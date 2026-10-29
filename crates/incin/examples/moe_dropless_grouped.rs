//! Dropless mixture-of-experts over option-C offsets (proposal P1).
//!
//! A runnable sketch of proposal P1 from
//! `docs/plan/research/0.2.0/moe-design-space.md`: a top-k MoE forward pass
//! with NO token dropped and NO padding, built from the offset-array
//! geometry issue #102 settled on (option C), with the B-interim spirit of
//! `102-nameable-spans.md` §3/§7 kept visible in the comments.
//!
//! What you are seeing, in order:
//!
//! 1. **Router.** A real [`Router`] module (`linear -> softmax -> top-k ->
//!    renormalize`) routes `T` tokens to `TOPK` of `E` experts each. The
//!    gate weights it returns stay differentiable; the expert ids do not.
//! 2. **Option-C geometry.** The `[T, K]` assignment flattens to `[T*K]`
//!    slot ids, `argsort` permutes them into expert order, the tokens are
//!    `repeat_interleave`d once per slot and gathered into the static
//!    `[T*K, D]` buffer, and `bincount + cumsum` (via
//!    [`Routing::expert_offsets`][incin::nn::Routing]) builds the `[E+1]`
//!    offsets. The per-expert token count `n_e` never becomes a tensor
//!    extent: it is a row range `offsets[e]..offsets[e+1]`, never a shape.
//! 3. **Grouped GEMM.** One `grouped_matmul` call multiplies every expert's
//!    row block by its own weight (`lhs [T*K, K] x rhs [E, K, N]` sliced by
//!    `offsets [E+1]`), twice (up-projection, ReLU, down-projection), on the
//!    CPU path that already exists. No per-expert loop, no padding.
//! 4. **Combine.** Gate weights ride along the same permutation, scale the
//!    grouped outputs, and `scatter_add` accumulates each slot back into its
//!    token row. Two writes to one row sum; nothing is overwritten.
//! 5. **Aux loss.** A Switch-style load-balancing scalar `E * sum(f * P)`
//!    travels beside the output as an `(output, aux)` tuple, mirroring the
//!    issue-#102 decision that the aux loss lives in `Module::Output`.
//!
//! Facade honesty: the forward pass below is all `Tensor` facade methods
//! (`softmax`/`topk`/`gather` via `Router::forward`, `flatten_runtime`,
//! `argsort`, `bincount`, `expert_offsets`, `repeat_interleave`,
//! `index_select`, `grouped_matmul`, `broadcast_mul`, `scatter_add`,
//! `mean`, `sum_all`). Unlike `quantization_qat.rs`, which must reach past
//! the facade into backend-authoring dispatch for `quantized_matmul`, the
//! grouped path already has a CPU implementation, so P1 runs here as
//! written — no dispatch fallback, no per-expert loop. The queued upgrade
//! is a *performant* CUDA grouped kernel (#85 -> #103), not a semantic one.
//! One step sits above the facade by design: `deterministic_router` builds
//! the gate from fixed literals through the module/param surface
//! (`var_from_tensor` + `Param::from_parts_checked` +
//! `Linear::from_raw_parts`, the `custom_dtype_walkthrough.rs` gate-3c
//! precedent), because there is no seeded initializer and `Router::build`
//! fills nondeterministic kaiming noise.
//!
//! Determinism: tokens, gate projection, and expert weights are all fixed
//! literals below, so every number printed is reproducible run to run.
//! (A `Router::build` gate would NOT be: it fills kaiming-uniform noise via
//! `VariableUniformRandom`, which the catalog marks nondeterministic — hence
//! the fixed gate in `deterministic_router`.)
//!
//! Run with: `cargo run -p incin --example moe_dropless_grouped --no-default-features --features incin-backends/cpu,incin/cpu`

#![cfg(feature = "cpu")]

use incin::nn::param::Trainable;
use incin::nn::{Linear, Param, Router};
use incin::prelude::*;
use incin_core::backend_authoring::{ShapeBuf, VariableBackend};

type B = DefaultBackend;

// Small enough to read in one sitting: 4 experts, top-2 routing over 6
// tokens of width 8, experts that widen to 16 and back.
const T: usize = 6;
const D: usize = 8;
const E: usize = 4;
const TOPK: usize = 2;
const DFF: usize = 16;
const SLOTS: usize = T * TOPK;

fn main() -> incin::Result<()> {
    section("1. Tokens in, routing out: linear -> softmax -> top-k");
    let x = tokens()?;
    let router = deterministic_router()?;
    // `forward` runs softmax, `topk`, `gather`, and the renormalizing
    // divide internally: probs [T, E], weights [T, K], indices [T, K].
    let routing = router.forward(x.clone())?;

    let indices = routing.indices.to_vec1::<u32>()?;
    let weights = routing.weights.to_vec1::<f32>()?;
    for token in 0..T {
        let pair = [indices[token * TOPK], indices[token * TOPK + 1]];
        let gate = [weights[token * TOPK], weights[token * TOPK + 1]];
        println!(
            "  token {token}: experts [{}, {}]   gates [{:.3}, {:.3}]",
            pair[0], pair[1], gate[0], gate[1]
        );
        // The router renormalizes the selected gates, so each row sums to 1.
        assert!(
            (gate[0] + gate[1] - 1.0).abs() < 1e-5,
            "token {token}'s gates must renormalize to 1"
        );
    }

    section("2. Option-C geometry: permute, counts, [E+1] offsets");
    // Flatten the [T, K] assignment to [T*K] slot ids. `flatten_runtime`
    // is the runtime-axis form (the factor is data layout, not a const),
    // and `to_dtype` moves the keys to the i64 the offsets tile uses.
    let flat = routing.indices.flatten_runtime(0, 1)?.to_dtype::<i64>()?;
    // The permutation that groups slots by expert: stable, so the same
    // assignment always groups the same way. One layout note, straight
    // from `Router::forward`'s own playbook: `argsort` (like `topk`) is
    // only defined for the default layout, while `flatten`/`to_dtype`
    // return row-major proofs, so the proof is dropped before selecting.
    let perm = flat.clone().forget_layout().argsort(0, false)?;
    let perm_host = perm.to_vec1::<u32>()?;
    let sorted_ids = flat.index_select(0isize, &perm)?.to_vec1::<i64>()?;
    // `bincount` counts the [T, K] assignment directly: geometry that
    // addressed the indices does not survive, only the histogram does.
    // `expert_offsets` scans it (bincount -> cumsum -> prepend zero) with
    // no host interop.
    let counts = routing.indices.bincount::<E>()?.to_vec1::<i64>()?;
    let offsets = routing.expert_offsets()?.to_vec1::<i64>()?;
    println!(
        "  per-expert counts: {counts:?}   (sum = {}, slots = {SLOTS})",
        {
            let total: i64 = counts.iter().sum();
            total
        }
    );
    println!("  offsets [E+1]:     {offsets:?}");
    println!("  slot -> expert:    {sorted_ids:?}");
    println!("  permutation:       {perm_host:?}");
    assert_eq!(
        counts.iter().sum::<i64>(),
        SLOTS as i64,
        "every slot lands on exactly one expert: nothing dropped"
    );
    assert_eq!(offsets[0], 0, "offsets start at row 0");
    assert_eq!(
        offsets[E], SLOTS as i64,
        "offsets end at the last grouped row"
    );
    for e in 0..E {
        assert!(
            offsets[e] <= offsets[e + 1],
            "offsets must tile [0, T*K) without overlap"
        );
        let span = &sorted_ids[offsets[e] as usize..offsets[e + 1] as usize];
        assert!(
            span.iter().all(|slot| *slot == e as i64),
            "expert {e} owns rows {}..{} and they hold {:?}",
            offsets[e],
            offsets[e + 1],
            span
        );
    }
    println!("  each expert's span holds exactly its own id: the sort and the offsets agree");

    section("3. Static buffer + grouped GEMM (the dropless multiply)");
    // One row per slot: token t appears TOPK times consecutively, so slot
    // s = t*K + j. Gathering by `perm` turns slot order into expert order.
    let buffer = x.repeat_interleave(TOPK, 0)?.index_select(0isize, &perm)?;
    assert_eq!(buffer.dims().as_ref(), &[SLOTS, D]);
    // A permutation only reorders: the grouped rows are the expanded rows.
    let mut grouped_rows = buffer.to_vec1::<f32>()?;
    let mut expanded_rows = x.repeat_interleave(TOPK, 0)?.to_vec1::<f32>()?;
    grouped_rows.sort_by(f32::total_cmp);
    expanded_rows.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(
        grouped_rows, expanded_rows,
        "the buffer must be a pure reorder of the expanded tokens"
    );
    println!("  buffer [T*K, D] = [{SLOTS}, {D}]: a pure reorder, no padding rows");

    // Stacked experts [E, K, N]: one small FFN per expert, fixed literals.
    let w_up = Tensor::<Dyn, B>::from_slice(&expert_values(0), vec![E, D, DFF])?;
    let w_down = Tensor::<Dyn, B>::from_slice(&expert_values(1), vec![E, DFF, D])?;
    // The P1 multiply: each expert's span meets its own weights in one
    // call. Empty spans contribute nothing and are not an error, which is
    // the empty-expert case a router legitimately produces.
    let hidden = buffer
        .grouped_matmul(&w_up, &routing.expert_offsets()?)?
        .relu()?;
    let grouped = hidden.grouped_matmul(&w_down, &routing.expert_offsets()?)?;
    assert_eq!(grouped.dims().as_ref(), &[SLOTS, D]);
    let grouped_host = grouped.to_vec1::<f32>()?;
    // Independent check in plain Rust: slice each expert's span and run the
    // same up/ReLU/down multiply on the host. This pins `grouped_matmul`
    // rather than re-running it.
    let expected = naive_grouped_gemm(&buffer.to_vec1::<f32>()?, &offsets);
    let worst = grouped_host
        .iter()
        .zip(&expected)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  grouped output shape {:?}; max |kernel - naive| = {worst:.2e}",
        grouped.dims()
    );
    assert!(
        worst < 1e-4,
        "grouped_matmul must match the per-expert naive multiply"
    );

    section("4. Gate weighting + scatter-add combine (nothing overwritten)");
    // The gates ride the same permutation into expert order, then scale
    // each grouped row through a [T*K, 1] column.
    let gate_col = routing
        .weights
        .flatten_runtime(0, 1)?
        .index_select(0isize, &perm)?
        .unsqueeze(1isize)?;
    let weighted = grouped.broadcast_mul(&gate_col)?;
    // Scatter index: grouped row r belongs to token perm[r] / TOPK, naming
    // every column of that row so the whole row lands at once.
    let mut scatter_host = Vec::with_capacity(SLOTS * D);
    for row in &perm_host {
        for _ in 0..D {
            scatter_host.push(row / TOPK as u32);
        }
    }
    let scatter_index = Tensor::<Dyn, B, u32>::from_slice(&scatter_host, vec![SLOTS, D])?;
    let base = Tensor::<Dyn, B>::zeros(vec![T, D])?;
    let out = base.scatter_add(0isize, &scatter_index, &weighted)?;
    assert_eq!(out.dims().as_ref(), &[T, D]);
    let out_host = out.to_vec1::<f32>()?;
    println!(
        "  combined output shape {:?} (one row per token)",
        out.dims()
    );
    for token in 0..T.min(2) {
        println!(
            "  out[{token}]: {:?}",
            &out_host[token * D..(token + 1) * D]
        );
    }
    // Dropless proof on the host: `expected` is in grouped (expert)
    // order, so row r holds slot perm[r], not slot r. Walk the grouped
    // rows, weight each by its own slot's gate, and accumulate into its
    // token's check row: every slot contributes exactly once.
    let flat_weights = routing.weights.flatten_runtime(0, 1)?.to_vec1::<f32>()?;
    let mut want_rows = [0.0f32; T * D];
    for r in 0..SLOTS {
        let slot = perm_host[r] as usize;
        let token = slot / TOPK;
        for d in 0..D {
            want_rows[token * D + d] += expected[r * D + d] * flat_weights[slot];
        }
    }
    for token in 0..T {
        for d in 0..D {
            let (got, want) = (out_host[token * D + d], want_rows[token * D + d]);
            assert!(
                (got - want).abs() < 1e-4,
                "token {token} dim {d}: combine lost a contribution"
            );
        }
    }
    println!("  all {T} rows equal their TOPK weighted contributions: no token dropped");

    section("5. Aux loss beside the output (the #102 Output tuple)");
    // Switch-style balance term E * sum(f * P): f is the routed fraction
    // per expert (NoGrad counts), P the mean gate probability per expert
    // (differentiable, so the gate still learns from this term).
    let frac = counts
        .iter()
        .map(|count| *count as f32 / SLOTS as f32)
        .collect::<Vec<_>>();
    let mean_prob = routing.probs.mean(0isize)?.to_vec1::<f32>()?;
    println!("  routed fraction f: {frac:.3?}");
    println!("  mean prob     P: {mean_prob:.3?}");
    let frac_t = routing
        .indices
        .bincount::<E>()?
        .to_dtype::<f32>()?
        .div_scalar(SLOTS as f64)?;
    let aux = frac_t
        .broadcast_mul(&routing.probs.mean(0isize)?)?
        .sum_all()?
        .mul_scalar(E as f64)?;
    let aux_value: f32 = aux.to_scalar()?;
    // The tuple the memo mandates: combined output plus balance scalar.
    let (output, aux_loss) = (out, aux);
    println!("  aux loss E*sum(f*P) = {aux_value:.6}");
    println!(
        "  output shape {:?}, aux shape {:?}: the (tensor, aux) pair",
        output.dims(),
        aux_loss.dims()
    );
    assert!(
        (0.0..=E as f32).contains(&aux_value),
        "a balance term over two distributions lives in [0, E]"
    );

    section("done: dropless MoE, six tokens through four experts");
    println!("  routing, counts, offsets, grouped GEMM, scatter-add combine, aux:");
    println!("  every stage above ran on CPU through facade ops, nothing padded, nothing dropped.");

    Ok(())
}

/// Six deterministic tokens in [-1, 1]: a fixed ramp, not a sample.
fn tokens() -> incin::Result<Tensor<Dyn, B>> {
    let mut values = Vec::with_capacity(T * D);
    for token in 0..T {
        for dim in 0..D {
            values.push(((token * 7 + dim * 3) % 11) as f32 / 5.0 - 1.0);
        }
    }
    Tensor::<Dyn, B>::from_slice(&values, vec![T, D])
}

/// The router with a fixed-literal gate projection `[E, D]`, so routing is
/// identical every run. `Router::build` is honest random init (kaiming
/// uniform through the catalog's nondeterministic `VariableUniformRandom`
/// row), which is what you want for training and not for a learning example.
/// The construction below is the in-tree literal-weights precedent from
/// `custom_dtype_walkthrough.rs` (gate 3c): `var_from_tensor` promotes a
/// `from_slice` tensor to a trainable variable, `from_parts_checked`
/// re-attaches the shape/dtype/device proofs, and `Linear::from_raw_parts`
/// plus the `Router` struct literal skip init entirely. Distinct rows spread
/// the six tokens across all four experts; the printed routing pins it.
fn deterministic_router() -> incin::Result<Router<E, TOPK, B>> {
    let mut values = Vec::with_capacity(E * D);
    for e in 0..E {
        for d in 0..D {
            values.push(((e * 11 + d * 5 + 3) % 9) as f32 / 4.0 - 1.0);
        }
    }
    let weight_tensor = Tensor::<Dyn, B>::from_slice(&values, vec![E, D])?;
    let var = B::var_from_tensor::<f32>(weight_tensor.inner())?;
    let weight = Param::<Dyn, B, f32, Trainable>::from_parts_checked(
        var,
        ShapeBuf::from_slice(&[E, D]),
        f32::init(()),
        <Cpu as Device>::init(()),
    )?;
    Ok(Router {
        gate: Linear::from_raw_parts(weight, None),
    })
}

/// Fixed expert weights: `which` selects the up (`0`, [E, D, DFF]) or down
/// (`1`, [E, DFF, D]) stack. Small magnitudes, mixed signs so the ReLU
/// between the two grouped multiplies has something to do.
fn expert_values(which: usize) -> Vec<f32> {
    let (rows, cols) = if which == 0 { (D, DFF) } else { (DFF, D) };
    let mut values = Vec::with_capacity(E * rows * cols);
    for e in 0..E {
        for i in 0..rows {
            for j in 0..cols {
                let code = (e * 131 + i * 17 + j * 7 + which * 41) % 13;
                values.push(code as f32 / 12.0 - 0.5);
            }
        }
    }
    values
}

/// Plain-Rust grouped GEMM: for each expert span, rows @ up, ReLU, @ down.
/// The independent reference section 3 checks the kernel against.
fn naive_grouped_gemm(buffer: &[f32], offsets: &[i64]) -> Vec<f32> {
    let w_up = expert_values(0);
    let w_down = expert_values(1);
    let mut out = vec![0.0f32; SLOTS * D];
    for e in 0..E {
        let (start, end) = (offsets[e] as usize, offsets[e + 1] as usize);
        for r in start..end {
            let mut hidden = [0.0f32; DFF];
            for j in 0..DFF {
                let mut acc = 0.0f32;
                for i in 0..D {
                    acc += buffer[r * D + i] * w_up[(e * D + i) * DFF + j];
                }
                hidden[j] = acc.max(0.0);
            }
            for j in 0..D {
                let mut acc = 0.0f32;
                for i in 0..DFF {
                    acc += hidden[i] * w_down[(e * DFF + i) * D + j];
                }
                out[r * D + j] = acc;
            }
        }
    }
    out
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}
