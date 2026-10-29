//! Fused attention equivalence, selection and fallback (issue #104).
//!
//! The core proof of this lane: the catalog's `FusedAttention` row (online
//! softmax, no score matrix) matches the composed
//! `ScaledDotProductAttention` row forward and backward, across head
//! configurations (MHA/GQA/MQA), causal masking on and off, and
//! decode-shaped rectangles -- including sequence lengths that cross the
//! kernel's query/key block boundaries, which is what pins the causal
//! block-skip as correct (its benchmark proof is device-bound and deferred).
//!
//! Selection is pinned too: with no capability row admitting fused on CPU,
//! `Auto` must pick the composed row and say so (`FusedUnavailable`); the
//! forced paths exist for benchmarking the crossover, not for production.
//!
//! The last test is CPU-only supporting evidence for the memory AC, loudly
//! labelled as such: total bytes allocated during one forward at three
//! sequence lengths, composed vs fused. It is not peak high-water, and it is
//! not the device measurement the acceptance criterion requires.

extern crate incin_core as incin;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::nn::{
    AttentionCrossover, AttentionPath, AttentionPreference, AttentionSelection,
    AttentionSelectionSource, fused_or_composed_attention, select_attention_path,
};
use incin_core::prelude::*;

type Cpu = CpuBackendImpl;

/// Deterministic pseudo-random values in a softmax-sane range.
fn rand_tensor(dims: Vec<usize>, seed: u64) -> Result<Tensor<Dyn, Cpu, f32, NoGrad>> {
    let n: usize = dims.iter().product();
    let values = (0..n)
        .map(|i| {
            (i as f32 * 0.731 + seed as f32 * 1.7).sin() * 1.3 + (i as f32 * 0.013).cos() * 0.4
        })
        .collect::<Vec<f32>>();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared tensors differ in length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// One query/key/value triple of head-split attention operands.
type Qkv = (
    Tensor<Dyn, Cpu, f32, NoGrad>,
    Tensor<Dyn, Cpu, f32, NoGrad>,
    Tensor<Dyn, Cpu, f32, NoGrad>,
);

struct Case {
    name: &'static str,
    batch: usize,
    heads_q: usize,
    heads_kv: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
}

impl Case {
    fn qkv(&self, seed: u64) -> Result<Qkv> {
        Ok((
            rand_tensor(
                vec![self.batch, self.heads_q, self.seq_q, self.head_dim],
                seed,
            )?,
            rand_tensor(
                vec![self.batch, self.heads_kv, self.seq_kv, self.head_dim],
                seed + 1,
            )?,
            rand_tensor(
                vec![self.batch, self.heads_kv, self.seq_kv, self.head_dim],
                seed + 2,
            )?,
        ))
    }
}

const FWD_CASES: &[Case] = &[
    Case {
        name: "mha-square",
        batch: 1,
        heads_q: 4,
        heads_kv: 4,
        seq_q: 6,
        seq_kv: 6,
        head_dim: 8,
    },
    Case {
        name: "mha-batched",
        batch: 2,
        heads_q: 2,
        heads_kv: 2,
        seq_q: 8,
        seq_kv: 8,
        head_dim: 8,
    },
    Case {
        name: "gqa-8-over-2",
        batch: 1,
        heads_q: 8,
        heads_kv: 2,
        seq_q: 8,
        seq_kv: 8,
        head_dim: 8,
    },
    Case {
        name: "mqa-4-over-1",
        batch: 1,
        heads_q: 4,
        heads_kv: 1,
        seq_q: 6,
        seq_kv: 6,
        head_dim: 4,
    },
    Case {
        name: "gqa-decode-rect",
        batch: 1,
        heads_q: 4,
        heads_kv: 2,
        seq_q: 2,
        seq_kv: 9,
        head_dim: 8,
    },
    // Crosses the kernel's 32-wide query/key blocks, so partial blocks and
    // the causal skip frontier all execute.
    Case {
        name: "mha-crosses-blocks",
        batch: 1,
        heads_q: 2,
        heads_kv: 2,
        seq_q: 40,
        seq_kv: 40,
        head_dim: 8,
    },
    Case {
        name: "gqa-decode-crosses-blocks",
        batch: 1,
        heads_q: 4,
        heads_kv: 2,
        seq_q: 5,
        seq_kv: 70,
        head_dim: 8,
    },
    // Tall rectangle (more queries than keys): the causal rule saturates to
    // `j <= i`, matching the plain triangle the composed path builds.
    Case {
        name: "mha-tall-rect",
        batch: 1,
        heads_q: 2,
        heads_kv: 2,
        seq_q: 6,
        seq_kv: 4,
        head_dim: 8,
    },
];

fn run_fused<G: RequiresGrad>(
    q: &Tensor<Dyn, Cpu, f32, G>,
    k: &Tensor<Dyn, Cpu, f32, G>,
    v: &Tensor<Dyn, Cpu, f32, G>,
    causal: bool,
) -> Result<Tensor<Dyn, Cpu, f32, G>> {
    let (out, selection) = fused_or_composed_attention(
        q,
        k,
        v,
        causal,
        None,
        AttentionPreference::ForceFused,
        AttentionCrossover::unknown(),
    )?;
    assert_eq!(selection.path, AttentionPath::Fused);
    assert_eq!(selection.source, AttentionSelectionSource::Forced);
    Ok(out)
}

fn run_composed<G: RequiresGrad>(
    q: &Tensor<Dyn, Cpu, f32, G>,
    k: &Tensor<Dyn, Cpu, f32, G>,
    v: &Tensor<Dyn, Cpu, f32, G>,
    causal: bool,
) -> Result<Tensor<Dyn, Cpu, f32, G>> {
    let (out, selection) = fused_or_composed_attention(
        q,
        k,
        v,
        causal,
        None,
        AttentionPreference::ForceComposed,
        AttentionCrossover::unknown(),
    )?;
    assert_eq!(selection.path, AttentionPath::Composed);
    assert_eq!(selection.source, AttentionSelectionSource::Forced);
    Ok(out)
}

/// Forward equivalence across the whole matrix: MHA/GQA/MQA, causal on and
/// off, square and decode rectangles, block-crossing lengths.
#[test]
fn fused_forward_matches_composed() -> Result<()> {
    let mut report = Vec::new();
    for case in FWD_CASES {
        for causal in [false, true] {
            let (q, k, v) = case.qkv(11)?;
            let fused = run_fused(&q, &k, &v, causal)?.to_vec1::<f32>()?;
            let composed = run_composed(&q, &k, &v, causal)?.to_vec1::<f32>()?;
            assert!(
                fused.iter().all(|x| x.is_finite()),
                "{} causal={causal}: fused produced non-finite values",
                case.name
            );
            let diff = max_abs_diff(&fused, &composed);
            report.push((case.name, causal, diff));
            assert!(
                diff < 1e-5,
                "{} causal={causal}: fused vs composed forward diff {diff:e} exceeds 1e-5",
                case.name
            );
        }
    }
    // Printed so the lane report can quote per-case numbers, not just a pass.
    for (name, causal, diff) in &report {
        println!("fwd {name} causal={causal}: max abs diff {diff:e}");
    }
    Ok(())
}

/// Backward equivalence: the fused single-tape-entry backward (recompute,
/// no stored attention matrix) matches the composed multi-entry backward.
#[test]
fn fused_backward_matches_composed() -> Result<()> {
    let mut report = Vec::new();
    for case in FWD_CASES.iter().filter(|c| c.seq_q <= 8) {
        for causal in [false, true] {
            let mut grad_diffs = Vec::new();
            // Same values both paths; fresh tensors per path because
            // `backward` drains the tape.
            let (qf, kf, vf) = case.qkv(101)?;
            let (qc, kc, vc) = case.qkv(101)?;
            let qf = qf.require_grad();
            let kf = kf.require_grad();
            let vf = vf.require_grad();
            let qc = qc.require_grad();
            let kc = kc.require_grad();
            let vc = vc.require_grad();
            let loss_f = run_fused(&qf, &kf, &vf, causal)?.sum_all()?;
            let loss_c = run_composed(&qc, &kc, &vc, causal)?.sum_all()?;
            // The losses themselves must agree before gradients can.
            let loss_diff = (loss_f.to_vec1::<f32>()?[0] - loss_c.to_vec1::<f32>()?[0]).abs();
            assert!(
                loss_diff < 1e-4,
                "{} causal={causal}: scalar losses differ by {loss_diff:e}",
                case.name
            );
            let grads_f = loss_f.backward()?;
            let grads_c = loss_c.backward()?;
            for (gname, (tf, tc)) in [("q", (&qf, &qc)), ("k", (&kf, &kc)), ("v", (&vf, &vc))] {
                let gf = grads_f.require(tf)?.to_vec1::<f32>()?;
                let gc = grads_c.require(tc)?.to_vec1::<f32>()?;
                assert!(
                    gf.iter().all(|x| x.is_finite()),
                    "{} causal={causal}: fused d{gname} non-finite",
                    case.name
                );
                let diff = max_abs_diff(&gf, &gc);
                grad_diffs.push((gname, diff));
                assert!(
                    diff < 1e-4,
                    "{} causal={causal}: d{gname} diff {diff:e} exceeds 1e-4",
                    case.name
                );
            }
            report.push((case.name, causal, loss_diff, grad_diffs));
        }
    }
    for (name, causal, loss_diff, grad_diffs) in &report {
        println!("bwd {name} causal={causal}: loss diff {loss_diff:e}, grads {grad_diffs:?}");
    }
    Ok(())
}

/// The causal block-skip is correctness-pinned here: fused+causal must equal
/// composed against the exact mask rows the module uses -- the full lower
/// triangle for square inputs, the narrowed trailing rows for decode
/// rectangles -- at lengths that cross block boundaries. A kernel that
/// computed the upper triangle and masked it would pass this too; the
/// benchmark proving it *skips* the work is device-bound and deferred.
#[test]
fn causal_skip_matches_the_composed_masks() -> Result<()> {
    // Square: full triangle, crossing the 32-wide blocks.
    let (q, k, v) = Case {
        name: "skip-square",
        batch: 1,
        heads_q: 2,
        heads_kv: 2,
        seq_q: 40,
        seq_kv: 40,
        head_dim: 8,
    }
    .qkv(7)?;
    let diff = max_abs_diff(
        &run_fused(&q, &k, &v, true)?.to_vec1::<f32>()?,
        &run_composed(&q, &k, &v, true)?.to_vec1::<f32>()?,
    );
    assert!(diff < 1e-5, "square causal skip diverged: {diff:e}");

    // Decode rectangle: 5 new queries against 70 cached keys, crossing both
    // block frontiers while the skip cuts the leading key blocks per row.
    let (q, k, v) = Case {
        name: "skip-decode",
        batch: 2,
        heads_q: 4,
        heads_kv: 2,
        seq_q: 5,
        seq_kv: 70,
        head_dim: 8,
    }
    .qkv(8)?;
    let diff = max_abs_diff(
        &run_fused(&q, &k, &v, true)?.to_vec1::<f32>()?,
        &run_composed(&q, &k, &v, true)?.to_vec1::<f32>()?,
    );
    assert!(diff < 1e-5, "decode causal skip diverged: {diff:e}");

    // Non-causal rectangles must not skip anything.
    let diff = max_abs_diff(
        &run_fused(&q, &k, &v, false)?.to_vec1::<f32>()?,
        &run_composed(&q, &k, &v, false)?.to_vec1::<f32>()?,
    );
    assert!(diff < 1e-5, "non-causal rectangle diverged: {diff:e}");
    Ok(())
}

/// Grouped-query configurations run in the same kernel: MQA and GQA go
/// through the identical `hq / groups` head mapping, pinned here by
/// equivalence rather than by inspection.
#[test]
fn grouped_query_runs_in_the_same_kernel() -> Result<()> {
    for (heads_q, heads_kv) in [(8, 2), (4, 1), (6, 3)] {
        let case = Case {
            name: "gqa-same-kernel",
            batch: 1,
            heads_q,
            heads_kv,
            seq_q: 7,
            seq_kv: 7,
            head_dim: 8,
        };
        let (q, k, v) = case.qkv(21)?;
        for causal in [false, true] {
            let diff = max_abs_diff(
                &run_fused(&q, &k, &v, causal)?.to_vec1::<f32>()?,
                &run_composed(&q, &k, &v, causal)?.to_vec1::<f32>()?,
            );
            assert!(
                diff < 1e-5,
                "heads {heads_q}-over-{heads_kv} causal={causal}: {diff:e}"
            );
        }
    }
    Ok(())
}

/// `Auto` on CPU today: the capability row admits fused, but no crossover
/// has been measured (`AttentionCrossover::unknown`), so the composed row
/// is selected *rather than attempted*, with the reason recorded.
/// `AboveCrossover`/`Fused` needs measured device crossover values, which
/// are deferred to the hardware runner (issue #104).
#[test]
fn auto_stays_composed_until_crossover_is_measured() -> Result<()> {
    let case = &FWD_CASES[2];
    let (q, k, v) = case.qkv(31)?;
    for causal in [false, true] {
        let (out, selection) = fused_or_composed_attention(
            &q,
            &k,
            &v,
            causal,
            None,
            AttentionPreference::Auto,
            AttentionCrossover::unknown(),
        )?;
        assert_eq!(
            (selection.path, selection.source),
            (
                AttentionPath::Composed,
                AttentionSelectionSource::BelowCrossover
            ),
            "Auto with an unmeasured crossover must stay composed; \
             AboveCrossover/Fused needs measured device crossover values"
        );
        // And the fallback output is the composed answer.
        let diff = max_abs_diff(
            &out.to_vec1::<f32>()?,
            &run_composed(&q, &k, &v, causal)?.to_vec1::<f32>()?,
        );
        assert!(
            diff == 0.0,
            "Auto fallback differs from forced composed: {diff:e}"
        );
    }
    Ok(())
}

/// The pure selection function: forced paths, missing admission, and the
/// crossover stub's three positions.
#[test]
fn selection_policy_table() {
    use AttentionPreference::*;
    use AttentionSelectionSource::*;
    // Forced paths ignore admission and crossover alike.
    assert_eq!(
        select_attention_path(1024, true, ForceFused, AttentionCrossover::unknown()),
        AttentionSelection {
            path: AttentionPath::Fused,
            source: Forced
        }
    );
    assert_eq!(
        select_attention_path(1024, true, ForceComposed, AttentionCrossover::measured(8)),
        AttentionSelection {
            path: AttentionPath::Composed,
            source: Forced
        }
    );
    // No admitting row: composed, whatever the crossover says.
    assert_eq!(
        select_attention_path(1024, false, Auto, AttentionCrossover::measured(8)),
        AttentionSelection {
            path: AttentionPath::Composed,
            source: FusedUnavailable
        }
    );
    // No measurement: fail closed toward the portable path.
    assert_eq!(
        select_attention_path(1024, true, Auto, AttentionCrossover::unknown()),
        AttentionSelection {
            path: AttentionPath::Composed,
            source: BelowCrossover
        }
    );
    // Measured crossover at 8: 7 stays composed, 8 and up go fused.
    let measured = AttentionCrossover::measured(8);
    assert_eq!(
        select_attention_path(7, true, Auto, measured).path,
        AttentionPath::Composed
    );
    assert_eq!(
        select_attention_path(8, true, Auto, measured),
        AttentionSelection {
            path: AttentionPath::Fused,
            source: AboveCrossover
        }
    );
    assert_eq!(
        select_attention_path(4096, true, Auto, measured).source,
        AboveCrossover
    );
}

// --- CPU-only memory evidence (supporting, NOT the device AC) ---

thread_local! {
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

impl Counting {
    fn measure(f: impl FnOnce()) -> usize {
        ALLOCATED_BYTES.with(|c| c.set(0));
        COUNTING.with(|c| c.set(true));
        f();
        COUNTING.with(|c| c.set(false));
        ALLOCATED_BYTES.with(|c| c.get())
    }
}

// SAFETY: every method forwards to `System` unchanged; the byte counter is a
// thread-local cell read that cannot allocate (mirrors the counting allocator
// in `incin/tests/hot_path_allocations.rs`).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding layout directly to the System allocator.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && COUNTING.with(Cell::get) {
            ALLOCATED_BYTES.with(|c| c.set(c.get() + layout.size()));
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding pointer and layout directly to the System allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// CPU-ONLY supporting evidence for the linear-memory AC: total bytes
/// allocated during one forward (not peak high-water) at three sequence
/// lengths, composed vs fused.
///
/// The composed total carries the `[H, T, T]` score buffer plus its
/// scale/mask/softmax intermediates; the fused total carries the output and
/// `O(T)` running state. What this shows is allocation growth, and it shows
/// it on CPU only. The acceptance criterion needs device peak-memory
/// benches, which are hardware-gated and deferred.
#[test]
fn cpu_allocation_growth_is_subquadratic_for_fused() -> Result<()> {
    let mut table = Vec::new();
    for seq in [16, 64, 256] {
        let (q, k, v) = Case {
            name: "mem",
            batch: 1,
            heads_q: 4,
            heads_kv: 4,
            seq_q: seq,
            seq_kv: seq,
            head_dim: 8,
        }
        .qkv(3)?;
        // Warm up outside the counter so one-time lazy state is not counted.
        let _ = run_composed(&q, &k, &v, false)?;
        let _ = run_fused(&q, &k, &v, false)?;
        let composed_bytes = Counting::measure(|| {
            let _ = run_composed(&q, &k, &v, false).unwrap();
        });
        let fused_bytes = Counting::measure(|| {
            let _ = run_fused(&q, &k, &v, false).unwrap();
        });
        table.push((seq, composed_bytes, fused_bytes));
    }
    for (seq, composed, fused) in &table {
        println!("alloc T={seq}: composed {composed} bytes, fused {fused} bytes");
    }
    let growth = |a: usize, b: usize| b as f64 / a as f64;
    let composed_growth = growth(table[1].1, table[2].1);
    let fused_growth = growth(table[1].2, table[2].2);
    println!("growth 64->256: composed {composed_growth:.2}x, fused {fused_growth:.2}x");
    // 4x the sequence: quadratic growth approaches 16x, linear stays near
    // 4x. The margins are wide on purpose -- this is a growth-shape check,
    // not a benchmark.
    assert!(
        composed_growth > 8.0,
        "composed growth {composed_growth:.2}x does not look quadratic; \
         the measurement setup may be broken"
    );
    assert!(
        fused_growth < composed_growth / 2.0,
        "fused growth {fused_growth:.2}x is not clearly sub-quadratic vs \
         composed {composed_growth:.2}x"
    );
    Ok(())
}
