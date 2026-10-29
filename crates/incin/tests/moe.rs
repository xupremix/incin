//! Integration coverage for `Router` and `MoE` on the public facade (#102).
//!
//! The dense masked path is what this pins: every expert runs, tokens that
//! did not select an expert contribute zero through the one-hot weight
//! column, and the gate stays differentiable. Capacity-factor drop and the
//! aux load-balancing loss remain deferred gaps for a later issue and are
//! deliberately not asserted here.
#![cfg(feature = "cpu")]

use incin::nn::{ComputeStats, MoE, Module, NamedLayers, Router, TrainMode};
use incin::prelude::*;
use incin::state::{collect_state, load_state};

type Cpu = incin_backends::cpu::CpuBackendImpl;

/// Deterministic non-constant input so zeros cannot mask a broken multiply.
fn ramp(dims: Vec<usize>) -> Result<Tensor<Dyn, Cpu, f32, NoGrad>> {
    let n: usize = dims.iter().product();
    let values = (0..n)
        .map(|v| ((v as f32) * 0.37).sin() * 0.9 + (v as f32) * 0.01)
        .collect::<Vec<f32>>();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims)
}

fn build_moe(d_model: usize, d_expert: usize) -> Result<MoE<2, 1, Linear<Dyn, Cpu>, Cpu>> {
    MoE::<2, 1, _, Cpu>::build(d_model, (), (), || {
        Linear::<Dyn, Cpu>::build((d_model, d_expert))
    })
}

fn nonzero(values: &[f32]) -> usize {
    values.iter().filter(|v| **v != 0.0).count()
}

/// Forward preserves the last dimension and the token count. The dense path
/// runs every expert on the full batch, so the output width is the expert
/// width, not the model width.
#[test]
fn forward_preserves_tokens_and_projects_to_the_expert_width() -> Result<()> {
    let moe = build_moe(8, 16)?;
    let x = ramp(vec![4, 8])?;
    let y = moe.forward(x)?;
    assert_eq!(y.dims().dims(), &[4, 16]);
    assert!(
        y.to_vec1::<f32>()?.iter().all(|v| v.is_finite()),
        "the dense masked path produced a non-finite value"
    );
    Ok(())
}

/// The gate is a softmax: every token's full distribution sums to one, and
/// the renormalized top-k weights sum to one for each token.
#[test]
fn router_probs_and_weights_are_probability_simplexes() -> Result<()> {
    let router = Router::<4, 2, Cpu>::build(8, (), ())?;
    let x = ramp(vec![5, 8])?;
    let routing = router.forward(x)?;

    assert_eq!(routing.probs.dims().dims(), &[5, 4]);
    assert_eq!(routing.weights.dims().dims(), &[5, 2]);
    assert_eq!(routing.indices.dims().dims(), &[5, 2]);

    let probs = routing.probs.to_vec1::<f32>()?;
    for token in 0..5 {
        let sum: f32 = probs[token * 4..(token + 1) * 4].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "token {token} gate probabilities sum to {sum}"
        );
    }

    let weights = routing.weights.to_vec1::<f32>()?;
    for token in 0..5 {
        let sum: f32 = weights[token * 2..(token + 1) * 2].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "token {token} top-k weights sum to {sum}"
        );
    }
    Ok(())
}

/// The gate is trainable: a backward pass must deliver a non-zero gradient
/// to `router.gate.weight` and to at least one expert weight. A router whose
/// gather accidentally detached the probabilities would fail the first; a
/// zero weight column everywhere would fail the second.
///
/// Uses `TOPK >= 2`: with `TOPK == 1` the renormalized weight is identically
/// `1.0` (`p_max / p_max`), so the gate gradient through the dense path is
/// exactly zero. Top-1 routing would need the deferred aux load-balancing
/// loss to train the gate (issue #102 gap).
#[test]
fn gate_and_expert_weights_receive_gradients() -> Result<()> {
    let moe = MoE::<4, 2, _, Cpu>::build(8, (), (), || Linear::<Dyn, Cpu>::build((8, 16)))?;
    let x = ramp(vec![4, 8])?.require_grad();
    let y = moe.forward(x)?;
    let target = Tensor::<Dyn, Cpu>::zeros(vec![4, 16])?;
    let grads = y.mse_loss(&target)?.backward()?;

    let gate = moe.router.gate.weight.as_tensor()?;
    let gate_grad = grads
        .require(&gate)
        .map_err(|e| Error::Msg(format!("no gradient reached the gate: {e}")))?
        .to_vec1::<f32>()?;
    assert!(
        nonzero(&gate_grad) > 0,
        "the gate received only zeros, so top-k routing cannot train"
    );

    let mut expert_hits = 0;
    for expert in &moe.experts {
        let w = expert.weight.as_tensor()?;
        if nonzero(&grads.require(&w)?.to_vec1::<f32>()?) > 0 {
            expert_hits += 1;
        }
    }
    assert!(
        expert_hits > 0,
        "no expert weight received a gradient, so the dense path is not training"
    );
    Ok(())
}

/// State keys name the router gate and each expert slot by index, and the
/// snapshot round-trips into a freshly built module that computes the same
/// thing.
#[test]
fn state_round_trips_through_experts_n_and_the_router_gate() -> Result<()> {
    let moe = build_moe(8, 16)?;
    let snapshot = collect_state::<Cpu, _>(&moe)?;

    // Bias-free gate: one weight. Two biased experts: weight + bias each.
    assert_eq!(
        snapshot.len(),
        5,
        "expected gate.weight plus two experts' weight+bias, got {:?}",
        snapshot
            .iter()
            .map(|(p, _)| p.as_str().to_string())
            .collect::<Vec<_>>()
    );
    let paths: Vec<&str> = snapshot.iter().map(|(p, _)| p.as_str()).collect();
    assert!(paths.contains(&"router.gate.weight"));
    assert!(paths.contains(&"experts.0.weight"));
    assert!(paths.contains(&"experts.0.bias"));
    assert!(paths.contains(&"experts.1.weight"));
    assert!(paths.contains(&"experts.1.bias"));

    let mut restored = build_moe(8, 16)?;
    load_state::<Cpu, _>(&mut restored, &snapshot)?;
    assert_eq!(collect_state::<Cpu, _>(&restored)?, snapshot);

    let x = ramp(vec![3, 8])?;
    let expected = moe.forward(x.clone())?.to_vec1::<f32>()?;
    let actual = restored.forward(x)?.to_vec1::<f32>()?;
    assert_eq!(
        expected, actual,
        "the restored module computes something else"
    );
    Ok(())
}

/// `expert_offsets` is the exclusive `[E + 1]` scan: starts at 0, each step
/// is the previous plus that expert's count, last entry is the total. Built
/// without host interop, so the values must still be exact integers.
#[test]
fn expert_offsets_are_the_exclusive_prefix_of_the_bincount() -> Result<()> {
    // Four tokens: two to expert 0, one to expert 1, one to expert 2.
    let indices = Tensor::<Dyn, Cpu, u32>::from_slice(&[0, 0, 1, 2], vec![4])?.into_row_major()?;
    let probs = Tensor::<Dyn, Cpu>::zeros(vec![4, 3])?.into_row_major()?;
    let weights = Tensor::<Dyn, Cpu>::zeros(vec![4, 1])?.into_row_major()?;
    let routing: incin::nn::Routing<3, Cpu, f32, NoGrad> = incin::nn::Routing {
        probs,
        weights,
        indices,
    };

    let offsets = routing.expert_offsets()?;
    assert_eq!(offsets.dims().dims(), &[4]);
    assert_eq!(
        offsets.to_vec1::<i64>()?,
        vec![0, 2, 3, 4],
        "exclusive offsets must be [0, c0, c0+c1, total]"
    );
    Ok(())
}

/// Freezing keeps the arithmetic identical and the state keys the same, and
/// unfreezing returns a module that still trains. Capacity-factor drop is
/// not implemented; this only pins the typestate transition.
#[test]
fn freeze_and_unfreeze_preserve_values_and_state() -> Result<()> {
    let moe = build_moe(8, 16)?;
    let x = ramp(vec![4, 8])?;
    let before = moe.forward(x.clone())?.to_vec1::<f32>()?;

    let frozen = moe.clone().freeze();
    let frozen_snapshot = collect_state::<Cpu, _>(&frozen)?;
    assert_eq!(
        frozen_snapshot.len(),
        5,
        "freezing changed which tensors are state"
    );
    let after = frozen.forward(x.clone())?.to_vec1::<f32>()?;
    assert_eq!(before, after, "freezing changed the computation");

    let thawed = frozen.unfreeze();
    assert_eq!(
        collect_state::<Cpu, _>(&thawed)?,
        collect_state::<Cpu, _>(&moe)?,
        "unfreeze changed the state keys or values"
    );
    let round_trip = thawed.forward(x)?.to_vec1::<f32>()?;
    assert_eq!(before, round_trip);
    Ok(())
}

/// `TrainMode::set_training` reaches the router and the expert array without
/// a compile-time refusal (the dense path has no dropout, so the forward is
/// unchanged either way — this pins the traversal, not a numeric difference).
#[test]
fn train_mode_traverses_the_router_and_experts() -> Result<()> {
    let mut moe = build_moe(8, 16)?;
    TrainMode::set_training(&mut moe, false);
    let x = ramp(vec![2, 8])?;
    let eval_a = moe.forward(x.clone())?.to_vec1::<f32>()?;
    let eval_b = moe.forward(x.clone())?.to_vec1::<f32>()?;
    assert_eq!(eval_a, eval_b, "eval mode is not deterministic");

    TrainMode::set_training(&mut moe, true);
    let train = moe.forward(x)?.to_vec1::<f32>()?;
    assert_eq!(eval_a, train, "train mode changed the dense arithmetic");
    Ok(())
}

/// NamedLayers names experts by index under the prefix, matching the
/// StatePath components the same array produces.
#[test]
fn named_layers_names_experts_by_index() -> Result<()> {
    let moe = build_moe(8, 16)?;
    let structure = moe.layer_structure("moe");
    assert_eq!(structure.len(), 1);

    let root = &structure[0];
    assert_eq!(root.name, "moe");
    assert_eq!(root.type_name, "MoE");

    let names: Vec<&str> = root.children.iter().map(|c| c.name.as_str()).collect();
    assert!(
        names.contains(&"moe.router"),
        "router missing from {names:?}"
    );
    assert!(
        names.contains(&"moe.experts.0"),
        "expert 0 missing from {names:?}"
    );
    assert!(
        names.contains(&"moe.experts.1"),
        "expert 1 missing from {names:?}"
    );
    Ok(())
}

/// Parameter and MAC counts sum across the gate and every expert.
#[test]
fn compute_stats_sums_gate_and_experts() -> Result<()> {
    let moe = build_moe(8, 16)?;
    let stats = moe.compute_stats(1);

    // Gate: bias-free [2, 8] = 16. Experts: two biased [16, 8] + [16] each
    // = 2 * (128 + 16) = 288. Total params = 304.
    assert_eq!(stats.params, 16 + 2 * (128 + 16));
    // MACs: gate 8*2 + two experts 8*16 each, batch 1.
    assert_eq!(stats.macs, 8 * 2 + 2 * (8 * 16));
    Ok(())
}

/// A mismatched last dimension is refused with `InvalidModuleState`-class
/// messaging rather than a panic or a silent broadcast.
#[test]
fn forward_refuses_a_width_that_does_not_match_the_gate() -> Result<()> {
    let moe = build_moe(8, 16)?;
    let bad = Tensor::<Dyn, Cpu>::zeros(vec![4, 7])?;
    let err = moe.forward(bad).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("7") && text.contains("8"),
        "error should name both widths, got: {text}"
    );
    Ok(())
}
