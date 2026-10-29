//! Dropless grouped-GEMM MoE module tests (issue #102 prototype).
//!
//! The module under test is `incin::experimental::{DroplessMoE, ExpertMlp}`:
//! option-C dropless routing (static `[T*K, D]` buffer + `[E+1]` offsets,
//! one `grouped_matmul` per projection) behind a typed `Module` with static
//! outer shapes, `E`/`TOPK` const generics, and the aux loss in `Output`.
//! These tests pin the prototype contract: geometry/aux values, gate-only
//! gradients, `NoGrad` routing indices by type, state roundtrip, and
//! expert-count mismatch refusal on load.
#![cfg(feature = "cpu")]

use incin::experimental::{DroplessMoE, ExpertMlp};
use incin::nn::Module;
use incin::prelude::*;
use incin_core::dist::Local;

type Cpu = incin_backends::cpu::CpuBackendImpl;

const T: usize = 6;
const D: usize = 8;
const DFF: usize = 16;
const E: usize = 4;
const TOPK: usize = 2;

type MoE = DroplessMoE<E, TOPK, Cpu>;

fn build_moe() -> Result<MoE> {
    MoE::build(D, DFF, (), (), || ExpertMlp::<Cpu>::build(D, DFF, (), ()))
}

fn is_finite_vec(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

#[test]
fn forward_shapes_and_aux() -> Result<()> {
    let moe = build_moe()?;
    let x = Tensor::<Dyn, Cpu>::randn(vec![T, D])?;
    let (y, aux) = moe.forward(x)?;
    assert_eq!(y.dims().as_ref(), &[T, D]);
    assert!(aux.dims().as_ref().is_empty());
    let aux_value = aux.to_vec1::<f32>()?;
    assert_eq!(aux_value.len(), 1);
    assert!(
        is_finite_vec(&aux_value) && aux_value[0] >= 0.0,
        "aux loss must be a finite nonnegative scalar, got {aux_value:?}"
    );
    let out = y.to_vec1::<f32>()?;
    assert_eq!(out.len(), T * D);
    assert!(is_finite_vec(&out), "combined output must be finite");
    Ok(())
}

#[test]
fn forward_is_deterministic_for_fixed_params() -> Result<()> {
    let moe = build_moe()?;
    let x = Tensor::<Dyn, Cpu>::randn(vec![T, D])?;
    let (y1, aux1) = moe.forward(x.clone())?;
    let (y2, aux2) = moe.forward(x)?;
    assert_eq!(y1.to_vec1::<f32>()?, y2.to_vec1::<f32>()?);
    assert_eq!(aux1.to_vec1::<f32>()?, aux2.to_vec1::<f32>()?);
    Ok(())
}

#[test]
fn gate_weights_receive_gradients() -> Result<()> {
    let moe = build_moe()?;
    let x = Tensor::<Dyn, Cpu>::randn(vec![T, D])?.require_grad();
    // Through the combined output ...
    // (one forward per backward: the tape drains on the walk that
    // consumes it, and there is no retain_graph — so the aux path gets
    // its own fresh forward below.)
    let (y, _) = moe.forward(x.clone())?;
    let grads = y.sum_all()?.backward()?;
    let gate = moe.router.gate.weight.as_tensor()?;
    let gate_grad = grads.require(&gate)?;
    let values = gate_grad.to_vec1::<f32>()?;
    assert!(
        is_finite_vec(&values),
        "gate gradient through the output must be finite"
    );
    assert!(
        values.iter().any(|v| *v != 0.0),
        "gate gradient through the output must be nonzero"
    );
    // ... and through the aux loss independently.
    let (_, aux) = moe.forward(x)?;
    let aux_grads = aux.backward()?;
    let aux_gate_grad = aux_grads.require(&gate)?;
    assert!(is_finite_vec(&aux_gate_grad.to_vec1::<f32>()?));
    Ok(())
}

#[test]
fn expert_weights_receive_gradients_from_grad_input() -> Result<()> {
    // Documented prototype contract: `grouped_matmul` derives its record
    // mode from the lhs marker, so expert gradients need a grad-marked
    // input (unlike the dense masked MoE, which trains experts from
    // NoGrad inputs — a genuine B-vs-C gap, recorded in the module docs).
    let moe = build_moe()?;
    let x = Tensor::<Dyn, Cpu>::randn(vec![T, D])?.require_grad();
    let (y, _) = moe.forward(x)?;
    let grads = y.sum_all()?.backward()?;
    let expert_w = moe.experts[0].up.weight.as_tensor()?;
    let expert_grad = grads.require(&expert_w)?;
    assert!(is_finite_vec(&expert_grad.to_vec1::<f32>()?));
    Ok(())
}

#[test]
fn routing_indices_carry_no_gradient_by_type() -> Result<()> {
    // Indices, permutation, offsets, and counts are NoGrad *by type*:
    // this helper only compiles for a NoGrad-marked u32 tensor, so the
    // call below is a compile-time proof, not a runtime check.
    fn assert_nograd(_: &Dense<Dyn, Cpu, u32, NoGrad, Local>) {}
    let moe = build_moe()?;
    let x = Tensor::<Dyn, Cpu>::randn(vec![T, D])?;
    let routing = moe.router.forward(x)?;
    assert_nograd(&routing.indices);
    Ok(())
}

#[test]
fn save_load_roundtrip_is_exact() -> Result<()> {
    let moe = build_moe()?;
    let path = std::env::temp_dir().join(format!(
        "incin-moe-dropless-{}.safetensors",
        std::process::id()
    ));
    moe.save(Format::Safetensors, &path)?;
    let mut reloaded = build_moe()?;
    reloaded.load(Format::Safetensors, &path)?;
    let x = Tensor::<Dyn, Cpu>::randn(vec![T, D])?;
    let (before, _) = moe.forward(x.clone())?;
    let (after, _) = reloaded.forward(x)?;
    assert_eq!(before.to_vec1::<f32>()?, after.to_vec1::<f32>()?);
    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn expert_count_mismatch_refuses_load() -> Result<()> {
    let moe = build_moe()?;
    let path = std::env::temp_dir().join(format!(
        "incin-moe-dropless-mismatch-{}.safetensors",
        std::process::id()
    ));
    moe.save(Format::Safetensors, &path)?;
    let mut other = DroplessMoE::<2, 1, Cpu>::build(D, DFF, (), (), || {
        ExpertMlp::<Cpu>::build(D, DFF, (), ())
    })?;
    let result = other.load(Format::Safetensors, &path);
    let _ = std::fs::remove_file(&path);
    assert!(
        result.is_err(),
        "loading E=4 state into an E=2 module must refuse, not partially load"
    );
    Ok(())
}
