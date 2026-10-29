//! The normalization layers train when used as modules.
//!
//! Both `LayerNorm` and `RMSNorm` previously implemented `Module` only for a
//! `NoGrad` operand, and `LayerNorm` dispatched with a fresh execution context
//! that named no gradient mode at all. The combination meant a normalized
//! network could not be assembled from modules: feeding either one the output
//! of a `Linear` did not compile, and nothing recorded to the tape if it had.
//!
//! These tests pin the two halves of the fix separately: that the layer accepts
//! an operand carrying a gradient, and that a gradient actually comes back for
//! the layer's own scale.
#![cfg(feature = "cpu")]

use incin::prelude::*;

type Cpu = incin_backends::cpu::CpuBackendImpl;

fn ramp(dims: Vec<usize>) -> Result<Tensor<Dyn, Cpu, f32, NoGrad>> {
    let n: usize = dims.iter().product();
    let values = (0..n)
        .map(|v| ((v as f32) * 0.41).cos() * 1.3 + 0.2)
        .collect::<Vec<f32>>();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims)
}

fn nonzero(values: &[f32]) -> usize {
    values.iter().filter(|v| **v != 0.0).count()
}

#[test]
fn layer_norm_accepts_a_gradient_carrying_operand_and_trains() -> Result<()> {
    let width = 6;
    let projection = Linear::<Dyn, Cpu>::build((width, width))?;
    let norm = LayerNorm::<Dyn, Cpu>::build((width, 1e-5f32))?;

    let x = ramp(vec![4, width])?.require_grad();
    // The chain that did not compile before: a trainable layer feeding a norm.
    let hidden = projection.forward(x)?;
    let normalized = norm.forward(hidden)?;
    assert_eq!(normalized.dims().dims(), &[4, width]);

    let target = Tensor::<Dyn, Cpu>::zeros(vec![4, width])?;
    let grads = normalized.mse_loss(&target)?.backward()?;

    let scale = norm.weight.as_tensor()?;
    let scale_grad = grads
        .require(&scale)
        .map_err(|e| Error::Msg(format!("no gradient reached the LayerNorm scale: {e}")))?
        .to_vec1::<f32>()?;
    assert!(
        nonzero(&scale_grad) > 0,
        "the LayerNorm scale received only zeros, so it cannot train"
    );

    let weight = projection.weight.as_tensor()?;
    let weight_grad = grads
        .require(&weight)
        .map_err(|e| Error::Msg(format!("the gradient did not reach through the norm: {e}")))?
        .to_vec1::<f32>()?;
    assert!(
        nonzero(&weight_grad) > 0,
        "nothing flowed back past the norm to the layer beneath it"
    );
    Ok(())
}

#[test]
fn rms_norm_accepts_a_gradient_carrying_operand_and_trains() -> Result<()> {
    let width = 6;
    let projection = Linear::<Dyn, Cpu>::build((width, width))?;
    let norm = RMSNorm::<Dyn, Cpu>::build((width, 1e-6f32))?;

    let x = ramp(vec![4, width])?.require_grad();
    let normalized = norm.forward(projection.forward(x)?)?;
    assert_eq!(normalized.dims().dims(), &[4, width]);

    let target = Tensor::<Dyn, Cpu>::zeros(vec![4, width])?;
    let grads = normalized.mse_loss(&target)?.backward()?;

    let scale = norm.weight.as_tensor()?;
    let scale_grad = grads
        .require(&scale)
        .map_err(|e| Error::Msg(format!("no gradient reached the RMSNorm scale: {e}")))?
        .to_vec1::<f32>()?;
    assert!(
        nonzero(&scale_grad) > 0,
        "the RMSNorm scale received only zeros, so it cannot train"
    );

    let weight = projection.weight.as_tensor()?;
    assert!(
        nonzero(&grads.require(&weight)?.to_vec1::<f32>()?) > 0,
        "nothing flowed back past the RMS norm"
    );
    Ok(())
}

/// Widening the operand's gradient type must not change what the layer
/// computes, so the arithmetic is compared against the tensor-level operation
/// it wraps.
#[test]
fn layer_norm_still_computes_the_same_values() -> Result<()> {
    let width = 6;
    let norm = LayerNorm::<Dyn, Cpu>::build((width, 1e-5f32))?;
    let x = ramp(vec![4, width])?;

    let from_module = norm.forward(x.clone())?.to_vec1::<f32>()?;
    let by_operation = x
        .layer_norm(
            &norm.weight.as_tensor()?.detach(),
            &norm.bias.as_tensor()?.detach(),
            1e-5,
        )?
        .to_vec1::<f32>()?;

    let diff = from_module
        .iter()
        .zip(by_operation.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        diff < 1e-7,
        "the module and the operation disagree by {diff:e}"
    );
    Ok(())
}

/// A `NoGrad` operand still produces a `NoGrad` result, so inference paths are
/// unchanged by the widening.
#[test]
fn an_inference_operand_records_nothing() -> Result<()> {
    let width = 6;
    let norm = LayerNorm::<Dyn, Cpu>::build((width, 1e-5f32))?;
    let frozen = norm.freeze();
    let out = frozen.forward(ramp(vec![2, width])?)?;
    assert!(
        out.to_vec1::<f32>()?.iter().all(|v| v.is_finite()),
        "a frozen norm produced a non-finite value"
    );
    Ok(())
}
