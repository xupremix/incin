//! End-to-end autograd for the scan/product reductions whose catalog rows
//! declare `GradientRule::Defined`: a gradient must arrive at the tracked
//! input through the public API, and an optimizer must be able to consume it.
//!
//! The kernel-level backwards are gradchecked in `incin-backends`; this file
//! pins the wiring - dispatch under an active GradMode, tape recording
//! through `execute_shaped`, and gradient delivery through `Gradients` - so a
//! future regression cannot hide at a layer boundary.
#![cfg(feature = "cpu")]

use incin::prelude::*;

/// B.
type B = incin_backends::cpu::CpuBackendImpl;

#[test]
fn cumsum_backward_delivers_suffix_sum_weights_through_the_public_api() -> Result<()> {
    let x = Tensor::<s![4], B, f32, Grad>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    let loss = x.cumsum(axis!(0))?.sum_all()?;
    // sum(cumsum(x)) = 4*x0 + 3*x1 + 2*x2 + 1*x3.
    assert!((loss.to_scalar::<f32>()? - 20.0).abs() < 1e-5);
    let grads = loss.backward()?;
    assert_eq!(
        grads.require(&x)?.to_vec1::<f32>()?,
        vec![4.0, 3.0, 2.0, 1.0]
    );
    Ok(())
}

#[test]
fn prod_all_backward_delivers_excluding_products_through_the_public_api() -> Result<()> {
    let x = Tensor::<s![3], B, f32, Grad>::from_slice(&[2.0, 4.0, 3.0], ())?;
    let loss = x.clone().prod_all()?;
    assert!((loss.to_scalar::<f32>()? - 24.0).abs() < 1e-5);
    let grads = loss.backward()?;
    assert_eq!(grads.require(&x)?.to_vec1::<f32>()?, vec![12.0, 6.0, 8.0]);
    Ok(())
}

#[test]
fn an_optimizer_step_consumes_a_gradient_that_flows_through_cumsum() -> Result<()> {
    // A linear model whose output passes through a scan before the loss: the
    // parameters must actually move. Before the CPU kernel recorded its
    // backward, this graph silently produced no parameter gradient at all.
    let model = Linear::<s![3, 2], B>::build(())?;
    let mut optim = SGD::<B>::from_module(&model, 0.5)?;

    let before = model.weight.as_tensor()?.to_vec1::<f32>()?;

    let x = Tensor::<s![1, 3], B>::ones(())?;
    let scanned = model.forward(x)?.cumsum(axis!(1))?;
    let loss = scanned.mul_exact(&scanned)?.sum_all()?;
    let grads = loss.backward()?;
    optim.step(&grads)?;

    let after = model.weight.as_tensor()?.to_vec1::<f32>()?;
    assert_ne!(
        before, after,
        "parameters must move when gradients flow through a cumsum"
    );
    Ok(())
}
