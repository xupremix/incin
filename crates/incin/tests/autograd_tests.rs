//! Integration coverage for `test_simple_linear_regression` on the documented public surface.
#![cfg(feature = "cpu")]

use incin::prelude::*;
use incin::{Optimizer, SGD};
use incin_core::exec::check_gradients;

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

#[test]
/// Test simple linear regression.
fn test_simple_linear_regression() -> Result<()> {
    let model = Linear::<s![1, 1], CpuBackendImpl>::build(())?;
    let mut optim = SGD::<CpuBackendImpl>::from_module(&model, 0.01)?;

    // We just create zeros to test the autograd and optimizer pipeline
    let x = Tensor::<s![4, 1], CpuBackendImpl>::zeros(())?;
    let y = Tensor::<s![4, 1], CpuBackendImpl>::zeros(())?;

    for _ in 0..2 {
        let pred = model.forward(x.clone())?;
        let loss = pred.mse_loss(&y)?;
        let grads = loss.backward()?;
        optim.step(&grads)?;
    }

    let pred = model.forward(x)?;
    let loss = pred.mse_loss(&y)?;

    // We just verify it compiles and runs without panicking.
    let _loss_val = loss.to_scalar::<f32>()?;

    Ok(())
}

#[test]
/// Test a checked backward pass over finite gradients.
///
/// `GRD-005` replaced `Backend::backward_with_nan_check` with a `NanPolicy`
/// axis: the check is a scope around the ordinary `backward`, and a failure is
/// a returned error rather than a panic.
fn test_backward_under_nan_checking_succeeds() -> Result<()> {
    let model = Linear::<s![1, 1], CpuBackendImpl>::build(())?;
    let mut optim = SGD::<CpuBackendImpl>::from_module(&model, 0.01)?;

    let x = Tensor::<s![4, 1], CpuBackendImpl>::zeros(())?;
    let y = Tensor::<s![4, 1], CpuBackendImpl>::zeros(())?;

    let pred = model.forward(x.clone())?;
    let loss = pred.mse_loss(&y)?;
    let grads = check_gradients(|| loss.backward())?;
    optim.step(&grads)?;

    Ok(())
}
