use kindle::prelude::*;
use kindle::{Adam, AdamW, SGD};

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = kindle_backends::cpu::CpuBackendImpl;

/// Get linear and grads.
fn get_linear_and_grads() -> Result<(
    Linear<s![10, 5], CpuBackendImpl>,
    kindle::Gradients<<CpuBackendImpl as Backend>::Grads>,
)> {
    let linear = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    let input = Tensor::<s![2, 10], CpuBackendImpl>::ones(())?;
    let target = Tensor::<s![2, 5], CpuBackendImpl>::zeros(())?;
    let out = linear.forward(input)?;
    let loss = out.mse_loss(&target)?;
    let grads = loss.backward()?;
    Ok((linear, grads))
}

#[test]
/// Test sgd.
fn test_sgd() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = SGD::<CpuBackendImpl>::new(linear.parameters(), 0.01);

    optim.step(&grads)?;

    Ok(())
}

#[test]
/// Test adam.
fn test_adam() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = Adam::<CpuBackendImpl>::new(linear.parameters(), 0.001);

    optim.step(&grads)?;

    Ok(())
}

#[test]
/// Test adamw.
fn test_adamw() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = AdamW::<CpuBackendImpl>::new(linear.parameters(), 0.001);

    optim.step(&grads)?;

    Ok(())
}
