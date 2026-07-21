use kindle::prelude::*;
use kindle::{Adam, AdamW, SGD};

/// Auto-generated documentation for CpuBackend.
type CpuBackend = DefaultBackend;

/// Auto-generated documentation for get_linear_and_grads.
fn get_linear_and_grads() -> Result<(
    Linear<s![10, 5], CpuBackend>,
    kindle::Gradients<<CpuBackend as Backend>::Grads>,
)> {
    let linear = Linear::<s![10, 5], CpuBackend>::new()?;
    let input = Tensor::<s![2, 10], CpuBackend>::ones(())?;
    let target = Tensor::<s![2, 5], CpuBackend>::zeros(())?;
    let out = linear.forward(input)?;
    let loss = out.mse_loss(&target)?;
    let grads = loss.backward()?;
    Ok((linear, grads))
}

#[test]
/// Auto-generated documentation for test_sgd.
fn test_sgd() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = SGD::<CpuBackend>::new(linear.parameters(), 0.01);

    optim.step(&grads)?;

    Ok(())
}

#[test]
/// Auto-generated documentation for test_adam.
fn test_adam() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = Adam::<CpuBackend>::new(linear.parameters(), 0.001);

    optim.step(&grads)?;

    Ok(())
}

#[test]
/// Auto-generated documentation for test_adamw.
fn test_adamw() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = AdamW::<CpuBackend>::new(linear.parameters(), 0.001);

    optim.step(&grads)?;

    Ok(())
}
