use kindle::optim::{Optimizer, SGD};
use kindle::prelude::*;

type CpuBackend = DefaultBackend;

#[test]
fn test_simple_linear_regression() -> Result<()> {
    let mut model = Linear::<s![1, 1], CpuBackend>::new()?;
    let mut optim = SGD::<CpuBackend>::new(model.parameters(), 0.01);

    // We just create zeros to test the autograd and optimizer pipeline
    let x = Tensor::<s![4, 1], CpuBackend>::zeros(())?;
    let y = Tensor::<s![4, 1], CpuBackend>::zeros(())?;

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
