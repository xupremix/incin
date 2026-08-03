#![cfg(feature = "cpu")]

use std::collections::BTreeMap;

use incin::prelude::*;
use incin::{Adam, AdamW, SGD};

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

/// Get linear and grads.
fn get_linear_and_grads() -> Result<(
    Linear<s![10, 5], CpuBackendImpl>,
    incin::Gradients<<CpuBackendImpl as Backend>::Grads>,
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

#[test]
fn test_adam_optimizer_state_dict_checkpointing() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim1 = Adam::<CpuBackendImpl>::new(linear.parameters(), 0.01);

    // Step 1
    optim1.step(&grads)?;
    assert_eq!(optim1.step_count(), 1);

    // Save optimizer state
    let mut state = BTreeMap::new();
    optim1.state_dict("", &mut state);
    assert!(!state.is_empty());

    // Create a new optimizer instance and load state
    let mut optim2 = Adam::<CpuBackendImpl>::new(linear.parameters(), 0.01);
    optim2.load_state_dict("", &state)?;
    optim2.set_step_count(optim1.step_count());

    assert_eq!(optim2.step_count(), 1);
    // Step again with restored momentum state
    optim2.step(&grads)?;
    assert_eq!(optim2.step_count(), 2);

    Ok(())
}

#[test]
fn test_adamw_optimizer_state_dict_checkpointing() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim1 = AdamW::<CpuBackendImpl>::new(linear.parameters(), 0.01);

    optim1.step(&grads)?;
    assert_eq!(optim1.step_count(), 1);

    let mut state = BTreeMap::new();
    optim1.state_dict("", &mut state);
    assert!(!state.is_empty());

    let mut optim2 = AdamW::<CpuBackendImpl>::new(linear.parameters(), 0.01);
    optim2.load_state_dict("", &state)?;
    optim2.set_step_count(optim1.step_count());

    assert_eq!(optim2.step_count(), 1);
    optim2.step(&grads)?;
    assert_eq!(optim2.step_count(), 2);

    Ok(())
}
