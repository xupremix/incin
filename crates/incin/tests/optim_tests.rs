#![cfg(feature = "cpu")]

use std::collections::BTreeMap;

use incin::prelude::*;
use incin::optim::ParameterGroup;
use incin::{Adam, AdamW, SGD};
use incin::backend_authoring::HostInterop;
use incin::backend_authoring::AutogradBackend;

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

#[test]
fn optimizer_parameter_groups_collect_through_typed_state_visitors() {
    let linear = Linear::<s![10, 5], CpuBackendImpl>::build(()).unwrap();
    let group = ParameterGroup::<CpuBackendImpl, f32>::from_module(&linear).unwrap();
    assert_eq!(group.len(), 2);
    assert!(!group.is_empty());
    let _optimizer = SGD::<CpuBackendImpl>::from_group(group, 0.01);
}

fn parameter_bytes(
    linear: &Linear<s![10, 5], CpuBackendImpl>,
) -> Result<BTreeMap<String, Vec<u8>>> {
    ParameterGroup::<CpuBackendImpl, f32>::from_module(linear)
        .unwrap()
        .iter()
        .into_iter()
        .map(|(name, var)| {
            let storage = CpuBackendImpl::var_as_tensor::<f32>(&var)?;
            Ok((name.clone(), CpuBackendImpl::to_bytes::<f32>(&storage)?))
        })
        .collect()
}

fn state_bytes(
    state: &BTreeMap<String, Tensor<Dyn, CpuBackendImpl>>,
) -> Result<BTreeMap<String, Vec<u8>>> {
    state
        .iter()
        .map(|(name, tensor)| {
            Ok((
                name.clone(),
                CpuBackendImpl::to_bytes::<f32>(tensor.inner())?,
            ))
        })
        .collect()
}

/// Get linear and grads.
fn get_linear_and_grads() -> Result<(
    Linear<s![10, 5], CpuBackendImpl>,
    incin::Gradients<<CpuBackendImpl as AutogradBackend>::Grads>,
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
    let mut optim = SGD::<CpuBackendImpl>::from_module(&linear, 0.01)?;

    optim.step(&grads)?;

    Ok(())
}

#[test]
/// Test adam.
fn test_adam() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = Adam::<CpuBackendImpl>::from_module(&linear, 0.001)?;

    optim.step(&grads)?;

    Ok(())
}

#[test]
/// Test adamw.
fn test_adamw() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = AdamW::<CpuBackendImpl>::from_module(&linear, 0.001)?;

    optim.step(&grads)?;

    Ok(())
}

#[test]
fn test_adam_optimizer_state_dict_checkpointing() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim1 = Adam::<CpuBackendImpl>::from_module(&linear, 0.01)?;

    // Step 1
    optim1.step(&grads)?;
    assert_eq!(optim1.step_count(), 1);

    // Save optimizer state
    let mut state = BTreeMap::new();
    optim1.state_dict("", &mut state)?;
    assert!(!state.is_empty());

    // Create a new optimizer instance and load state
    let mut optim2 = Adam::<CpuBackendImpl>::from_module(&linear, 0.01)?;
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
    let mut optim1 = AdamW::<CpuBackendImpl>::from_module(&linear, 0.01)?;

    optim1.step(&grads)?;
    assert_eq!(optim1.step_count(), 1);

    let mut state = BTreeMap::new();
    optim1.state_dict("", &mut state)?;
    assert!(!state.is_empty());

    let mut optim2 = AdamW::<CpuBackendImpl>::from_module(&linear, 0.01)?;
    optim2.load_state_dict("", &state)?;
    optim2.set_step_count(optim1.step_count());

    assert_eq!(optim2.step_count(), 1);
    optim2.step(&grads)?;
    assert_eq!(optim2.step_count(), 2);

    Ok(())
}

#[test]
fn adam_step_rolls_back_parameters_state_and_counter_on_backend_failure() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let before = parameter_bytes(&linear)?;
    let mut optim = Adam::<CpuBackendImpl>::from_module(&linear, 0.001)?;
    let mut state_before = BTreeMap::new();
    optim.state_dict("", &mut state_before)?;

    let failure = incin::test_utils::fail_assign_on(2);
    let error = optim.step(&grads).unwrap_err();
    drop(failure);

    assert!(matches!(
        error,
        Error::Backend(BackendError::Execution { .. })
    ));
    assert_eq!(parameter_bytes(&linear)?, before);
    assert_eq!(optim.step_count(), 0);
    let mut state_after = BTreeMap::new();
    optim.state_dict("", &mut state_after)?;
    assert_eq!(
        state_after.keys().collect::<Vec<_>>(),
        state_before.keys().collect::<Vec<_>>()
    );

    // The injected failure is one-shot and the rollback leaves a usable
    // optimizer, not merely one whose visible values happen to match.
    optim.step(&grads)?;
    assert_eq!(optim.step_count(), 1);
    Ok(())
}

#[test]
fn adam_step_overflow_preserves_parameters_and_state() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let before = parameter_bytes(&linear)?;
    let mut optim = Adam::<CpuBackendImpl>::from_module(&linear, 0.001)?;
    optim.set_step_count(usize::MAX);

    assert!(matches!(
        optim.step(&grads),
        Err(Error::ArithmeticOverflow {
            operation: "adam_step",
            ..
        })
    ));
    assert_eq!(parameter_bytes(&linear)?, before);
    assert_eq!(optim.step_count(), usize::MAX);
    let mut state = BTreeMap::new();
    optim.state_dict("", &mut state)?;
    assert!(state.is_empty());
    Ok(())
}

#[test]
fn malformed_adam_state_load_is_typed_and_transactional() -> Result<()> {
    let (linear, grads) = get_linear_and_grads()?;
    let mut optim = Adam::<CpuBackendImpl>::from_module(&linear, 0.001)?;
    optim.step(&grads)?;

    let mut valid_state = BTreeMap::new();
    optim.state_dict("", &mut valid_state)?;
    let before = state_bytes(&valid_state)?;
    let first_key = valid_state.keys().next().cloned().unwrap();
    valid_state.remove(&first_key);

    assert!(matches!(
        optim.load_state_dict("", &valid_state),
        Err(Error::InvalidModuleState {
            operation: "adam_load_state_dict",
            ..
        })
    ));
    let mut after_state = BTreeMap::new();
    optim.state_dict("", &mut after_state)?;
    assert_eq!(state_bytes(&after_state)?, before);
    assert_eq!(optim.step_count(), 1);
    Ok(())
}
