#![cfg(feature = "cpu")]
#![allow(clippy::type_complexity)]

use std::collections::BTreeMap;

use incin::backend_authoring::{HostInterop, VariableBackend};
use incin::optim::ParameterGroup;
use incin::prelude::*;
use incin::{Adam, AdamW, SGD};

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
        .map(|(name, var)| {
            let storage = CpuBackendImpl::var_as_tensor::<f32>(var)?;
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
    incin::Gradients<CpuBackendImpl>,
)> {
    let linear = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    let grads = grads_for(&linear)?;
    Ok((linear, grads))
}

/// Gradients for `linear` as it is *now*.
///
/// A committed optimizer step assigns fresh storage to every parameter it
/// updates, and gradients are looked up by the identity of the storage they
/// were recorded against. So a `Gradients` value is spent once: reusing it for
/// a second step matches nothing. That used to be invisible, because a step
/// that matched nothing still returned `Ok(())` and still incremented the step
/// counter, which is exactly what the checkpoint cases below asserted on.
fn grads_for(
    linear: &Linear<s![10, 5], CpuBackendImpl>,
) -> Result<incin::Gradients<CpuBackendImpl>> {
    let input = Tensor::<s![2, 10], CpuBackendImpl>::ones(())?;
    let target = Tensor::<s![2, 5], CpuBackendImpl>::zeros(())?;
    let out = linear.forward(input)?;
    let loss = out.mse_loss(&target)?;
    loss.backward()
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
    // Step again with restored momentum state, against gradients recorded after
    // the first step replaced the parameter storage.
    let grads = grads_for(&linear)?;
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
    // As above: fresh gradients, because the first step replaced the storage
    // the previous ones were recorded against.
    let grads = grads_for(&linear)?;
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

/// A step whose gradients reach no parameter in the group is an error, not a
/// silent no-op.
///
/// Before this contract, every optimizer here skipped a parameter it had no
/// gradient for and returned `Ok(())` regardless — so a training loop that
/// never reached the backward pass, or reached it on a different thread from
/// the one that recorded the forward pass, ran to completion with parameters
/// that never moved. Skipping *some* parameters stays legal, because a
/// parameter the forward pass did not use genuinely has nothing to apply.
#[test]
fn a_step_that_reaches_no_parameter_is_refused_rather_than_committing_nothing() {
    let model = Linear::<s![4, 2], CpuBackendImpl>::build(()).unwrap();

    // Gradients from a graph that has nothing to do with `model`, which is the
    // shape every one of the failure modes above arrives in.
    let unrelated = Tensor::<s![2], CpuBackendImpl>::ones(())
        .unwrap()
        .require_grad()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();

    let mut sgd = SGD::<CpuBackendImpl>::from_module(&model, 0.01).unwrap();
    let error = sgd
        .step(&unrelated)
        .expect_err("a step that updates nothing must not report success");
    let rendered = error.to_string();
    assert!(
        rendered.contains("no parameter in this group received a gradient"),
        "the error must name the actual problem, got: {rendered}"
    );

    let mut adam = Adam::<CpuBackendImpl>::from_module(&model, 1e-3).unwrap();
    adam.step(&unrelated)
        .expect_err("Adam must refuse the same step");

    let mut adamw = AdamW::<CpuBackendImpl>::from_module(&model, 1e-4).unwrap();
    adamw
        .step(&unrelated)
        .expect_err("AdamW must refuse the same step");
}

/// The refusal above does not fire when the backward pass really did reach the
/// group, including when it reached only part of it.
#[test]
fn a_step_that_reaches_the_group_still_commits() {
    let model = Linear::<s![4, 2], CpuBackendImpl>::build(()).unwrap();
    let input = Tensor::<s![1, 4], CpuBackendImpl>::ones(())
        .unwrap()
        .require_grad();
    let grads = model
        .forward(input)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();

    let mut sgd = SGD::<CpuBackendImpl>::from_module(&model, 0.01).unwrap();
    sgd.step(&grads)
        .expect("a step whose gradients reach the group must succeed");
}

/// `clip_grad_norm` rescales the whole gradient set to the requested total
/// norm, and reports the norm the set had before it did.
#[test]
fn clipping_rescales_the_group_to_the_requested_total_norm() {
    use incin::optim::clip_grad_norm;

    let model = Linear::<s![10, 5], CpuBackendImpl>::build(()).unwrap();
    let group = ParameterGroup::<CpuBackendImpl, f32>::from_module(&model).unwrap();
    let mut grads = grads_for(&model).unwrap();

    let squared_total = |grads: &incin::Gradients<CpuBackendImpl>| -> f64 {
        let mut total = 0.0;
        for (_, var) in group.iter() {
            let tensor = <CpuBackendImpl as VariableBackend>::var_as_tensor::<f32>(var).unwrap();
            let Some(grad) =
                <CpuBackendImpl as incin::backend_authoring::AutogradBackend>::get_grad::<f32>(
                    &tensor,
                    grads.as_backend(),
                )
                .unwrap()
            else {
                continue;
            };
            for value in
                <CpuBackendImpl as incin::backend_authoring::HostReadback>::float_to_vec1::<f32>(
                    &grad,
                )
                .unwrap()
            {
                total += value * value;
            }
        }
        total
    };

    let before = squared_total(&grads).sqrt();
    assert!(
        before > 0.0,
        "the fixture must produce a non-zero gradient to clip"
    );

    // A threshold well under the current norm, so the call has to do work.
    let target = before / 4.0;
    let reported = clip_grad_norm(&group, &mut grads, target).unwrap();
    assert!(
        (reported - before).abs() <= 1e-5,
        "the reported norm must be the one before clipping: got {reported}, expected {before}"
    );

    let after = squared_total(&grads).sqrt();
    assert!(
        (after - target).abs() <= 1e-4,
        "the clipped norm must be the requested one: got {after}, expected {target}"
    );
}

/// A group already inside the threshold is left exactly as it was.
#[test]
fn clipping_below_the_threshold_changes_nothing() {
    use incin::optim::clip_grad_norm;

    let model = Linear::<s![10, 5], CpuBackendImpl>::build(()).unwrap();
    let group = ParameterGroup::<CpuBackendImpl, f32>::from_module(&model).unwrap();
    let mut grads = grads_for(&model).unwrap();

    let snapshot = |grads: &incin::Gradients<CpuBackendImpl>| -> Vec<f64> {
        let mut values = Vec::new();
        for (_, var) in group.iter() {
            let tensor = <CpuBackendImpl as VariableBackend>::var_as_tensor::<f32>(var).unwrap();
            if let Some(grad) =
                <CpuBackendImpl as incin::backend_authoring::AutogradBackend>::get_grad::<f32>(
                    &tensor,
                    grads.as_backend(),
                )
                .unwrap()
            {
                values.extend(<CpuBackendImpl as incin::backend_authoring::HostReadback>::float_to_vec1::<f32>(&grad).unwrap());
            }
        }
        values
    };

    let before = snapshot(&grads);
    let reported = clip_grad_norm(&group, &mut grads, 1.0e9).unwrap();
    assert!(reported < 1.0e9);
    assert_eq!(before, snapshot(&grads));
}

/// A non-positive threshold is a configuration error, not a silent clamp to
/// zero gradients.
#[test]
fn clipping_rejects_a_threshold_that_is_not_positive_and_finite() {
    use incin::optim::clip_grad_norm;

    let model = Linear::<s![10, 5], CpuBackendImpl>::build(()).unwrap();
    let group = ParameterGroup::<CpuBackendImpl, f32>::from_module(&model).unwrap();
    let mut grads = grads_for(&model).unwrap();

    for threshold in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        clip_grad_norm(&group, &mut grads, threshold)
            .expect_err("a threshold of {threshold} must be refused");
    }
}

fn gradient_values(
    group: &ParameterGroup<CpuBackendImpl, f32>,
    grads: &incin::Gradients<CpuBackendImpl>,
) -> Vec<f64> {
    let mut values = Vec::new();
    for (_, var) in group.iter() {
        let tensor = <CpuBackendImpl as VariableBackend>::var_as_tensor::<f32>(var).unwrap();
        if let Some(grad) =
            <CpuBackendImpl as incin::backend_authoring::AutogradBackend>::get_grad::<f32>(
                &tensor,
                grads.as_backend(),
            )
            .unwrap()
        {
            values.extend(
                <CpuBackendImpl as incin::backend_authoring::HostReadback>::float_to_vec1::<f32>(
                    &grad,
                )
                .unwrap(),
            );
        }
    }
    values
}

/// `clip_grad_value` clamps every element independently into
/// `[-clip_value, clip_value]`, unlike `clip_grad_norm`'s whole-set rescale:
/// an element already inside the bound is untouched, one outside it is
/// flattened exactly to the bound it crossed.
#[test]
fn clip_grad_value_clamps_every_element_independently() {
    use incin::optim::clip_grad_value;

    let model = Linear::<s![10, 5], CpuBackendImpl>::build(()).unwrap();
    let group = ParameterGroup::<CpuBackendImpl, f32>::from_module(&model).unwrap();
    let mut grads = grads_for(&model).unwrap();

    let before = gradient_values(&group, &grads);
    assert!(
        before.iter().any(|&v| v.abs() > 0.0),
        "the fixture must produce a non-zero gradient to clip"
    );

    // A bound tight enough that at least one element must actually clip,
    // loose enough that at least one stays untouched, so both branches of
    // the per-element clamp are exercised rather than only one.
    let max_abs = before.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    let clip_value = max_abs / 2.0;
    assert!(
        clip_value > 0.0,
        "the fixture's largest-magnitude gradient must be nonzero"
    );

    clip_grad_value(&group, &mut grads, clip_value).unwrap();
    let after = gradient_values(&group, &grads);

    assert_eq!(before.len(), after.len());
    assert!(
        after.iter().any(|&v| (v.abs() - clip_value).abs() <= 1e-5),
        "at least one element must have been flattened exactly to the bound"
    );
    for (&pre, &post) in before.iter().zip(after.iter()) {
        assert!(
            post.abs() <= clip_value + 1e-5,
            "every element must be within the bound after clamping: {post} vs {clip_value}"
        );
        if pre.abs() <= clip_value {
            assert!(
                (pre - post).abs() <= 1e-5,
                "an element already inside the bound must be left untouched: {pre} vs {post}"
            );
        }
    }
}

/// A gradient set already inside the bound is left exactly as it was.
#[test]
fn clip_grad_value_below_the_bound_changes_nothing() {
    use incin::optim::clip_grad_value;

    let model = Linear::<s![10, 5], CpuBackendImpl>::build(()).unwrap();
    let group = ParameterGroup::<CpuBackendImpl, f32>::from_module(&model).unwrap();
    let mut grads = grads_for(&model).unwrap();

    let before = gradient_values(&group, &grads);
    clip_grad_value(&group, &mut grads, 1.0e9).unwrap();
    assert_eq!(before, gradient_values(&group, &grads));
}

/// A non-positive bound is a configuration error, not a silent clamp to zero.
#[test]
fn clip_grad_value_rejects_a_bound_that_is_not_positive_and_finite() {
    use incin::optim::clip_grad_value;

    let model = Linear::<s![10, 5], CpuBackendImpl>::build(()).unwrap();
    let group = ParameterGroup::<CpuBackendImpl, f32>::from_module(&model).unwrap();
    let mut grads = grads_for(&model).unwrap();

    for bound in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        clip_grad_value(&group, &mut grads, bound).expect_err("a bound of {bound} must be refused");
    }
}
