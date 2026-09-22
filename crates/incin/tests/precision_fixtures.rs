//! `UX-001`, issue #2: precision planning and f32 trainer execution fixtures.

#![cfg(all(feature = "train", feature = "cpu"))]

use incin::experimental::training::{TrainError, Trainer};
use incin::prelude::*;
use incin_core::backend_authoring::StorageBackend;
use incin_core::exec::{
    ExecutionContext, ExecutionPolicy, PrecisionChoice, RuntimePrecisionPolicy,
};
use incin_core::tensor::dtype::{ConstDType, f16};

type Backend = DefaultBackend;
type TestResult = core::result::Result<(), Box<dyn core::error::Error>>;

/// `UX-001`: explicitly disabling the mixed-f16 safeguard is a typed refusal.
#[test]
fn mixed_f16_without_scaling_is_rejected() {
    let policy = RuntimePrecisionPolicy::mixed_f16();
    assert_eq!(policy.active_dtype(), Some(<f16 as ConstDType>::DESCRIPTOR));
    assert_eq!(
        policy.accumulator(),
        PrecisionChoice::Exact(<f32 as ConstDType>::DESCRIPTOR)
    );
    let error = Trainer::plan()
        .devices(DeviceSet::cpu())
        .precision(policy)
        .loss_scaling(LossScaling::None)
        .build()
        .expect_err("mixed-f16 requires scaling");
    match &error {
        TrainError::UnsupportedPrecision {
            active_dtype,
            accumulator,
            ..
        } => {
            assert_eq!(*active_dtype, <f16 as ConstDType>::DESCRIPTOR);
            assert_eq!(*accumulator, <f32 as ConstDType>::DESCRIPTOR);
        }
        other => panic!("expected UnsupportedPrecision, got {other:?}"),
    }
    assert!(
        error
            .to_string()
            .contains("trainer plan requires loss scaling")
    );
}

/// `UX-001`: precision resets earlier overrides; later overrides feed fresh state.
#[test]
fn scaler_uses_the_effective_plan_policy() -> TestResult {
    let plan = Trainer::plan()
        .devices(DeviceSet::cpu())
        .loss_scaling(LossScaling::None)
        .precision(RuntimePrecisionPolicy::mixed_f16())
        .build()?;
    assert_eq!(plan.loss_scaling(), LossScaling::dynamic_default());
    assert_eq!(plan.loss_scale_state().policy(), plan.loss_scaling());
    assert_eq!(plan.loss_scale_state().scale(), 65536.0);

    let scaling = LossScaling::dynamic(1024.0, 2.0, 0.5, 2);
    let plan = Trainer::plan()
        .devices(DeviceSet::cpu())
        .precision(RuntimePrecisionPolicy::mixed_f16())
        .loss_scaling(scaling)
        .build()?;
    let mut state = plan.loss_scale_state();
    assert_eq!(state.policy(), scaling);
    assert_eq!(state.scale(), 1024.0);
    state.update(true);
    assert_eq!(state.scale(), 512.0);
    assert_eq!(plan.loss_scale_state().scale(), 1024.0);

    for policy in [
        RuntimePrecisionPolicy::fp32(),
        RuntimePrecisionPolicy::mixed_bf16(),
        RuntimePrecisionPolicy::exact::<f16>(),
        RuntimePrecisionPolicy::mixed_f16().with_accumulator(PrecisionChoice::Native),
    ] {
        let plan = Trainer::plan()
            .devices(DeviceSet::cpu())
            .precision(policy)
            .loss_scaling(LossScaling::None)
            .build()?;
        assert_eq!(plan.loss_scale_state().scale(), 1.0);
    }
    Ok(())
}

fn parameters(model: &Linear<Dyn, Backend>) -> Result<Vec<f32>> {
    let weight = model.weight.as_tensor()?;
    let bias = model.bias.as_ref().expect("linear has bias").as_tensor()?;
    assert_eq!(weight.dtype(), <f32 as ConstDType>::DESCRIPTOR);
    assert_eq!(bias.dtype(), <f32 as ConstDType>::DESCRIPTOR);
    assert_eq!(
        Backend::storage_dtype::<f32>(weight.inner()),
        Some(<f32 as ConstDType>::DESCRIPTOR)
    );
    assert_eq!(
        Backend::storage_dtype::<f32>(bias.inner()),
        Some(<f32 as ConstDType>::DESCRIPTOR)
    );
    let mut values = weight.to_vec1::<f32>()?;
    values.extend(bias.to_vec1::<f32>()?);
    Ok(values)
}

/// `UX-001`: the retained bf16 policy does not autocast module boundaries.
/// F32 weights and bias remain f32 after a real trainer optimizer update.
#[test]
fn mixed_bf16_retains_f32_master_weights() -> TestResult {
    let policy = RuntimePrecisionPolicy::mixed_bf16();
    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cpu())
            .precision(policy)
            .build()?,
    );
    assert_eq!(trainer.report().precision(), policy);
    assert_eq!(
        policy.active_dtype(),
        Some(<bf16 as ConstDType>::DESCRIPTOR)
    );
    for code in ["precision", "loss-scaling"] {
        assert!(trainer.report().decisions().iter().any(|d| d.code == code));
    }
    let mut scaler = trainer.report().loss_scale_state();
    assert_eq!(scaler.policy(), LossScaling::None);
    let mut model = Linear::<Dyn, Backend>::build((2, 2))?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.05)?;
    let before = parameters(&model)?;
    let input = Tensor::<Dyn, Backend>::ones(vec![1, 2])?;
    let outcome = trainer.fit_scaled(
        &mut model,
        &mut optimizer,
        &mut scaler,
        [input],
        |model, input| Ok(model.forward(input)?.sum_all()?.forget_layout()),
    )?;
    assert_eq!(outcome.batches, 1);
    assert!(outcome.final_loss.is_some_and(f32::is_finite));
    assert_ne!(parameters(&model)?, before);
    assert_eq!(trainer.report().precision(), policy);
    Ok(())
}

/// `UX-001`: sum(linear(ones)) has unit gradients for every weight and bias.
/// Multiplying that loss by infinity injects non-finite gradients through the
/// trainer's backward pass, not by bypassing it with a manual optimizer call.
#[test]
fn mixed_f16_unscales_skips_overflow_and_recovers() -> TestResult {
    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cpu())
            .precision(RuntimePrecisionPolicy::mixed_f16())
            .loss_scaling(LossScaling::dynamic(1024.0, 2.0, 0.5, 2))
            .build()?,
    );
    let mut scaler = trainer.report().loss_scale_state();
    let mut model = Linear::<Dyn, Backend>::build((2, 2))?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.05)?;
    let input = Tensor::<Dyn, Backend>::ones(vec![1, 2])?;

    for (factor, scale, finite_steps) in [
        (1.0, 1024.0, 1),
        (f64::INFINITY, 512.0, 0),
        (1.0, 512.0, 1),
        (1.0, 1024.0, 0),
    ] {
        let before = parameters(&model)?;
        let outcome = trainer.fit_scaled(
            &mut model,
            &mut optimizer,
            &mut scaler,
            [input.clone()],
            |model, input| {
                Ok(model
                    .forward(input)?
                    .sum_all()?
                    .mul_scalar(factor)?
                    .forget_layout())
            },
        )?;
        assert_eq!(outcome.batches, 1);
        assert_eq!(scaler.scale(), scale);
        assert_eq!(scaler.steps_since_last_overflow(), finite_steps);
        let after = parameters(&model)?;
        if factor.is_finite() {
            assert!(outcome.final_loss.is_some_and(f32::is_finite));
            for (actual, initial) in after.iter().zip(&before) {
                let expected = initial - 0.05;
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "unscaled update mismatch: actual={actual}, expected={expected}"
                );
            }
        } else {
            assert!(outcome.final_loss.is_some_and(|loss| !loss.is_finite()));
            assert_eq!(
                before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "overflow must leave every weight and bias bit unchanged"
            );
        }
    }
    Ok(())
}

/// `UX-001`: ordinary fp32 fit still learns without opting into the scaled loop.
#[test]
fn fp32_fit_reduces_loss() -> TestResult {
    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cpu())
            .precision(RuntimePrecisionPolicy::fp32())
            .epochs(10)
            .build()?,
    );
    let mut model = Linear::<Dyn, Backend>::build((2, 1))?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.05)?;
    let input = Tensor::<Dyn, Backend>::ones(vec![1, 2])?;
    let target = model
        .forward(input.clone())?
        .detach()
        .add_scalar(2.0)?
        .forget_layout();
    let probe = |model: &Linear<Dyn, Backend>| {
        model
            .forward(input.clone())?
            .mse_loss(&target)?
            .to_scalar::<f32>()
    };
    let before = probe(&model)?;
    let data = [(input.clone(), target.clone())];
    let outcome = trainer.fit(
        &mut model,
        &mut optimizer,
        &data,
        |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
    )?;
    let after = probe(&model)?;
    assert_eq!(outcome.epochs, 10);
    assert_eq!(outcome.batches, 10);
    assert!(outcome.final_loss.is_some_and(f32::is_finite));
    assert!(
        after < before * 0.01,
        "loss did not decrease: {before} -> {after}"
    );
    Ok(())
}

/// `UX-001`, issue #2: a fit step runs with the plan's precision ambient and
/// hands that precision to every `ExecutionContext` built during the step.
/// Outside the call the caller's policy is back, unchanged.
#[test]
fn fit_step_runs_under_the_plan_precision_scope() -> TestResult {
    let policy = RuntimePrecisionPolicy::mixed_bf16();
    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cpu())
            .precision(policy)
            .build()?,
    );
    let ambient = ExecutionPolicy::current();
    let during = ambient.with_precision(policy);

    let mut model = Linear::<Dyn, Backend>::build((2, 2))?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.05)?;
    let input = Tensor::<Dyn, Backend>::ones(vec![1, 2])?;
    let outcome = trainer.fit(&mut model, &mut optimizer, [input], |model, input| {
        assert_eq!(
            ExecutionPolicy::current(),
            during,
            "the step must run under the plan's precision scope"
        );
        let context = ExecutionContext::from_scope(Backend::default());
        assert_eq!(
            context.precision_policy(),
            policy,
            "contexts built during the step must carry the plan's precision"
        );
        Ok(model.forward(input)?.sum_all()?.forget_layout())
    })?;
    assert_eq!(outcome.batches, 1);
    assert_eq!(
        ExecutionPolicy::current(),
        ambient,
        "fit must restore the caller's ambient policy"
    );
    Ok(())
}

/// `UX-001`, issue #2: a failing step unwinds through `?` and the `scope`
/// restore guard, so the enclosing policy survives an error the same way it
/// survives a successful return. The outer scope, not the default, is what
/// must be observable afterwards.
#[test]
fn a_failed_fit_step_restores_the_enclosing_precision_scope() -> TestResult {
    let plan_policy = RuntimePrecisionPolicy::fp32();
    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cpu())
            .precision(plan_policy)
            .build()?,
    );
    let ambient = ExecutionPolicy::current();
    let outer = ambient.with_precision(RuntimePrecisionPolicy::fp64());

    let mut model = Linear::<Dyn, Backend>::build((2, 2))?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.05)?;
    let input = Tensor::<Dyn, Backend>::ones(vec![1, 2])?;

    let (result, after_failure) = outer.scope(|| {
        assert_eq!(ExecutionPolicy::current(), outer);
        let result = trainer.fit(&mut model, &mut optimizer, [input], |model, input| {
            assert_eq!(
                ExecutionPolicy::current(),
                outer.with_precision(plan_policy),
                "fit's scope overrides only the precision axis of the enclosing scope"
            );
            let wrong = Tensor::<Dyn, Backend>::zeros(vec![7, 7])?;
            model.forward(input)?.mse_loss(&wrong)
        });
        let observed = ExecutionPolicy::current();
        (result, observed)
    });

    assert_eq!(
        after_failure, outer,
        "a failed fit must restore the enclosing scope before returning"
    );
    let error = result.expect_err("the mismatched target must fail the step");
    assert!(
        matches!(error, TrainError::Step { .. }),
        "expected a step failure, got {error:?}"
    );
    assert_eq!(
        ExecutionPolicy::current(),
        ambient,
        "leaving the outer scope restores the ambient policy"
    );
    Ok(())
}

/// `UX-001`, issue #2: `fit_scaled` installs the same plan-precision scope as
/// `fit` around its scaled loop.
#[test]
fn fit_scaled_step_runs_under_the_plan_precision_scope() -> TestResult {
    let policy = RuntimePrecisionPolicy::mixed_bf16();
    let trainer = Trainer::new(
        Trainer::plan()
            .devices(DeviceSet::cpu())
            .precision(policy)
            .build()?,
    );
    let ambient = ExecutionPolicy::current();
    let during = ambient.with_precision(policy);

    let mut scaler = trainer.report().loss_scale_state();
    let mut model = Linear::<Dyn, Backend>::build((2, 2))?;
    let mut optimizer = SGD::<Backend>::from_module(&model, 0.05)?;
    let input = Tensor::<Dyn, Backend>::ones(vec![1, 2])?;
    let outcome = trainer.fit_scaled(
        &mut model,
        &mut optimizer,
        &mut scaler,
        [input],
        |model, input| {
            assert_eq!(
                ExecutionPolicy::current(),
                during,
                "the scaled step must run under the plan's precision scope"
            );
            Ok(model.forward(input)?.sum_all()?.forget_layout())
        },
    )?;
    assert_eq!(outcome.batches, 1);
    assert_eq!(
        ExecutionPolicy::current(),
        ambient,
        "fit_scaled must restore the caller's ambient policy"
    );
    Ok(())
}
