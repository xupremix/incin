//! Shared validation and update-staging logic behind every optimizer's
//! `step`/`load_state_dict`, kept in one file rather than split further
//! since the `Adam`-family trio (`load_adam_state`, `validate_adam_config`,
//! `prepare_adam_update`) and the two functions every optimizer calls
//! (`validate_storage_pair`, `commit_parameter_updates`) both chain through
//! `invalid_optimizer_config`, the one error constructor every function
//! here (and `super::clip`, and every concrete optimizer) needs.

use super::group::PreparedUpdate;
use super::traits::OptimizerBackend;
use crate::err::{Error, ErrorMessage, Result};
use crate::shapes::Dyn;
use crate::tensor::backend::VariableBackend;
use crate::tensor::base::Tensor;
use crate::tensor::dtype::DType;
use alloc::string::{String, ToString};

pub(super) fn invalid_optimizer_config(operation: &'static str, reason: &'static str) -> Error {
    Error::InvalidModuleState {
        operation,
        reason: ErrorMessage::new(reason),
    }
}

/// Refuse a step in which no parameter in a non-empty group received a
/// gradient.
///
/// Every optimizer here skips a parameter it has no gradient for, which is
/// correct on its own: a parameter the forward pass did not use has nothing to
/// apply. Skipping *every* parameter is a different event. It means the
/// backward pass did not reach this group at all - because it was never run,
/// because the graph was detached, or because the tape that recorded the
/// forward pass belongs to another thread and the reverse walk on this one
/// found nothing to drain. In each case the previous behaviour was to commit
/// nothing and return `Ok(())`, so the training loop ran to completion with
/// parameters that never moved. A run that finishes wrong is the failure mode
/// this crate refuses everywhere else, and it costs one comparison per step to
/// refuse it here too.
pub(super) fn require_gradients_reached_the_group(
    operation: &'static str,
    parameters: usize,
    updated: usize,
) -> Result<()> {
    if parameters > 0 && updated == 0 {
        return Err(invalid_optimizer_config(
            operation,
            "no parameter in this group received a gradient: the backward pass did not \
             reach it. A tape is thread-local, so a backward call on a thread other than \
             the one that recorded the forward pass drains an empty graph and produces \
             exactly this state.",
        ));
    }
    Ok(())
}

pub(super) fn validate_learning_rate(operation: &'static str, lr: f64) -> Result<()> {
    if !lr.is_finite() || lr < 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "learning rate must be finite and non-negative",
        ));
    }
    Ok(())
}

pub(super) fn validate_adam_config(
    operation: &'static str,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: Option<f64>,
) -> Result<()> {
    validate_learning_rate(operation, lr)?;
    if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
        return Err(invalid_optimizer_config(
            operation,
            "beta1 must be finite and in [0, 1)",
        ));
    }
    if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
        return Err(invalid_optimizer_config(
            operation,
            "beta2 must be finite and in [0, 1)",
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "epsilon must be finite and positive",
        ));
    }
    if weight_decay.is_some_and(|decay| !decay.is_finite() || decay < 0.0) {
        return Err(invalid_optimizer_config(
            operation,
            "weight decay must be finite and non-negative",
        ));
    }
    Ok(())
}

pub(super) fn validate_storage_pair<B: VariableBackend, K: DType>(
    operation: &'static str,
    parameter: &B::Storage<K>,
    other: &B::Storage<K>,
) -> Result<()> {
    if B::shape(parameter) != B::shape(other) {
        return Err(invalid_optimizer_config(
            operation,
            "parameter, gradient, and optimizer-state shapes must match",
        ));
    }
    if let (Some(expected), Some(actual)) = (B::storage_dtype(parameter), B::storage_dtype(other))
        && expected != actual
    {
        return Err(Error::DTypeMismatch {
            operation,
            expected,
            actual,
        });
    }
    if let (Some(expected), Some(actual)) = (B::storage_device(parameter), B::storage_device(other))
        && expected != actual
    {
        return Err(Error::PlacementMismatch {
            operation,
            expected,
            actual,
        });
    }
    Ok(())
}

type AdamState<S> = (
    alloc::collections::BTreeMap<String, S>,
    alloc::collections::BTreeMap<String, S>,
);

pub(super) fn load_adam_state<B: VariableBackend, K: DType>(
    operation: &'static str,
    prefix: &str,
    params: &alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
) -> Result<AdamState<B::Storage<K>>> {
    let prefix = if prefix.is_empty() {
        alloc::string::String::new()
    } else {
        alloc::format!("{}.", prefix)
    };
    let m_prefix = alloc::format!("{}m.", prefix);
    let v_prefix = alloc::format!("{}v.", prefix);
    let mut next_m = alloc::collections::BTreeMap::new();
    let mut next_v = alloc::collections::BTreeMap::new();

    for (key, tensor) in dict {
        let (name, destination) = if let Some(name) = key.strip_prefix(&m_prefix) {
            (name, &mut next_m)
        } else if let Some(name) = key.strip_prefix(&v_prefix) {
            (name, &mut next_v)
        } else {
            continue;
        };
        let parameter = params.get(name).ok_or_else(|| {
            invalid_optimizer_config(operation, "state dictionary names an unknown parameter")
        })?;
        let parameter = B::var_as_tensor::<K>(parameter)?;
        validate_storage_pair::<B, K>(operation, &parameter, tensor.inner())?;
        destination.insert(name.to_string(), tensor.inner().clone());
    }

    for name in next_m.keys().chain(next_v.keys()) {
        if !next_m.contains_key(name) || !next_v.contains_key(name) {
            return Err(invalid_optimizer_config(
                operation,
                "Adam state dictionary must contain both moments for each parameter",
            ));
        }
    }
    Ok((next_m, next_v))
}

pub(super) fn commit_parameter_updates<B: VariableBackend, K: DType>(
    operation: &'static str,
    params: &mut alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    updates: &[PreparedUpdate<B::Storage<K>>],
) -> Result<()> {
    for update in updates {
        let var = params
            .get_mut(&update.name)
            .ok_or(Error::InternalInvariant {
                operation,
                reason: "prepared optimizer update lost its parameter",
            })?;
        if let Err(commit_error) = B::assign_var::<K>(var, &update.updated) {
            for rollback in updates {
                let rollback_var =
                    params
                        .get_mut(&rollback.name)
                        .ok_or(Error::InternalInvariant {
                            operation,
                            reason: "optimizer rollback lost its parameter",
                        })?;
                if B::assign_var::<K>(rollback_var, &rollback.before).is_err() {
                    return Err(Error::InternalInvariant {
                        operation,
                        reason: "backend rejected optimizer rollback",
                    });
                }
            }
            return Err(commit_error);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare_adam_update<B: OptimizerBackend<K>, K: DType>(
    operation: &'static str,
    tensor: &B::Storage<K>,
    grad: &B::Storage<K>,
    previous_m: Option<&B::Storage<K>>,
    previous_v: Option<&B::Storage<K>>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step: usize,
) -> Result<(B::Storage<K>, B::Storage<K>, B::Storage<K>)> {
    validate_storage_pair::<B, K>(operation, tensor, grad)?;
    if let Some(m) = previous_m {
        validate_storage_pair::<B, K>(operation, tensor, m)?;
    }
    if let Some(v) = previous_v {
        validate_storage_pair::<B, K>(operation, tensor, v)?;
    }

    let m_t = if let Some(m) = previous_m {
        let retained = B::optimizer_mul_scalar(m, beta1)?;
        let incoming = B::optimizer_mul_scalar(grad, 1.0 - beta1)?;
        B::optimizer_add(&retained, &incoming)?
    } else {
        B::optimizer_mul_scalar(grad, 1.0 - beta1)?
    };
    let grad_sq = B::optimizer_mul(grad, grad)?;
    let v_t = if let Some(v) = previous_v {
        let retained = B::optimizer_mul_scalar(v, beta2)?;
        let incoming = B::optimizer_mul_scalar(&grad_sq, 1.0 - beta2)?;
        B::optimizer_add(&retained, &incoming)?
    } else {
        B::optimizer_mul_scalar(&grad_sq, 1.0 - beta2)?
    };

    let t_step = step as f64;
    let bias_correction1 = 1.0 - beta1.powf(t_step);
    let bias_correction2 = 1.0 - beta2.powf(t_step);
    if !bias_correction1.is_finite()
        || !bias_correction2.is_finite()
        || bias_correction1 <= 0.0
        || bias_correction2 <= 0.0
    {
        return Err(Error::ArithmeticOverflow {
            operation,
            expression: "Adam bias correction",
        });
    }

    let m_hat = B::optimizer_mul_scalar(&m_t, 1.0 / bias_correction1)?;
    let v_hat = B::optimizer_mul_scalar(&v_t, 1.0 / bias_correction2)?;
    let sqrt_v_hat = B::optimizer_sqrt(&v_hat)?;
    let denom = B::optimizer_add_scalar(&sqrt_v_hat, eps)?;
    let normalized = B::optimizer_div(&m_hat, &denom)?;
    let step_value = B::optimizer_mul_scalar(&normalized, lr)?;
    let decayed = if weight_decay == 0.0 {
        tensor.clone()
    } else {
        let decay = B::optimizer_mul_scalar(tensor, weight_decay * lr)?;
        B::optimizer_sub(tensor, &decay)?
    };
    let updated = B::optimizer_sub(&decayed, &step_value)?;
    Ok((updated, m_t, v_t))
}
