//! Shared validation and update-staging logic behind every optimizer's
//! `step`/`load_state_dict`, kept in one file rather than split further
//! since the `Adam`-family trio (`load_adam_state`, `validate_adam_config`,
//! `prepare_adam_update`) and the two functions every optimizer calls
//! (`validate_storage_pair`, `commit_parameter_updates`) both chain through
//! `invalid_optimizer_config`, the one error constructor every function
//! here (and `super::clip`, and every concrete optimizer) needs.

use super::group::PreparedUpdate;
use super::traits::OptimizerBackend;
use crate::backend_authoring::{Capabilities, Execute};
use crate::err::{Error, ErrorMessage, FloatToIntPolicy, Result, convert_f64_to_i64};
use crate::exec::catalog::{FullAttributes, op};
use crate::exec::dispatch;
use crate::exec::{ExecutionContext, GradMode};
use crate::shapes::{Dyn, ShapeBuf, ShapeValue};
use crate::tensor::backend::{HostReadback, VariableBackend};
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
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

/// Refuse a step in which only *some* parameters received a gradient.
///
/// [`require_gradients_reached_the_group`] fires when zero parameters were
/// reached; this fires when the coverage is partial (`0 < updated <
/// parameters`). The lenient `step` keeps skipping unreached parameters for
/// PyTorch compatibility (an unused parameter genuinely has nothing to
/// apply); `step_strict` calls this so a silently detached branch cannot hide
/// inside an otherwise successful step.
pub(super) fn require_full_gradient_coverage(
    operation: &'static str,
    parameters: usize,
    updated: usize,
) -> Result<()> {
    require_gradients_reached_the_group(operation, parameters, updated)?;
    if updated != parameters {
        return Err(invalid_optimizer_config(
            operation,
            "strict step requires every parameter in this group to have received a \
             gradient, but only some did: a parameter the forward pass did not use, \
             or a detached branch, was silently skipped.",
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

/// Validates an SGD configuration before any backend runs.
///
/// `nesterov` without a positive `momentum` is a typed refusal, not a silent
/// fallback to plain momentum: the caller asked for lookahead updates and
/// there is no velocity buffer to look ahead with, so running anyway would
/// train a different optimizer than the one requested.
pub(super) fn validate_sgd_config(
    operation: &'static str,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    nesterov: bool,
) -> Result<()> {
    validate_learning_rate(operation, lr)?;
    if !momentum.is_finite() || momentum < 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "momentum must be finite and non-negative",
        ));
    }
    if !weight_decay.is_finite() || weight_decay < 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "weight decay must be finite and non-negative",
        ));
    }
    if nesterov && momentum <= 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "nesterov momentum requires a positive momentum: without a velocity \
             buffer there is nothing to look ahead with",
        ));
    }
    Ok(())
}

/// Validates an RMSprop configuration before any backend runs.
pub(super) fn validate_rmsprop_config(
    operation: &'static str,
    lr: f64,
    alpha: f64,
    eps: f64,
    momentum: f64,
    weight_decay: f64,
) -> Result<()> {
    validate_learning_rate(operation, lr)?;
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err(invalid_optimizer_config(
            operation,
            "alpha must be finite and in [0, 1)",
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "epsilon must be finite and positive",
        ));
    }
    if !momentum.is_finite() || momentum < 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "momentum must be finite and non-negative",
        ));
    }
    if !weight_decay.is_finite() || weight_decay < 0.0 {
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

/// Suffix of the scalar entry carrying the Adam-family step counter.
const ADAM_STEP_SUFFIX: &str = "step";

/// Persist the Adam-family step counter alongside the moment buffers.
///
/// Bias correction divides by `1 - beta^t`, so a resumed run that restores
/// `m`/`v` but not `t` silently mis-corrects every update until the counter
/// coincidentally realigns. The counter travels as a scalar tensor under
/// `{prefix.}step`, so it survives the same serialization formats the moments
/// do. `f64` holds every realistic step count exactly; the value is produced
/// by the `Full` creation op, which every backend implements.
pub(super) fn save_adam_step<B, K>(
    prefix: &str,
    step: usize,
    dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
) -> Result<()>
where
    B: VariableBackend + Capabilities + Execute<op::Full>,
    K: DType,
    <B as Execute<op::Full>>::Output: Into<B::Storage<K>>,
{
    let dtype_field: K::Field = Default::default();
    let device_field: <B::Device as Device>::Field = Default::default();
    let dtype = K::descriptor(&dtype_field);
    let device = B::Device::to_incin(&device_field)?;
    let context = ExecutionContext::from_scope(B::default()).with_grad_mode(GradMode::Disabled);
    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[])).map_err(Error::Shape)?;
    let inner = dispatch::execute_shaped::<op::Full, B, Dyn>(
        &context,
        FullAttributes {
            shape: alloc::vec![],
            dtype,
            device,
            value: step as f64,
        },
        &[],
        &expected,
    )
    .map(Into::into)
    .map_err(Error::from)?;
    let tensor = Tensor::<Dyn, B, K>::from_parts(
        inner,
        ShapeBuf::from_slice(&[]),
        dtype_field,
        device_field,
        core::marker::PhantomData,
    )?;
    let key = if prefix.is_empty() {
        String::from("step")
    } else {
        alloc::format!("{prefix}.{ADAM_STEP_SUFFIX}")
    };
    dict.insert(key, tensor);
    Ok(())
}

/// Restore the Adam-family step counter saved by [`save_adam_step`].
///
/// Returns `Ok(None)` when the dictionary predates the counter entry: those
/// checkpoints restore moments only, exactly as before, and the caller keeps
/// whatever counter it holds (set one explicitly with `set_step_count`). A
/// present-but-malformed entry is a typed error, never a silent default.
pub(super) fn load_adam_step<B, K>(
    operation: &'static str,
    prefix: &str,
    dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
) -> Result<Option<usize>>
where
    B: VariableBackend + HostReadback,
    K: DType,
{
    let key = if prefix.is_empty() {
        String::from("step")
    } else {
        alloc::format!("{prefix}.{ADAM_STEP_SUFFIX}")
    };
    let Some(tensor) = dict.get(&key) else {
        return Ok(None);
    };
    let storage = tensor.inner();
    if !B::shape(storage).is_empty() {
        return Err(invalid_optimizer_config(
            operation,
            "optimizer step entry must be a scalar tensor",
        ));
    }
    let values = B::float_to_vec1::<K>(storage)?;
    if values.len() != 1 {
        return Err(invalid_optimizer_config(
            operation,
            "optimizer step entry must hold exactly one value",
        ));
    }
    let value = values[0];
    if !value.is_finite() || value < 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "optimizer step counter must be a finite non-negative number",
        ));
    }
    // The counter is written as `step as f64`, so a well-formed entry is
    // integral. A fractional value means a corrupted checkpoint, and `as
    // usize` would silently truncate it (and saturate an out-of-range one),
    // so this goes through the exact float-to-int conversion instead.
    let from = crate::tensor::dtype::DTypeId::F64.descriptor();
    let step_i64 = convert_f64_to_i64(operation, from, value, FloatToIntPolicy::Exact)?;
    usize::try_from(step_i64)
        .map_err(|_| invalid_optimizer_config(operation, "optimizer step counter out of range"))
        .map(Some)
}

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

/// Persists one named optimizer-state buffer map (an SGD velocity map, an
/// RMSprop square-average map) into a state dictionary.
///
/// Entries travel as `{prefix.}{buffer}.{parameter}`, the PyTorch buffer
/// names (`momentum_buffer`, `square_avg`) under this crate's dotted-prefix
/// convention. Unlike [`save_adam_step`] there is no counter entry: neither
/// SGD nor RMSprop bias-corrects, so the buffers alone resume the trajectory
/// exactly.
pub(super) fn save_state_buffers<B, K>(
    buffer: &str,
    prefix: &str,
    states: &alloc::collections::BTreeMap<String, B::Storage<K>>,
    dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
) -> Result<()>
where
    B: VariableBackend,
    K: DType,
{
    let p = if prefix.is_empty() {
        alloc::string::String::new()
    } else {
        alloc::format!("{}.", prefix)
    };
    for (name, state) in states {
        let shape = B::shape(state);
        let tensor = Tensor::<Dyn, B, K>::from_parts(
            state.clone(),
            ShapeBuf::from_slice(&shape),
            Default::default(),
            Default::default(),
            core::marker::PhantomData,
        )?;
        dict.insert(alloc::format!("{p}{buffer}.{name}"), tensor);
    }
    Ok(())
}

/// Restores one named optimizer-state buffer map saved by
/// [`save_state_buffers`].
///
/// Every entry must name a known parameter and match its storage in shape,
/// dtype, and device, exactly like [`load_adam_state`](load_adam_state);
/// entries under other prefixes pass through untouched so one dictionary can
/// carry several optimizers' states. Parameters absent from the dictionary
/// restart from a zero buffer, the same state a fresh optimizer holds, so a
/// checkpoint saved before the first step loads as a no-op rather than an
/// error.
pub(super) fn load_state_buffers<B, K>(
    operation: &'static str,
    prefix: &str,
    buffer: &str,
    params: &alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
) -> Result<alloc::collections::BTreeMap<String, B::Storage<K>>>
where
    B: VariableBackend,
    K: DType,
{
    let p = if prefix.is_empty() {
        alloc::string::String::new()
    } else {
        alloc::format!("{}.", prefix)
    };
    let key_prefix = alloc::format!("{p}{buffer}.");
    let mut next = alloc::collections::BTreeMap::new();
    for (key, tensor) in dict {
        let Some(name) = key.strip_prefix(&key_prefix) else {
            continue;
        };
        let parameter = params.get(name).ok_or_else(|| {
            invalid_optimizer_config(operation, "state dictionary names an unknown parameter")
        })?;
        let parameter = B::var_as_tensor::<K>(parameter)?;
        validate_storage_pair::<B, K>(operation, &parameter, tensor.inner())?;
        next.insert(name.to_string(), tensor.inner().clone());
    }
    Ok(next)
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

/// Stages one classic-SGD parameter update: `w ← w - lr * direction`.
///
/// Weight decay here is the classic (coupled) form: the penalty joins the
/// gradient *before* the momentum buffer, so the buffer accumulates penalized
/// gradients and the penalty itself is smoothed across steps. That is the
/// opposite of [`prepare_adam_update`](prepare_adam_update)'s decoupled form
/// (penalty applied straight to the parameter, outside the moments); the two
/// coincide only at `weight_decay == 0.0`, and the doc comments on `SGD` and
/// `AdamW` state which each optimizer uses.
///
/// With `momentum == 0.0` there is no buffer to maintain and the returned
/// velocity is `None`; the two backend ops issued are exactly the ones the
/// pre-momentum `SGD` step issued, so default-configured runs reproduce
/// earlier trajectories bit for bit.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare_sgd_update<B: OptimizerBackend<K>, K: DType>(
    operation: &'static str,
    tensor: &B::Storage<K>,
    grad: &B::Storage<K>,
    previous_velocity: Option<&B::Storage<K>>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    nesterov: bool,
) -> Result<(B::Storage<K>, Option<B::Storage<K>>)> {
    validate_storage_pair::<B, K>(operation, tensor, grad)?;
    if let Some(velocity) = previous_velocity {
        validate_storage_pair::<B, K>(operation, tensor, velocity)?;
    }
    let penalized;
    let penalized_grad: &B::Storage<K> = if weight_decay == 0.0 {
        grad
    } else {
        let decay = B::optimizer_mul_scalar(tensor, weight_decay)?;
        penalized = B::optimizer_add(grad, &decay)?;
        &penalized
    };
    let (direction, next_velocity) = if momentum == 0.0 {
        (B::optimizer_mul_scalar(penalized_grad, lr)?, None)
    } else {
        let velocity = if let Some(previous) = previous_velocity {
            let retained = B::optimizer_mul_scalar(previous, momentum)?;
            B::optimizer_add(&retained, penalized_grad)?
        } else {
            penalized_grad.clone()
        };
        // Nesterov lookahead evaluates the gradient at the extrapolated
        // position, which PyTorch writes as `d + momentum * v` on the
        // already-updated buffer.
        let direction = if nesterov {
            let lookahead = B::optimizer_mul_scalar(&velocity, momentum)?;
            B::optimizer_add(penalized_grad, &lookahead)?
        } else {
            velocity.clone()
        };
        (B::optimizer_mul_scalar(&direction, lr)?, Some(velocity))
    };
    let updated = B::optimizer_sub(tensor, &direction)?;
    Ok((updated, next_velocity))
}

/// Stages one RMSprop parameter update (non-centered form).
///
/// `v ← alpha * v + (1 - alpha) * g²`, then `w ← w - lr * g / (sqrt(v) +
/// eps)`, with an optional momentum buffer on the normalized gradient *after*
/// the division, matching PyTorch's `rmsprop` with `centered=False`. Weight
/// decay is L2 regularization folded into the gradient (not the denominator),
/// the same coupled choice [`prepare_sgd_update`](prepare_sgd_update) makes.
/// There is no `centered` variant: the variance-tracking buffer would be a
/// third state map for a form this crate's backends never needed.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare_rmsprop_update<B: OptimizerBackend<K>, K: DType>(
    operation: &'static str,
    tensor: &B::Storage<K>,
    grad: &B::Storage<K>,
    previous_square_avg: Option<&B::Storage<K>>,
    previous_momentum: Option<&B::Storage<K>>,
    lr: f64,
    alpha: f64,
    eps: f64,
    momentum: f64,
    weight_decay: f64,
) -> Result<(B::Storage<K>, B::Storage<K>, Option<B::Storage<K>>)> {
    validate_storage_pair::<B, K>(operation, tensor, grad)?;
    if let Some(square_avg) = previous_square_avg {
        validate_storage_pair::<B, K>(operation, tensor, square_avg)?;
    }
    if let Some(buffer) = previous_momentum {
        validate_storage_pair::<B, K>(operation, tensor, buffer)?;
    }
    let penalized;
    let penalized_grad: &B::Storage<K> = if weight_decay == 0.0 {
        grad
    } else {
        let decay = B::optimizer_mul_scalar(tensor, weight_decay)?;
        penalized = B::optimizer_add(grad, &decay)?;
        &penalized
    };
    let grad_sq = B::optimizer_mul(penalized_grad, penalized_grad)?;
    let square_avg = if let Some(previous) = previous_square_avg {
        let retained = B::optimizer_mul_scalar(previous, alpha)?;
        let incoming = B::optimizer_mul_scalar(&grad_sq, 1.0 - alpha)?;
        B::optimizer_add(&retained, &incoming)?
    } else {
        B::optimizer_mul_scalar(&grad_sq, 1.0 - alpha)?
    };
    let root = B::optimizer_sqrt(&square_avg)?;
    let denom = B::optimizer_add_scalar(&root, eps)?;
    let normalized = B::optimizer_div(penalized_grad, &denom)?;
    let (direction, next_momentum) = if momentum == 0.0 {
        (normalized, None)
    } else {
        let buffer = if let Some(previous) = previous_momentum {
            let retained = B::optimizer_mul_scalar(previous, momentum)?;
            B::optimizer_add(&retained, &normalized)?
        } else {
            normalized.clone()
        };
        (buffer.clone(), Some(buffer))
    };
    let step_value = B::optimizer_mul_scalar(&direction, lr)?;
    let updated = B::optimizer_sub(tensor, &step_value)?;
    Ok((updated, square_avg, next_momentum))
}
