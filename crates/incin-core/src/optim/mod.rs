use crate::dist::Local;
use crate::err::{Error, ErrorMessage, Result};
use crate::nn::param::{Param, TrainState};
use crate::nn::{ParameterVisitor, StatePath, VisitParameters};
use crate::shapes::Dyn;
use crate::shapes::{Shape, ShapeBuf, ShapeValue};
use crate::tensor::backend::{AutogradBackend, HostReadback, VariableBackend};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::{ConstDType, DType};
use crate::{
    backend_authoring::{Capabilities, Execute},
    exec::request::TensorHandle,
    exec::{
        catalog::{NoAttributes, ScalarAttributes, op},
        dispatch,
    },
};
use alloc::string::{String, ToString};

pub mod scheduler;
pub use crate::autograd::Gradients;
pub use scheduler::{ConstantLR, LRScheduler, LinearLR};
#[cfg(feature = "std")]
pub use scheduler::{CosineAnnealingLR, StepLR};

/// Trait defining a generic optimization algorithm.
///
/// Implementors receive a reference to the [`Gradients`] computed from a backward pass
/// and apply the appropriate parameter update rule to all tracked variables.
pub trait Optimizer<B: VariableBackend + AutogradBackend> {
    /// Steps the optimizer using the given gradients, updating the tracked parameters.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()>;
}

/// Extension trait for optimizers that support mixed-precision loss scaling.
pub trait ScaledOptimizer<B: VariableBackend + AutogradBackend>: Optimizer<B> {
    /// Steps the optimizer with loss scaling support, unscaling gradients in-place and skipping
    /// updates on overflow.
    ///
    /// Returns `Ok(true)` if the step was applied (gradients were finite and unscaled).
    /// Returns `Ok(false)` if non-finite gradients were found (overflow), skipping the update.
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool>;
}

/// A homogeneous, optimizer-owned collection of trainable variables.
pub struct ParameterGroup<B: VariableBackend, K: ConstDType> {
    params: alloc::collections::BTreeMap<String, B::Var<K>>,
}

impl<B: VariableBackend, K: ConstDType> ParameterGroup<B, K> {
    /// Collects trainable parameters with dtype `K` through the canonical
    /// heterogeneous module visitor. Parameters of other dtypes are ignored;
    /// this lets one model contain optimizer-incompatible auxiliary dtypes
    /// without creating a second module traversal architecture.
    pub fn from_module<M>(module: &M) -> Result<Self>
    where
        M: VisitParameters<B>,
    {
        let mut collector = ParameterCollector::<B, K> {
            params: alloc::collections::BTreeMap::new(),
        };
        module.visit_parameters(&StatePath::root(), &mut collector)?;
        Ok(Self {
            params: collector.params,
        })
    }

    /// Creates a group from an already collected homogeneous map.
    #[must_use]
    pub fn from_map(params: alloc::collections::BTreeMap<String, B::Var<K>>) -> Self {
        Self { params }
    }

    /// Returns the number of collected variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Returns whether the group contains no variables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Iterates over the collected variables in canonical path order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &B::Var<K>)> {
        self.params.iter()
    }

    fn into_map(self) -> alloc::collections::BTreeMap<String, B::Var<K>> {
        self.params
    }
}

struct ParameterCollector<B: VariableBackend, K: ConstDType> {
    params: alloc::collections::BTreeMap<String, B::Var<K>>,
}

impl<B: VariableBackend, K: ConstDType> ParameterVisitor<B> for ParameterCollector<B, K> {
    fn visit_param<S, LeafK, Train>(
        &mut self,
        path: &StatePath,
        param: &Param<S, B, LeafK, Train>,
    ) -> Result<()>
    where
        S: Shape,
        LeafK: DType,
        Train: TrainState,
    {
        if param.dtype_descriptor() != K::DESCRIPTOR {
            return Ok(());
        }
        let variable =
            param
                .variable_any()
                .downcast_ref::<B::Var<K>>()
                .ok_or(Error::InternalInvariant {
                    operation: "collect parameter group",
                    reason: "dtype matched but backend variable type did not",
                })?;
        self.params.insert(path.to_string(), variable.clone());
        Ok(())
    }
}

struct PreparedUpdate<S> {
    name: String,
    before: S,
    updated: S,
    first_moment: Option<S>,
    second_moment: Option<S>,
}

/// The canonical execution profile required by generic optimizers.
///
/// Backend authors satisfy this profile by implementing the exact operation
/// descriptors listed in the blanket implementation below.
pub trait OptimizerBackend<K: DType>: VariableBackend {
    fn optimizer_add(lhs: &Self::Storage<K>, rhs: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_sub(lhs: &Self::Storage<K>, rhs: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_mul(lhs: &Self::Storage<K>, rhs: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_div(lhs: &Self::Storage<K>, rhs: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_sqrt(storage: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_mul_scalar(storage: &Self::Storage<K>, value: f64) -> Result<Self::Storage<K>>;
    fn optimizer_add_scalar(storage: &Self::Storage<K>, value: f64) -> Result<Self::Storage<K>>;
}

impl<B, K: DType> OptimizerBackend<K> for B
where
    B: VariableBackend
        + Capabilities
        + Execute<op::Add>
        + Execute<op::Sub>
        + Execute<op::Mul>
        + Execute<op::Div>
        + Execute<op::Sqrt>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Div>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sqrt>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<K>>,
{
    fn optimizer_add(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_binary::<op::Add, B, K>(lhs, rhs)
    }

    fn optimizer_sub(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_binary::<op::Sub, B, K>(lhs, rhs)
    }

    fn optimizer_mul(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_binary::<op::Mul, B, K>(lhs, rhs)
    }

    fn optimizer_div(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_binary::<op::Div, B, K>(lhs, rhs)
    }

    fn optimizer_sqrt(storage: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_unary::<op::Sqrt, B, K>(storage)
    }

    fn optimizer_mul_scalar(storage: &B::Storage<K>, value: f64) -> Result<B::Storage<K>> {
        execute_scalar::<op::MulScalar, B, K>(storage, value)
    }

    fn optimizer_add_scalar(storage: &B::Storage<K>, value: f64) -> Result<B::Storage<K>> {
        execute_scalar::<op::AddScalar, B, K>(storage, value)
    }
}

fn execute_binary<O, B, K>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>
where
    O: crate::exec::catalog::Operation<Attributes = NoAttributes>,
    B: VariableBackend + Capabilities + Execute<O>,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let expected =
        ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&B::shape(lhs))).map_err(Error::Shape)?;
    let inputs = [
        TensorHandle::from_storage::<B, K, Local>(lhs),
        TensorHandle::from_storage::<B, K, Local>(rhs),
    ];
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(&context, NoAttributes, &inputs, &expected)
        .map(Into::into)
        .map_err(Error::from)
}

fn execute_unary<O, B, K>(storage: &B::Storage<K>) -> Result<B::Storage<K>>
where
    O: crate::exec::catalog::Operation<Attributes = NoAttributes>,
    B: VariableBackend + Capabilities + Execute<O>,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&B::shape(storage)))
        .map_err(Error::Shape)?;
    let input = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(&context, NoAttributes, &[input], &expected)
        .map(Into::into)
        .map_err(Error::from)
}

fn execute_scalar<O, B, K>(storage: &B::Storage<K>, value: f64) -> Result<B::Storage<K>>
where
    O: crate::exec::catalog::Operation<Attributes = ScalarAttributes>,
    B: VariableBackend + Capabilities + Execute<O>,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&B::shape(storage)))
        .map_err(Error::Shape)?;
    let input = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(&context, ScalarAttributes { value }, &[input], &expected)
        .map(Into::into)
        .map_err(Error::from)
}

fn invalid_optimizer_config(operation: &'static str, reason: &'static str) -> Error {
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
/// backward pass did not reach this group at all — because it was never run,
/// because the graph was detached, or because the tape that recorded the
/// forward pass belongs to another thread and the reverse walk on this one
/// found nothing to drain. In each case the previous behaviour was to commit
/// nothing and return `Ok(())`, so the training loop ran to completion with
/// parameters that never moved. A run that finishes wrong is the failure mode
/// this crate refuses everywhere else, and it costs one comparison per step to
/// refuse it here too.
fn require_gradients_reached_the_group(
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

fn validate_learning_rate(operation: &'static str, lr: f64) -> Result<()> {
    if !lr.is_finite() || lr < 0.0 {
        return Err(invalid_optimizer_config(
            operation,
            "learning rate must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_adam_config(
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

fn validate_storage_pair<B: VariableBackend, K: DType>(
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

fn load_adam_state<B: VariableBackend, K: DType>(
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

fn commit_parameter_updates<B: VariableBackend, K: DType>(
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
fn prepare_adam_update<B: OptimizerBackend<K>, K: DType>(
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

/// Rescales a parameter group's gradients so their global L2 norm is at most
/// `max_norm`, and returns the norm they had before rescaling.
///
/// This is the standard total-norm form: the norm is taken over the
/// concatenation of every gradient in the group, not per parameter, so the
/// direction of the update is preserved and only its length changes. A group
/// already under the threshold is left untouched and the returned norm is the
/// one it had.
///
/// Call it between the backward pass and [`Optimizer::step`]:
///
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::optim::{ParameterGroup, clip_grad_norm};
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
/// let input = Tensor::<s![1, 4], DefaultBackend>::ones(())?.require_grad();
/// let mut gradients = model.forward(input)?.sum_all()?.backward()?;
///
/// let group = ParameterGroup::<DefaultBackend, f32>::from_module(&model)?;
/// let before = clip_grad_norm(&group, &mut gradients, 1.0)?;
/// assert!(before >= 0.0);
///
/// let mut optimizer = SGD::<DefaultBackend>::from_module(&model, 0.01)?;
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
///
/// # Errors
///
/// Returns an error when `max_norm` is not finite and positive, when a
/// gradient cannot be read back to the host, or when the backend refuses the
/// rescale.
pub fn clip_grad_norm<B, K>(
    params: &ParameterGroup<B, K>,
    grads: &mut Gradients<B>,
    max_norm: f64,
) -> Result<f64>
where
    B: VariableBackend + AutogradBackend + OptimizerBackend<K> + HostReadback,
    K: ConstDType,
{
    const OPERATION: &str = "clip_grad_norm";
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(invalid_optimizer_config(
            OPERATION,
            "the maximum norm must be finite and greater than zero",
        ));
    }

    // Two passes over the group, because the scale factor is a property of the
    // whole set: nothing can be rescaled until every gradient has been
    // measured. The first pass also collects the storage handles so the second
    // does not repeat the parameter lookup.
    let mut squared_total = 0.0f64;
    let mut present = alloc::vec::Vec::new();
    for (_, var) in params.iter() {
        let tensor = B::var_as_tensor::<K>(var)?;
        let Some(grad) = B::get_grad::<K>(&tensor, grads.as_backend())? else {
            continue;
        };
        for value in B::float_to_vec1::<K>(&grad)? {
            squared_total += value * value;
        }
        present.push((tensor, grad));
    }

    let total_norm = squared_total.sqrt();
    if !total_norm.is_finite() {
        return Err(invalid_optimizer_config(
            OPERATION,
            "the gradient norm is not finite, so no finite rescale exists. Inspect the \
             backward pass rather than clipping a NaN into range.",
        ));
    }
    if total_norm <= max_norm {
        return Ok(total_norm);
    }

    // The epsilon keeps the divisor away from zero. It cannot matter here —
    // this branch already established `total_norm > max_norm > 0` — but it
    // keeps the expression the same one every reference implementation writes,
    // which is worth more than the branch it would save.
    let scale = max_norm / (total_norm + 1e-6);
    for (tensor, grad) in present {
        let scaled = B::optimizer_mul_scalar(&grad, scale)?;
        B::set_grad::<K>(&tensor, grads.as_backend_mut(), scaled)?;
    }
    Ok(total_norm)
}

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Applies the update rule: `w ← w - lr * ∂L/∂w`.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
///
/// // The gradients must come from a backward pass over *this* model. A step
/// // whose gradients reach none of the group's parameters is refused rather
/// // than silently committing nothing.
/// let input = Tensor::<s![1, 4], DefaultBackend>::ones(())?.require_grad();
/// let loss = model.forward(input)?.sum_all()?;
/// let gradients = loss.backward()?;
///
/// let mut optimizer = SGD::<DefaultBackend>::from_module(&model, 0.01)?;
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
pub struct SGD<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    /// `lr`.
    pub lr: f64,
    _marker: core::marker::PhantomData<K>,
}

impl<B: VariableBackend, K: DType> SGD<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(
        params: alloc::collections::BTreeMap<
            String,
            <B as crate::tensor::backend::VariableBackend>::Var<K>,
        >,
        lr: f64,
    ) -> Self {
        Self {
            params,
            lr,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates an optimizer from the canonical module-derived parameter group.
    pub fn from_group(group: ParameterGroup<B, K>, lr: f64) -> Self
    where
        K: ConstDType,
    {
        Self::new(group.into_map(), lr)
    }

    /// Collects a module's trainable parameters and creates the optimizer.
    pub fn from_module<M>(module: &M, lr: f64) -> Result<Self>
    where
        M: VisitParameters<B>,
        K: ConstDType,
    {
        Ok(Self::from_group(ParameterGroup::from_module(module)?, lr))
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Optimizer<B> for SGD<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "sgd_step";
        validate_learning_rate(OPERATION, self.lr)?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                validate_storage_pair::<B, K>(OPERATION, &t, &grad)?;
                let grad_scaled = B::optimizer_mul_scalar(&grad, self.lr)?;
                let updated = B::optimizer_sub(&t, &grad_scaled)?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: None,
                    second_moment: None,
                });
            }
        }
        require_gradients_reached_the_group(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend + crate::tensor::backend::HostReadback, K: ConstDType>
    ScaledOptimizer<B> for SGD<B, K>
{
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool> {
        if !scaler.unscale_and_update_vars(self.params.values(), grads)? {
            return Ok(false);
        }
        self.step(grads)?;
        Ok(true)
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
///
/// AdamW modifies the standard Adam algorithm by decoupling the weight decay from the
/// gradient updates. This leads to better generalization performance, particularly when
/// training transformer models and deep networks.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
///
/// // The gradients must come from a backward pass over *this* model. A step
/// // whose gradients reach none of the group's parameters is refused rather
/// // than silently committing nothing.
/// let input = Tensor::<s![1, 4], DefaultBackend>::ones(())?.require_grad();
/// let loss = model.forward(input)?.sum_all()?;
/// let gradients = loss.backward()?;
///
/// let mut optimizer = AdamW::<DefaultBackend>::from_module(&model, 1e-4)?;
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
pub struct AdamW<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    /// `lr`.
    pub lr: f64,
    /// `beta1`.
    pub beta1: f64,
    /// `beta2`.
    pub beta2: f64,
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f64,
    /// `weight_decay`.
    pub weight_decay: f64,
    m: alloc::collections::BTreeMap<String, B::Storage<K>>,
    v: alloc::collections::BTreeMap<String, B::Storage<K>>,
    step: usize,
}

impl<B: VariableBackend, K: DType> AdamW<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(
        params: alloc::collections::BTreeMap<
            String,
            <B as crate::tensor::backend::VariableBackend>::Var<K>,
        >,
        lr: f64,
    ) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: alloc::collections::BTreeMap::new(),
            v: alloc::collections::BTreeMap::new(),
            step: 0,
        }
    }

    /// Creates an optimizer from the canonical module-derived parameter group.
    pub fn from_group(group: ParameterGroup<B, K>, lr: f64) -> Self
    where
        K: ConstDType,
    {
        Self::new(group.into_map(), lr)
    }

    /// Collects a module's trainable parameters and creates the optimizer.
    pub fn from_module<M>(module: &M, lr: f64) -> Result<Self>
    where
        M: VisitParameters<B>,
        K: ConstDType,
    {
        Ok(Self::from_group(ParameterGroup::from_module(module)?, lr))
    }

    /// Gets the current step counter.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Sets the current step counter value.
    pub fn set_step_count(&mut self, step: usize) {
        self.step = step;
    }

    /// Exports optimizer state tensors (`m` and `v` momentum buffers).
    pub fn state_dict(
        &self,
        prefix: &str,
        dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        let p = if prefix.is_empty() {
            alloc::string::String::new()
        } else {
            alloc::format!("{}.", prefix)
        };
        for (name, m_val) in &self.m {
            let shape = B::shape(m_val);
            let tensor = Tensor::<Dyn, B, K>::from_parts(
                m_val.clone(),
                ShapeBuf::from_slice(&shape),
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            )?;
            dict.insert(alloc::format!("{}m.{}", p, name), tensor);
        }
        for (name, v_val) in &self.v {
            let shape = B::shape(v_val);
            let tensor = Tensor::<Dyn, B, K>::from_parts(
                v_val.clone(),
                ShapeBuf::from_slice(&shape),
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            )?;
            dict.insert(alloc::format!("{}v.{}", p, name), tensor);
        }
        Ok(())
    }

    /// Loads optimizer state tensors from a dictionary.
    pub fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        let (next_m, next_v) =
            load_adam_state::<B, K>("adamw_load_state_dict", prefix, &self.params, dict)?;
        self.m = next_m;
        self.v = next_v;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Optimizer<B> for AdamW<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "adamw_step";
        validate_adam_config(
            OPERATION,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            Some(self.weight_decay),
        )?;
        let next_step = self.step.checked_add(1).ok_or(Error::ArithmeticOverflow {
            operation: OPERATION,
            expression: "optimizer step + 1",
        })?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, m_t, v_t) = prepare_adam_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.m.get(name),
                    self.v.get(name),
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    next_step,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: Some(m_t),
                    second_moment: Some(v_t),
                });
            }
        }
        require_gradients_reached_the_group(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            self.m.insert(
                update.name.clone(),
                update.first_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared AdamW update lost first moment",
                })?,
            );
            self.v.insert(
                update.name,
                update.second_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared AdamW update lost second moment",
                })?,
            );
        }
        self.step = next_step;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend + crate::tensor::backend::HostReadback, K: ConstDType>
    ScaledOptimizer<B> for AdamW<B, K>
{
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool> {
        if !scaler.unscale_and_update_vars(self.params.values(), grads)? {
            return Ok(false);
        }
        self.step(grads)?;
        Ok(true)
    }
}

/// Adam optimization algorithm.
///
/// Implements the standard Adam optimizer with momentum and variance tracking.
/// For models sensitive to weight decay (like Transformers), prefer [`AdamW`].
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
///
/// // The gradients must come from a backward pass over *this* model. A step
/// // whose gradients reach none of the group's parameters is refused rather
/// // than silently committing nothing.
/// let input = Tensor::<s![1, 4], DefaultBackend>::ones(())?.require_grad();
/// let loss = model.forward(input)?.sum_all()?;
/// let gradients = loss.backward()?;
///
/// let mut optimizer = Adam::<DefaultBackend>::from_module(&model, 1e-3)?;
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
pub struct Adam<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    /// `lr`.
    pub lr: f64,
    /// `beta1`.
    pub beta1: f64,
    /// `beta2`.
    pub beta2: f64,
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f64,
    m: alloc::collections::BTreeMap<String, B::Storage<K>>,
    v: alloc::collections::BTreeMap<String, B::Storage<K>>,
    step: usize,
}

impl<B: VariableBackend, K: DType> Adam<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(
        params: alloc::collections::BTreeMap<
            String,
            <B as crate::tensor::backend::VariableBackend>::Var<K>,
        >,
        lr: f64,
    ) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: alloc::collections::BTreeMap::new(),
            v: alloc::collections::BTreeMap::new(),
            step: 0,
        }
    }

    /// Creates an optimizer from the canonical module-derived parameter group.
    pub fn from_group(group: ParameterGroup<B, K>, lr: f64) -> Self
    where
        K: ConstDType,
    {
        Self::new(group.into_map(), lr)
    }

    /// Collects a module's trainable parameters and creates the optimizer.
    pub fn from_module<M>(module: &M, lr: f64) -> Result<Self>
    where
        M: VisitParameters<B>,
        K: ConstDType,
    {
        Ok(Self::from_group(ParameterGroup::from_module(module)?, lr))
    }

    /// Gets the current step counter.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Sets the current step counter value.
    pub fn set_step_count(&mut self, step: usize) {
        self.step = step;
    }

    /// Exports optimizer state tensors (`m` and `v` momentum buffers).
    pub fn state_dict(
        &self,
        prefix: &str,
        dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        let p = if prefix.is_empty() {
            alloc::string::String::new()
        } else {
            alloc::format!("{}.", prefix)
        };
        for (name, m_val) in &self.m {
            let shape = B::shape(m_val);
            let tensor = Tensor::<Dyn, B, K>::from_parts(
                m_val.clone(),
                ShapeBuf::from_slice(&shape),
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            )?;
            dict.insert(alloc::format!("{}m.{}", p, name), tensor);
        }
        for (name, v_val) in &self.v {
            let shape = B::shape(v_val);
            let tensor = Tensor::<Dyn, B, K>::from_parts(
                v_val.clone(),
                ShapeBuf::from_slice(&shape),
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            )?;
            dict.insert(alloc::format!("{}v.{}", p, name), tensor);
        }
        Ok(())
    }

    /// Loads optimizer state tensors from a dictionary.
    pub fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        let (next_m, next_v) =
            load_adam_state::<B, K>("adam_load_state_dict", prefix, &self.params, dict)?;
        self.m = next_m;
        self.v = next_v;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Optimizer<B> for Adam<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "adam_step";
        validate_adam_config(OPERATION, self.lr, self.beta1, self.beta2, self.eps, None)?;
        let next_step = self.step.checked_add(1).ok_or(Error::ArithmeticOverflow {
            operation: OPERATION,
            expression: "optimizer step + 1",
        })?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, m_t, v_t) = prepare_adam_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.m.get(name),
                    self.v.get(name),
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    0.0,
                    next_step,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: Some(m_t),
                    second_moment: Some(v_t),
                });
            }
        }
        require_gradients_reached_the_group(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            self.m.insert(
                update.name.clone(),
                update.first_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared Adam update lost first moment",
                })?,
            );
            self.v.insert(
                update.name,
                update.second_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared Adam update lost second moment",
                })?,
            );
        }
        self.step = next_step;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend + crate::tensor::backend::HostReadback, K: ConstDType>
    ScaledOptimizer<B> for Adam<B, K>
{
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool> {
        if !scaler.unscale_and_update_vars(self.params.values(), grads)? {
            return Ok(false);
        }
        self.step(grads)?;
        Ok(true)
    }
}
