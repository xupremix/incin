use crate::err::{Error, ErrorMessage, Result};
use crate::shapes::{Shape, ShapeBuf, ShapeValue};
use crate::shapes::Dyn;
use crate::tensor::base::Tensor;
use crate::tensor::backend::{AutogradBackend, Backend, StorageBackend, VariableBackend};
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{Grad, NoGrad, RequiresGrad};
use crate::dist::Local;
use crate::nn::param::{Param, TrainState};
use alloc::string::{String, ToString};
use crate::{
    backend_authoring::{Capabilities, Execute},
    exec::request::TensorHandle,
    exec::{
        catalog::{NoAttributes, ScalarAttributes, op},
        dispatch,
    },
};

pub mod scheduler;
pub use scheduler::*;
pub use crate::autograd::Gradients;

/// Trait defining a generic optimization algorithm.
///
/// Implementors receive a reference to the [`Gradients`] computed from a backward pass
/// and apply the appropriate parameter update rule to all tracked variables.
pub trait Optimizer<B: VariableBackend + AutogradBackend> {
    /// Steps the optimizer using the given gradients, updating the tracked parameters.
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()>;
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
    params: &alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
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
    params: &mut alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
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

#[allow(clippy::too_many_arguments)]
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

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Applies the update rule: `w ← w - lr * ∂L/∂w`.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
/// let gradients = Tensor::<s![], DefaultBackend>::zeros(())?
///     .require_grad()
///     .backward()?;
///
/// let mut optimizer = SGD::<DefaultBackend>::new(model.parameters(), 0.01);
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
pub struct SGD<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
    /// `lr`.
    pub lr: f64,
    _marker: core::marker::PhantomData<K>,
}

impl<B: VariableBackend, K: DType> SGD<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(params: alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>, lr: f64) -> Self {
        Self {
            params,
            lr,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Optimizer<B> for SGD<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
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
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        Ok(())
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
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
/// let gradients = Tensor::<s![], DefaultBackend>::zeros(())?
///     .require_grad()
///     .backward()?;
///
/// let mut optimizer = AdamW::<DefaultBackend>::new(model.parameters(), 1e-4);
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
pub struct AdamW<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
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
    pub fn new(params: alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>, lr: f64) -> Self {
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
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
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

/// Adam optimization algorithm.
///
/// Implements the standard Adam optimizer with momentum and variance tracking.
/// For models sensitive to weight decay (like Transformers), prefer [`AdamW`].
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
/// let gradients = Tensor::<s![], DefaultBackend>::zeros(())?
///     .require_grad()
///     .backward()?;
///
/// let mut optimizer = Adam::<DefaultBackend>::new(model.parameters(), 1e-3);
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
pub struct Adam<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>,
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
    pub fn new(params: alloc::collections::BTreeMap<String, <B as crate::tensor::backend::VariableBackend>::Var<K>>, lr: f64) -> Self {
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
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
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
