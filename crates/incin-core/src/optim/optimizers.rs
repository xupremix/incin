//! `SGD`, `AdamW` and `Adam`, the three concrete optimizers. Kept in one
//! file rather than one each: all three share the same nine names from
//! `super::group`/`super::support`/`super::traits` (`ParameterGroup`,
//! `PreparedUpdate`, `Optimizer`, `OptimizerBackend`, `commit_parameter_updates`,
//! `require_gradients_reached_the_group`, `validate_learning_rate`, plus
//! `Gradients`/`ScaledOptimizer`), and `AdamW`/`Adam` additionally share
//! `validate_adam_config`/`load_adam_state`/`prepare_adam_update` - splitting
//! them apart would mean writing that same import list three times over a
//! contiguous 551-line span rather than reading it once.

use super::group::{ParameterGroup, PreparedUpdate};
use super::support::{
    commit_parameter_updates, load_adam_state, prepare_adam_update,
    require_gradients_reached_the_group, validate_adam_config, validate_learning_rate,
    validate_storage_pair,
};
use super::traits::{Optimizer, OptimizerBackend, ScaledOptimizer};
use crate::autograd::Gradients;
use crate::err::{Error, Result};
use crate::nn::VisitParameters;
use crate::shapes::{Dyn, ShapeBuf};
use crate::tensor::backend::{AutogradBackend, VariableBackend};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::{ConstDType, DType};
use alloc::string::String;

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
