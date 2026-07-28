use crate::prelude::*;

pub mod scheduler;
pub use scheduler::*;

/// Encapsulates the backend-specific gradients obtained from a backward pass.
///
/// This is a newtype wrapper around the backend's raw gradient container (e.g., `candle_core::backprop::GradStore`).
/// Obtain it by calling `.backward()` on a scalar loss tensor. Pass it to [`Optimizer::step`] to update parameters.
pub struct Gradients<G>(pub G);

/// Trait defining a generic optimization algorithm.
///
/// Implementors receive a reference to the [`Gradients`] computed from a backward pass
/// and apply the appropriate parameter update rule to all tracked variables.
pub trait Optimizer<B: Backend> {
    /// Steps the optimizer using the given gradients, updating the tracked parameters.
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()>;
}

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Applies the update rule: `w ← w - lr * ∂L/∂w`.
///
/// ## Examples
/// ```rust,ignore
/// use incin::prelude::*;
///
/// let optimizer = SGD::new(model.parameters(), 0.01);
/// optimizer.step(&gradients)?;
/// ```
pub struct SGD<B: Backend, K: DType = f32> {
    params: alloc::collections::BTreeMap<String, B::RawVar>,
    /// `lr`.
    pub lr: f64,
    _marker: core::marker::PhantomData<K>,
}

impl<B: Backend, K: DType> SGD<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(params: alloc::collections::BTreeMap<String, B::RawVar>, lr: f64) -> Self {
        Self {
            params,
            lr,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: Backend, K: DType> Optimizer<B> for SGD<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        for var in self.params.values_mut() {
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, &grads.0)? {
                // t = t - lr * grad
                let grad_scaled = B::mul_scalar_float::<K>(&grad, self.lr)?;
                let updated = B::sub::<K>(&t, &grad_scaled)?;
                B::assign_var::<K>(&mut *var, &updated)?;
            }
        }
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
/// ```rust,ignore
/// use incin::prelude::*;
///
/// let optimizer = AdamW::new(model.parameters(), 1e-4);
/// optimizer.step(&gradients)?;
/// ```
pub struct AdamW<B: Backend, K: DType = f32> {
    params: alloc::collections::BTreeMap<String, B::RawVar>,
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

impl<B: Backend, K: DType> AdamW<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(params: alloc::collections::BTreeMap<String, B::RawVar>, lr: f64) -> Self {
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
    ) {
        let p = if prefix.is_empty() {
            alloc::string::String::new()
        } else {
            alloc::format!("{}.", prefix)
        };
        for (name, m_val) in &self.m {
            let shape = B::shape(m_val);
            if let Ok(tensor) = Tensor::<Dyn, B, K>::from_parts(
                m_val.clone(),
                shape,
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            ) {
                dict.insert(alloc::format!("{}m.{}", p, name), tensor);
            }
        }
        for (name, v_val) in &self.v {
            let shape = B::shape(v_val);
            if let Ok(tensor) = Tensor::<Dyn, B, K>::from_parts(
                v_val.clone(),
                shape,
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            ) {
                dict.insert(alloc::format!("{}v.{}", p, name), tensor);
            }
        }
    }

    /// Loads optimizer state tensors from a dictionary.
    pub fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        let p = if prefix.is_empty() {
            alloc::string::String::new()
        } else {
            alloc::format!("{}.", prefix)
        };
        let m_prefix = alloc::format!("{}m.", p);
        let v_prefix = alloc::format!("{}v.", p);

        for (key, tensor) in dict {
            if let Some(name) = key.strip_prefix(&m_prefix) {
                self.m.insert(name.to_string(), tensor.inner().clone());
            } else if let Some(name) = key.strip_prefix(&v_prefix) {
                self.v.insert(name.to_string(), tensor.inner().clone());
            }
        }
        Ok(())
    }
}

impl<B: Backend<FloatElem = f32>> crate::nn::module::StateDict<B> for AdamW<B, f32> {
    fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        AdamW::load_state_dict(self, prefix, dict)
    }

    fn state_dict(
        &self,
        prefix: &str,
        dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) {
        AdamW::state_dict(self, prefix, dict);
    }
}

impl<B: Backend, K: DType> Optimizer<B> for AdamW<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        self.step += 1;
        let t_step = self.step as f64;
        let _bias_correction1 = 1.0 - self.beta1.powf(t_step);
        let _bias_correction2 = 1.0 - self.beta2.powf(t_step);

        for (name, var) in self.params.iter_mut() {
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, &grads.0)? {
                let device = B::storage_device::<K>(&t).unwrap_or_else(DeviceId::cpu);
                if !self.m.contains_key(name) {
                    let zero =
                        B::var_zeros::<K>(B::shape::<K>(&t).as_slice(), DTypeId::F32, &device)?;
                    self.m.insert(name.clone(), B::var_as_tensor::<K>(&zero)?);
                }
                if !self.v.contains_key(name) {
                    let zero =
                        B::var_zeros::<K>(B::shape::<K>(&t).as_slice(), DTypeId::F32, &device)?;
                    self.v.insert(name.clone(), B::var_as_tensor::<K>(&zero)?);
                }

                let mut m_t = self.m.remove(name).unwrap();
                let mut v_t = self.v.remove(name).unwrap();

                B::adamw_step::<K>(
                    var,
                    &grad,
                    &mut m_t,
                    &mut v_t,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    self.step,
                )?;

                self.m.insert(name.clone(), m_t);
                self.v.insert(name.clone(), v_t);
            }
        }
        Ok(())
    }
}

/// Adam optimization algorithm.
///
/// Implements the standard Adam optimizer with momentum and variance tracking.
/// For models sensitive to weight decay (like Transformers), prefer [`AdamW`].
///
/// ## Examples
/// ```rust,ignore
/// use incin::prelude::*;
///
/// let optimizer = Adam::new(model.parameters(), 1e-3);
/// optimizer.step(&gradients)?;
/// ```
pub struct Adam<B: Backend, K: DType = f32> {
    params: alloc::collections::BTreeMap<String, B::RawVar>,
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

impl<B: Backend, K: DType> Adam<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(params: alloc::collections::BTreeMap<String, B::RawVar>, lr: f64) -> Self {
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
    ) {
        let p = if prefix.is_empty() {
            alloc::string::String::new()
        } else {
            alloc::format!("{}.", prefix)
        };
        for (name, m_val) in &self.m {
            let shape = B::shape(m_val);
            if let Ok(tensor) = Tensor::<Dyn, B, K>::from_parts(
                m_val.clone(),
                shape,
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            ) {
                dict.insert(alloc::format!("{}m.{}", p, name), tensor);
            }
        }
        for (name, v_val) in &self.v {
            let shape = B::shape(v_val);
            if let Ok(tensor) = Tensor::<Dyn, B, K>::from_parts(
                v_val.clone(),
                shape,
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            ) {
                dict.insert(alloc::format!("{}v.{}", p, name), tensor);
            }
        }
    }

    /// Loads optimizer state tensors from a dictionary.
    pub fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        let p = if prefix.is_empty() {
            alloc::string::String::new()
        } else {
            alloc::format!("{}.", prefix)
        };
        let m_prefix = alloc::format!("{}m.", p);
        let v_prefix = alloc::format!("{}v.", p);

        for (key, tensor) in dict {
            if let Some(name) = key.strip_prefix(&m_prefix) {
                self.m.insert(name.to_string(), tensor.inner().clone());
            } else if let Some(name) = key.strip_prefix(&v_prefix) {
                self.v.insert(name.to_string(), tensor.inner().clone());
            }
        }
        Ok(())
    }
}

impl<B: Backend<FloatElem = f32>> crate::nn::module::StateDict<B> for Adam<B, f32> {
    fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        Adam::load_state_dict(self, prefix, dict)
    }

    fn state_dict(
        &self,
        prefix: &str,
        dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B>>,
    ) {
        Adam::state_dict(self, prefix, dict);
    }
}

impl<B: Backend, K: DType> Optimizer<B> for Adam<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        self.step += 1;
        let t_step = self.step as f64;
        let bias_correction1 = 1.0 - self.beta1.powf(t_step);
        let bias_correction2 = 1.0 - self.beta2.powf(t_step);

        for (name, var) in self.params.iter_mut() {
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, &grads.0)? {
                let m_t = if let Some(m) = self.m.get(name) {
                    let term1 = B::mul_scalar_float::<K>(m, self.beta1)?;
                    let term2 = B::mul_scalar_float::<K>(&grad, 1.0 - self.beta1)?;
                    B::add::<K>(&term1, &term2)?
                } else {
                    B::mul_scalar_float::<K>(&grad, 1.0 - self.beta1)?
                };

                let grad_sq = B::mul::<K>(&grad, &grad)?;
                let v_t = if let Some(v) = self.v.get(name) {
                    let term1 = B::mul_scalar_float::<K>(v, self.beta2)?;
                    let term2 = B::mul_scalar_float::<K>(&grad_sq, 1.0 - self.beta2)?;
                    B::add::<K>(&term1, &term2)?
                } else {
                    B::mul_scalar_float::<K>(&grad_sq, 1.0 - self.beta2)?
                };

                self.m.insert(name.clone(), m_t.clone());
                self.v.insert(name.clone(), v_t.clone());

                let m_hat = B::mul_scalar_float::<K>(&m_t, 1.0 / bias_correction1)?;
                let v_hat = B::mul_scalar_float::<K>(&v_t, 1.0 / bias_correction2)?;

                // step = lr * m_hat / (sqrt(v_hat) + eps)
                let denom = B::add_scalar_float::<K>(&B::sqrt::<K>(&v_hat)?, self.eps)?;
                let step = B::mul_scalar_float::<K>(&B::div::<K>(&m_hat, &denom)?, self.lr)?;

                let updated = B::sub::<K>(&t, &step)?;
                B::assign_var::<K>(&mut *var, &updated)?;
            }
        }
        Ok(())
    }
}
