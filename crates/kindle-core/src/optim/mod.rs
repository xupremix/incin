use crate::prelude::*;

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
/// use kindle::prelude::*;
///
/// let optimizer = SGD::new(model.parameters(), 0.01);
/// optimizer.step(&gradients)?;
/// ```
pub struct SGD<B: Backend> {
    params: std::collections::HashMap<String, B::RawVar>,
    pub lr: f64,
}

impl<B: Backend> SGD<B> {
    pub fn new(params: std::collections::HashMap<String, B::RawVar>, lr: f64) -> Self {
        Self { params, lr }
    }
}

impl<B: Backend> Optimizer<B> for SGD<B> {
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        for var in self.params.values_mut() {
            if let Some(grad) = B::get_grad(var, &grads.0)? {
                let t = B::var_as_tensor(var)?;
                // t = t - lr * grad
                let grad_scaled = B::mul_scalar(&grad, self.lr)?;
                let updated = B::sub(&t, &grad_scaled)?;
                B::assign_var(var, &updated)?;
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
/// use kindle::prelude::*;
///
/// let optimizer = AdamW::new(model.parameters(), 1e-4);
/// optimizer.step(&gradients)?;
/// ```
pub struct AdamW<B: Backend> {
    params: std::collections::HashMap<String, B::RawVar>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    m: std::collections::HashMap<String, B::RawTensor>,
    v: std::collections::HashMap<String, B::RawTensor>,
    step: usize,
}

impl<B: Backend> AdamW<B> {
    pub fn new(params: std::collections::HashMap<String, B::RawVar>, lr: f64) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: std::collections::HashMap::new(),
            v: std::collections::HashMap::new(),
            step: 0,
        }
    }
}

impl<B: Backend> Optimizer<B> for AdamW<B> {
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        self.step += 1;
        let t_step = self.step as f64;
        let bias_correction1 = 1.0 - self.beta1.powf(t_step);
        let bias_correction2 = 1.0 - self.beta2.powf(t_step);

        for (name, var) in self.params.iter_mut() {
            if let Some(grad) = B::get_grad(var, &grads.0)? {
                let mut t = B::var_as_tensor(var)?;

                // Weight decay
                if self.weight_decay > 0.0 {
                    let decay = B::mul_scalar(&t, self.weight_decay * self.lr)?;
                    t = B::sub(&t, &decay)?;
                }

                let m_t = if let Some(m) = self.m.get(name) {
                    let term1 = B::mul_scalar(m, self.beta1)?;
                    let term2 = B::mul_scalar(&grad, 1.0 - self.beta1)?;
                    B::add(&term1, &term2)?
                } else {
                    B::mul_scalar(&grad, 1.0 - self.beta1)?
                };

                let grad_sq = B::mul(&grad, &grad)?;
                let v_t = if let Some(v) = self.v.get(name) {
                    let term1 = B::mul_scalar(v, self.beta2)?;
                    let term2 = B::mul_scalar(&grad_sq, 1.0 - self.beta2)?;
                    B::add(&term1, &term2)?
                } else {
                    B::mul_scalar(&grad_sq, 1.0 - self.beta2)?
                };

                self.m.insert(name.clone(), m_t.clone());
                self.v.insert(name.clone(), v_t.clone());

                let m_hat = B::mul_scalar(&m_t, 1.0 / bias_correction1)?;
                let v_hat = B::mul_scalar(&v_t, 1.0 / bias_correction2)?;

                // step = lr * m_hat / (sqrt(v_hat) + eps)
                let denom = B::add_scalar(&B::sqrt(&v_hat)?, self.eps)?;
                let step = B::mul_scalar(&B::div(&m_hat, &denom)?, self.lr)?;

                let updated = B::sub(&t, &step)?;
                B::assign_var(var, &updated)?;
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
/// use kindle::prelude::*;
///
/// let optimizer = Adam::new(model.parameters(), 1e-3);
/// optimizer.step(&gradients)?;
/// ```
pub struct Adam<B: Backend> {
    params: std::collections::HashMap<String, B::RawVar>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    m: std::collections::HashMap<String, B::RawTensor>,
    v: std::collections::HashMap<String, B::RawTensor>,
    step: usize,
}

impl<B: Backend> Adam<B> {
    pub fn new(params: std::collections::HashMap<String, B::RawVar>, lr: f64) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: std::collections::HashMap::new(),
            v: std::collections::HashMap::new(),
            step: 0,
        }
    }
}

impl<B: Backend> Optimizer<B> for Adam<B> {
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        self.step += 1;
        let t_step = self.step as f64;
        let bias_correction1 = 1.0 - self.beta1.powf(t_step);
        let bias_correction2 = 1.0 - self.beta2.powf(t_step);

        for (name, var) in self.params.iter_mut() {
            if let Some(grad) = B::get_grad(var, &grads.0)? {
                let t = B::var_as_tensor(var)?;

                let m_t = if let Some(m) = self.m.get(name) {
                    let term1 = B::mul_scalar(m, self.beta1)?;
                    let term2 = B::mul_scalar(&grad, 1.0 - self.beta1)?;
                    B::add(&term1, &term2)?
                } else {
                    B::mul_scalar(&grad, 1.0 - self.beta1)?
                };

                let grad_sq = B::mul(&grad, &grad)?;
                let v_t = if let Some(v) = self.v.get(name) {
                    let term1 = B::mul_scalar(v, self.beta2)?;
                    let term2 = B::mul_scalar(&grad_sq, 1.0 - self.beta2)?;
                    B::add(&term1, &term2)?
                } else {
                    B::mul_scalar(&grad_sq, 1.0 - self.beta2)?
                };

                self.m.insert(name.clone(), m_t.clone());
                self.v.insert(name.clone(), v_t.clone());

                let m_hat = B::mul_scalar(&m_t, 1.0 / bias_correction1)?;
                let v_hat = B::mul_scalar(&v_t, 1.0 / bias_correction2)?;

                // step = lr * m_hat / (sqrt(v_hat) + eps)
                let denom = B::add_scalar(&B::sqrt(&v_hat)?, self.eps)?;
                let step = B::mul_scalar(&B::div(&m_hat, &denom)?, self.lr)?;

                let updated = B::sub(&t, &step)?;
                B::assign_var(var, &updated)?;
            }
        }
        Ok(())
    }
}
