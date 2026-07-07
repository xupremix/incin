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
    params: Vec<B::RawVar>,
    lr: f64,
}

impl<B: Backend> SGD<B> {
    /// Create a new SGD optimizer.
    pub fn new(params: Vec<B::RawVar>, lr: f64) -> Self {
        Self { params, lr }
    }
}

impl<B: Backend> Optimizer<B> for SGD<B> {
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        B::step_sgd(&mut self.params, &grads.0, self.lr)
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
/// 
/// Implements the optimizer from [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
/// Unlike standard Adam, weight decay is applied directly to the parameters, not the gradients.
pub struct AdamW<B: Backend> {
    params: Vec<B::RawVar>,
    lr: f64,
}

impl<B: Backend> AdamW<B> {
    /// Create a new AdamW optimizer.
    pub fn new(params: Vec<B::RawVar>, lr: f64) -> Self {
        Self { params, lr }
    }
}

impl<B: Backend> Optimizer<B> for AdamW<B> {
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        B::step_adamw(&mut self.params, &grads.0, self.lr)
    }
}

/// Adam optimizer.
/// 
/// Implements the optimizer from [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
/// Uses first and second moment estimates of the gradient to compute adaptive learning rates.
pub struct Adam<B: Backend> {
    params: Vec<B::RawVar>,
    lr: f64,
}

impl<B: Backend> Adam<B> {
    /// Create a new Adam optimizer.
    pub fn new(params: Vec<B::RawVar>, lr: f64) -> Self {
        Self { params, lr }
    }
}

impl<B: Backend> Optimizer<B> for Adam<B> {
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        B::step_adam(&mut self.params, &grads.0, self.lr)
    }
}
