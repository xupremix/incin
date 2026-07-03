use crate::prelude::*;

/// Encapsulates the backend-specific gradients obtained from a backward pass.
pub struct Gradients<G>(pub G);

/// Trait defining a generic optimization algorithm.
pub trait Optimizer<B: Backend<Dyn>> {
    /// Steps the optimizer using the given gradients, updating the tracked parameters.
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()>;
}

/// Stochastic Gradient Descent Optimizer.
pub struct SGD<B: Backend<Dyn>> {
    params: Vec<B::RawVar>,
    lr: f64,
}

impl<B: Backend<Dyn>> SGD<B> {
    /// Create a new SGD optimizer.
    pub fn new(params: Vec<B::RawVar>, lr: f64) -> Self {
        Self { params, lr }
    }
}

impl<B: Backend<Dyn>> Optimizer<B> for SGD<B> {
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        B::step_sgd(&mut self.params, &grads.0, self.lr)
    }
}

/// AdamW Optimizer.
pub struct AdamW<B: Backend<Dyn>> {
    params: Vec<B::RawVar>,
    lr: f64,
}

impl<B: Backend<Dyn>> AdamW<B> {
    /// Create a new AdamW optimizer.
    pub fn new(params: Vec<B::RawVar>, lr: f64) -> Self {
        Self { params, lr }
    }
}

impl<B: Backend<Dyn>> Optimizer<B> for AdamW<B> {
    fn step(&mut self, grads: &Gradients<B::Grads>) -> Result<()> {
        B::step_adamw(&mut self.params, &grads.0, self.lr)
    }
}
