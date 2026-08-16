//! Backend-owned gradient handles produced by autograd passes.

use crate::err::{Error, Result};
use crate::shapes::Shape;
use crate::tensor::backend::{AutogradBackend, Backend};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;

/// Encapsulates the backend-specific gradients obtained from a backward pass.
///
/// Autograd owns this wrapper. Optimizers consume it, but do not own the
/// result of graph differentiation.
pub struct Gradients<B: Backend + AutogradBackend>(B::Grads);

impl<B: Backend + AutogradBackend> Gradients<B> {
    pub(crate) fn from_backend(inner: B::Grads) -> Self {
        Self(inner)
    }

    /// Returns the gradient for a tensor when the backward pass produced one.
    pub fn get<S, K, G>(&self, tensor: &Tensor<S, B, K, G>) -> Result<Option<Tensor<S, B, K>>>
    where
        S: Shape,
        K: DType,
        G: RequiresGrad,
    {
        let Some(storage) = B::get_grad(tensor.inner(), &self.0)? else {
            return Ok(None);
        };
        Tensor::from_gradient_storage(tensor, storage).map(Some)
    }

    /// Returns the gradient for a tensor or a structured missing-gradient error.
    pub fn require<S, K, G>(&self, tensor: &Tensor<S, B, K, G>) -> Result<Tensor<S, B, K>>
    where
        S: Shape,
        K: DType,
        G: RequiresGrad,
    {
        self.get(tensor)?.ok_or_else(|| {
            Error::Backward(crate::err::BackwardError::Recipe {
                operation: crate::shapes::error::OperationKind::Storage,
                reason: "the backward pass produced no gradient for this tensor",
            })
        })
    }

    /// Borrows the backend-specific gradient container.
    #[must_use]
    pub fn as_backend(&self) -> &B::Grads {
        &self.0
    }

    /// Mutably borrows the backend container for backend-authoring and
    /// distributed gradient synchronization.
    #[must_use]
    pub fn as_backend_mut(&mut self) -> &mut B::Grads {
        &mut self.0
    }

    /// Consumes this handle and returns its backend-specific container.
    #[must_use]
    pub fn into_backend(self) -> B::Grads {
        self.0
    }
}
