//! Backend-owned gradient handles produced by tensor backward passes.

/// Encapsulates the backend-specific gradients obtained from a backward pass.
///
/// This wrapper belongs to the tensor/autograd layer. Optimizers consume it,
/// but do not own the result of graph differentiation.
pub struct Gradients<G>(G);

impl<G> Gradients<G> {
    pub(crate) fn from_backend(inner: G) -> Self {
        Self(inner)
    }

    /// Borrows the backend-specific gradient container.
    #[must_use]
    pub fn as_backend(&self) -> &G {
        &self.0
    }

    /// Mutably borrows the backend container for backend-authoring and
    /// distributed gradient synchronization.
    #[must_use]
    pub fn as_backend_mut(&mut self) -> &mut G {
        &mut self.0
    }

    /// Consumes this handle and returns its backend-specific container.
    #[must_use]
    pub fn into_backend(self) -> G {
        self.0
    }
}
