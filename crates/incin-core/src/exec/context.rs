//! Backend-owning execution context foundation.
//!
//! EXE-006 needs the context in `ExecutionRequest` per decision D-002. GRD-001
//! extends this same type with gradient, determinism, fallback, allocator, and
//! autotuning policy; it does not introduce a parallel context.

use crate::tensor::backend::StorageBackend;

/// Explicit owner of the backend used by descriptor execution.
#[derive(Debug, Clone)]
pub struct ExecutionContext<B: StorageBackend> {
    pub backend: B,
}

impl<B: StorageBackend> ExecutionContext<B> {
    #[must_use]
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    #[must_use]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[must_use]
    pub fn into_backend(self) -> B {
        self.backend
    }
}
