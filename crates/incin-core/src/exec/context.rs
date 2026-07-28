//! The explicit execution context: a backend plus the policy it runs under.
//!
//! `EXE-006` needed only the backend, per decision D-002, and that is what this
//! type carried. `GRD-001` gives it the policy PROPOSALS.md sec. 1.2.5
//! declares, so that a decision like "this run must be reproducible" or "never
//! silently copy to the host" is a value the caller passes rather than a
//! global someone else configured.
//!
//! Two of the fields sec. 1.2.5 lists are deliberately absent here. `GradMode`
//! is derived from the existing `Grad`/`NoGrad` markers and belongs to
//! `GRD-002`, which owns that derivation; `AutotunePolicy` lives in
//! `incin-backends::tuning` and belongs to `TUN-003`. Both extend this same
//! type when they land. Neither is declared early as an inert field, because a
//! field a caller can set and nothing reads is worse than a missing one.

use crate::exec::policy::{
    AllocatorPolicy, Determinism, ExecutionPolicy, FallbackPolicy, MathMode,
};
use crate::tensor::backend::StorageBackend;

/// Explicit owner of the backend used by descriptor execution, and of the
/// policy that execution runs under.
#[derive(Debug, Clone)]
pub struct ExecutionContext<B: StorageBackend> {
    pub backend: B,
    pub policy: ExecutionPolicy,
}

impl<B: StorageBackend> ExecutionContext<B> {
    /// Own `backend` under the default policy.
    ///
    /// Per D-006 this always takes a backend value, never a device. Selecting
    /// a device is a separate fallible step that yields the backend.
    #[must_use]
    pub const fn new(backend: B) -> Self {
        Self {
            backend,
            policy: ExecutionPolicy::new(),
        }
    }

    /// Own `backend` under an explicitly supplied policy.
    #[must_use]
    pub const fn with_policy(backend: B, policy: ExecutionPolicy) -> Self {
        Self { backend, policy }
    }

    /// Own `backend` under whatever policy is ambient on this thread.
    ///
    /// This is the bridge between the scoped convenience form and the explicit
    /// one: it reads the ambient policy once, here, and the resulting context
    /// keeps that value even if the scope it was read from later ends.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_scope(backend: B) -> Self {
        Self {
            backend,
            policy: ExecutionPolicy::current(),
        }
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

    #[must_use]
    pub const fn policy(&self) -> ExecutionPolicy {
        self.policy
    }

    #[must_use]
    pub const fn math_mode(&self) -> MathMode {
        self.policy.math_mode
    }

    #[must_use]
    pub const fn determinism(&self) -> Determinism {
        self.policy.determinism
    }

    #[must_use]
    pub const fn fallback(&self) -> FallbackPolicy {
        self.policy.fallback
    }

    #[must_use]
    pub const fn allocator(&self) -> AllocatorPolicy {
        self.policy.allocator
    }

    #[must_use]
    pub const fn with_math_mode(mut self, math_mode: MathMode) -> Self {
        self.policy.math_mode = math_mode;
        self
    }

    #[must_use]
    pub const fn with_determinism(mut self, determinism: Determinism) -> Self {
        self.policy.determinism = determinism;
        self
    }

    #[must_use]
    pub const fn with_fallback(mut self, fallback: FallbackPolicy) -> Self {
        self.policy.fallback = fallback;
        self
    }

    #[must_use]
    pub const fn with_allocator(mut self, allocator: AllocatorPolicy) -> Self {
        self.policy.allocator = allocator;
        self
    }
}
