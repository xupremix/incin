//! The explicit execution context: a backend plus the policy it runs under.
//!
//! `EXE-006` needed only the backend, per decision D-002, and that is what this
//! type carried. `GRD-001` gives it the policy PROPOSALS.md sec. 1.2.5
//! declares, so that a decision like "never silently copy to the host" is a
//! value the caller passes rather than a global someone else configured.
//!
//! `GRD-002` supplies the `GradMode` sec. 1.2.5 lists, derived from the
//! existing `Grad`/`NoGrad` markers rather than declared beside them. One
//! field of that section is still absent: `AutotunePolicy` lives in
//! `incin-backends::tuning` and belongs to `TUN-003`, which extends this same
//! type when it lands. It is not declared early as an inert field, because a
//! field a caller can set and nothing reads is worse than a missing one.

use crate::exec::policy::{ExecutionPolicy, FallbackPolicy, GradMode, MathMode};
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
    ///
    /// Without `std` there is no thread-local to read and no
    /// [`ExecutionPolicy::scope`] to have installed anything, so the answer is
    /// the default policy — the same reasoning, and the same resolution, as
    /// [`GradMode::current`](crate::exec::GradMode::current). Offering it in
    /// both configurations rather than gating it on `std` keeps a caller like
    /// `Tensor::add` from having to exist only in one of them.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_scope(backend: B) -> Self {
        Self {
            backend,
            policy: ExecutionPolicy::current(),
        }
    }

    /// Own `backend` under the default policy. See the `std` form above.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn from_scope(backend: B) -> Self {
        Self::new(backend)
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
    pub const fn fallback(&self) -> FallbackPolicy {
        self.policy.fallback
    }

    /// The ceiling this context puts on gradient recording.
    ///
    /// A context that permits recording does not cause it: an operation
    /// records only if its operand's `G` also permits it. A context that
    /// disables it does decide, for everything run under it.
    #[must_use]
    pub const fn grad_mode(&self) -> GradMode {
        self.policy.grad_mode
    }

    #[must_use]
    pub const fn training(&self) -> bool {
        self.policy.training
    }

    #[must_use]
    pub const fn with_math_mode(mut self, math_mode: MathMode) -> Self {
        self.policy.math_mode = math_mode;
        self
    }

    #[must_use]
    pub const fn with_fallback(mut self, fallback: FallbackPolicy) -> Self {
        self.policy.fallback = fallback;
        self
    }

    #[must_use]
    pub const fn with_grad_mode(mut self, grad_mode: GradMode) -> Self {
        self.policy.grad_mode = grad_mode;
        self
    }

    #[must_use]
    pub const fn with_training(mut self, training: bool) -> Self {
        self.policy.training = training;
        self
    }

    #[must_use]
    pub const fn precision_policy(&self) -> crate::exec::RuntimePrecisionPolicy {
        self.policy.precision
    }

    #[must_use]
    pub const fn with_precision_policy(
        mut self,
        precision: crate::exec::RuntimePrecisionPolicy,
    ) -> Self {
        self.policy.precision = precision;
        self
    }
}
