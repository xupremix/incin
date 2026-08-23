//! Execution policy shared by capability queries and kernel cache keys.
//!
//! Every type here answers a question that is orthogonal to *what* an
//! operation computes: how far its floating-point arithmetic may be
//! transformed, what an executor is allowed to do when it has no kernel, and
//! whether execution records gradients. These are the decisions stable
//! execution currently enforces.
//!
//! [`ExecutionPolicy`] groups the whole set. It is the half of an
//! [`ExecutionContext`](crate::exec::ExecutionContext) that names no backend,
//! which is what makes it something a scope can carry.

/// Floating-point transformation policy.
///
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MathMode {
    #[default]
    /// Favor numerical precision even when slower paths are required.
    Precise,
    /// Favor throughput; precision shortcuts become available.
    Fast,
}

impl MathMode {
    #[must_use]
    /// Stable string spelling used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Precise => "precise",
            Self::Fast => "fast",
        }
    }
}

/// Whether a tuning request requires deterministic candidates.
///
/// This remains separate from [`ExecutionPolicy`]: backend tuning enforces it
/// when selecting candidates, while general descriptor execution has no
/// corresponding deterministic-kernel admission contract.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Determinism {
    #[default]
    /// Determinism may be relaxed where documented.
    Permitted,
    /// Determinism required; non-deterministic kernels refuse to launch.
    Required,
}

impl Determinism {
    #[must_use]
    /// Stable string spelling used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Permitted => "permitted",
            Self::Required => "required",
        }
    }

    /// True when this policy forbids a nondeterministic implementation.
    #[must_use]
    pub const fn is_required(self) -> bool {
        matches!(self, Self::Required)
    }
}

/// What an executor may do when the requested operation has no kernel on the
/// device the context owns.
///
/// The default permits same-device composition but not transfer. A silent host
/// round-trip is the single easiest way to turn a GPU program into a slower
/// CPU program without anything in the code saying so.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FallbackPolicy {
    /// Refuse composed and transfer fallback.
    Deny,
    /// Allow an operation to be composed from other operations on the same
    /// device. No data crosses a device boundary and no layout is rewritten.
    #[default]
    AllowComposition,
    /// Allow moving or materializing data, including a host round-trip.
    /// Implies everything `AllowComposition` allows.
    AllowTransfer,
}

impl FallbackPolicy {
    #[must_use]
    /// Stable string spelling used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Deny => "deny",
            Self::AllowComposition => "allow-composition",
            Self::AllowTransfer => "allow-transfer",
        }
    }

    /// True when an operation may be built out of other operations that stay
    /// on this device.
    #[must_use]
    pub const fn allows_composition(self) -> bool {
        matches!(self, Self::AllowComposition | Self::AllowTransfer)
    }

    /// True when data may cross a device boundary or be materialized.
    #[must_use]
    pub const fn allows_transfer(self) -> bool {
        matches!(self, Self::AllowTransfer)
    }
}

/// Whether execution records an autograd node for the operations run under it.
///
/// This is the runtime form of the type-level `Grad`/`NoGrad` markers, and it
/// is derived from them rather than set beside them - see
/// [`RequiresGrad::grad_mode`](crate::tensor::grad::RequiresGrad::grad_mode).
/// The markers decide which tensor APIs exist; this decides what the layer
/// below the frontend does, because a backend kernel receives storage and
/// never sees `G`.
///
/// `Enabled` is the default and is a permission, not an instruction: an
/// operation records only if nothing in scope has disabled it. That asymmetry
/// is what makes a disabled gradient scope work. A `Grad` operand inside such
/// a scope
/// records nothing, while a `NoGrad` operand records nothing anywhere.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GradMode {
    /// Operations may record an autograd node.
    #[default]
    Enabled,
    /// Operations record nothing and retain no backward-only tensor.
    Disabled,
}

impl GradMode {
    #[must_use]
    /// Stable string spelling used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
        }
    }

    /// True when an operation run under this mode may push a tape entry.
    ///
    /// The tapes call this, so it is the single place the guarantee is
    /// spelled: a mode that does not record produces no node and retains no
    /// saved tensor.
    #[must_use]
    pub const fn records(self) -> bool {
        matches!(self, Self::Enabled)
    }

    /// The stricter of two modes: recording requires both to permit it.
    ///
    /// Ordering the combination this way is the whole contract. An operand
    /// that permits recording cannot re-enable it inside a scope that
    /// disabled it, which is what a disabled gradient scope has to mean if it is to be
    /// usable around code whose operands are `Grad`.
    #[must_use]
    pub const fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Enabled, Self::Enabled) => Self::Enabled,
            _ => Self::Disabled,
        }
    }

    /// The mode ambient on this thread, or [`GradMode::Enabled`] outside any
    /// scope.
    ///
    /// The tapes read this. It is the only thing standing between a `NoGrad`
    /// frontend operation and a recorded node, because a backend kernel
    /// receives storage and never sees `G`.
    ///
    /// Without `std` there is no thread-local to read and no policy scope to have
    /// installed anything, so the answer is the default. That is the true
    /// state rather than a weakened guarantee: nothing in a `no_std` build can
    /// express a disabled scope, and every tape in the workspace lives in a
    /// backend that requires `std`.
    ///
    #[must_use]
    pub fn current() -> Self {
        #[cfg(feature = "std")]
        {
            ExecutionPolicy::current().grad_mode
        }
        #[cfg(not(feature = "std"))]
        {
            Self::Enabled
        }
    }

    /// Run `body` under this mode combined with the ambient one, per
    /// [`and`](Self::and).
    ///
    /// This is what an operation whose result's `G` is known calls; the policy
    /// scope
    /// is what a *caller* calls. The difference is the direction: this can
    /// only tighten, so it needs to install nothing when it is `Enabled`, and
    /// the common path - an operation on a `Grad` tensor - reads no
    /// thread-local at all.
    ///
    /// Without `std` there is nothing to install into; see
    /// [`current`](Self::current) for why that is the true answer rather than
    /// a weakened one.
    ///
    #[inline]
    pub fn restrict<R>(self, body: impl FnOnce() -> R) -> R {
        #[cfg(feature = "std")]
        {
            match self {
                Self::Enabled => body(),
                Self::Disabled => self.scope(body),
            }
        }
        #[cfg(not(feature = "std"))]
        {
            body()
        }
    }
}

/// Whether a backward pass validates each gradient as it is produced.
///
/// PROPOSALS.md sec. 3.9 requires this to be "an execution policy applied
/// consistently across backends, not a panic-only backend helper", and before
/// `GRD-005` it was the second of those: a `Backend::backward_with_nan_check`
/// beside `Backend::backward`, panicking. A caller who wanted the check
/// without the abort had no spelling, and one who wanted the abort without the
/// check had no reason to want it.
///
/// As an axis it composes with the rest: a debugging run sets it beside
/// `MathMode::Precise` and leaves everything else alone, and the one walk all
/// backends share is what reads it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NanPolicy {
    /// Do not inspect gradient values. The default, and the only one that
    /// costs nothing: the check reads every element of every gradient, which
    /// on a device backend is a full readback per contribution.
    #[default]
    Permit,
    /// Fail at the first non-finite gradient, naming the tensor it was found
    /// on.
    ///
    /// The point is *where* rather than *whether*: a training run notices a
    /// `NaN` loss on its own, and what it cannot do is say which operation
    /// produced it.
    Reject,
}

impl NanPolicy {
    #[must_use]
    /// Stable string spelling used in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Permit => "permit",
            Self::Reject => "reject",
        }
    }

    /// True when a backward pass must inspect the values it produces.
    #[must_use]
    pub const fn checks(self) -> bool {
        matches!(self, Self::Reject)
    }

    /// The policy ambient on this thread, or [`NanPolicy::Permit`] outside any
    /// scope.
    ///
    /// The shared backward walk reads this. Without `std` there is no scope to
    /// have installed anything, so the answer is the default - see
    /// [`GradMode::current`] for why that is the true answer rather than a
    /// weakened one.
    #[must_use]
    pub fn current() -> Self {
        #[cfg(feature = "std")]
        {
            ExecutionPolicy::current().nan_policy
        }
        #[cfg(not(feature = "std"))]
        {
            Self::Permit
        }
    }
}

/// The complete policy half of an execution context: everything that is a
/// decision rather than a device.
///
/// Grouping these is what lets a scope carry them. A thread-local default
/// cannot be generic over a backend type, so the ambient value a scope
/// installs has to be the part of a context that names no backend, and this is
/// exactly that part.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionPolicy {
    /// Math mode in force.
    pub math_mode: MathMode,
    /// Fallback policy in force.
    pub fallback: FallbackPolicy,
    /// Training flag in force.
    pub training: bool,
    /// Gradient recording mode in force.
    pub grad_mode: GradMode,
    /// NaN handling policy in force.
    pub nan_policy: NanPolicy,
    /// Runtime precision policy in force.
    pub precision: crate::exec::RuntimePrecisionPolicy,
}

impl ExecutionPolicy {
    /// The default policy, spelled out: precise arithmetic, same-device
    /// composition allowed, evaluation mode, and gradient recording permitted.
    ///
    /// `GradMode::Enabled` is the only default here that permits rather than
    /// forbids, and it has to be: the ambient value is the *ceiling* a
    /// tensor's own `G` is combined with, so a default of `Disabled` would
    /// mean no `Grad` tensor anywhere records without the caller opting back
    /// in.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            math_mode: MathMode::Precise,
            fallback: FallbackPolicy::AllowComposition,
            training: false,
            grad_mode: GradMode::Enabled,
            nan_policy: NanPolicy::Permit,
            precision: crate::exec::RuntimePrecisionPolicy::fp32(),
        }
    }

    #[must_use]
    /// Sets one math mode.
    pub const fn with_math_mode(mut self, math_mode: MathMode) -> Self {
        self.math_mode = math_mode;
        self
    }

    #[must_use]
    /// Sets one fallback policy.
    pub const fn with_fallback(mut self, fallback: FallbackPolicy) -> Self {
        self.fallback = fallback;
        self
    }

    #[must_use]
    /// Sets one training flag.
    pub const fn with_training(mut self, training: bool) -> Self {
        self.training = training;
        self
    }

    #[must_use]
    /// Sets one gradient-recording mode.
    pub const fn with_grad_mode(mut self, grad_mode: GradMode) -> Self {
        self.grad_mode = grad_mode;
        self
    }

    #[must_use]
    /// Sets one NaN policy.
    pub const fn with_nan_policy(mut self, nan_policy: NanPolicy) -> Self {
        self.nan_policy = nan_policy;
        self
    }

    #[must_use]
    /// Sets one precision policy.
    pub const fn with_precision(mut self, precision: crate::exec::RuntimePrecisionPolicy) -> Self {
        self.precision = precision;
        self
    }

    #[must_use]
    /// Reads the precision policy.
    pub const fn precision(&self) -> crate::exec::RuntimePrecisionPolicy {
        self.precision
    }
}

#[cfg(feature = "std")]
mod scope {
    use super::{ExecutionPolicy, GradMode, NanPolicy};
    use core::cell::Cell;

    std::thread_local! {
        /// The ambient policy for this thread. Thread-local rather than global
        /// because PROPOSALS.md sec. 1.2.5 calls explicit contexts the
        /// canonical thread-safe interface: a scope is a per-thread
        /// convenience, and one thread's scope must not be observable from
        /// another.
        static CURRENT: Cell<ExecutionPolicy> = const { Cell::new(ExecutionPolicy::new()) };
    }

    /// Restores the policy that was ambient when a scope was entered.
    ///
    /// This is a guard rather than a restore at the end of `scope` so that an
    /// unwinding panic cannot leave a thread with a policy no enclosing scope
    /// asked for. A poisoned ambient policy would outlive the scope that set
    /// it and silently apply to unrelated later work on the same thread.
    struct Restore(ExecutionPolicy);

    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|current| current.set(self.0));
        }
    }

    impl ExecutionPolicy {
        /// The policy ambient on this thread, or the default outside any
        /// scope.
        #[must_use]
        pub fn current() -> Self {
            CURRENT.with(Cell::get)
        }

        /// Run `body` with this policy ambient on the current thread,
        /// restoring the previous one on the way out.
        ///
        /// Scopes nest: `body` may enter another scope, and leaving it
        /// restores this one rather than the default.
        pub fn scope<R>(self, body: impl FnOnce() -> R) -> R {
            let _restore = Restore(CURRENT.with(|current| current.replace(self)));
            body()
        }
    }

    impl NanPolicy {
        /// Run `body` with this policy ambient, leaving every other axis as it
        /// was.
        pub fn scope<R>(self, body: impl FnOnce() -> R) -> R {
            ExecutionPolicy::current().with_nan_policy(self).scope(body)
        }
    }

    /// Run `body` with every gradient checked for a non-finite value.
    ///
    /// The debugging counterpart to [`ExecutionPolicy::scope`]: a backward pass inside this
    /// fails at the operation that first produced a `NaN` instead of at
    /// whatever notices later.
    pub fn check_gradients<R>(body: impl FnOnce() -> R) -> R {
        NanPolicy::Reject.scope(body)
    }

    impl GradMode {
        /// Run `body` with this mode ambient, leaving every other policy axis
        /// as it was.
        ///
        /// This installs the mode rather than combining with the enclosing
        /// one, so `GradMode::Enabled.scope(..)` inside a disabled scope
        /// records again - the caller who writes that is asking for it by
        /// name. Combining is what an *operand's* mode does, and that happens
        /// where the operand is known, not here.
        ///
        /// Reusing [`ExecutionPolicy::scope`] rather than adding a second
        /// thread-local is deliberate: two ambient mechanisms would restore
        /// independently, and a panic unwinding through both would have to
        /// leave them consistent by coincidence.
        pub fn scope<R>(self, body: impl FnOnce() -> R) -> R {
            ExecutionPolicy::current().with_grad_mode(self).scope(body)
        }
    }
}

#[cfg(feature = "std")]
pub use scope::check_gradients;
