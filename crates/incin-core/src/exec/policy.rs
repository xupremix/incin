//! Execution policy shared by capability queries and kernel cache keys.
//!
//! Every type here answers a question that is orthogonal to *what* an
//! operation computes: how far its floating-point arithmetic may be
//! transformed, whether repeated runs must agree, what an executor is allowed
//! to do when it has no kernel, and where the memory comes from. They are
//! separate axes because they compose independently, and because collapsing
//! any two of them into one enum is what forces a cache to alias two requests
//! a caller deliberately distinguished.
//!
//! [`ExecutionPolicy`] groups the whole set. It is the half of an
//! [`ExecutionContext`](crate::exec::ExecutionContext) that names no backend,
//! which is what makes it something a scope can carry.

/// Floating-point transformation policy.
///
/// Determinism is deliberately orthogonal and lives in [`Determinism`]; a
/// deterministic request must not alias either numerical mode in a cache.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MathMode {
    #[default]
    Precise,
    Fast,
}

impl MathMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Precise => "precise",
            Self::Fast => "fast",
        }
    }
}

/// Whether repeated executions of the same request must produce identical
/// results.
///
/// `Permitted` is the default and does not mean results *will* vary; it means
/// nothing is promised. A reduction whose kernel accumulates in an order that
/// depends on how many blocks the device scheduled is permitted, and returns
/// the same answer to within rounding on every run without guaranteeing it.
/// `Required` is a filter, not a request: an executor that cannot prove it has
/// a deterministic path must refuse rather than fall back to a probably-stable
/// one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Determinism {
    #[default]
    Permitted,
    Required,
}

impl Determinism {
    #[must_use]
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
/// The default forbids everything, which is the whole point. A silent host
/// round-trip is the single easiest way to turn a GPU program into a slower
/// CPU program without anything in the code saying so, and PROPOSALS.md
/// sec. 1.2.4 requires a fallback to be an explicitly enabled policy rather
/// than a decision an executor makes on the caller's behalf.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FallbackPolicy {
    /// Refuse. An unsupported operation is an error naming the device.
    #[default]
    Deny,
    /// Allow an operation to be composed from other operations on the same
    /// device. No data crosses a device boundary and no layout is rewritten.
    AllowComposition,
    /// Allow moving or materializing data, including a host round-trip.
    /// Implies everything `AllowComposition` allows.
    AllowTransfer,
}

impl FallbackPolicy {
    #[must_use]
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

/// Where an executor's intermediate memory comes from.
///
/// This names a strategy, not an allocator implementation: the vocabulary has
/// to exist before a context can carry it, and a caller choosing `Direct` for
/// a reproducible memory profile is making a different request than one
/// choosing `Pooled` for throughput.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AllocatorPolicy {
    /// Every allocation goes to the device allocator and is released when it
    /// is dropped. Predictable, and slower under churn.
    #[default]
    Direct,
    /// Freed blocks are retained and handed out again. Faster under churn, and
    /// reports a high-water mark rather than a live total.
    Pooled,
}

impl AllocatorPolicy {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Pooled => "pooled",
        }
    }
}

/// Whether execution records an autograd node for the operations run under it.
///
/// This is the runtime form of the type-level `Grad`/`NoGrad` markers, and it
/// is derived from them rather than set beside them — see
/// [`RequiresGrad::grad_mode`](crate::tensor::grad::RequiresGrad::grad_mode).
/// The markers decide which tensor APIs exist; this decides what the layer
/// below the frontend does, because a backend kernel receives storage and
/// never sees `G`.
///
/// `Enabled` is the default and is a permission, not an instruction: an
/// operation records only if nothing in scope has disabled it. That asymmetry
/// is what makes [`no_grad`] work — a `Grad` operand inside a `no_grad` scope
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
    /// disabled it, which is what a `no_grad` block has to mean if it is to be
    /// usable around code whose operands are `Grad`.
    #[must_use]
    pub const fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Enabled, Self::Enabled) => Self::Enabled,
            _ => Self::Disabled,
        }
    }

    /// Run `body` under this mode combined with the ambient one, per
    /// [`and`](Self::and).
    ///
    /// This is what an operation whose result's `G` is known calls; [`scope`]
    /// is what a *caller* calls. The difference is the direction: this can
    /// only tighten, so it needs to install nothing when it is `Enabled`, and
    /// the common path — an operation on a `Grad` tensor — reads no
    /// thread-local at all.
    ///
    /// Without `std` there is no thread-local to install into, and no tape to
    /// install it for: every tape in the workspace lives in a backend that
    /// requires `std`. Running `body` unchanged is the whole `no_std` story
    /// here, not a silently weakened guarantee.
    ///
    /// [`scope`]: Self::scope
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

/// The complete policy half of an execution context: everything that is a
/// decision rather than a device.
///
/// Grouping these is what lets a scope carry them. A thread-local default
/// cannot be generic over a backend type, so the ambient value a scope
/// installs has to be the part of a context that names no backend, and this is
/// exactly that part.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionPolicy {
    pub math_mode: MathMode,
    pub determinism: Determinism,
    pub fallback: FallbackPolicy,
    pub allocator: AllocatorPolicy,
    pub grad_mode: GradMode,
}

impl ExecutionPolicy {
    /// The default policy, spelled out: precise arithmetic, no determinism
    /// promise, no fallback of any kind, a direct allocator, and gradient
    /// recording permitted.
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
            determinism: Determinism::Permitted,
            fallback: FallbackPolicy::Deny,
            allocator: AllocatorPolicy::Direct,
            grad_mode: GradMode::Enabled,
        }
    }

    #[must_use]
    pub const fn with_math_mode(mut self, math_mode: MathMode) -> Self {
        self.math_mode = math_mode;
        self
    }

    #[must_use]
    pub const fn with_determinism(mut self, determinism: Determinism) -> Self {
        self.determinism = determinism;
        self
    }

    #[must_use]
    pub const fn with_fallback(mut self, fallback: FallbackPolicy) -> Self {
        self.fallback = fallback;
        self
    }

    #[must_use]
    pub const fn with_allocator(mut self, allocator: AllocatorPolicy) -> Self {
        self.allocator = allocator;
        self
    }

    #[must_use]
    pub const fn with_grad_mode(mut self, grad_mode: GradMode) -> Self {
        self.grad_mode = grad_mode;
        self
    }
}

#[cfg(feature = "std")]
mod scope {
    use super::{ExecutionPolicy, GradMode};
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

    impl GradMode {
        /// The mode ambient on this thread, or [`GradMode::Enabled`] outside
        /// any scope.
        ///
        /// The tapes read this. It is the only thing standing between a
        /// `NoGrad` frontend operation and a recorded node, because a backend
        /// kernel receives storage and never sees `G`.
        #[must_use]
        pub fn current() -> Self {
            ExecutionPolicy::current().grad_mode
        }

        /// Run `body` with this mode ambient, leaving every other policy axis
        /// as it was.
        ///
        /// This installs the mode rather than combining with the enclosing
        /// one, so `GradMode::Enabled.scope(..)` inside a [`no_grad`] block
        /// records again — the caller who writes that is asking for it by
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

    /// Run `body` with gradient recording disabled, whatever the operands say.
    ///
    /// This is the inference form. Every operation inside records nothing and
    /// retains no saved tensor, including operations on `Grad` tensors: an
    /// operand's mode is combined with the ambient one through
    /// [`GradMode::and`], and a `Grad` operand permits recording rather than
    /// demanding it.
    pub fn no_grad<R>(body: impl FnOnce() -> R) -> R {
        GradMode::Disabled.scope(body)
    }
}

#[cfg(feature = "std")]
pub use scope::no_grad;
