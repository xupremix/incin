//! Structured shape-resolution errors.
//!
//! Every shape rule that cannot be discharged by trait resolution reports its
//! failure through [`ShapeError`]. The design goal, from `PROPOSALS.md` §1.2.3,
//! is that a failure names four things without the caller having to reconstruct
//! them: *which operation* was being resolved, *where* in the shape it failed,
//! *what rule* was violated, and *what the operands actually were*.
//!
//! Statically invalid operations remain compile errors; this type describes the
//! mixed and dynamic cases, which return `Result` rather than panicking or
//! fabricating a scalar or empty shape.
//!
//! Every variant is `Copy` and allocation-free - `&'static str` and `usize`
//! only - so a shape rule can report a precise diagnostic from `no_std` code
//! and from a context that must not allocate.

use core::fmt;

/// The operation whose shape or dtype rule was being resolved.
///
/// This is the single operation vocabulary for the crate. It spans two levels
/// of granularity on purpose:
///
/// * coarse *families* ([`Storage`](Self::Storage) through
///   [`Normalization`](Self::Normalization)), retained only for policy
///   classification and legacy geometry diagnostics;
/// * exact catalog identities, which are the only identities allowed to prove
///   execution support.
///
/// Keeping both levels in one enum prevents parallel vocabularies, while
/// [`is_exact`](Self::is_exact) prevents a family from being mistaken for a
/// capability declaration.
macro_rules! define_operation_kind {
    ($(($variant:ident, $name:literal, $family:ident, $profile:ident, $attrs:ident, $min:expr, $max:expr, $legacy:literal),)*) => {
        /// Exact identities generated from the canonical operation catalog.
        #[non_exhaustive]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
        pub enum OperationKind {
            /// Allocating or copying storage, independent of any compute.
            Storage,
            /// Filling a tensor with a constant.
            Fill,
            /// Sampling a tensor from a distribution.
            Random,
            /// Elementwise compute, unary or binary.
            Pointwise,
            /// Reduction along one or more axes.
            Reduction,
            /// Normalization over a set of axes.
            Normalization,
            /// Broadcasting two shapes to a common shape.
            Broadcast,
            /// Reinterpreting a shape with the same element count.
            Reshape,
            /// Collapsing a contiguous axis range into one axis.
            Flatten,
            /// Removing a length-1 axis.
            Squeeze,
            /// Inserting a length-1 axis.
            Unsqueeze,
            /// Reordering axes.
            Permute,
            /// Exchanging two axes.
            Transpose,
            /// Taking a strided sub-range of one or more axes.
            Slice,
            /// Joining tensors along an existing axis.
            Concat,
            /// Joining tensors along a new axis.
            Stack,
            /// Matrix multiplication, including its batched form.
            MatMul,
            /// One-dimensional convolution.
            Conv1d,
            /// Two-dimensional convolution.
            Conv2d,
            /// One-dimensional pooling.
            Pool1d,
            /// Two-dimensional pooling.
            Pool2d,
            /// Two-dimensional pooling to a caller-chosen output extent.
            AdaptiveAvgPool2d,
            /// Gathering rows of a table by index.
            Embedding,
            $(
                #[doc = concat!("Canonical semantic operation `", $name, "`.")]
                $variant,
            )*
        }
    };
}

incin_operation_catalog!(define_operation_kind);

macro_rules! impl_catalog_operation_kind {
    ($(($variant:ident, $name:literal, $family:ident, $profile:ident, $attrs:ident, $min:expr, $max:expr, $legacy:literal),)*) => {
        impl OperationKind {
            /// Whether this is an executable exact identity rather than a
            /// descriptive family/legacy geometry identity.
            #[must_use]
            pub const fn is_exact(self) -> bool {
                matches!(self, $(Self::$variant)|*)
            }

            const fn exact_family(self) -> Option<Self> {
                match self {
                    $(Self::$variant => Some(Self::$family),)*
                    _ => None,
                }
            }

            const fn exact_name(self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => Some($name),)*
                    _ => None,
                }
            }
        }
    };
}

incin_operation_catalog!(impl_catalog_operation_kind);

impl OperationKind {
    /// The coarse family this operation resolves dtype policy at.
    ///
    /// Dtype support is a property of a *class* of work, not of an individual
    /// operation: a backend supports floating-point pointwise arithmetic, not
    /// `add` specifically. `EXE-001` folds every operation onto the six coarse
    /// variants so `incin-backends` can keep resolving policy at that
    /// granularity without a second enum to do it with.
    ///
    /// The groupings follow what the dtype rule actually is:
    ///
    /// * operations that only move or re-address bytes - every shape
    ///   manipulation, plus [`Embedding`](Self::Embedding), which gathers rows
    ///   by integer index - are [`Storage`](Self::Storage), and so accept
    ///   whatever dtype the backend can hold;
    /// * operations that accumulate - [`MatMul`](Self::MatMul), the
    ///   convolutions, and the poolings - are [`Reduction`](Self::Reduction),
    ///   because what makes them float-only and what earns them a widened
    ///   accumulator is that they sum many values into one.
    ///
    /// The result is always one of the six coarse variants, so the function is
    /// idempotent: `k.family().family() == k.family()`.
    #[must_use]
    pub const fn family(self) -> Self {
        if let Some(family) = self.exact_family() {
            return family;
        }
        match self {
            Self::Storage
            | Self::Broadcast
            | Self::Reshape
            | Self::Flatten
            | Self::Squeeze
            | Self::Unsqueeze
            | Self::Permute
            | Self::Transpose
            | Self::Slice
            | Self::Concat
            | Self::Stack
            | Self::Embedding => Self::Storage,
            Self::Fill => Self::Fill,
            Self::Random => Self::Random,
            Self::Pointwise => Self::Pointwise,
            Self::Reduction
            | Self::MatMul
            | Self::Conv1d
            | Self::Conv2d
            | Self::Pool1d
            | Self::Pool2d
            | Self::AdaptiveAvgPool2d => Self::Reduction,
            Self::Normalization => Self::Normalization,
            // Exact variants returned above. This arm keeps the method
            // forward-compatible with the non-exhaustive vocabulary.
            _ => Self::Storage,
        }
    }

    /// The lowercase name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        if let Some(name) = self.exact_name() {
            return name;
        }
        match self {
            Self::Storage => "storage",
            Self::Fill => "fill",
            Self::Random => "random",
            Self::Pointwise => "pointwise",
            Self::Reduction => "reduction",
            Self::Normalization => "normalization",
            Self::Broadcast => "broadcast",
            Self::Reshape => "reshape",
            Self::Flatten => "flatten",
            Self::Squeeze => "squeeze",
            Self::Unsqueeze => "unsqueeze",
            Self::Permute => "permute",
            Self::Transpose => "transpose",
            Self::Slice => "slice",
            Self::Concat => "concat",
            Self::Stack => "stack",
            Self::MatMul => "matmul",
            Self::Conv1d => "conv1d",
            Self::Conv2d => "conv2d",
            Self::Pool1d => "pool1d",
            Self::Pool2d => "pool2d",
            Self::AdaptiveAvgPool2d => "adaptive_avg_pool2d",
            Self::Embedding => "embedding",
            _ => "unknown_operation",
        }
    }
}

impl fmt::Display for OperationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The position in a shape at which a rule failed.
///
/// [`Named`](Self::Named) exists because the position alone is a poor
/// diagnostic for operations whose layout fixes the meaning of each axis:
/// "axis 1 mismatch" is much less useful than "axis 'channels' mismatch" for a
/// convolution.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    /// A positional axis, counted from the front of the shape.
    Index(usize),
    /// An axis identified by the role its operation gives it, such as
    /// `"channels"` or `"height"`.
    Named(&'static str),
    /// The rule constrains the shape as a whole rather than one axis - for
    /// example reshape's element-count equality.
    Whole,
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(i) => write!(f, "axis {i}"),
            Self::Named(name) => write!(f, "axis '{name}'"),
            Self::Whole => f.write_str("the shape as a whole"),
        }
    }
}

/// What an operation required of an operand's rank.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RankExpectation {
    /// Exactly this rank.
    Exactly(usize),
    /// This rank or higher.
    AtLeast(usize),
    /// This rank or lower - usually the rank ceiling of a shape rule.
    AtMost(usize),
    /// Within an inclusive range.
    Between {
        /// Lowest accepted rank.
        min: usize,
        /// Highest accepted rank.
        max: usize,
    },
    /// The same rank as another operand, whose rank is recorded so the message
    /// can state both sides.
    SameAs {
        /// Name of the operand that set the expectation, such as `"lhs"`.
        operand: &'static str,
        /// That operand's rank.
        rank: usize,
    },
}

impl fmt::Display for RankExpectation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exactly(n) => write!(f, "exactly {n}"),
            Self::AtLeast(n) => write!(f, "at least {n}"),
            Self::AtMost(n) => write!(f, "at most {n}"),
            Self::Between { min, max } => write!(f, "between {min} and {max}"),
            Self::SameAs { operand, rank } => write!(f, "the same rank as {operand} ({rank})"),
        }
    }
}

/// The rule that related two dimensions at the same axis.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimensionConstraint {
    /// The two dimensions must be equal.
    Equal,
    /// The two dimensions must be equal, or one of them must be 1.
    Broadcastable,
    /// The left dimension must be an exact multiple of the right.
    DivisibleBy,
    /// The left dimension must be at least the right.
    AtLeast,
}

impl DimensionConstraint {
    /// The rule as a clause that reads after "which must be".
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::Equal => "equal",
            Self::Broadcastable => "equal, or one of them 1",
            Self::DivisibleBy => "an exact multiple",
            Self::AtLeast => "greater than or equal",
        }
    }
}

impl fmt::Display for DimensionConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.describe())
    }
}

/// A shape rule that could not be discharged.
///
/// Returned by every fallible shape computation. Statically invalid programs do
/// not reach this type - they fail to compile.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeError {
    /// An operand had the wrong number of dimensions.
    #[error("{operation}: expected rank {expected}, got {actual}")]
    RankMismatch {
        /// Operation being resolved.
        operation: OperationKind,
        /// What the operation required.
        expected: RankExpectation,
        /// The rank the operand actually had.
        actual: usize,
    },

    /// Two dimensions at the same axis violated the operation's rule.
    #[error("{operation}: {axis} mismatch: {lhs} vs {rhs}, which must be {constraint}")]
    DimensionMismatch {
        /// Operation being resolved.
        operation: OperationKind,
        /// Where the rule failed.
        axis: Axis,
        /// The left operand's dimension.
        lhs: usize,
        /// The right operand's dimension.
        rhs: usize,
        /// The rule that related them.
        constraint: DimensionConstraint,
    },

    /// An axis range was empty, reversed, or reached past the operand's rank.
    #[error("{operation}: axis range {start}..{end} is invalid for rank {rank}")]
    InvalidAxisRange {
        /// Operation being resolved.
        operation: OperationKind,
        /// Inclusive start of the requested range.
        start: usize,
        /// Exclusive end of the requested range.
        end: usize,
        /// Rank of the operand the range was applied to.
        rank: usize,
    },

    /// A scalar operation parameter was outside its legal domain - most often a
    /// stride or kernel extent of 0.
    #[error("{operation}: parameter '{parameter}' has invalid value {value}")]
    InvalidParameter {
        /// Operation being resolved.
        operation: OperationKind,
        /// Name of the parameter, as it is spelled in the public API.
        parameter: &'static str,
        /// The rejected value.
        value: usize,
    },

    /// A shape computation overflowed `usize`.
    ///
    /// `expression` names the failing term of the operation's formula, so a
    /// diagnostic points at one multiplication rather than the whole rule.
    #[error("{operation}: arithmetic overflow evaluating '{expression}'")]
    ArithmeticOverflow {
        /// Operation being resolved.
        operation: OperationKind,
        /// The term that overflowed, written as it appears in the formula.
        expression: &'static str,
    },

    /// A computed dimension list did not fit the typed shape it must produce.
    ///
    /// Raised where a rule has already computed its output dimensions as plain
    /// numbers and must rebuild the typed field from them: either the rank
    /// differs, or an axis the target type fixes at compile time disagrees.
    ///
    /// The axis is deliberately absent. Recovering it needs per-axis knowledge
    /// that the erased `&[usize]` form has thrown away - which is the reason
    /// `SHP-004` reconstructs the field axis by axis wherever the arity is
    /// known, and reports [`DimensionMismatch`](Self::DimensionMismatch) with a
    /// real axis there. This variant covers what is left.
    #[error("{operation}: the computed rank-{rank} shape does not fit the target shape type")]
    TargetShapeRejected {
        /// Operation being resolved.
        operation: OperationKind,
        /// Rank of the dimension list that was rejected.
        rank: usize,
    },

    /// An axis index reached past the operand's rank.
    #[error("axis {axis} is invalid for rank {rank}")]
    InvalidAxis {
        /// Axis index.
        axis: usize,
        /// Rank of the operand.
        rank: usize,
    },

    /// An axis selector specified a duplicate axis.
    #[error("axis {axis} specified multiple times in selector sequence")]
    DuplicateAxis {
        /// Axis index that was duplicated.
        axis: usize,
    },

    /// A named selector did not occur in the current structural shape.
    #[error("named axis '{name}' is not present in the shape")]
    MissingNamedAxis { name: &'static str },

    /// A named selector occurred more than once and cannot be resolved implicitly.
    #[error("named axis '{name}' is ambiguous: it occurs more than once")]
    AmbiguousNamedAxis { name: &'static str },

    /// Two positionally paired broadcast axes carry different semantic names.
    #[error("broadcast axis {axis} has conflicting names '{lhs}' and '{rhs}'")]
    ConflictingNamedAxes {
        /// Right-aligned output axis.
        axis: usize,
        /// Name from the left operand.
        lhs: &'static str,
        /// Name from the right operand.
        rhs: &'static str,
    },

    /// The rule resolved successfully but produced a zero-length axis.
    ///
    /// This is separate from [`DimensionMismatch`](Self::DimensionMismatch)
    /// because no input dimension is wrong: a kernel simply does not fit its
    /// padded input. It is the case the pre-`SHP-005` spatial code silently
    /// produced instead of reporting.
    #[error("{operation}: {axis} would have length 0")]
    EmptyOutput {
        /// Operation being resolved.
        operation: OperationKind,
        /// The axis that collapsed.
        axis: Axis,
    },
}

impl ShapeError {
    /// The operation that failed, for callers that route on it rather than
    /// matching every variant.
    #[must_use]
    pub const fn operation(&self) -> OperationKind {
        match self {
            Self::RankMismatch { operation, .. }
            | Self::DimensionMismatch { operation, .. }
            | Self::InvalidAxisRange { operation, .. }
            | Self::InvalidParameter { operation, .. }
            | Self::ArithmeticOverflow { operation, .. }
            | Self::TargetShapeRejected { operation, .. }
            | Self::EmptyOutput { operation, .. } => *operation,
            _ => OperationKind::Storage,
        }
    }

    /// The axis the failure is attributed to, when the variant names one.
    pub const fn axis(&self) -> Option<Axis> {
        match self {
            Self::DimensionMismatch { axis, .. } | Self::EmptyOutput { axis, .. } => Some(*axis),
            Self::InvalidAxis { axis, .. } | Self::DuplicateAxis { axis } => {
                Some(Axis::Index(*axis))
            }
            _ => None,
        }
    }
}
