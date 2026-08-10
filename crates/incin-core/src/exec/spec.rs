//! Frozen operation descriptors.
//!
//! A descriptor is the resolved form of one operation: everything a backend
//! needs to launch it, computed once from a shape that the frontend has already
//! proved legal. `PROPOSALS.md` §1.2.1 lists the four that anchor the design,
//! chosen because their launch parameters are direct consequences of shape
//! proofs --- [`MatMulSpec`], [`BroadcastSpec`], [`ReductionSpec`], and
//! [`Conv2dSpec`]. [`Pool2dSpec`] and [`ReshapeSpec`] joined them under
//! `EXE-003`, which needs a descriptor for each of the six operations it
//! lowers; decision `D-018` records why neither reuses one of the first four.
//!
//! # What "frozen" means
//!
//! Two properties, and [`DescriptorSchemaVersion`] exists to police the second:
//!
//! * **A descriptor is derived, never asserted.** Every constructor here takes
//!   operand *shapes* and computes the rest. There is no way to hand a
//!   descriptor an output shape that disagrees with its inputs, or a broadcast
//!   mask that disagrees with its strides, because neither is an argument. This
//!   is what lets `EXE-007` onward delete validation from kernels instead of
//!   moving it.
//! * **The field set is a versioned contract.** Kernel caches, autotune
//!   records, and specialization keys are all derived from descriptor contents.
//!   Adding, removing, or reinterpreting a field invalidates them, so it must
//!   bump [`DescriptorSchemaVersion::CURRENT`] --- a change a pinning test makes
//!   deliberate rather than accidental.
//!
//! Descriptors are `#[non_exhaustive]` with public fields: readable everywhere,
//! constructible only through the checked constructors in this module.
//!
//! # What a descriptor does not hold
//!
//! Storage offsets, dtype, device, and alignment are *per-tensor* facts, not
//! per-operation ones, and belong to `TensorMeta` (`EXE-004`). Keeping them out
//! is what lets one descriptor be reused across operands, cached, and used as a
//! specialization key. A descriptor holds logical geometry only.
//!
//! # Rank handling
//!
//! Descriptors use [`AxisSet`] for semantic axis collections. Its inline
//! [`AxisMask`] representation is only an optimization; axes beyond 63 spill
//! into owned storage. Descriptor construction therefore has no frontend
//! rank ceiling. Backend capability and resource-policy limits are checked at
//! their respective boundaries.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

use super::sealed::Sealed;
use crate::shapes::buf::{ShapeBuf, StrideBuf};
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation, ShapeError};
use crate::shapes::spatial::spatial_out_size;

// --- schema version ---------------------------------------------------------

/// The version of the descriptor field layout.
///
/// Anything derived from descriptor *contents* and kept across runs --- a kernel
/// cache, an autotune record, a serialized execution plan --- is only valid for
/// the schema it was produced under. Comparing versions is how a stale entry is
/// recognized instead of being replayed against fields that have since changed
/// meaning.
///
/// Bump [`CURRENT`](Self::CURRENT) whenever a descriptor in this module gains,
/// loses, or reinterprets a field. Adding a whole new descriptor does not
/// require a bump: nothing cached can refer to it yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DescriptorSchemaVersion(u32);

impl DescriptorSchemaVersion {
    /// The schema the descriptors in this module are currently frozen at.
    ///
    /// v2 gave [`ReductionSpec`] and [`Pool2dSpec`] the operator that runs
    /// inside their geometry. A v1 cache entry keyed on either records a window
    /// without saying what accumulated in it, so it cannot be replayed.
    ///
    /// v3 gave [`BroadcastSpec`] the same treatment, with an operator that may
    /// be absent. A v2 entry keyed on one cannot say whether it described a
    /// stretch or the geometry of a binary operation, and the two produce
    /// different output storage from identical strides.
    pub const CURRENT: Self = Self(3);

    /// Name a specific version, for reading a cache entry or plan back.
    #[must_use]
    pub const fn new(version: u32) -> Self {
        Self(version)
    }

    /// The raw version number, for embedding in a cache key or file header.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether data written under `self` may be read under `other`.
    ///
    /// Exact equality, deliberately. A descriptor schema has no compatible
    /// subset: a field whose meaning changed is not detectable by a range
    /// check, and re-deriving a descriptor is cheap next to executing one
    /// against a stale cache entry.
    #[must_use]
    pub const fn is_compatible_with(self, other: Self) -> bool {
        self.0 == other.0
    }
}

impl fmt::Display for DescriptorSchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// --- operators --------------------------------------------------------------

/// Which accumulation a [`ReductionSpec`] performs.
///
/// The geometry of a reduction says how many elements collapse into each result
/// element. It does not say what happens to them, and a backend cannot execute
/// one without knowing: `sum` and `max` walk identical loops and compute
/// different answers. This is the part the descriptor names.
///
/// The set is closed at the five accumulations whose result has the shape
/// [`ReductionSpec`] derives. `argmax` and `argmin` collapse the same axes but
/// return indices, so their result dtype differs from their input's and they are
/// a different operation; `cumsum` and `topk` do not collapse an axis at all.
/// None of the three is expressible here, deliberately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Add the collapsed elements.
    Sum,
    /// Average the collapsed elements.
    Mean,
    /// Take the largest collapsed element.
    Max,
    /// Take the smallest collapsed element.
    Min,
    /// Multiply the collapsed elements.
    Prod,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Min => "min",
            Self::Prod => "prod",
        })
    }
}

/// Which elementwise combination a [`BroadcastSpec`] performs.
///
/// [`BroadcastSpec`] is the one descriptor whose operator may be absent, because
/// it is the one descriptor that is useful without one: it describes iteration
/// geometry, and a named broadcast stretches an operand without combining it
/// with anything. `Option<BinaryOp>` is that distinction --- `None` is a stretch,
/// `Some` is a stretch a kernel then folds two operands through.
///
/// The set is closed at the four operations the shared broadcasting path
/// implements. `maximum` and `minimum` are deliberately absent: they read only
/// the left operand's shape and index both with it, so they require equal shapes
/// and would ask this geometry for something no kernel behind it performs.
/// Comparisons are absent for the reason `argmax` is absent from [`ReduceOp`] ---
/// their result dtype is not their input's, so they are a different operation
/// wearing the same shape rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Add the paired elements.
    Add,
    /// Subtract the right element from the left.
    Sub,
    /// Multiply the paired elements.
    Mul,
    /// Divide the left element by the right.
    Div,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
        })
    }
}

/// Which accumulation a [`Pool2dSpec`] performs inside its window.
///
/// Separate from [`ReduceOp`] even though both are window accumulations, because
/// the two sets are not the same one: pooling has no product or sum form in any
/// backend here, and `Average` over a padded window is not `Mean` over the
/// elements present. Merging them would let a descriptor ask for a pool no
/// kernel implements and would have to be refused later, at the backend, instead
/// of being unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolOp {
    /// Take the largest element in the window.
    Max,
    /// Average the window, counting padded positions as zero.
    Average,
}

impl fmt::Display for PoolOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Max => "max",
            Self::Average => "average",
        })
    }
}

// --- axis mask --------------------------------------------------------------

/// Compact inline storage for the semantic [`AxisSet`] axis collection.
///
/// Reductions name the axes they collapse, and broadcasts name the axes they
/// stretch. Both are sets over a small range, both are read in kernel inner
/// loops, and neither may allocate. A bitmask gives all three, plus a form that
/// passes to a native kernel as a single scalar --- see [`bits`](Self::bits).
///
/// Axes are counted from the front of the shape, so axis 0 is the outermost.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct AxisMask(u64);

impl AxisMask {
    /// The number of axes a mask can address inline (64).
    pub const MAX_AXES: usize = u64::BITS as usize;

    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// Reinterpret a raw 32-bit pattern.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits as u64)
    }

    /// Reinterpret a raw 64-bit pattern.
    #[must_use]
    pub const fn from_bits_u64(bits: u64) -> Self {
        Self(bits)
    }

    /// The raw 32-bit pattern (saturates if axes >= 32 are present).
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0 as u32
    }

    /// The raw 64-bit pattern.
    #[must_use]
    pub const fn bits_u64(self) -> u64 {
        self.0
    }

    /// Whether `axis` is in the set.
    #[must_use]
    pub const fn contains(self, axis: usize) -> bool {
        axis < Self::MAX_AXES && (self.0 >> axis) & 1 == 1
    }

    /// How many axes are in the set.
    #[must_use]
    pub const fn count(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set with `axis` added, or `None` if `axis` is out of range.
    #[must_use]
    pub const fn insert(self, axis: usize) -> Option<Self> {
        if axis < Self::MAX_AXES {
            Some(Self(self.0 | (1u64 << axis)))
        } else {
            None
        }
    }

    /// The set with `axis` removed. An out-of-range axis was never present, so
    /// removing it is a no-op rather than an error.
    #[must_use]
    pub const fn remove(self, axis: usize) -> Self {
        if axis < Self::MAX_AXES {
            Self(self.0 & !(1u64 << axis))
        } else {
            self
        }
    }

    /// Every axis of a shape of this rank, or `None` if the rank exceeds
    /// [`MAX_AXES`](Self::MAX_AXES).
    #[must_use]
    pub const fn all_below(rank: usize) -> Option<Self> {
        if rank > Self::MAX_AXES {
            None
        } else if rank == Self::MAX_AXES {
            Some(Self(u64::MAX))
        } else {
            Some(Self((1u64 << rank) - 1))
        }
    }

    /// The axes of a shape of this rank that are *not* in the set.
    ///
    /// The rank is required because a mask does not know it: the complement of
    /// "axis 1" is a different set for a rank-2 shape than for a rank-5 one.
    #[must_use]
    pub const fn complement_within(self, rank: usize) -> Option<Self> {
        match Self::all_below(rank) {
            Some(all) => Some(Self(all.0 & !self.0)),
            None => None,
        }
    }

    /// Build a mask from listed axes, checking each against `rank`.
    ///
    /// A repeated axis is rejected rather than absorbed. In a set, the second
    /// mention has no effect, which means `sum(dims = [1, 1])` would silently
    /// behave as `sum(dims = [1])` --- an argument list that says one thing and
    /// does another is worth a diagnostic.
    pub fn try_from_axes(
        operation: OperationKind,
        rank: usize,
        axes: impl IntoIterator<Item = usize>,
    ) -> Result<Self, ShapeError> {
        if rank > Self::MAX_AXES {
            return Err(ShapeError::RankMismatch {
                operation,
                expected: RankExpectation::AtMost(Self::MAX_AXES),
                actual: rank,
            });
        }
        let mut mask = Self::EMPTY;
        for axis in axes {
            if axis >= rank || mask.contains(axis) {
                return Err(ShapeError::InvalidParameter {
                    operation,
                    parameter: "axis",
                    value: axis,
                });
            }
            mask = match mask.insert(axis) {
                Some(wider) => wider,
                // Unreachable: `axis < rank`, and `rank <= MAX_AXES` above.
                None => {
                    return Err(ShapeError::RankMismatch {
                        operation,
                        expected: RankExpectation::AtMost(Self::MAX_AXES),
                        actual: rank,
                    });
                }
            };
        }
        Ok(mask)
    }

    /// The axes in the set, in ascending order.
    pub fn axes(self) -> impl Iterator<Item = usize> {
        let mut bits = self.0;
        core::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let axis = bits.trailing_zeros() as usize;
            // Clear the lowest set bit.
            bits &= bits - 1;
            Some(axis)
        })
    }

    /// Whether the set is a single unbroken run of axes.
    ///
    /// Descriptors whose geometry collapses a shape into contiguous regions ---
    /// [`ReductionSpec`] is the one here --- are only expressible when the axes
    /// they act on are adjacent.
    #[must_use]
    pub const fn is_contiguous_run(self) -> bool {
        if self.0 == 0 {
            return true;
        }
        let lowest = self.0.trailing_zeros();
        let past_highest = u64::BITS - self.0.leading_zeros();
        self.0.count_ones() == past_highest - lowest
    }
}

/// Dynamic/arbitrary rank descriptor axis set.
///
/// The representation is deliberately private: callers use this semantic
/// collection, while the <=64-bit mask remains an implementation detail.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct AxisSet(AxisSetRepr);

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
enum AxisSetRepr {
    Empty,
    Inline(AxisMask),
    Spilled(alloc::vec::Vec<usize>),
}

impl Default for AxisSet {
    fn default() -> Self {
        Self::EMPTY
    }
}

pub trait IntoAxisSet {
    fn into_axis_set(self) -> AxisSet;
}

impl IntoAxisSet for AxisSet {
    fn into_axis_set(self) -> AxisSet {
        self
    }
}

impl IntoAxisSet for AxisMask {
    fn into_axis_set(self) -> AxisSet {
        self.axes()
            .fold(AxisSet::EMPTY, |set, axis| set.insert(axis))
    }
}

impl AxisSet {
    pub const EMPTY: Self = Self(AxisSetRepr::Empty);

    pub fn contains(&self, axis: usize) -> bool {
        match &self.0 {
            AxisSetRepr::Empty => false,
            AxisSetRepr::Inline(mask) => mask.contains(axis),
            AxisSetRepr::Spilled(axes) => axes.contains(&axis),
        }
    }

    pub fn count(&self) -> usize {
        match &self.0 {
            AxisSetRepr::Empty => 0,
            AxisSetRepr::Inline(mask) => mask.count(),
            AxisSetRepr::Spilled(axes) => axes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.0 {
            AxisSetRepr::Empty => true,
            AxisSetRepr::Inline(mask) => mask.is_empty(),
            AxisSetRepr::Spilled(axes) => axes.is_empty(),
        }
    }

    pub fn axes(&self) -> alloc::vec::IntoIter<usize> {
        match &self.0 {
            AxisSetRepr::Empty => alloc::vec::Vec::new().into_iter(),
            AxisSetRepr::Inline(mask) => mask.axes().collect::<alloc::vec::Vec<_>>().into_iter(),
            AxisSetRepr::Spilled(axes) => axes.clone().into_iter(),
        }
    }

    pub fn all_below(rank: usize) -> Self {
        (0..rank).fold(Self::EMPTY, |set, axis| set.insert(axis))
    }

    pub fn is_contiguous_run(&self) -> bool {
        let axes: alloc::vec::Vec<_> = self.axes().collect();
        axes.windows(2).all(|pair| pair[1] == pair[0] + 1)
    }

    pub fn try_from_axes(
        operation: OperationKind,
        rank: usize,
        axes: impl IntoIterator<Item = usize>,
    ) -> Result<Self, ShapeError> {
        let mut set = Self::EMPTY;
        for axis in axes {
            if axis >= rank || set.contains(axis) {
                return Err(ShapeError::InvalidParameter {
                    operation,
                    parameter: "axis",
                    value: axis,
                });
            }
            set = set.insert(axis);
        }
        Ok(set)
    }

    pub fn insert(self, axis: usize) -> Self {
        match self.0 {
            AxisSetRepr::Empty => {
                if axis < AxisMask::MAX_AXES {
                    Self(AxisSetRepr::Inline(AxisMask::EMPTY.insert(axis).unwrap()))
                } else {
                    Self(AxisSetRepr::Spilled(alloc::vec![axis]))
                }
            }
            AxisSetRepr::Inline(mask) => {
                if axis < AxisMask::MAX_AXES {
                    Self(AxisSetRepr::Inline(mask.insert(axis).unwrap()))
                } else {
                    let mut axes: alloc::vec::Vec<usize> = (0..AxisMask::MAX_AXES)
                        .filter(|&a| mask.contains(a))
                        .collect();
                    axes.push(axis);
                    axes.sort_unstable();
                    Self(AxisSetRepr::Spilled(axes))
                }
            }
            AxisSetRepr::Spilled(mut axes) => {
                if !axes.contains(&axis) {
                    axes.push(axis);
                    axes.sort_unstable();
                }
                Self(AxisSetRepr::Spilled(axes))
            }
        }
    }
}

impl BitOr for AxisMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitAnd for AxisMask {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitOrAssign for AxisMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAndAssign for AxisMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl fmt::Debug for AxisMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AxisMask")?;
        f.debug_set().entries(self.axes()).finish()
    }
}

// --- the taxonomy -----------------------------------------------------------

/// What every operation descriptor reports about itself.
///
/// This trait *is* the operation taxonomy: it binds each descriptor to the one
/// [`OperationKind`] it resolves and the schema its fields are frozen at, so
/// generic execution code (`Execute<O>`, `EXE-006`) can key caches, route
/// capability queries, and report errors without matching on descriptor types.
///
/// Sealed against outside implementations: a descriptor is the contract between
/// a shape proof and native execution, so defining one belongs next to the shape
/// rule that produces it. Backend authors consume descriptors; they never need
/// to add one, and one defined elsewhere would carry no proof of validation.
pub trait OperationSpec: Clone + fmt::Debug + Sealed {
    /// The operation this descriptor resolves.
    const KIND: OperationKind;

    /// Exact semantic identity for errors, capability lookup, and capture.
    /// Attribute-polymorphic geometry descriptors override this method.
    #[must_use]
    fn operation(&self) -> OperationKind {
        Self::KIND
    }

    /// The schema this descriptor's fields are frozen at.
    ///
    /// Defaulted to [`DescriptorSchemaVersion::CURRENT`] because descriptors
    /// normally move together. A descriptor may override it to version
    /// independently once one of them has a reason to.
    const SCHEMA: DescriptorSchemaVersion = DescriptorSchemaVersion::CURRENT;

    /// The operation's output dimensions.
    fn output(&self) -> &ShapeBuf;

    /// How many elements the operation writes.
    ///
    /// Every constructor in this module already rejects an output whose element
    /// count overflows `usize`, so this cannot fail for a descriptor built
    /// through one. It stays fallible because that is a property of the
    /// constructors rather than of the trait, and a caller holding a `&dyn`
    /// descriptor has no way to know which constructor produced it.
    fn output_elements(&self) -> Result<usize, ShapeError> {
        self.output().checked_numel(self.operation())
    }
}

/// Descriptor types accepted by the execution contract.
///
/// Legacy geometry descriptors implement this through `OperationSpec`; exact
/// catalog descriptors implement it directly. The validation wrapper remains
/// the seal, so backend authors consume but cannot mint invocations.
pub trait ExecutionDescriptor: Clone + fmt::Debug {
    /// Return the validated output shape when the descriptor carries one.
    ///
    /// Shape-only executors use this metadata to model storage without
    /// inventing a second shape calculation. Descriptors whose output is not
    /// represented by a logical tensor leave the default as `None`.
    fn output_shape(&self) -> Option<&ShapeBuf> {
        None
    }
}

impl<O: OperationSpec> ExecutionDescriptor for O {
    fn output_shape(&self) -> Option<&ShapeBuf> {
        Some(self.output())
    }
}

/// Reject an output that cannot be allocated or indexed.
///
/// A descriptor whose element count overflows `usize` describes work no backend
/// can perform, and the useful place to say so is where the operands are still
/// in hand. Note that an *empty* output is fine: a zero dimension collapses the
/// product to 0 rather than overflowing, so a legitimately empty tensor passes.
fn check_output(operation: OperationKind, output: &ShapeBuf) -> Result<(), ShapeError> {
    output.checked_numel(operation).map(|_| ())
}

/// Reject a rank that cannot be addressed (no-op in rank-independent model).
fn check_rank_ceiling(_operation: OperationKind, _rank: usize) -> Result<(), ShapeError> {
    Ok(())
}

/// The checked product of a run of dimensions.
///
/// Routed through [`ShapeBuf::checked_numel`] rather than a local fold so that
/// the empty-tensor and overflow rules stay in one place --- a zero dimension
/// collapses the product regardless of axis order.
fn extent(operation: OperationKind, dims: &[usize]) -> Result<usize, ShapeError> {
    ShapeBuf::from_slice(dims).checked_numel(operation)
}

// --- broadcast --------------------------------------------------------------

/// Elementwise iteration geometry over two operands.
///
/// Both operands are normalized to the output rank, with a zero stride at every
/// axis they are stretched along. That is the whole point: a kernel walks the
/// output shape and indexes both operands with the same loop, because a
/// broadcast axis contributes `coordinate * 0` to the offset. No inner-loop
/// branch on which operand was smaller.
///
/// This descriptor also serves binary [`Pointwise`](OperationKind::Pointwise)
/// operations, which have no shape rule of their own beyond broadcasting. It
/// replaces the `IterationPlan` and `OperandIteration` pair in
/// `incin-backends/src/iteration.rs`, whose strides are `Vec<usize>` built per
/// call and whose errors are formatted strings.
///
/// Storage offsets are absent on purpose: they are per-tensor facts that belong
/// to `TensorMeta` (`EXE-004`), and leaving them out is what makes one
/// descriptor reusable across calls on different views of the same shapes.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BroadcastSpec {
    /// The result dimensions.
    pub output: ShapeBuf,
    /// Left operand strides, normalized to the output rank.
    pub lhs_strides: StrideBuf,
    /// Right operand strides, normalized to the output rank.
    pub rhs_strides: StrideBuf,
    /// Axes the left operand is stretched along.
    pub lhs_broadcast_mask: AxisSet,
    /// Axes the right operand is stretched along.
    pub rhs_broadcast_mask: AxisSet,
    /// What the kernel folds the paired elements through, if anything.
    ///
    /// `None` is a stretch: the geometry is resolved and the output is the left
    /// operand read through the broadcast strides. See [`BinaryOp`] for why this
    /// is the only descriptor whose operator is optional.
    pub op: Option<BinaryOp>,
}

impl BroadcastSpec {
    /// The operation this descriptor resolves. Named apart from the
    /// [`OperationSpec::KIND`] it feeds so the two are never ambiguous.
    const OP: OperationKind = OperationKind::Broadcast;

    /// Resolve two operand layouts into a shared iteration geometry.
    ///
    /// Each operand is right-aligned against the output, as broadcasting
    /// requires: trailing axes line up and missing leading axes are treated as
    /// length 1.
    ///
    /// The output shape and both masks are *derived* from the arguments, so a
    /// descriptor cannot claim a shape its operands do not produce.
    pub fn new(
        lhs: &ShapeBuf,
        lhs_strides: &StrideBuf,
        rhs: &ShapeBuf,
        rhs_strides: &StrideBuf,
        op: Option<BinaryOp>,
    ) -> Result<Self, ShapeError> {
        check_operand(Self::OP, "lhs", lhs, lhs_strides)?;
        check_operand(Self::OP, "rhs", rhs, rhs_strides)?;

        let output = Self::resolve_shape(lhs, rhs)?;
        check_output(Self::OP, &output)?;
        let (lhs_strides, lhs_broadcast_mask) = align_operand(&output, lhs, lhs_strides)?;
        let (rhs_strides, rhs_broadcast_mask) = align_operand(&output, rhs, rhs_strides)?;

        Ok(Self {
            output,
            lhs_strides,
            rhs_strides,
            lhs_broadcast_mask,
            rhs_broadcast_mask,
            op,
        })
    }

    /// Resolve two dense row-major operands.
    ///
    /// The common case, and the one worth not making callers spell: both
    /// operands are contiguous, so their strides follow from their shapes.
    pub fn contiguous(
        lhs: &ShapeBuf,
        rhs: &ShapeBuf,
        op: Option<BinaryOp>,
    ) -> Result<Self, ShapeError> {
        let lhs_strides = StrideBuf::contiguous_for(lhs, Self::OP)?;
        let rhs_strides = StrideBuf::contiguous_for(rhs, Self::OP)?;
        Self::new(lhs, &lhs_strides, rhs, &rhs_strides, op)
    }

    /// The broadcast result of two shapes, without building a descriptor.
    ///
    /// Exposed because shape resolution is useful on its own --- a caller may
    /// want the output dimensions to allocate before it has strides to lower.
    pub fn resolve_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> Result<ShapeBuf, ShapeError> {
        let rank = lhs.rank().max(rhs.rank());
        check_rank_ceiling(Self::OP, rank)?;

        let mut dims = ShapeBuf::default();
        for axis in 0..rank {
            let left = aligned_dim(lhs, rank, axis);
            let right = aligned_dim(rhs, rank, axis);
            let resolved = match (left, right) {
                (l, r) if l == r => l,
                (1, r) => r,
                (l, 1) => l,
                (l, r) => {
                    return Err(ShapeError::DimensionMismatch {
                        operation: Self::OP,
                        axis: Axis::Index(axis),
                        lhs: l,
                        rhs: r,
                        constraint: DimensionConstraint::Broadcastable,
                    });
                }
            };
            dims.push(resolved);
        }
        Ok(dims)
    }
}

impl Sealed for BroadcastSpec {}

impl OperationSpec for BroadcastSpec {
    const KIND: OperationKind = Self::OP;

    fn operation(&self) -> OperationKind {
        match self.op {
            None => OperationKind::BroadcastAs,
            Some(BinaryOp::Add) => OperationKind::Add,
            Some(BinaryOp::Sub) => OperationKind::Sub,
            Some(BinaryOp::Mul) => OperationKind::Mul,
            Some(BinaryOp::Div) => OperationKind::Div,
        }
    }

    fn output(&self) -> &ShapeBuf {
        &self.output
    }
}

/// The dimension an operand contributes at an output axis, once right-aligned.
///
/// Leading axes the operand does not have are length 1, which is exactly the
/// broadcasting rule.
fn aligned_dim(operand: &ShapeBuf, rank: usize, axis: usize) -> usize {
    let offset = rank - operand.rank();
    if axis < offset {
        1
    } else {
        operand.dims()[axis - offset]
    }
}

/// Reject an operand whose stride count does not match its rank.
fn check_operand(
    operation: OperationKind,
    name: &'static str,
    shape: &ShapeBuf,
    strides: &StrideBuf,
) -> Result<(), ShapeError> {
    check_rank_ceiling(operation, shape.rank())?;
    if strides.len() != shape.rank() {
        return Err(ShapeError::RankMismatch {
            operation,
            expected: RankExpectation::SameAs {
                operand: name,
                rank: shape.rank(),
            },
            actual: strides.len(),
        });
    }
    Ok(())
}

/// Normalize one operand to the output rank, reporting the axes it is stretched
/// along.
///
/// An axis is a broadcast axis when the operand is length 1 there and the
/// output is not. Its stride becomes 0, so the kernel's coordinate arithmetic
/// keeps returning the same element.
fn align_operand(
    output: &ShapeBuf,
    operand: &ShapeBuf,
    strides: &StrideBuf,
) -> Result<(StrideBuf, AxisSet), ShapeError> {
    let rank = output.rank();
    let offset = rank - operand.rank();
    let mut aligned = StrideBuf::default();
    let mut mask = AxisSet::EMPTY;

    for axis in 0..rank {
        let out_dim = output.dims()[axis];
        if axis < offset {
            // An axis the operand does not have at all: length 1 by the
            // alignment rule, and therefore stretched unless the output is 1 too.
            aligned.push(0);
            if out_dim != 1 {
                mask = mask.insert(axis);
            }
            continue;
        }

        let dim = operand.dims()[axis - offset];
        if dim == out_dim {
            aligned.push(strides.strides()[axis - offset]);
        } else if dim == 1 {
            aligned.push(0);
            mask = mask.insert(axis);
        } else {
            return Err(ShapeError::DimensionMismatch {
                operation: OperationKind::Broadcast,
                axis: Axis::Index(axis),
                lhs: dim,
                rhs: out_dim,
                constraint: DimensionConstraint::Broadcastable,
            });
        }
    }

    Ok((aligned, mask))
}

// --- matrix multiplication --------------------------------------------------

/// Batched matrix multiplication launch parameters.
///
/// `M`, `N`, `K` and the per-axis batch strides are precisely the arguments a
/// batched GEMM takes, and every one of them follows from the shape rule the
/// frontend has already discharged. Resolving them here is what stops each
/// backend from re-deriving them from raw dimension vectors.
///
/// Batch axes broadcast, as they do for elementwise operations, and a broadcast
/// batch axis gets a stride of 0 --- the same convention [`BroadcastSpec`] uses.
/// A GEMM that reuses one operand across the batch therefore needs no separate
/// code path.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatMulSpec {
    /// Rank of the left operand, matrix axes included.
    pub lhs_rank: usize,
    /// Rank of the right operand, matrix axes included.
    pub rhs_rank: usize,
    /// The result dimensions: the broadcast batch shape followed by `m` and `n`.
    pub output: ShapeBuf,
    /// The broadcast batch dimensions, without the two matrix axes.
    pub batch: ShapeBuf,
    /// Rows of the left operand and of the result.
    pub m: usize,
    /// Columns of the right operand and of the result.
    pub n: usize,
    /// The contracted extent, shared by both operands.
    pub k: usize,
    /// Elements to advance in the left operand per step along each batch axis.
    pub lhs_batch_strides: StrideBuf,
    /// Elements to advance in the right operand per step along each batch axis.
    pub rhs_batch_strides: StrideBuf,
    /// Elements to advance in the result per step along each batch axis.
    pub output_batch_strides: StrideBuf,
    /// Whether the backend should read the left operand transposed.
    pub transpose_lhs: bool,
    /// Whether the backend should read the right operand transposed.
    pub transpose_rhs: bool,
}

impl MatMulSpec {
    /// See [`BroadcastSpec::OP`].
    const OP: OperationKind = OperationKind::MatMul;

    /// Resolve two operand shapes into GEMM parameters.
    ///
    /// Both operands must have rank 2 or more. The last two axes are the matrix;
    /// everything before them is a batch axis and broadcasts.
    pub fn new(lhs: &ShapeBuf, rhs: &ShapeBuf) -> Result<Self, ShapeError> {
        let (lhs_rank, rhs_rank) = (lhs.rank(), rhs.rank());
        for rank in [lhs_rank, rhs_rank] {
            if rank < 2 {
                return Err(ShapeError::RankMismatch {
                    operation: Self::OP,
                    expected: RankExpectation::AtLeast(2),
                    actual: rank,
                });
            }
            check_rank_ceiling(Self::OP, rank)?;
        }

        let m = lhs.dims()[lhs_rank - 2];
        let k = lhs.dims()[lhs_rank - 1];
        let rhs_k = rhs.dims()[rhs_rank - 2];
        let n = rhs.dims()[rhs_rank - 1];
        if k != rhs_k {
            return Err(ShapeError::DimensionMismatch {
                operation: Self::OP,
                axis: Axis::Named("contraction"),
                lhs: k,
                rhs: rhs_k,
                constraint: DimensionConstraint::Equal,
            });
        }

        let lhs_batch = ShapeBuf::from_slice(&lhs.dims()[..lhs_rank - 2]);
        let rhs_batch = ShapeBuf::from_slice(&rhs.dims()[..rhs_rank - 2]);
        let batch = BroadcastSpec::resolve_shape(&lhs_batch, &rhs_batch).map_err(retag)?;
        check_rank_ceiling(Self::OP, batch.rank() + 2)?;

        let mut output = batch.clone();
        output.push(m);
        output.push(n);
        check_output(Self::OP, &output)?;

        // A batch step moves past one whole matrix, so the unit of every batch
        // stride is that matrix's element count, not one element.
        let lhs_batch_strides = batch_strides(&batch, &lhs_batch, checked_area(m, k)?)?;
        let rhs_batch_strides = batch_strides(&batch, &rhs_batch, checked_area(k, n)?)?;
        let output_batch_strides = batch_strides(&batch, &batch, checked_area(m, n)?)?;

        Ok(Self {
            lhs_rank,
            rhs_rank,
            output,
            batch,
            m,
            n,
            k,
            lhs_batch_strides,
            rhs_batch_strides,
            output_batch_strides,
            transpose_lhs: false,
            transpose_rhs: false,
        })
    }

    /// Mark either operand as stored transposed.
    ///
    /// These are layout facts, not shape facts: the descriptor's `m`, `n`, `k`,
    /// and output are unchanged, and the flags only tell the backend how to read
    /// storage it has been handed. Lowering (`EXE-003`) sets them from operand
    /// strides, which is why they are a separate step rather than a constructor
    /// argument --- [`new`](Self::new) sees shapes, not layouts.
    #[must_use]
    pub fn transposed(mut self, lhs: bool, rhs: bool) -> Self {
        self.transpose_lhs = lhs;
        self.transpose_rhs = rhs;
        self
    }
}

impl Sealed for MatMulSpec {}

impl OperationSpec for MatMulSpec {
    const KIND: OperationKind = Self::OP;

    fn operation(&self) -> OperationKind {
        OperationKind::MatMulExact
    }

    fn output(&self) -> &ShapeBuf {
        &self.output
    }
}

/// Re-attribute a broadcast diagnostic raised while resolving matmul batch axes.
///
/// The batch axes broadcast, but the caller asked for a matrix multiplication,
/// and an error that says "broadcast" sends them looking at the wrong operation.
fn retag(error: ShapeError) -> ShapeError {
    match error {
        ShapeError::DimensionMismatch {
            axis,
            lhs,
            rhs,
            constraint,
            ..
        } => ShapeError::DimensionMismatch {
            operation: MatMulSpec::OP,
            axis,
            lhs,
            rhs,
            constraint,
        },
        ShapeError::RankMismatch {
            expected, actual, ..
        } => ShapeError::RankMismatch {
            operation: MatMulSpec::OP,
            expected,
            actual,
        },
        other => other,
    }
}

/// The element count of one `rows * columns` matrix.
fn checked_area(rows: usize, columns: usize) -> Result<usize, ShapeError> {
    extent(MatMulSpec::OP, &[rows, columns])
}

/// Per-batch-axis element strides for one operand of a batched matmul.
///
/// `operand` is right-aligned against the broadcast `batch` shape. An axis the
/// operand is length 1 along while the batch is not gets stride 0, so every
/// batch index reads the same matrix.
fn batch_strides(
    batch: &ShapeBuf,
    operand: &ShapeBuf,
    matrix_elements: usize,
) -> Result<StrideBuf, ShapeError> {
    let rank = batch.rank();
    let offset = rank - operand.rank();
    let mut strides = StrideBuf::default();

    for axis in 0..rank {
        if axis < offset || operand.dims()[axis - offset] == 1 && batch.dims()[axis] != 1 {
            strides.push(0);
            continue;
        }
        // Row-major over the operand's own batch axes, measured in matrices.
        let trailing = extent(MatMulSpec::OP, &operand.dims()[axis - offset + 1..])?;
        strides.push(trailing.checked_mul(matrix_elements).ok_or(
            ShapeError::ArithmeticOverflow {
                operation: MatMulSpec::OP,
                expression: "batch stride * matrix element count",
            },
        )?);
    }

    Ok(strides)
}

// --- reduction --------------------------------------------------------------

/// A reduction collapsed into three regions.
///
/// Every reduction over adjacent axes is the same loop nest: `outer`
/// independent slices, each reducing `reduced` elements into `inner`
/// accumulators. Expressing it this way is what lets one kernel serve
/// `sum(dim = 0)` on a matrix and `mean(dim = 2)` on a rank-5 tensor --- the ranks
/// differ, the three extents do not.
///
/// The axes must form one unbroken run. A scattered set such as `{0, 2}` has no
/// such decomposition without first permuting the tensor, so it is rejected here
/// rather than mis-lowered; a later task may lower it as a transpose followed by
/// a contiguous reduction.
///
/// The three extents are the loop; [`op`](Self::op) is what runs inside it. Both
/// halves are needed to execute, and neither determines the other --- which is why
/// the operator is a field the constructors take rather than one they derive.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReductionSpec {
    /// The result dimensions.
    pub output: ShapeBuf,
    /// The accumulation applied across the collapsed axes.
    pub op: ReduceOp,
    /// The axes being collapsed.
    pub axes: AxisSet,
    /// Elements in the region outside the reduced axes.
    pub outer: usize,
    /// Elements collapsed into each result element.
    pub reduced: usize,
    /// Elements in the region inside the reduced axes.
    pub inner: usize,
    /// Whether reduced axes stay in the output as length 1.
    pub keep_dims: bool,
}

impl ReductionSpec {
    /// See [`BroadcastSpec::OP`].
    const OP: OperationKind = OperationKind::Reduction;

    /// Resolve an input shape and a set of axes into a reduction geometry.
    ///
    /// An empty axis set is legal and reduces nothing: `reduced` is 1 and the
    /// output equals the input. Keeping that case total means callers that build
    /// an axis list dynamically do not need a special case for "no axes given".
    pub fn new(
        input: &ShapeBuf,
        axes: impl IntoAxisSet,
        keep_dims: bool,
        op: ReduceOp,
    ) -> Result<Self, ShapeError> {
        let axes = axes.into_axis_set();
        let rank = input.rank();
        check_rank_ceiling(Self::OP, rank)?;

        for axis in axes.axes() {
            if axis >= rank {
                return Err(ShapeError::InvalidParameter {
                    operation: Self::OP,
                    parameter: "axis",
                    value: axis,
                });
            }
        }
        if !axes.is_contiguous_run() {
            // The run is described by its endpoints, which is what an axis-range
            // diagnostic already says.
            let listed: alloc::vec::Vec<_> = axes.axes().collect();
            let start = listed.first().copied().unwrap_or(0);
            let end = listed.last().map_or(0, |last| last + 1);
            return Err(ShapeError::InvalidAxisRange {
                operation: Self::OP,
                start,
                end,
                rank,
            });
        }

        let listed: alloc::vec::Vec<_> = axes.axes().collect();
        let (start, end) = match (listed.first(), listed.last()) {
            (Some(first), Some(last)) => (*first, *last + 1),
            // No axes: an empty run placed at the end, so `outer` is everything.
            _ => (rank, rank),
        };

        let outer = extent(Self::OP, &input.dims()[..start])?;
        let reduced = extent(Self::OP, &input.dims()[start..end])?;
        let inner = extent(Self::OP, &input.dims()[end..])?;

        let output = input
            .dims()
            .iter()
            .enumerate()
            .filter_map(|(axis, &dim)| match (axes.contains(axis), keep_dims) {
                (true, true) => Some(1),
                (true, false) => None,
                (false, _) => Some(dim),
            })
            .collect();
        check_output(Self::OP, &output)?;

        Ok(Self {
            output,
            op,
            axes,
            outer,
            reduced,
            inner,
            keep_dims,
        })
    }

    /// Resolve a reduction over listed axes.
    ///
    /// Rejects an out-of-range or repeated axis before any geometry is computed.
    pub fn over_axes(
        input: &ShapeBuf,
        axes: impl IntoIterator<Item = usize>,
        keep_dims: bool,
        op: ReduceOp,
    ) -> Result<Self, ShapeError> {
        let mask = AxisSet::try_from_axes(Self::OP, input.rank(), axes)?;
        Self::new(input, mask, keep_dims, op)
    }

    /// Resolve a reduction over every axis, producing a scalar.
    pub fn over_all(input: &ShapeBuf, keep_dims: bool, op: ReduceOp) -> Result<Self, ShapeError> {
        let mask = AxisSet::all_below(input.rank());
        Self::new(input, mask, keep_dims, op)
    }
}

impl Sealed for ReductionSpec {}

impl OperationSpec for ReductionSpec {
    const KIND: OperationKind = Self::OP;

    fn operation(&self) -> OperationKind {
        let input_rank = if self.keep_dims {
            self.output.rank()
        } else {
            self.output.rank() + self.axes.count()
        };
        let all = self.axes.count() == input_rank;
        match (self.op, all, self.keep_dims) {
            (ReduceOp::Sum, true, _) => OperationKind::SumAll,
            (ReduceOp::Mean, true, _) => OperationKind::MeanAll,
            (ReduceOp::Max, true, _) => OperationKind::MaxAll,
            (ReduceOp::Min, true, _) => OperationKind::MinAll,
            (ReduceOp::Prod, true, _) => OperationKind::ProdAll,
            (ReduceOp::Sum, false, false) => OperationKind::SumDim,
            (ReduceOp::Sum, false, true) => OperationKind::SumKeepDim,
            (ReduceOp::Mean, false, false) => OperationKind::MeanDim,
            (ReduceOp::Mean, false, true) => OperationKind::MeanKeepDim,
            (ReduceOp::Max, false, false) => OperationKind::MaxDim,
            (ReduceOp::Max, false, true) => OperationKind::MaxKeepDim,
            (ReduceOp::Min, false, false) => OperationKind::MinDim,
            (ReduceOp::Min, false, true) => OperationKind::MinKeepDim,
            (ReduceOp::Prod, false, _) => OperationKind::ProdDim,
        }
    }

    fn output(&self) -> &ShapeBuf {
        &self.output
    }
}

// --- two-dimensional convolution --------------------------------------------

/// Two-dimensional convolution geometry.
///
/// Every extent a convolution kernel needs, with the spatial output sizes
/// resolved through [`spatial_out_size`] rather than recomputed. That shared
/// call is the point: the output-size formula has four ways to fail, and
/// `SHP-005` already spells each of them out as a distinct diagnostic. A second
/// copy of the formula here would be a second place for them to drift.
///
/// Input and output are `NCHW`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conv2dSpec {
    /// The result dimensions, `[n, c_out, h_out, w_out]`.
    pub output: ShapeBuf,
    /// Batch size.
    pub n: usize,
    /// Input channels.
    pub c_in: usize,
    /// Output channels.
    pub c_out: usize,
    /// Input height.
    pub h_in: usize,
    /// Input width.
    pub w_in: usize,
    /// Output height.
    pub h_out: usize,
    /// Output width.
    pub w_out: usize,
    /// Kernel extent, `[height, width]`.
    pub kernel: [usize; 2],
    /// Stride, `[height, width]`.
    pub stride: [usize; 2],
    /// Zero padding applied to both sides of each axis, `[height, width]`.
    pub padding: [usize; 2],
    /// Dilation, `[height, width]`.
    pub dilation: [usize; 2],
    /// Number of channel groups.
    pub groups: usize,
}

impl Conv2dSpec {
    /// See [`BroadcastSpec::OP`].
    const OP: OperationKind = OperationKind::Conv2d;

    /// Resolve an `NCHW` input and convolution parameters into a geometry.
    ///
    /// Rejects a `groups` of 0, an input or output channel count that `groups`
    /// does not divide, a stride, kernel, or dilation of 0, and a kernel that
    /// does not fit its padded input.
    pub fn new(
        input: &ShapeBuf,
        c_out: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self, ShapeError> {
        if input.rank() != 4 {
            return Err(ShapeError::RankMismatch {
                operation: Self::OP,
                expected: RankExpectation::Exactly(4),
                actual: input.rank(),
            });
        }
        let [n, c_in, h_in, w_in] = [
            input.dims()[0],
            input.dims()[1],
            input.dims()[2],
            input.dims()[3],
        ];

        if groups == 0 {
            return Err(ShapeError::InvalidParameter {
                operation: Self::OP,
                parameter: "groups",
                value: groups,
            });
        }
        for (name, channels) in [("in_channels", c_in), ("out_channels", c_out)] {
            if channels % groups != 0 {
                return Err(ShapeError::DimensionMismatch {
                    operation: Self::OP,
                    axis: Axis::Named(name),
                    lhs: channels,
                    rhs: groups,
                    constraint: DimensionConstraint::DivisibleBy,
                });
            }
        }

        let h_out = spatial_out_size(
            Self::OP,
            Axis::Named("height"),
            h_in,
            kernel[0],
            stride[0],
            padding[0],
            dilation[0],
        )?;
        let w_out = spatial_out_size(
            Self::OP,
            Axis::Named("width"),
            w_in,
            kernel[1],
            stride[1],
            padding[1],
            dilation[1],
        )?;

        let output = ShapeBuf::from_slice(&[n, c_out, h_out, w_out]);
        check_output(Self::OP, &output)?;

        Ok(Self {
            output,
            n,
            c_in,
            c_out,
            h_in,
            w_in,
            h_out,
            w_out,
            kernel,
            stride,
            padding,
            dilation,
            groups,
        })
    }
}

impl Sealed for Conv2dSpec {}

impl OperationSpec for Conv2dSpec {
    const KIND: OperationKind = Self::OP;

    fn operation(&self) -> OperationKind {
        OperationKind::Conv2dExact
    }

    fn output(&self) -> &ShapeBuf {
        &self.output
    }
}

// --- two-dimensional pooling ------------------------------------------------

/// Two-dimensional pooling geometry.
///
/// The sliding window is the same one [`Conv2dSpec`] describes, and the output
/// sizes come from the same [`spatial_out_size`]. What differs is the channel
/// axis: a convolution replaces it, a pool passes it through, so there is no
/// `c_out` and no `groups` here. Sharing [`Conv2dSpec`] with `c_out = c_in` and
/// `groups = c_in` would express the same geometry, but it would also report
/// [`OperationKind::Conv2d`], and a capability query or a kernel cache keyed on
/// that would answer for the wrong operation.
///
/// Which reduction runs inside the window is not geometry, but a backend still
/// cannot execute a pool without it, so [`op`](Self::op) names it. It is the one
/// field here that is not derived from the input shape, and it is what
/// distinguishes this descriptor from the window it shares with [`Conv2dSpec`].
///
/// Input and output are `NCHW`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pool2dSpec {
    /// The result dimensions, `[n, channels, h_out, w_out]`.
    pub output: ShapeBuf,
    /// The accumulation applied inside each window.
    pub op: PoolOp,
    /// Batch size.
    pub n: usize,
    /// Channels, unchanged from input to output.
    pub channels: usize,
    /// Input height.
    pub h_in: usize,
    /// Input width.
    pub w_in: usize,
    /// Output height.
    pub h_out: usize,
    /// Output width.
    pub w_out: usize,
    /// Window extent, `[height, width]`.
    pub kernel: [usize; 2],
    /// Stride, `[height, width]`.
    pub stride: [usize; 2],
    /// Zero padding applied to both sides of each axis, `[height, width]`.
    pub padding: [usize; 2],
    /// Dilation, `[height, width]`.
    pub dilation: [usize; 2],
}

impl Pool2dSpec {
    /// See [`BroadcastSpec::OP`].
    const OP: OperationKind = OperationKind::Pool2d;

    /// Resolve an `NCHW` input and window parameters into a geometry.
    ///
    /// Rejects a stride, window, or dilation of 0, and a window that does not
    /// fit its padded input.
    pub fn new(
        input: &ShapeBuf,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        op: PoolOp,
    ) -> Result<Self, ShapeError> {
        if input.rank() != 4 {
            return Err(ShapeError::RankMismatch {
                operation: Self::OP,
                expected: RankExpectation::Exactly(4),
                actual: input.rank(),
            });
        }
        let [n, channels, h_in, w_in] = [
            input.dims()[0],
            input.dims()[1],
            input.dims()[2],
            input.dims()[3],
        ];

        let h_out = spatial_out_size(
            Self::OP,
            Axis::Named("height"),
            h_in,
            kernel[0],
            stride[0],
            padding[0],
            dilation[0],
        )?;
        let w_out = spatial_out_size(
            Self::OP,
            Axis::Named("width"),
            w_in,
            kernel[1],
            stride[1],
            padding[1],
            dilation[1],
        )?;

        let output = ShapeBuf::from_slice(&[n, channels, h_out, w_out]);
        check_output(Self::OP, &output)?;

        Ok(Self {
            output,
            op,
            n,
            channels,
            h_in,
            w_in,
            h_out,
            w_out,
            kernel,
            stride,
            padding,
            dilation,
        })
    }
}

impl Sealed for Pool2dSpec {}

impl OperationSpec for Pool2dSpec {
    const KIND: OperationKind = Self::OP;

    fn operation(&self) -> OperationKind {
        match self.op {
            PoolOp::Max => OperationKind::MaxPool2d,
            PoolOp::Average => OperationKind::AvgPool2d,
        }
    }

    fn output(&self) -> &ShapeBuf {
        &self.output
    }
}

// --- reshape ----------------------------------------------------------------

/// A reinterpretation of one shape as another with the same element count.
///
/// Reshape is the one descriptor here whose output is an *operand* rather than
/// a derivation: the caller chooses the target shape, and no rule can compute
/// it from the input. What the constructor derives is the thing that makes the
/// pair legal --- the element count --- and it refuses any pair where the two
/// counts differ. The rest of the "derived, never asserted" property holds
/// unchanged, because [`elements`](Self::elements) is never an argument.
///
/// Keeping the input shape alongside the output is what lets a backend decide
/// between re-addressing storage and copying it. That decision also needs the
/// input's strides, which are a per-tensor fact belonging to `TensorMeta`
/// (`EXE-004`), so it is not made here.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReshapeSpec {
    /// The dimensions being reinterpreted.
    pub input: ShapeBuf,
    /// The dimensions they are reinterpreted as.
    pub output: ShapeBuf,
    /// The element count both shapes share.
    pub elements: usize,
}

impl ReshapeSpec {
    /// See [`BroadcastSpec::OP`].
    const OP: OperationKind = OperationKind::Reshape;

    /// Resolve an input shape and a target shape into a reinterpretation.
    ///
    /// Rejects a target whose element count differs from the input's, and
    /// either shape whose count overflows `usize`.
    pub fn new(input: &ShapeBuf, output: &ShapeBuf) -> Result<Self, ShapeError> {
        check_rank_ceiling(Self::OP, input.rank())?;
        check_rank_ceiling(Self::OP, output.rank())?;

        let elements = input.checked_numel(Self::OP)?;
        let target = output.checked_numel(Self::OP)?;
        if elements != target {
            return Err(ShapeError::DimensionMismatch {
                operation: Self::OP,
                // The rule constrains the shapes as wholes; no single axis is
                // at fault, and naming one would send a reader to the wrong
                // place.
                axis: Axis::Whole,
                lhs: elements,
                rhs: target,
                constraint: DimensionConstraint::Equal,
            });
        }

        Ok(Self {
            input: input.clone(),
            output: output.clone(),
            elements,
        })
    }
}

impl Sealed for ReshapeSpec {}

impl OperationSpec for ReshapeSpec {
    const KIND: OperationKind = Self::OP;

    fn operation(&self) -> OperationKind {
        OperationKind::ReshapeExact
    }

    fn output(&self) -> &ShapeBuf {
        &self.output
    }
}

#[cfg(test)]
mod exact_identity_tests {
    use super::*;

    #[test]
    fn attribute_polymorphic_descriptors_report_exact_identities() {
        let matrix = ShapeBuf::from_slice(&[2, 2]);
        assert_eq!(
            BroadcastSpec::contiguous(&matrix, &matrix, Some(BinaryOp::Add))
                .unwrap()
                .operation(),
            OperationKind::Add
        );
        assert_eq!(
            BroadcastSpec::contiguous(&matrix, &matrix, None)
                .unwrap()
                .operation(),
            OperationKind::BroadcastAs
        );
        assert_eq!(
            ReductionSpec::over_all(&matrix, false, ReduceOp::Mean)
                .unwrap()
                .operation(),
            OperationKind::MeanAll
        );
        assert_eq!(
            ReductionSpec::over_axes(&matrix, [1], true, ReduceOp::Sum)
                .unwrap()
                .operation(),
            OperationKind::SumKeepDim
        );
        assert_eq!(
            Pool2dSpec::new(
                &ShapeBuf::from_slice(&[1, 1, 4, 4]),
                [2, 2],
                [2, 2],
                [0, 0],
                [1, 1],
                PoolOp::Average,
            )
            .unwrap()
            .operation(),
            OperationKind::AvgPool2d
        );
        assert_eq!(
            ReshapeSpec::new(&matrix, &ShapeBuf::from_slice(&[4]))
                .unwrap()
                .operation(),
            OperationKind::ReshapeExact
        );
    }
}
