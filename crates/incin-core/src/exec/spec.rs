//! Shared execution metadata.
//! Exact operation descriptors live in [`super::catalog`]. This module retains
//! shared axis storage, schema versioning, distributed reduction vocabulary,
//! and the neutral execution metadata trait.
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

use crate::shapes::buf::ShapeBuf;
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation, ShapeError};

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
    /// v2 added the operator identity to reduction and pooling geometry. A v1
    /// cache entry keyed on either records a window without saying what
    /// accumulated in it, so it cannot be replayed.
    ///
    /// v3 added optional operator identity to broadcast geometry. A v2 entry
    /// cannot say whether it described a stretch or the geometry of a binary
    /// operation, and the two produce different output storage from identical
    /// strides.
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

/// Which accumulation a distributed reduction performs.
///
/// The geometry of a reduction says how many elements collapse into each result
/// element. It does not say what happens to them, and a backend cannot execute
/// one without knowing: `sum` and `max` walk identical loops and compute
/// different answers. This is the part the descriptor names.
///
/// The set is closed at the five accumulations whose result has the shape
/// the exact reduction descriptor derives. `argmax` and `argmin` collapse the same axes but
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
    /// Descriptors whose geometry collapses a shape into contiguous regions are
    /// only expressible when the axes they act on are adjacent.
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

// --- execution contract ------------------------------------------------------

/// Descriptor accepted by the backend execution contract.
///
/// Exact operation descriptors implement this neutral metadata interface. The
/// trait intentionally contains no legacy operation taxonomy or geometry
/// representation.
pub trait ExecutionDescriptor: Clone + fmt::Debug {
    /// Return the validated output shape when the descriptor carries one.
    fn output_shape(&self) -> Option<&ShapeBuf> {
        None
    }
}
