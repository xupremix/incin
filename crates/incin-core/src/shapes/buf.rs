//! Checked, inline-first shape and stride buffers.
//!
//! `PROPOSALS.md` §1.2.1 asks for one runtime shape representation with two
//! properties the `Vec<usize>` pairs scattered through the backends do not
//! have:
//!
//! * **Checked arithmetic.** Element counts and byte lengths are computed with
//!   `checked_mul` and returned as `Result`. Today `contiguous_strides`
//!   (`incin-backends/src/cpu/stride.rs`) panics on overflow and
//!   `checked_numel` there reports it as a formatted string; both become a
//!   [`ShapeError`] naming the operation and the failing term.
//! * **No cached derived values.** A `ShapeBuf` stores dimensions and nothing
//!   else. There is no `numel` field that a later mutation could leave stale —
//!   the RFC calls this out explicitly, because a cached count that disagrees
//!   with its dimensions is exactly how an undersized allocation gets indexed
//!   with an oversized stride.
//!
//! Ranks up to [`INLINE_RANK`] are stored inline, so the overwhelmingly common
//! case allocates nothing. The spill to the heap is an implementation detail:
//! two buffers holding the same dimensions are equal whichever side of the
//! boundary they are on.

use alloc::vec::Vec;
use core::fmt;
use core::ops::Deref;

use super::error::{OperationKind, RankExpectation, ShapeError};

/// Ranks up to this bound are stored inline, without allocating.
///
/// This is a storage optimization only. It is deliberately independent of
/// framework representability, typed rank, and backend rank capability.
pub const INLINE_RANK: usize = 8;

/// A short sequence of `T` held inline until it outgrows [`INLINE_RANK`].
///
/// The inline/heap split is deliberately not observable through the public API
/// beyond [`is_inline`](Self::is_inline), which exists for tests that pin the
/// spill boundary. Equality, ordering, hashing, and iteration all go through
/// the slice, so a value that spills stays equal to one that did not.
#[derive(Clone)]
pub struct InlineOrHeap<T: Copy + Default> {
    repr: Repr<T>,
}

#[derive(Clone)]
enum Repr<T: Copy + Default> {
    Inline { len: usize, items: [T; INLINE_RANK] },
    Heap(Vec<T>),
}

impl InlineOrHeap<usize> {
    /// The empty buffer, usable in a `const` context.
    ///
    /// `new` cannot be `const` because `T::default()` is not a `const fn` in a
    /// generic context. The dimension and stride buffers are always `usize`, so
    /// they get a `const` empty value spelled out for that one element type.
    pub const EMPTY: Self = Self {
        repr: Repr::Inline {
            len: 0,
            items: [0; INLINE_RANK],
        },
    };
}

impl<T: Copy + Default> InlineOrHeap<T> {
    /// An empty buffer. Allocates nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            repr: Repr::Inline {
                len: 0,
                items: [T::default(); INLINE_RANK],
            },
        }
    }

    /// Copy `items` into a buffer, spilling to the heap only if it is longer
    /// than [`INLINE_RANK`].
    #[must_use]
    pub fn from_slice(items: &[T]) -> Self {
        if items.len() <= INLINE_RANK {
            let mut inline = [T::default(); INLINE_RANK];
            inline[..items.len()].copy_from_slice(items);
            Self {
                repr: Repr::Inline {
                    len: items.len(),
                    items: inline,
                },
            }
        } else {
            Self {
                repr: Repr::Heap(items.to_vec()),
            }
        }
    }

    /// The buffer's contents.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        match &self.repr {
            Repr::Inline { len, items } => &items[..*len],
            Repr::Heap(v) => v.as_slice(),
        }
    }

    /// The buffer's contents, mutably. The length cannot change through this
    /// handle, so no derived value can go stale behind it.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.repr {
            Repr::Inline { len, items } => &mut items[..*len],
            Repr::Heap(v) => v.as_mut_slice(),
        }
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the contents are currently stored inline.
    ///
    /// Exposed so tests can pin the spill boundary. Behavior must never depend
    /// on it.
    #[must_use]
    pub fn is_inline(&self) -> bool {
        matches!(self.repr, Repr::Inline { .. })
    }

    /// Append one element, spilling to the heap if the inline capacity is full.
    pub fn push(&mut self, value: T) {
        match &mut self.repr {
            Repr::Inline { len, items } if *len < INLINE_RANK => {
                items[*len] = value;
                *len += 1;
            }
            Repr::Inline { len, items } => {
                let mut heap = Vec::with_capacity(*len + 1);
                heap.extend_from_slice(&items[..*len]);
                heap.push(value);
                self.repr = Repr::Heap(heap);
            }
            Repr::Heap(v) => v.push(value),
        }
    }

    /// Remove and return the last element.
    ///
    /// A buffer that has spilled stays on the heap. Migrating back would make
    /// [`is_inline`](Self::is_inline) depend on history rather than length, and
    /// nothing observable depends on the representation.
    pub fn pop(&mut self) -> Option<T> {
        match &mut self.repr {
            Repr::Inline { len, items } => {
                if *len == 0 {
                    None
                } else {
                    *len -= 1;
                    Some(items[*len])
                }
            }
            Repr::Heap(v) => v.pop(),
        }
    }
}

impl<T: Copy + Default> Default for InlineOrHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default> Deref for InlineOrHeap<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy + Default + PartialEq> PartialEq for InlineOrHeap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Default + Eq> Eq for InlineOrHeap<T> {}

impl<T: Copy + Default + fmt::Debug> fmt::Debug for InlineOrHeap<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T: Copy + Default> FromIterator<T> for InlineOrHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out = Self::new();
        for item in iter {
            out.push(item);
        }
        out
    }
}

/// A tensor's runtime dimensions.
///
/// Holds dimensions and nothing else: every derived quantity is recomputed on
/// demand, so none of them can disagree with the shape.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct ShapeBuf {
    dims: InlineOrHeap<usize>,
}

impl ShapeBuf {
    /// The rank-0 shape, usable in a `const` context.
    pub const SCALAR: Self = Self {
        dims: InlineOrHeap::EMPTY,
    };

    /// The rank-0 shape, which holds a single scalar element.
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            dims: InlineOrHeap::new(),
        }
    }

    /// Build from dimensions.
    #[must_use]
    pub fn from_slice(dims: &[usize]) -> Self {
        Self {
            dims: InlineOrHeap::from_slice(dims),
        }
    }

    /// The dimensions.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        self.dims.as_slice()
    }

    /// The dimensions, mutably. The rank cannot change through this handle.
    pub fn dims_mut(&mut self) -> &mut [usize] {
        self.dims.as_mut_slice()
    }

    /// Number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Whether the dimensions are stored inline. See
    /// [`InlineOrHeap::is_inline`].
    #[must_use]
    pub fn is_inline(&self) -> bool {
        self.dims.is_inline()
    }

    /// Append a dimension.
    pub fn push(&mut self, dim: usize) {
        self.dims.push(dim);
    }

    /// Remove and return the last dimension.
    pub fn pop(&mut self) -> Option<usize> {
        self.dims.pop()
    }

    /// Total element count, or `None` if the product overflows `usize`.
    ///
    /// A rank-0 shape holds one element, so the empty product is 1. Any zero
    /// dimension gives 0.
    ///
    /// The zero case is short-circuited rather than folded, and that is load
    /// bearing: a running product would make the answer depend on axis order.
    /// `[MAX, 0, MAX]` reaches the zero and collapses, while `[MAX, MAX, 0]`
    /// overflows before it gets there — the same dimensions, two different
    /// answers. An empty tensor holds no elements no matter how its axes are
    /// written.
    pub fn numel(&self) -> Option<usize> {
        if self.is_empty_tensor() {
            return Some(0);
        }
        // Every remaining dimension is at least 1, so partial products are
        // monotone: the fold overflows exactly when the true product does,
        // regardless of the order it visits axes in.
        self.dims()
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }

    /// Total element count, reported as a [`ShapeError`] on overflow.
    ///
    /// `operation` is supplied by the caller because the buffer does not know
    /// which rule is being resolved, and a diagnostic that cannot name the
    /// operation is not worth much.
    pub fn checked_numel(&self, operation: OperationKind) -> Result<usize, ShapeError> {
        self.numel().ok_or(ShapeError::ArithmeticOverflow {
            operation,
            expression: "product of dimensions",
        })
    }

    /// Byte length of a dense buffer holding this shape, for elements of
    /// `element_size` bytes.
    ///
    /// Both multiplications are checked. The element count can fit `usize` and
    /// the byte length still not, which is precisely the case that silently
    /// undersizes an allocation when the multiply is unchecked.
    pub fn checked_byte_len(
        &self,
        element_size: usize,
        operation: OperationKind,
    ) -> Result<usize, ShapeError> {
        self.checked_numel(operation)?
            .checked_mul(element_size)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation,
                expression: "element count * element size",
            })
    }

    /// Whether any dimension is 0, which makes the shape hold no elements.
    #[must_use]
    pub fn is_empty_tensor(&self) -> bool {
        self.dims().contains(&0)
    }
}

// ShapeBuf is the canonical runtime shape value.  Serialize it as its logical
// dimension sequence rather than exposing the inline/heap storage choice.
impl serde::Serialize for ShapeBuf {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.dims().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ShapeBuf {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let dims = alloc::vec::Vec::<usize>::deserialize(deserializer)?;
        Ok(Self::from_slice(&dims))
    }
}

impl PartialEq<Vec<usize>> for ShapeBuf {
    fn eq(&self, other: &Vec<usize>) -> bool {
        self.dims() == other.as_slice()
    }
}

impl PartialEq<&[usize]> for ShapeBuf {
    fn eq(&self, other: &&[usize]) -> bool {
        self.dims() == *other
    }
}

impl<const N: usize> PartialEq<[usize; N]> for ShapeBuf {
    fn eq(&self, other: &[usize; N]) -> bool {
        self.dims() == other.as_slice()
    }
}

impl Deref for ShapeBuf {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        self.dims()
    }
}

impl fmt::Debug for ShapeBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.dims(), f)
    }
}

impl From<&[usize]> for ShapeBuf {
    fn from(dims: &[usize]) -> Self {
        Self::from_slice(dims)
    }
}

impl FromIterator<usize> for ShapeBuf {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self {
            dims: iter.into_iter().collect(),
        }
    }
}

impl AsRef<[usize]> for ShapeBuf {
    fn as_ref(&self) -> &[usize] {
        self.dims()
    }
}

impl AsMut<[usize]> for ShapeBuf {
    fn as_mut(&mut self) -> &mut [usize] {
        self.dims_mut()
    }
}

impl<I: core::slice::SliceIndex<[usize]>> core::ops::Index<I> for ShapeBuf {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.dims()[index]
    }
}

impl<I: core::slice::SliceIndex<[usize]>> core::ops::IndexMut<I> for ShapeBuf {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.dims_mut()[index]
    }
}

impl From<ShapeBuf> for Vec<usize> {
    fn from(buf: ShapeBuf) -> Self {
        buf.dims().to_vec()
    }
}

impl IntoIterator for ShapeBuf {
    type Item = usize;
    type IntoIter = alloc::vec::IntoIter<usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.dims().to_vec().into_iter()
    }
}

impl<'a> IntoIterator for &'a ShapeBuf {
    type Item = &'a usize;
    type IntoIter = core::slice::Iter<'a, usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.dims().iter()
    }
}

/// A tensor's per-axis element strides.
///
/// Separate from [`ShapeBuf`] because the two are not interchangeable: strides
/// may repeat (a broadcast axis has stride 0) and need not be ordered, so no
/// shape invariant applies to them.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct StrideBuf {
    strides: InlineOrHeap<usize>,
}

impl StrideBuf {
    /// The rank-0 stride list, usable in a `const` context.
    pub const EMPTY: Self = Self {
        strides: InlineOrHeap::EMPTY,
    };

    /// Build from strides.
    #[must_use]
    pub fn from_slice(strides: &[usize]) -> Self {
        Self {
            strides: InlineOrHeap::from_slice(strides),
        }
    }

    /// The strides.
    #[must_use]
    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }

    /// Number of axes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strides.len()
    }

    /// Whether there are no axes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strides.is_empty()
    }

    /// Whether the strides are stored inline. See [`InlineOrHeap::is_inline`].
    #[must_use]
    pub fn is_inline(&self) -> bool {
        self.strides.is_inline()
    }

    /// Append a stride.
    ///
    /// Mirrors [`ShapeBuf::push`], for the normalizing loops that build one
    /// stride per *output* axis and therefore cannot know the length up front.
    pub fn push(&mut self, stride: usize) {
        self.strides.push(stride);
    }

    /// Remove and return the last stride.
    pub fn pop(&mut self) -> Option<usize> {
        self.strides.pop()
    }

    /// Row-major (C-contiguous) strides for `shape`.
    ///
    /// The last axis has stride 1 and each earlier axis is the product of all
    /// later dimensions. This replaces `cpu::stride::contiguous_strides`, whose
    /// overflow path is a `panic!`: an unchecked multiply here can wrap to a
    /// small stride that is then used to index a buffer sized from the same
    /// wrapped arithmetic.
    pub fn contiguous_for(shape: &ShapeBuf, operation: OperationKind) -> Result<Self, ShapeError> {
        let dims = shape.dims();
        let mut strides = InlineOrHeap::from_slice(&[]);
        for _ in dims {
            strides.push(1usize);
        }
        let slice = strides.as_mut_slice();
        for axis in (0..dims.len().saturating_sub(1)).rev() {
            slice[axis] = slice[axis + 1].checked_mul(dims[axis + 1]).ok_or(
                ShapeError::ArithmeticOverflow {
                    operation,
                    expression: "stride * trailing dimension",
                },
            )?;
        }
        Ok(Self { strides })
    }

    /// Whether these strides are the row-major strides for `shape`.
    ///
    /// A shape whose contiguous strides overflow is not contiguous under any
    /// stride buffer, so overflow answers `false` rather than propagating.
    #[must_use]
    pub fn is_contiguous_for(&self, shape: &ShapeBuf) -> bool {
        Self::contiguous_for(shape, OperationKind::Storage).is_ok_and(|expected| expected == *self)
    }

    /// Number of elements the view spans, from its first element through its
    /// last inclusive.
    ///
    /// This is what a bounds check needs, and it is *not* the element count: a
    /// strided or broadcast view can span far more or far fewer elements than
    /// it addresses. A shape with any zero dimension spans 0.
    pub fn checked_span(
        &self,
        shape: &ShapeBuf,
        operation: OperationKind,
    ) -> Result<usize, ShapeError> {
        if self.len() != shape.rank() {
            return Err(ShapeError::RankMismatch {
                operation,
                expected: RankExpectation::SameAs {
                    operand: "shape",
                    rank: shape.rank(),
                },
                actual: self.len(),
            });
        }
        if shape.is_empty_tensor() {
            return Ok(0);
        }
        let mut span = 1usize;
        for (&dim, &stride) in shape.dims().iter().zip(self.strides()) {
            let extent = (dim - 1).checked_mul(stride).ok_or({
                ShapeError::ArithmeticOverflow {
                    operation,
                    expression: "(dimension - 1) * stride",
                }
            })?;
            span = span
                .checked_add(extent)
                .ok_or(ShapeError::ArithmeticOverflow {
                    operation,
                    expression: "sum of axis extents",
                })?;
        }
        Ok(span)
    }
}

impl PartialEq<Vec<usize>> for StrideBuf {
    fn eq(&self, other: &Vec<usize>) -> bool {
        self.strides() == other.as_slice()
    }
}

impl Deref for StrideBuf {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        self.strides()
    }
}

impl fmt::Debug for StrideBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.strides(), f)
    }
}

impl From<&[usize]> for StrideBuf {
    fn from(strides: &[usize]) -> Self {
        Self::from_slice(strides)
    }
}

impl FromIterator<usize> for StrideBuf {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self {
            strides: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_numel_distinguishes_scalar_zero_and_overflow() {
        assert_eq!(
            ShapeBuf::scalar()
                .checked_numel(OperationKind::Storage)
                .unwrap(),
            1
        );
        assert_eq!(
            ShapeBuf::from_slice(&[usize::MAX, 0, usize::MAX])
                .checked_numel(OperationKind::Storage)
                .unwrap(),
            0
        );
        assert!(matches!(
            ShapeBuf::from_slice(&[usize::MAX, 2]).checked_numel(OperationKind::Storage),
            Err(ShapeError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn contiguous_strides_reject_unrepresentable_layouts_even_when_empty() {
        let overflowing = ShapeBuf::from_slice(&[2, usize::MAX, usize::MAX]);
        assert!(matches!(
            StrideBuf::contiguous_for(&overflowing, OperationKind::Storage),
            Err(ShapeError::ArithmeticOverflow { .. })
        ));

        let empty = ShapeBuf::from_slice(&[0, usize::MAX, usize::MAX]);
        assert!(matches!(
            StrideBuf::contiguous_for(&empty, OperationKind::Storage),
            Err(ShapeError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn checked_span_rejects_rank_and_arithmetic_overflow() {
        let shape = ShapeBuf::from_slice(&[2, 2]);
        assert!(matches!(
            StrideBuf::from_slice(&[1]).checked_span(&shape, OperationKind::Slice),
            Err(ShapeError::RankMismatch { .. })
        ));
        assert!(matches!(
            StrideBuf::from_slice(&[usize::MAX, usize::MAX])
                .checked_span(&shape, OperationKind::Slice),
            Err(ShapeError::ArithmeticOverflow { .. })
        ));
    }
}
