//! Checked, backend-neutral physical tensor metadata.
//!
//! Logical operation descriptors deliberately exclude these facts: shape
//! views, storage offsets, runtime dtype/device identity, and alignment belong
//! to a concrete storage handle. `TensorMeta` validates them together so no
//! backend can update one raw vector while leaving another stale.

use core::fmt;

use crate::shapes::{OperationKind, ShapeBuf, ShapeError, StrideBuf};
use crate::tensor::device::DeviceId;
use crate::tensor::dtype::{ConstDType, DTypeDescriptor};

/// The shared layout vocabulary used by tensor metadata, capability queries,
/// and kernel specialization.
///
/// A single tensor is classified as [`Contiguous`](Self::Contiguous) or
/// [`Strided`](Self::Strided). The remaining variants describe relationships
/// between operands or operation-specific access patterns and replace the
/// former backend-local unary, binary, and kernel layout enums.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum LayoutClass {
    /// General strided access.
    Strided,
    /// Dense row-major access.
    Contiguous,
    /// A scalar left operand broadcast over a contiguous right operand.
    ScalarLeft,
    /// A contiguous left operand with a scalar right operand.
    ScalarRight,
    /// Reduction over a physically contiguous final axis.
    ContiguousLastAxis,
    /// Rows are independently contiguous.
    RowWise,
    /// Channels are independently contiguous.
    ChannelWise,
}

impl LayoutClass {
    /// Stable diagnostic/cache-key spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strided => "strided",
            Self::Contiguous => "contiguous",
            Self::ScalarLeft => "scalar-left",
            Self::ScalarRight => "scalar-right",
            Self::ContiguousLastAxis => "contiguous-last-axis",
            Self::RowWise => "row-wise",
            Self::ChannelWise => "channel-wise",
        }
    }
}

/// A proved power-of-two byte alignment.
///
/// The value is a guarantee ("aligned to at least N bytes"), not a preferred
/// vector width. View offsets can weaken it but can never strengthen it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Alignment(usize);

impl Alignment {
    /// One-byte alignment, which every allocation and view satisfies.
    pub const BYTE: Self = Self(1);

    /// The alignment guaranteed for values of `T`.
    ///
    /// Rust alignments are always non-zero powers of two, so this constructor
    /// cannot fail and is the preferred path for typed host allocations.
    pub const fn of<T>() -> Self {
        Self(core::mem::align_of::<T>())
    }

    /// Construct a power-of-two alignment guarantee.
    pub fn new(bytes: usize) -> Result<Self, MetaError> {
        if bytes.is_power_of_two() {
            Ok(Self(bytes))
        } else {
            Err(MetaError::InvalidAlignment { bytes })
        }
    }

    /// Guaranteed byte alignment.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self.0
    }

    /// Whether this guarantee satisfies `required_bytes`.
    #[must_use]
    pub const fn supports(self, required_bytes: usize) -> bool {
        required_bytes.is_power_of_two() && self.0 >= required_bytes
    }

    fn after_offset_bytes(self, offset_bytes: usize) -> Self {
        if offset_bytes == 0 {
            return self;
        }
        let offset_alignment = 1usize << offset_bytes.trailing_zeros();
        Self(core::cmp::min(self.0, offset_alignment))
    }
}

impl Default for Alignment {
    fn default() -> Self {
        Self::BYTE
    }
}

/// Failure while binding logical metadata to a physical allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaError {
    /// Shape/stride rank or arithmetic validation failed.
    Shape(ShapeError),
    /// An alignment guarantee must be a non-zero power of two.
    InvalidAlignment {
        /// Rejected byte alignment.
        bytes: usize,
    },
    /// The view's inclusive span does not fit the allocation.
    OutOfBounds {
        /// First addressed element.
        offset_elements: usize,
        /// Number of elements from the first through last address.
        span_elements: usize,
        /// Allocation capacity in elements.
        capacity_elements: usize,
    },
}

impl From<ShapeError> for MetaError {
    fn from(error: ShapeError) -> Self {
        Self::Shape(error)
    }
}

impl fmt::Display for MetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(error) => fmt::Display::fmt(error, f),
            Self::InvalidAlignment { bytes } => write!(
                f,
                "invalid storage alignment {bytes}: expected a non-zero power of two"
            ),
            Self::OutOfBounds {
                offset_elements,
                span_elements,
                capacity_elements,
            } => write!(
                f,
                "storage view out of bounds: offset {offset_elements} + span {span_elements} exceeds capacity {capacity_elements}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MetaError {}

/// Checked physical metadata shared by every backend storage handle.
///
/// Backend storage embeds this value behind shared access only. Its readable
/// fields are exposed through a shared-only view; `TensorMeta` does not
/// implement `DerefMut`, and its private wrapper prevents constructing or
/// mutating an execution proof without the checked constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    fields: TensorMetaFields,
}

/// Read-only field view of checked tensor metadata.
///
/// Constructing this plain record does not construct a [`TensorMeta`]; only
/// [`TensorMeta::try_new`] and [`TensorMeta::contiguous`] can do that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetaFields {
    /// Runtime dimensions.
    pub shape: ShapeBuf,
    /// Per-axis element strides.
    pub strides: StrideBuf,
    /// First addressed element in the allocation.
    pub offset_elements: usize,
    /// Physical element dtype.
    pub dtype: DTypeDescriptor,
    /// Physical runtime device.
    pub device: DeviceId,
    /// Layout class derived from shape and strides.
    pub layout: LayoutClass,
    /// Effective byte alignment after applying the view offset.
    pub alignment: Alignment,
}

impl core::ops::Deref for TensorMeta {
    type Target = TensorMetaFields;

    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}

impl TensorMeta {
    /// Metadata for a handle that owns no allocation.
    ///
    /// A backend whose storage is an enum over several devices has one variant
    /// that holds nothing - it exists so the enum has a shape when no backend
    /// feature is enabled. [`StorageBackend::metadata`] returns `&TensorMeta`
    /// infallibly and so cannot report "there is no allocation here"; this
    /// constant is what that variant describes instead. It is a well-formed
    /// rank-0 contiguous descriptor rather than a sentinel, so every accessor
    /// on it answers consistently - but it describes no real buffer, and a
    /// backend that reaches it has been handed a handle it cannot execute.
    ///
    /// [`StorageBackend::metadata`]: crate::tensor::backend::StorageBackend::metadata
    pub const UNALLOCATED: Self = Self {
        fields: TensorMetaFields {
            shape: ShapeBuf::SCALAR,
            strides: StrideBuf::EMPTY,
            offset_elements: 0,
            dtype: <f32 as ConstDType>::DESCRIPTOR,
            device: DeviceId::CPU,
            layout: LayoutClass::Contiguous,
            alignment: Alignment::BYTE,
        },
    };

    /// Validate and bind a strided view to an allocation.
    pub fn try_new(
        shape: ShapeBuf,
        strides: StrideBuf,
        offset_elements: usize,
        dtype: DTypeDescriptor,
        device: DeviceId,
        allocation_alignment: Alignment,
        capacity_elements: usize,
    ) -> Result<Self, MetaError> {
        let span_elements = strides.checked_span(&shape, OperationKind::Storage)?;
        let end =
            offset_elements
                .checked_add(span_elements)
                .ok_or(ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Storage,
                    expression: "storage offset + view span",
                })?;
        if end > capacity_elements {
            return Err(MetaError::OutOfBounds {
                offset_elements,
                span_elements,
                capacity_elements,
            });
        }
        let offset_bytes = dtype.size_bytes(offset_elements, OperationKind::Storage)?;
        let layout = if strides.is_contiguous_for(&shape) {
            LayoutClass::Contiguous
        } else {
            LayoutClass::Strided
        };
        Ok(Self {
            fields: TensorMetaFields {
                shape,
                strides,
                offset_elements,
                dtype,
                device,
                layout,
                alignment: allocation_alignment.after_offset_bytes(offset_bytes),
            },
        })
    }

    /// Build checked row-major metadata at offset zero.
    pub fn contiguous(
        shape: ShapeBuf,
        dtype: DTypeDescriptor,
        device: DeviceId,
        allocation_alignment: Alignment,
        capacity_elements: usize,
    ) -> Result<Self, MetaError> {
        let strides = StrideBuf::contiguous_for(&shape, OperationKind::Storage)?;
        Self::try_new(
            shape,
            strides,
            0,
            dtype,
            device,
            allocation_alignment,
            capacity_elements,
        )
    }

    /// Runtime dimensions.
    #[must_use]
    pub const fn shape(&self) -> &ShapeBuf {
        &self.fields.shape
    }

    /// Per-axis element strides.
    #[must_use]
    pub const fn strides(&self) -> &StrideBuf {
        &self.fields.strides
    }

    /// First addressed element in the allocation.
    #[must_use]
    pub const fn offset_elements(&self) -> usize {
        self.fields.offset_elements
    }

    /// Physical element dtype.
    #[must_use]
    pub const fn dtype(&self) -> DTypeDescriptor {
        self.fields.dtype
    }

    /// Physical runtime device.
    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.fields.device
    }

    /// Derived layout class.
    #[must_use]
    pub const fn layout(&self) -> LayoutClass {
        self.fields.layout
    }

    /// Effective byte alignment after applying the view offset.
    #[must_use]
    pub const fn alignment(&self) -> Alignment {
        self.fields.alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        alignment: usize,
        capacity: usize,
    ) -> Result<TensorMeta, MetaError> {
        TensorMeta::try_new(
            ShapeBuf::from_slice(shape),
            StrideBuf::from_slice(strides),
            offset,
            <f32 as ConstDType>::DESCRIPTOR,
            DeviceId::cpu(),
            Alignment::new(alignment)?,
            capacity,
        )
    }

    #[test]
    fn contiguous_layout_is_derived_not_asserted() {
        let meta = meta(&[2, 3], &[3, 1], 0, 16, 6).unwrap();
        assert_eq!(meta.layout(), LayoutClass::Contiguous);
        assert_eq!(meta.shape().dims(), &[2, 3]);
        assert_eq!(meta.strides().strides(), &[3, 1]);
    }

    #[test]
    fn transposed_and_broadcast_views_are_strided() {
        let transposed = meta(&[3, 2], &[1, 3], 0, 16, 6).unwrap();
        let broadcast = meta(&[4, 3], &[0, 1], 0, 16, 3).unwrap();
        assert_eq!(transposed.layout(), LayoutClass::Strided);
        assert_eq!(broadcast.layout(), LayoutClass::Strided);
    }

    #[test]
    fn a_view_offset_weakens_but_never_strengthens_alignment() {
        let four = meta(&[1], &[1], 1, 16, 4).unwrap();
        let eight = meta(&[1], &[1], 2, 16, 4).unwrap();
        assert_eq!(four.alignment().bytes(), 4);
        assert_eq!(eight.alignment().bytes(), 8);
        assert!(eight.alignment().supports(4));
        assert!(!eight.alignment().supports(16));
    }

    #[test]
    fn zero_offset_preserves_the_allocation_alignment() {
        let meta = meta(&[2], &[1], 0, 32, 2).unwrap();
        assert_eq!(meta.alignment().bytes(), 32);
    }

    #[test]
    fn empty_views_may_sit_at_but_not_after_the_allocation_end() {
        assert!(meta(&[0, usize::MAX], &[usize::MAX, 1], 7, 8, 7).is_ok());
        assert!(matches!(
            meta(&[0], &[1], 8, 8, 7),
            Err(MetaError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn strided_span_and_offset_are_checked_against_capacity() {
        assert!(matches!(
            meta(&[2, 2], &[3, 1], 2, 16, 6),
            Err(MetaError::OutOfBounds {
                offset_elements: 2,
                span_elements: 5,
                capacity_elements: 6,
            })
        ));
    }

    #[test]
    fn rank_and_arithmetic_failures_remain_structured() {
        assert!(matches!(
            meta(&[2, 3], &[1], 0, 8, 6),
            Err(MetaError::Shape(ShapeError::RankMismatch { .. }))
        ));
        assert!(matches!(
            meta(&[usize::MAX, 2], &[2, 1], 0, 8, usize::MAX),
            Err(MetaError::Shape(ShapeError::ArithmeticOverflow { .. }))
        ));
    }

    #[test]
    fn invalid_alignment_is_rejected() {
        assert!(matches!(
            Alignment::new(0),
            Err(MetaError::InvalidAlignment { bytes: 0 })
        ));
        assert!(matches!(
            Alignment::new(24),
            Err(MetaError::InvalidAlignment { bytes: 24 })
        ));
    }
}
