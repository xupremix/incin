use crate::prelude::Dyn;
use crate::shapes::error::{OperationKind, ShapeError};

use core::{fmt::Debug, marker::PhantomData};
pub use half::{bf16, f16};

/// A type-level tensor element type (`f32`, `f16`, `bf16`, `u8`, `i64`,
/// `Q8_0`, or `Dyn` for a runtime-chosen dtype). Paired with a `Device`
/// via `BackendFor<T>` to select a concrete backend.
pub trait DType: 'static + Clone + Debug + Send + Sync + PartialEq {
    /// The user-facing constructor argument (`()` for compile-time-fixed
    /// dtypes, `DTypeId` for `Dyn`).
    type Arg;
    /// The runtime-stored representation (a `PhantomData` for compile-
    /// time-fixed dtypes, `DTypeId` for `Dyn`).
    type Field: Debug + Clone + Default;
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field;
    /// Resolves this dtype's runtime `DTypeId`.
    fn to_incin(dtype: &Self::Field) -> DTypeId;
}

/// Marker for floating-point dtypes (`f32`/`f64`/`f16`/`bf16`).
pub trait FloatDType: DType {}
/// Marker for integer dtypes (`u8`/`u32`/`i64`).
pub trait IntDType: DType {}
/// Marker for the boolean dtype.
pub trait BoolDType: DType {}

pub mod sealed {
    pub trait TensorElementSealed {}
}

/// Marker trait enforcing that a tensor element type is POD, Zeroable, safe, and sealed (`SEC-005`).
pub trait TensorElement:
    sealed::TensorElementSealed
    + bytemuck::Pod
    + bytemuck::Zeroable
    + Copy
    + Debug
    + Send
    + Sync
    + 'static
{
}

impl<T> TensorElement for T where
    T: sealed::TensorElementSealed
        + bytemuck::Pod
        + bytemuck::Zeroable
        + Copy
        + Debug
        + Send
        + Sync
        + 'static
{
}

/// A `DType` whose identity is fully known at compile time (as opposed to
/// `Dyn`, which is resolved at runtime) — takes no constructor argument.
pub trait ConstDType: DType<Arg = ()> {
    /// The Rust element type corresponding to this dtype.
    type Elem: TensorElement;
    /// The compile-time-known `DTypeId`.
    const DTYPE: DTypeId;
}

/// Marker for block-quantized dtypes (e.g. `Q8_0`) — storage formats with
/// their own internal scale/block structure, not plain scalar elements.
pub trait QuantDType: DType {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Q8_0 block quantization: groups of 32 elements share one `f32` scale,
/// each element stored as a scaled `i8`.
pub struct Q8_0;

#[non_exhaustive]
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
/// The runtime-identifiable element type a storage handle holds — every
/// `DType::to_incin` resolves to one of these.
pub enum DTypeId {
    /// 8-bit unsigned integer.
    U8,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit signed integer.
    I64,
    /// 16-bit brain floating point.
    BF16,
    /// 16-bit (IEEE 754 half-precision) floating point.
    F16,
    /// 32-bit floating point.
    #[default]
    F32,
    /// 64-bit floating point.
    F64,
    /// Q8_0 block-quantized 8-bit integer.
    Q8_0,
}

impl DTypeId {
    /// The lowercase name used in diagnostics, generated documentation, and
    /// `cargo incin doctor`'s report.
    ///
    /// The counterpart of [`OperationKind::name`](crate::prelude::OperationKind::name)
    /// and [`LayoutClass::as_str`](crate::exec::LayoutClass::as_str): one
    /// spelling per dtype, so the capability tables, the doctor's probe lines
    /// and a shape error cannot disagree about what to call `F32`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Q8_0 => "q8_0",
        }
    }

    /// Returns the size in bytes of a single element of this dtype.
    ///
    /// This is the *scalar* element width. It is not a storage width for
    /// block-quantized dtypes: `Q8_0` reports 1 because one quantized value is
    /// one `i8`, but 32 of them share an `f16` scale and occupy a 34-byte
    /// block. Use [`size_bytes`](Self::size_bytes) to size an allocation.
    pub fn element_size(&self) -> usize {
        match self {
            DTypeId::U8 | DTypeId::Q8_0 => 1,
            DTypeId::F16 | DTypeId::BF16 => 2,
            DTypeId::F32 | DTypeId::U32 => 4,
            DTypeId::F64 | DTypeId::I64 => 8,
        }
    }

    /// Logical values packed into one physical storage block.
    ///
    /// Scalar dtypes store one value per block. `Q8_0` packs 32.
    #[must_use]
    pub const fn block_elements(self) -> usize {
        match self {
            DTypeId::Q8_0 => 32,
            _ => 1,
        }
    }

    /// Bytes occupied by one physical storage block.
    ///
    /// For scalar dtypes this equals [`element_size`](Self::element_size). A
    /// `Q8_0` block is an `f16` scale followed by 32 `i8` values: 34 bytes for
    /// 32 logical values, which is why byte lengths cannot be derived from the
    /// element width alone.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            DTypeId::U8 => 1,
            DTypeId::F16 | DTypeId::BF16 => 2,
            DTypeId::F32 | DTypeId::U32 => 4,
            DTypeId::F64 | DTypeId::I64 => 8,
            DTypeId::Q8_0 => 34,
        }
    }

    /// Bytes occupied by `elements` logical values of this dtype.
    ///
    /// This is the single byte-arithmetic entry point for storage sizing. It
    /// is checked, so an element count that fits `usize` but whose byte length
    /// does not is reported rather than silently truncated into an undersized
    /// allocation. A block-quantized count that does not fill whole blocks is
    /// rejected: half a `Q8_0` block has no representation.
    pub fn size_bytes(
        self,
        elements: usize,
        operation: OperationKind,
    ) -> Result<usize, ShapeError> {
        let per_block = self.block_elements();
        if !elements.is_multiple_of(per_block) {
            return Err(ShapeError::InvalidParameter {
                operation,
                parameter: "elements",
                value: elements,
            });
        }
        (elements / per_block)
            .checked_mul(self.block_bytes())
            .ok_or(ShapeError::ArithmeticOverflow {
                operation,
                expression: "block count * block size",
            })
    }
}

macro_rules! impl_dtype {
    ($($repr:ident $t:ty),* $(,)?) => {
        $(
            impl sealed::TensorElementSealed for $t {}

            impl DType for $t {
                /// No argument needed — the dtype is fixed by the Rust type itself.
                type Arg = ();
                /// Zero-sized: the value is fixed by the type.
                type Field = PhantomData<$t>;
                /// The compile-time-known `DTypeId` for this Rust type.
                fn to_incin(_: &Self::Field) -> DTypeId {
                    DTypeId::$repr
                }
                /// No-op: nothing to convert.
                fn init(_: Self::Arg) -> Self::Field {
                    PhantomData
                }
            }
            impl ConstDType for $t {
                /// This Rust type itself.
                type Elem = $t;
                /// The compile-time-known `DTypeId`.
                const DTYPE: DTypeId = DTypeId::$repr;
            }
        )*
    };
}

unsafe impl bytemuck::Zeroable for Q8_0 {}
unsafe impl bytemuck::Pod for Q8_0 {}

impl_dtype!(
    F32 f32,
    F64 f64,
    U8 u8,
    U32 u32,
    I64 i64,
    F16 f16,
    BF16 bf16,
    Q8_0 Q8_0,
);

impl FloatDType for f32 {}
impl FloatDType for f64 {}
impl FloatDType for f16 {}
impl FloatDType for bf16 {}

impl IntDType for u8 {}
impl IntDType for u32 {}
impl IntDType for i64 {}

impl QuantDType for Q8_0 {}

/// Marker trait for plain scalar dtypes (excluding block quantization).
pub trait PlainDType: DType {
    /// Plain scalar element type.
    type Elem: TensorElement;
}

impl PlainDType for f32 {
    type Elem = f32;
}
impl PlainDType for f64 {
    type Elem = f64;
}
impl PlainDType for u8 {
    type Elem = u8;
}
impl PlainDType for u32 {
    type Elem = u32;
}
impl PlainDType for i64 {
    type Elem = i64;
}
impl PlainDType for f16 {
    type Elem = f16;
}
impl PlainDType for bf16 {
    type Elem = bf16;
}

impl DType for Dyn {
    /// The runtime-chosen dtype.
    type Arg = DTypeId;
    /// Stored directly — `Dyn`'s whole point is deferring dtype choice
    /// to runtime, so `Field` is just the `DTypeId` itself.
    type Field = DTypeId;

    /// Stores the `DTypeId` verbatim.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }

    /// Already a `DTypeId` — returned as-is.
    fn to_incin(dtype: &Self::Field) -> DTypeId {
        *dtype
    }
}
