use crate::prelude::Dyn;

use core::{fmt::Debug, marker::PhantomData};
pub use half::{bf16, f16};

/// `DType`.
pub trait DType: 'static + Clone + Debug + Send + Sync + PartialEq {
    /// `Arg`.
    type Arg;
    /// `Field`.
    type Field: Debug + Clone;
    /// `init`.
    fn init(arg: Self::Arg) -> Self::Field;
    /// `to_kindle`.
    fn to_kindle(dtype: &Self::Field) -> DTypeId;
}

/// `FloatDType`.
pub trait FloatDType: DType {}
/// `IntDType`.
pub trait IntDType: DType {}
/// `BoolDType`.
pub trait BoolDType: DType {}

/// `ConstDType`.
pub trait ConstDType: DType<Arg = ()> {
    /// The Rust element type corresponding to this dtype.
    type Elem: 'static + Copy + Debug + Send + Sync;
    /// `DTYPE`.
    const DTYPE: DTypeId;
}

/// `QuantDType`.
pub trait QuantDType: DType {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// `Q8_0`.
pub struct Q8_0;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// `DTypeId`.
pub enum DTypeId {
    /// `U8`.
    U8,
    /// `U32`.
    U32,
    /// `I64`.
    I64,
    /// `BF16`.
    BF16,
    /// `F16`.
    F16,
    /// `F32`.
    F32,
    /// `F64`.
    F64,
    /// `Q8_0`.
    Q8_0,
}

impl DTypeId {
    /// Returns the size in bytes of a single element of this dtype.
    pub fn element_size(&self) -> usize {
        match self {
            DTypeId::U8 | DTypeId::Q8_0 => 1,
            DTypeId::F16 | DTypeId::BF16 => 2,
            DTypeId::F32 | DTypeId::U32 => 4,
            DTypeId::F64 | DTypeId::I64 => 8,
        }
    }
}

macro_rules! impl_dtype {
    ($($repr:ident $t:ty),* $(,)?) => {
        $(
            impl DType for $t {
                /// `Arg`.
                type Arg = ();
                /// `Field`.
                type Field = PhantomData<$t>;
                /// `to_kindle`.
                fn to_kindle(_: &Self::Field) -> DTypeId {
                    DTypeId::$repr
                }
                /// `init`.
                fn init(_: Self::Arg) -> Self::Field {
                    PhantomData
                }
            }
            impl ConstDType for $t {
                /// `Elem`.
                type Elem = $t;
                /// `DTYPE`.
                const DTYPE: DTypeId = DTypeId::$repr;
            }
        )*
    };
}

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

impl DType for Dyn {
    /// `Arg`.
    type Arg = DTypeId;
    /// `Field`.
    type Field = DTypeId;

    /// `init`.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }

    /// `to_kindle`.
    fn to_kindle(dtype: &Self::Field) -> DTypeId {
        *dtype
    }
}
