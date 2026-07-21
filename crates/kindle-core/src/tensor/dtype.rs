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
    fn to_kindle(dtype: &Self::Field) -> KindleDType;
}

/// `DynDType`.
pub trait DynDType: DType {}

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
    const DTYPE: KindleDType;
}

/// `QuantDType`.
pub trait QuantDType: DType {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// `Q8_0`.
pub struct Q8_0;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// `KindleDType`.
pub enum KindleDType {
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

impl KindleDType {
    /// Returns the size in bytes of a single element of this dtype.
    pub fn element_size(&self) -> usize {
        match self {
            KindleDType::U8 | KindleDType::Q8_0 => 1,
            KindleDType::F16 | KindleDType::BF16 => 2,
            KindleDType::F32 | KindleDType::U32 => 4,
            KindleDType::F64 | KindleDType::I64 => 8,
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
                fn to_kindle(_: &Self::Field) -> KindleDType {
                    KindleDType::$repr
                }
                /// `init`.
                fn init(_: Self::Arg) -> Self::Field {
                    PhantomData
                }
            }
            impl DynDType for $t { }
            impl ConstDType for $t {
                /// `Elem`.
                type Elem = $t;
                /// `DTYPE`.
                const DTYPE: KindleDType = KindleDType::$repr;
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
    type Arg = KindleDType;
    /// `Field`.
    type Field = KindleDType;

    /// `init`.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }

    /// `to_kindle`.
    fn to_kindle(dtype: &Self::Field) -> KindleDType {
        *dtype
    }
}
impl DynDType for Dyn {}
