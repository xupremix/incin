use crate::prelude::Dyn;

use core::{fmt::Debug, marker::PhantomData};
pub use half::{bf16, f16};

/// Auto-generated documentation for DType.
pub trait DType: 'static + Clone + Debug + Send + Sync + PartialEq {
    /// Auto-generated documentation for Arg.
    type Arg;
    /// Auto-generated documentation for Field.
    type Field: Debug + Clone;
    /// Auto-generated documentation for init.
    fn init(arg: Self::Arg) -> Self::Field;
    /// Auto-generated documentation for to_kindle.
    fn to_kindle(dtype: &Self::Field) -> KindleDType;
}

/// Auto-generated documentation for DynDType.
pub trait DynDType: DType {}

/// Auto-generated documentation for FloatDType.
pub trait FloatDType: DType {}
/// Auto-generated documentation for IntDType.
pub trait IntDType: DType {}
/// Auto-generated documentation for BoolDType.
pub trait BoolDType: DType {}

/// Auto-generated documentation for ConstDType.
pub trait ConstDType: DType<Arg = ()> {
    /// The Rust element type corresponding to this dtype.
    type Elem: 'static + Copy + Debug + Send + Sync;
    /// Auto-generated documentation for DTYPE.
    const DTYPE: KindleDType;
}

/// Auto-generated documentation for QuantDType.
pub trait QuantDType: DType {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Auto-generated documentation for Q8_0.
pub struct Q8_0;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Auto-generated documentation for KindleDType.
pub enum KindleDType {
    /// Auto-generated documentation for U8.
    U8,
    /// Auto-generated documentation for U32.
    U32,
    /// Auto-generated documentation for I64.
    I64,
    /// Auto-generated documentation for BF16.
    BF16,
    /// Auto-generated documentation for F16.
    F16,
    /// Auto-generated documentation for F32.
    F32,
    /// Auto-generated documentation for F64.
    F64,
    /// Auto-generated documentation for Q8_0.
    Q8_0,
}

macro_rules! impl_dtype {
    ($($repr:ident $t:ty),* $(,)?) => {
        $(
            impl DType for $t {
                /// Auto-generated documentation for Arg.
                type Arg = ();
                /// Auto-generated documentation for Field.
                type Field = PhantomData<$t>;
                /// Auto-generated documentation for to_kindle.
                fn to_kindle(_: &Self::Field) -> KindleDType {
                    KindleDType::$repr
                }
                /// Auto-generated documentation for init.
                fn init(_: Self::Arg) -> Self::Field {
                    PhantomData
                }
            }
            impl DynDType for $t { }
            impl ConstDType for $t {
                /// Auto-generated documentation for Elem.
                type Elem = $t;
                /// Auto-generated documentation for DTYPE.
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
    /// Auto-generated documentation for Arg.
    type Arg = KindleDType;
    /// Auto-generated documentation for Field.
    type Field = KindleDType;

    /// Auto-generated documentation for init.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }

    /// Auto-generated documentation for to_kindle.
    fn to_kindle(dtype: &Self::Field) -> KindleDType {
        *dtype
    }
}
impl DynDType for Dyn {}
