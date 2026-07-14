use crate::prelude::Dyn;

use core::{fmt::Debug, marker::PhantomData};
pub use half::{bf16, f16};

pub trait DType: 'static + Clone + Debug + Send + Sync + PartialEq {
    type Arg;
    type Field: Debug + Clone;
    fn init(arg: Self::Arg) -> Self::Field;
    fn to_kindle(dtype: &Self::Field) -> KindleDType;
}

pub trait DynDType: DType {}

pub trait FloatDType: DType {}
pub trait IntDType: DType {}
pub trait BoolDType: DType {}

pub trait ConstDType: DType<Arg = ()> {
    /// The Rust element type corresponding to this dtype.
    type Elem: 'static + Copy + Debug + Send + Sync;
    const DTYPE: KindleDType;
}

pub trait QuantDType: DType {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Q8_0;


#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KindleDType {
    U8,
    U32,
    I64,
    BF16,
    F16,
    F32,
    F64,
    Q8_0,
}

macro_rules! impl_dtype {
    ($($repr:ident $t:ty),* $(,)?) => {
        $(
            impl DType for $t {
                type Arg = ();
                type Field = PhantomData<$t>;
                fn to_kindle(_: &Self::Field) -> KindleDType {
                    KindleDType::$repr
                }
                fn init(_: Self::Arg) -> Self::Field {
                    PhantomData
                }
            }
            impl DynDType for $t { }
            impl ConstDType for $t {
                type Elem = $t;
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
    type Arg = KindleDType;
    type Field = KindleDType;

    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }

    fn to_kindle(dtype: &Self::Field) -> KindleDType {
        *dtype
    }
}
impl DynDType for Dyn {}
