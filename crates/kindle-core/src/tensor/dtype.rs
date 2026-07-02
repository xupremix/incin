use crate::{candle, prelude::Dyn};

use core::{fmt::Debug, marker::PhantomData};
pub use half::{bf16, f16};

pub trait DType: 'static + Clone + Debug + Send + Sync + PartialEq {
    type Arg;
    type Field: Debug + Clone;
    type DType;
    fn init(arg: Self::Arg) -> Self::Field;
    fn dtype(dtype: &Self::Field) -> Self::DType;
}

pub trait DynDType: DType {}

pub trait ConstDType: DType<Arg = ()> {
    /// The Rust element type corresponding to this dtype.
    type Elem: 'static + Copy + Debug + Send + Sync;
    const DTYPE: <Self as DType>::DType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KindleDType {
    U8,
    U32,
    I64,
    BF16,
    F16,
    F32,
    F64,
}

macro_rules! impl_dtype {
    ($($repr:ident $t:ty),* $(,)?) => {
        $(
            impl DType for $t {
                type Arg = ();
                type Field = PhantomData<$t>;
                type DType = candle::DType;
                fn dtype(_: &Self::Field) -> Self::DType {
                    candle::DType::$repr
                }
                fn init(_: Self::Arg) -> Self::Field {
                    PhantomData
                }
            }
            impl DynDType for $t { }
            impl ConstDType for $t {
                type Elem = $t;
                const DTYPE: Self::DType = candle::DType::$repr;
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
);

impl DType for Dyn {
    type Arg = KindleDType;
    type Field = KindleDType;
    type DType = candle::DType;

    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }

    fn dtype(dtype: &Self::Field) -> Self::DType {
        match dtype {
            KindleDType::U8 => candle::DType::U8,
            KindleDType::U32 => candle::DType::U32,
            KindleDType::I64 => candle::DType::I64,
            KindleDType::BF16 => candle::DType::BF16,
            KindleDType::F16 => candle::DType::F16,
            KindleDType::F32 => candle::DType::F32,
            KindleDType::F64 => candle::DType::F64,
        }
    }
}
impl DynDType for Dyn {}
