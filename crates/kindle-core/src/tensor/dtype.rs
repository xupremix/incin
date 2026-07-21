use crate::prelude::Dyn;

use core::{fmt::Debug, marker::PhantomData};
pub use half::{bf16, f16};

/// Core abstraction for `DType` within the Kindle framework..
pub trait DType: 'static + Clone + Debug + Send + Sync + PartialEq {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field: Debug + Clone;
    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field;
    /// Core abstraction for `to_kindle` within the Kindle framework..
    fn to_kindle(dtype: &Self::Field) -> KindleDType;
}

/// Core abstraction for `DynDType` within the Kindle framework..
pub trait DynDType: DType {}

/// Core abstraction for `FloatDType` within the Kindle framework..
pub trait FloatDType: DType {}
/// Core abstraction for `IntDType` within the Kindle framework..
pub trait IntDType: DType {}
/// Core abstraction for `BoolDType` within the Kindle framework..
pub trait BoolDType: DType {}

/// Core abstraction for `ConstDType` within the Kindle framework..
pub trait ConstDType: DType<Arg = ()> {
    /// The Rust element type corresponding to this dtype.
    type Elem: 'static + Copy + Debug + Send + Sync;
    /// Core abstraction for `DTYPE` within the Kindle framework..
    const DTYPE: KindleDType;
}

/// Core abstraction for `QuantDType` within the Kindle framework..
pub trait QuantDType: DType {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Core abstraction for `Q8_0` within the Kindle framework..
pub struct Q8_0;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Core abstraction for `KindleDType` within the Kindle framework..
pub enum KindleDType {
    /// Core abstraction for `U8` within the Kindle framework..
    U8,
    /// Core abstraction for `U32` within the Kindle framework..
    U32,
    /// Core abstraction for `I64` within the Kindle framework..
    I64,
    /// Core abstraction for `BF16` within the Kindle framework..
    BF16,
    /// Core abstraction for `F16` within the Kindle framework..
    F16,
    /// Core abstraction for `F32` within the Kindle framework..
    F32,
    /// Core abstraction for `F64` within the Kindle framework..
    F64,
    /// Core abstraction for `Q8_0` within the Kindle framework..
    Q8_0,
}

macro_rules! impl_dtype {
    ($($repr:ident $t:ty),* $(,)?) => {
        $(
            impl DType for $t {
                /// Core abstraction for `Arg` within the Kindle framework..
                type Arg = ();
                /// Core abstraction for `Field` within the Kindle framework..
                type Field = PhantomData<$t>;
                /// Core abstraction for `to_kindle` within the Kindle framework..
                fn to_kindle(_: &Self::Field) -> KindleDType {
                    KindleDType::$repr
                }
                /// Core abstraction for `init` within the Kindle framework..
                fn init(_: Self::Arg) -> Self::Field {
                    PhantomData
                }
            }
            impl DynDType for $t { }
            impl ConstDType for $t {
                /// Core abstraction for `Elem` within the Kindle framework..
                type Elem = $t;
                /// Core abstraction for `DTYPE` within the Kindle framework..
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
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = KindleDType;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = KindleDType;

    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }

    /// Core abstraction for `to_kindle` within the Kindle framework..
    fn to_kindle(dtype: &Self::Field) -> KindleDType {
        *dtype
    }
}
impl DynDType for Dyn {}
