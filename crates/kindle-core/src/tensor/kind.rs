use crate::tensor::backend::Backend;
use crate::prelude::KindleDType;

pub trait TensorKind: 'static + Send + Sync + core::fmt::Debug + Clone + Copy {
    type Primitive<B: Backend>: Clone;
    fn dtype<B: Backend>() -> KindleDType;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Float;

impl TensorKind for Float {
    type Primitive<B: Backend> = B::FloatTensorPrimitive;
    fn dtype<B: Backend>() -> KindleDType {
        <B::FloatElem as crate::prelude::DType>::to_kindle(&Default::default())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Int;

impl TensorKind for Int {
    type Primitive<B: Backend> = B::IntTensorPrimitive;
    fn dtype<B: Backend>() -> KindleDType {
        <B::IntElem as crate::prelude::DType>::to_kindle(&Default::default())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Bool;

impl TensorKind for Bool {
    type Primitive<B: Backend> = B::BoolTensorPrimitive;
    fn dtype<B: Backend>() -> KindleDType {
        <B::BoolElem as crate::prelude::DType>::to_kindle(&Default::default())
    }
}
