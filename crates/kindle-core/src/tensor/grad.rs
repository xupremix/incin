use crate::prelude::Dyn;
use core::fmt::Debug;
use core::marker::PhantomData;

pub trait RequiresGrad: 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    type Arg;
    type Field: Clone + Debug + Send + Sync + PartialEq;
    fn requires_grad(grad: &Self::Field) -> bool;
    fn init(arg: Self::Arg) -> Self::Field;
}

pub trait DynRequiresGrad: RequiresGrad {}

pub trait ConstRequiresGrad: RequiresGrad<Arg = ()> {
    const REQUIRES_GRAD: bool;
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct Grad;
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct NoGrad;

impl RequiresGrad for Dyn {
    type Arg = bool;
    type Field = bool;
    fn requires_grad(grad: &Self::Field) -> bool {
        *grad
    }
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl RequiresGrad for Grad {
    type Arg = ();
    type Field = PhantomData<Self>;
    fn requires_grad(_: &Self::Field) -> bool {
        true
    }
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl RequiresGrad for NoGrad {
    type Arg = ();
    type Field = PhantomData<Self>;
    fn requires_grad(_: &Self::Field) -> bool {
        false
    }
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

impl DynRequiresGrad for Dyn {}
impl DynRequiresGrad for Grad {}
impl DynRequiresGrad for NoGrad {}

impl ConstRequiresGrad for Grad {
    const REQUIRES_GRAD: bool = true;
}
impl ConstRequiresGrad for NoGrad {
    const REQUIRES_GRAD: bool = false;
}
