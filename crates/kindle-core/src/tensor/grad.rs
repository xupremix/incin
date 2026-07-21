use crate::prelude::Dyn;
use core::fmt::Debug;
use core::marker::PhantomData;

/// `RequiresGrad`.
pub trait RequiresGrad: 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    /// `Arg`.
    type Arg;
    /// `Field`.
    type Field: Clone + Debug + Send + Sync + PartialEq;
    /// `requires_grad`.
    fn requires_grad(grad: &Self::Field) -> bool;
    /// `init`.
    fn init(arg: Self::Arg) -> Self::Field;
}

/// `DynRequiresGrad`.
pub trait DynRequiresGrad: RequiresGrad {}

/// `ConstRequiresGrad`.
pub trait ConstRequiresGrad: RequiresGrad<Arg = ()> {
    /// `REQUIRES_GRAD`.
    const REQUIRES_GRAD: bool;
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// `Grad`.
pub struct Grad;
#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// `NoGrad`.
pub struct NoGrad;

impl RequiresGrad for Dyn {
    /// `Arg`.
    type Arg = bool;
    /// `Field`.
    type Field = bool;
    /// `requires_grad`.
    fn requires_grad(grad: &Self::Field) -> bool {
        *grad
    }
    /// `init`.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl RequiresGrad for Grad {
    /// `Arg`.
    type Arg = ();
    /// `Field`.
    type Field = PhantomData<Self>;
    /// `requires_grad`.
    fn requires_grad(_: &Self::Field) -> bool {
        true
    }
    /// `init`.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl RequiresGrad for NoGrad {
    /// `Arg`.
    type Arg = ();
    /// `Field`.
    type Field = PhantomData<Self>;
    /// `requires_grad`.
    fn requires_grad(_: &Self::Field) -> bool {
        false
    }
    /// `init`.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

impl DynRequiresGrad for Dyn {}
impl DynRequiresGrad for Grad {}
impl DynRequiresGrad for NoGrad {}

impl ConstRequiresGrad for Grad {
    /// `REQUIRES_GRAD`.
    const REQUIRES_GRAD: bool = true;
}
impl ConstRequiresGrad for NoGrad {
    /// `REQUIRES_GRAD`.
    const REQUIRES_GRAD: bool = false;
}
