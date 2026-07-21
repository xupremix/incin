use crate::prelude::Dyn;
use core::fmt::Debug;
use core::marker::PhantomData;

/// Auto-generated documentation for RequiresGrad.
pub trait RequiresGrad: 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    /// Auto-generated documentation for Arg.
    type Arg;
    /// Auto-generated documentation for Field.
    type Field: Clone + Debug + Send + Sync + PartialEq;
    /// Auto-generated documentation for requires_grad.
    fn requires_grad(grad: &Self::Field) -> bool;
    /// Auto-generated documentation for init.
    fn init(arg: Self::Arg) -> Self::Field;
}

/// Auto-generated documentation for DynRequiresGrad.
pub trait DynRequiresGrad: RequiresGrad {}

/// Auto-generated documentation for ConstRequiresGrad.
pub trait ConstRequiresGrad: RequiresGrad<Arg = ()> {
    /// Auto-generated documentation for REQUIRES_GRAD.
    const REQUIRES_GRAD: bool;
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// Auto-generated documentation for Grad.
pub struct Grad;
#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// Auto-generated documentation for NoGrad.
pub struct NoGrad;

impl RequiresGrad for Dyn {
    /// Auto-generated documentation for Arg.
    type Arg = bool;
    /// Auto-generated documentation for Field.
    type Field = bool;
    /// Auto-generated documentation for requires_grad.
    fn requires_grad(grad: &Self::Field) -> bool {
        *grad
    }
    /// Auto-generated documentation for init.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl RequiresGrad for Grad {
    /// Auto-generated documentation for Arg.
    type Arg = ();
    /// Auto-generated documentation for Field.
    type Field = PhantomData<Self>;
    /// Auto-generated documentation for requires_grad.
    fn requires_grad(_: &Self::Field) -> bool {
        true
    }
    /// Auto-generated documentation for init.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl RequiresGrad for NoGrad {
    /// Auto-generated documentation for Arg.
    type Arg = ();
    /// Auto-generated documentation for Field.
    type Field = PhantomData<Self>;
    /// Auto-generated documentation for requires_grad.
    fn requires_grad(_: &Self::Field) -> bool {
        false
    }
    /// Auto-generated documentation for init.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

impl DynRequiresGrad for Dyn {}
impl DynRequiresGrad for Grad {}
impl DynRequiresGrad for NoGrad {}

impl ConstRequiresGrad for Grad {
    /// Auto-generated documentation for REQUIRES_GRAD.
    const REQUIRES_GRAD: bool = true;
}
impl ConstRequiresGrad for NoGrad {
    /// Auto-generated documentation for REQUIRES_GRAD.
    const REQUIRES_GRAD: bool = false;
}
