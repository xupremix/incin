use crate::prelude::Dyn;
use core::fmt::Debug;
use core::marker::PhantomData;

/// Core abstraction for `RequiresGrad` within the Kindle framework..
pub trait RequiresGrad: 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field: Clone + Debug + Send + Sync + PartialEq;
    /// Core abstraction for `requires_grad` within the Kindle framework..
    fn requires_grad(grad: &Self::Field) -> bool;
    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field;
}

/// Core abstraction for `DynRequiresGrad` within the Kindle framework..
pub trait DynRequiresGrad: RequiresGrad {}

/// Core abstraction for `ConstRequiresGrad` within the Kindle framework..
pub trait ConstRequiresGrad: RequiresGrad<Arg = ()> {
    /// Core abstraction for `REQUIRES_GRAD` within the Kindle framework..
    const REQUIRES_GRAD: bool;
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// Core abstraction for `Grad` within the Kindle framework..
pub struct Grad;
#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// Core abstraction for `NoGrad` within the Kindle framework..
pub struct NoGrad;

impl RequiresGrad for Dyn {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = bool;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = bool;
    /// Core abstraction for `requires_grad` within the Kindle framework..
    fn requires_grad(grad: &Self::Field) -> bool {
        *grad
    }
    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl RequiresGrad for Grad {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = ();
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = PhantomData<Self>;
    /// Core abstraction for `requires_grad` within the Kindle framework..
    fn requires_grad(_: &Self::Field) -> bool {
        true
    }
    /// Core abstraction for `init` within the Kindle framework..
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl RequiresGrad for NoGrad {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = ();
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = PhantomData<Self>;
    /// Core abstraction for `requires_grad` within the Kindle framework..
    fn requires_grad(_: &Self::Field) -> bool {
        false
    }
    /// Core abstraction for `init` within the Kindle framework..
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

impl DynRequiresGrad for Dyn {}
impl DynRequiresGrad for Grad {}
impl DynRequiresGrad for NoGrad {}

impl ConstRequiresGrad for Grad {
    /// Core abstraction for `REQUIRES_GRAD` within the Kindle framework..
    const REQUIRES_GRAD: bool = true;
}
impl ConstRequiresGrad for NoGrad {
    /// Core abstraction for `REQUIRES_GRAD` within the Kindle framework..
    const REQUIRES_GRAD: bool = false;
}
