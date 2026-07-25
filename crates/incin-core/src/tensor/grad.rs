use crate::prelude::Dyn;
use core::fmt::Debug;
use core::marker::PhantomData;

/// A type-level marker for whether a `Tensor` tracks gradients — `Grad`
/// (always tracks), `NoGrad` (never tracks), or `Dyn` (decided at runtime).
pub trait RequiresGrad: 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    /// The user-facing constructor argument (`()` for the compile-time-
    /// fixed markers, `bool` for `Dyn`).
    type Arg;
    /// The runtime-stored representation (a `PhantomData` for the
    /// compile-time-fixed markers, `bool` for `Dyn`).
    type Field: Clone + Debug + Send + Sync + PartialEq + Default;
    /// Returns whether gradient tracking is enabled.
    fn requires_grad(grad: &Self::Field) -> bool;
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field;
}

/// Marker for `RequiresGrad` implementors whose value is resolved at
/// runtime rather than fixed by the type alone (currently only `Dyn`).
pub trait DynRequiresGrad: RequiresGrad {}

/// A `RequiresGrad` whose value is fully known at compile time (`Grad` or
/// `NoGrad`, as opposed to `Dyn`) — takes no constructor argument.
pub trait ConstRequiresGrad: RequiresGrad<Arg = ()> {
    /// The compile-time-known gradient-tracking value.
    const REQUIRES_GRAD: bool;
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// Marker: this tensor always tracks gradients.
pub struct Grad;
#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// Marker: this tensor never tracks gradients.
pub struct NoGrad;

impl RequiresGrad for Dyn {
    /// The runtime-chosen gradient-tracking flag.
    type Arg = bool;
    /// Stored directly as the flag itself.
    type Field = bool;
    /// Returns the stored flag.
    fn requires_grad(grad: &Self::Field) -> bool {
        *grad
    }
    /// Stores the flag verbatim.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl RequiresGrad for Grad {
    /// No argument needed — always tracks gradients.
    type Arg = ();
    /// Zero-sized: the value is fixed by the type.
    type Field = PhantomData<Self>;
    /// Always `true`.
    fn requires_grad(_: &Self::Field) -> bool {
        true
    }
    /// No-op: nothing to convert.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl RequiresGrad for NoGrad {
    /// No argument needed — never tracks gradients.
    type Arg = ();
    /// Zero-sized: the value is fixed by the type.
    type Field = PhantomData<Self>;
    /// Always `false`.
    fn requires_grad(_: &Self::Field) -> bool {
        false
    }
    /// No-op: nothing to convert.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

impl DynRequiresGrad for Dyn {}
impl DynRequiresGrad for Grad {}
impl DynRequiresGrad for NoGrad {}

impl ConstRequiresGrad for Grad {
    /// Always `true`.
    const REQUIRES_GRAD: bool = true;
}
impl ConstRequiresGrad for NoGrad {
    /// Always `false`.
    const REQUIRES_GRAD: bool = false;
}
