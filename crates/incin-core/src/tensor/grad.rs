use crate::exec::policy::GradMode;
use crate::prelude::Dyn;
use core::fmt::Debug;
use core::marker::PhantomData;

/// A type-level marker for whether a `Tensor` tracks gradients — `Grad`
/// (always tracks), `NoGrad` (never tracks), or `Dyn` (decided at runtime).
pub trait RequiresGrad:
    GradJoin<Self, Output = Self> + 'static + Clone + Debug + Send + Sync + Eq + PartialEq
{
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

    /// The execution-layer form of this marker (`GRD-002`).
    ///
    /// The backend that actually records an autograd node receives storage
    /// and never sees `G`, so the marker has to cross that boundary as a
    /// value. This is that value, and it is *derived* rather than supplied per
    /// impl: a marker cannot claim it tracks gradients and then decline to
    /// record, because there is only one answer and this reads it.
    fn grad_mode(grad: &Self::Field) -> GradMode {
        if Self::requires_grad(grad) {
            GradMode::Enabled
        } else {
            GradMode::Disabled
        }
    }
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

/// Type-level gradient capability join rule (`GRD-003`).
///
/// Computes the type-level OR join of two `RequiresGrad` markers:
/// - Grad OR Grad -> Grad
/// - Grad OR NoGrad -> Grad
/// - NoGrad OR Grad -> Grad
/// - NoGrad OR NoGrad -> NoGrad
/// - Grad OR Dyn -> Grad
/// - Dyn OR Grad -> Grad
/// - NoGrad OR Dyn -> Dyn
/// - Dyn OR NoGrad -> Dyn
/// - Dyn OR Dyn -> Dyn
pub trait GradJoin<Rhs: RequiresGrad>: 'static + Send + Sync {
    /// The joined gradient requirement output type.
    type Output: RequiresGrad;

    /// Combines the runtime field representation of two operands.
    fn join_field(
        lhs: &<Self as RequiresGrad>::Field,
        rhs: &Rhs::Field,
    ) -> <<Self as GradJoin<Rhs>>::Output as RequiresGrad>::Field
    where
        Self: RequiresGrad;
}

/// Type alias for joined gradient requirement of `L` and `R`.
pub type JoinedGrad<L, R> = <L as GradJoin<R>>::Output;

impl<G2: RequiresGrad> GradJoin<G2> for Grad {
    type Output = Grad;
    fn join_field(_: &PhantomData<Grad>, _: &G2::Field) -> PhantomData<Grad> {
        PhantomData
    }
}

impl<G2: RequiresGrad> GradJoin<G2> for NoGrad {
    type Output = G2;
    fn join_field(_: &PhantomData<NoGrad>, rhs: &G2::Field) -> G2::Field {
        rhs.clone()
    }
}

impl GradJoin<Grad> for Dyn {
    type Output = Grad;
    fn join_field(_: &bool, _: &PhantomData<Grad>) -> PhantomData<Grad> {
        PhantomData
    }
}

impl GradJoin<NoGrad> for Dyn {
    type Output = Dyn;
    fn join_field(lhs: &bool, _: &PhantomData<NoGrad>) -> bool {
        *lhs
    }
}

impl GradJoin<Dyn> for Dyn {
    type Output = Dyn;
    fn join_field(lhs: &bool, rhs: &bool) -> bool {
        *lhs || *rhs
    }
}
