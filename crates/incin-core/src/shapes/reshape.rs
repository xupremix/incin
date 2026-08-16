//! Compile-time shape reshaping and element count verification.
use crate::shapes::shape::{ShapeArgs, ShapeSpec};
use crate::shapes::{DynShape, Shape, ShapeValue};
use core::ops::Mul;
use typenum::{Prod, U1, Unsigned};

/// Computes the total number of elements in a static shape.
pub trait ElementCount {
    /// `Count`.
    type Count: typenum::Unsigned;
}

impl ElementCount for crate::shapes::Nil {
    type Count = U1;
}

/// Type-level product fold for the canonical recursive shape representation.
/// This is the arbitrary-rank counterpart of the old tuple ladder.
impl<H, T> ElementCount for crate::shapes::DimCons<H, T>
where
    H: crate::shapes::ConcreteStaticExtent,
    T: crate::shapes::Shape + ElementCount,
    H::Nat: Mul<<T as ElementCount>::Count>,
    Prod<H::Nat, <T as ElementCount>::Count>: Unsigned,
{
    type Count = Prod<H::Nat, <T as ElementCount>::Count>;
}

/// Witness that two type-level element counts are identical. Implemented
/// reflexively — only `SameCount<N>` for the *same* `N` exists — so
/// requiring `A: SameCount<B>` is a compile-time assertion that `A == B`,
/// but one that fails as an unsatisfied trait bound (E0277) rather than an
/// associated-type projection mismatch (E0271). `#[diagnostic::
/// on_unimplemented]` can only decorate E0277; routing the comparison
/// through this bound (instead of `ElementCount<Count = ...>` equality
/// directly) is what makes the reshape mismatch below render as a labeled,
/// call-site-anchored message instead of a raw `UInt<...>` projection wall.
#[diagnostic::on_unimplemented(
    message = "Cannot reshape: source has {Self} elements but the target shape has {Rhs} elements",
    label = "element count changes here",
    note = "reshape must preserve the total number of elements"
)]
pub trait SameCount<Rhs> {}
impl<N> SameCount<N> for N {}

/// A trait that guarantees two shapes have the exact same number of elements at compile-time.
#[diagnostic::on_unimplemented(
    message = "Cannot reshape from `{Self}` to `{Target}`",
    label = "element count mismatch for reshape",
    note = "reshape requires the total number of elements to remain constant"
)]
/// `ReshapeShape`.
pub trait ReshapeShape<Target: Shape>: Shape {}

// Blanket implementation for any two static shapes that share the exact same ElementCount.
impl<S1, S2> ReshapeShape<S2> for S1
where
    S1: Shape + ElementCount,
    S2: Shape + ElementCount,
    <S1 as ElementCount>::Count: SameCount<<S2 as ElementCount>::Count>,
{
}

/// A hybrid trait for dynamic and partial dynamic reshaping.
pub trait TryReshape<Target: Shape>: Shape {}

// Any pair of dynamic shapes can attempt to reshape at runtime.
impl<S1: DynShape, S2: DynShape> TryReshape<S2> for S1 {}

/// Shape specifications accepted by the ergonomic [`Tensor::reshape`](crate::tensor::Tensor::reshape)
/// API.
///
/// Fully static [`ShapeValue`] inputs retain the compile-time element-count
/// proof. Runtime [`ShapeArgs`], arrays, and vectors use the checked runtime
/// path because their dimensions are not known to the type system.
pub trait ReshapeSpec<Source: Shape>: ShapeSpec {}

impl<Source, Target> ReshapeSpec<Source> for ShapeValue<Target>
where
    Source: Shape + ReshapeShape<Target>,
    Target: Shape + DynShape,
{
}

impl<Source, Target> ReshapeSpec<Source> for ShapeArgs<Target>
where
    Source: DynShape,
    Target: Shape + DynShape,
{
}

impl<Source, const N: usize> ReshapeSpec<Source> for [usize; N] where Source: DynShape {}

impl<Source> ReshapeSpec<Source> for alloc::vec::Vec<usize> where Source: DynShape {}

impl<Source> ReshapeSpec<Source> for &[usize] where Source: DynShape {}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use typenum::{U2, U4, U8};

    /// `assert_reshape_eq`.
    fn assert_reshape_eq<S1, S2: Shape>()
    where
        S1: Shape + ReshapeShape<S2>,
    {
    }

    #[test]
    /// `reshape_same_rank_same_numel`.
    fn reshape_same_rank_same_numel() {
        /// `S1`.
        type S1 = crate::shapes::DimCons<U2, crate::shapes::DimCons<U8, crate::shapes::Nil>>;
        /// `S2`.
        type S2 = crate::shapes::DimCons<U4, crate::shapes::DimCons<U4, crate::shapes::Nil>>;
        assert_reshape_eq::<S1, S2>();
    }

    #[test]
    /// `reshape_different_rank_same_numel`.
    fn reshape_different_rank_same_numel() {
        /// `S1`.
        type S1 = crate::shapes::DimCons<
            U2,
            crate::shapes::DimCons<U2, crate::shapes::DimCons<U4, crate::shapes::Nil>>,
        >;
        /// `S2`.
        type S2 = crate::shapes::DimCons<U4, crate::shapes::DimCons<U4, crate::shapes::Nil>>;
        assert_reshape_eq::<S1, S2>();
    }
}
