//! Compile-time shape reshaping and element count verification.
use crate::prelude::*;
use core::ops::Mul;
use typenum::{Prod, U1, Unsigned};

/// Computes the total number of elements in a static shape.
pub trait ElementCount {
    /// `Count`.
    type Count: typenum::Unsigned;
}

impl ElementCount for () {
    /// `Count`.
    type Count = U1;
}

impl<A: Unsigned> ElementCount for (A,) {
    /// `Count`.
    type Count = A;
}

impl<A: Unsigned, B: Unsigned> ElementCount for (A, B)
where
    A: Mul<B>,
    Prod<A, B>: Unsigned,
{
    /// `Count`.
    type Count = Prod<A, B>;
}

impl<A: Unsigned, B: Unsigned, C: Unsigned> ElementCount for (A, B, C)
where
    A: Mul<B>,
    Prod<A, B>: Mul<C>,
    Prod<Prod<A, B>, C>: Unsigned,
{
    /// `Count`.
    type Count = Prod<Prod<A, B>, C>;
}

impl<A: Unsigned, B: Unsigned, C: Unsigned, D: Unsigned> ElementCount for (A, B, C, D)
where
    A: Mul<B>,
    Prod<A, B>: Mul<C>,
    Prod<Prod<A, B>, C>: Mul<D>,
    Prod<Prod<Prod<A, B>, C>, D>: Unsigned,
{
    /// `Count`.
    type Count = Prod<Prod<Prod<A, B>, C>, D>;
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
        type S1 = (U2, U8);
        /// `S2`.
        type S2 = (U4, U4);
        assert_reshape_eq::<S1, S2>();
    }

    #[test]
    /// `reshape_different_rank_same_numel`.
    fn reshape_different_rank_same_numel() {
        /// `S1`.
        type S1 = (U2, U2, U4);
        /// `S2`.
        type S2 = (U4, U4);
        assert_reshape_eq::<S1, S2>();
    }
}
