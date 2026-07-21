//! Compile-time shape reshaping and element count verification.
use crate::prelude::*;
use core::ops::Mul;
use typenum::{Prod, U1, Unsigned};

/// Computes the total number of elements in a static shape.
pub trait ElementCount {
    /// Core abstraction for `Count` within the Kindle framework..
    type Count: typenum::Unsigned;
}

impl ElementCount for () {
    /// Core abstraction for `Count` within the Kindle framework..
    type Count = U1;
}

impl<A: Unsigned> ElementCount for (A,) {
    /// Core abstraction for `Count` within the Kindle framework..
    type Count = A;
}

impl<A: Unsigned, B: Unsigned> ElementCount for (A, B)
where
    A: Mul<B>,
    Prod<A, B>: Unsigned,
{
    /// Core abstraction for `Count` within the Kindle framework..
    type Count = Prod<A, B>;
}

impl<A: Unsigned, B: Unsigned, C: Unsigned> ElementCount for (A, B, C)
where
    A: Mul<B>,
    Prod<A, B>: Mul<C>,
    Prod<Prod<A, B>, C>: Unsigned,
{
    /// Core abstraction for `Count` within the Kindle framework..
    type Count = Prod<Prod<A, B>, C>;
}

impl<A: Unsigned, B: Unsigned, C: Unsigned, D: Unsigned> ElementCount for (A, B, C, D)
where
    A: Mul<B>,
    Prod<A, B>: Mul<C>,
    Prod<Prod<A, B>, C>: Mul<D>,
    Prod<Prod<Prod<A, B>, C>, D>: Unsigned,
{
    /// Core abstraction for `Count` within the Kindle framework..
    type Count = Prod<Prod<Prod<A, B>, C>, D>;
}

/// A trait that guarantees two shapes have the exact same number of elements at compile-time.
#[diagnostic::on_unimplemented(
    message = "Cannot reshape from `{Self}` to `{Target}`",
    label = "Element count mismatch for reshape",
    note = "Reshape requires the total number of elements to remain constant"
)]
/// Core abstraction for `ReshapeShape` within the Kindle framework..
pub trait ReshapeShape<Target: Shape>: Shape {}

// Blanket implementation for any two static shapes that share the exact same ElementCount.
impl<S1, S2> ReshapeShape<S2> for S1
where
    S1: Shape + ElementCount,
    S2: Shape + ElementCount<Count = <S1 as ElementCount>::Count>,
{
}

/// A hybrid trait for dynamic and partial dynamic reshaping.
pub trait TryReshape<Target: Shape>: Shape {}

// Any pair of dynamic shapes can attempt to reshape at runtime.
impl<S1: DynShape, S2: DynShape> TryReshape<S2> for S1 {}

#[cfg(test)]
/// Core abstraction for `tests` within the Kindle framework..
mod tests {
    use super::*;
    use typenum::{U2, U4, U8};

    /// Core abstraction for `assert_reshape_eq` within the Kindle framework..
    fn assert_reshape_eq<S1, S2: Shape>()
    where
        S1: Shape + ReshapeShape<S2>,
    {
    }

    #[test]
    /// Core abstraction for `reshape_same_rank_same_numel` within the Kindle framework..
    fn reshape_same_rank_same_numel() {
        /// Core abstraction for `S1` within the Kindle framework..
        type S1 = (U2, U8);
        /// Core abstraction for `S2` within the Kindle framework..
        type S2 = (U4, U4);
        assert_reshape_eq::<S1, S2>();
    }

    #[test]
    /// Core abstraction for `reshape_different_rank_same_numel` within the Kindle framework..
    fn reshape_different_rank_same_numel() {
        /// Core abstraction for `S1` within the Kindle framework..
        type S1 = (U2, U2, U4);
        /// Core abstraction for `S2` within the Kindle framework..
        type S2 = (U4, U4);
        assert_reshape_eq::<S1, S2>();
    }
}
