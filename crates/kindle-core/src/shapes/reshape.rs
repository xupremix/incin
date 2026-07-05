//! Compile-time shape reshaping and element count verification.
use crate::prelude::*;
use std::ops::Mul;
use typenum::{Prod, U1, Unsigned};

/// Computes the total number of elements in a static shape.
pub trait ElementCount {
    type Count: typenum::Unsigned;
}

impl ElementCount for () {
    type Count = U1;
}

impl<A: Unsigned> ElementCount for (A,) {
    type Count = A;
}

impl<A: Unsigned, B: Unsigned> ElementCount for (A, B)
where
    A: Mul<B>,
    Prod<A, B>: Unsigned,
{
    type Count = Prod<A, B>;
}

impl<A: Unsigned, B: Unsigned, C: Unsigned> ElementCount for (A, B, C)
where
    A: Mul<B>,
    Prod<A, B>: Mul<C>,
    Prod<Prod<A, B>, C>: Unsigned,
{
    type Count = Prod<Prod<A, B>, C>;
}

impl<A: Unsigned, B: Unsigned, C: Unsigned, D: Unsigned> ElementCount for (A, B, C, D)
where
    A: Mul<B>,
    Prod<A, B>: Mul<C>,
    Prod<Prod<A, B>, C>: Mul<D>,
    Prod<Prod<Prod<A, B>, C>, D>: Unsigned,
{
    type Count = Prod<Prod<Prod<A, B>, C>, D>;
}

/// A trait that guarantees two shapes have the exact same number of elements at compile-time.
pub trait ReshapeShape<Target: Shape>: Shape {}

// Blanket implementation for any two static shapes that share the exact same ElementCount.
impl<S1, S2> ReshapeShape<S2> for S1
where
    S1: Shape + ElementCount,
    S2: Shape + ElementCount<Count = <S1 as ElementCount>::Count>,
{}

/// A hybrid trait for dynamic and partial dynamic reshaping.
pub trait TryReshape<Target: Shape>: Shape {}

// Any pair of dynamic shapes can attempt to reshape at runtime.
impl<S1: DynShape, S2: DynShape> TryReshape<S2> for S1 {}

