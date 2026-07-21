use crate::prelude::*;
use core::ops::Add;
use typenum::{U0, U1, U2, U3, U4, U5};

#[diagnostic::on_unimplemented(
    message = "Cannot concatenate shape `{Self}` with `{S2}` along axis `{Axis}`",
    label = "Shape mismatch during concatenation",
    note = "Concatenation requires all dimensions except the given axis to match exactly"
)]
/// Auto-generated documentation for ConcatShape.
pub trait ConcatShape<S2, Axis> {
    /// Auto-generated documentation for Output.
    type Output: Shape;
}

/// Auto-generated documentation for TryConcatShape.
pub trait TryConcatShape<S2> {
    /// Auto-generated documentation for Output.
    type Output: Shape;
}

impl<S1: Shape, S2: Shape> TryConcatShape<S2> for S1 {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
}

impl<D0, D0_> ConcatShape<(D0_,), U0> for (D0,)
where
    D0: Dim,
    D0_: Dim,
    D0: Add<D0_>,
    <D0 as Add<D0_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (<D0 as Add<D0_>>::Output,);
}

impl<D0, D1, D0_> ConcatShape<(D0_, D1), U0> for (D0, D1)
where
    D0: Dim,
    D1: Dim,
    D0_: Dim,
    D0: Add<D0_>,
    <D0 as Add<D0_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (<D0 as Add<D0_>>::Output, D1);
}

impl<D0, D1, D1_> ConcatShape<(D0, D1_), U1> for (D0, D1)
where
    D0: Dim,
    D1: Dim,
    D1_: Dim,
    D1: Add<D1_>,
    <D1 as Add<D1_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, <D1 as Add<D1_>>::Output);
}

impl<D0, D1, D2, D0_> ConcatShape<(D0_, D1, D2), U0> for (D0, D1, D2)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D0_: Dim,
    D0: Add<D0_>,
    <D0 as Add<D0_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (<D0 as Add<D0_>>::Output, D1, D2);
}

impl<D0, D1, D2, D1_> ConcatShape<(D0, D1_, D2), U1> for (D0, D1, D2)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D1_: Dim,
    D1: Add<D1_>,
    <D1 as Add<D1_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, <D1 as Add<D1_>>::Output, D2);
}

impl<D0, D1, D2, D2_> ConcatShape<(D0, D1, D2_), U2> for (D0, D1, D2)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D2_: Dim,
    D2: Add<D2_>,
    <D2 as Add<D2_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, <D2 as Add<D2_>>::Output);
}

impl<D0, D1, D2, D3, D0_> ConcatShape<(D0_, D1, D2, D3), U0> for (D0, D1, D2, D3)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D0_: Dim,
    D0: Add<D0_>,
    <D0 as Add<D0_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (<D0 as Add<D0_>>::Output, D1, D2, D3);
}

impl<D0, D1, D2, D3, D1_> ConcatShape<(D0, D1_, D2, D3), U1> for (D0, D1, D2, D3)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D1_: Dim,
    D1: Add<D1_>,
    <D1 as Add<D1_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, <D1 as Add<D1_>>::Output, D2, D3);
}

impl<D0, D1, D2, D3, D2_> ConcatShape<(D0, D1, D2_, D3), U2> for (D0, D1, D2, D3)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D2_: Dim,
    D2: Add<D2_>,
    <D2 as Add<D2_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, <D2 as Add<D2_>>::Output, D3);
}

impl<D0, D1, D2, D3, D3_> ConcatShape<(D0, D1, D2, D3_), U3> for (D0, D1, D2, D3)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D3_: Dim,
    D3: Add<D3_>,
    <D3 as Add<D3_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, D2, <D3 as Add<D3_>>::Output);
}

impl<D0, D1, D2, D3, D4, D0_> ConcatShape<(D0_, D1, D2, D3, D4), U0> for (D0, D1, D2, D3, D4)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D0_: Dim,
    D0: Add<D0_>,
    <D0 as Add<D0_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (<D0 as Add<D0_>>::Output, D1, D2, D3, D4);
}

impl<D0, D1, D2, D3, D4, D1_> ConcatShape<(D0, D1_, D2, D3, D4), U1> for (D0, D1, D2, D3, D4)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D1_: Dim,
    D1: Add<D1_>,
    <D1 as Add<D1_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, <D1 as Add<D1_>>::Output, D2, D3, D4);
}

impl<D0, D1, D2, D3, D4, D2_> ConcatShape<(D0, D1, D2_, D3, D4), U2> for (D0, D1, D2, D3, D4)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D2_: Dim,
    D2: Add<D2_>,
    <D2 as Add<D2_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, <D2 as Add<D2_>>::Output, D3, D4);
}

impl<D0, D1, D2, D3, D4, D3_> ConcatShape<(D0, D1, D2, D3_, D4), U3> for (D0, D1, D2, D3, D4)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D3_: Dim,
    D3: Add<D3_>,
    <D3 as Add<D3_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, D2, <D3 as Add<D3_>>::Output, D4);
}

impl<D0, D1, D2, D3, D4, D4_> ConcatShape<(D0, D1, D2, D3, D4_), U4> for (D0, D1, D2, D3, D4)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D4_: Dim,
    D4: Add<D4_>,
    <D4 as Add<D4_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, D2, D3, <D4 as Add<D4_>>::Output);
}

impl<D0, D1, D2, D3, D4, D5, D0_> ConcatShape<(D0_, D1, D2, D3, D4, D5), U0>
    for (D0, D1, D2, D3, D4, D5)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D5: Dim,
    D0_: Dim,
    D0: Add<D0_>,
    <D0 as Add<D0_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (<D0 as Add<D0_>>::Output, D1, D2, D3, D4, D5);
}

impl<D0, D1, D2, D3, D4, D5, D1_> ConcatShape<(D0, D1_, D2, D3, D4, D5), U1>
    for (D0, D1, D2, D3, D4, D5)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D5: Dim,
    D1_: Dim,
    D1: Add<D1_>,
    <D1 as Add<D1_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, <D1 as Add<D1_>>::Output, D2, D3, D4, D5);
}

impl<D0, D1, D2, D3, D4, D5, D2_> ConcatShape<(D0, D1, D2_, D3, D4, D5), U2>
    for (D0, D1, D2, D3, D4, D5)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D5: Dim,
    D2_: Dim,
    D2: Add<D2_>,
    <D2 as Add<D2_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, <D2 as Add<D2_>>::Output, D3, D4, D5);
}

impl<D0, D1, D2, D3, D4, D5, D3_> ConcatShape<(D0, D1, D2, D3_, D4, D5), U3>
    for (D0, D1, D2, D3, D4, D5)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D5: Dim,
    D3_: Dim,
    D3: Add<D3_>,
    <D3 as Add<D3_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, D2, <D3 as Add<D3_>>::Output, D4, D5);
}

impl<D0, D1, D2, D3, D4, D5, D4_> ConcatShape<(D0, D1, D2, D3, D4_, D5), U4>
    for (D0, D1, D2, D3, D4, D5)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D5: Dim,
    D4_: Dim,
    D4: Add<D4_>,
    <D4 as Add<D4_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, D2, D3, <D4 as Add<D4_>>::Output, D5);
}

impl<D0, D1, D2, D3, D4, D5, D5_> ConcatShape<(D0, D1, D2, D3, D4, D5_), U5>
    for (D0, D1, D2, D3, D4, D5)
where
    D0: Dim,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    D4: Dim,
    D5: Dim,
    D5_: Dim,
    D5: Add<D5_>,
    <D5 as Add<D5_>>::Output: Dim,
{
    /// Auto-generated documentation for Output.
    type Output = (D0, D1, D2, D3, D4, <D5 as Add<D5_>>::Output);
}
