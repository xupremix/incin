//! Compile-time broadcasting shape verification.
use crate::prelude::*;
use crate::tensor::matmul::StaticDim;

/// Trait that verifies two shapes are broadcastable and determines the output shape.
#[diagnostic::on_unimplemented(
    message = "Cannot broadcast shape `{Self}` to `{Rhs}`",
    label = "Shape mismatch during broadcast",
    note = "Broadcast requires dimensions to be equal, or one of them to be 1"
)]
/// Auto-generated documentation for BroadcastShape.
pub trait BroadcastShape<Rhs: Shape>: Shape {
    /// Auto-generated documentation for Output.
    type Output: Shape;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Rhs as Shape>::Field,
    ) -> <Self::Output as Shape>::Field;
}

impl BroadcastShape<()> for () {
    /// Auto-generated documentation for Output.
    type Output = ();
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(_: &(), _: &()) {}
}
impl<A: StaticDim> BroadcastShape<(A,)> for (A,) {
    /// Auto-generated documentation for Output.
    type Output = (A,);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for (A, B) {
    /// Auto-generated documentation for Output.
    type Output = (A, B);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (A, B, C) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)>
    for (A, B, C, D)
{
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}

impl<A: StaticDim> BroadcastShape<(A,)> for () {
    /// Auto-generated documentation for Output.
    type Output = (A,);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for () {
    /// Auto-generated documentation for Output.
    type Output = (A, B);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for () {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for () {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim> BroadcastShape<()> for (A,) {
    /// Auto-generated documentation for Output.
    type Output = (A,);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for (B,) {
    /// Auto-generated documentation for Output.
    type Output = (A, B);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (C,) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for (D,) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<()> for (A, B) {
    /// Auto-generated documentation for Output.
    type Output = (A, B);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(B,)> for (A, B) {
    /// Auto-generated documentation for Output.
    type Output = (A, B);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(B,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (B, C) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)>
    for (C, D)
{
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<()> for (A, B, C) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(C,)> for (A, B, C) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(C,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(B, C)> for (A, B, C) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)>
    for (B, C, D)
{
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<()> for (A, B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(D,)> for (A, B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(D,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(C, D)>
    for (A, B, C, D)
{
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(B, C, D)>
    for (A, B, C, D)
{
    /// Auto-generated documentation for Output.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}

impl BroadcastShape<(usize,)> for (usize,) {
    /// Auto-generated documentation for Output.
    type Output = (usize,);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0.max(rhs.0),)
    }
}
impl BroadcastShape<()> for (usize,) {
    /// Auto-generated documentation for Output.
    type Output = (usize,);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0,)
    }
}
impl BroadcastShape<(usize,)> for () {
    /// Auto-generated documentation for Output.
    type Output = (usize,);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0,)
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for (usize, B) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0.max(rhs.0), Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<()> for (usize, B) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for () {
    /// Auto-generated documentation for Output.
    type Output = (usize, B);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(B,)> for (usize, B) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(B,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for (B,) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (usize, B, C) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0.max(rhs.0), Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<()> for (usize, B, C) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for () {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(C,)> for (usize, B, C) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(C,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (C,) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(B, C)> for (usize, B, C) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (B, C) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(usize, B, C, D)>
    for (usize, B, C, D)
{
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            lhs.0.max(rhs.0),
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<()> for (usize, B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            lhs.0,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(usize, B, C, D)> for () {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            rhs.0,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(D,)> for (usize, B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(D,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            lhs.0,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(usize, B, C, D)> for (D,) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            rhs.0,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(C, D)> for (usize, B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            lhs.0,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(usize, B, C, D)> for (C, D) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            rhs.0,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(B, C, D)> for (usize, B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            lhs.0,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(usize, B, C, D)> for (B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = (usize, B, C, D);
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            rhs.0,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}

impl BroadcastShape<Dyn> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        // Simplistic dynamic shape broadcasting computation
        let mut out = alloc::vec![];
        let max_len = lhs.len().max(rhs.len());
        for i in 0..max_len {
            let l = if i < max_len - lhs.len() {
                1
            } else {
                lhs[i - (max_len - lhs.len())]
            };
            let r = if i < max_len - rhs.len() {
                1
            } else {
                rhs[i - (max_len - rhs.len())]
            };
            out.push(l.max(r));
        }
        out
    }
}
impl BroadcastShape<()> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<Dyn> for () {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim> BroadcastShape<(A,)> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A,) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim> BroadcastShape<Dyn> for (A,) {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<Dyn> for (A, B) {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<Dyn> for (A, B, C) {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<Dyn> for (A, B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<(usize,)> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize,) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<Dyn> for (usize,) {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim> BroadcastShape<Dyn> for (usize, B) {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B, C) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<Dyn> for (usize, B, C) {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(usize, B, C, D)> for Dyn {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B, C, D) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<Dyn> for (usize, B, C, D) {
    /// Auto-generated documentation for Output.
    type Output = Dyn;
    /// Auto-generated documentation for output_shape.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
