//! Compile-time broadcasting shape verification.
use crate::prelude::*;
use crate::tensor::matmul::StaticDim;

/// Resolve one runtime (`Dyn`) broadcast dimension, panicking with a clear
/// message on incompatible sizes instead of silently fabricating a wrong
/// result via a bare `.max()`. NumPy/PyTorch broadcast rule: two dims are
/// compatible iff they're equal or one of them is 1.
///
/// Every `BroadcastShape` call site reachable through the public `Tensor` API
/// (`broadcast_add`/`sub`/`mul`/`div` and the `+`/`-`/`*`/`/` operator
/// overloads in `tensor::ops::binary`) already calls into the backend's own
/// validated `broadcast_shape` first and propagates its `Err` via `?` before
/// this value is ever used — so today this assert is not reachable from that
/// path. It exists as defense-in-depth for any future or direct caller of
/// `BroadcastShape::output_shape` that doesn't already validate independently.
#[inline]
fn checked_broadcast_dim(lhs: usize, rhs: usize) -> usize {
    assert!(
        lhs == rhs || lhs == 1 || rhs == 1,
        "cannot broadcast dynamic dimension: {lhs} vs {rhs} (dims must be equal, or one of them must be 1)"
    );
    lhs.max(rhs)
}

/// Trait that verifies two shapes are broadcastable and determines the output shape.
#[diagnostic::on_unimplemented(
    message = "Cannot broadcast shape `{Self}` to `{Rhs}`",
    label = "Shape mismatch during broadcast",
    note = "Broadcast requires dimensions to be equal, or one of them to be 1"
)]
/// Core abstraction for `BroadcastShape` within the Kindle framework..
pub trait BroadcastShape<Rhs: Shape>: Shape {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output: Shape;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Rhs as Shape>::Field,
    ) -> <Self::Output as Shape>::Field;
}

impl BroadcastShape<()> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = ();
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(_: &(), _: &()) {}
}
impl<A: StaticDim> BroadcastShape<(A,)> for (A,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A,);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for (A, B) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (A, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}

impl<A: StaticDim> BroadcastShape<(A,)> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A,);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim> BroadcastShape<()> for (A,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A,);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for (B,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (C,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for (D,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<()> for (A, B) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(B,)> for (A, B) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(B,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<()> for (A, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(C,)> for (A, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(C,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(B, C)> for (A, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<()> for (A, B, C, D) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(D,)> for (A, B, C, D) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}

impl BroadcastShape<(usize,)> for (usize,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize,);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (checked_broadcast_dim(lhs.0, rhs.0),)
    }
}
impl BroadcastShape<()> for (usize,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize,);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0,)
    }
}
impl BroadcastShape<(usize,)> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize,);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0,)
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for (usize, B) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (checked_broadcast_dim(lhs.0, rhs.0), Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<()> for (usize, B) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(B,)> for (usize, B) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(B,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for (B,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (usize, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            checked_broadcast_dim(lhs.0, rhs.0),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<()> for (usize, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(C,)> for (usize, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(C,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (C,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(B, C)> for (usize, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (
            checked_broadcast_dim(lhs.0, rhs.0),
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<()> for (usize, B, C, D) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (usize, B, C, D);
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<Dyn> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim> BroadcastShape<(A,)> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A,) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim> BroadcastShape<Dyn> for (A,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<Dyn> for (A, B) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<Dyn> for (A, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<Dyn> for (A, B, C, D) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<(usize,)> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize,) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<Dyn> for (usize,) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim> BroadcastShape<Dyn> for (usize, B) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B, C) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<Dyn> for (usize, B, C) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(usize, B, C, D)> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B, C, D) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<Dyn> for (usize, B, C, D) {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
    /// Core abstraction for `output_shape` within the Kindle framework..
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
