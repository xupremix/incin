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
/// Compile-time-checked NumPy-style broadcast shape rule: `Self`
/// broadcast against `Rhs` produces `Output`.
pub trait BroadcastShape<Rhs: Shape>: Shape {
    /// The resulting shape after broadcasting `Self` against `Rhs`.
    type Output: Shape;
    /// Computes the runtime `Field` (dimension values) of `Output`,
    /// resolving any `usize` (runtime) dimensions via `checked_broadcast_dim`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Rhs as Shape>::Field,
    ) -> <Self::Output as Shape>::Field;
}

impl BroadcastShape<()> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = ();
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(_: &(), _: &()) {}
}
impl<A: StaticDim> BroadcastShape<(A,)> for (A,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A,);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for (A, B) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (A, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}

impl<A: StaticDim> BroadcastShape<(A,)> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A,);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim> BroadcastShape<()> for (A,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A,);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for (B,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (C,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for (D,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<()> for (A, B) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(B,)> for (A, B) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(B,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for (B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<()> for (A, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(C,)> for (A, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(C,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(B, C)> for (A, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<()> for (A, B, C, D) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(D,)> for (A, B, C, D) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (A, B, C, D);
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(B, C, D) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        Default::default()
    }
}

impl BroadcastShape<(usize,)> for (usize,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize,);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (checked_broadcast_dim(lhs.0, rhs.0),)
    }
}
impl BroadcastShape<()> for (usize,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize,);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0,)
    }
}
impl BroadcastShape<(usize,)> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize,);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0,)
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for (usize, B) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (checked_broadcast_dim(lhs.0, rhs.0), Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<()> for (usize, B) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(B,)> for (usize, B) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(B,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for (B,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (usize, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(C,)> for (usize, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(C,) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (C,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize, B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (rhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(B, C)> for (usize, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(B, C) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for (B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize, B, C, D);
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
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
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<Dyn> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim> BroadcastShape<(A,)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A,) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim> BroadcastShape<Dyn> for (A,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<(A, B)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim> BroadcastShape<Dyn> for (A, B) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<(A, B, C)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B, C) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim> BroadcastShape<Dyn> for (A, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(A, B, C, D)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(A, B, C, D) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<A: StaticDim, B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<Dyn> for (A, B, C, D) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<(usize,)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize,) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<Dyn> for (usize,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim> BroadcastShape<(usize, B)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim> BroadcastShape<Dyn> for (usize, B) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<(usize, B, C)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B, C) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim> BroadcastShape<Dyn> for (usize, B, C) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<(usize, B, C, D)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize, B, C, D) as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        lhs.clone() // At runtime candle will compute output shape properly
    }
}
impl<B: StaticDim, C: StaticDim, D: StaticDim> BroadcastShape<Dyn> for (usize, B, C, D) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        rhs.clone() // At runtime candle will compute output shape properly
    }
}
