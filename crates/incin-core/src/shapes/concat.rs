use crate::shapes::Dyn;
use crate::shapes::broadcast::ReverseShape;
use crate::shapes::dim::Dim;
use crate::shapes::idx::FromEnd;
use crate::shapes::shape::{DimCons, Shape};
use core::ops::Add;

#[diagnostic::on_unimplemented(
    message = "Cannot concatenate shape `{Self}` with `{S2}` along axis `{Axis}`",
    label = "Shape mismatch during concatenation",
    note = "Concatenation requires all dimensions except the given axis to match exactly"
)]
/// Compile-time-checked concatenation shape rule: concatenating `Self`
/// with `S2` along dimension `Axis` produces `Output` (every other
/// dimension must already match, per the diagnostic below).
pub trait ConcatShape<S2, Axis> {
    /// The resulting shape after concatenating `Self` with `S2` along `Axis`.
    type Output: Shape;
}

/// Fallback concatenation rule for shape pairs with no compile-time-known
/// axis-sum relationship: always resolves to `Dyn`, deferring the actual
/// dimension check to runtime.
pub trait TryConcatShape<S2> {
    /// Always `Dyn` — the concrete size is only known at runtime.
    type Output: Shape;
}

impl<S1: Shape, S2: Shape> TryConcatShape<S2> for S1 {
    /// Always `Dyn` — the concrete size is only known at runtime.
    type Output = Dyn;
}

impl<H1: Dim + Add<H2>, H2: Dim, T: Shape> ConcatShape<DimCons<H2, T>, crate::shapes::idx::Here>
    for DimCons<H1, T>
where
    <H1 as Add<H2>>::Output: Dim,
{
    type Output = DimCons<<H1 as Add<H2>>::Output, T>;
}

impl<H: Dim, T1: Shape, T2: Shape, SubCursor>
    ConcatShape<DimCons<H, T2>, crate::shapes::idx::Next<SubCursor>> for DimCons<H, T1>
where
    T1: ConcatShape<T2, SubCursor>,
{
    type Output = DimCons<H, <T1 as ConcatShape<T2, SubCursor>>::Output>;
}

impl<S, S2, Cursor, SR, S2R, O, Output> ConcatShape<S2, FromEnd<Cursor>> for S
where
    S: Shape + ReverseShape<Output = SR>,
    S2: Shape + ReverseShape<Output = S2R>,
    Cursor: crate::shapes::shape::ForwardCursor,
    SR: ConcatShape<S2R, Cursor, Output = O>,
    O: Shape + ReverseShape<Output = Output>,
    Output: Shape,
{
    type Output = Output;
}

// Tuple-specific generated implementations were retired. Structural callers
// use the recursive `DimCons` implementation above.
