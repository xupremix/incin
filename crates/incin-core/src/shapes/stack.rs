use crate::shapes::Shape;
use crate::shapes::idx::StaticCursor;
use typenum::U2;

#[diagnostic::on_unimplemented(
    message = "Cannot stack shape `{Self}` along axis `{Axis}`",
    label = "Invalid axis for stacking",
    note = "Stacking requires the axis to be <= the rank of the tensor"
)]
/// Compile-time-checked stacking shape rule: stacking 2 tensors of
/// shape `Self` along a new dimension inserted at `Axis` produces
/// `Output` (one rank higher, with a size-2 dimension at `Axis`).
pub trait StackShape<Axis> {
    /// `Self` with a new size-2 dimension inserted at `Axis`.
    type Output: Shape;
}

impl<S: Shape, Axis> StackShape<Axis> for S
where
    Axis: StaticCursor,
    S: crate::shapes::InsertAt<Axis, U2>,
{
    type Output = <S as crate::shapes::InsertAt<Axis, U2>>::Output;
}

// Tuple-specific generated implementations were retired. Structural callers
// use `InsertAt` through the generic implementation above.
