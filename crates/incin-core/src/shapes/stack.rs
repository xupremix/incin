use crate::prelude::*;
use typenum::{U0, U1, U2, U3, U4, U5, U6, U7};

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

// `$pre` are the axes before the inserted one and `$post` those after it;
// `$u` is the insertion index as a `typenum`. A rank-N shape has N+1 insertion
// points, which is why the sweep emits one more invocation than the rank.
//
// This replaced 27 hand-written impls covering ranks 1 through 6. Stacking
// *adds* an axis, so its `Output` is rank N+1 and is bounded by `Shape` --
// making `MAX_RANK - 1` the correct input ceiling, not a gap.
macro_rules! impl_stack_shape {
    ($($pre:ident),* ; $($post:ident),* ; $u:ty) => {
        impl<$($pre,)* $($post,)*> StackShape<$u> for ($($pre,)* $($post,)*)
        where
            $($pre: Dim,)*
            $($post: Dim,)*
        {
            /// `Self` with a new size-2 dimension inserted at `Axis`.
            type Output = ($($pre,)* U2, $($post,)*);
        }
    };
}

incin_macros::rank_sweep!(axis_insert => impl_stack_shape, max = 7);
