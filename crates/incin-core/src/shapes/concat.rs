use crate::prelude::*;
use core::ops::Add;
use typenum::{U0, U1, U2, U3, U4, U5, U6, U7};

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

// `$pre` are the axes before the concatenation axis, `$ax` is that axis, and
// `$post` are the axes after it. Only `$ax` changes: it becomes the sum of both
// operands' sizes along it. `$u` is the axis index as a `typenum`.
//
// This replaced 21 hand-written impls covering ranks 1 through 6. The rule is
// rank-preserving, so its ceiling is `MAX_RANK`.
macro_rules! impl_concat_shape {
    ($($pre:ident),* ; $ax:ident ; $($post:ident),* ; $u:ty) => {
        impl<$($pre,)* $ax, $($post,)* Rhs> ConcatShape<($($pre,)* Rhs, $($post,)*), $u>
            for ($($pre,)* $ax, $($post,)*)
        where
            $($pre: Dim,)*
            $ax: Dim,
            $($post: Dim,)*
            Rhs: Dim,
            $ax: Add<Rhs>,
            <$ax as Add<Rhs>>::Output: Dim,
        {
            /// The concatenated shape: the target axis becomes the sum of both
            /// operands' sizes along it, every other dimension unchanged.
            type Output = ($($pre,)* <$ax as Add<Rhs>>::Output, $($post,)*);
        }
    };
}

incin_macros::rank_sweep!(axis_split => impl_concat_shape);
