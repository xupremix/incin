use crate::prelude::Shape;

#[diagnostic::on_unimplemented(
    message = "Cannot transpose dimensions `{D1}` and `{D2}` on shape `{Self}`",
    label = "Invalid transpose",
    note = "Transpose requires both dimensions to be < the rank of the tensor"
)]
/// Compile-time-checked shape rule for swapping dimensions `D1`/`D2`.
pub trait Transpose<const D1: usize, const D2: usize>: Shape {
    /// `Self` with dimensions `D1` and `D2` swapped.
    type Output: Shape;
}

#[diagnostic::on_unimplemented(
    message = "Cannot reduce dimension `{D}` on shape `{Self}`",
    label = "Invalid reduction dimension",
    note = "Reduction requires the dimension to be < the rank of the tensor"
)]
/// Compile-time-checked shape rule for reducing (removing) dimension `D`.
pub trait ReduceDim<const D: usize>: Shape {
    /// `Self` with dimension `D` removed.
    type Output: Shape;
}

#[diagnostic::on_unimplemented(
    message = "Cannot reduce dimension `{D}` (keepdim) on shape `{Self}`",
    label = "Invalid reduction dimension",
    note = "Reduction requires the dimension to be < the rank of the tensor"
)]
/// Compile-time-checked shape rule for reducing dimension `D` while
/// keeping it in the shape at size 1.
pub trait ReduceKeepDim<const D: usize>: Shape {
    /// `Self` with dimension `D`'s size set to 1.
    type Output: Shape;
}

#[diagnostic::on_unimplemented(
    message = "Cannot flatten shape `{Self}` from dimension `{START}` to `{END}`",
    label = "Invalid flatten range",
    note = "Flatten requires START <= END and END < the rank of the tensor"
)]
/// Compile-time-checked shape rule for collapsing dimensions
/// `[START, END]` into a single dimension.
pub trait Flatten<const START: usize, const END: usize>: Shape {
    /// `Self` with dimensions `[START, END]` collapsed into one.
    type Output: Shape;
}

impl<const START: usize, const END: usize> Flatten<START, END> for crate::prelude::Dyn {
    /// Always `Dyn` — the concrete size is only known at runtime.
    type Output = crate::prelude::Dyn;
}

impl<const D1: usize, const D2: usize> Transpose<D1, D2> for crate::prelude::Dyn {
    /// Always `Dyn` — the concrete size is only known at runtime.
    type Output = crate::prelude::Dyn;
}

impl<const D: usize> ReduceDim<D> for crate::prelude::Dyn {
    /// Always `Dyn` — the concrete size is only known at runtime.
    type Output = crate::prelude::Dyn;
}

impl<const D: usize> ReduceKeepDim<D> for crate::prelude::Dyn {
    /// Always `Dyn` — the concrete size is only known at runtime.
    type Output = crate::prelude::Dyn;
}

// Generate the trait implementations for permutations and reductions
incin_macros::generate_shape_ops!();
