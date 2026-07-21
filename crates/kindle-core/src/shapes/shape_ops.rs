use crate::prelude::Shape;

#[diagnostic::on_unimplemented(
    message = "Cannot transpose dimensions `{D1}` and `{D2}` on shape `{Self}`",
    label = "Invalid transpose",
    note = "Transpose requires both dimensions to be < the rank of the tensor"
)]
/// Auto-generated documentation for Transpose.
pub trait Transpose<const D1: usize, const D2: usize>: Shape {
    /// Auto-generated documentation for Output.
    type Output: Shape;
}

#[diagnostic::on_unimplemented(
    message = "Cannot reduce dimension `{D}` on shape `{Self}`",
    label = "Invalid reduction dimension",
    note = "Reduction requires the dimension to be < the rank of the tensor"
)]
/// Auto-generated documentation for ReduceDim.
pub trait ReduceDim<const D: usize>: Shape {
    /// Auto-generated documentation for Output.
    type Output: Shape;
}

#[diagnostic::on_unimplemented(
    message = "Cannot reduce dimension `{D}` (keepdim) on shape `{Self}`",
    label = "Invalid reduction dimension",
    note = "Reduction requires the dimension to be < the rank of the tensor"
)]
/// Auto-generated documentation for ReduceKeepDim.
pub trait ReduceKeepDim<const D: usize>: Shape {
    /// Auto-generated documentation for Output.
    type Output: Shape;
}

#[diagnostic::on_unimplemented(
    message = "Cannot flatten shape `{Self}` from dimension `{START}` to `{END}`",
    label = "Invalid flatten range",
    note = "Flatten requires START <= END and END < the rank of the tensor"
)]
/// Auto-generated documentation for Flatten.
pub trait Flatten<const START: usize, const END: usize>: Shape {
    /// Auto-generated documentation for Output.
    type Output: Shape;
}

impl<const START: usize, const END: usize> Flatten<START, END> for crate::prelude::Dyn {
    /// Auto-generated documentation for Output.
    type Output = crate::prelude::Dyn;
}

impl<const D1: usize, const D2: usize> Transpose<D1, D2> for crate::prelude::Dyn {
    /// Auto-generated documentation for Output.
    type Output = crate::prelude::Dyn;
}

impl<const D: usize> ReduceDim<D> for crate::prelude::Dyn {
    /// Auto-generated documentation for Output.
    type Output = crate::prelude::Dyn;
}

impl<const D: usize> ReduceKeepDim<D> for crate::prelude::Dyn {
    /// Auto-generated documentation for Output.
    type Output = crate::prelude::Dyn;
}

// Generate the trait implementations for permutations and reductions
kindle_macros::generate_shape_ops!();
