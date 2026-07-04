use crate::prelude::Shape;

pub trait Transpose<const D1: usize, const D2: usize>: Shape {
    type Output: Shape;
}

pub trait ReduceDim<const D: usize>: Shape {
    type Output: Shape;
}

pub trait ReduceKeepDim<const D: usize>: Shape {
    type Output: Shape;
}

pub trait Flatten<const START: usize, const END: usize>: Shape {
    type Output: Shape;
}

impl<const START: usize, const END: usize> Flatten<START, END> for crate::prelude::Dyn {
    type Output = crate::prelude::Dyn;
}

impl<const D1: usize, const D2: usize> Transpose<D1, D2> for crate::prelude::Dyn {
    type Output = crate::prelude::Dyn;
}

impl<const D: usize> ReduceDim<D> for crate::prelude::Dyn {
    type Output = crate::prelude::Dyn;
}

impl<const D: usize> ReduceKeepDim<D> for crate::prelude::Dyn {
    type Output = crate::prelude::Dyn;
}

// Generate the trait implementations for permutations and reductions
kindle_macros::generate_shape_ops!();
