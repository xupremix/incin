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

// Generate the trait implementations for permutations and reductions
kindle_macros::generate_shape_ops!();
