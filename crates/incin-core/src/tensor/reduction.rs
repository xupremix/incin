//! Runtime reduction choices shared by tensor and neural-network operations.

/// How an operation combines values across its reduction dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    /// Average the reduced values.
    #[default]
    Mean,
    /// Add the reduced values.
    Sum,
    /// Preserve the unreduced values.
    None,
}

/// Type-level reduction choice used by loss-shaped tensor operations.
pub trait ReductionMode: Clone + Default + 'static {
    /// Returns the runtime reduction choice.
    fn as_enum() -> Reduction;
}

#[derive(Debug, Clone, Copy, Default)]
/// Mean reduction.
pub struct Mean;
impl ReductionMode for Mean {
    fn as_enum() -> Reduction {
        Reduction::Mean
    }
}

#[derive(Debug, Clone, Copy, Default)]
/// Sum reduction.
pub struct Sum;
impl ReductionMode for Sum {
    fn as_enum() -> Reduction {
        Reduction::Sum
    }
}

#[derive(Debug, Clone, Copy, Default)]
/// No reduction.
pub struct NoneReduction;
impl ReductionMode for NoneReduction {
    fn as_enum() -> Reduction {
        Reduction::None
    }
}

/// Output-shape rule for mean-squared error reduction.
pub trait MseReductionShape<S: crate::shapes::Shape> {
    /// Result shape.
    type Output: crate::shapes::Shape;
}
impl<S: crate::shapes::Shape> MseReductionShape<S> for Mean {
    type Output = crate::shapes::Nil;
}
impl<S: crate::shapes::Shape> MseReductionShape<S> for Sum {
    type Output = crate::shapes::Nil;
}
impl<S: crate::shapes::Shape> MseReductionShape<S> for NoneReduction {
    type Output = S;
}

/// Output-shape rule for cross-entropy reduction.
pub trait CrossEntropyReductionShape<S: crate::shapes::Shape> {
    /// Result shape.
    type Output: crate::shapes::Shape;
}
impl<S: crate::shapes::Shape> CrossEntropyReductionShape<S> for Mean {
    type Output = crate::shapes::Nil;
}
impl<S: crate::shapes::Shape> CrossEntropyReductionShape<S> for Sum {
    type Output = crate::shapes::Nil;
}
impl<
    S: crate::shapes::Shape
        + crate::shapes::shape_ops::ReduceAt<crate::shapes::idx::Next<crate::shapes::idx::Here>>,
> CrossEntropyReductionShape<S> for NoneReduction
{
    type Output = <S as crate::shapes::shape_ops::ReduceAt<
        crate::shapes::idx::Next<crate::shapes::idx::Here>,
    >>::Output;
}

/// Output-shape rule for binary cross-entropy reduction.
pub trait BceReductionShape<S: crate::shapes::Shape> {
    /// Result shape.
    type Output: crate::shapes::Shape;
}
impl<S: crate::shapes::Shape> BceReductionShape<S> for Mean {
    type Output = crate::shapes::Nil;
}
impl<S: crate::shapes::Shape> BceReductionShape<S> for Sum {
    type Output = crate::shapes::Nil;
}
impl<S: crate::shapes::Shape> BceReductionShape<S> for NoneReduction {
    type Output = S;
}

/// Output-shape rule for L1 reduction.
pub trait L1ReductionShape<S: crate::shapes::Shape> {
    /// Result shape.
    type Output: crate::shapes::Shape;
}
impl<S: crate::shapes::Shape> L1ReductionShape<S> for Mean {
    type Output = crate::shapes::Nil;
}
impl<S: crate::shapes::Shape> L1ReductionShape<S> for Sum {
    type Output = crate::shapes::Nil;
}
impl<S: crate::shapes::Shape> L1ReductionShape<S> for NoneReduction {
    type Output = S;
}
