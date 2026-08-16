use crate::err::Result;
use crate::nn::Module;
use crate::shapes::Dyn;
use crate::shapes::FlattenAt;
use crate::shapes::idx::StaticCursor;
use crate::shapes::{DynShape, Shape};
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;
use crate::tensor::ops::manipulation::FlattenSelector;
use core::marker::PhantomData;

/// Ergonomic flatten module driven by two axis selectors.
#[derive(Debug, Clone, Copy, Default)]
pub struct Flatten<Start, End> {
    start: Start,
    end: End,
}

impl<Start, End> Flatten<Start, End> {
    /// Creates a flattening module for an inclusive axis range.
    #[must_use]
    pub const fn new(start: Start, end: End) -> Self {
        Self { start, end }
    }
}

/// Advanced structural flatten module for internal shape-proof code.
#[derive(Debug, Clone, Copy, Default)]
pub struct StructuralFlatten<Start, End>(PhantomData<fn() -> (Start, End)>);

/// Runtime-axis flattening module for ordinary model code.
///
/// The inclusive axis range uses signed indices, so `FlattenAxes::new(1, -1)`
/// flattens every dimension after a leading batch axis without exposing proof
/// cursor types. The output shape is `Dyn` because the selected positions are
/// runtime values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlattenAxes {
    start: isize,
    end: isize,
}

impl FlattenAxes {
    /// Creates a checked runtime-axis flattening module.
    #[must_use]
    pub const fn new(start: isize, end: isize) -> Self {
        Self { start, end }
    }
}

impl<Start, End, B: crate::tensor::backend::VariableBackend> crate::nn::VisitParameters<B>
    for Flatten<Start, End>
{
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        _: &crate::nn::StatePath,
        _: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl<B: crate::tensor::backend::VariableBackend> crate::nn::VisitParameters<B> for FlattenAxes {
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        _: &crate::nn::StatePath,
        _: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}

impl<S, B, K, G> Module<Tensor<S, B, K, G>> for FlattenAxes
where
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend
        + crate::backend_authoring::Execute<
            crate::backend_authoring::op::FlattenExact,
            Output = <B as crate::backend_authoring::StorageBackend>::Storage<K>,
        > + crate::exec::Capabilities,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
{
    type Output = Tensor<Dyn, B, K, G>;
    type Error = crate::err::Error;

    fn forward(&self, x: Tensor<S, B, K, G>) -> Result<Self::Output> {
        x.flatten_runtime(self.start, self.end)
    }
}

impl<S, B, K, G, Start: Copy, End: Copy> Module<Tensor<S, B, K, G>> for Flatten<Start, End>
where
    S: Shape + DynShape,
    (): FlattenSelector<S, Start, End>,
    <() as FlattenSelector<S, Start, End>>::Output: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend
        + crate::backend_authoring::Execute<
            crate::backend_authoring::op::FlattenExact,
            Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
        > + crate::exec::Capabilities,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
{
    type Output = Tensor<<() as FlattenSelector<S, Start, End>>::Output, B, K, G>;
    type Error = crate::err::Error;

    fn forward(&self, x: Tensor<S, B, K, G>) -> Result<Self::Output> {
        x.flatten(self.start, self.end)
    }
}

impl<S, B, K, G, Start, End> Module<Tensor<S, B, K, G>> for StructuralFlatten<Start, End>
where
    Start: StaticCursor,
    End: StaticCursor,
    S: Shape + DynShape + FlattenAt<Start, End>,
    <S as FlattenAt<Start, End>>::Output: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend
        + crate::backend_authoring::Execute<
            crate::backend_authoring::op::FlattenExact,
            Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
        > + crate::exec::Capabilities,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
{
    type Output = Tensor<<S as FlattenAt<Start, End>>::Output, B, K, G>;
    type Error = crate::err::Error;

    fn forward(&self, x: Tensor<S, B, K, G>) -> Result<Self::Output> {
        x.flatten_structural::<Start, End>()
    }
}
