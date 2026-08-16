use crate::err::Result;
use crate::nn::Module;
use crate::shapes::Dyn;
use crate::shapes::FlattenAt;
use crate::shapes::idx::StaticCursor;
use crate::shapes::idx::{Here, Next};
use crate::shapes::{DynShape, Shape};
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;
use core::marker::PhantomData;

/// Structural flatten module. Axis positions are selector types rather than
/// const-generic rank-ladder parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct Flatten<Start, End>(PhantomData<fn() -> (Start, End)>);

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

impl<Start, End> Flatten<Start, End> {
    pub fn new() -> Self {
        Self(PhantomData)
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

/// Runtime-rank models commonly flatten the image axes after a dynamic batch
/// axis.  Keep that migration path on the same module type while the exact
/// structural implementation above remains available for statically-known
/// shapes.
impl<B, K, G> Module<Tensor<Dyn, B, K, G>> for Flatten<Next<Here>, Next<Next<Here>>>
where
    B: crate::tensor::backend::VariableBackend
        + crate::backend_authoring::Execute<
            crate::backend_authoring::op::FlattenExact,
            Output = <B as crate::tensor::backend::StorageBackend>::Storage<K>,
        > + crate::exec::Capabilities,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
{
    type Output = Tensor<Dyn, B, K, G>;
    type Error = crate::err::Error;

    fn forward(&self, x: Tensor<Dyn, B, K, G>) -> Result<Self::Output> {
        x.flatten_runtime(1, 3)
    }
}

impl<S, B, K, G, Start, End> Module<Tensor<S, B, K, G>> for Flatten<Start, End>
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
