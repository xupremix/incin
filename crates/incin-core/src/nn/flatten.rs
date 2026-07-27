use crate::nn::Module;
use crate::prelude::*;

#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
/// Collapses dimensions `[START, END]` (inclusive) into a single dimension.
///
/// `START` and `END` are const-generic axis indices, while the sizes stored in
/// the input shape remain type-level `Dim` values. The module therefore keeps
/// the complete output shape in its `Module::Output` type.
pub struct Flatten<const START: usize, const END: usize> {}

impl<const START: usize, const END: usize> Default for Flatten<START, END> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const START: usize, const END: usize> Flatten<START, END> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Self {
        Self {}
    }
}

impl<S, B, K, G, const START: usize, const END: usize> Module<Tensor<S, B, K, G>>
    for Flatten<START, END>
where
    S: Shape + crate::shapes::DynShape + crate::shapes::Flatten<START, END>,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S::Output, B, K, G>;
    /// The error type returned if the forward pass fails.
    type Error = crate::prelude::Error;

    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<S, B, K, G>) -> Result<Self::Output> {
        x.flatten::<START, END>()
    }
}
