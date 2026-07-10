use crate::nn::Module;
use crate::prelude::*;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Flatten<const START: usize, const END: usize> {}

impl<const START: usize, const END: usize> Default for Flatten<START, END> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const START: usize, const END: usize> Flatten<START, END> {
    pub fn new() -> Self {
        Self {}
    }
}

impl<S, B, K, D, G, const START: usize, const END: usize> Module<Tensor<S, B, K, D, G>> for Flatten<START, END>
where
    S: Shape + crate::shapes::DynShape + crate::shapes::Flatten<START, END>,
    B: Backend,
    K: crate::tensor::dtype::DType,
    D: crate::tensor::device::Device,
    G: RequiresGrad,
{
    type Output = Tensor<S::Output, B, K, D, G>;
    type Error = crate::prelude::Error;

    fn forward(&self, x: Tensor<S, B, K, D, G>) -> Result<Self::Output> {
        x.flatten::<START, END>()
    }
}
