use crate::nn::Module;
use crate::prelude::*;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Flatten<const START: usize, const END: usize> {}

impl<const START: usize, const END: usize> Flatten<START, END> {
    pub fn new() -> Self {
        Self {}
    }
}

impl<S, B, G, const START: usize, const END: usize> Module<Tensor<S, B, G>> for Flatten<START, END>
where
    S: Shape + crate::shapes::DynShape + crate::shapes::Flatten<START, END>,
    B: Backend,
    G: RequiresGrad,
{
    type Output = Tensor<S::Output, B, G>;
    type Error = crate::prelude::Error;

    fn forward(&self, x: Tensor<S, B, G>) -> Result<Self::Output> {
        x.flatten::<START, END>()
    }
}
