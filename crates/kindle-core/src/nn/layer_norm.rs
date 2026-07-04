use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct LayerNorm<S: Shape, B: Backend<S> + Backend<Dyn>> {
    pub weight: Param<Dyn, B>,
    pub bias: Param<Dyn, B>,
    pub eps: f32,
    _phantom: PhantomData<S>,
}

impl<S: Shape, B: Backend<S> + Backend<Dyn>> LayerNorm<S, B> {
    pub fn new<A>(_args: A, _eps: f32) -> Result<Self>
    where
        A: ArgInto<(Dyn, f32, Cpu, Grad)>,
        S: DynShape,
    {
        Err(Error::Msg(
            "LayerNorm init not fully implemented. Please use from_safetensors!".to_string(),
        ))
    }
}



impl<S: Shape + DynShape, B: Backend<S> + Backend<Dyn, RawTensor = <B as Backend<S>>::RawTensor>>
    Module<Tensor<S, B>> for LayerNorm<S, B>
{
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = self.bias.as_tensor()?;
        x.layer_norm(&weight, &bias, self.eps)
    }
}
