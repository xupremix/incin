use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct LayerNorm<C: Dim, B: Backend> {
    pub weight: Param<(C,), B>,
    pub bias: Param<(C,), B>,
    pub eps: f32,
    _phantom: PhantomData<C>,
}

impl<C: Dim, B: Backend> LayerNorm<C, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new<A>(_args: A, _eps: f32) -> Result<Self>
    where
        A: ArgInto<((C,), f32, Cpu, Grad)>,
    {
        Err(Error::Msg(
            "LayerNorm init not fully implemented. Please use from_safetensors!".to_string(),
        ))
    }
}

impl<S: Shape + DynShape + crate::shapes::EndsWith<C>, C: Dim, B: Backend> Module<Tensor<S, B>> for LayerNorm<C, B> {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?.into_dyn();
        let bias = self.bias.as_tensor()?.into_dyn();
        x.layer_norm(&weight, &bias, self.eps)
    }
}
