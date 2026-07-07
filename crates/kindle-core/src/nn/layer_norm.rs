use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

pub trait LayerNormShape: Shape + DynShape {
    type Channels: Dim;
}

impl<C: Dim> LayerNormShape for (C,) {
    type Channels = C;
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct LayerNorm<S: LayerNormShape, B: Backend> {
    pub weight: Param<(S::Channels,), B>,
    pub bias: Param<(S::Channels,), B>,
    #[module(ignore)]
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<B>,
}

impl<S: LayerNormShape, B: Backend> LayerNorm<S, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(args: <S::Channels as Dim>::Arg, eps: f32) -> Result<Self> {
        let _c = S::Channels::from_arg(args.clone());
        Ok(Self {
            weight: Param::<(S::Channels,), B>::ones((args.clone(),))?,
            bias: Param::<(S::Channels,), B>::zeros((args.clone(),))?,
            eps,
            _phantom: PhantomData,
        })
    }

    pub fn new_dyn(c_size: usize, eps: f32) -> Result<Self> {
        let c = S::Channels::from_size(c_size)
            .ok_or_else(|| Error::Msg("Invalid channel size".into()))?;
        Self::new(c.arg(), eps)
    }
}

impl<S: LayerNormShape, InS: Shape + DynShape + crate::shapes::EndsWith<S::Channels>, B: Backend>
    Module<Tensor<InS, B>> for LayerNorm<S, B>
{
    type Output = Tensor<InS, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<InS, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?.into_dyn();
        let bias = self.bias.as_tensor()?.into_dyn();
        let out = B::layer_norm(x.inner(), weight.inner(), bias.inner(), self.eps)?;
        Ok(Tensor::from_parts_unchecked(
            out,
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad,
        ))
    }
}
