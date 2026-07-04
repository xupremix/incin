use crate::nn::{Buffer, Module, Param};
use crate::prelude::*;

use core::marker::PhantomData;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct BatchNorm2d<C: Dim, B: Backend> {
    pub weight: Param<(C,), B>,
    pub bias: Param<(C,), B>,
    pub running_mean: Buffer<(C,), B>,
    pub running_var: Buffer<(C,), B>,
    pub eps: f32,
    _phantom: PhantomData<C>,
}

impl<C: Dim, B: Backend> BatchNorm2d<C, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new<A>(args: A, _device: &KindleDevice) -> Result<Self>
    where
        A: Clone + ArgInto<(<C as Dim>::Arg,)>,
    {
        Ok(Self {
            weight: Param::<(C,), B>::ones(args.clone())?,
            bias: Param::<(C,), B>::zeros(args.clone())?,
            running_mean: Buffer::<(C,), B>::zeros(args.clone())?,
            running_var: Buffer::<(C,), B>::ones(args)?,
            eps: 1e-5,
            _phantom: PhantomData,
        })
    }
}
impl<N: Dim, C: Dim, H: Dim, W: Dim, B: Backend> Module<Tensor<(N, C, H, W), B>> for BatchNorm2d<C, B> {
    type Output = Tensor<(N, C, H, W), B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<(N, C, H, W), B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?.into_dyn();
        let bias = self.bias.as_tensor()?.into_dyn();
        let running_mean = self.running_mean.as_tensor()?.into_dyn();
        let running_var = self.running_var.as_tensor()?.into_dyn();
        x.batch_norm(&weight, &bias, &running_mean, &running_var, self.eps)
    }
}
