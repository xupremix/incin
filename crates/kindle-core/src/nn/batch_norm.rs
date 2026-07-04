use crate::nn::{Buffer, Module, Param};
use crate::prelude::*;

use core::marker::PhantomData;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct BatchNorm2d<S: Shape, B: Backend> {
    pub weight: Param<Dyn, B>,
    pub bias: Param<Dyn, B>,
    pub running_mean: Buffer<Dyn, B>,
    pub running_var: Buffer<Dyn, B>,
    pub eps: f32,
    _phantom: PhantomData<S>,
}

impl<S: Shape + DynShape, B: Backend> BatchNorm2d<S, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(num_features: usize, _device: &KindleDevice) -> Result<Self>
    where
        B::DType: crate::prelude::ConstDType,
        B::Device: crate::prelude::ConstDevice,
    {
        let _dtype = KindleDType::F32;
        let _dims: &[usize] = &[num_features];
        Ok(Self {
            weight: Param::<Dyn, B>::ones([num_features])?,
            bias: Param::<Dyn, B>::zeros([num_features])?,
            running_mean: Buffer::<Dyn, B>::zeros([num_features])?,
            running_var: Buffer::<Dyn, B>::ones([num_features])?,
            eps: 1e-5,
            _phantom: PhantomData,
        })
    }
}
impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for BatchNorm2d<S, B> {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = self.bias.as_tensor()?;
        let running_mean = self.running_mean.as_tensor()?;
        let running_var = self.running_var.as_tensor()?;
        x.batch_norm(&weight, &bias, &running_mean, &running_var, self.eps)
    }
}
