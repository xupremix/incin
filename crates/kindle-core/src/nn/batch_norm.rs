use crate::nn::{Buffer, Module, Param};
use crate::prelude::*;

use core::marker::PhantomData;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct BatchNorm2d<S: Shape, B: Backend<S> + Backend<Dyn>> {
    pub weight: Param<Dyn, B>,
    pub bias: Param<Dyn, B>,
    pub running_mean: Buffer<Dyn, B>,
    pub running_var: Buffer<Dyn, B>,
    pub eps: f32,
    _phantom: PhantomData<S>,
}

impl<S: Shape + DynShape, B: Backend<S> + Backend<Dyn, RawVar = <B as Backend<S>>::RawVar>> BatchNorm2d<S, B> {
    pub fn new(num_features: usize, device: &KindleDevice) -> Result<Self> {
        let dtype = KindleDType::F32;
        let dims: &[usize] = &[num_features];
        Ok(Self {
            weight: Param {
                inner: <B as Backend<Dyn>>::var_ones(dims, dtype, device)?,
                _shape: dims.to_vec(),
                _dtype: PhantomData,
                _device: PhantomData,
            },
            bias: Param {
                inner: <B as Backend<Dyn>>::var_zeros(dims, dtype, device)?,
                _shape: dims.to_vec(),
                _dtype: PhantomData,
                _device: PhantomData,
            },
            running_mean: Buffer {
                inner: <B as Backend<Dyn>>::var_zeros(dims, dtype, device)?,
                _shape: dims.to_vec(),
                _dtype: PhantomData,
                _device: PhantomData,
            },
            running_var: Buffer {
                inner: <B as Backend<Dyn>>::var_ones(dims, dtype, device)?,
                _shape: dims.to_vec(),
                _dtype: PhantomData,
                _device: PhantomData,
            },
            eps: 1e-5,
            _phantom: PhantomData,
        })
    }
}
impl<S: Shape + DynShape, B: Backend<S> + Backend<Dyn, RawTensor = <B as Backend<S>>::RawTensor>>
    Module<Tensor<S, B>> for BatchNorm2d<S, B>
{
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
