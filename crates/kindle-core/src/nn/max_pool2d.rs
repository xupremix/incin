use crate::nn::{Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel_size, stride }
    }
}

impl<B: Backend<Dyn>> Parameters<B> for MaxPool2d {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

// Pooling output shapes can be computed if we have strong typing, but for now we implement it for Dyn tensors
impl<B: Backend<Dyn>> Module<Tensor<Dyn, B>> for MaxPool2d {
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<Dyn, B>) -> core::result::Result<Self::Output, Error> {
        let out = <B as Backend<Dyn>>::max_pool2d(x.inner(), self.kernel_size, self.stride)?;
        let shape = <B as Backend<Dyn>>::shape(&out);
        Ok(Tensor::from_parts(out, shape, x._dtype.clone(), x._device.clone(), core::marker::PhantomData))
    }
}
