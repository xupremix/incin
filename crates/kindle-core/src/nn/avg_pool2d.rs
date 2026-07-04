use crate::nn::{Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

use typenum::Unsigned;

#[derive(Debug, Clone)]
pub struct AvgPool2d<K: Unsigned, S: Unsigned> {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    _phantom: core::marker::PhantomData<(K, S)>,
}

impl<K: Unsigned, S: Unsigned> AvgPool2d<K, S> {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel_size, stride, _phantom: core::marker::PhantomData }
    }
}

impl<K: Unsigned, S: Unsigned, B: Backend> Parameters<B> for AvgPool2d<K, S> {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<
    I: Shape + DynShape + crate::shapes::Pool2dShape<K, S>,
    K: Unsigned,
    S: Unsigned,
    B: Backend
> Module<Tensor<I, B>> for AvgPool2d<K, S>
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let out = <B as Backend>::avg_pool2d(x.inner(), self.kernel_size, self.stride)?;
        
        let mut dims = <I as DynShape>::dims(x.shape_field()).into();
        // Fallback for Dyn inputs
        if dims.len() == 4 {
            dims[2] = (dims[2] - K::USIZE) / S::USIZE + 1;
            dims[3] = (dims[3] - K::USIZE) / S::USIZE + 1;
        }
        
        let shape = I::Output::from_dyn(&dims).unwrap();
        Ok(Tensor::from_parts(out, shape, x._dtype.clone(), x._device.clone(), core::marker::PhantomData))
    }
}
