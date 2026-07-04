use crate::nn::{Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

use typenum::Unsigned;

#[derive(Debug, Clone)]
pub struct AvgPool2d<K: Unsigned, S: Unsigned> {
    _phantom: core::marker::PhantomData<(K, S)>,
}

impl<K: Unsigned, S: Unsigned> AvgPool2d<K, S> {
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<K: Unsigned, S: Unsigned, B: Backend> Parameters<B> for AvgPool2d<K, S> {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<I: Shape + DynShape + crate::shapes::Pool2dShape<K, S>, K: Unsigned, S: Unsigned, B: Backend>
    Module<Tensor<I, B>> for AvgPool2d<K, S>
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let out =
            <B as Backend>::avg_pool2d(x.inner(), (K::USIZE, K::USIZE), (S::USIZE, S::USIZE))?;

        let shape = <I as crate::shapes::Pool2dShape<K, S>>::compute_output_shape(x.shape_field());
        Ok(Tensor::from_parts(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
