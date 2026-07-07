use crate::nn::{Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

use typenum::Unsigned;

#[derive(Debug, Clone)]
pub struct MaxPool2d<K: Unsigned, S: Unsigned, P: Unsigned = typenum::U0, D: Unsigned = typenum::U1> {
    _phantom: core::marker::PhantomData<(K, S, P, D)>,
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> MaxPool2d<K, S, P, D> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            _phantom: core::marker::PhantomData,
        })
    }
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned, B: Backend> Parameters<B> for MaxPool2d<K, S, P, D> {
    fn named_parameters(&self, _prefix: &str, _map: &mut std::collections::HashMap<String, B::RawVar>) {}
}

impl<I: Shape + DynShape + crate::shapes::Pool2dShape<K, S, P, D>, K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned, B: Backend>
    Module<Tensor<I, B>> for MaxPool2d<K, S, P, D>
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let out =
            <B as Backend>::max_pool2d(x.inner(), (K::USIZE, K::USIZE), (S::USIZE, S::USIZE), (P::USIZE, P::USIZE), (D::USIZE, D::USIZE))?;

        let shape = <I as crate::shapes::Pool2dShape<K, S, P, D>>::compute_output_shape(x.shape_field());
        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
