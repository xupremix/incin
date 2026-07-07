use crate::nn::{Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

use typenum::Unsigned;

#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d<HOut: Unsigned, WOut: Unsigned> {
    _phantom: core::marker::PhantomData<(HOut, WOut)>,
}

impl<HOut: Unsigned, WOut: Unsigned> AdaptiveAvgPool2d<HOut, WOut> {
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<HOut: Unsigned, WOut: Unsigned, B: Backend> Parameters<B> for AdaptiveAvgPool2d<HOut, WOut> {
    fn named_parameters(&self, _prefix: &str, _map: &mut std::collections::HashMap<String, B::RawVar>) {}
}

impl<
        I: Shape + DynShape + crate::shapes::AdaptiveAvgPool2dShape<HOut, WOut>,
        HOut: Unsigned,
        WOut: Unsigned,
        B: Backend,
    > Module<Tensor<I, B>> for AdaptiveAvgPool2d<HOut, WOut>
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let out = <B as Backend>::adaptive_avg_pool2d(x.inner(), (HOut::USIZE, WOut::USIZE))?;

        let shape = <I as crate::shapes::AdaptiveAvgPool2dShape<HOut, WOut>>::compute_output_shape(
            x.shape_field(),
        );
        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
