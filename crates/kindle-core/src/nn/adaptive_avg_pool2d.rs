use crate::nn::{Module, Parameters};
use crate::prelude::*;

use typenum::Unsigned;

#[derive(Debug, Clone)]
/// Core abstraction for `AdaptiveAvgPool2d` within the Kindle framework..
pub struct AdaptiveAvgPool2d<HOut: Unsigned, WOut: Unsigned> {
    _phantom: core::marker::PhantomData<(HOut, WOut)>,
}

impl<HOut: Unsigned, WOut: Unsigned> Default for AdaptiveAvgPool2d<HOut, WOut> {
    fn default() -> Self {
        Self::new()
    }
}

impl<HOut: Unsigned, WOut: Unsigned> AdaptiveAvgPool2d<HOut, WOut> {
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<HOut: Unsigned, WOut: Unsigned, B: Backend> Parameters<B> for AdaptiveAvgPool2d<HOut, WOut> {
    /// Core abstraction for `named_parameters` within the Kindle framework..
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<
    I: Shape + DynShape + crate::shapes::AdaptiveAvgPool2dShape<HOut, WOut>,
    HOut: Unsigned,
    WOut: Unsigned,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
> Module<Tensor<I, B>> for AdaptiveAvgPool2d<HOut, WOut>
{
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Tensor<I::Output, B>;
    /// Core abstraction for `Error` within the Kindle framework..
    type Error = Error;

    #[inline]
    /// Core abstraction for `forward` within the Kindle framework..
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let out = B::adaptive_avg_pool2d(x.inner(), (HOut::USIZE, WOut::USIZE))?;

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
