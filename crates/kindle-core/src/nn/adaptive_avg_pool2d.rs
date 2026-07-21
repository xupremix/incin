use crate::nn::{Module, Parameters};
use crate::prelude::*;

use typenum::Unsigned;

#[derive(Debug, Clone)]
/// `AdaptiveAvgPool2d`.
pub struct AdaptiveAvgPool2d<HOut: Unsigned, WOut: Unsigned> {
    _phantom: core::marker::PhantomData<(HOut, WOut)>,
}

impl<HOut: Unsigned, WOut: Unsigned> Default for AdaptiveAvgPool2d<HOut, WOut> {
    fn default() -> Self {
        Self::new()
    }
}

impl<HOut: Unsigned, WOut: Unsigned> AdaptiveAvgPool2d<HOut, WOut> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<HOut: Unsigned, WOut: Unsigned, B: Backend> Parameters<B> for AdaptiveAvgPool2d<HOut, WOut> {
    /// Collects named trainable parameters into `map` under the given `prefix`.
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
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
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
