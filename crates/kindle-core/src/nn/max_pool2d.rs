use crate::nn::{Module, Parameters};
use crate::prelude::*;

use typenum::Unsigned;

#[derive(Debug, Clone)]
/// Core abstraction for `MaxPool2d` within the Kindle framework..
pub struct MaxPool2d<K: Unsigned, S: Unsigned, P: Unsigned = typenum::U0, D: Unsigned = typenum::U1>
{
    _phantom: core::marker::PhantomData<(K, S, P, D)>,
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned> MaxPool2d<K, S, P, D> {
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new() -> Result<Self> {
        Ok(Self {
            _phantom: core::marker::PhantomData,
        })
    }
}

impl<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned, B: Backend> Parameters<B>
    for MaxPool2d<K, S, P, D>
{
    /// Core abstraction for `named_parameters` within the Kindle framework..
    fn named_parameters(
        &self,
        _prefix: &str,
        _map: &mut alloc::collections::BTreeMap<String, B::RawVar>,
    ) {
    }
}

impl<
    I: Shape + DynShape + crate::shapes::Pool2dShape<K, S, P, D>,
    K: Unsigned,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
> Module<Tensor<I, B>> for MaxPool2d<K, S, P, D>
{
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Tensor<I::Output, B>;
    /// Core abstraction for `Error` within the Kindle framework..
    type Error = Error;

    #[inline]
    /// Core abstraction for `forward` within the Kindle framework..
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let out = B::max_pool2d(
            x.inner(),
            (K::USIZE, K::USIZE),
            (S::USIZE, S::USIZE),
            (P::USIZE, P::USIZE),
            (D::USIZE, D::USIZE),
        )?;

        let shape =
            <I as crate::shapes::Pool2dShape<K, S, P, D>>::compute_output_shape(x.shape_field());
        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
