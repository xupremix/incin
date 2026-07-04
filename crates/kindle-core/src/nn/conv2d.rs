use crate::nn::{Module, Param};
use crate::prelude::*;
use crate::shapes::Conv2dShape;
use typenum::Unsigned;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Conv2d<K: Unsigned, S: Unsigned, P: Unsigned, D: Unsigned, W: Shape, B: Backend<W> + Backend<Dyn>> {
    pub weight: Param<W, B>,
    pub bias: Option<Param<Dyn, B>>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    _phantom: core::marker::PhantomData<(K, S, P, D)>,
}

impl<I, K, S, P, D, W, B> Module<Tensor<I, B>> for Conv2d<K, S, P, D, W, B>
where
    K: Unsigned,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
    I: Shape + DynShape + Conv2dShape<K, S, P, D>,
    W: Shape,
    B: Backend<W, RawTensor = <B as Backend<I>>::RawTensor>
        + Backend<Dyn, RawTensor = <B as Backend<I>>::RawTensor>
        + Backend<I>
        + Backend<I::Output, RawTensor = <B as Backend<I>>::RawTensor>,
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = match &self.bias {
            Some(b) => Some(b.as_tensor()?),
            None => None,
        };

        // Note: the backend conv2d currently expects a single usize for symmetric params
        // or we need to update it if we switch to asymmetric.
        let out = <B as Backend<I>>::conv2d(
            x.inner(),
            weight.inner(),
            bias.as_ref().map(|b| b.inner()),
            self.stride,
            self.padding,
            self.dilation,
        )?;
        
        let mut dims = <I as DynShape>::dims(x.shape_field()).into();
        if dims.len() == 4 {
            dims[2] = (dims[2] + 2 * P::USIZE - D::USIZE * (K::USIZE - 1) - 1) / S::USIZE + 1;
            dims[3] = (dims[3] + 2 * P::USIZE - D::USIZE * (K::USIZE - 1) - 1) / S::USIZE + 1;
        }
        
        let shape = I::Output::from_dyn(&dims).unwrap();

        Ok(Tensor::from_parts(out, shape, x._dtype.clone(), weight._device.clone(), core::marker::PhantomData))
    }
}
