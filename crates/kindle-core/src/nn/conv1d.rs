use crate::nn::{Module, Param};
use crate::prelude::*;
use crate::tensor::conv1d::Conv1dShape;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Conv1d<W: Shape, B: Backend<W> + Backend<Dyn>> {
    pub weight: Param<W, B>,
    pub bias: Option<Param<Dyn, B>>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}



impl<I, W, B> Module<Tensor<I, B>> for Conv1d<W, B>
where
    I: Shape + Conv1dShape<W>,
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

        let out = <B as Backend<I>>::conv1d(
            x.inner(),
            weight.inner(),
            bias.as_ref().map(|b| b.inner()),
            self.stride,
            self.padding,
            self.dilation,
        )?;
        
        let shape = I::Output::from_dyn(&<B as Backend<I>>::shape(&out)).unwrap();

        Ok(Tensor::from_parts(out, shape, x._dtype.clone(), weight._device.clone(), core::marker::PhantomData))
    }
}
