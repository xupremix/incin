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
    B: Backend<W> + Backend<Dyn> + Backend<I> + Backend<I::Output>,
{
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, _x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let _weight = self.weight.as_tensor()?;
        let _bias = match &self.bias {
            Some(b) => Some(b.as_tensor()?),
            None => None,
        };

        // Note: Backend might need conv1d op!
        // We simulate it using unsupported for now since Candle conv1d is 1d.
        Err(Error::Msg(
            "Conv1d forward not implemented on Backend trait yet".to_string(),
        ))
    }
}
