use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Embedding<S: Shape, B: Backend<S> + Backend<Dyn>> {
    pub weight: Param<Dyn, B>,
    _phantom: PhantomData<S>,
}

impl<S: Shape + DynShape, B: Backend<S> + Backend<Dyn, RawTensor = <B as Backend<S>>::RawTensor>>
    Module<Tensor<S, B>> for Embedding<S, B>
{
    type Output = Tensor<Dyn, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let out = <B as Backend<Dyn>>::embedding(x.inner(), weight.inner())?;
        let shape = <B as Backend<Dyn>>::shape(&out);
        Ok(Tensor::from_parts(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
