use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

use typenum::Unsigned;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Embedding<E: Unsigned + Dim, S: Shape, B: Backend<S> + Backend<Dyn>> {
    pub weight: Param<Dyn, B>,
    _phantom: PhantomData<(E, S)>,
}

impl<E: Unsigned + Dim, S: Shape + DynShape + AppendDim<E>, B> Module<Tensor<S, B>> for Embedding<E, S, B>
where
    B: Backend<S> + Backend<Dyn, RawTensor = <B as Backend<S>>::RawTensor> + Backend<<S as AppendDim<E>>::Output, RawTensor = <B as Backend<S>>::RawTensor>
{
    type Output = Tensor<<S as AppendDim<E>>::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let out = <B as Backend<Dyn>>::embedding(x.inner(), weight.inner())?;
        
        let mut dims = <S as DynShape>::dims(x.shape_field()).into();
        dims.push(E::USIZE);
        
        let shape = <S as AppendDim<E>>::Output::from_dyn(&dims).unwrap();
        
        Ok(Tensor::from_parts(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
