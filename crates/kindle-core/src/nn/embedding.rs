use crate::nn::{Module, Param};
use crate::prelude::*;

use typenum::Unsigned;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Embedding<V: Dim, E: Dim, B: Backend> {
    pub weight: Param<(V, E), B>,
}

impl<V: Dim, E: Dim + typenum::Unsigned, S: Shape + DynShape + AppendDim<E>, B> Module<Tensor<S, B>>
    for Embedding<V, E, B>
where
    B: Backend,
{
    type Output = Tensor<<S as AppendDim<E>>::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let out = <B as Backend>::embedding(x.inner(), weight.inner())?;

        let mut dims = <S as DynShape>::dims(x.shape_field()).into();
        dims.push(E::USIZE);

        let shape = <S as AppendDim<E>>::Output::from_dyn(&dims).unwrap();

        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
