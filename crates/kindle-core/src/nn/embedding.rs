use crate::nn::{Module, Param};
use crate::prelude::*;

pub trait EmbeddingShape: Shape + DynShape {
    type Vocab: Dim;
    type Embed: Dim;
}

impl<V: Dim, E: Dim> EmbeddingShape for (V, E) {
    type Vocab = V;
    type Embed = E;
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Embedding<S: EmbeddingShape, B: Backend> {
    pub weight: Param<(S::Vocab, S::Embed), B>,
}

impl<S: EmbeddingShape, InS: Shape + DynShape + AppendDim<S::Embed>, B> Module<Tensor<InS, B>>
    for Embedding<S, B>
where
    S::Embed: typenum::Unsigned,
    B: Backend,
{
    type Output = Tensor<<InS as AppendDim<S::Embed>>::Output, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<InS, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let out = <B as Backend>::embedding(x.inner(), weight.inner())?;

        let mut dims = <InS as DynShape>::dims(x.shape_field()).into();
        dims.push(<S::Embed as typenum::Unsigned>::USIZE);

        let shape = <InS as AppendDim<S::Embed>>::Output::from_dyn(&dims).unwrap();

        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        ))
    }
}
