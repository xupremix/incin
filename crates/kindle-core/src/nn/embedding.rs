use crate::nn::{Module, Param};
use crate::prelude::*;

pub trait EmbeddingShape: Shape + DynShape {
    type Vocab: Dim;
    type Embed: Dim;
    type BuildArg: crate::tensor::arg_into::NotUnit;
    type Target;

    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<V: Dim, E: Dim> EmbeddingShape for (V, E) {
    type Vocab = V;
    type Embed = E;
    type BuildArg = (<V as Dim>::Arg, <E as Dim>::Arg);
    type Target = (<V as Dim>::Arg, <E as Dim>::Arg);

    fn build_args(target: Self::Target) -> Self::BuildArg {
        target
    }
}

impl EmbeddingShape for Dyn {
    type Vocab = usize;
    type Embed = usize;
    type BuildArg = alloc::vec::Vec<usize>;
    type Target = (usize, usize);

    fn build_args(target: Self::Target) -> Self::BuildArg {
        alloc::vec![target.0, target.1]
    }
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Embedding<S: EmbeddingShape, B: Backend> {
    pub weight: Param<(S::Vocab, S::Embed), B>,
}

impl<S: EmbeddingShape, B: Backend> Embedding<S, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Vocab, S::Embed): Shape<Arg = S::BuildArg>,
{
    pub fn new_with(args: S::Target) -> Result<Self> {
        let w_args = S::build_args(args);
        let w_args_data = crate::tensor::arg_into::TensorArgsData {
            shape: w_args,
            dtype: (),
            device: (),
            grad: (),
        };

        // Kaiming uniform init or standard normal? Usually standard normal.
        let weight = Param::<(S::Vocab, S::Embed), B>::new_init_raw(
            w_args_data,
            crate::nn::init::Init::Randn,
        )?;
        Ok(Self { weight })
    }
}

impl<S, B> Embedding<S, B>
where
    S: EmbeddingShape<Target = ()>,
    B: Backend,
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Vocab, S::Embed): Shape<Arg = S::BuildArg>,
{
    pub fn new() -> Result<Self> {
        Self::new_with(())
    }
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
        let out = B::embedding(x.inner(), weight.inner())?;

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
