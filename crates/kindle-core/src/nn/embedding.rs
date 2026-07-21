use crate::nn::{Module, Param};
use crate::prelude::*;

/// Auto-generated documentation for EmbeddingShape.
pub trait EmbeddingShape: Shape + DynShape {
    /// Auto-generated documentation for Vocab.
    type Vocab: Dim;
    /// Auto-generated documentation for Embed.
    type Embed: Dim;
    /// Auto-generated documentation for BuildArg.
    type BuildArg: crate::tensor::arg_into::NotUnit;
    /// Auto-generated documentation for Target.
    type Target;

    /// Auto-generated documentation for build_args.
    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<V: Dim, E: Dim> EmbeddingShape for (V, E) {
    /// Auto-generated documentation for Vocab.
    type Vocab = V;
    /// Auto-generated documentation for Embed.
    type Embed = E;
    /// Auto-generated documentation for BuildArg.
    type BuildArg = (<V as Dim>::Arg, <E as Dim>::Arg);
    /// Auto-generated documentation for Target.
    type Target = (<V as Dim>::Arg, <E as Dim>::Arg);

    /// Auto-generated documentation for build_args.
    fn build_args(target: Self::Target) -> Self::BuildArg {
        target
    }
}

impl EmbeddingShape for Dyn {
    /// Auto-generated documentation for Vocab.
    type Vocab = usize;
    /// Auto-generated documentation for Embed.
    type Embed = usize;
    /// Auto-generated documentation for BuildArg.
    type BuildArg = alloc::vec::Vec<usize>;
    /// Auto-generated documentation for Target.
    type Target = (usize, usize);

    /// Auto-generated documentation for build_args.
    fn build_args(target: Self::Target) -> Self::BuildArg {
        alloc::vec![target.0, target.1]
    }
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
/// Auto-generated documentation for Embedding.
pub struct Embedding<S: EmbeddingShape, B: Backend> {
    /// Auto-generated documentation for weight.
    pub weight: Param<(S::Vocab, S::Embed), B>,
}

impl<S: EmbeddingShape, B: Backend> Embedding<S, B>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Vocab, S::Embed): Shape<Arg = S::BuildArg>,
{
    /// Auto-generated documentation for new_with.
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
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Vocab, S::Embed): Shape<Arg = S::BuildArg>,
{
    /// Auto-generated documentation for new.
    pub fn new() -> Result<Self> {
        Self::new_with(())
    }
}

impl<S: EmbeddingShape, InS: Shape + DynShape + AppendDim<S::Embed>, B> Module<Tensor<InS, B>>
    for Embedding<S, B>
where
    S::Embed: typenum::Unsigned,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
{
    /// Auto-generated documentation for Output.
    type Output = Tensor<<InS as AppendDim<S::Embed>>::Output, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
