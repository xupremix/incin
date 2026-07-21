use crate::nn::{Module, Param};
use crate::prelude::*;

/// Core abstraction for `EmbeddingShape` within the Kindle framework..
pub trait EmbeddingShape: Shape + DynShape {
    /// Core abstraction for `Vocab` within the Kindle framework..
    type Vocab: Dim;
    /// Core abstraction for `Embed` within the Kindle framework..
    type Embed: Dim;
    /// Core abstraction for `BuildArg` within the Kindle framework..
    type BuildArg: crate::tensor::arg_into::NotUnit;
    /// Core abstraction for `Target` within the Kindle framework..
    type Target;

    /// Core abstraction for `build_args` within the Kindle framework..
    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<V: Dim, E: Dim> EmbeddingShape for (V, E) {
    /// Core abstraction for `Vocab` within the Kindle framework..
    type Vocab = V;
    /// Core abstraction for `Embed` within the Kindle framework..
    type Embed = E;
    /// Core abstraction for `BuildArg` within the Kindle framework..
    type BuildArg = (<V as Dim>::Arg, <E as Dim>::Arg);
    /// Core abstraction for `Target` within the Kindle framework..
    type Target = (<V as Dim>::Arg, <E as Dim>::Arg);

    /// Core abstraction for `build_args` within the Kindle framework..
    fn build_args(target: Self::Target) -> Self::BuildArg {
        target
    }
}

impl EmbeddingShape for Dyn {
    /// Core abstraction for `Vocab` within the Kindle framework..
    type Vocab = usize;
    /// Core abstraction for `Embed` within the Kindle framework..
    type Embed = usize;
    /// Core abstraction for `BuildArg` within the Kindle framework..
    type BuildArg = alloc::vec::Vec<usize>;
    /// Core abstraction for `Target` within the Kindle framework..
    type Target = (usize, usize);

    /// Core abstraction for `build_args` within the Kindle framework..
    fn build_args(target: Self::Target) -> Self::BuildArg {
        alloc::vec![target.0, target.1]
    }
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
/// Core abstraction for `Embedding` within the Kindle framework..
pub struct Embedding<S: EmbeddingShape, B: Backend> {
    /// Core abstraction for `weight` within the Kindle framework..
    pub weight: Param<(S::Vocab, S::Embed), B>,
}

impl<S: EmbeddingShape, B: Backend> Embedding<S, B>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Vocab, S::Embed): Shape<Arg = S::BuildArg>,
{
    /// Core abstraction for `new_with` within the Kindle framework..
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
    /// Core abstraction for `new` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Tensor<<InS as AppendDim<S::Embed>>::Output, B>;
    /// Core abstraction for `Error` within the Kindle framework..
    type Error = Error;

    #[inline]
    /// Core abstraction for `forward` within the Kindle framework..
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
