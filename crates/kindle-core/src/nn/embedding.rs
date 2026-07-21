use crate::nn::{Module, Param};
use crate::prelude::*;

/// `EmbeddingShape`.
pub trait EmbeddingShape: Shape + DynShape {
    /// `Vocab`.
    type Vocab: Dim;
    /// `Embed`.
    type Embed: Dim;
    /// `BuildArg`.
    type BuildArg: crate::tensor::arg_into::NotUnit;
    /// The runtime arguments needed to instantiate this layer.
    type Target;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<V: Dim, E: Dim> EmbeddingShape for (V, E) {
    /// `Vocab`.
    type Vocab = V;
    /// `Embed`.
    type Embed = E;
    /// `BuildArg`.
    type BuildArg = (<V as Dim>::Arg, <E as Dim>::Arg);
    /// The runtime arguments needed to instantiate this layer.
    type Target = (<V as Dim>::Arg, <E as Dim>::Arg);

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: Self::Target) -> Self::BuildArg {
        target
    }
}

impl EmbeddingShape for Dyn {
    /// `Vocab`.
    type Vocab = usize;
    /// `Embed`.
    type Embed = usize;
    /// `BuildArg`.
    type BuildArg = alloc::vec::Vec<usize>;
    /// The runtime arguments needed to instantiate this layer.
    type Target = (usize, usize);

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: Self::Target) -> Self::BuildArg {
        alloc::vec![target.0, target.1]
    }
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
/// `Embedding`.
pub struct Embedding<S: EmbeddingShape, B: Backend> {
    /// The learnable weight matrix parameter.
    pub weight: Param<(S::Vocab, S::Embed), B>,
}

impl<S: EmbeddingShape, B: Backend> Embedding<S, B>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Vocab, S::Embed): Shape<Arg = S::BuildArg>,
{
    /// Creates a new instance with explicitly provided shape arguments.
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
    /// Creates a new instance with default (statically inferred) shape arguments.
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
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<<InS as AppendDim<S::Embed>>::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
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
