use crate::nn::{Module, Param};
use crate::prelude::*;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::field_from_dims;

/// A shape marker trait specifying an [`Embedding`] layer's vocabulary size
/// and embedding dimension, analogous to [`crate::nn::linear::LinearShape`].
/// The typical usage is `(Vocab, Embed)` for a static layer, or `Dyn` for
/// runtime-determined sizes.
pub trait EmbeddingShape: Shape + DynShape {
    /// The vocabulary size (number of distinct token ids).
    type Vocab: Dim;
    /// The embedding dimension (length of each row vector).
    type Embed: Dim;
    /// The shape argument type used to construct the weight tensor.
    type BuildArg: crate::tensor::arg_into::NotUnit;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: (<Self::Vocab as Dim>::Arg, <Self::Embed as Dim>::Arg))
    -> Self::BuildArg;
}

impl<V: Dim, E: Dim> EmbeddingShape for (V, E) {
    /// The vocabulary size.
    type Vocab = V;
    /// The embedding dimension.
    type Embed = E;
    /// A `(vocab_arg, embed_arg)` pair.
    type BuildArg = (<V as Dim>::Arg, <E as Dim>::Arg);

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::Vocab as Dim>::Arg, <Self::Embed as Dim>::Arg),
    ) -> Self::BuildArg {
        target
    }
}

impl EmbeddingShape for Dyn {
    /// The vocabulary size, resolved at runtime.
    type Vocab = usize;
    /// The embedding dimension, resolved at runtime.
    type Embed = usize;
    /// The weight tensor's shape as a `Vec<usize>`.
    type BuildArg = alloc::vec::Vec<usize>;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::Vocab as Dim>::Arg, <Self::Embed as Dim>::Arg),
    ) -> Self::BuildArg {
        alloc::vec![target.0, target.1]
    }
}

#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
/// An embedding table: maps integer token ids to dense vectors via row
/// lookup in a learnable `[vocab_size, embed_dim]` weight matrix.
pub struct Embedding<S: EmbeddingShape, B: Backend> {
    /// The learnable weight matrix parameter.
    pub weight: Param<(S::Vocab, S::Embed), B>,
}

impl<S: EmbeddingShape, B: Backend> Embedding<S, B>
where
    B: SupportsDType<B::FloatElem>,
    (S::Vocab, S::Embed): Shape<Arg = S::BuildArg>,
{
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::Vocab as Dim>::Arg,
                <S::Embed as Dim>::Arg,
                <B::FloatElem as DType>::Arg,
                <B::Device as Device>::Arg,
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (vocab, embed, dtype, device) = args.into_layer_arg();
        let shape = S::build_args((vocab, embed));
        let weight = Param::<(S::Vocab, S::Embed), B>::new_init_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape,
                dtype,
                device,
                grad: (),
            },
            crate::nn::init::Init::Randn,
        )?;
        Ok(Self { weight })
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

        let mut dims = <InS as Shape>::dims(x.shape_field()).into();
        dims.push(<S::Embed as typenum::Unsigned>::USIZE);

        let shape = field_from_dims::<<InS as AppendDim<S::Embed>>::Output>(
            OperationKind::Embedding,
            &dims,
        )?;

        Tensor::from_parts(
            out,
            shape,
            x._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        )
    }
}
