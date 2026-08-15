use crate::backend_authoring::SupportsDType;
use crate::dist::placement::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::{NoAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::{Module, Param};
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;
use crate::shapes::{AppendDim, Dim, Dyn, DynShape, Shape, ShapeValue};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;

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
    /// The static shape type of the embedding weight parameter.
    type WeightShape: Shape<Arg = Self::BuildArg> + DynShape;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: (<Self::Vocab as Dim>::Arg, <Self::Embed as Dim>::Arg))
    -> Self::BuildArg;
}

impl<V: Dim, E: Dim> EmbeddingShape
    for crate::shapes::shape::DimCons<
        V,
        crate::shapes::shape::DimCons<E, crate::shapes::shape::Nil>,
    >
{
    type Vocab = V;
    type Embed = E;
    type BuildArg = (<V as Dim>::Arg, (<E as Dim>::Arg, ()));
    type WeightShape = crate::shapes::shape::DimCons<
        V,
        crate::shapes::shape::DimCons<E, crate::shapes::shape::Nil>,
    >;

    fn build_args(
        target: (<Self::Vocab as Dim>::Arg, <Self::Embed as Dim>::Arg),
    ) -> Self::BuildArg {
        (target.0, (target.1, ()))
    }
}

impl EmbeddingShape for Dyn {
    /// The vocabulary size, resolved at runtime.
    type Vocab = usize;
    /// The embedding dimension, resolved at runtime.
    type Embed = usize;
    /// The weight tensor's shape as a `Vec<usize>`.
    type BuildArg = alloc::vec::Vec<usize>;
    type WeightShape = Dyn;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::Vocab as Dim>::Arg, <Self::Embed as Dim>::Arg),
    ) -> Self::BuildArg {
        alloc::vec![target.0, target.1]
    }
}

use crate::nn::param::{Frozen, TrainState, Trainable};
use core::marker::PhantomData;

#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
/// An embedding table: maps integer token ids to dense vectors via row
/// lookup in a learnable `[vocab_size, embed_dim]` weight matrix.
pub struct Embedding<
    S: EmbeddingShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The learnable weight matrix parameter.
    pub weight: Param<S::WeightShape, B, K, Train>,
}

impl<S: EmbeddingShape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    Embedding<S, B, K, Train>
{
    /// Constructs an Embedding from a raw weight parameter.
    pub fn from_raw_parts(weight: Param<S::WeightShape, B, K, Train>) -> Self {
        Self { weight }
    }

    /// Freezes this layer's parameters.
    pub fn freeze(self) -> Embedding<S, B, K, Frozen> {
        Embedding {
            weight: self.weight.freeze(),
        }
    }

    /// Unfreezes this layer's parameters.
    pub fn unfreeze(self) -> Embedding<S, B, K, Trainable> {
        Embedding {
            weight: self.weight.unfreeze(),
        }
    }
}

/// A builder for constructing an [`Embedding`] layer with a target.
#[derive(Debug, Clone)]
pub struct EmbeddingBuilder<S: EmbeddingShape, Train: TrainState = Trainable> {
    pub shape: ShapeValue<S>,
    pub weight_init: crate::nn::init::Init,
    pub _train: PhantomData<Train>,
}

/// Creates a new builder for an [`Embedding`] layer with shape `shape`.
pub fn embedding<S: EmbeddingShape>(shape: ShapeValue<S>) -> EmbeddingBuilder<S> {
    EmbeddingBuilder {
        shape,
        weight_init: crate::nn::init::Init::Randn,
        _train: PhantomData,
    }
}

impl<S: EmbeddingShape, Train: TrainState> EmbeddingBuilder<S, Train> {
    /// Configures weight initialization.
    pub fn weight_init(mut self, init: crate::nn::init::Init) -> Self {
        self.weight_init = init;
        self
    }

    /// Marks the resulting layer as frozen (non-trainable).
    pub fn frozen(self) -> EmbeddingBuilder<S, Frozen> {
        EmbeddingBuilder {
            shape: self.shape,
            weight_init: self.weight_init,
            _train: PhantomData,
        }
    }
}

impl<
    S: EmbeddingShape,
    B: crate::tensor::backend::VariableBackend
        + crate::tensor::backend::SupportsDType<K>
        + crate::nn::param::ParameterInit<K>,
    K: DType,
> Embedding<S, B, K, Trainable>
where
    B: SupportsDType<K>,
{
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::Vocab as Dim>::Arg,
                <S::Embed as Dim>::Arg,
                <K as DType>::Arg,
                <B::Device as Device>::Arg,
            )>,
    {
        let (vocab, embed, dtype, device) = args.into_layer_arg();
        let shape = S::build_args((vocab, embed));
        let weight = Param::<S::WeightShape, B, K, Trainable>::new_init_raw(
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

impl<
    S: EmbeddingShape,
    InS: Shape + DynShape + AppendDim<S::Embed>,
    B,
    InK: DType,
    K: DType,
    Train: TrainState,
> Module<Tensor<InS, B, InK>> for Embedding<S, B, K, Train>
where
    B: crate::tensor::backend::VariableBackend
        + crate::exec::Capabilities
        + Execute<op::EmbeddingExact>,
    <B as Execute<op::EmbeddingExact>>::Output: Into<B::Storage<K>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<<InS as AppendDim<S::Embed>>::Output, B, K>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<InS, B, InK>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let x_handle = TensorHandle::from_storage::<B, InK, Local>(x.inner());
        let weight_handle = TensorHandle::from_storage::<B, K, Local>(weight.inner());
        let context = ExecutionContext::from_scope(B::default());
        let mut dims = x.shape_buf().as_ref().to_vec();
        // Read the validated weight extent so named static axes and runtime
        // axes share the same execution path.
        dims.push(weight.shape_buf()[1]);
        let shape = shape_buf_from_dims::<<InS as AppendDim<S::Embed>>::Output>(
            OperationKind::Embedding,
            &dims,
        )?;
        let output_shape =
            crate::shapes::ShapeValue::<<InS as AppendDim<S::Embed>>::Output>::try_new(shape)
                .map_err(crate::err::Error::Shape)?;
        let out = dispatch::execute_shaped::<
            op::EmbeddingExact,
            B,
            <InS as AppendDim<S::Embed>>::Output,
        >(
            &context,
            NoAttributes,
            &[x_handle, weight_handle],
            &output_shape,
        )
        .map_err(crate::err::Error::from)?;

        Tensor::from_shape_value(
            out.into(),
            output_shape,
            weight._dtype.clone(),
            x._device.clone(),
            core::marker::PhantomData,
        )
    }
}
