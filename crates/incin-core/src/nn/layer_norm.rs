use crate::backend_authoring::SupportsDType;
use crate::dist::placement::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::{LayerNormAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::module::ShapeInfo;
use crate::nn::{Module, Param};
use crate::shapes::{Dim, Dyn, DynShape, Shape, ShapeValue};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use alloc::string::String;
use core::marker::PhantomData;

/// A shape marker trait specifying a [`LayerNorm`] layer's normalized
/// dimension size. The typical usage is `(Channels,)` for a static layer,
/// or `Dyn` for a runtime-determined size.
pub trait LayerNormShape: Shape + DynShape {
    /// The size of the dimension being normalized (the weight/bias length).
    type Channels: Dim;
    /// The shape argument type used to construct the weight/bias tensors.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// The static shape type of the weight and bias parameters.
    type ParamShape: Shape<Arg = Self::BuildArg> + DynShape;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> LayerNormShape for crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil> {
    type Channels = C;
    type BuildArg = (<C as Dim>::Arg, ());
    type ParamShape = crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil>;

    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target, ())
    }
}

impl LayerNormShape for Dyn {
    type Channels = usize;
    type BuildArg = alloc::vec::Vec<usize>;
    type ParamShape = Dyn;
    fn build_args(target: usize) -> Self::BuildArg {
        alloc::vec![target]
    }
}

use crate::nn::param::{Frozen, TrainState, Trainable};

#[derive(Debug)]
#[incin_macros::module(internal)]
/// Layer normalization: normalizes the last dimension to zero mean and
/// unit variance, then applies a learnable affine `weight`/`bias`.
pub struct LayerNorm<
    S: LayerNormShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The learnable weight matrix parameter.
    pub weight: Param<S::ParamShape, B, K, Train>,
    /// The optional learnable bias vector parameter.
    pub bias: Param<S::ParamShape, B, K, Train>,
    #[module(ignore)]
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<(S, B, K, Train)>,
}

impl<S, B, K, Train> ShapeInfo for LayerNorm<S, B, K, Train>
where
    S: LayerNormShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
{
    fn shape_info(&self) -> Option<String> {
        None
    }
}

impl<S: LayerNormShape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    LayerNorm<S, B, K, Train>
{
    /// Constructs a LayerNorm from raw weight/bias parameters and epsilon.
    pub fn from_raw_parts(
        weight: Param<S::ParamShape, B, K, Train>,
        bias: Param<S::ParamShape, B, K, Train>,
        eps: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            eps,
            _phantom: PhantomData,
        }
    }

    /// Freezes this layer's parameters.
    pub fn freeze(self) -> LayerNorm<S, B, K, Frozen> {
        LayerNorm {
            weight: self.weight.freeze(),
            bias: self.bias.freeze(),
            eps: self.eps,
            _phantom: PhantomData,
        }
    }

    /// Unfreezes this layer's parameters.
    pub fn unfreeze(self) -> LayerNorm<S, B, K, Trainable> {
        LayerNorm {
            weight: self.weight.unfreeze(),
            bias: self.bias.unfreeze(),
            eps: self.eps,
            _phantom: PhantomData,
        }
    }
}

/// A builder for constructing a [`LayerNorm`] layer with a target.
#[derive(Debug, Clone)]
pub struct LayerNormBuilder<S: LayerNormShape, Train: TrainState = Trainable> {
    pub shape: ShapeValue<S>,
    pub eps: f32,
    pub weight_init: crate::nn::init::Init,
    pub bias_init: crate::nn::init::Init,
    pub _train: PhantomData<Train>,
}

/// Creates a new builder for a [`LayerNorm`] layer with shape `shape` and epsilon `eps`.
pub fn layer_norm<S: LayerNormShape>(shape: ShapeValue<S>, eps: f32) -> LayerNormBuilder<S> {
    LayerNormBuilder {
        shape,
        eps,
        weight_init: crate::nn::init::ones(),
        bias_init: crate::nn::init::zeros(),
        _train: PhantomData,
    }
}

impl<S: LayerNormShape, Train: TrainState> LayerNormBuilder<S, Train> {
    /// Configures weight initialization.
    pub fn weight_init(mut self, init: crate::nn::init::Init) -> Self {
        self.weight_init = init;
        self
    }

    /// Configures bias initialization.
    pub fn bias_init(mut self, init: crate::nn::init::Init) -> Self {
        self.bias_init = init;
        self
    }

    /// Marks the resulting layer as frozen (non-trainable).
    pub fn frozen(self) -> LayerNormBuilder<S, Frozen> {
        LayerNormBuilder {
            shape: self.shape,
            eps: self.eps,
            weight_init: self.weight_init,
            bias_init: self.bias_init,
            _train: PhantomData,
        }
    }
}

impl<
    S: LayerNormShape,
    B: crate::tensor::backend::VariableBackend
        + crate::tensor::backend::SupportsDType<K>
        + crate::nn::param::ParameterInit<K>,
    K: DType,
> LayerNorm<S, B, K, Trainable>
where
    B: SupportsDType<K>,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::Channels as Dim>::Arg,
                <K as DType>::Arg,
                <B::Device as Device>::Arg,
                f32,
            )>,
    {
        let (channels, dtype, device, eps) = args.into_layer_arg();
        let shape = S::build_args(channels);
        let weight = Param::<S::ParamShape, B, K, Trainable>::ones_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape: shape.clone(),
                dtype: dtype.clone(),
                device: device.clone(),
                grad: (),
            },
        )?;
        let bias = Param::<S::ParamShape, B, K, Trainable>::zeros_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape,
                dtype,
                device,
                grad: (),
            },
        )?;
        Ok(Self {
            weight,
            bias,
            eps,
            _phantom: PhantomData,
        })
    }
}

impl<
    S: LayerNormShape,
    InS: Shape + DynShape + crate::shapes::EndsWith<S::Channels>,
    B: crate::tensor::backend::VariableBackend + crate::exec::Capabilities + Execute<op::LayerNorm>,
    K: DType,
    Train: TrainState,
> Module<Tensor<InS, B, K>> for LayerNorm<S, B, K, Train>
where
    <B as Execute<op::LayerNorm>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<InS, B, K>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<InS, B, K>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = self.bias.as_tensor()?;

        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(x.inner()),
            TensorHandle::from_storage::<B, K, Local>(weight.inner()),
            TensorHandle::from_storage::<B, K, Local>(bias.inner()),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let out_inner = dispatch::execute_shaped::<op::LayerNorm, B, InS>(
            &context,
            LayerNormAttributes {
                normalized_shape: weight.shape_buf().as_ref().to_vec(),
                epsilon: self.eps as f64,
                has_bias: true,
            },
            &inputs,
            &x._shape,
        )
        .map_err(crate::err::Error::from)?;

        Tensor::from_shape_value(
            out_inner.into(),
            x._shape.clone(),
            x._dtype.clone(),
            weight._device,
            *x.grad_field(),
        )
    }
}
