use crate::dist::placement::Local;
use crate::exec::catalog::{BatchNormAttributes, Descriptor, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::{Buffer, Module, Param};
use crate::prelude::*;
use crate::tensor::backend::Execute;

use core::marker::PhantomData;

/// A shape marker trait specifying a [`BatchNorm2d`] layer's channel
/// count. The typical usage is `(Channels,)` for a static layer, or `Dyn`
/// for a runtime-determined size.
pub trait BatchNormShape: Shape + DynShape {
    /// The number of channels being normalized.
    type Channels: Dim;
    /// The shape argument type used to construct the weight/bias/
    /// running-stat tensors.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// The static shape type of the affine and running-stat parameters.
    type ParamShape: Shape<Arg = Self::BuildArg> + DynShape;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> BatchNormShape for crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil> {
    type Channels = C;
    type BuildArg = (<C as Dim>::Arg, ());
    type ParamShape = crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil>;

    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target, ())
    }
}

impl BatchNormShape for Dyn {
    type Channels = usize;
    type BuildArg = alloc::vec::Vec<usize>;
    type ParamShape = Dyn;
    fn build_args(target: usize) -> Self::BuildArg {
        alloc::vec![target]
    }
}

use crate::nn::param::{Frozen, TrainState, Trainable};

#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
/// A 2D Batch Normalization layer, as described in [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
///
/// Normalizes the input tensor across the batch and spatial dimensions for each channel independently,
/// applying learnable affine scaling (`weight`) and shift (`bias`) parameters.
///
/// The type parameter `S` is the shape marker.
///
/// ## Running Statistics
/// `running_mean` and `running_var` are non-trainable buffers updated during training (training mode
/// must be handled at the backend level). During inference, these stored statistics are used.
pub struct BatchNorm2d<S: BatchNormShape, B: Backend, K: DType = f32, Train: TrainState = Trainable>
{
    /// The learnable weight matrix parameter.
    pub weight: Param<S::ParamShape, B, K, Train>,
    /// The optional learnable bias vector parameter.
    pub bias: Param<S::ParamShape, B, K, Train>,
    /// Running mean buffer used during batch normalization inference.
    pub running_mean: Buffer<S::ParamShape, B, K>,
    /// Running variance buffer used during batch normalization inference.
    pub running_var: Buffer<S::ParamShape, B, K>,
    #[module(ignore)]
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f32,
    #[module(ignore)]
    /// Momentum factor for updating running statistics.
    pub momentum: f32,
    #[module(ignore)]
    _phantom: PhantomData<(B, K, Train)>,
}

impl<S: BatchNormShape, B: Backend, K: DType, Train: TrainState> BatchNorm2d<S, B, K, Train> {
    /// Constructs a BatchNorm2d from raw parts.
    pub fn from_raw_parts(
        weight: Param<S::ParamShape, B, K, Train>,
        bias: Param<S::ParamShape, B, K, Train>,
        running_mean: Buffer<S::ParamShape, B, K>,
        running_var: Buffer<S::ParamShape, B, K>,
        eps: f32,
        momentum: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            _phantom: PhantomData,
        }
    }

    /// Freezes this layer's learnable parameters (weight and bias).
    pub fn freeze(self) -> BatchNorm2d<S, B, K, Frozen> {
        BatchNorm2d {
            weight: self.weight.freeze(),
            bias: self.bias.freeze(),
            running_mean: self.running_mean,
            running_var: self.running_var,
            eps: self.eps,
            momentum: self.momentum,
            _phantom: PhantomData,
        }
    }

    /// Unfreezes this layer's learnable parameters (weight and bias).
    pub fn unfreeze(self) -> BatchNorm2d<S, B, K, Trainable> {
        BatchNorm2d {
            weight: self.weight.unfreeze(),
            bias: self.bias.unfreeze(),
            running_mean: self.running_mean,
            running_var: self.running_var,
            eps: self.eps,
            momentum: self.momentum,
            _phantom: PhantomData,
        }
    }
}

/// A builder for constructing a [`BatchNorm2d`] layer with a target.
#[derive(Debug, Clone)]
pub struct BatchNorm2dBuilder<S: BatchNormShape, Train: TrainState = Trainable> {
    pub shape: ShapeValue<S>,
    pub eps: f32,
    pub momentum: f32,
    pub weight_init: crate::nn::init::Init,
    pub bias_init: crate::nn::init::Init,
    pub running_mean_init: crate::nn::init::Init,
    pub running_var_init: crate::nn::init::Init,
    pub _train: PhantomData<Train>,
}

/// Creates a new builder for a [`BatchNorm2d`] layer with shape `shape`, epsilon `eps`, and momentum `momentum`.
pub fn batch_norm2d<S: BatchNormShape>(
    shape: ShapeValue<S>,
    eps: f32,
    momentum: f32,
) -> BatchNorm2dBuilder<S> {
    BatchNorm2dBuilder {
        shape,
        eps,
        momentum,
        weight_init: crate::nn::init::ones(),
        bias_init: crate::nn::init::zeros(),
        running_mean_init: crate::nn::init::zeros(),
        running_var_init: crate::nn::init::ones(),
        _train: PhantomData,
    }
}

impl<S: BatchNormShape, Train: TrainState> BatchNorm2dBuilder<S, Train> {
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
    pub fn frozen(self) -> BatchNorm2dBuilder<S, Frozen> {
        BatchNorm2dBuilder {
            shape: self.shape,
            eps: self.eps,
            momentum: self.momentum,
            weight_init: self.weight_init,
            bias_init: self.bias_init,
            running_mean_init: self.running_mean_init,
            running_var_init: self.running_var_init,
            _train: PhantomData,
        }
    }
}

impl<
    S: BatchNormShape,
    B: Backend
        + crate::tensor::backend::CreationOps<B>
        + crate::tensor::backend::FloatOps<B>
        + crate::tensor::backend::NumericOps<B>
        + crate::nn::param::ParameterInit<K>,
    K: DType,
> BatchNorm2d<S, B, K, Trainable>
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
                f32,
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (channels, dtype, device, eps, momentum) = args.into_layer_arg();
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
                shape: shape.clone(),
                dtype: dtype.clone(),
                device: device.clone(),
                grad: (),
            },
        )?;
        let running_mean =
            Buffer::<S::ParamShape, B, K>::zeros_raw(crate::tensor::arg_into::TensorArgsData {
                shape: shape.clone(),
                dtype: dtype.clone(),
                device: device.clone(),
                grad: (),
            })?;
        let running_var =
            Buffer::<S::ParamShape, B, K>::ones_raw(crate::tensor::arg_into::TensorArgsData {
                shape,
                dtype,
                device,
                grad: (),
            })?;
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            _phantom: PhantomData,
        })
    }
}

impl<
    S: BatchNormShape,
    InS: Shape + HasChannels2D<S::Channels>,
    B: Backend + crate::exec::Capabilities + Execute<op::BatchNorm>,
    K: DType,
    Train: TrainState,
> Module<Tensor<InS, B, K>> for BatchNorm2d<S, B, K, Train>
where
    <B as Execute<op::BatchNorm>>::Output: Into<B::Storage<K>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<InS, B, K>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<InS, B, K>) -> core::result::Result<Self::Output, Self::Error> {
        let weight = self.weight.as_tensor()?.into_dyn();
        let bias = self.bias.as_tensor()?.into_dyn();
        let running_mean = self.running_mean.as_tensor()?.into_dyn();
        let running_var = self.running_var.as_tensor()?.into_dyn();

        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(x.inner()),
            TensorHandle::from_storage::<B, K, Local>(weight.inner()),
            TensorHandle::from_storage::<B, K, Local>(bias.inner()),
            TensorHandle::from_storage::<B, K, Local>(running_mean.inner()),
            TensorHandle::from_storage::<B, K, Local>(running_var.inner()),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let out = dispatch::execute::<op::BatchNorm, B>(
            &context,
            BatchNormAttributes {
                epsilon: self.eps as f64,
                momentum: self.momentum as f64,
                training: false,
                has_weight: true,
                has_bias: true,
                has_running_mean: true,
                has_running_variance: true,
            },
            &inputs,
        )
        .map_err(crate::prelude::Error::from)?;
        Tensor::from_shape_value(
            out.into(),
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad,
        )
    }
}
