use crate::backend_authoring::SupportsDType;
use crate::dist::placement::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::{BatchNormAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::param::{Frozen, TrainState, Trainable};
use crate::nn::{Buffer, Module, Param, TrainMode};
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation};
use crate::shapes::{Dim, Dyn, DynShape, Layout, Shape, ShapeError};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use core::marker::PhantomData;

/// A shape marker trait specifying a [`BatchNorm1d`] layer's feature
/// count. The typical usage is `(Features,)` for a static layer, or `Dyn`
/// for a runtime-determined size.
pub trait BatchNorm1dShape: Shape + DynShape {
    /// The number of features being normalized.
    type Channels: Dim;
    /// The shape argument type used to construct the weight/bias/
    /// running-stat tensors.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// The static shape type of the affine and running-stat parameters.
    type ParamShape: Shape<Arg = Self::BuildArg> + DynShape;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> BatchNorm1dShape for crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil> {
    type Channels = C;
    type BuildArg = (<C as Dim>::Arg, ());
    type ParamShape = crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil>;

    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target, ())
    }
}

impl BatchNorm1dShape for Dyn {
    type Channels = usize;
    type BuildArg = alloc::vec::Vec<usize>;
    type ParamShape = Dyn;
    fn build_args(target: usize) -> Self::BuildArg {
        alloc::vec![target]
    }
}

/// A 1D Batch Normalization layer, as described in [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
///
/// Mirrors [`BatchNorm2d`](crate::nn::BatchNorm2d) for rank-2 `[N, C]` and
/// rank-3 `[N, C, L]` inputs: each channel is normalized independently over
/// the batch (and length) axes, then scaled by `weight` and shifted by
/// `bias`.
///
/// * `affine = false` leaves `weight`/`bias` absent from the state dict.
/// * `track_running_stats = false` leaves `running_mean`/`running_var`
///   absent; evaluation then falls back to batch statistics, matching the
///   PyTorch behavior for untracked layers.
/// * Unlike `BatchNorm2d`, this layer honors [`TrainMode`]: training mode
///   normalizes by batch statistics, evaluation mode by the running ones.
///   Running statistics are never mutated by `forward` (they arrive as shared
///   references the execution contract does not carry mutations through).
///
/// The type parameter `S` is the shape marker.
#[derive(Debug, Clone)]
#[incin_macros::module(internal, no_train_mode)]
pub struct BatchNorm1d<
    S: BatchNorm1dShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The optional learnable per-channel scale (`None` when built with `affine = false`).
    pub weight: Option<Param<S::ParamShape, B, K, Train>>,
    /// The optional learnable per-channel shift (`None` when built with `affine = false`).
    pub bias: Option<Param<S::ParamShape, B, K, Train>>,
    /// Running mean buffer used during evaluation (`None` when built with `track_running_stats = false`).
    pub running_mean: Option<Buffer<S::ParamShape, B, K>>,
    /// Running variance buffer used during evaluation (`None` when built with `track_running_stats = false`).
    pub running_var: Option<Buffer<S::ParamShape, B, K>>,
    #[module(ignore)]
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f32,
    #[module(ignore)]
    /// Momentum factor for updating running statistics.
    pub momentum: f32,
    #[module(ignore)]
    /// Whether `forward` normalizes by batch statistics (`true`) or running ones (`false`).
    pub is_training: bool,
    #[module(ignore)]
    _phantom: PhantomData<(B, K, Train)>,
}

impl<S: BatchNorm1dShape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    BatchNorm1d<S, B, K, Train>
{
    /// Constructs a BatchNorm1d from raw parts. Starts in training mode.
    pub fn from_raw_parts(
        weight: Option<Param<S::ParamShape, B, K, Train>>,
        bias: Option<Param<S::ParamShape, B, K, Train>>,
        running_mean: Option<Buffer<S::ParamShape, B, K>>,
        running_var: Option<Buffer<S::ParamShape, B, K>>,
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
            is_training: true,
            _phantom: PhantomData,
        }
    }

    /// Freezes this layer's learnable parameters (weight and bias).
    pub fn freeze(self) -> BatchNorm1d<S, B, K, Frozen> {
        BatchNorm1d {
            weight: self.weight.map(|w| w.freeze()),
            bias: self.bias.map(|b| b.freeze()),
            running_mean: self.running_mean,
            running_var: self.running_var,
            eps: self.eps,
            momentum: self.momentum,
            is_training: self.is_training,
            _phantom: PhantomData,
        }
    }

    /// Unfreezes this layer's learnable parameters (weight and bias).
    pub fn unfreeze(self) -> BatchNorm1d<S, B, K, Trainable> {
        BatchNorm1d {
            weight: self.weight.map(|w| w.unfreeze()),
            bias: self.bias.map(|b| b.unfreeze()),
            running_mean: self.running_mean,
            running_var: self.running_var,
            eps: self.eps,
            momentum: self.momentum,
            is_training: self.is_training,
            _phantom: PhantomData,
        }
    }
}

impl<S: BatchNorm1dShape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    TrainMode for BatchNorm1d<S, B, K, Train>
{
    /// Directly sets `is_training`, which selects batch statistics
    /// (`true`) or running statistics (`false`) in the next `forward`.
    fn set_training(&mut self, training: bool) {
        self.is_training = training;
    }
}

impl<
    S: BatchNorm1dShape,
    B: crate::tensor::backend::VariableBackend
        + crate::tensor::backend::SupportsDType<K>
        + crate::nn::param::ParameterInit<K>,
    K: DType,
> BatchNorm1d<S, B, K, Trainable>
where
    B: SupportsDType<K>,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    /// Builds the layer from every argument stated explicitly.
    pub fn build_full(
        channels: <S::Channels as Dim>::Arg,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
        eps: f32,
        momentum: f32,
        affine: bool,
        track_running_stats: bool,
    ) -> Result<Self> {
        let shape = S::build_args(channels);
        let param_args = || crate::tensor::arg_into::TensorArgsData {
            shape: shape.clone(),
            dtype: dtype.clone(),
            device: device.clone(),
            grad: (),
        };
        let (weight, bias) = if affine {
            (
                Some(Param::<S::ParamShape, B, K, Trainable>::ones_raw(
                    param_args(),
                )?),
                Some(Param::<S::ParamShape, B, K, Trainable>::zeros_raw(
                    param_args(),
                )?),
            )
        } else {
            (None, None)
        };
        let (running_mean, running_var) = if track_running_stats {
            (
                Some(Buffer::<S::ParamShape, B, K>::zeros_raw(param_args())?),
                Some(Buffer::<S::ParamShape, B, K>::ones_raw(param_args())?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            is_training: true,
            _phantom: PhantomData,
        })
    }

    /// Builds the layer from channel arguments.
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::Channels as Dim>::Arg,
                <K as DType>::Arg,
                <B::Device as Device>::Arg,
                f32,
                f32,
                bool,
                bool,
            )>,
    {
        let (channels, dtype, device, eps, momentum, affine, track_running_stats) =
            args.into_layer_arg();
        Self::build_full(
            channels,
            dtype,
            device,
            eps,
            momentum,
            affine,
            track_running_stats,
        )
    }
}

impl<
    S: BatchNorm1dShape,
    InS: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + crate::exec::Capabilities + Execute<op::BatchNorm>,
    K: DType,
    Train: TrainState,
    L: Layout<InS>,
> Module<Tensor<InS, B, K, crate::tensor::grad::NoGrad, Local, L>> for BatchNorm1d<S, B, K, Train>
where
    <B as Execute<op::BatchNorm>>::Output: Into<B::Storage<K>>,
{
    /// `Dense`: every call dispatches and writes a fresh buffer.
    type Output = crate::shapes::Dense<InS, B, K, crate::tensor::grad::NoGrad, Local>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<InS, B, K, crate::tensor::grad::NoGrad, Local, L>,
    ) -> core::result::Result<Self::Output, Self::Error> {
        let dims = x.dims();
        let rank = dims.as_ref().len();
        if rank != 2 && rank != 3 {
            return Err(Error::Shape(ShapeError::RankMismatch {
                operation: OperationKind::BatchNorm,
                expected: RankExpectation::Between { min: 2, max: 3 },
                actual: rank,
            }));
        }
        let channels = dims.as_ref()[1];
        let check_len = |len: usize| -> Result<()> {
            if len == channels {
                Ok(())
            } else {
                Err(Error::Shape(ShapeError::DimensionMismatch {
                    operation: OperationKind::BatchNorm,
                    axis: Axis::Named("channels"),
                    lhs: len,
                    rhs: channels,
                    constraint: DimensionConstraint::Equal,
                }))
            }
        };

        let weight = self
            .weight
            .as_ref()
            .map(|w| w.as_tensor())
            .transpose()?
            .map(|t| t.into_dyn());
        let bias = self
            .bias
            .as_ref()
            .map(|b| b.as_tensor())
            .transpose()?
            .map(|t| t.into_dyn());
        let running_mean = self
            .running_mean
            .as_ref()
            .map(|m| m.as_tensor())
            .transpose()?
            .map(|t| t.into_dyn());
        let running_var = self
            .running_var
            .as_ref()
            .map(|v| v.as_tensor())
            .transpose()?
            .map(|t| t.into_dyn());
        if let Some(param) = weight.as_ref() {
            check_len(param.dims().as_ref().first().copied().unwrap_or(0))?;
        }
        if let Some(param) = bias.as_ref() {
            check_len(param.dims().as_ref().first().copied().unwrap_or(0))?;
        }
        if let Some(param) = running_mean.as_ref() {
            check_len(param.dims().as_ref().first().copied().unwrap_or(0))?;
        }
        if let Some(param) = running_var.as_ref() {
            check_len(param.dims().as_ref().first().copied().unwrap_or(0))?;
        }

        // Evaluation without tracked statistics falls back to batch
        // statistics, matching PyTorch for `track_running_stats = false`.
        let has_running = running_mean.is_some() && running_var.is_some();
        let training = self.is_training || !has_running;

        let mut inputs = alloc::vec::Vec::with_capacity(5);
        inputs.push(TensorHandle::from_storage::<B, K, Local>(&x.inner));
        if let Some(weight) = weight.as_ref() {
            inputs.push(TensorHandle::from_storage::<B, K, Local>(&weight.inner));
        }
        if let Some(bias) = bias.as_ref() {
            inputs.push(TensorHandle::from_storage::<B, K, Local>(&bias.inner));
        }
        if !training {
            if let Some(running_mean) = running_mean.as_ref() {
                inputs.push(TensorHandle::from_storage::<B, K, Local>(
                    &running_mean.inner,
                ));
            }
            if let Some(running_var) = running_var.as_ref() {
                inputs.push(TensorHandle::from_storage::<B, K, Local>(
                    &running_var.inner,
                ));
            }
        }
        let context = ExecutionContext::from_scope(B::default()).with_training(training);
        let out = dispatch::execute_shaped::<op::BatchNorm, B, InS>(
            &context,
            BatchNormAttributes {
                epsilon: f64::from(self.eps),
                momentum: f64::from(self.momentum),
                training,
                has_weight: weight.is_some(),
                has_bias: bias.is_some(),
                has_running_mean: !training && running_mean.is_some(),
                has_running_variance: !training && running_var.is_some(),
            },
            &inputs,
            &x._shape,
        )
        .map_err(crate::err::Error::from)?;
        Tensor::from_shape_value(
            out.into(),
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad,
        )
    }
}
