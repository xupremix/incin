use crate::backend_authoring::SupportsDType;
use crate::dist::placement::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::{NoAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::param::{Frozen, TrainState, Trainable};
use crate::nn::{Module, Param};
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation};
use crate::shapes::{Dim, Dyn, DynShape, Layout, Shape, ShapeBuf, ShapeError, ShapeValue};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use core::marker::PhantomData;

/// A shape marker trait specifying an [`InstanceNorm`] layer's channel
/// count. The typical usage is `(Channels,)` for a static layer, or `Dyn`
/// for a runtime-determined size.
pub trait InstanceNormShape: Shape + DynShape {
    /// The number of channels being normalized.
    type Channels: Dim;
    /// The shape argument type used to construct the weight/bias tensors.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// The static shape type of the affine weight and bias parameters.
    type ParamShape: Shape<Arg = Self::BuildArg> + DynShape;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> InstanceNormShape for crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil> {
    type Channels = C;
    type BuildArg = (<C as Dim>::Arg, ());
    type ParamShape = crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil>;

    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target, ())
    }
}

impl InstanceNormShape for Dyn {
    type Channels = usize;
    type BuildArg = alloc::vec::Vec<usize>;
    type ParamShape = Dyn;
    fn build_args(target: usize) -> Self::BuildArg {
        alloc::vec![target]
    }
}

/// Instance Normalization, as described in [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).
///
/// Normalizes each channel of each sample independently over the batch and
/// every trailing (spatial) axis, then applies an optional learnable
/// per-channel affine `weight`/`bias` (`affine = false` leaves both absent
/// from the state dict).
///
/// This is `group_norm` with one group per channel, so it follows the same
/// descriptor rule: inputs are `[batch, channels, ...]` with rank >= 2,
/// covering both the 1D (`[N, C, L]`) and 2D (`[N, C, H, W]`) forms. Like
/// [`BatchNorm2d`](crate::nn::BatchNorm2d) this layer is inference-scoped:
/// inputs are `NoGrad` and no running statistics are kept.
///
/// The type parameter `S` is the shape marker.
///
/// ## Example
///
/// ```rust
/// # extern crate incin_core as incin;
/// # use incin_backends::prelude::*;
/// # use incin_backends::cpu::CpuBackendImpl;
/// # use incin_core::tensor::device::Cpu;
/// use incin::nn::{InstanceNorm, Module};
/// use incin::prelude::*;
///
/// type Backend = CpuBackendImpl<Cpu>;
/// // Without affine parameters the layer is the bare `instance_norm` op.
/// let m = InstanceNorm::<Dyn, Backend>::build((4usize, 1e-5f32, false)).unwrap();
/// let x = Cpu.ones(vec![1, 4, 2, 2]).unwrap();
/// let got = m.forward(x).unwrap().to_vec1::<f32>().unwrap();
/// let x = Cpu.ones(vec![1, 4, 2, 2]).unwrap();
/// let expected = x.instance_norm(1e-5).unwrap().to_vec1::<f32>().unwrap();
/// assert_eq!(got, expected);
/// // A constant input normalizes to (near) zero per channel.
/// assert!(got.iter().all(|v| v.abs() < 1e-4));
/// ```
#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
pub struct InstanceNorm<
    S: InstanceNormShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The optional learnable per-channel scale parameter (`None` when built with `affine = false`).
    pub weight: Option<Param<S::ParamShape, B, K, Train>>,
    /// The optional learnable per-channel shift parameter (`None` when built with `affine = false`).
    pub bias: Option<Param<S::ParamShape, B, K, Train>>,
    #[module(ignore)]
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<(B, K, Train)>,
}

impl<S: InstanceNormShape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    InstanceNorm<S, B, K, Train>
{
    /// Constructs an InstanceNorm from raw parts.
    pub fn from_raw_parts(
        weight: Option<Param<S::ParamShape, B, K, Train>>,
        bias: Option<Param<S::ParamShape, B, K, Train>>,
        eps: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            eps,
            _phantom: PhantomData,
        }
    }

    /// Freezes this layer's affine parameters (weight and bias).
    pub fn freeze(self) -> InstanceNorm<S, B, K, Frozen> {
        InstanceNorm {
            weight: self.weight.map(|w| w.freeze()),
            bias: self.bias.map(|b| b.freeze()),
            eps: self.eps,
            _phantom: PhantomData,
        }
    }

    /// Unfreezes this layer's affine parameters (weight and bias).
    pub fn unfreeze(self) -> InstanceNorm<S, B, K, Trainable> {
        InstanceNorm {
            weight: self.weight.map(|w| w.unfreeze()),
            bias: self.bias.map(|b| b.unfreeze()),
            eps: self.eps,
            _phantom: PhantomData,
        }
    }
}

impl<
    S: InstanceNormShape,
    B: crate::tensor::backend::VariableBackend
        + crate::tensor::backend::SupportsDType<K>
        + crate::nn::param::ParameterInit<K>,
    K: DType,
> InstanceNorm<S, B, K, Trainable>
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
        affine: bool,
    ) -> Result<Self> {
        let shape = S::build_args(channels);
        let (weight, bias) = if affine {
            (
                Some(Param::<S::ParamShape, B, K, Trainable>::ones_raw(
                    crate::tensor::arg_into::TensorArgsData {
                        shape: shape.clone(),
                        dtype: dtype.clone(),
                        device: device.clone(),
                        grad: (),
                    },
                )?),
                Some(Param::<S::ParamShape, B, K, Trainable>::zeros_raw(
                    crate::tensor::arg_into::TensorArgsData {
                        shape,
                        dtype,
                        device,
                        grad: (),
                    },
                )?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            weight,
            bias,
            eps,
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
                bool,
            )>,
    {
        let (channels, dtype, device, eps, affine) = args.into_layer_arg();
        Self::build_full(channels, dtype, device, eps, affine)
    }
}

impl<
    S: InstanceNormShape,
    InS: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend
        + crate::exec::Capabilities
        + Execute<op::InstanceNorm>
        + Execute<op::Mul>
        + Execute<op::Add>
        + Execute<op::ReshapeExact>,
    K: DType,
    Train: TrainState,
    L: Layout<InS>,
> Module<Tensor<InS, B, K, crate::tensor::grad::NoGrad, Local, L>> for InstanceNorm<S, B, K, Train>
where
    <B as Execute<op::InstanceNorm>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
{
    /// `Dense`: normalization rewrites every value into a fresh buffer.
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
        // The descriptor rule: a channel axis at position 1 needs rank >= 2.
        if rank < 2 {
            return Err(Error::Shape(ShapeError::RankMismatch {
                operation: OperationKind::InstanceNorm,
                expected: RankExpectation::AtLeast(2),
                actual: rank,
            }));
        }
        let channels = dims.as_ref()[1];

        let normed = x.instance_norm(f64::from(self.eps))?;
        let out_shape_dyn =
            ShapeValue::<Dyn>::try_new(x.shape_buf().clone()).map_err(Error::Shape)?;
        let mut current = normed.inner;
        let context = ExecutionContext::from_scope(B::default());

        // Same `[1, C, 1, ...]` broadcast reshape as [`GroupNorm`](crate::nn::GroupNorm):
        // the instance-norm op carries no affine operands.
        let mut broadcast_shape = alloc::vec![1usize; rank];
        broadcast_shape[1] = channels;
        let broadcast_shape = ShapeBuf::from_slice(&broadcast_shape);
        if let Some(weight) = self.weight.as_ref() {
            if weight.shape_dims() != alloc::vec![channels] {
                return Err(Error::Shape(ShapeError::DimensionMismatch {
                    operation: OperationKind::InstanceNorm,
                    axis: Axis::Named("channels"),
                    lhs: weight.shape_dims().first().copied().unwrap_or(0),
                    rhs: channels,
                    constraint: DimensionConstraint::Equal,
                }));
            }
            let param = weight.as_tensor()?;
            let scaled = crate::tensor::ops::manipulation::reshape_storage_exact::<B, K>(
                &param.inner,
                &broadcast_shape,
            )?;
            let inputs = [
                TensorHandle::from_storage::<B, K, Local>(&current),
                TensorHandle::from_storage::<B, K, Local>(&scaled),
            ];
            current = dispatch::execute_shaped::<op::Mul, B, Dyn>(
                &context,
                NoAttributes,
                &inputs,
                &out_shape_dyn,
            )
            .map_err(Error::from)?
            .into();
        }
        if let Some(bias) = self.bias.as_ref() {
            if bias.shape_dims() != alloc::vec![channels] {
                return Err(Error::Shape(ShapeError::DimensionMismatch {
                    operation: OperationKind::InstanceNorm,
                    axis: Axis::Named("channels"),
                    lhs: bias.shape_dims().first().copied().unwrap_or(0),
                    rhs: channels,
                    constraint: DimensionConstraint::Equal,
                }));
            }
            let param = bias.as_tensor()?;
            let shifted = crate::tensor::ops::manipulation::reshape_storage_exact::<B, K>(
                &param.inner,
                &broadcast_shape,
            )?;
            let inputs = [
                TensorHandle::from_storage::<B, K, Local>(&current),
                TensorHandle::from_storage::<B, K, Local>(&shifted),
            ];
            current = dispatch::execute_shaped::<op::Add, B, Dyn>(
                &context,
                NoAttributes,
                &inputs,
                &out_shape_dyn,
            )
            .map_err(Error::from)?
            .into();
        }

        Tensor::from_shape_value(
            current,
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::{DimCons, Nil};
    use typenum::consts::U2;

    type C2 = DimCons<U2, Nil>;

    #[test]
    fn static_shape_build_args_pair_the_channel() {
        assert_eq!(<C2 as InstanceNormShape>::build_args(()), ((), ()));
    }

    #[test]
    fn dyn_shape_build_args_carry_the_count() {
        assert_eq!(
            <Dyn as InstanceNormShape>::build_args(4usize),
            alloc::vec![4]
        );
    }
}
