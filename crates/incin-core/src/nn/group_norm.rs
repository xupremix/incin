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

/// A shape marker trait specifying a [`GroupNorm`] layer's channel count.
/// The typical usage is `(Channels,)` for a static layer, or `Dyn` for a
/// runtime-determined size.
pub trait GroupNormShape: Shape + DynShape {
    /// The number of channels being normalized.
    type Channels: Dim;
    /// The shape argument type used to construct the weight/bias tensors.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// The static shape type of the affine weight and bias parameters.
    type ParamShape: Shape<Arg = Self::BuildArg> + DynShape;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> GroupNormShape for crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil> {
    type Channels = C;
    type BuildArg = (<C as Dim>::Arg, ());
    type ParamShape = crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil>;

    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target, ())
    }
}

impl GroupNormShape for Dyn {
    type Channels = usize;
    type BuildArg = alloc::vec::Vec<usize>;
    type ParamShape = Dyn;
    fn build_args(target: usize) -> Self::BuildArg {
        alloc::vec![target]
    }
}

/// Group Normalization, as described in [Group Normalization](https://arxiv.org/abs/1803.08494).
///
/// Normalizes each sample's channels in `num_groups` groups over the batch
/// and every trailing (spatial) axis, then applies an optional learnable
/// per-channel affine `weight`/`bias` (`affine = false` leaves both absent
/// from the state dict).
///
/// The normalization itself wraps the canonical `group_norm` tensor op; the
/// affine step broadcasts the `[C]` parameters to `[1, C, 1, ...]`, mirroring
/// the broadcast the batch-norm kernel applies internally. Like
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
/// use incin::nn::{GroupNorm, Module, collect_state};
/// use incin::prelude::*;
///
/// type Backend = CpuBackendImpl<Cpu>;
/// // Two channels in two groups: each channel normalizes alone.
/// let m = GroupNorm::<Dyn, Backend>::build((2usize, 2usize, 1e-5f32, true)).unwrap();
/// let x = Cpu.ones(vec![1, 2, 2]).unwrap();
/// let out = m.forward(x).unwrap();
/// // A constant input normalizes to (near) zero in every group.
/// assert!(out.to_vec1::<f32>().unwrap().iter().all(|v| v.abs() < 1e-4));
/// // The affine parameters are registered for `state_dict`.
/// assert_eq!(collect_state::<Backend, _>(&m).unwrap().len(), 2);
/// // Rank-1 inputs have no channel axis and are refused.
/// let bad = Cpu.ones(vec![4]).unwrap();
/// assert!(m.forward(bad).is_err());
/// ```
#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
pub struct GroupNorm<
    S: GroupNormShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The optional learnable per-channel scale parameter (`None` when built with `affine = false`).
    pub weight: Option<Param<S::ParamShape, B, K, Train>>,
    /// The optional learnable per-channel shift parameter (`None` when built with `affine = false`).
    pub bias: Option<Param<S::ParamShape, B, K, Train>>,
    #[module(ignore)]
    /// Number of groups the channels are split into; must divide the channel count.
    pub num_groups: usize,
    #[module(ignore)]
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<(B, K, Train)>,
}

/// Rejects a zero group count before any backend runs. A free function
/// (rather than an associated one) so it is callable without naming a
/// backend-parameterized layer type.
fn check_groups(num_groups: usize) -> Result<()> {
    if num_groups == 0 {
        return Err(Error::Shape(ShapeError::InvalidParameter {
            operation: OperationKind::GroupNorm,
            parameter: "num_groups",
            value: 0,
        }));
    }
    Ok(())
}

impl<S: GroupNormShape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    GroupNorm<S, B, K, Train>
{
    /// Constructs a GroupNorm from raw parts.
    pub fn from_raw_parts(
        weight: Option<Param<S::ParamShape, B, K, Train>>,
        bias: Option<Param<S::ParamShape, B, K, Train>>,
        num_groups: usize,
        eps: f32,
    ) -> Result<Self> {
        check_groups(num_groups)?;
        Ok(Self {
            weight,
            bias,
            num_groups,
            eps,
            _phantom: PhantomData,
        })
    }

    /// Freezes this layer's affine parameters (weight and bias).
    pub fn freeze(self) -> GroupNorm<S, B, K, Frozen> {
        GroupNorm {
            weight: self.weight.map(|w| w.freeze()),
            bias: self.bias.map(|b| b.freeze()),
            num_groups: self.num_groups,
            eps: self.eps,
            _phantom: PhantomData,
        }
    }

    /// Unfreezes this layer's affine parameters (weight and bias).
    pub fn unfreeze(self) -> GroupNorm<S, B, K, Trainable> {
        GroupNorm {
            weight: self.weight.map(|w| w.unfreeze()),
            bias: self.bias.map(|b| b.unfreeze()),
            num_groups: self.num_groups,
            eps: self.eps,
            _phantom: PhantomData,
        }
    }
}

impl<
    S: GroupNormShape,
    B: crate::tensor::backend::VariableBackend
        + crate::tensor::backend::SupportsDType<K>
        + crate::nn::param::ParameterInit<K>,
    K: DType,
> GroupNorm<S, B, K, Trainable>
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
        num_groups: usize,
        eps: f32,
        affine: bool,
    ) -> Result<Self> {
        check_groups(num_groups)?;
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
            num_groups,
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
                usize,
                f32,
                bool,
            )>,
    {
        let (channels, dtype, device, num_groups, eps, affine) = args.into_layer_arg();
        Self::build_full(channels, dtype, device, num_groups, eps, affine)
    }
}

impl<
    S: GroupNormShape,
    InS: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend
        + crate::exec::Capabilities
        + Execute<op::GroupNorm>
        + Execute<op::Mul>
        + Execute<op::Add>
        + Execute<op::ReshapeExact>,
    K: DType,
    Train: TrainState,
    L: Layout<InS>,
> Module<Tensor<InS, B, K, crate::tensor::grad::NoGrad, Local, L>> for GroupNorm<S, B, K, Train>
where
    <B as Execute<op::GroupNorm>>::Output: Into<B::Storage<K>>,
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
        check_groups(self.num_groups)?;
        let dims = x.dims();
        let rank = dims.as_ref().len();
        // The descriptor rule: a channel axis at position 1 needs rank >= 2.
        if rank < 2 {
            return Err(Error::Shape(ShapeError::RankMismatch {
                operation: OperationKind::GroupNorm,
                expected: RankExpectation::AtLeast(2),
                actual: rank,
            }));
        }
        let channels = dims.as_ref()[1];
        if channels % self.num_groups != 0 {
            return Err(Error::Shape(ShapeError::DimensionMismatch {
                operation: OperationKind::GroupNorm,
                axis: Axis::Named("channels"),
                lhs: channels,
                rhs: self.num_groups,
                constraint: DimensionConstraint::DivisibleBy,
            }));
        }

        let normed = x.group_norm(self.num_groups, f64::from(self.eps))?;
        let out_shape_dyn =
            ShapeValue::<Dyn>::try_new(x.shape_buf().clone()).map_err(Error::Shape)?;
        let mut current = normed.inner;
        let context = ExecutionContext::from_scope(B::default());

        // Batch norm's kernel reshapes `[C]` parameters to `[1, C, 1, ...]`
        // before broadcasting; the group-norm op carries no affine operands,
        // so this layer performs the same reshape explicitly at the storage
        // level, keeping the whole forward in `NoGrad` space.
        let mut broadcast_shape = alloc::vec![1usize; rank];
        broadcast_shape[1] = channels;
        let broadcast_shape = ShapeBuf::from_slice(&broadcast_shape);
        if let Some(weight) = self.weight.as_ref() {
            if weight.shape_dims() != alloc::vec![channels] {
                return Err(Error::Shape(ShapeError::DimensionMismatch {
                    operation: OperationKind::GroupNorm,
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
                    operation: OperationKind::GroupNorm,
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
        assert_eq!(<C2 as GroupNormShape>::build_args(()), ((), ()));
    }

    #[test]
    fn dyn_shape_build_args_carry_the_count() {
        assert_eq!(<Dyn as GroupNormShape>::build_args(4usize), alloc::vec![4]);
    }

    #[test]
    fn zero_groups_are_a_typed_refusal() {
        let err = check_groups(0).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::InvalidParameter {
                operation: OperationKind::GroupNorm,
                parameter: "num_groups",
                value: 0,
            })
        ));
    }

    #[test]
    fn nonzero_groups_pass_validation() {
        assert!(check_groups(1).is_ok());
        assert!(check_groups(32).is_ok());
    }
}
