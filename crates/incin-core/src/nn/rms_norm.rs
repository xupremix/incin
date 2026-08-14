use crate::exec::catalog::{Descriptor, op};
use crate::nn::{Module, Param};
use crate::prelude::{Backend, Device, DType, Dim, DynShape, Error, Result, Shape, ShapeBuf, ShapeError, ShapeValue, SupportsDType, Tensor, Dyn};
use alloc::string::String;
use crate::shapes::idx::{FromEnd, Here};
use crate::shapes::shape_ops::ReduceKeepAt;
use crate::tensor::backend::Execute;
use core::marker::PhantomData;

/// Shape traits for RMSNorm.
pub trait RMSNormShape: Shape + DynShape {
    type Channels: Dim;
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    type ParamShape: Shape<Arg = Self::BuildArg> + DynShape;
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> RMSNormShape for crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil> {
    type Channels = C;
    type BuildArg = (<C as Dim>::Arg, ());
    type ParamShape = crate::shapes::shape::DimCons<C, crate::shapes::shape::Nil>;

    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target, ())
    }
}

impl RMSNormShape for Dyn {
    type Channels = usize;
    type BuildArg = alloc::vec::Vec<usize>;
    type ParamShape = Dyn;
    fn build_args(target: usize) -> Self::BuildArg {
        alloc::vec![target]
    }
}

use crate::nn::param::{Frozen, TrainState, Trainable};

/// Root Mean Square Normalization (RMSNorm).
///
/// RMSNorm normalizes the input over the channel dimension without centering the mean,
/// improving training speed and stability. Widely used in modern LLMs (e.g. LLaMA).
#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
pub struct RMSNorm<S: RMSNormShape, B: Backend, K: DType = f32, Train: TrainState = Trainable> {
    pub weight: Param<S::ParamShape, B, K, Train>,
    #[module(ignore)]
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<(S, B, K, Train)>,
}

impl<S: RMSNormShape, B: Backend, K: DType, Train: TrainState> RMSNorm<S, B, K, Train> {
    /// Constructs an RMSNorm from a raw weight parameter and epsilon.
    pub fn from_raw_parts(weight: Param<S::ParamShape, B, K, Train>, eps: f32) -> Self {
        Self {
            weight,
            eps,
            _phantom: PhantomData,
        }
    }

    /// Freezes this layer's parameters.
    pub fn freeze(self) -> RMSNorm<S, B, K, Frozen> {
        RMSNorm {
            weight: self.weight.freeze(),
            eps: self.eps,
            _phantom: PhantomData,
        }
    }

    /// Unfreezes this layer's parameters.
    pub fn unfreeze(self) -> RMSNorm<S, B, K, Trainable> {
        RMSNorm {
            weight: self.weight.unfreeze(),
            eps: self.eps,
            _phantom: PhantomData,
        }
    }
}

/// A builder for constructing an [`RMSNorm`] layer with a target.
#[derive(Debug, Clone)]
pub struct RMSNormBuilder<S: RMSNormShape, Train: TrainState = Trainable> {
    pub shape: ShapeValue<S>,
    pub eps: f32,
    pub weight_init: crate::nn::init::Init,
    pub _train: PhantomData<Train>,
}

/// Creates a new builder for an [`RMSNorm`] layer with shape `shape` and epsilon `eps`.
pub fn rms_norm<S: RMSNormShape>(shape: ShapeValue<S>, eps: f32) -> RMSNormBuilder<S> {
    RMSNormBuilder {
        shape,
        eps,
        weight_init: crate::nn::init::ones(),
        _train: PhantomData,
    }
}

impl<S: RMSNormShape, Train: TrainState> RMSNormBuilder<S, Train> {
    /// Configures weight initialization.
    pub fn weight_init(mut self, init: crate::nn::init::Init) -> Self {
        self.weight_init = init;
        self
    }

    /// Marks the resulting layer as frozen (non-trainable).
    pub fn frozen(self) -> RMSNormBuilder<S, Frozen> {
        RMSNormBuilder {
            shape: self.shape,
            eps: self.eps,
            weight_init: self.weight_init,
            _train: PhantomData,
        }
    }
}

impl<
    S: RMSNormShape,
    B: Backend + crate::tensor::backend::SupportsDType<K> + crate::nn::param::ParameterInit<K>,
    K: DType,
> RMSNorm<S, B, K, Trainable>
{
    pub fn build<A>(args: A) -> Result<Self>
    where
        B: SupportsDType<K>,
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::Channels as Dim>::Arg,
                <K as DType>::Arg,
                <B::Device as Device>::Arg,
                f32,
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (channels, dtype, device, eps) = args.into_layer_arg();
        let shape = S::build_args(channels);
        let weight = Param::<S::ParamShape, B, K, Trainable>::ones_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape,
                dtype,
                device,
                grad: (),
            },
        )?;
        Ok(Self {
            weight,
            eps,
            _phantom: PhantomData,
        })
    }
}

impl<
    S: RMSNormShape,
    InS: Shape + DynShape + crate::shapes::EndsWith<S::Channels> + ReduceKeepAt<FromEnd<Here>>,
    B: Backend
        + crate::exec::Capabilities
        + Execute<op::Mul>
        + Execute<op::Div>
        + Execute<op::DivScalar>
        + Execute<op::Sqrt>
        + Execute<op::SumKeepDim>
        + Execute<op::AddScalar>,
    K: DType,
    Train: TrainState,
> Module<Tensor<InS, B, K>> for RMSNorm<S, B, K, Train>
where
    <InS as ReduceKeepAt<FromEnd<Here>>>::Output: DynShape,
    <B as Execute<op::SumKeepDim>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Div>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::DivScalar>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sqrt>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<K>>,
{
    type Output = Tensor<InS, B, K, Train::TensorGrad>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<InS, B, K>) -> core::result::Result<Self::Output, Error> {
        // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        let weight = self.weight.as_tensor()?.into_dyn();

        let channels = x.shape_buf().as_ref().last().copied().ok_or_else(|| {
            Error::Shape(crate::shapes::error::ShapeError::RankMismatch {
                operation: crate::shapes::error::OperationKind::MeanDim,
                expected: crate::shapes::error::RankExpectation::AtLeast(1),
                actual: 0,
            })
        })?;
        if channels == 0 {
            return Err(Error::Shape(
                crate::shapes::error::ShapeError::InvalidParameter {
                    operation: crate::shapes::error::OperationKind::MeanDim,
                    parameter: "channels",
                    value: channels,
                },
            ));
        }

        // x^2
        let x_sq = x.mul(&x)?;

        // mean(x^2)
        let mean_sq = x_sq
            .sum_keepdim::<FromEnd<Here>>()?
            .div_scalar(channels as f64)?
            .into_dyn();

        // mean(x^2) + eps
        let var = mean_sq.add_scalar(self.eps)?;

        // sqrt(mean(x^2) + eps)
        let std = var.sqrt()?;

        // x / std
        let dyn_x = x.into_dyn();
        let normed = dyn_x.broadcast_div(&std)?;

        // normed * weight
        let out = normed.broadcast_mul(&weight)?;

        out.into_shape::<InS>()
    }
}
