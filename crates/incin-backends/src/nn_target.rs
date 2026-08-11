//! Central target-aware parameter materialization and NN layer initialization on [`TensorTarget`].

use incin_core::backend_authoring::{Execute, SupportsDType, op};
use incin_core::nn::batch_norm::{BatchNorm2d, BatchNorm2dBuilder, BatchNormShape};
use incin_core::nn::conv1d::{Conv1d, Conv1dBuilder, Conv1dShape};
use incin_core::nn::conv2d::{Conv2d, Conv2dBuilder, Conv2dShape};
use incin_core::nn::embedding::{Embedding, EmbeddingBuilder, EmbeddingShape};
use incin_core::nn::init::{Init, InitContext, InitPlan, ParameterRole};
use incin_core::nn::layer_norm::{LayerNorm, LayerNormBuilder, LayerNormShape};
use incin_core::nn::linear::{Linear, LinearBuilder, LinearShape};
use incin_core::nn::lstm::{LSTM, LSTMBuilder, LSTMCell, LSTMCellBuilder, LstmShape};
use incin_core::nn::optional::OptionalField;
use incin_core::nn::param::{Param, TrainState, Trainable};
use incin_core::nn::rms_norm::{RMSNorm, RMSNormBuilder, RMSNormShape};
use incin_core::nn::rnn::{RNN, RNNBuilder, RNNCell, RNNCellBuilder, RnnShape};
use incin_core::prelude::{
    Backend, DynShape, FloatDType, Result, Shape, ShapeBuf, ShapeValue, StorageBackend,
};

use crate::target::{GeneratedFill, TargetBackend, TargetExt, TensorTarget};

/// Build a target-side shape value from generated dimensions through the
/// canonical ShapeBuf validation boundary.
fn shape_value_from_dims<S: Shape>(dims: &[usize]) -> Result<ShapeValue<S>> {
    ShapeValue::try_new(ShapeBuf::from_slice(dims)).map_err(incin_core::prelude::Error::Shape)
}
type Shape2<A, B> =
    incin_core::shapes::DimCons<A, incin_core::shapes::DimCons<B, incin_core::shapes::Nil>>;

fn materialize_storage_plan<T, S>(
    target: &T,
    shape_val: ShapeValue<S>,
    init: Init,
    context: InitContext,
) -> Result<<TargetBackend<T> as Backend>::RawVar>
where
    T: TensorTarget + TargetExt + Clone,
    S: Shape + DynShape,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    let plan = init.plan(context)?;
    match plan {
        InitPlan::Zeros => {
            let tensor = target.state_tensor(shape_val, GeneratedFill::Zeros)?;
            let storage = tensor.into_inner();
            TargetBackend::<T>::var_from_tensor(&storage)
        }
        InitPlan::Ones => {
            let tensor = target.state_tensor(shape_val, GeneratedFill::Ones)?;
            let storage = tensor.into_inner();
            TargetBackend::<T>::var_from_tensor(&storage)
        }
        InitPlan::Constant(val) => {
            let tensor = target.state_tensor(shape_val, GeneratedFill::Ones)?;
            if val == 1.0 {
                let storage = tensor.into_inner();
                TargetBackend::<T>::var_from_tensor(&storage)
            } else {
                let scaled = tensor.mul_scalar(val)?.into_inner();
                TargetBackend::<T>::var_from_tensor(&scaled)
            }
        }
        InitPlan::Uniform { low, high } => {
            let tensor = target.state_tensor(shape_val, GeneratedFill::Uniform)?;
            let range = high - low;
            let scaled = if range != 1.0 {
                tensor.mul_scalar(range)?
            } else {
                tensor
            };
            let shifted = if low != 0.0 {
                scaled.add_scalar(low)?.into_inner()
            } else {
                scaled.into_inner()
            };
            TargetBackend::<T>::var_from_tensor(&shifted)
        }
        InitPlan::Normal { mean, std } => {
            let tensor = target.state_tensor(shape_val, GeneratedFill::Normal)?;
            let scaled = if std != 1.0 {
                tensor.mul_scalar(std)?
            } else {
                tensor
            };
            let shifted = if mean != 0.0 {
                scaled.add_scalar(mean)?.into_inner()
            } else {
                scaled.into_inner()
            };
            TargetBackend::<T>::var_from_tensor(&shifted)
        }
    }
}

/// Central target-aware parameter materializer.
pub fn materialize_parameter<T, S, Train>(
    target: &T,
    shape_val: ShapeValue<S>,
    init: Init,
    context: InitContext,
) -> Result<Param<S, TargetBackend<T>, T::ParameterDtype, Train>>
where
    T: TensorTarget + TargetExt + Clone,
    S: Shape + DynShape,
    Train: TrainState,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    // `ShapeValue` has already established the type/value relationship. Keep
    // that canonical ShapeBuf rather than reconstructing another shape
    // representation.
    let shape_buf = shape_val.shape_buf().clone();
    let raw_var = materialize_storage_plan(target, shape_val, init, context)?;
    let dtype_field = target.parameter_dtype_field();
    let device_field = <T::Device as incin_core::prelude::Device>::init(target.device_arg());

    Param::<S, TargetBackend<T>, T::ParameterDtype, Train>::from_parts_checked(
        raw_var,
        shape_buf,
        dtype_field,
        device_field,
    )
}

/// Central target-aware buffer materializer.
pub fn materialize_buffer<T, S>(
    target: &T,
    shape_val: ShapeValue<S>,
    init: Init,
    context: InitContext,
) -> Result<incin_core::nn::Buffer<S, TargetBackend<T>, T::ParameterDtype>>
where
    T: TensorTarget + TargetExt + Clone,
    S: Shape + DynShape,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    let shape_buf = shape_val.shape_buf().clone();
    let raw_var = materialize_storage_plan(target, shape_val, init, context)?;
    let dtype_field = target.parameter_dtype_field();
    let device_field = <T::Device as incin_core::prelude::Device>::init(target.device_arg());

    incin_core::nn::Buffer::<S, TargetBackend<T>, T::ParameterDtype>::from_parts_checked(
        raw_var,
        shape_buf,
        dtype_field,
        device_field,
    )
}

/// Extension trait for target-based initialization of layer builders.
pub trait InitOnTarget<T: TensorTarget> {
    /// The initialized layer output type.
    type Output;
    /// Initializes the layer on target `target`.
    fn init(self, target: &T) -> Result<Self::Output>;
}

impl<S, Bias, Train, T> InitOnTarget<T> for LinearBuilder<S, Bias, Train>
where
    S: LinearShape,
    Bias: OptionalField,
    Bias::Arg: Default,
    Train: TrainState,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = Linear<S, TargetBackend<T>, Bias, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let shape_val = self.shape;
        let dims = shape_val.dims();
        let in_features = dims[0];
        let out_features = dims[1];

        let w_dims = [out_features, in_features];
        let w_shape_val = shape_value_from_dims::<S::WeightShape>(&w_dims)?;

        let context_w = InitContext::new(ParameterRole::Weight).with_fan(in_features, out_features);
        let weight = materialize_parameter::<T, S::WeightShape, Train>(
            target,
            w_shape_val,
            self.weight_init,
            context_w,
        )?;

        let bias = if Bias::init(Default::default()) {
            let b_dims = [out_features];
            let b_shape_val = shape_value_from_dims::<S::BiasShape>(&b_dims)?;
            let context_b =
                InitContext::new(ParameterRole::Bias).with_fan(in_features, out_features);
            Some(materialize_parameter::<T, S::BiasShape, Train>(
                target,
                b_shape_val,
                self.bias_init,
                context_b,
            )?)
        } else {
            None
        };

        Ok(Linear::from_raw_parts(weight, bias))
    }
}

/// Direct `Linear::new_on_target` constructor extension trait for targets.
pub trait LinearNewOnTarget<S: LinearShape, T: TensorTarget> {
    /// The initialized layer output type.
    type Output;
    /// Direct linear layer construction on target.
    fn new_on_target(shape: ShapeValue<S>, target: &T) -> Result<Self::Output>;
}

impl<S, T> LinearNewOnTarget<S, T>
    for Linear<S, TargetBackend<T>, incin_core::nn::optional::True, T::ParameterDtype, Trainable>
where
    S: LinearShape,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output =
        Linear<S, TargetBackend<T>, incin_core::nn::optional::True, T::ParameterDtype, Trainable>;

    fn new_on_target(shape: ShapeValue<S>, target: &T) -> Result<Self::Output> {
        incin_core::nn::linear::linear(shape).init(target)
    }
}

impl<S, Train, T> InitOnTarget<T> for EmbeddingBuilder<S, Train>
where
    S: EmbeddingShape,
    Train: TrainState,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = Embedding<S, TargetBackend<T>, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let dims = self.shape.dims();
        let vocab = dims[0];
        let embed = dims[1];
        let w_dims = [vocab, embed];
        let w_shape_val = shape_value_from_dims::<S::WeightShape>(&w_dims)?;

        let context_w = InitContext::new(ParameterRole::Weight);
        let weight = materialize_parameter::<T, S::WeightShape, Train>(
            target,
            w_shape_val,
            self.weight_init,
            context_w,
        )?;

        Ok(Embedding::from_raw_parts(weight))
    }
}

impl<S, Train, T> InitOnTarget<T> for LayerNormBuilder<S, Train>
where
    S: LayerNormShape,
    Train: TrainState,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = LayerNorm<S, TargetBackend<T>, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let dims = self.shape.dims();
        let channels = dims[0];
        let w_dims = [channels];
        let w_shape_val = shape_value_from_dims::<S::ParamShape>(&w_dims)?;

        let context_w = InitContext::new(ParameterRole::Scale);
        let weight = materialize_parameter::<T, S::ParamShape, Train>(
            target,
            w_shape_val.clone(),
            self.weight_init,
            context_w,
        )?;

        let context_b = InitContext::new(ParameterRole::Offset);
        let bias = materialize_parameter::<T, S::ParamShape, Train>(
            target,
            w_shape_val,
            self.bias_init,
            context_b,
        )?;

        Ok(LayerNorm::from_raw_parts(weight, bias, self.eps))
    }
}

impl<S, Train, T> InitOnTarget<T> for RMSNormBuilder<S, Train>
where
    S: RMSNormShape,
    Train: TrainState,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = RMSNorm<S, TargetBackend<T>, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let dims = self.shape.dims();
        let channels = dims[0];
        let w_dims = [channels];
        let w_shape_val = shape_value_from_dims::<S::ParamShape>(&w_dims)?;

        let context_w = InitContext::new(ParameterRole::Scale);
        let weight = materialize_parameter::<T, S::ParamShape, Train>(
            target,
            w_shape_val,
            self.weight_init,
            context_w,
        )?;

        Ok(RMSNorm::from_raw_parts(weight, self.eps))
    }
}

impl<S, Train, T> InitOnTarget<T> for BatchNorm2dBuilder<S, Train>
where
    S: BatchNormShape,
    Train: TrainState,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = BatchNorm2d<S, TargetBackend<T>, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let dims = self.shape.dims();
        let channels = dims[0];
        let w_dims = [channels];
        let w_shape_val = shape_value_from_dims::<S::ParamShape>(&w_dims)?;

        let weight = materialize_parameter::<T, S::ParamShape, Train>(
            target,
            w_shape_val.clone(),
            self.weight_init,
            InitContext::new(ParameterRole::Scale),
        )?;

        let bias = materialize_parameter::<T, S::ParamShape, Train>(
            target,
            w_shape_val.clone(),
            self.bias_init,
            InitContext::new(ParameterRole::Offset),
        )?;

        let running_mean = materialize_buffer::<T, S::ParamShape>(
            target,
            w_shape_val.clone(),
            self.running_mean_init,
            InitContext::new(ParameterRole::Other),
        )?;

        let running_var = materialize_buffer::<T, S::ParamShape>(
            target,
            w_shape_val,
            self.running_var_init,
            InitContext::new(ParameterRole::Other),
        )?;

        Ok(BatchNorm2d::from_raw_parts(
            weight,
            bias,
            running_mean,
            running_var,
            self.eps,
            self.momentum,
        ))
    }
}

impl<S, Bias, Train, T> InitOnTarget<T> for Conv1dBuilder<S, Bias, Train>
where
    S: Conv1dShape,
    Bias: OptionalField,
    Bias::Arg: Default,
    Train: TrainState,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = Conv1d<S, TargetBackend<T>, Bias, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let dims = self.shape.dims();
        let out_channels = dims[0];
        let in_channels = dims[1];
        let kernel_size = dims[2];
        let stride = dims[3];
        let padding = dims[4];
        let dilation = dims[5];

        let fan_in = in_channels * kernel_size;
        let fan_out = out_channels * kernel_size;

        let w_dims = [out_channels, in_channels, kernel_size];
        let w_shape_val = shape_value_from_dims::<S::WeightShape>(&w_dims)?;

        let context_w = InitContext::new(ParameterRole::Weight).with_fan(fan_in, fan_out);
        let weight = materialize_parameter::<T, S::WeightShape, Train>(
            target,
            w_shape_val,
            self.weight_init,
            context_w,
        )?;

        let bias = if Bias::init(Default::default()) {
            let b_dims = [out_channels];
            let b_shape_val = shape_value_from_dims::<S::BiasShape>(&b_dims)?;
            let context_b = InitContext::new(ParameterRole::Bias).with_fan(fan_in, fan_out);
            Some(materialize_parameter::<T, S::BiasShape, Train>(
                target,
                b_shape_val,
                self.bias_init,
                context_b,
            )?)
        } else {
            None
        };

        Ok(Conv1d::from_raw_parts(
            weight, bias, stride, padding, dilation, 1,
        ))
    }
}

impl<S, Bias, Train, T> InitOnTarget<T> for Conv2dBuilder<S, Bias, Train>
where
    S: Conv2dShape,
    Bias: OptionalField,
    Bias::Arg: Default,
    Train: TrainState,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = Conv2d<S, TargetBackend<T>, Bias, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let dims = self.shape.dims();
        let out_channels = dims[0];
        let in_channels = dims[1];
        let kernel_size = dims[2];

        let fan_in = in_channels * kernel_size * kernel_size;
        let fan_out = out_channels * kernel_size * kernel_size;

        let w_dims = [out_channels, in_channels, kernel_size, kernel_size];
        let w_shape_val = shape_value_from_dims::<S::WeightShape>(&w_dims)?;

        let context_w = InitContext::new(ParameterRole::Weight).with_fan(fan_in, fan_out);
        let weight = materialize_parameter::<T, S::WeightShape, Train>(
            target,
            w_shape_val,
            self.weight_init,
            context_w,
        )?;

        let bias = if Bias::init(Default::default()) {
            let b_dims = [out_channels];
            let b_shape_val = shape_value_from_dims::<S::BiasShape>(&b_dims)?;
            let context_b = InitContext::new(ParameterRole::Bias).with_fan(fan_in, fan_out);
            Some(materialize_parameter::<T, S::BiasShape, Train>(
                target,
                b_shape_val,
                self.bias_init,
                context_b,
            )?)
        } else {
            None
        };

        Ok(Conv2d::from_raw_parts(weight, bias))
    }
}

// ---------------------------------------------------------------------------
// InitOnTarget for RNNCellBuilder / RNNBuilder
// ---------------------------------------------------------------------------

impl<S, BiasIh, BiasHh, Train, T> InitOnTarget<T> for RNNCellBuilder<S, BiasIh, BiasHh, Train>
where
    S: RnnShape,
    BiasIh: OptionalField,
    BiasIh::Arg: Default,
    BiasHh: OptionalField,
    BiasHh::Arg: Default,
    Train: TrainState,
    S::IhShape: incin_core::nn::linear::LinearShape,
    S::HhShape: incin_core::nn::linear::LinearShape,
    LinearBuilder<S::IhShape, BiasIh, Train>: InitOnTarget<T>,
    LinearBuilder<S::HhShape, BiasHh, Train>: InitOnTarget<T>,
    <LinearBuilder<S::IhShape, BiasIh, Train> as InitOnTarget<T>>::Output:
        Into<Linear<S::IhShape, TargetBackend<T>, BiasIh, T::ParameterDtype, Train>>,
    <LinearBuilder<S::HhShape, BiasHh, Train> as InitOnTarget<T>>::Output:
        Into<Linear<S::HhShape, TargetBackend<T>, BiasHh, T::ParameterDtype, Train>>,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = RNNCell<S, TargetBackend<T>, BiasIh, BiasHh, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let dims = self.shape.dims();
        let in_features = dims[0];
        let out_features = dims[1];

        // wi: Linear<Shape2<S::In, S::Out>, ...>
        let wi_shape_val = shape_value_from_dims::<S::IhShape>(&[in_features, out_features])?;
        let wi_builder = LinearBuilder::<S::IhShape, BiasIh, Train> {
            shape: wi_shape_val,
            weight_init: self.input_weight_init,
            bias_init: self.input_bias_init,
            _phantom: core::marker::PhantomData,
        };
        let wi = wi_builder.init(target)?.into();

        // wh: Linear<Shape2<S::Out, S::Out>, ...>
        let wh_shape_val = shape_value_from_dims::<S::HhShape>(&[out_features, out_features])?;
        let wh_builder = LinearBuilder::<S::HhShape, BiasHh, Train> {
            shape: wh_shape_val,
            weight_init: self.hidden_weight_init,
            bias_init: self.hidden_bias_init,
            _phantom: core::marker::PhantomData,
        };
        let wh = wh_builder.init(target)?.into();

        Ok(RNNCell::new(wi, wh))
    }
}

impl<S, BiasIh, BiasHh, Train, T> InitOnTarget<T> for RNNBuilder<S, BiasIh, BiasHh, Train>
where
    S: RnnShape,
    BiasIh: OptionalField,
    BiasIh::Arg: Default,
    BiasHh: OptionalField,
    BiasHh::Arg: Default,
    Train: TrainState,
    Shape2<S::In, S::Out>: incin_core::nn::linear::LinearShape,
    Shape2<S::Out, S::Out>: incin_core::nn::linear::LinearShape,
    RNNCellBuilder<S, BiasIh, BiasHh, Train>: InitOnTarget<T>,
    <RNNCellBuilder<S, BiasIh, BiasHh, Train> as InitOnTarget<T>>::Output:
        Into<RNNCell<S, TargetBackend<T>, BiasIh, BiasHh, T::ParameterDtype, Train>>,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = RNN<S, TargetBackend<T>, BiasIh, BiasHh, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let cell = self.cell.init(target)?.into();
        Ok(RNN::new(cell))
    }
}

// ---------------------------------------------------------------------------
// InitOnTarget for LSTMCellBuilder / LSTMBuilder
// ---------------------------------------------------------------------------

impl<S, BiasIh, BiasHh, Train, T> InitOnTarget<T> for LSTMCellBuilder<S, BiasIh, BiasHh, Train>
where
    S: LstmShape,
    BiasIh: OptionalField,
    BiasIh::Arg: Default,
    BiasHh: OptionalField,
    BiasHh::Arg: Default,
    Train: TrainState,
    S::IhShape: incin_core::nn::linear::LinearShape,
    S::HhShape: incin_core::nn::linear::LinearShape,
    LinearBuilder<S::IhShape, BiasIh, Train>: InitOnTarget<T>,
    LinearBuilder<S::HhShape, BiasHh, Train>: InitOnTarget<T>,
    <LinearBuilder<S::IhShape, BiasIh, Train> as InitOnTarget<T>>::Output:
        Into<Linear<S::IhShape, TargetBackend<T>, BiasIh, T::ParameterDtype, Train>>,
    <LinearBuilder<S::HhShape, BiasHh, Train> as InitOnTarget<T>>::Output:
        Into<Linear<S::HhShape, TargetBackend<T>, BiasHh, T::ParameterDtype, Train>>,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = LSTMCell<S, TargetBackend<T>, BiasIh, BiasHh, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let dims = self.shape.dims();
        let in_features = dims[0];
        let out_features = dims[1];

        let iw = self.input_weight_init;
        let hw = self.hidden_weight_init;
        let ib = self.input_bias_init;
        let hb = self.hidden_bias_init;

        let make_wi = |weight_init: incin_core::nn::init::Init,
                       bias_init: incin_core::nn::init::Init|
         -> Result<
            Linear<S::IhShape, TargetBackend<T>, BiasIh, T::ParameterDtype, Train>,
        > {
            let shape_val = shape_value_from_dims::<S::IhShape>(&[in_features, out_features])?;
            let builder = LinearBuilder::<S::IhShape, BiasIh, Train> {
                shape: shape_val,
                weight_init,
                bias_init,
                _phantom: core::marker::PhantomData,
            };
            Ok(builder.init(target)?.into())
        };

        let make_wh = |weight_init: incin_core::nn::init::Init,
                       bias_init: incin_core::nn::init::Init|
         -> Result<
            Linear<S::HhShape, TargetBackend<T>, BiasHh, T::ParameterDtype, Train>,
        > {
            let shape_val = shape_value_from_dims::<S::HhShape>(&[out_features, out_features])?;
            let builder = LinearBuilder::<S::HhShape, BiasHh, Train> {
                shape: shape_val,
                weight_init,
                bias_init,
                _phantom: core::marker::PhantomData,
            };
            Ok(builder.init(target)?.into())
        };

        Ok(LSTMCell {
            wi_i: make_wi(iw, ib)?,
            wi_f: make_wi(iw, ib)?,
            wi_g: make_wi(iw, ib)?,
            wi_o: make_wi(iw, ib)?,
            wh_i: make_wh(hw, hb)?,
            wh_f: make_wh(hw, hb)?,
            wh_g: make_wh(hw, hb)?,
            wh_o: make_wh(hw, hb)?,
        })
    }
}

impl<S, BiasIh, BiasHh, Train, T> InitOnTarget<T> for LSTMBuilder<S, BiasIh, BiasHh, Train>
where
    S: LstmShape,
    BiasIh: OptionalField,
    BiasIh::Arg: Default,
    BiasHh: OptionalField,
    BiasHh::Arg: Default,
    Train: TrainState,
    S::IhShape: incin_core::nn::linear::LinearShape,
    S::HhShape: incin_core::nn::linear::LinearShape,
    LSTMCellBuilder<S, BiasIh, BiasHh, Train>: InitOnTarget<T>,
    <LSTMCellBuilder<S, BiasIh, BiasHh, Train> as InitOnTarget<T>>::Output:
        Into<LSTMCell<S, TargetBackend<T>, BiasIh, BiasHh, T::ParameterDtype, Train>>,
    T: TensorTarget + TargetExt + Clone,
    T::ParameterDtype: FloatDType,
    TargetBackend<T>: Backend<Device = T::Device>
        + SupportsDType<T::ParameterDtype>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>
        + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Zeros,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::Ones,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::UniformRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::backend_authoring::Execute<
            incin_core::backend_authoring::op::NormalRandom,
            Output = <TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>,
        > + incin_core::exec::Capabilities
        + Default,
    <TargetBackend<T> as Execute<op::MulScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
    <TargetBackend<T> as Execute<op::AddScalar>>::Output:
        Into<<TargetBackend<T> as StorageBackend>::Storage<T::ParameterDtype>>,
{
    type Output = LSTM<S, TargetBackend<T>, BiasIh, BiasHh, T::ParameterDtype, Train>;

    fn init(self, target: &T) -> Result<Self::Output> {
        let cell = self.cell.init(target)?.into();
        Ok(LSTM::new(cell))
    }
}
