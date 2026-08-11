use crate::exec::catalog::{Conv2dAttributes, Descriptor, op};
use crate::exec::context::ExecutionContext;
use crate::exec::request::TensorHandle;
use crate::nn::init::{InitContext, ParameterRole};
use crate::nn::param::{Frozen, TrainState, Trainable, execute_plan_raw};
use crate::nn::{Module, Param};
use crate::prelude::*;
use crate::tensor::backend::Execute;
use typenum::Unsigned;

/// A shape marker trait specifying a [`Conv2d`] layer's channel counts and
/// compile-time-fixed kernel/stride/padding/dilation. The typical usage is
/// `(OutC, InC, K, S, P, D)` for a fully static, square-kernel layer.
pub trait Conv2dShape: Shape + DynShape {
    /// Number of output channels.
    type OutC: Dim;
    /// Number of input channels.
    type InC: Dim;
    /// Kernel (window) size — square, applied to both spatial dimensions.
    type K: Dim<Arg = ()>;
    /// Stride.
    type S: Dim<Arg = ()>;
    /// Padding.
    type P: Dim<Arg = ()>;
    /// Dilation.
    type D: Dim<Arg = ()>;
    /// The shape argument type used to construct the weight tensor.
    type WeightArg: crate::tensor::arg_into::NotUnit;
    /// The shape argument type used to construct the bias tensor.
    type BiasArg: crate::tensor::arg_into::NotUnit;
    /// The static shape type of the weight parameter tensor.
    type WeightShape: Shape<Arg = Self::WeightArg> + DynShape;
    /// The static shape type of the bias parameter tensor.
    type BiasShape: Shape<Arg = Self::BiasArg> + DynShape;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::OutC as Dim>::Arg, <Self::InC as Dim>::Arg),
    ) -> core::result::Result<(usize, usize, Self::WeightArg, Self::BiasArg), ShapeError>;
}

/* legacy tuple Conv2dShape implementation removed: use DimCons/Nil */
/*
impl<OutC: Dim, InC: Dim, K: Dim<Arg = ()>, S: Dim<Arg = ()>, P: Dim<Arg = ()>, D: Dim<Arg = ()>>
    Conv2dShape for (OutC, InC, K, S, P, D)
{
    /// Number of output channels.
    type OutC = OutC;
    /// Number of input channels.
    type InC = InC;
    /// Kernel (window) size.
    type K = K;
    /// Stride.
    type S = S;
    /// Padding.
    type P = P;
    /// Dilation.
    type D = D;
    /// The shape argument type used to construct the weight tensor.
    type WeightArg = (
        <OutC as Dim>::Arg,
        <InC as Dim>::Arg,
        <K as Dim>::Arg,
        <K as Dim>::Arg,
    );
    /// The shape argument type used to construct the bias tensor.
    type BiasArg = (<OutC as Dim>::Arg,);
    /// The static shape type of the weight parameter tensor.
    type WeightShape = (OutC, InC, K, K);
    /// The static shape type of the bias parameter tensor.
    type BiasShape = (OutC,);

    #[inline]
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::OutC as Dim>::Arg, <Self::InC as Dim>::Arg),
    ) -> core::result::Result<(usize, usize, Self::WeightArg, Self::BiasArg), ShapeError> {
        let out_channels = OutC::resolve_arg(target.0.clone())?;
        let in_channels = InC::resolve_arg(target.1.clone())?;
        Ok((
            out_channels,
            in_channels,
            (
                target.0.clone(),
                target.1,
                (),
                (),
            ),
            (target.0,),
        ))
    }
}
*/

impl<OutC: Dim, InC: Dim, K: Dim<Arg = ()>, S: Dim<Arg = ()>, P: Dim<Arg = ()>, D: Dim<Arg = ()>>
    Conv2dShape
    for crate::shapes::shape::DimCons<
        OutC,
        crate::shapes::shape::DimCons<
            InC,
            crate::shapes::shape::DimCons<
                K,
                crate::shapes::shape::DimCons<
                    S,
                    crate::shapes::shape::DimCons<
                        P,
                        crate::shapes::shape::DimCons<D, crate::shapes::shape::Nil>,
                    >,
                >,
            >,
        >,
    >
{
    type OutC = OutC;
    type InC = InC;
    type K = K;
    type S = S;
    type P = P;
    type D = D;
    type WeightArg = (
        <OutC as Dim>::Arg,
        (<InC as Dim>::Arg, (<K as Dim>::Arg, (<K as Dim>::Arg, ()))),
    );
    type BiasArg = (<OutC as Dim>::Arg, ());
    type WeightShape = crate::shapes::shape::DimCons<
        OutC,
        crate::shapes::shape::DimCons<
            InC,
            crate::shapes::shape::DimCons<
                K,
                crate::shapes::shape::DimCons<K, crate::shapes::shape::Nil>,
            >,
        >,
    >;
    type BiasShape = crate::shapes::shape::DimCons<OutC, crate::shapes::shape::Nil>;

    #[inline]
    fn build_args(
        target: (<Self::OutC as Dim>::Arg, <Self::InC as Dim>::Arg),
    ) -> core::result::Result<(usize, usize, Self::WeightArg, Self::BiasArg), ShapeError> {
        let out_channels = OutC::resolve_arg(target.0.clone())?;
        let in_channels = InC::resolve_arg(target.1.clone())?;
        Ok((
            out_channels,
            in_channels,
            (target.0.clone(), (target.1, ((), ((), ())))),
            (target.0, ()),
        ))
    }
}

/// A 2D convolutional layer operating on 4D tensors of shape `[Batch, Channels, H, W]`.
#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
pub struct Conv2d<
    S: Conv2dShape,
    B: Backend,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The learnable weight matrix parameter.
    pub weight: Param<S::WeightShape, B, K, Train>,
    /// The optional learnable bias vector parameter.
    pub bias: Option<Param<S::BiasShape, B, K, Train>>,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias, K, Train)>,
}

impl<
    S: Conv2dShape,
    B: Backend,
    Bias: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> Conv2d<S, B, Bias, K, Train>
{
    /// Constructs a Conv2d from raw parts.
    pub fn from_raw_parts(
        weight: Param<S::WeightShape, B, K, Train>,
        bias: Option<Param<S::BiasShape, B, K, Train>>,
    ) -> Self {
        Self {
            weight,
            bias,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Freezes this layer's parameters.
    pub fn freeze(self) -> Conv2d<S, B, Bias, K, Frozen> {
        Conv2d {
            weight: self.weight.freeze(),
            bias: self.bias.map(|b| b.freeze()),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Unfreezes this layer's parameters.
    pub fn unfreeze(self) -> Conv2d<S, B, Bias, K, Trainable> {
        Conv2d {
            weight: self.weight.unfreeze(),
            bias: self.bias.map(|b| b.unfreeze()),
            _phantom: core::marker::PhantomData,
        }
    }
}

/// A builder for constructing a [`Conv2d`] layer with a target.
#[derive(Debug, Clone)]
pub struct Conv2dBuilder<
    S: Conv2dShape,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
    Train: TrainState = Trainable,
> {
    pub shape: ShapeValue<S>,
    pub weight_init: crate::nn::init::Init,
    pub bias_init: crate::nn::init::Init,
    pub _bias: core::marker::PhantomData<Bias>,
    pub _train: core::marker::PhantomData<Train>,
}

/// Creates a new builder for a [`Conv2d`] layer with shape `shape`.
pub fn conv2d<S: Conv2dShape>(shape: ShapeValue<S>) -> Conv2dBuilder<S> {
    Conv2dBuilder {
        shape,
        weight_init: crate::nn::init::kaiming_uniform(),
        bias_init: crate::nn::init::kaiming_uniform(),
        _bias: core::marker::PhantomData,
        _train: core::marker::PhantomData,
    }
}

impl<S: Conv2dShape, Bias: crate::nn::optional::OptionalField, Train: TrainState>
    Conv2dBuilder<S, Bias, Train>
{
    /// Disables bias parameter for this convolution layer.
    pub fn no_bias(self) -> Conv2dBuilder<S, crate::nn::optional::False, Train> {
        Conv2dBuilder {
            shape: self.shape,
            weight_init: self.weight_init,
            bias_init: self.bias_init,
            _bias: core::marker::PhantomData,
            _train: self._train,
        }
    }

    /// Marks the resulting layer as frozen (non-trainable).
    pub fn frozen(self) -> Conv2dBuilder<S, Bias, Frozen> {
        Conv2dBuilder {
            shape: self.shape,
            weight_init: self.weight_init,
            bias_init: self.bias_init,
            _bias: self._bias,
            _train: core::marker::PhantomData,
        }
    }

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
}

impl<S, B, Bias, K: DType> Conv2d<S, B, Bias, K, Trainable>
where
    S: Conv2dShape,
    B: Backend
        + SupportsDType<K>
        + crate::tensor::backend::CreationOps<B>
        + crate::tensor::backend::FloatOps<B>
        + crate::tensor::backend::NumericOps<B>,
    Bias: crate::nn::optional::OptionalField,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::OutC as Dim>::Arg,
                <S::InC as Dim>::Arg,
                <K as DType>::Arg,
                <B::Device as Device>::Arg,
                <Bias as crate::nn::optional::OptionalField>::Arg,
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (out_c, in_c, dtype_arg, device_arg, bias_arg) = args.into_layer_arg();
        let (cout, cin, weight_shape_arg, bias_shape_arg) =
            S::build_args((out_c, in_c)).map_err(Error::Shape)?;
        let kernel_size = S::K::static_size().map_err(Error::Shape)?;
        let fan_in = cin * kernel_size * kernel_size;
        let fan_out = cout * kernel_size * kernel_size;

        let dtype_field = <K as DType>::init(dtype_arg);
        let device_field = <B::Device as Device>::init(device_arg);
        let weight_shape_field =
            <S::WeightShape as Shape>::resolve(weight_shape_arg).map_err(Error::Shape)?;
        let bias_shape_field =
            <S::BiasShape as Shape>::resolve(bias_shape_arg).map_err(Error::Shape)?;

        let init = crate::nn::init::kaiming_uniform();
        let context_w = InitContext::new(ParameterRole::Weight).with_fan(fan_in, fan_out);
        let plan_w = init.plan(context_w)?;
        let weight_dims = weight_shape_field.clone();
        let raw_w =
            execute_plan_raw::<B, K>(weight_dims.as_ref(), &dtype_field, &device_field, plan_w)?;
        let weight = Param::<S::WeightShape, B, K, Trainable>::from_parts_checked(
            raw_w,
            weight_shape_field,
            dtype_field.clone(),
            device_field.clone(),
        )?;

        let bias = if Bias::init(bias_arg) {
            let context_b = InitContext::new(ParameterRole::Bias).with_fan(fan_in, fan_out);
            let plan_b = init.plan(context_b)?;
            let bias_dims = bias_shape_field.clone();
            let raw_b =
                execute_plan_raw::<B, K>(bias_dims.as_ref(), &dtype_field, &device_field, plan_b)?;
            Some(Param::<S::BiasShape, B, K, Trainable>::from_parts_checked(
                raw_b,
                bias_shape_field,
                dtype_field,
                device_field,
            )?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            _phantom: core::marker::PhantomData,
        })
    }
}

impl<I, S, B, COut: Dim, CIn: Dim, K: DType, Train: TrainState> Module<Tensor<I, B, K>>
    for Conv2d<S, B, crate::nn::optional::True, K, Train>
where
    S: Conv2dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels2D<CIn>,
    B: Backend
        + crate::tensor::backend::ModuleOps<B>
        + crate::tensor::backend::TensorOps<B>
        + Execute<Descriptor<op::Conv2dExact>>
        + Execute<Descriptor<op::ReshapeExact>>,
    <B as Execute<Descriptor<op::Conv2dExact>>>::Output: Into<B::Storage<K>>,
    <B as Execute<Descriptor<op::ReshapeExact>>>::Output: Into<B::Storage<K>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B, K>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B, K>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = self.bias.as_ref().unwrap().as_tensor()?;

        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size = crate::shapes::ShapeBuf::from_slice(&x_shape[0..rank - 3])
            .checked_numel(crate::shapes::error::OperationKind::Conv2d)?;
        let in_channels = x_shape[rank - 3];
        let height = x_shape[rank - 2];
        let width = x_shape[rank - 1];

        let x_inner = if rank > 4 {
            crate::tensor::ops::manipulation::reshape_storage_exact::<B, K>(
                &x.inner,
                &ShapeBuf::from_slice(&[batch_size, in_channels, height, width]),
            )?
        } else {
            x.inner.clone()
        };

        let shape =
            <I as crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                &x.shape_buf_value(),
                weight.dims()[0],
            )?;

        let intermediate_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[
            batch_size,
            shape[shape.len() - 3],
            shape[shape.len() - 2],
            shape[shape.len() - 1],
        ]))
        .map_err(Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&x_inner),
            TensorHandle::from_storage::<B, K, Local>(&weight.inner),
            TensorHandle::from_storage::<B, K, Local>(&bias.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let out = crate::exec::dispatch::execute_shaped::<op::Conv2dExact, B, Dyn>(
            &context,
            Conv2dAttributes {
                stride: [S::S::static_size().map_err(Error::Shape)?; 2],
                padding: [S::P::static_size().map_err(Error::Shape)?; 2],
                dilation: [S::D::static_size().map_err(Error::Shape)?; 2],
                groups: 1,
                has_bias: true,
            },
            &inputs,
            &intermediate_shape,
        )?
        .into();

        let out_shape = shape.clone();
        let out = if rank > 4 {
            crate::tensor::ops::manipulation::reshape_storage_exact::<B, K>(&out, &out_shape)?
        } else {
            out
        };

        Tensor::from_parts(
            out,
            shape,
            x._dtype.clone(),
            weight._device.clone(),
            *x.grad_field(),
        )
    }
}

impl<I, S, B, COut: Dim, CIn: Dim, K: DType, Train: TrainState> Module<Tensor<I, B, K>>
    for Conv2d<S, B, crate::nn::optional::False, K, Train>
where
    S: Conv2dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels2D<CIn>,
    B: Backend
        + crate::tensor::backend::ModuleOps<B>
        + crate::tensor::backend::TensorOps<B>
        + Execute<Descriptor<op::Conv2dExact>>
        + Execute<Descriptor<op::ReshapeExact>>,
    <B as Execute<Descriptor<op::Conv2dExact>>>::Output: Into<B::Storage<K>>,
    <B as Execute<Descriptor<op::ReshapeExact>>>::Output: Into<B::Storage<K>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B, K>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B, K>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;

        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size = crate::shapes::ShapeBuf::from_slice(&x_shape[0..rank - 3])
            .checked_numel(crate::shapes::error::OperationKind::Conv2d)?;
        let in_channels = x_shape[rank - 3];
        let height = x_shape[rank - 2];
        let width = x_shape[rank - 1];

        let x_inner = if rank > 4 {
            crate::tensor::ops::manipulation::reshape_storage_exact::<B, K>(
                &x.inner,
                &ShapeBuf::from_slice(&[batch_size, in_channels, height, width]),
            )?
        } else {
            x.inner.clone()
        };

        let shape =
            <I as crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                &x.shape_buf_value(),
                weight.dims()[0],
            )?;

        let intermediate_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[
            batch_size,
            shape[shape.len() - 3],
            shape[shape.len() - 2],
            shape[shape.len() - 1],
        ]))
        .map_err(Error::Shape)?;
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&x_inner),
            TensorHandle::from_storage::<B, K, Local>(&weight.inner),
        ];
        let context = ExecutionContext::from_scope(B::default());
        let out = crate::exec::dispatch::execute_shaped::<op::Conv2dExact, B, Dyn>(
            &context,
            Conv2dAttributes {
                stride: [S::S::static_size().map_err(Error::Shape)?; 2],
                padding: [S::P::static_size().map_err(Error::Shape)?; 2],
                dilation: [S::D::static_size().map_err(Error::Shape)?; 2],
                groups: 1,
                has_bias: false,
            },
            &inputs,
            &intermediate_shape,
        )?
        .into();

        let out_shape = shape.clone();
        let out = if rank > 4 {
            crate::tensor::ops::manipulation::reshape_storage_exact::<B, K>(&out, &out_shape)?
        } else {
            out
        };

        Tensor::from_parts(
            out,
            shape,
            x._dtype.clone(),
            weight._device.clone(),
            *x.grad_field(),
        )
    }
}

impl<I, S, B, COut: Dim, CIn: Dim, K: DType, Train: TrainState> Module<Tensor<I, B, K>>
    for Conv2d<S, B, Dyn, K, Train>
where
    S: Conv2dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels2D<CIn>,
    B: Backend
        + crate::tensor::backend::ModuleOps<B>
        + crate::tensor::backend::TensorOps<B>
        + Execute<Descriptor<op::Conv2dExact>>
        + Execute<Descriptor<op::ReshapeExact>>,
    <B as Execute<Descriptor<op::Conv2dExact>>>::Output: Into<B::Storage<K>>,
    <B as Execute<Descriptor<op::ReshapeExact>>>::Output: Into<B::Storage<K>>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B, K>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B, K>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = match &self.bias {
            Some(b) => Some(b.as_tensor()?),
            None => None,
        };

        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size = crate::shapes::ShapeBuf::from_slice(&x_shape[0..rank - 3])
            .checked_numel(crate::shapes::error::OperationKind::Conv2d)?;
        let in_channels = x_shape[rank - 3];
        let height = x_shape[rank - 2];
        let width = x_shape[rank - 1];

        let x_inner = if rank > 4 {
            crate::tensor::ops::manipulation::reshape_storage_exact::<B, K>(
                &x.inner,
                &ShapeBuf::from_slice(&[batch_size, in_channels, height, width]),
            )?
        } else {
            x.inner.clone()
        };

        let shape =
            <I as crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                &x.shape_buf_value(),
                weight.dims()[0],
            )?;

        let intermediate_shape = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[
            batch_size,
            shape[shape.len() - 3],
            shape[shape.len() - 2],
            shape[shape.len() - 1],
        ]))
        .map_err(Error::Shape)?;
        let mut inputs = alloc::vec::Vec::with_capacity(3);
        inputs.push(TensorHandle::from_storage::<B, K, Local>(&x_inner));
        inputs.push(TensorHandle::from_storage::<B, K, Local>(&weight.inner));
        if let Some(bias) = bias.as_ref() {
            inputs.push(TensorHandle::from_storage::<B, K, Local>(&bias.inner));
        }
        let context = ExecutionContext::from_scope(B::default());
        let out = crate::exec::dispatch::execute_shaped::<op::Conv2dExact, B, Dyn>(
            &context,
            Conv2dAttributes {
                stride: [S::S::static_size().map_err(Error::Shape)?; 2],
                padding: [S::P::static_size().map_err(Error::Shape)?; 2],
                dilation: [S::D::static_size().map_err(Error::Shape)?; 2],
                groups: 1,
                has_bias: bias.is_some(),
            },
            &inputs,
            &intermediate_shape,
        )?
        .into();

        let out_shape = shape.clone();
        let out = if rank > 4 {
            crate::tensor::ops::manipulation::reshape_storage_exact::<B, K>(&out, &out_shape)?
        } else {
            out
        };

        Tensor::from_parts(
            out,
            shape,
            x._dtype.clone(),
            weight._device.clone(),
            *x.grad_field(),
        )
    }
}
