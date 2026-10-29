use crate::backend_authoring::SupportsDType;
use crate::dist::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::{ConvTranspose2dAttributes, op};
use crate::exec::context::ExecutionContext;
use crate::exec::request::TensorHandle;
use crate::nn::init::{InitContext, ParameterRole};
use crate::nn::param::{Frozen, TrainState, Trainable, execute_plan_raw};
use crate::nn::{Module, Param};
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation};
use crate::shapes::{Dim, DynShape, Layout, Shape, ShapeBuf, ShapeError, ShapeValue};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;

/// A shape marker trait specifying a [`ConvTranspose2d`] layer's channel
/// counts and compile-time-fixed square kernel/stride/padding/
/// output-padding/dilation. The typical usage is
/// `(InC, OutC, K, S, P, OP, D)` for a fully static layer.
pub trait ConvTranspose2dShape: Shape + DynShape {
    /// Number of input channels.
    type InC: Dim;
    /// Number of output channels.
    type OutC: Dim;
    /// Kernel (window) size - square, applied to both spatial dimensions.
    type K: Dim<Arg = ()>;
    /// Stride - square, applied to both spatial dimensions.
    type S: Dim<Arg = ()>;
    /// Padding - square, applied to both spatial dimensions.
    type P: Dim<Arg = ()>;
    /// Output padding - square, applied to both spatial dimensions.
    type OP: Dim<Arg = ()>;
    /// Dilation - square, applied to both spatial dimensions.
    type D: Dim<Arg = ()>;
    /// The shape argument type used to construct the weight tensor.
    type WeightArg: crate::tensor::arg_into::NotUnit;
    /// The shape argument type used to construct the bias tensor.
    type BiasArg: crate::tensor::arg_into::NotUnit;
    /// The static shape type of the weight parameter tensor (`[Cin, Cout, K, K]`).
    type WeightShape: Shape<Arg = Self::WeightArg> + DynShape;
    /// The static shape type of the bias parameter tensor (`[Cout]`).
    type BiasShape: Shape<Arg = Self::BiasArg> + DynShape;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::InC as Dim>::Arg, <Self::OutC as Dim>::Arg),
    ) -> core::result::Result<(Self::WeightArg, Self::BiasArg), ShapeError>;
}

impl<
    InC: Dim,
    OutC: Dim,
    K: Dim<Arg = ()>,
    S: Dim<Arg = ()>,
    P: Dim<Arg = ()>,
    OP: Dim<Arg = ()>,
    D: Dim<Arg = ()>,
> ConvTranspose2dShape
    for crate::shapes::shape::DimCons<
        InC,
        crate::shapes::shape::DimCons<
            OutC,
            crate::shapes::shape::DimCons<
                K,
                crate::shapes::shape::DimCons<
                    S,
                    crate::shapes::shape::DimCons<
                        P,
                        crate::shapes::shape::DimCons<
                            OP,
                            crate::shapes::shape::DimCons<D, crate::shapes::shape::Nil>,
                        >,
                    >,
                >,
            >,
        >,
    >
{
    type InC = InC;
    type OutC = OutC;
    type K = K;
    type S = S;
    type P = P;
    type OP = OP;
    type D = D;
    type WeightArg = (
        <InC as Dim>::Arg,
        (<OutC as Dim>::Arg, (<K as Dim>::Arg, (<K as Dim>::Arg, ()))),
    );
    type BiasArg = (<OutC as Dim>::Arg, ());
    type WeightShape = crate::shapes::shape::DimCons<
        InC,
        crate::shapes::shape::DimCons<
            OutC,
            crate::shapes::shape::DimCons<
                K,
                crate::shapes::shape::DimCons<K, crate::shapes::shape::Nil>,
            >,
        >,
    >;
    type BiasShape = crate::shapes::shape::DimCons<OutC, crate::shapes::shape::Nil>;

    #[inline]
    fn build_args(
        target: (<Self::InC as Dim>::Arg, <Self::OutC as Dim>::Arg),
    ) -> core::result::Result<(Self::WeightArg, Self::BiasArg), ShapeError> {
        Ok((
            (target.0.clone(), (target.1.clone(), ((), ((), ())))),
            (target.1, ()),
        ))
    }
}

/// A 2D transposed convolutional layer operating on 3D unbatched
/// `[Cin, H, W]` or 4D batched `[N, Cin, H, W]` tensors.
///
/// Transposed convolution upsamples: with stride `s` each input position
/// spreads over an `s x s` output window (see
/// [`natural_transpose_out_size`](crate::shapes::error::OperationKind) for
/// the shape rule, mirrored in [`transpose_out_size`] below).
///
/// Geometry is square-static (matching [`Conv2d`](crate::nn::Conv2d)); the
/// weight follows the transposed convention `[Cin, Cout, K, K]`. Only
/// `groups == 1` is supported — grouped transposed convolution is
/// unimplemented on every backend, so any other group count is refused with
/// a typed error at build and forward time rather than silently narrowed.
#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
pub struct ConvTranspose2d<
    S: ConvTranspose2dShape,
    B: crate::tensor::backend::VariableBackend,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    /// The learnable weight parameter in `[Cin, Cout, K, K]` layout.
    pub weight: Param<S::WeightShape, B, K, Train>,
    /// The optional learnable bias vector parameter (`[Cout]`).
    pub bias: Option<Param<S::BiasShape, B, K, Train>>,
    #[module(ignore)]
    /// Convolution groups. Only `1` is supported.
    pub groups: usize,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias, K, Train)>,
}

impl<
    S: ConvTranspose2dShape,
    B: crate::tensor::backend::VariableBackend,
    Bias: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
> ConvTranspose2d<S, B, Bias, K, Train>
{
    /// Rejects group counts no backend implements.
    fn check_groups(groups: usize) -> Result<()> {
        if groups != 1 {
            return Err(Error::Shape(ShapeError::InvalidParameter {
                operation: OperationKind::ConvTranspose2d,
                parameter: "groups",
                value: groups,
            }));
        }
        Ok(())
    }

    /// Constructs a ConvTranspose2d from raw parts.
    pub fn from_raw_parts(
        weight: Param<S::WeightShape, B, K, Train>,
        bias: Option<Param<S::BiasShape, B, K, Train>>,
        groups: usize,
    ) -> Result<Self> {
        Self::check_groups(groups)?;
        Ok(Self {
            weight,
            bias,
            groups,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Freezes this layer's parameters.
    pub fn freeze(self) -> ConvTranspose2d<S, B, Bias, K, Frozen> {
        ConvTranspose2d {
            weight: self.weight.freeze(),
            bias: self.bias.map(|b| b.freeze()),
            groups: self.groups,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Unfreezes this layer's parameters.
    pub fn unfreeze(self) -> ConvTranspose2d<S, B, Bias, K, Trainable> {
        ConvTranspose2d {
            weight: self.weight.unfreeze(),
            bias: self.bias.map(|b| b.unfreeze()),
            groups: self.groups,
            _phantom: core::marker::PhantomData,
        }
    }
}

/// One spatial axis of the transposed-convolution output size:
/// `(len - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1`,
/// plus `output_padding` applied once afterwards.
///
/// Mirrors the kernel's `natural_transpose_out_size` (saturating, never
/// panicking) with typed errors for degenerate geometry and overflow,
/// attributed to [`OperationKind::ConvTranspose2d`].
pub fn transpose_out_size(
    len: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_padding: usize,
) -> Result<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return Err(Error::Shape(ShapeError::InvalidParameter {
            operation: OperationKind::ConvTranspose2d,
            parameter: "kernel, stride, and dilation must be nonzero",
            value: 0,
        }));
    }
    let unpadded = len
        .saturating_sub(1)
        .checked_mul(stride)
        .and_then(|span| {
            dilation
                .checked_mul(kernel - 1)
                .and_then(|kernel| span.checked_add(kernel))
        })
        .and_then(|span| span.checked_add(1))
        .ok_or(Error::Shape(ShapeError::ArithmeticOverflow {
            operation: OperationKind::ConvTranspose2d,
            expression: "transposed-convolution output dimension",
        }))?;
    let twice_padding =
        padding
            .checked_mul(2)
            .ok_or(Error::Shape(ShapeError::ArithmeticOverflow {
                operation: OperationKind::ConvTranspose2d,
                expression: "transposed-convolution padding",
            }))?;
    unpadded
        .saturating_sub(twice_padding)
        .checked_add(output_padding)
        .ok_or(Error::Shape(ShapeError::ArithmeticOverflow {
            operation: OperationKind::ConvTranspose2d,
            expression: "transposed-convolution output padding",
        }))
}

impl<S, B, Bias, K: DType> ConvTranspose2d<S, B, Bias, K, Trainable>
where
    S: ConvTranspose2dShape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::tensor::backend::SupportsDType<K>
        + crate::nn::param::ParameterInit<K>,
    Bias: crate::nn::optional::OptionalField,
    <K as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    /// Builds the layer from every argument stated explicitly.
    pub fn build_full(
        in_channels: <S::InC as Dim>::Arg,
        out_channels: <S::OutC as Dim>::Arg,
        dtype: <K as DType>::Arg,
        device: <B::Device as Device>::Arg,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        Self::check_groups(groups)?;
        let (weight_shape_arg, bias_shape_arg) =
            S::build_args((in_channels.clone(), out_channels.clone())).map_err(Error::Shape)?;
        let kernel_size = S::K::static_size().map_err(Error::Shape)?;
        let in_c = S::InC::resolve_arg(in_channels).map_err(Error::Shape)?;
        let out_c = S::OutC::resolve_arg(out_channels).map_err(Error::Shape)?;
        let fan_in = in_c * kernel_size * kernel_size;
        let fan_out = out_c * kernel_size * kernel_size;

        let dtype_field = <K as DType>::init(dtype);
        let device_field = <B::Device as Device>::init(device);
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

        let bias = if bias {
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
            groups,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Builds the layer from channel arguments.
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::InC as Dim>::Arg,
                <S::OutC as Dim>::Arg,
                <K as DType>::Arg,
                <B::Device as Device>::Arg,
                usize,
                <Bias as crate::nn::optional::OptionalField>::Arg,
            )>,
    {
        let (in_c, out_c, dtype_arg, device_arg, groups, bias_arg) = args.into_layer_arg();
        Self::build_full(
            in_c,
            out_c,
            dtype_arg,
            device_arg,
            groups,
            <Bias as crate::nn::optional::OptionalField>::init(bias_arg),
        )
    }
}

impl<
    I: Shape + DynShape,
    S: ConvTranspose2dShape,
    B: crate::tensor::backend::VariableBackend
        + crate::exec::Capabilities
        + Execute<op::ConvTranspose2d>,
    Bias: crate::nn::optional::OptionalField,
    K: DType,
    Train: TrainState,
    L: Layout<I>,
> Module<Tensor<I, B, K, crate::tensor::grad::NoGrad, Local, L>>
    for ConvTranspose2d<S, B, Bias, K, Train>
where
    <B as Execute<op::ConvTranspose2d>>::Output: Into<B::Storage<K>>,
{
    /// `Dense<Dyn>`: the output spatial extents are runtime values of the
    /// geometry, so the shape is computed numerically rather than typed.
    type Output =
        crate::shapes::Dense<crate::shapes::Dyn, B, K, crate::tensor::grad::NoGrad, Local>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(
        &self,
        x: Tensor<I, B, K, crate::tensor::grad::NoGrad, Local, L>,
    ) -> core::result::Result<Self::Output, Error> {
        Self::check_groups(self.groups)?;
        let weight = self.weight.as_tensor()?;
        let bias = self.bias.as_ref().map(|b| b.as_tensor()).transpose()?;

        let dims = x.dims();
        let rank = dims.as_ref().len();
        // The catalog admits batched (rank 4) and unbatched (rank 3) activations.
        if rank != 3 && rank != 4 {
            return Err(Error::Shape(ShapeError::RankMismatch {
                operation: OperationKind::ConvTranspose2d,
                expected: RankExpectation::Between { min: 3, max: 4 },
                actual: rank,
            }));
        }
        let in_channels = dims.as_ref()[rank - 3];
        if in_channels != weight.dims().as_ref()[0] {
            return Err(Error::Shape(ShapeError::DimensionMismatch {
                operation: OperationKind::ConvTranspose2d,
                axis: Axis::Named("channels"),
                lhs: in_channels,
                rhs: weight.dims().as_ref()[0],
                constraint: DimensionConstraint::Equal,
            }));
        }

        let kernel = S::K::static_size().map_err(Error::Shape)?;
        let stride = S::S::static_size().map_err(Error::Shape)?;
        let padding = S::P::static_size().map_err(Error::Shape)?;
        let output_padding = S::OP::static_size().map_err(Error::Shape)?;
        let dilation = S::D::static_size().map_err(Error::Shape)?;
        let height = dims.as_ref()[rank - 2];
        let width = dims.as_ref()[rank - 1];
        let out_height =
            transpose_out_size(height, kernel, stride, padding, dilation, output_padding)?;
        let out_width =
            transpose_out_size(width, kernel, stride, padding, dilation, output_padding)?;
        let out_channels = weight.dims().as_ref()[1];

        let mut out_dims = alloc::vec::Vec::with_capacity(rank);
        if rank == 4 {
            out_dims.push(dims.as_ref()[0]);
        }
        out_dims.push(out_channels);
        out_dims.push(out_height);
        out_dims.push(out_width);
        let output_shape =
            ShapeValue::<crate::shapes::Dyn>::try_new(ShapeBuf::from_slice(&out_dims))
                .map_err(Error::Shape)?;

        let mut inputs = alloc::vec::Vec::with_capacity(3);
        inputs.push(TensorHandle::from_storage::<B, K, Local>(&x.inner));
        inputs.push(TensorHandle::from_storage::<B, K, Local>(&weight.inner));
        if let Some(bias) = bias.as_ref() {
            inputs.push(TensorHandle::from_storage::<B, K, Local>(&bias.inner));
        }
        let context = ExecutionContext::from_scope(B::default());
        let out =
            crate::exec::dispatch::execute_shaped::<op::ConvTranspose2d, B, crate::shapes::Dyn>(
                &context,
                ConvTranspose2dAttributes {
                    stride: [stride, stride],
                    padding: [padding, padding],
                    output_padding: [output_padding, output_padding],
                    dilation: [dilation, dilation],
                    groups: self.groups,
                    has_bias: bias.is_some(),
                },
                &inputs,
                &output_shape,
            )
            .map_err(crate::err::Error::from)?;
        Tensor::from_shape_value(
            out.into(),
            output_shape,
            x._dtype.clone(),
            weight._device.clone(),
            x._grad,
        )
    }
}
