use crate::nn::{Module, Param};
use crate::prelude::*;
use typenum::Unsigned;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
/// `Conv1d`.
pub struct Conv1d<
    S: Conv1dShape,
    B: Backend,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
> {
    /// The learnable weight matrix parameter.
    pub weight: Param<S::WeightShape, B>,
    /// The optional learnable bias vector parameter.
    pub bias: Option<Param<S::BiasShape, B>>,
    #[module(ignore)]
    /// Step size of the convolution kernel window.
    pub stride: usize,
    #[module(ignore)]
    /// Zero-padding added to both sides of the input.
    pub padding: usize,
    #[module(ignore)]
    /// Spacing between kernel elements.
    pub dilation: usize,
    #[module(ignore)]
    /// Number of blocked connections from input channels to output channels.
    pub groups: usize,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias)>,
}

/// `Conv1dShape`.
pub trait Conv1dShape: Shape + DynShape {
    /// `OutC`.
    type OutC: Dim;
    /// `InC`.
    type InC: Dim;
    /// `K`.
    type K: Unsigned + Dim<Arg = ()>;
    /// `S`.
    type S: Unsigned + Dim<Arg = ()>;
    /// `P`.
    type P: Unsigned + Dim<Arg = ()>;
    /// `D`.
    type D: Unsigned + Dim<Arg = ()>;
    /// The shape argument type used to construct the weight tensor.
    type WeightArg: crate::tensor::arg_into::NotUnit;
    /// The shape argument type used to construct the bias tensor.
    type BiasArg: crate::tensor::arg_into::NotUnit;
    /// The static shape type of the weight parameter tensor.
    type WeightShape: Shape<Arg = Self::WeightArg> + DynShape;
    /// The static shape type of the bias parameter tensor.
    type BiasShape: Shape<Arg = Self::BiasArg> + DynShape;
    /// The runtime arguments needed to instantiate this layer.
    type Target;

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg);
}

impl<
    OutC: Dim,
    InC: Dim,
    K: Unsigned + Dim<Arg = ()>,
    S: Unsigned + Dim<Arg = ()>,
    P: Unsigned + Dim<Arg = ()>,
    D: Unsigned + Dim<Arg = ()>,
> Conv1dShape for (OutC, InC, K, S, P, D)
{
    /// `OutC`.
    type OutC = OutC;
    /// `InC`.
    type InC = InC;
    /// `K`.
    type K = K;
    /// `S`.
    type S = S;
    /// `P`.
    type P = P;
    /// `D`.
    type D = D;
    /// The shape argument type used to construct the weight tensor.
    type WeightArg = (<OutC as Dim>::Arg, <InC as Dim>::Arg, <K as Dim>::Arg);
    /// The shape argument type used to construct the bias tensor.
    type BiasArg = (<OutC as Dim>::Arg,);
    /// The static shape type of the weight parameter tensor.
    type WeightShape = (OutC, InC, K);
    /// The static shape type of the bias parameter tensor.
    type BiasShape = (OutC,);
    /// The runtime arguments needed to instantiate this layer.
    type Target = (OutC::Arg, InC::Arg);

    #[inline]
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
        let out_channels = OutC::from_arg(target.0.clone()).size();
        let in_channels = InC::from_arg(target.1.clone()).size();
        (
            out_channels,
            in_channels,
            (target.0, target.1, K::from_arg(()).arg()),
            (Default::default(),),
        )
    }
}

// ── Bias = True ────────────────────────────────────────────────────────────────
impl<S: Conv1dShape, B: Backend + crate::tensor::backend::ModuleOps<B>>
    Conv1d<S, B, crate::nn::optional::True>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    /// Creates a new instance with explicitly provided shape arguments.
    pub fn new_with(args: S::Target) -> Result<Self> {
        let (_cout, _cin, w_args, b_args) = S::build_args(args);
        let fan_in = _cin * S::K::USIZE;
        let init = crate::nn::init::Init::KaimingUniform {
            fan_in,
            a: f64::sqrt(5.0),
        };
        let weight = Param::<S::WeightShape, B>::new_init_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape: w_args,
                dtype: (),
                device: (),
                grad: (),
            },
            init,
        )?;
        let bias = Some(Param::<S::BiasShape, B>::new_init_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape: b_args,
                dtype: (),
                device: (),
                grad: (),
            },
            init,
        )?);
        Ok(Self {
            weight,
            bias,
            stride: S::S::USIZE,
            padding: S::P::USIZE,
            dilation: S::D::USIZE,
            groups: 1,
            _phantom: core::marker::PhantomData,
        })
    }
}
impl<S, B> Conv1d<S, B, crate::nn::optional::True>
where
    S: Conv1dShape<Target = ((), ())>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Result<Self> {
        Self::new_with(((), ()))
    }
}

// ── Bias = False ───────────────────────────────────────────────────────────────
impl<S: Conv1dShape, B: Backend + crate::tensor::backend::ModuleOps<B>>
    Conv1d<S, B, crate::nn::optional::False>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    /// Creates a new instance with explicitly provided shape arguments.
    pub fn new_with(args: S::Target) -> Result<Self> {
        let (_cout, _cin, w_args, _b_args) = S::build_args(args);
        let fan_in = _cin * S::K::USIZE;
        let init = crate::nn::init::Init::KaimingUniform {
            fan_in,
            a: f64::sqrt(5.0),
        };
        let weight = Param::<S::WeightShape, B>::new_init_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape: w_args,
                dtype: (),
                device: (),
                grad: (),
            },
            init,
        )?;
        Ok(Self {
            weight,
            bias: None,
            stride: S::S::USIZE,
            padding: S::P::USIZE,
            dilation: S::D::USIZE,
            groups: 1,
            _phantom: core::marker::PhantomData,
        })
    }
}
impl<S, B> Conv1d<S, B, crate::nn::optional::False>
where
    S: Conv1dShape<Target = ((), ())>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Result<Self> {
        Self::new_with(((), ()))
    }
}

// ── Bias = Dyn ─────────────────────────────────────────────────────────────────
impl<S: Conv1dShape, B: Backend + crate::tensor::backend::ModuleOps<B>> Conv1d<S, B, Dyn>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    /// Creates a new instance with explicitly provided shape arguments.
    pub fn new_with(args: S::Target, has_bias: bool) -> Result<Self> {
        let (_cout, _cin, w_args, b_args) = S::build_args(args);
        let fan_in = _cin * S::K::USIZE;
        let init = crate::nn::init::Init::KaimingUniform {
            fan_in,
            a: f64::sqrt(5.0),
        };
        let weight = Param::<S::WeightShape, B>::new_init_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape: w_args,
                dtype: (),
                device: (),
                grad: (),
            },
            init,
        )?;
        let bias = if has_bias {
            Some(Param::<S::BiasShape, B>::new_init_raw(
                crate::tensor::arg_into::TensorArgsData {
                    shape: b_args,
                    dtype: (),
                    device: (),
                    grad: (),
                },
                init,
            )?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride: S::S::USIZE,
            padding: S::P::USIZE,
            dilation: S::D::USIZE,
            groups: 1,
            _phantom: core::marker::PhantomData,
        })
    }
}
impl<S, B> Conv1d<S, B, Dyn>
where
    S: Conv1dShape<Target = ((), ())>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(has_bias: bool) -> Result<Self> {
        Self::new_with(((), ()), has_bias)
    }
}

impl<I, S, B, COut: Dim, CIn: Dim> Module<Tensor<I, B>> for Conv1d<S, B, crate::nn::optional::True>
where
    S: Conv1dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv1d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels1D<CIn>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = Some(self.bias.as_ref().unwrap().as_tensor()?.detach());

        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size: usize = x_shape[0..rank - 2].iter().product();
        let in_channels = x_shape[rank - 2];
        let length = x_shape[rank - 1];

        let x_inner = if rank > 3 {
            B::reshape(&x.inner, &[batch_size, in_channels, length])?
        } else {
            x.inner.clone()
        };

        let out = B::conv1d(
            &x_inner,
            &weight.inner,
            bias.as_ref().map(|b| b.inner()),
            S::S::USIZE,
            S::P::USIZE,
            S::D::USIZE,
            self.groups,
        )?;

        let shape =
            <I as crate::shapes::SpatialConv1d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                x.shape_field(),
                weight.dims()[0],
            );

        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 3 {
            B::reshape(&out, out_shape.as_ref())?
        } else {
            out
        };

        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            weight._device.clone(),
            *x.grad_field(),
        ))
    }
}

impl<I, S, B, COut: Dim, CIn: Dim> Module<Tensor<I, B>> for Conv1d<S, B, crate::nn::optional::False>
where
    S: Conv1dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv1d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels1D<CIn>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;

        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size: usize = x_shape[0..rank - 2].iter().product();
        let in_channels = x_shape[rank - 2];
        let length = x_shape[rank - 1];

        let x_inner = if rank > 3 {
            B::reshape(&x.inner, &[batch_size, in_channels, length])?
        } else {
            x.inner.clone()
        };

        let out = B::conv1d(
            &x_inner,
            &weight.inner,
            None,
            S::S::USIZE,
            S::P::USIZE,
            S::D::USIZE,
            self.groups,
        )?;

        let shape =
            <I as crate::shapes::SpatialConv1d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                x.shape_field(),
                weight.dims()[0],
            );

        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 3 {
            B::reshape(&out, out_shape.as_ref())?
        } else {
            out
        };

        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            weight._device.clone(),
            *x.grad_field(),
        ))
    }
}

impl<I, S, B, COut: Dim, CIn: Dim> Module<Tensor<I, B>> for Conv1d<S, B, Dyn>
where
    S: Conv1dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv1d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels1D<CIn>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<I::Output, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<I, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?;
        let bias = match &self.bias {
            Some(b) => Some(b.as_tensor()?.detach()),
            None => None,
        };

        let x_shape = x.dims();
        let x_shape = x_shape.as_ref();
        let rank = x_shape.len();
        let batch_size: usize = x_shape[0..rank - 2].iter().product();
        let in_channels = x_shape[rank - 2];
        let length = x_shape[rank - 1];

        let x_inner = if rank > 3 {
            B::reshape(&x.inner, &[batch_size, in_channels, length])?
        } else {
            x.inner.clone()
        };

        let out = B::conv1d(
            &x_inner,
            &weight.inner,
            bias.as_ref().map(|b| b.inner()),
            S::S::USIZE,
            S::P::USIZE,
            S::D::USIZE,
            self.groups,
        )?;

        let shape =
            <I as crate::shapes::SpatialConv1d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                x.shape_field(),
                weight.dims()[0],
            );

        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 3 {
            B::reshape(&out, out_shape.as_ref())?
        } else {
            out
        };

        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
            x._dtype.clone(),
            weight._device.clone(),
            *x.grad_field(),
        ))
    }
}
