use crate::nn::{Module, Param};
use crate::prelude::*;
use typenum::Unsigned;

#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
/// A 1-D convolution layer: `y = x * W + b`, sliding a `[out_c, in_c, k]`
/// kernel over the input's trailing (length) dimension.
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

/// A shape marker trait specifying a [`Conv1d`] layer's channel counts and
/// compile-time-fixed kernel/stride/padding/dilation. The typical usage is
/// `(OutC, InC, K, S, P, D)` for a fully static layer.
pub trait Conv1dShape: Shape + DynShape {
    /// Number of output channels.
    type OutC: Dim;
    /// Number of input channels.
    type InC: Dim;
    /// Kernel (window) size.
    type K: Unsigned + Dim<Arg = ()>;
    /// Stride.
    type S: Unsigned + Dim<Arg = ()>;
    /// Padding.
    type P: Unsigned + Dim<Arg = ()>;
    /// Dilation.
    type D: Unsigned + Dim<Arg = ()>;
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
    ) -> (usize, usize, Self::WeightArg, Self::BiasArg);
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
    type WeightArg = (<OutC as Dim>::Arg, <InC as Dim>::Arg, <K as Dim>::Arg);
    /// The shape argument type used to construct the bias tensor.
    type BiasArg = (<OutC as Dim>::Arg,);
    /// The static shape type of the weight parameter tensor.
    type WeightShape = (OutC, InC, K);
    /// The static shape type of the bias parameter tensor.
    type BiasShape = (OutC,);

    #[inline]
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(
        target: (<Self::OutC as Dim>::Arg, <Self::InC as Dim>::Arg),
    ) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
        let out_channels = OutC::from_arg(target.0.clone()).size();
        let in_channels = InC::from_arg(target.1.clone()).size();
        (
            out_channels,
            in_channels,
            (target.0.clone(), target.1, K::from_arg(()).arg()),
            (target.0,),
        )
    }
}

impl<S, B, Bias> Conv1d<S, B, Bias>
where
    S: Conv1dShape,
    B: Backend + SupportsDType<B::FloatElem>,
    Bias: crate::nn::optional::OptionalField,
    <B::FloatElem as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::OutC as Dim>::Arg,
                <S::InC as Dim>::Arg,
                <B::FloatElem as DType>::Arg,
                <B::Device as Device>::Arg,
                <Bias as crate::nn::optional::OptionalField>::Arg,
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (out_c, in_c, dtype, device, bias_arg) = args.into_layer_arg();
        let (_cout, cin, weight_shape, bias_shape) = S::build_args((out_c, in_c));
        let init = crate::nn::init::Init::KaimingUniform {
            fan_in: cin * S::K::USIZE,
            a: f64::sqrt(5.0),
        };
        let weight = Param::<S::WeightShape, B>::new_init_raw(
            crate::tensor::arg_into::TensorArgsData {
                shape: weight_shape,
                dtype: dtype.clone(),
                device: device.clone(),
                grad: (),
            },
            init,
        )?;
        let bias = if Bias::init(bias_arg) {
            Some(Param::<S::BiasShape, B>::new_init_raw(
                crate::tensor::arg_into::TensorArgsData {
                    shape: bias_shape,
                    dtype,
                    device,
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
            )?;

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
            )?;

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
            )?;

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
