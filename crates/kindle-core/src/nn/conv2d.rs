use crate::nn::{Module, Param};
use crate::prelude::*;
use typenum::Unsigned;

/// A 2D convolutional layer operating on 4D tensors of shape `[Batch, Channels, H, W]`.
///
/// The layer type parameters give the compiler enough information to verify output shapes statically:
/// * `K` — Kernel size (square, e.g. `typenum::U3` for a 3×3 kernel).
/// * `S` — Stride.
/// * `P` — Padding.
/// * `D` — Dilation.
/// * `W` — Weight shape (e.g. `(COut, CIn, K, K)` for a fully static layer).
///
/// The high-level entry point is `Tensor::conv2d(...)`, which accepts `S`, `P`, and a weight shape,
/// and verifies at compile time that the output dimension is non-negative.
///
/// ## Examples
/// ```rust,ignore
/// use kindle::prelude::*;
///
/// // A statically typed 3×3, stride-1, padding-0, dilation-1 conv: 3 in → 64 out
/// type S = s![64, 3, 3, 1, 0, 1]; // (OutC, InC, K, Stride, Padding, Dilation)
/// let conv = Conv2d::<S, MyBackend>::build(())?;
/// ```
#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Conv2d<
    S: Conv2dShape,
    B: Backend,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
> {
    /// The learnable weight matrix parameter.
    pub weight: Param<S::WeightShape, B>,
    /// The optional learnable bias vector parameter.
    pub bias: Option<Param<S::BiasShape, B>>,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias)>,
}

/// A shape marker trait specifying a [`Conv2d`] layer's channel counts and
/// compile-time-fixed kernel/stride/padding/dilation. The typical usage is
/// `(OutC, InC, K, S, P, D)` for a fully static, square-kernel layer.
pub trait Conv2dShape: Shape + DynShape {
    /// Number of output channels.
    type OutC: Dim;
    /// Number of input channels.
    type InC: Dim;
    /// Kernel (window) size — square, applied to both spatial dimensions.
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
> Conv2dShape for (OutC, InC, K, S, P, D)
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
    ) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
        let out_channels = OutC::from_arg(target.0.clone()).size();
        let in_channels = InC::from_arg(target.1.clone()).size();
        K::from_arg(()).arg();
        (
            out_channels,
            in_channels,
            (target.0.clone(), target.1, (), ()),
            (target.0,),
        )
    }
}

impl<S, B, Bias> Conv2d<S, B, Bias>
where
    S: Conv2dShape,
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
            fan_in: cin * S::K::USIZE * S::K::USIZE,
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
            _phantom: core::marker::PhantomData,
        })
    }
}

impl<I, S, B, COut: Dim, CIn: Dim> Module<Tensor<I, B>> for Conv2d<S, B, crate::nn::optional::True>
where
    S: Conv2dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels2D<CIn>,
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
        let batch_size: usize = x_shape[0..rank - 3].iter().product();
        let in_channels = x_shape[rank - 3];
        let height = x_shape[rank - 2];
        let width = x_shape[rank - 1];

        let x_inner = if rank > 4 {
            B::reshape(&x.inner, &[batch_size, in_channels, height, width])?
        } else {
            x.inner.clone()
        };

        let out = B::conv2d(
            &x_inner,
            &weight.inner,
            Some(self.bias.as_ref().unwrap().as_tensor()?.inner()),
            S::S::USIZE,
            S::P::USIZE,
            S::D::USIZE,
            1,
        )?;

        let shape =
            <I as crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                x.shape_field(),
                weight.dims()[0],
            );

        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 4 {
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

impl<I, S, B, COut: Dim, CIn: Dim> Module<Tensor<I, B>> for Conv2d<S, B, crate::nn::optional::False>
where
    S: Conv2dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels2D<CIn>,
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
        let batch_size: usize = x_shape[0..rank - 3].iter().product();
        let in_channels = x_shape[rank - 3];
        let height = x_shape[rank - 2];
        let width = x_shape[rank - 1];

        let x_inner = if rank > 4 {
            B::reshape(&x.inner, &[batch_size, in_channels, height, width])?
        } else {
            x.inner.clone()
        };

        let out = B::conv2d(
            &x_inner,
            &weight.inner,
            None,
            S::S::USIZE,
            S::P::USIZE,
            S::D::USIZE,
            1,
        )?;

        let shape =
            <I as crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                x.shape_field(),
                weight.dims()[0],
            );

        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 4 {
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

impl<I, S, B, COut: Dim, CIn: Dim> Module<Tensor<I, B>> for Conv2d<S, B, Dyn>
where
    S: Conv2dShape<OutC = COut, InC = CIn>,
    I: Shape
        + DynShape
        + crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>
        + crate::shapes::HasChannels2D<CIn>,
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
        let batch_size: usize = x_shape[0..rank - 3].iter().product();
        let in_channels = x_shape[rank - 3];
        let height = x_shape[rank - 2];
        let width = x_shape[rank - 1];

        let x_inner = if rank > 4 {
            B::reshape(&x.inner, &[batch_size, in_channels, height, width])?
        } else {
            x.inner.clone()
        };

        let out = B::conv2d(
            &x_inner,
            &weight.inner,
            bias.as_ref().map(|b| b.inner()),
            S::S::USIZE,
            S::P::USIZE,
            S::D::USIZE,
            1,
        )?;

        let shape =
            <I as crate::shapes::SpatialConv2d<COut, S::K, S::S, S::P, S::D>>::compute_output_shape(
                x.shape_field(),
                weight.dims()[0],
            );

        let out_shape = <I::Output as DynShape>::dims(&shape);
        let out = if rank > 4 {
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
