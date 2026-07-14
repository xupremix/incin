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
/// type S = s![64, 3, 3, 3]; // (COut, CIn, K, K)
/// let conv = Conv2d::<typenum::U3, typenum::U1, typenum::U0, typenum::U1, S, MyBackend>::new()?;
/// ```
#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct Conv2d<
    S: Conv2dShape,
    B: Backend,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
> {
    pub weight: Param<S::WeightShape, B>,
    pub bias: Option<Param<S::BiasShape, B>>,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias)>,
}

pub trait Conv2dShape: Shape + DynShape {
    type OutC: Dim;
    type InC: Dim;
    type K: Unsigned + Dim<Arg = ()>;
    type S: Unsigned + Dim<Arg = ()>;
    type P: Unsigned + Dim<Arg = ()>;
    type D: Unsigned + Dim<Arg = ()>;
    type WeightArg: crate::tensor::arg_into::NotUnit;
    type BiasArg: crate::tensor::arg_into::NotUnit;
    type WeightShape: Shape<Arg = Self::WeightArg> + DynShape;
    type BiasShape: Shape<Arg = Self::BiasArg> + DynShape;
    type Target;

    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg);
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
    type OutC = OutC;
    type InC = InC;
    type K = K;
    type S = S;
    type P = P;
    type D = D;
    type WeightArg = (
        <OutC as Dim>::Arg,
        <InC as Dim>::Arg,
        <K as Dim>::Arg,
        <K as Dim>::Arg,
    );
    type BiasArg = (<OutC as Dim>::Arg,);
    type WeightShape = (OutC, InC, K, K);
    type BiasShape = (OutC,);
    type Target = (OutC::Arg, InC::Arg);

    #[inline]
    fn build_args(target: Self::Target) -> (usize, usize, Self::WeightArg, Self::BiasArg) {
        let out_channels = OutC::from_arg(target.0.clone()).size();
        let in_channels = InC::from_arg(target.1.clone()).size();
        K::from_arg(()).arg();
        (
            out_channels,
            in_channels,
            (target.0, target.1, (), ()),
            (Default::default(),),
        )
    }
}

// ── Bias = True ────────────────────────────────────────────────────────────────
impl<S: Conv2dShape, B: Backend + crate::tensor::backend::ModuleOps<B>> Conv2d<S, B, crate::nn::optional::True>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new_with(args: S::Target) -> Result<Self> {
        let (_cout, _cin, w_args, b_args) = S::build_args(args);
        let fan_in = _cin * S::K::USIZE * S::K::USIZE;
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
            _phantom: core::marker::PhantomData,
        })
    }
}
impl<S, B> Conv2d<S, B, crate::nn::optional::True>
where
    S: Conv2dShape<Target = ((), ())>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new() -> Result<Self> {
        Self::new_with(((), ()))
    }
}

// ── Bias = False ───────────────────────────────────────────────────────────────
impl<S: Conv2dShape, B: Backend + crate::tensor::backend::ModuleOps<B>> Conv2d<S, B, crate::nn::optional::False>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new_with(args: S::Target) -> Result<Self> {
        let (_cout, _cin, w_args, _b_args) = S::build_args(args);
        let fan_in = _cin * S::K::USIZE * S::K::USIZE;
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
            _phantom: core::marker::PhantomData,
        })
    }
}
impl<S, B> Conv2d<S, B, crate::nn::optional::False>
where
    S: Conv2dShape<Target = ((), ())>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new() -> Result<Self> {
        Self::new_with(((), ()))
    }
}

// ── Bias = Dyn ─────────────────────────────────────────────────────────────────
impl<S: Conv2dShape, B: Backend + crate::tensor::backend::ModuleOps<B>> Conv2d<S, B, Dyn>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new_with(args: S::Target, has_bias: bool) -> Result<Self> {
        let (_cout, _cin, w_args, b_args) = S::build_args(args);
        let fan_in = _cin * S::K::USIZE * S::K::USIZE;
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
            _phantom: core::marker::PhantomData,
        })
    }
}
impl<S, B> Conv2d<S, B, Dyn>
where
    S: Conv2dShape<Target = ((), ())>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(has_bias: bool) -> Result<Self> {
        Self::new_with(((), ()), has_bias)
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
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
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
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
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
    type Output = Tensor<I::Output, B>;
    type Error = Error;

    #[inline]
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
