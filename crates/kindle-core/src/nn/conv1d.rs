use crate::nn::{Module, Param};
use crate::prelude::*;
use typenum::Unsigned;

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
/// Auto-generated documentation for Conv1d.
pub struct Conv1d<
    S: Conv1dShape,
    B: Backend,
    Bias: crate::nn::optional::OptionalField = crate::nn::optional::True,
> {
    /// Auto-generated documentation for weight.
    pub weight: Param<S::WeightShape, B>,
    /// Auto-generated documentation for bias.
    pub bias: Option<Param<S::BiasShape, B>>,
    #[module(ignore)]
    /// Auto-generated documentation for stride.
    pub stride: usize,
    #[module(ignore)]
    /// Auto-generated documentation for padding.
    pub padding: usize,
    #[module(ignore)]
    /// Auto-generated documentation for dilation.
    pub dilation: usize,
    #[module(ignore)]
    /// Auto-generated documentation for groups.
    pub groups: usize,
    #[module(ignore)]
    _phantom: core::marker::PhantomData<(S, B, Bias)>,
}

/// Auto-generated documentation for Conv1dShape.
pub trait Conv1dShape: Shape + DynShape {
    /// Auto-generated documentation for OutC.
    type OutC: Dim;
    /// Auto-generated documentation for InC.
    type InC: Dim;
    /// Auto-generated documentation for K.
    type K: Unsigned + Dim<Arg = ()>;
    /// Auto-generated documentation for S.
    type S: Unsigned + Dim<Arg = ()>;
    /// Auto-generated documentation for P.
    type P: Unsigned + Dim<Arg = ()>;
    /// Auto-generated documentation for D.
    type D: Unsigned + Dim<Arg = ()>;
    /// Auto-generated documentation for WeightArg.
    type WeightArg: crate::tensor::arg_into::NotUnit;
    /// Auto-generated documentation for BiasArg.
    type BiasArg: crate::tensor::arg_into::NotUnit;
    /// Auto-generated documentation for WeightShape.
    type WeightShape: Shape<Arg = Self::WeightArg> + DynShape;
    /// Auto-generated documentation for BiasShape.
    type BiasShape: Shape<Arg = Self::BiasArg> + DynShape;
    /// Auto-generated documentation for Target.
    type Target;

    /// Auto-generated documentation for build_args.
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
    /// Auto-generated documentation for OutC.
    type OutC = OutC;
    /// Auto-generated documentation for InC.
    type InC = InC;
    /// Auto-generated documentation for K.
    type K = K;
    /// Auto-generated documentation for S.
    type S = S;
    /// Auto-generated documentation for P.
    type P = P;
    /// Auto-generated documentation for D.
    type D = D;
    /// Auto-generated documentation for WeightArg.
    type WeightArg = (<OutC as Dim>::Arg, <InC as Dim>::Arg, <K as Dim>::Arg);
    /// Auto-generated documentation for BiasArg.
    type BiasArg = (<OutC as Dim>::Arg,);
    /// Auto-generated documentation for WeightShape.
    type WeightShape = (OutC, InC, K);
    /// Auto-generated documentation for BiasShape.
    type BiasShape = (OutC,);
    /// Auto-generated documentation for Target.
    type Target = (OutC::Arg, InC::Arg);

    #[inline]
    /// Auto-generated documentation for build_args.
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
    /// Auto-generated documentation for new_with.
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
    /// Auto-generated documentation for new.
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
    /// Auto-generated documentation for new_with.
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
    /// Auto-generated documentation for new.
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
    /// Auto-generated documentation for new_with.
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
    /// Auto-generated documentation for new.
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
    /// Auto-generated documentation for Output.
    type Output = Tensor<I::Output, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
    /// Auto-generated documentation for Output.
    type Output = Tensor<I::Output, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
    /// Auto-generated documentation for Output.
    type Output = Tensor<I::Output, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
