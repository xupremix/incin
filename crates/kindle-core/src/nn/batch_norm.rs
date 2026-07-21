use crate::nn::{Buffer, Module, Param};
use crate::prelude::*;

use core::marker::PhantomData;

/// A 2D Batch Normalization layer, as described in [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
///
/// Normalizes the input tensor across the batch and spatial dimensions for each channel independently,
/// applying learnable affine scaling (`weight`) and shift (`bias`) parameters.
///
/// The type parameter `S` is the shape marker.
///
/// ## Running Statistics
/// `running_mean` and `running_var` are non-trainable buffers updated during training (training mode
/// must be handled at the backend level). During inference, these stored statistics are used.
///
/// ## Examples
/// ```rust,ignore
/// use kindle::prelude::*;
///
/// // BatchNorm for 64-channel feature maps
/// let bn = BatchNorm2d::<(typenum::U64,), MyBackend>::new(typenum::U64::new(), 1e-5, 0.1)?;
/// ```
pub trait BatchNormShape: Shape + DynShape {
    /// Auto-generated documentation for Channels.
    type Channels: Dim;
    /// Auto-generated documentation for BuildArg.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// Auto-generated documentation for Target.
    type Target;
    /// Auto-generated documentation for build_args.
    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<C: Dim> BatchNormShape for (C,) {
    /// Auto-generated documentation for Channels.
    type Channels = C;
    /// Auto-generated documentation for BuildArg.
    type BuildArg = (<C as Dim>::Arg,);
    /// Auto-generated documentation for Target.
    type Target = (<C as Dim>::Arg,);

    /// Auto-generated documentation for build_args.
    fn build_args(target: Self::Target) -> Self::BuildArg {
        target
    }
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
/// Auto-generated documentation for BatchNorm2d.
pub struct BatchNorm2d<S: BatchNormShape, B: Backend> {
    /// Auto-generated documentation for weight.
    pub weight: Param<(S::Channels,), B>,
    /// Auto-generated documentation for bias.
    pub bias: Param<(S::Channels,), B>,
    /// Auto-generated documentation for running_mean.
    pub running_mean: Buffer<(S::Channels,), B>,
    /// Auto-generated documentation for running_var.
    pub running_var: Buffer<(S::Channels,), B>,
    #[module(ignore)]
    /// Auto-generated documentation for eps.
    pub eps: f32,
    #[module(ignore)]
    /// Auto-generated documentation for momentum.
    pub momentum: f32,
    #[module(ignore)]
    _phantom: PhantomData<B>,
}

impl<S: BatchNormShape, B: Backend> BatchNorm2d<S, B>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Channels,): Shape<Arg = S::BuildArg>,
{
    /// Auto-generated documentation for new_with.
    pub fn new_with(args: S::Target, eps: f32, momentum: f32) -> Result<Self> {
        let b_args = S::build_args(args);

        let args_data = crate::tensor::arg_into::TensorArgsData {
            shape: b_args,
            dtype: (),
            device: (),
            grad: (),
        };

        let weight = Param::<(S::Channels,), B>::ones_raw(args_data.clone())?;
        let bias = Param::<(S::Channels,), B>::zeros_raw(args_data.clone())?;
        let running_mean = Buffer::<(S::Channels,), B>::zeros_raw(args_data.clone())?;
        let running_var = Buffer::<(S::Channels,), B>::ones_raw(args_data.clone())?;

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            _phantom: PhantomData,
        })
    }
}

impl<S, B> BatchNorm2d<S, B>
where
    S: BatchNormShape<Target = ((),)>,
    B: Backend,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Channels,): Shape<Arg = S::BuildArg>,
{
    /// Auto-generated documentation for new.
    pub fn new(eps: f32, momentum: f32) -> Result<Self> {
        Self::new_with(((),), eps, momentum)
    }
}

impl<
    S: BatchNormShape,
    InS: Shape + HasChannels2D<S::Channels>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
> Module<Tensor<InS, B>> for BatchNorm2d<S, B>
{
    /// Auto-generated documentation for Output.
    type Output = Tensor<InS, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
    fn forward(&self, x: Tensor<InS, B>) -> core::result::Result<Self::Output, Self::Error> {
        let weight = self.weight.as_tensor()?.into_dyn();
        let bias = self.bias.as_tensor()?.into_dyn();
        let running_mean = self.running_mean.as_tensor()?.into_dyn();
        let running_var = self.running_var.as_tensor()?.into_dyn();

        let out = B::batch_norm(
            x.inner(),
            Some(weight.inner()),
            Some(bias.inner()),
            Some(running_mean.inner()),
            Some(running_var.inner()),
            self.eps,
            self.momentum as f64,
        )?;
        Ok(Tensor::from_parts_unchecked(
            out,
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad,
        ))
    }
}
