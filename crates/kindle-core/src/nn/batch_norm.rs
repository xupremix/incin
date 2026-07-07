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
    type Channels: Dim;
}

impl<C: Dim> BatchNormShape for (C,) {
    type Channels = C;
}

#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct BatchNorm2d<S: BatchNormShape, B: Backend> {
    pub weight: Param<(S::Channels,), B>,
    pub bias: Param<(S::Channels,), B>,
    pub running_mean: Buffer<(S::Channels,), B>,
    pub running_var: Buffer<(S::Channels,), B>,
    #[module(ignore)]
    pub eps: f32,
    #[module(ignore)]
    pub momentum: f32,
    #[module(ignore)]
    _phantom: PhantomData<B>,
}

impl<S: BatchNormShape, B: Backend> BatchNorm2d<S, B>
where
    B::DType: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
{
    pub fn new(args: <S::Channels as Dim>::Arg, eps: f32, momentum: f32) -> Result<Self> {
        let _c = S::Channels::from_arg(args.clone());
        Ok(Self {
            weight: Param::<(S::Channels,), B>::ones((args.clone(),))?,
            bias: Param::<(S::Channels,), B>::zeros((args.clone(),))?,
            running_mean: Buffer::<(S::Channels,), B>::zeros((args.clone(),))?,
            running_var: Buffer::<(S::Channels,), B>::ones((args,))?,
            eps,
            momentum,
            _phantom: PhantomData,
        })
    }

    pub fn new_dyn(c_size: usize, eps: f32, momentum: f32) -> Result<Self> {
        let c = S::Channels::from_size(c_size)
            .ok_or_else(|| Error::Msg("Invalid channel size".into()))?;
        Self::new(c.arg(), eps, momentum)
    }
}

impl<S: BatchNormShape, InS: Shape + HasChannels2D<S::Channels>, B: Backend> Module<Tensor<InS, B>>
    for BatchNorm2d<S, B>
{
    type Output = Tensor<InS, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<InS, B>) -> core::result::Result<Self::Output, Self::Error> {
        let weight = self.weight.as_tensor()?.into_dyn();
        let bias = self.bias.as_tensor()?.into_dyn();
        let running_mean = self.running_mean.as_tensor()?.into_dyn();
        let running_var = self.running_var.as_tensor()?.into_dyn();
        
        let out = B::batch_norm(
            x.inner(),
            weight.inner(),
            bias.inner(),
            running_mean.inner(),
            running_var.inner(),
            self.eps,
        )?;
        Ok(Tensor::from_parts_unchecked(
            out,
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad.clone(),
        ))
    }
}
