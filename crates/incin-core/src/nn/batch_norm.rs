use crate::nn::{Buffer, Module, Param};
use crate::prelude::*;

use core::marker::PhantomData;

/// A shape marker trait specifying a [`BatchNorm2d`] layer's channel
/// count. The typical usage is `(Channels,)` for a static layer, or `Dyn`
/// for a runtime-determined size.
pub trait BatchNormShape: Shape + DynShape {
    /// The number of channels being normalized.
    type Channels: Dim;
    /// The shape argument type used to construct the weight/bias/
    /// running-stat tensors.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> BatchNormShape for (C,) {
    /// The channel count.
    type Channels = C;
    /// A single-element `(channels_arg,)` tuple.
    type BuildArg = (<C as Dim>::Arg,);

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target,)
    }
}

impl BatchNormShape for Dyn {
    type Channels = usize;
    type BuildArg = (usize,);
    fn build_args(target: usize) -> Self::BuildArg {
        (target,)
    }
}

#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
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
/// use incin::prelude::*;
///
/// // BatchNorm for 64-channel feature maps
/// let bn = BatchNorm2d::<(typenum::U64,), MyBackend>::build((1e-5, 0.1))?;
/// ```
pub struct BatchNorm2d<S: BatchNormShape, B: Backend> {
    /// The learnable weight matrix parameter.
    pub weight: Param<(S::Channels,), B>,
    /// The optional learnable bias vector parameter.
    pub bias: Param<(S::Channels,), B>,
    /// Running mean buffer used during batch normalization inference.
    pub running_mean: Buffer<(S::Channels,), B>,
    /// Running variance buffer used during batch normalization inference.
    pub running_var: Buffer<(S::Channels,), B>,
    #[module(ignore)]
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f32,
    #[module(ignore)]
    /// Momentum factor for updating running statistics.
    pub momentum: f32,
    #[module(ignore)]
    _phantom: PhantomData<B>,
}

impl<S: BatchNormShape, B: Backend> BatchNorm2d<S, B>
where
    B: SupportsDType<B::FloatElem>,
    (S::Channels,): Shape<Arg = S::BuildArg>,
    <B::FloatElem as DType>::Arg: Clone,
    <B::Device as Device>::Arg: Clone,
{
    pub fn build<A>(args: A) -> Result<Self>
    where
        A: crate::tensor::arg_into::LayerArgInto<(
                <S::Channels as Dim>::Arg,
                <B::FloatElem as DType>::Arg,
                <B::Device as Device>::Arg,
                f32,
                f32,
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (channels, dtype, device, eps, momentum) = args.into_layer_arg();
        let shape = S::build_args(channels);
        let weight =
            Param::<(S::Channels,), B>::ones_raw(crate::tensor::arg_into::TensorArgsData {
                shape: shape.clone(),
                dtype: dtype.clone(),
                device: device.clone(),
                grad: (),
            })?;
        let bias =
            Param::<(S::Channels,), B>::zeros_raw(crate::tensor::arg_into::TensorArgsData {
                shape: shape.clone(),
                dtype: dtype.clone(),
                device: device.clone(),
                grad: (),
            })?;
        let running_mean =
            Buffer::<(S::Channels,), B>::zeros_raw(crate::tensor::arg_into::TensorArgsData {
                shape: shape.clone(),
                dtype: dtype.clone(),
                device: device.clone(),
                grad: (),
            })?;
        let running_var =
            Buffer::<(S::Channels,), B>::ones_raw(crate::tensor::arg_into::TensorArgsData {
                shape,
                dtype,
                device,
                grad: (),
            })?;
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

impl<
    S: BatchNormShape,
    InS: Shape + HasChannels2D<S::Channels>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
> Module<Tensor<InS, B>> for BatchNorm2d<S, B>
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<InS, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
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
