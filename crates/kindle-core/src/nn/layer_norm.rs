use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

/// `LayerNormShape`.
pub trait LayerNormShape: Shape + DynShape {
    /// `Channels`.
    type Channels: Dim;
    /// `BuildArg`.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> LayerNormShape for (C,) {
    /// `Channels`.
    type Channels = C;
    /// `BuildArg`.
    type BuildArg = (<C as Dim>::Arg,);

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target,)
    }
}

impl LayerNormShape for Dyn {
    type Channels = usize;
    type BuildArg = (usize,);
    fn build_args(target: usize) -> Self::BuildArg {
        (target,)
    }
}

#[derive(Debug)]
#[kindle_macros::module(internal)]
/// `LayerNorm`.
pub struct LayerNorm<S: LayerNormShape, B: Backend> {
    /// The learnable weight matrix parameter.
    pub weight: Param<(S::Channels,), B>,
    /// The optional learnable bias vector parameter.
    pub bias: Param<(S::Channels,), B>,
    #[module(ignore)]
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<(S, B)>,
}

impl<S: LayerNormShape, B: Backend> LayerNorm<S, B>
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
            )>,
    {
        use crate::tensor::arg_into::LayerArgInto;
        let (channels, dtype, device, eps) = args.into_layer_arg();
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
                shape,
                dtype,
                device,
                grad: (),
            })?;
        Ok(Self {
            weight,
            bias,
            eps,
            _phantom: PhantomData,
        })
    }
}

impl<
    S: LayerNormShape,
    InS: Shape + DynShape + crate::shapes::EndsWith<S::Channels>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
> Module<Tensor<InS, B>> for LayerNorm<S, B>
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<InS, B>;
    /// The error type returned if the forward pass fails.
    type Error = Error;

    #[inline]
    /// Runs the forward pass of this module on the given input.
    fn forward(&self, x: Tensor<InS, B>) -> core::result::Result<Self::Output, Error> {
        let weight = self.weight.as_tensor()?.into_dyn();
        let bias = self.bias.as_tensor()?.into_dyn();
        let out = B::layer_norm(x.inner(), weight.inner(), Some(bias.inner()), self.eps)?;
        Ok(Tensor::from_parts_unchecked(
            out,
            x._shape.clone(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad,
        ))
    }
}
