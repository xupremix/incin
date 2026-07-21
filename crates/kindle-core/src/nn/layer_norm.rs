use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

/// Auto-generated documentation for LayerNormShape.
pub trait LayerNormShape: Shape + DynShape {
    /// Auto-generated documentation for Channels.
    type Channels: Dim;
    /// Auto-generated documentation for BuildArg.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// Auto-generated documentation for Target.
    type Target;
    /// Auto-generated documentation for build_args.
    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<C: Dim> LayerNormShape for (C,) {
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

#[derive(Debug)]
#[kindle_macros::module(internal)]
/// Auto-generated documentation for LayerNorm.
pub struct LayerNorm<S: LayerNormShape, B: Backend> {
    /// Auto-generated documentation for weight.
    pub weight: Param<(S::Channels,), B>,
    /// Auto-generated documentation for bias.
    pub bias: Param<(S::Channels,), B>,
    #[module(ignore)]
    /// Auto-generated documentation for eps.
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<(S, B)>,
}

impl<S: LayerNormShape, B: Backend> LayerNorm<S, B>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Channels,): Shape<Arg = S::BuildArg>,
{
    /// Auto-generated documentation for new_with.
    pub fn new_with(args: S::Target, eps: f32) -> Result<Self> {
        let b_args = S::build_args(args);

        let args_data = crate::tensor::arg_into::TensorArgsData {
            shape: b_args,
            dtype: (),
            device: (),
            grad: (),
        };

        let weight = Param::<(S::Channels,), B>::ones_raw(args_data.clone())?;

        let bias = Param::<(S::Channels,), B>::zeros_raw(args_data.clone())?;

        Ok(Self {
            weight,
            bias,
            eps,
            _phantom: PhantomData,
        })
    }
}

impl<S, B> LayerNorm<S, B>
where
    S: LayerNormShape<Target = ((),)>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Channels,): Shape<Arg = S::BuildArg>,
{
    /// Auto-generated documentation for new.
    pub fn new(eps: f32) -> Result<Self> {
        Self::new_with(((),), eps)
    }
}

impl<
    S: LayerNormShape,
    InS: Shape + DynShape + crate::shapes::EndsWith<S::Channels>,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
> Module<Tensor<InS, B>> for LayerNorm<S, B>
{
    /// Auto-generated documentation for Output.
    type Output = Tensor<InS, B>;
    /// Auto-generated documentation for Error.
    type Error = Error;

    #[inline]
    /// Auto-generated documentation for forward.
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
