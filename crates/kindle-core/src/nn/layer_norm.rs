use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

/// Core abstraction for `LayerNormShape` within the Kindle framework..
pub trait LayerNormShape: Shape + DynShape {
    /// Core abstraction for `Channels` within the Kindle framework..
    type Channels: Dim;
    /// Core abstraction for `BuildArg` within the Kindle framework..
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// Core abstraction for `Target` within the Kindle framework..
    type Target;
    /// Core abstraction for `build_args` within the Kindle framework..
    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<C: Dim> LayerNormShape for (C,) {
    /// Core abstraction for `Channels` within the Kindle framework..
    type Channels = C;
    /// Core abstraction for `BuildArg` within the Kindle framework..
    type BuildArg = (<C as Dim>::Arg,);
    /// Core abstraction for `Target` within the Kindle framework..
    type Target = (<C as Dim>::Arg,);

    /// Core abstraction for `build_args` within the Kindle framework..
    fn build_args(target: Self::Target) -> Self::BuildArg {
        target
    }
}

#[derive(Debug)]
#[kindle_macros::module(internal)]
/// Core abstraction for `LayerNorm` within the Kindle framework..
pub struct LayerNorm<S: LayerNormShape, B: Backend> {
    /// Core abstraction for `weight` within the Kindle framework..
    pub weight: Param<(S::Channels,), B>,
    /// Core abstraction for `bias` within the Kindle framework..
    pub bias: Param<(S::Channels,), B>,
    #[module(ignore)]
    /// Core abstraction for `eps` within the Kindle framework..
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
    /// Core abstraction for `new_with` within the Kindle framework..
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
    /// Core abstraction for `new` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Tensor<InS, B>;
    /// Core abstraction for `Error` within the Kindle framework..
    type Error = Error;

    #[inline]
    /// Core abstraction for `forward` within the Kindle framework..
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
