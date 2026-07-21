use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

/// `LayerNormShape`.
pub trait LayerNormShape: Shape + DynShape {
    /// `Channels`.
    type Channels: Dim;
    /// `BuildArg`.
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    /// The runtime arguments needed to instantiate this layer.
    type Target;
    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<C: Dim> LayerNormShape for (C,) {
    /// `Channels`.
    type Channels = C;
    /// `BuildArg`.
    type BuildArg = (<C as Dim>::Arg,);
    /// The runtime arguments needed to instantiate this layer.
    type Target = (<C as Dim>::Arg,);

    /// Converts the target arguments into concrete shape args for weight and bias tensors.
    fn build_args(target: Self::Target) -> Self::BuildArg {
        target
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
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Channels,): Shape<Arg = S::BuildArg>,
{
    /// Creates a new instance with explicitly provided shape arguments.
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
    /// Creates a new instance with default (statically inferred) shape arguments.
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
