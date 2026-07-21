use crate::nn::{Module, Param};
use crate::prelude::*;
use core::marker::PhantomData;

/// Shape traits for RMSNorm.
pub trait RMSNormShape: Shape + DynShape {
    type Channels: Dim;
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    type Target;
    fn build_args(target: Self::Target) -> Self::BuildArg;
}

impl<C: Dim> RMSNormShape for (C,) {
    type Channels = C;
    type BuildArg = (<C as Dim>::Arg,);
    type Target = (<C as Dim>::Arg,);

    fn build_args(target: Self::Target) -> Self::BuildArg {
        target
    }
}

/// Root Mean Square Normalization (RMSNorm).
///
/// RMSNorm normalizes the input over the channel dimension without centering the mean,
/// improving training speed and stability. Widely used in modern LLMs (e.g. LLaMA).
#[derive(Debug, Clone)]
#[kindle_macros::module(internal)]
pub struct RMSNorm<S: RMSNormShape, B: Backend> {
    pub weight: Param<(S::Channels,), B>,
    #[module(ignore)]
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<(S, B)>,
}

impl<S: RMSNormShape, B: Backend> RMSNorm<S, B>
where
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Channels,): Shape<Arg = S::BuildArg>,
{
    pub fn new_with(args: S::Target, eps: f32) -> Result<Self> {
        let b_args = S::build_args(args);

        let args_data = crate::tensor::arg_into::TensorArgsData {
            shape: b_args,
            dtype: (),
            device: (),
            grad: (),
        };

        let weight = Param::<(S::Channels,), B>::ones_raw(args_data.clone())?;

        Ok(Self {
            weight,
            eps,
            _phantom: PhantomData,
        })
    }
}

impl<S, B> RMSNorm<S, B>
where
    S: RMSNormShape<Target = ((),)>,
    B: Backend,
    B::FloatElem: crate::prelude::ConstDType,
    B::Device: crate::prelude::ConstDevice,
    (S::Channels,): Shape<Arg = S::BuildArg>,
{
    pub fn new(eps: f32) -> Result<Self> {
        Self::new_with(((),), eps)
    }
}

impl<
    S: RMSNormShape,
    InS: Shape + DynShape + crate::shapes::EndsWith<S::Channels>,
    B: Backend + crate::tensor::backend::ReductionOps<B> + crate::tensor::backend::FloatOps<B>,
> Module<Tensor<InS, B>> for RMSNorm<S, B>
{
    type Output = Tensor<InS, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<InS, B>) -> core::result::Result<Self::Output, Error> {
        // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        let weight = self.weight.as_tensor()?.into_dyn();
        
        let x_dims = InS::dims(&x._shape);
        let dim = x_dims.as_ref().len().saturating_sub(1);
        
        // x^2
        let x_sq = x.mul(&x)?;
        
        // mean(x^2)
        // We use backend dynamic reduce to keep shapes dynamic
        let mean_sq_inner = B::mean_keepdim(x_sq.inner(), dim)?;
        let mut mean_shape = InS::dims(&x._shape).into();
        if !mean_shape.is_empty() {
            mean_shape[dim] = 1;
        }
        let mean_sq = Tensor::<Dyn, B, B::FloatElem, B::Device, Grad>::from_parts_unchecked(
            mean_sq_inner,
            Dyn::from_dyn(&mean_shape).unwrap(),
            x._dtype.clone(),
            x._device.clone(),
            x._grad, // We propagate grad context
        );
        
        // mean(x^2) + eps
        let var = mean_sq.add_scalar(self.eps)?;
        
        // sqrt(mean(x^2) + eps)
        let std = var.sqrt()?;
        
        // x / std
        let dyn_x = x.into_dyn();
        let normed = dyn_x.broadcast_div(&std)?;
        
        // normed * weight
        let out = normed.broadcast_mul(&weight)?;
        
        out.into_shape::<InS>()
    }
}
