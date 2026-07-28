use crate::nn::{Module, Param};
use crate::prelude::*;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::field_from_dims;
use core::marker::PhantomData;

/// Shape traits for RMSNorm.
pub trait RMSNormShape: Shape + DynShape {
    type Channels: Dim;
    type BuildArg: crate::tensor::arg_into::NotUnit + Clone;
    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg;
}

impl<C: Dim> RMSNormShape for (C,) {
    type Channels = C;
    type BuildArg = (<C as Dim>::Arg,);

    fn build_args(target: <Self::Channels as Dim>::Arg) -> Self::BuildArg {
        (target,)
    }
}

impl RMSNormShape for Dyn {
    type Channels = usize;
    type BuildArg = (usize,);
    fn build_args(target: usize) -> Self::BuildArg {
        (target,)
    }
}

/// Root Mean Square Normalization (RMSNorm).
///
/// RMSNorm normalizes the input over the channel dimension without centering the mean,
/// improving training speed and stability. Widely used in modern LLMs (e.g. LLaMA).
#[derive(Debug, Clone)]
#[incin_macros::module(internal)]
pub struct RMSNorm<S: RMSNormShape, B: Backend> {
    pub weight: Param<(S::Channels,), B>,
    #[module(ignore)]
    pub eps: f32,
    #[module(ignore)]
    _phantom: PhantomData<(S, B)>,
}

impl<S: RMSNormShape, B: Backend> RMSNorm<S, B>
where
    B: SupportsDType<B::FloatElem>,
    (S::Channels,): Shape<Arg = S::BuildArg>,
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
                shape,
                dtype,
                device,
                grad: (),
            })?;
        Ok(Self {
            weight,
            eps,
            _phantom: PhantomData,
        })
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
        let mean_sq = Tensor::<Dyn, B, B::FloatElem, Grad>::from_parts(
            mean_sq_inner,
            field_from_dims::<Dyn>(OperationKind::Normalization, &mean_shape)?,
            x._dtype.clone(),
            x._device.clone(),
            x._grad, // We propagate grad context
        )?;

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
