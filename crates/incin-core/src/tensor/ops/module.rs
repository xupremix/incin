//! Module operations (LayerNorm, BatchNorm, etc) for neural networks.
use crate::prelude::{Backend, Dyn, DynShape, RequiresGrad, Result, Shape, Tensor};

impl<
    S: Shape + DynShape,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S, B, K, G>
{
    #[inline]
    /// `layer_norm`.
    pub fn layer_norm(
        &self,
        weight: &Tensor<Dyn, B, K, G>,
        bias: &Tensor<Dyn, B, K, G>,
        eps: f32,
    ) -> Result<Tensor<S, B, K, G>> {
        // weight and bias should technically be 1D tensors matching the last dimension, but we use DynShape for now
        let inner = B::layer_norm::<K>(&self.inner, &weight.inner, Some(&bias.inner), eps)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    #[inline]
    /// `batch_norm`.
    pub fn batch_norm(
        &self,
        weight: &Tensor<Dyn, B, K, G>,
        bias: &Tensor<Dyn, B, K, G>,
        running_mean: &Tensor<Dyn, B, K, G>,
        running_var: &Tensor<Dyn, B, K, G>,
        eps: f32,
    ) -> Result<Tensor<S, B, K, G>> {
        let inner = B::batch_norm::<K>(
            &self.inner,
            Some(&weight.inner),
            Some(&bias.inner),
            Some(&running_mean.inner),
            Some(&running_var.inner),
            eps,
            0.1,
        )?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}
