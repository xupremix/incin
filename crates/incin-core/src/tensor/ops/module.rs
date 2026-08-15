//! Module operations (LayerNorm, BatchNorm, etc) for neural networks.
use crate::exec::catalog::{BatchNormAttributes, Descriptor, LayerNormAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::dist::Local;
use crate::err::{Error, Result};
use crate::shapes::{Dyn, DynShape, Shape};
use crate::tensor::backend::Backend;
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;
use crate::tensor::backend::Execute;

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
{
    #[inline]
    /// `layer_norm`.
    pub fn layer_norm(
        &self,
        weight: &Tensor<Dyn, B, K, G>,
        bias: &Tensor<Dyn, B, K, G>,
        eps: f32,
    ) -> Result<Tensor<S, B, K, G>>
    where
        B: Execute<op::LayerNorm>,
        <B as Execute<op::LayerNorm>>::Output: Into<B::Storage<K>>,
    {
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&weight.inner),
            TensorHandle::from_storage::<B, K, Local>(&bias.inner),
        ];
        let attributes = LayerNormAttributes {
            normalized_shape: weight.dims().into(),
            epsilon: eps as f64,
            has_bias: true,
        };
        let shape = self._shape.clone();
        let context =
            crate::tensor::grad::execution_context::<B, G>(&self._grad).with_training(true);
        let inner =
            dispatch::execute_shaped::<op::LayerNorm, B, S>(&context, attributes, &inputs, &shape)
                .map_err(Error::from)?;
        Tensor::from_shape_value(
            inner.into(),
            shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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
    ) -> Result<Tensor<S, B, K, G>>
    where
        B: Execute<op::BatchNorm>,
        <B as Execute<op::BatchNorm>>::Output: Into<B::Storage<K>>,
    {
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&self.inner),
            TensorHandle::from_storage::<B, K, Local>(&weight.inner),
            TensorHandle::from_storage::<B, K, Local>(&bias.inner),
            TensorHandle::from_storage::<B, K, Local>(&running_mean.inner),
            TensorHandle::from_storage::<B, K, Local>(&running_var.inner),
        ];
        let attributes = BatchNormAttributes {
            epsilon: eps as f64,
            momentum: 0.1,
            training: true,
            has_weight: true,
            has_bias: true,
            has_running_mean: true,
            has_running_variance: true,
        };
        let shape = self._shape.clone();
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner =
            dispatch::execute_shaped::<op::BatchNorm, B, S>(&context, attributes, &inputs, &shape)
                .map_err(Error::from)?;
        Tensor::from_shape_value(
            inner.into(),
            shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
