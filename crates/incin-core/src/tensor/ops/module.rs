//! Module operations (LayerNorm, BatchNorm, etc) for neural networks.
use crate::dist::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::{BatchNormAttributes, LayerNormAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::{Dyn, DynShape, Shape};
use crate::tensor::backend::Backend;
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
{
    #[inline]
    /// `layer_norm`.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let x = Cpu.tensor([[1.0f32, 2.0, 3.0, 4.0]]).unwrap();
    /// let weight = Cpu.ones(vec![4]).unwrap();
    /// let bias = Cpu.zeros(vec![4]).unwrap();
    /// let y = x.layer_norm(&weight, &bias, 1e-5).unwrap();
    /// assert_eq!(y.dims().dims(), &[1, 4]);
    /// // Normalized rows have (approximately) zero mean.
    /// let mean = y.to_vec1::<f32>().unwrap().iter().sum::<f32>() / 4.0;
    /// assert!(mean.abs() < 1e-4);
    /// ```
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
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let x = Cpu.tensor([[1.0f32, 2.0], [3.0, 4.0]]).unwrap();
    /// let weight = Cpu.ones(vec![2]).unwrap();
    /// let bias = Cpu.zeros(vec![2]).unwrap();
    /// let running_mean = Cpu.zeros(vec![2]).unwrap();
    /// let running_var = Cpu.ones(vec![2]).unwrap();
    /// let y = x
    ///     .batch_norm(&weight, &bias, &running_mean, &running_var, 1e-5)
    ///     .unwrap();
    /// assert_eq!(y.dims().dims(), &[2, 2]);
    /// // Training mode normalizes each channel's batch mean to (near) zero.
    /// let v = y.to_vec1::<f32>().unwrap();
    /// let ch0 = (v[0] + v[2]) / 2.0;
    /// let ch1 = (v[1] + v[3]) / 2.0;
    /// assert!(ch0.abs() < 1e-4);
    /// assert!(ch1.abs() < 1e-4);
    /// ```
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
