//! Common loss functions (MSE, L1, BCE, CrossEntropy) for training.
//!
//! This module provides standard loss functions used to train neural networks.
//! Loss functions automatically compute and track their required reduction shape
//! (e.g. reducing down to a scalar or maintaining a batched shape) using type-level
//! logic to ensure that backpropagation can flow correctly from the scalar loss.
use crate::nn::loss::{
    BceReductionShape, CrossEntropyReductionShape, L1ReductionShape, Mean, MseReductionShape,
    Reduction, ReductionMode,
};
use crate::prelude::{Backend, RequiresGrad, Result, Shape, Tensor};
use crate::shapes::error::OperationKind;
use crate::shapes::shape::field_from_dims;
use alloc::vec::Vec;

impl<
    S: Shape + crate::prelude::DynShape,
    B: Backend + crate::tensor::backend::LossOps<B>,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S, B, K, G>
{
    /// Computes the Cross Entropy loss between predictions and target labels.
    /// Uses the default `Mean` reduction.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use incin::prelude::*;
    /// let pred = Tensor::<s![2, 10], DefaultBackend>::zeros(()).unwrap();
    /// let target = Tensor::<s![2], DefaultBackend>::zeros(()).unwrap();
    /// let loss = pred.cross_entropy_loss(&target).unwrap();
    /// ```
    pub fn cross_entropy_loss<S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<(), B, K, G>> {
        self.cross_entropy_loss_with::<Mean, S2>(target)
    }

    /// `cross_entropy_loss_with`.
    pub fn cross_entropy_loss_with<R, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + CrossEntropyReductionShape<S>,
    {
        let inner = B::cross_entropy_loss(&self.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
            if !out_shape_dims.is_empty() {
                out_shape_dims.remove(1); // usually class dim
            }
        }
        let out_shape = field_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Computes the Mean Squared Error (MSE) loss.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use incin::prelude::*;
    /// let pred = Tensor::<s![2], DefaultBackend>::ones(()).unwrap();
    /// let target = Tensor::<s![2], DefaultBackend>::zeros(()).unwrap();
    /// let loss = pred.mse_loss(&target).unwrap();
    /// ```
    pub fn mse_loss<S2: Shape>(&self, target: &Tensor<S2, B, K, G>) -> Result<Tensor<(), B, K, G>> {
        self.mse_loss_with::<Mean, S2>(target)
    }

    /// `mse_loss_with`.
    pub fn mse_loss_with<R, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + MseReductionShape<S>,
    {
        let inner = B::mse_loss(&self.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape = field_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// `l1_loss`.
    pub fn l1_loss<S2: Shape>(&self, target: &Tensor<S2, B, K, G>) -> Result<Tensor<(), B, K, G>> {
        self.l1_loss_with::<Mean, S2>(target)
    }

    /// `l1_loss_with`.
    pub fn l1_loss_with<R, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + L1ReductionShape<S>,
    {
        let inner = B::l1_loss(&self.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape = field_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// `bce_with_logits_loss`.
    pub fn bce_with_logits_loss<S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<(), B, K, G>> {
        self.bce_with_logits_loss_with::<Mean, S2>(target)
    }

    /// `bce_with_logits_loss_with`.
    pub fn bce_with_logits_loss_with<R, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + BceReductionShape<S>,
    {
        let inner = B::bce_with_logits_loss(&self.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape = field_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}
