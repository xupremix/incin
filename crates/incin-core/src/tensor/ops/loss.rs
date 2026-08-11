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
use crate::dist::placement::Local;
use crate::exec::catalog::{Descriptor, LossAttributes, LossReduction, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;
use crate::tensor::backend::Execute;
use alloc::vec::Vec;

fn execute_loss_descriptor<O, S: Shape, S2: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>(
    prediction: &Tensor<S, B, K, G>,
    target: &Tensor<S2, B, K, G>,
    reduction: Reduction,
) -> Result<<B as Execute<Descriptor<O>>>::Output>
where
    O: crate::exec::catalog::Operation<Attributes = crate::exec::catalog::LossAttributes>,
    B: Execute<Descriptor<O>> + crate::exec::Capabilities,
{
    let inputs = [
        TensorHandle::from_storage::<B, K, Local>(&prediction.inner),
        TensorHandle::from_storage::<B, K, Local>(&target.inner),
    ];
    let reduction = match reduction {
        Reduction::None => LossReduction::None,
        Reduction::Mean => LossReduction::Mean,
        Reduction::Sum => LossReduction::Sum,
    };
    let context = ExecutionContext::from_scope(B::default());
    dispatch::execute::<O, B>(&context, LossAttributes { reduction }, &inputs)
        .map_err(crate::prelude::Error::from)
}

impl<
    S: Shape + crate::prelude::DynShape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S, B, K, G>
{
    /// Computes the Cross Entropy loss between predictions and target labels.
    /// Uses the default `Mean` reduction.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let pred = Tensor::<s![2, 10], DefaultBackend>::zeros(()).unwrap();
    /// let target = Tensor::<s![2], DefaultBackend>::zeros(()).unwrap();
    /// let loss = pred.cross_entropy_loss(&target).unwrap();
    /// ```
    pub fn cross_entropy_loss<S2: Shape, KT: crate::tensor::dtype::DType>(
        &self,
        target: &Tensor<S2, B, KT, G>,
    ) -> Result<Tensor<(), B, K, G>>
    where
        B: Execute<Descriptor<op::CrossEntropyLoss>>,
        <B as Execute<Descriptor<op::CrossEntropyLoss>>>::Output: Into<B::Storage<K>>,
    {
        self.cross_entropy_loss_with::<Mean, S2, KT>(target)
    }

    /// `cross_entropy_loss_with`.
    pub fn cross_entropy_loss_with<R, S2: Shape, KT: crate::tensor::dtype::DType>(
        &self,
        target: &Tensor<S2, B, KT, G>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + CrossEntropyReductionShape<S>,
        B: Execute<Descriptor<op::CrossEntropyLoss>>,
        <B as Execute<Descriptor<op::CrossEntropyLoss>>>::Output: Into<B::Storage<K>>,
    {
        let prediction = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let target_handle = TensorHandle::from_storage::<B, KT, Local>(&target.inner);
        let reduction = match R::as_enum() {
            Reduction::None => LossReduction::None,
            Reduction::Mean => LossReduction::Mean,
            Reduction::Sum => LossReduction::Sum,
        };
        let context = ExecutionContext::from_scope(B::default());
        let inner = dispatch::execute::<op::CrossEntropyLoss, B>(
            &context,
            LossAttributes { reduction },
            &[prediction, target_handle],
        )
        .map_err(crate::prelude::Error::from)?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
            if !out_shape_dims.is_empty() {
                out_shape_dims.remove(1); // usually class dim
            }
        }
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(
            inner.into(),
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Computes the Mean Squared Error (MSE) loss.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let pred = Tensor::<s![2], DefaultBackend>::ones(()).unwrap();
    /// let target = Tensor::<s![2], DefaultBackend>::zeros(()).unwrap();
    /// let loss = pred.mse_loss(&target).unwrap();
    /// ```
    pub fn mse_loss<S2: Shape>(&self, target: &Tensor<S2, B, K, G>) -> Result<Tensor<(), B, K, G>>
    where
        B: Execute<Descriptor<op::MseLoss>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::MseLoss>>>::Output: Into<B::Storage<K>>,
    {
        self.mse_loss_with::<Mean, S2>(target)
    }

    /// `mse_loss_with`.
    pub fn mse_loss_with<R, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + MseReductionShape<S>,
        B: Execute<Descriptor<op::MseLoss>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::MseLoss>>>::Output: Into<B::Storage<K>>,
    {
        let inner = execute_loss_descriptor::<op::MseLoss, S, S2, B, K, G>(
            self,
            target,
            R::as_enum(),
        )?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(
            inner.into(),
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// `l1_loss`.
    pub fn l1_loss<S2: Shape>(&self, target: &Tensor<S2, B, K, G>) -> Result<Tensor<(), B, K, G>>
    where
        B: Execute<Descriptor<op::L1Loss>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::L1Loss>>>::Output: Into<B::Storage<K>>,
    {
        self.l1_loss_with::<Mean, S2>(target)
    }

    /// `l1_loss_with`.
    pub fn l1_loss_with<R, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + L1ReductionShape<S>,
        B: Execute<Descriptor<op::L1Loss>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::L1Loss>>>::Output: Into<B::Storage<K>>,
    {
        let inner = execute_loss_descriptor::<op::L1Loss, S, S2, B, K, G>(
            self,
            target,
            R::as_enum(),
        )?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(
            inner.into(),
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// `bce_with_logits_loss`.
    pub fn bce_with_logits_loss<S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<(), B, K, G>>
    where
        B: Execute<Descriptor<op::BceWithLogitsLoss>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::BceWithLogitsLoss>>>::Output: Into<B::Storage<K>>,
    {
        self.bce_with_logits_loss_with::<Mean, S2>(target)
    }

    /// `bce_with_logits_loss_with`.
    pub fn bce_with_logits_loss_with<R, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + BceReductionShape<S>,
        B: Execute<Descriptor<op::BceWithLogitsLoss>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::BceWithLogitsLoss>>>::Output: Into<B::Storage<K>>,
    {
        let inner = execute_loss_descriptor::<op::BceWithLogitsLoss, S, S2, B, K, G>(
            self,
            target,
            R::as_enum(),
        )?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(
            inner.into(),
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
