//! Common loss functions (MSE, L1, BCE, CrossEntropy) for training.
//!
//! This module provides standard loss functions used to train neural networks.
//! Loss functions automatically compute and track their required reduction shape
//! (e.g. reducing down to a scalar or maintaining a batched shape) using type-level
//! logic to ensure that backpropagation can flow correctly from the scalar loss.
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::catalog::{LossAttributes, LossReduction, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::Shape;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;
use crate::tensor::backend::Backend;
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;
use crate::tensor::reduction::{
    BceReductionShape, CrossEntropyReductionShape, L1ReductionShape, Mean, MseReductionShape,
    Reduction, ReductionMode,
};
use alloc::vec::Vec;

fn execute_loss_descriptor<
    O,
    S: Shape,
    S2: Shape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
    G2: RequiresGrad,
    L: crate::shapes::Layout<S>,
    L2: crate::shapes::Layout<S2>,
>(
    prediction: &Tensor<S, B, K, G, Local, L>,
    target: &Tensor<S2, B, K, G2, Local, L2>,
    reduction: Reduction,
) -> Result<<B as Execute<O>>::Output>
where
    O: crate::exec::catalog::Operation<Attributes = crate::exec::catalog::LossAttributes>,
    B: Execute<O> + crate::exec::Capabilities,
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
    let context = crate::tensor::grad::execution_context::<B, G>(&prediction._grad);
    dispatch::execute::<O, B>(&context, LossAttributes { reduction }, &inputs)
        .map_err(crate::err::Error::from)
}

impl<
    S: Shape + crate::shapes::DynShape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
    L: crate::shapes::Layout<S>,
> Tensor<S, B, K, G, crate::dist::Local, L>
{
    /// Computes the Cross Entropy loss between predictions and target labels.
    /// Uses the default `Mean` reduction.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let pred = Cpu.zeros(shape![2, 10]).unwrap();
    /// let target = Cpu.tensor([0i64, 0]).unwrap();
    /// let loss = pred.cross_entropy_loss(&target).unwrap();
    /// ```
    pub fn cross_entropy_loss<
        S2: Shape,
        KT: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, KT, G2, Local, L2>,
    ) -> Result<Tensor<crate::shapes::Nil, B, K, G>>
    where
        B: Execute<op::CrossEntropyLoss>,
        <B as Execute<op::CrossEntropyLoss>>::Output: Into<B::Storage<K>>,
    {
        self.cross_entropy_loss_with::<Mean, S2, KT, G2, L2>(target)
    }

    /// `cross_entropy_loss_with` picks the reduction at the type level.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let pred = Cpu.zeros(shape![2, 10]).unwrap();
    /// let target = Cpu.tensor([0i64, 0]).unwrap();
    /// // `NoneReduction` keeps one loss per example instead of averaging.
    /// let loss = pred.cross_entropy_loss_with::<NoneReduction, _, _, _, _>(&target).unwrap();
    /// assert_eq!(loss.dims().dims(), &[2]);
    /// // Uniform logits give -ln(1/10) per row.
    /// let vals = loss.to_vec1::<f32>().unwrap();
    /// let expected = 10.0f32.ln();
    /// assert!((vals[0] - expected).abs() < 1e-4);
    /// assert!((vals[1] - expected).abs() < 1e-4);
    /// ```
    pub fn cross_entropy_loss_with<
        R,
        S2: Shape,
        KT: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, KT, G2, Local, L2>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + CrossEntropyReductionShape<S>,
        B: Execute<op::CrossEntropyLoss>,
        <B as Execute<op::CrossEntropyLoss>>::Output: Into<B::Storage<K>>,
    {
        let prediction = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let target_handle = TensorHandle::from_storage::<B, KT, Local>(&target.inner);
        let reduction = match R::as_enum() {
            Reduction::None => LossReduction::None,
            Reduction::Mean => LossReduction::Mean,
            Reduction::Sum => LossReduction::Sum,
        };
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = dispatch::execute::<op::CrossEntropyLoss, B>(
            &context,
            LossAttributes { reduction },
            &[prediction, target_handle],
        )
        .map_err(crate::err::Error::from)?;
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
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let pred = Cpu.ones(shape![2]).unwrap();
    /// let target = Cpu.zeros(shape![2]).unwrap();
    /// let loss = pred.mse_loss(&target).unwrap();
    /// ```
    pub fn mse_loss<S2: Shape, G2: RequiresGrad, L2: crate::shapes::Layout<S2>>(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
    ) -> Result<Tensor<crate::shapes::Nil, B, K, G>>
    where
        B: Execute<op::MseLoss> + crate::exec::Capabilities,
        <B as Execute<op::MseLoss>>::Output: Into<B::Storage<K>>,
    {
        self.mse_loss_with::<Mean, S2, G2, L2>(target)
    }

    /// `mse_loss_with` picks the reduction at the type level.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let pred = Cpu.tensor([1.0f32, 2.0]).unwrap();
    /// let target = Cpu.zeros(shape![2]).unwrap();
    /// // Unreduced: one squared error per element, same shape as `pred`.
    /// let loss = pred.mse_loss_with::<NoneReduction, _, _, _>(&target).unwrap();
    /// assert_eq!(loss.to_vec1::<f32>().unwrap(), vec![1.0, 4.0]);
    /// ```
    pub fn mse_loss_with<R, S2: Shape, G2: RequiresGrad, L2: crate::shapes::Layout<S2>>(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + MseReductionShape<S>,
        B: Execute<op::MseLoss> + crate::exec::Capabilities,
        <B as Execute<op::MseLoss>>::Output: Into<B::Storage<K>>,
    {
        let inner = execute_loss_descriptor::<op::MseLoss, S, S2, B, K, G, G2, _, _>(
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

    /// Computes the Mean Absolute Error (L1) loss, reduced to a scalar.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let pred = Cpu.tensor([1.0f32, 2.0]).unwrap();
    /// let target = Cpu.tensor([0.0f32, 4.0]).unwrap();
    /// // (|1 - 0| + |2 - 4|) / 2 = 1.5
    /// let loss = pred.l1_loss(&target).unwrap();
    /// assert!((loss.to_scalar::<f32>().unwrap() - 1.5).abs() < 1e-6);
    /// ```
    pub fn l1_loss<S2: Shape, G2: RequiresGrad, L2: crate::shapes::Layout<S2>>(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
    ) -> Result<Tensor<crate::shapes::Nil, B, K, G>>
    where
        B: Execute<op::L1Loss> + crate::exec::Capabilities,
        <B as Execute<op::L1Loss>>::Output: Into<B::Storage<K>>,
    {
        self.l1_loss_with::<Mean, S2, G2, L2>(target)
    }

    /// `l1_loss_with` picks the reduction at the type level.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let pred = Cpu.tensor([1.0f32, 2.0]).unwrap();
    /// let target = Cpu.tensor([0.0f32, 4.0]).unwrap();
    /// // Unreduced: one absolute error per element.
    /// let loss = pred.l1_loss_with::<NoneReduction, _, _, _>(&target).unwrap();
    /// assert_eq!(loss.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    /// ```
    pub fn l1_loss_with<R, S2: Shape, G2: RequiresGrad, L2: crate::shapes::Layout<S2>>(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + L1ReductionShape<S>,
        B: Execute<op::L1Loss> + crate::exec::Capabilities,
        <B as Execute<op::L1Loss>>::Output: Into<B::Storage<K>>,
    {
        let inner = execute_loss_descriptor::<op::L1Loss, S, S2, B, K, G, G2, _, _>(
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

    /// Computes the binary cross-entropy loss from logits, reduced to a scalar.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let logits = Cpu.zeros(shape![2]).unwrap();
    /// let target = Cpu.tensor([1.0f32, 0.0]).unwrap();
    /// // At logit 0 both labels cost ln(2); the mean of two ln(2)s is ln(2).
    /// let loss = logits.bce_with_logits_loss(&target).unwrap();
    /// assert!((loss.to_scalar::<f32>().unwrap() - 0.6931472).abs() < 1e-5);
    /// ```
    pub fn bce_with_logits_loss<S2: Shape, G2: RequiresGrad, L2: crate::shapes::Layout<S2>>(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
    ) -> Result<Tensor<crate::shapes::Nil, B, K, G>>
    where
        B: Execute<op::BceWithLogitsLoss> + crate::exec::Capabilities,
        <B as Execute<op::BceWithLogitsLoss>>::Output: Into<B::Storage<K>>,
    {
        self.bce_with_logits_loss_with::<Mean, S2, G2, L2>(target)
    }

    /// `bce_with_logits_loss_with` picks the reduction at the type level.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let logits = Cpu.zeros(shape![2]).unwrap();
    /// let target = Cpu.tensor([1.0f32, 0.0]).unwrap();
    /// // Unreduced: one loss per element.
    /// let loss = logits.bce_with_logits_loss_with::<NoneReduction, _, _, _>(&target).unwrap();
    /// let vals = loss.to_vec1::<f32>().unwrap();
    /// assert_eq!(vals.len(), 2);
    /// assert!((vals[0] - 0.6931472).abs() < 1e-5);
    /// assert!((vals[1] - 0.6931472).abs() < 1e-5);
    /// ```
    pub fn bce_with_logits_loss_with<
        R,
        S2: Shape,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: ReductionMode + BceReductionShape<S>,
        B: Execute<op::BceWithLogitsLoss> + crate::exec::Capabilities,
        <B as Execute<op::BceWithLogitsLoss>>::Output: Into<B::Storage<K>>,
    {
        let inner = execute_loss_descriptor::<op::BceWithLogitsLoss, S, S2, B, K, G, G2, _, _>(
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
