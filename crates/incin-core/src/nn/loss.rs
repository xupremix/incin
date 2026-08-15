use crate::backend_authoring::Backend;
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::catalog::{LossAttributes, LossReduction, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;
use crate::shapes::{Dim, DimCons, Dyn, Nil, Shape};
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::grad::{NoGrad, RequiresGrad};
pub use crate::tensor::reduction::{
    BceReductionShape, CrossEntropyReductionShape, L1ReductionShape, Mean, MseReductionShape,
    NoneReduction, Reduction, ReductionMode, Sum,
};
use alloc::vec::Vec;

/// Trait to statically verify that two shapes are identical for MSE loss.
pub trait MSEShape<S2: Shape> {}
impl<S: Shape> MSEShape<S> for S {}

/// Mean Squared Error Loss.
#[derive(Debug, Clone, Default)]
pub struct MSELoss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl<R: ReductionMode> MSELoss<R> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the Mean Squared Error between predictions and targets.
    pub fn forward<
        S: Shape + crate::shapes::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<op::MseLoss>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, G>,
        target: &Tensor<S, B, K, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: MseReductionShape<S>,
        <B as Execute<op::MseLoss>>::Output: Into<B::Storage<K>>,
    {
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&pred.inner),
            TensorHandle::from_storage::<B, K, Local>(&target.inner),
        ];
        let reduction = match R::as_enum() {
            Reduction::None => LossReduction::None,
            Reduction::Mean => LossReduction::Mean,
            Reduction::Sum => LossReduction::Sum,
        };
        let context = ExecutionContext::from_scope(B::default());
        let inner =
            dispatch::execute::<op::MseLoss, B>(&context, LossAttributes { reduction }, &inputs)
                .map_err(crate::err::Error::from)?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = pred.dims().into();
        }
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(
            inner.into(),
            out_shape,
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        )
    }
}

/// Trait to statically verify the shapes for CrossEntropyLoss.
/// Ensures the prediction tensor is `[Batch, Classes]` and the target is `[Batch]`.
pub trait CrossEntropyShape<S2: Shape> {}

// Static implementation: [Batch, Classes] vs [Batch]
impl<Batch: Dim, Classes: Dim> CrossEntropyShape<DimCons<Batch, Nil>>
    for DimCons<Batch, DimCons<Classes, Nil>>
{
}

// Dynamic fallback
impl CrossEntropyShape<Dyn> for Dyn {}
impl<Batch: Dim, Classes: Dim> CrossEntropyShape<Dyn> for DimCons<Batch, DimCons<Classes, Nil>> {}
impl<Batch: Dim> CrossEntropyShape<DimCons<Batch, Nil>> for Dyn {}

/// Cross Entropy Loss.
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyLoss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl<R: ReductionMode> CrossEntropyLoss<R> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the Cross Entropy Loss between predictions and targets.
    /// The target tensor MUST have `u32` elements at compile time.
    pub fn forward<
        S1,
        S2: Shape,
        B: Backend + crate::exec::Capabilities + Execute<op::CrossEntropyLoss>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S1, B, K, G>,
        target: &Tensor<S2, B, u32, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        S1: Shape + crate::shapes::DynShape + CrossEntropyShape<S2>,
        R: CrossEntropyReductionShape<S1>,
        <B as Execute<op::CrossEntropyLoss>>::Output: Into<B::Storage<K>>,
    {
        // binds `BackendWithDType<u32>::RawTensor` to be identical to `Self::RawTensor`.
        let prediction = TensorHandle::from_storage::<B, K, Local>(&pred.inner);
        let target_handle = TensorHandle::from_storage::<B, u32, Local>(&target.inner);
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
        .map_err(crate::err::Error::from)?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = pred.dims().into();
            if !out_shape_dims.is_empty() {
                out_shape_dims.remove(1); // Usually class dim
            }
        }
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(
            inner.into(),
            out_shape,
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        )
    }
}

/// Trait to statically verify that two shapes are identical for L1 loss.
pub trait L1Shape<S2: crate::shapes::Shape> {}
impl<S: crate::shapes::Shape> L1Shape<S> for S {}

/// Mean Absolute Error (L1) Loss.
#[derive(Debug, Clone, Default)]
pub struct L1Loss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl<R: ReductionMode> L1Loss<R> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the L1 Loss between predictions and targets.
    pub fn forward<
        S: Shape + crate::shapes::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<op::L1Loss>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, G>,
        target: &Tensor<S, B, K, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: L1ReductionShape<S>,
        <B as Execute<op::L1Loss>>::Output: Into<B::Storage<K>>,
    {
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&pred.inner),
            TensorHandle::from_storage::<B, K, Local>(&target.inner),
        ];
        let reduction = match R::as_enum() {
            Reduction::None => LossReduction::None,
            Reduction::Mean => LossReduction::Mean,
            Reduction::Sum => LossReduction::Sum,
        };
        let context = ExecutionContext::from_scope(B::default());
        let inner =
            dispatch::execute::<op::L1Loss, B>(&context, LossAttributes { reduction }, &inputs)
                .map_err(crate::err::Error::from)?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = pred.dims().into();
        }
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(
            inner.into(),
            out_shape,
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        )
    }
}

/// Trait to statically verify that two shapes are identical for BCEWithLogits loss.
pub trait BCEWithLogitsShape<S2: crate::shapes::Shape> {}
impl<S: crate::shapes::Shape> BCEWithLogitsShape<S> for S {}

/// Binary Cross Entropy with Logits Loss.
#[derive(Debug, Clone, Default)]
pub struct BCEWithLogitsLoss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl<R: ReductionMode> BCEWithLogitsLoss<R> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the BCE With Logits Loss between predictions and targets.
    pub fn forward<
        S: Shape + crate::shapes::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<op::BceWithLogitsLoss>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, G>,
        target: &Tensor<S, B, K, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: BceReductionShape<S>,
        <B as Execute<op::BceWithLogitsLoss>>::Output: Into<B::Storage<K>>,
    {
        let inputs = [
            TensorHandle::from_storage::<B, K, Local>(&pred.inner),
            TensorHandle::from_storage::<B, K, Local>(&target.inner),
        ];
        let reduction = match R::as_enum() {
            Reduction::None => LossReduction::None,
            Reduction::Mean => LossReduction::Mean,
            Reduction::Sum => LossReduction::Sum,
        };
        let context = ExecutionContext::from_scope(B::default());
        let inner = dispatch::execute::<op::BceWithLogitsLoss, B>(
            &context,
            LossAttributes { reduction },
            &inputs,
        )
        .map_err(crate::err::Error::from)?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = pred.dims().into();
        }
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(
            inner.into(),
            out_shape,
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        )
    }
}
