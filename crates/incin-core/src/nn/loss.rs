use crate::prelude::*;
use crate::dist::placement::Local;
use crate::exec::catalog::{Descriptor, LossAttributes, LossReduction, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::tensor::backend::Execute;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;

/// Specifies the runtime reduction to apply to the output of a loss function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    #[default]
    /// `Mean`.
    Mean,
    /// `Sum`.
    Sum,
    /// `None`.
    None,
}

/// `ReductionMode`.
pub trait ReductionMode: Clone + Default + 'static {
    /// `as_enum`.
    fn as_enum() -> Reduction;
}

#[derive(Debug, Clone, Copy, Default)]
/// `Mean`.
pub struct Mean;
impl ReductionMode for Mean {
    /// `as_enum`.
    fn as_enum() -> Reduction {
        Reduction::Mean
    }
}

#[derive(Debug, Clone, Copy, Default)]
/// `Sum`.
pub struct Sum;
impl ReductionMode for Sum {
    /// `as_enum`.
    fn as_enum() -> Reduction {
        Reduction::Sum
    }
}

#[derive(Debug, Clone, Copy, Default)]
/// `NoneReduction`.
pub struct NoneReduction;
impl ReductionMode for NoneReduction {
    /// `as_enum`.
    fn as_enum() -> Reduction {
        Reduction::None
    }
}

/// `MseReductionShape`.
pub trait MseReductionShape<S: Shape> {
    /// The output tensor type produced by this module's forward pass.
    type Output: Shape;
}
impl<S: Shape> MseReductionShape<S> for Mean {
    /// The output tensor type produced by this module's forward pass.
    type Output = ();
}
impl<S: Shape> MseReductionShape<S> for Sum {
    /// The output tensor type produced by this module's forward pass.
    type Output = ();
}
impl<S: Shape> MseReductionShape<S> for NoneReduction {
    /// The output tensor type produced by this module's forward pass.
    type Output = S;
}

/// `CrossEntropyReductionShape`.
pub trait CrossEntropyReductionShape<S: Shape> {
    /// The output tensor type produced by this module's forward pass.
    type Output: Shape;
}
impl<S: Shape> CrossEntropyReductionShape<S> for Mean {
    /// The output tensor type produced by this module's forward pass.
    type Output = ();
}
impl<S: Shape> CrossEntropyReductionShape<S> for Sum {
    /// The output tensor type produced by this module's forward pass.
    type Output = ();
}
impl<
    S: Shape + crate::shapes::shape_ops::ReduceAt<crate::shapes::idx::Next<crate::shapes::idx::Here>>,
> CrossEntropyReductionShape<S> for NoneReduction
{
    /// The output tensor type produced by this module's forward pass.
    type Output = <S as crate::shapes::shape_ops::ReduceAt<
        crate::shapes::idx::Next<crate::shapes::idx::Here>,
    >>::Output;
}

/// `BceReductionShape`.
pub trait BceReductionShape<S: Shape> {
    /// The output tensor type produced by this module's forward pass.
    type Output: Shape;
}
impl<S: Shape> BceReductionShape<S> for Mean {
    /// The output tensor type produced by this module's forward pass.
    type Output = ();
}
impl<S: Shape> BceReductionShape<S> for Sum {
    /// The output tensor type produced by this module's forward pass.
    type Output = ();
}
impl<S: Shape> BceReductionShape<S> for NoneReduction {
    /// The output tensor type produced by this module's forward pass.
    type Output = S;
}

/// `L1ReductionShape`.
pub trait L1ReductionShape<S: Shape> {
    /// The output tensor type produced by this module's forward pass.
    type Output: Shape;
}
impl<S: Shape> L1ReductionShape<S> for Mean {
    /// The output tensor type produced by this module's forward pass.
    type Output = ();
}
impl<S: Shape> L1ReductionShape<S> for Sum {
    /// The output tensor type produced by this module's forward pass.
    type Output = ();
}
impl<S: Shape> L1ReductionShape<S> for NoneReduction {
    /// The output tensor type produced by this module's forward pass.
    type Output = S;
}

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
        S: Shape + crate::prelude::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<Descriptor<op::MseLoss>>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, G>,
        target: &Tensor<S, B, K, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: MseReductionShape<S>,
        <B as Execute<Descriptor<op::MseLoss>>>::Output: Into<B::Storage<K>>,
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
        let inner = dispatch::execute::<op::MseLoss, B>(
            &context,
            LossAttributes { reduction },
            &inputs,
        )
        .map_err(crate::prelude::Error::from)?;
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
        B: Backend + crate::exec::Capabilities + Execute<Descriptor<op::CrossEntropyLoss>>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S1, B, K, G>,
        target: &Tensor<S2, B, u32, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        S1: Shape + crate::prelude::DynShape + CrossEntropyShape<S2>,
        R: CrossEntropyReductionShape<S1>,
        <B as Execute<Descriptor<op::CrossEntropyLoss>>>::Output: Into<B::Storage<K>>,
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
        .map_err(crate::prelude::Error::from)?;
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
pub trait L1Shape<S2: crate::prelude::Shape> {}
impl<S: crate::prelude::Shape> L1Shape<S> for S {}

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
        S: Shape + crate::prelude::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<Descriptor<op::L1Loss>>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, G>,
        target: &Tensor<S, B, K, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: L1ReductionShape<S>,
        <B as Execute<Descriptor<op::L1Loss>>>::Output: Into<B::Storage<K>>,
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
        let inner = dispatch::execute::<op::L1Loss, B>(
            &context,
            LossAttributes { reduction },
            &inputs,
        )
        .map_err(crate::prelude::Error::from)?;
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
pub trait BCEWithLogitsShape<S2: crate::prelude::Shape> {}
impl<S: crate::prelude::Shape> BCEWithLogitsShape<S> for S {}

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
        S: Shape + crate::prelude::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<Descriptor<op::BceWithLogitsLoss>>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, G>,
        target: &Tensor<S, B, K, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, G>>
    where
        R: BceReductionShape<S>,
        <B as Execute<Descriptor<op::BceWithLogitsLoss>>>::Output: Into<B::Storage<K>>,
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
        .map_err(crate::prelude::Error::from)?;
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
