use crate::prelude::*;

/// Specifies the runtime reduction to apply to the output of a loss function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    #[default]
    Mean,
    Sum,
    None,
}

pub trait ReductionMode: Clone + Default + 'static {
    fn as_enum() -> Reduction;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Mean;
impl ReductionMode for Mean {
    fn as_enum() -> Reduction {
        Reduction::Mean
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sum;
impl ReductionMode for Sum {
    fn as_enum() -> Reduction {
        Reduction::Sum
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoneReduction;
impl ReductionMode for NoneReduction {
    fn as_enum() -> Reduction {
        Reduction::None
    }
}

pub trait MseReductionShape<S: Shape> {
    type Output: Shape;
}
impl<S: Shape> MseReductionShape<S> for Mean {
    type Output = ();
}
impl<S: Shape> MseReductionShape<S> for Sum {
    type Output = ();
}
impl<S: Shape> MseReductionShape<S> for NoneReduction {
    type Output = S;
}

pub trait CrossEntropyReductionShape<S: Shape> {
    type Output: Shape;
}
impl<S: Shape> CrossEntropyReductionShape<S> for Mean {
    type Output = ();
}
impl<S: Shape> CrossEntropyReductionShape<S> for Sum {
    type Output = ();
}
impl<S: Shape + crate::shapes::shape_ops::ReduceDim<1>> CrossEntropyReductionShape<S>
    for NoneReduction
{
    type Output = S::Output;
}

pub trait BceReductionShape<S: Shape> {
    type Output: Shape;
}
impl<S: Shape> BceReductionShape<S> for Mean {
    type Output = ();
}
impl<S: Shape> BceReductionShape<S> for Sum {
    type Output = ();
}
impl<S: Shape> BceReductionShape<S> for NoneReduction {
    type Output = S;
}

pub trait L1ReductionShape<S: Shape> {
    type Output: Shape;
}
impl<S: Shape> L1ReductionShape<S> for Mean {
    type Output = ();
}
impl<S: Shape> L1ReductionShape<S> for Sum {
    type Output = ();
}
impl<S: Shape> L1ReductionShape<S> for NoneReduction {
    type Output = S;
}

/// Trait to statically verify that two shapes are identical for MSE loss.
pub trait MSEShape<S2: Shape> {}
impl<S: Shape> MSEShape<S> for S {}

/// Mean Squared Error Loss.
#[derive(Debug, Clone, Default)]
pub struct MSELoss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl<R: ReductionMode> MSELoss<R> {
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the Mean Squared Error between predictions and targets.
    pub fn forward<
        S: Shape + crate::prelude::DynShape,
        B: Backend + crate::tensor::backend::LossOps<B>,
        K: crate::tensor::dtype::DType,
        D: crate::tensor::device::Device,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, D, G>,
        target: &Tensor<S, B, K, D, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, D, G>>
    where
        R: MseReductionShape<S>,
    {
        let inner = B::mse_loss(&pred.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = pred.dims().into();
        }
        let out_shape = <R::Output as Shape>::from_dyn(&out_shape_dims).unwrap();
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        ))
    }
}

/// Trait to statically verify the shapes for CrossEntropyLoss.
/// Ensures the prediction tensor is `[Batch, Classes]` and the target is `[Batch]`.
pub trait CrossEntropyShape<S2: Shape> {}

// Static implementation: [Batch, Classes] vs [Batch]
impl<Batch: Dim, Classes: Dim> CrossEntropyShape<(Batch,)> for (Batch, Classes) {}

// Dynamic fallback
impl CrossEntropyShape<Dyn> for Dyn {}
impl<Batch: Dim, Classes: Dim> CrossEntropyShape<Dyn> for (Batch, Classes) {}
impl<Batch: Dim> CrossEntropyShape<(Batch,)> for Dyn {}

/// Cross Entropy Loss.
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyLoss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl<R: ReductionMode> CrossEntropyLoss<R> {
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the Cross Entropy Loss between predictions and targets.
    /// The target tensor MUST have `u32` elements at compile time.
    pub fn forward<
        S1: Shape + crate::prelude::DynShape,
        S2: Shape,
        B: Backend + crate::tensor::backend::LossOps<B>,
        K: crate::tensor::dtype::DType,
        D: crate::tensor::device::Device,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S1, B, K, D, G>,
        target: &Tensor<S2, B, u32, D, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, D, G>>
    where
        S1: CrossEntropyShape<S2>,
        R: CrossEntropyReductionShape<S1>,
    {
        // binds `BackendWithDType<u32>::RawTensor` to be identical to `Self::RawTensor`.
        let inner = B::cross_entropy_loss(&pred.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = pred.dims().into();
            if !out_shape_dims.is_empty() {
                out_shape_dims.remove(1); // Usually class dim
            }
        }
        let out_shape = <R::Output as Shape>::from_dyn(&out_shape_dims).unwrap();
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        ))
    }
}

/// Trait to statically verify that two shapes are identical for L1 loss.
pub trait L1Shape<S2: crate::prelude::Shape> {}
impl<S: crate::prelude::Shape> L1Shape<S> for S {}

/// Mean Absolute Error (L1) Loss.
#[derive(Debug, Clone, Default)]
pub struct L1Loss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl<R: ReductionMode> L1Loss<R> {
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the L1 Loss between predictions and targets.
    pub fn forward<
        S: Shape + crate::prelude::DynShape,
        B: Backend + crate::tensor::backend::LossOps<B>,
        K: crate::tensor::dtype::DType,
        D: crate::tensor::device::Device,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, D, G>,
        target: &Tensor<S, B, K, D, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, D, G>>
    where
        R: L1ReductionShape<S>,
    {
        let inner = B::l1_loss(&pred.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = pred.dims().into();
        }
        let out_shape = <R::Output as Shape>::from_dyn(&out_shape_dims).unwrap();
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        ))
    }
}

/// Trait to statically verify that two shapes are identical for BCEWithLogits loss.
pub trait BCEWithLogitsShape<S2: crate::prelude::Shape> {}
impl<S: crate::prelude::Shape> BCEWithLogitsShape<S> for S {}

/// Binary Cross Entropy with Logits Loss.
#[derive(Debug, Clone, Default)]
pub struct BCEWithLogitsLoss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl<R: ReductionMode> BCEWithLogitsLoss<R> {
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the BCE With Logits Loss between predictions and targets.
    pub fn forward<
        S: Shape + crate::prelude::DynShape,
        B: Backend + crate::tensor::backend::LossOps<B>,
        K: crate::tensor::dtype::DType,
        D: crate::tensor::device::Device,
        G: RequiresGrad,
    >(
        &self,
        pred: &Tensor<S, B, K, D, G>,
        target: &Tensor<S, B, K, D, NoGrad>,
    ) -> Result<Tensor<R::Output, B, K, D, G>>
    where
        R: BceReductionShape<S>,
    {
        let inner = B::bce_with_logits_loss(&pred.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = pred.dims().into();
        }
        let out_shape = <R::Output as Shape>::from_dyn(&out_shape_dims).unwrap();
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        ))
    }
}
