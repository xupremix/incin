use crate::prelude::*;

/// Trait to statically verify that two shapes are identical for MSE loss.
pub trait MSEShape<S2: Shape> {}
impl<S: Shape> MSEShape<S> for S {}

/// Mean Squared Error Loss.
#[derive(Debug, Clone, Default)]
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }

    /// Forward pass computing the Mean Squared Error between predictions and targets.
    pub fn forward<S: Shape, B: Backend, G: RequiresGrad>(
        &self,
        pred: &Tensor<S, B, G>,
        target: &Tensor<S, B, NoGrad>,
    ) -> Result<Tensor<(), B, G>> {
        let inner = B::mse_loss(&pred.inner, &target.inner)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            Default::default(),
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        ))
    }
}

/// Trait to statically verify the shapes for CrossEntropyLoss.
/// Ensures the prediction tensor is [Batch, Classes] and the target is [Batch].
pub trait CrossEntropyShape<S2: Shape> {}

// Static implementation: [Batch, Classes] vs [Batch]
impl<Batch: Dim, Classes: Dim> CrossEntropyShape<(Batch,)> for (Batch, Classes) {}

// Dynamic fallback
impl CrossEntropyShape<Dyn> for Dyn {}
impl<Batch: Dim, Classes: Dim> CrossEntropyShape<Dyn> for (Batch, Classes) {}
impl<Batch: Dim> CrossEntropyShape<(Batch,)> for Dyn {}

/// Cross Entropy Loss.
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }

    /// Forward pass computing the Cross Entropy Loss between predictions and targets.
    /// The target tensor MUST have `u32` elements at compile time.
    pub fn forward<S1: Shape, S2: Shape, B: Backend, G: RequiresGrad>(
        &self,
        pred: &Tensor<S1, B, G>,
        target: &Tensor<S2, B::BackendWithDType<u32>, NoGrad>,
    ) -> Result<Tensor<(), B, G>>
    where
        S1: CrossEntropyShape<S2>,
    {
        // We can pass `target.inner` directly here because the `Backend` trait explicitly
        // binds `BackendWithDType<u32>::RawTensor` to be identical to `Self::RawTensor`.
        let inner = B::cross_entropy_loss(&pred.inner, &target.inner)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            Default::default(),
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
pub struct L1Loss;

impl L1Loss {
    pub fn new() -> Self {
        Self
    }

    /// Forward pass computing the L1 Loss between predictions and targets.
    pub fn forward<S: crate::prelude::Shape, B: crate::prelude::Backend, G: crate::prelude::RequiresGrad>(
        &self,
        pred: &crate::prelude::Tensor<S, B, G>,
        target: &crate::prelude::Tensor<S, B, crate::prelude::NoGrad>,
    ) -> crate::prelude::Result<crate::prelude::Tensor<(), B, G>> {
        let inner = B::l1_loss(&pred.inner, &target.inner)?;
        Ok(crate::prelude::Tensor::from_parts_unchecked(
            inner,
            Default::default(),
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
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        Self
    }

    /// Forward pass computing the BCE With Logits Loss between predictions and targets.
    pub fn forward<S: crate::prelude::Shape, B: crate::prelude::Backend, G: crate::prelude::RequiresGrad>(
        &self,
        pred: &crate::prelude::Tensor<S, B, G>,
        target: &crate::prelude::Tensor<S, B, crate::prelude::NoGrad>,
    ) -> crate::prelude::Result<crate::prelude::Tensor<(), B, G>> {
        let inner = B::bce_with_logits_loss(&pred.inner, &target.inner)?;
        Ok(crate::prelude::Tensor::from_parts_unchecked(
            inner,
            Default::default(),
            pred._dtype.clone(),
            pred._device.clone(),
            pred._grad.clone(),
        ))
    }
}
