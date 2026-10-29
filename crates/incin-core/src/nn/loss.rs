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
use crate::tensor::grad::{GradJoin, NoGrad, RequiresGrad};
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

impl MSELoss<Mean> {
    /// Mean reduction, the same default `torch.nn.MSELoss()` uses.
    ///
    /// A type-parameter default does not drive inference for an associated
    /// function, so a single generic `new` would force `MSELoss::<Mean>::new()`
    /// at every call site. This concrete constructor is what lets
    /// `MSELoss::new()` resolve on its own. Use
    /// [`with_reduction`](MSELoss::with_reduction) to choose another reduction.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<R: ReductionMode> MSELoss<R> {
    /// Selects the reduction explicitly, as in
    /// `MSELoss::<Sum>::with_reduction()`.
    pub fn with_reduction() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the Mean Squared Error between predictions and targets.
    pub fn forward<
        S: Shape + crate::shapes::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<op::MseLoss>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
        L1: crate::shapes::Layout<S>,
        L2: crate::shapes::Layout<S>,
    >(
        &self,
        pred: &Tensor<S, B, K, G, Local, L1>,
        target: &Tensor<S, B, K, NoGrad, Local, L2>,
    ) -> Result<crate::shapes::Dense<R::Output, B, K, G, Local>>
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

impl CrossEntropyLoss<Mean> {
    /// Mean reduction, the same default `torch.nn.CrossEntropyLoss()` uses.
    ///
    /// A type-parameter default does not drive inference for an associated
    /// function, so a single generic `new` would force `CrossEntropyLoss::<Mean>::new()`
    /// at every call site. This concrete constructor is what lets
    /// `CrossEntropyLoss::new()` resolve on its own. Use
    /// [`with_reduction`](CrossEntropyLoss::with_reduction) to choose another reduction.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<R: ReductionMode> CrossEntropyLoss<R> {
    /// Selects the reduction explicitly, as in
    /// `CrossEntropyLoss::<Sum>::with_reduction()`.
    pub fn with_reduction() -> Self {
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
        L1: crate::shapes::Layout<S1>,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        pred: &Tensor<S1, B, K, G, Local, L1>,
        target: &Tensor<S2, B, u32, NoGrad, Local, L2>,
    ) -> Result<crate::shapes::Dense<R::Output, B, K, G, Local>>
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

impl L1Loss<Mean> {
    /// Mean reduction, the same default `torch.nn.L1Loss()` uses.
    ///
    /// A type-parameter default does not drive inference for an associated
    /// function, so a single generic `new` would force `L1Loss::<Mean>::new()`
    /// at every call site. This concrete constructor is what lets
    /// `L1Loss::new()` resolve on its own. Use
    /// [`with_reduction`](L1Loss::with_reduction) to choose another reduction.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<R: ReductionMode> L1Loss<R> {
    /// Selects the reduction explicitly, as in
    /// `L1Loss::<Sum>::with_reduction()`.
    pub fn with_reduction() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the L1 Loss between predictions and targets.
    pub fn forward<
        S: Shape + crate::shapes::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<op::L1Loss>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
        L1: crate::shapes::Layout<S>,
        L2: crate::shapes::Layout<S>,
    >(
        &self,
        pred: &Tensor<S, B, K, G, Local, L1>,
        target: &Tensor<S, B, K, NoGrad, Local, L2>,
    ) -> Result<crate::shapes::Dense<R::Output, B, K, G, Local>>
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

impl BCEWithLogitsLoss<Mean> {
    /// Mean reduction, the same default `torch.nn.BCEWithLogitsLoss()` uses.
    ///
    /// A type-parameter default does not drive inference for an associated
    /// function, so a single generic `new` would force `BCEWithLogitsLoss::<Mean>::new()`
    /// at every call site. This concrete constructor is what lets
    /// `BCEWithLogitsLoss::new()` resolve on its own. Use
    /// [`with_reduction`](BCEWithLogitsLoss::with_reduction) to choose another reduction.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<R: ReductionMode> BCEWithLogitsLoss<R> {
    /// Selects the reduction explicitly, as in
    /// `BCEWithLogitsLoss::<Sum>::with_reduction()`.
    pub fn with_reduction() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the BCE With Logits Loss between predictions and targets.
    pub fn forward<
        S: Shape + crate::shapes::DynShape,
        B: Backend + crate::exec::Capabilities + Execute<op::BceWithLogitsLoss>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
        L1: crate::shapes::Layout<S>,
        L2: crate::shapes::Layout<S>,
    >(
        &self,
        pred: &Tensor<S, B, K, G, Local, L1>,
        target: &Tensor<S, B, K, NoGrad, Local, L2>,
    ) -> Result<crate::shapes::Dense<R::Output, B, K, G, Local>>
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

/// Smooth L1 (Huber) Loss, matching `torch.nn.SmoothL1Loss`.
///
/// Unlike the losses above this has no catalog op behind it: there is no
/// `SmoothL1Loss` descriptor to dispatch, so the forward pass delegates to the
/// composed tensor implementation (`sub`/`abs`/`mul`/`where_cond`/...), which
/// threads gradients through each primitive's own tape entry. `beta` is the
/// transition point between the quadratic and linear regions (default `1.0`).
#[derive(Debug, Clone)]
pub struct SmoothL1Loss<R: ReductionMode = Mean> {
    /// Transition point between the quadratic (`|x| < beta`) and linear
    /// (`|x| >= beta`) regions. Must be finite and positive.
    pub beta: f64,
    _reduction: core::marker::PhantomData<R>,
}

impl SmoothL1Loss<Mean> {
    /// Mean reduction with `beta = 1.0`, the `torch.nn.SmoothL1Loss()` default.
    pub fn new() -> Self {
        Self {
            beta: 1.0,
            _reduction: core::marker::PhantomData,
        }
    }
}

impl Default for SmoothL1Loss<Mean> {
    /// Default is [`SmoothL1Loss::new`]: mean reduction, `beta = 1.0`.
    fn default() -> Self {
        Self::new()
    }
}

impl<R: ReductionMode> SmoothL1Loss<R> {
    /// Selects the reduction explicitly, as in
    /// `SmoothL1Loss::<Sum>::with_reduction()`. Keeps `beta = 1.0`.
    pub fn with_reduction() -> Self {
        Self {
            beta: 1.0,
            _reduction: core::marker::PhantomData,
        }
    }

    /// Sets the Huber transition point, as in
    /// `SmoothL1Loss::with_beta(0.5)`.
    pub fn with_beta(beta: f64) -> Self {
        Self {
            beta,
            _reduction: core::marker::PhantomData,
        }
    }

    /// Forward pass computing the Smooth L1 loss between predictions and targets.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::nn::loss::SmoothL1Loss;
    /// use incin::prelude::*;
    /// let loss_fn = SmoothL1Loss::new();
    /// let pred = Cpu.tensor([0.0f32, 0.5, 1.0, 2.0]).unwrap();
    /// let target = Cpu.zeros(shape![4]).unwrap();
    /// // diffs [0, 0.5, 1, 2] with beta=1 give [0, 0.125, 0.5, 1.5].
    /// let loss = loss_fn.forward(&pred, &target).unwrap();
    /// assert!((loss.to_scalar::<f32>().unwrap() - 0.53125).abs() < 1e-6);
    /// ```
    pub fn forward<
        S: Shape + crate::shapes::DynShape,
        B: Backend
            + crate::exec::Capabilities
            + Execute<op::Sub>
            + Execute<op::Abs>
            + Execute<op::Mul>
            + Execute<op::MulScalar>
            + Execute<op::AddScalar>
            + Execute<op::SubScalar>
            + Execute<op::CmpLt>
            + Execute<op::WhereCond>
            + Execute<op::MeanAll>
            + Execute<op::SumAll>,
        K: crate::tensor::dtype::DType + crate::tensor::ops::quantization::FloatCapable,
        G: RequiresGrad + GradJoin<NoGrad, Output = G> + GradJoin<G, Output = G>,
        L1: crate::shapes::Layout<S> + crate::shapes::Restatable + crate::shapes::Layout<Dyn>,
        L2: crate::shapes::Layout<S> + crate::shapes::Restatable + crate::shapes::Layout<Dyn>,
    >(
        &self,
        pred: &Tensor<S, B, K, G, Local, L1>,
        target: &Tensor<S, B, K, NoGrad, Local, L2>,
    ) -> Result<crate::shapes::Dense<R::Output, B, K, G, Local>>
    where
        R: MseReductionShape<S>,
        <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Abs>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::AddScalar>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SubScalar>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::CmpLt>>::Output: Into<B::Storage<bool>>,
        <B as Execute<op::WhereCond>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::MeanAll>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SumAll>>::Output: Into<B::Storage<K>>,
    {
        pred.smooth_l1_loss_with::<R, S, NoGrad, L2>(target, self.beta)
    }
}

/// Negative Log-Likelihood Loss over log-probabilities.
///
/// `NLLLoss` is cross-entropy's second half: the input already holds
/// log-probabilities (e.g. from [`crate::nn::activation::LogSoftmax`]), so
/// each element is `-input[row, target[row]]`. Like [`SmoothL1Loss`] it has
/// no catalog op and delegates to the composed tensor implementation
/// (`unsqueeze`/`gather`/`neg` plus reductions), sharing cross-entropy's
/// reduction-shape rule.
#[derive(Debug, Clone, Default)]
pub struct NLLLoss<R: ReductionMode = Mean>(core::marker::PhantomData<R>);

impl NLLLoss<Mean> {
    /// Mean reduction, the same default `torch.nn.NLLLoss()` uses.
    pub fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<R: ReductionMode> NLLLoss<R> {
    /// Selects the reduction explicitly, as in
    /// `NLLLoss::<Sum>::with_reduction()`.
    pub fn with_reduction() -> Self {
        Self(core::marker::PhantomData)
    }

    /// Forward pass computing the NLL loss between log-probabilities and class targets.
    /// The target tensor MUST have `u32` elements at compile time.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::nn::loss::NLLLoss;
    /// use incin::prelude::*;
    /// let loss_fn = NLLLoss::new();
    /// let log_probs = Cpu.tensor([
    ///     [0.2f32.ln(), 0.3f32.ln(), 0.5f32.ln()],
    ///     [0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()],
    /// ])
    /// .unwrap();
    /// let target = Cpu.tensor([2u32, 0]).unwrap();
    /// // (-ln(0.5) + -ln(0.6)) / 2.
    /// let loss = loss_fn.forward(&log_probs, &target).unwrap();
    /// let expected = (-0.5f32.ln() + -0.6f32.ln()) / 2.0;
    /// assert!((loss.to_scalar::<f32>().unwrap() - expected).abs() < 1e-5);
    /// ```
    pub fn forward<
        S1,
        S2: Shape,
        B: Backend
            + crate::exec::Capabilities
            + Execute<op::UnsqueezeExact>
            + Execute<op::Gather>
            + Execute<op::ReshapeExact>
            + Execute<op::Neg>
            + Execute<op::MeanAll>
            + Execute<op::SumAll>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
        L1: crate::shapes::Layout<S1>,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        log_probs: &Tensor<S1, B, K, G, Local, L1>,
        target: &Tensor<S2, B, u32, NoGrad, Local, L2>,
    ) -> Result<crate::shapes::Dense<R::Output, B, K, G, Local>>
    where
        S1: Shape + crate::shapes::DynShape,
        S2: Shape + crate::shapes::DynShape,
        R: CrossEntropyReductionShape<S1>,
        L1: crate::shapes::Restatable + crate::shapes::Layout<Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<Dyn>,
        <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<u32>>,
        <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Neg>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::MeanAll>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SumAll>>::Output: Into<B::Storage<K>>,
    {
        log_probs.nll_loss_with::<R, S2, u32, NoGrad, L2>(target)
    }
}

impl<R: ReductionMode> CrossEntropyLoss<R> {
    /// Token-level forward over `[B, T, C]` logits and `[B, T]` targets.
    ///
    /// The rank-2 [`CrossEntropyShape`] trait (and the catalog descriptor
    /// behind it) only admits `[B, C]` logits, so this entry point flattens
    /// both operands to `[B*T, C]` / `[B*T]`, runs the rank-2 core, and (for
    /// `None`) restores the `[B, T]` geometry. There is no `ignore_index`
    /// convention to carry: the existing cross-entropy has none, so the 3D
    /// path matches it exactly. The output is dynamically shaped: a scalar
    /// for `Mean`/`Sum`, `[B, T]` for `None`.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::nn::loss::CrossEntropyLoss;
    /// use incin::prelude::*;
    /// let loss_fn = CrossEntropyLoss::new();
    /// let logits = Cpu.tensor([[[1.0f32, 2.0], [0.5, -0.5]]]).unwrap();
    /// let targets = Cpu.tensor([[1u32, 0]]).unwrap();
    /// // Token-CE equals the flattened rank-2 CE.
    /// let token = loss_fn.forward_token(&logits, &targets).unwrap();
    /// let flat_logits = logits.reshape(shape![2, 2]).unwrap();
    /// let flat_targets = targets.reshape(shape![2]).unwrap();
    /// let flat = loss_fn.forward(&flat_logits, &flat_targets).unwrap();
    /// assert!((token.to_scalar::<f32>().unwrap() - flat.to_scalar::<f32>().unwrap()).abs() < 1e-6);
    /// ```
    pub fn forward_token<
        S1,
        S2: Shape,
        B: Backend
            + crate::exec::Capabilities
            + Execute<op::CrossEntropyLoss>
            + Execute<op::ReshapeExact>,
        K: crate::tensor::dtype::DType,
        G: RequiresGrad,
        L1: crate::shapes::Layout<S1>,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        logits: &Tensor<S1, B, K, G, Local, L1>,
        target: &Tensor<S2, B, u32, NoGrad, Local, L2>,
    ) -> Result<crate::shapes::Dense<Dyn, B, K, G, Local>>
    where
        S1: Shape + crate::shapes::DynShape,
        S2: Shape + crate::shapes::DynShape,
        L1: crate::shapes::Restatable + crate::shapes::Layout<Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<Dyn>,
        <B as Execute<op::CrossEntropyLoss>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>> + Into<B::Storage<u32>>,
    {
        logits.cross_entropy_token_loss_with::<R, S2, u32, NoGrad, L2>(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::reduction::Sum;

    #[test]
    fn smooth_l1_defaults_to_beta_one() {
        let loss = SmoothL1Loss::new();
        assert_eq!(loss.beta, 1.0);
        assert_eq!(SmoothL1Loss::<Mean>::with_reduction().beta, 1.0);
        assert_eq!(SmoothL1Loss::<Sum>::with_reduction().beta, 1.0);
        assert_eq!(SmoothL1Loss::default().beta, 1.0);
    }

    #[test]
    fn smooth_l1_with_beta_stores_beta_for_any_reduction() {
        assert_eq!(SmoothL1Loss::<Mean>::with_beta(0.5).beta, 0.5);
        assert_eq!(SmoothL1Loss::<Sum>::with_beta(2.0).beta, 2.0);
        assert_eq!(SmoothL1Loss::<NoneReduction>::with_beta(0.25).beta, 0.25);
    }

    #[test]
    fn nll_constructors_resolve_for_all_reductions() {
        let _ = NLLLoss::new();
        let _ = NLLLoss::<Sum>::with_reduction();
        let _ = NLLLoss::<NoneReduction>::with_reduction();
        let _ = NLLLoss::<Mean>::default();
    }

    #[test]
    fn smooth_l1_is_clone_and_debug() {
        let loss = SmoothL1Loss::<Mean>::with_beta(0.75);
        let cloned = loss.clone();
        assert_eq!(cloned.beta, 0.75);
        assert!(alloc::format!("{loss:?}").contains("0.75"));
    }
}
