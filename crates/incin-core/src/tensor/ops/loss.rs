//! Common loss functions (MSE, L1, BCE, CrossEntropy) for training.
//!
//! This module provides standard loss functions used to train neural networks.
//! Loss functions automatically compute and track their required reduction shape
//! (e.g. reducing down to a scalar or maintaining a batched shape) using type-level
//! logic to ensure that backpropagation can flow correctly from the scalar loss.
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::catalog::{LossAttributes, LossReduction, ShapeAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::Shape;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;
use crate::tensor::backend::Backend;
use crate::tensor::backend::Execute;
use crate::tensor::base::Tensor;
use crate::tensor::grad::{GradJoin, RequiresGrad};
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

    /// Per-element Smooth L1 (Huber) loss, composed from existing ops.
    ///
    /// With `d = pred - target`, each element is `0.5 * d^2 / beta` when
    /// `|d| < beta` and `|d| - 0.5 * beta` otherwise. The strict `<` in the
    /// branch mask matches the CPU convention of the sibling losses: the kink
    /// at `|d| == beta` takes the linear branch, so its subgradient is
    /// `sign(d)` exactly like `torch.nn.SmoothL1Loss`. There is deliberately
    /// no catalog op behind this: `sub`/`abs`/`mul`/`mul_scalar`/`add_scalar`/
    /// `sub_scalar`/`cmp_lt`/`where_cond` each push their own tape entry, so
    /// the backward pass is correct by composition.
    fn smooth_l1_per_element<
        S2: Shape + crate::shapes::DynShape,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
        beta: f64,
    ) -> Result<crate::shapes::Dense<crate::shapes::Dyn, B, K, G, Local>>
    where
        L: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        K: crate::tensor::dtype::DType + crate::tensor::ops::quantization::FloatCapable,
        G: GradJoin<G2, Output = G> + GradJoin<G, Output = G>,
        B: Execute<op::Sub>
            + Execute<op::Abs>
            + Execute<op::Mul>
            + Execute<op::MulScalar>
            + Execute<op::AddScalar>
            + Execute<op::SubScalar>
            + Execute<op::CmpLt>
            + Execute<op::WhereCond>,
        <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Abs>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::AddScalar>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SubScalar>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::CmpLt>>::Output: Into<B::Storage<bool>>,
        <B as Execute<op::WhereCond>>::Output: Into<B::Storage<K>>,
    {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(crate::err::Error::ShapeMismatch {
                op: "smooth_l1_loss",
                expected: alloc::vec![],
                got: alloc::vec![],
                msg: alloc::format!("smooth_l1_loss beta must be finite and positive, got {beta}"),
            });
        }
        let pred = self.clone().into_dyn();
        let targ = target.clone().into_dyn();
        let diff = pred.sub_exact(&targ)?;
        let abs_diff = diff.abs()?;
        // Beta-filled threshold sharing the working shape. It feeds only the
        // grad-disabled comparison below, so its grad lineage is inert.
        let threshold = diff.mul_scalar(0.0)?.add_scalar(beta)?;
        let is_quadratic = abs_diff.lt(&threshold)?;
        let quad = diff.mul_exact(&diff)?.mul_scalar(0.5 / beta)?;
        let linear = abs_diff.sub_scalar(0.5 * beta)?;
        is_quadratic.where_cond(&quad, &linear)
    }

    /// Computes the Smooth L1 (Huber) loss between predictions and targets.
    /// Uses the default `Mean` reduction.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let pred = Cpu.tensor([0.0f32, 0.5, 1.0, 2.0]).unwrap();
    /// let target = Cpu.zeros(shape![4]).unwrap();
    /// // diffs [0, 0.5, 1, 2] with beta=1 give [0, 0.125, 0.5, 1.5].
    /// let loss = pred.smooth_l1_loss(&target, 1.0).unwrap();
    /// assert!((loss.to_scalar::<f32>().unwrap() - 0.53125).abs() < 1e-6);
    /// ```
    pub fn smooth_l1_loss<
        S2: Shape + crate::shapes::DynShape,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
        beta: f64,
    ) -> Result<Tensor<crate::shapes::Nil, B, K, G, Local, crate::shapes::RowMajor>>
    where
        L: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        K: crate::tensor::dtype::DType + crate::tensor::ops::quantization::FloatCapable,
        G: GradJoin<G2, Output = G> + GradJoin<G, Output = G>,
        B: Execute<op::Sub>
            + Execute<op::Abs>
            + Execute<op::Mul>
            + Execute<op::MulScalar>
            + Execute<op::AddScalar>
            + Execute<op::SubScalar>
            + Execute<op::CmpLt>
            + Execute<op::WhereCond>
            + Execute<op::MeanAll>
            + Execute<op::SumAll>
            + crate::exec::Capabilities,
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
        self.smooth_l1_loss_with::<Mean, S2, G2, L2>(target, beta)
    }

    /// `smooth_l1_loss_with` picks the reduction at the type level.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// use incin::tensor::reduction::Sum;
    /// # fn main() -> Result<()> {
    /// let pred = Cpu.tensor([0.0f32, 0.5, 1.0, 2.0])?.require_grad();
    /// let target = Cpu.zeros(shape![4])?;
    /// // Unreduced: one Huber loss per element.
    /// let per = pred.smooth_l1_loss_with::<NoneReduction, _, _, _>(&target, 1.0)?;
    /// assert_eq!(per.to_vec1::<f32>()?, vec![0.0, 0.125, 0.5, 1.5]);
    /// // The kink at |x| == beta takes the linear branch: d/dx is sign(x).
    /// let kink = Cpu.tensor([1.0f32, -1.0])?.require_grad();
    /// let zero = Cpu.zeros(shape![2])?;
    /// let loss = kink.smooth_l1_loss_with::<Sum, _, _, _>(&zero, 1.0)?;
    /// assert!((loss.to_scalar::<f32>()? - 1.0).abs() < 1e-6);
    /// let grads = loss.backward()?;
    /// assert_eq!(grads.require(&kink)?.to_vec1::<f32>()?, vec![1.0, -1.0]);
    /// # Ok(()) }
    /// ```
    pub fn smooth_l1_loss_with<
        R,
        S2: Shape + crate::shapes::DynShape,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, K, G2, Local, L2>,
        beta: f64,
    ) -> Result<Tensor<R::Output, B, K, G, Local, crate::shapes::RowMajor>>
    where
        R: ReductionMode + MseReductionShape<S>,
        L: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        K: crate::tensor::dtype::DType + crate::tensor::ops::quantization::FloatCapable,
        G: GradJoin<G2, Output = G> + GradJoin<G, Output = G>,
        B: Execute<op::Sub>
            + Execute<op::Abs>
            + Execute<op::Mul>
            + Execute<op::MulScalar>
            + Execute<op::AddScalar>
            + Execute<op::SubScalar>
            + Execute<op::CmpLt>
            + Execute<op::WhereCond>
            + Execute<op::MeanAll>
            + Execute<op::SumAll>
            + crate::exec::Capabilities,
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
        let per = self.smooth_l1_per_element(target, beta)?;
        // The joined grad field is `G` by the `GradJoin` bounds, but its
        // *value* comes from the composition (a `Dyn`-grad join can differ
        // from the prediction's own flag), so it is carried explicitly.
        let dtype = per._dtype.clone();
        let device = per._device.clone();
        let grad = per._grad.clone();
        let mut out_shape_dims: Vec<usize> = vec![];
        let storage = match R::as_enum() {
            Reduction::None => {
                out_shape_dims = per.shape_buf().as_ref().to_vec();
                per.inner
            }
            Reduction::Mean => per.mean_all()?.inner,
            Reduction::Sum => per.sum_all()?.inner,
        };
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(storage, out_shape, dtype, device, grad)
    }

    /// Per-element negative log-likelihood, composed from existing ops.
    ///
    /// The input holds log-probabilities (e.g. from `log_softmax`); each
    /// element is `-log_prob[row, target[row]]`. Gathered with `gather` along
    /// the class axis and negated, mirroring the second half of
    /// `cross_entropy_loss_storage`: like cross-entropy, the target side never
    /// receives a gradient. There is deliberately no catalog op behind this.
    fn nll_per_element<
        S2: Shape + crate::shapes::DynShape,
        KT: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, KT, G2, Local, L2>,
    ) -> Result<crate::shapes::Dense<crate::shapes::Dyn, B, K, G, Local>>
    where
        L: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        B: Execute<op::UnsqueezeExact>
            + Execute<op::Gather>
            + Execute<op::ReshapeExact>
            + Execute<op::Neg>
            + crate::exec::Capabilities,
        <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<KT>>,
        <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Neg>>::Output: Into<B::Storage<K>>,
    {
        let log_probs = self.clone().into_dyn();
        let classes = target.clone().into_dyn();
        // `[B]` class ids become the `[B, 1]` index the class-axis gather needs.
        let index = classes.unsqueeze(1isize)?;
        let picked = log_probs.gather(1isize, &index)?;
        // `[B, 1]` back to `[B]`, then negate: NLL is `-log p[target]`.
        let flat =
            picked.reshape_infer(crate::shapes::InferShape::<crate::shapes::Dyn>::new(vec![
                None,
            ]))?;
        flat.neg()
    }

    /// Computes the negative log-likelihood loss from log-probabilities.
    /// Uses the default `Mean` reduction.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let log_probs = Cpu.tensor([[0.2f32.ln(), 0.3f32.ln(), 0.5f32.ln()]]).unwrap();
    /// let target = Cpu.tensor([2u32]).unwrap();
    /// // -ln(0.5) for the single row.
    /// let loss = log_probs.nll_loss(&target).unwrap();
    /// assert!((loss.to_scalar::<f32>().unwrap() - -0.5f32.ln()).abs() < 1e-5);
    /// ```
    pub fn nll_loss<
        S2: Shape + crate::shapes::DynShape,
        KT: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, KT, G2, Local, L2>,
    ) -> Result<Tensor<crate::shapes::Nil, B, K, G, Local, crate::shapes::RowMajor>>
    where
        L: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        B: Execute<op::UnsqueezeExact>
            + Execute<op::Gather>
            + Execute<op::ReshapeExact>
            + Execute<op::Neg>
            + Execute<op::MeanAll>
            + Execute<op::SumAll>
            + crate::exec::Capabilities,
        <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<KT>>,
        <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Neg>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::MeanAll>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SumAll>>::Output: Into<B::Storage<K>>,
    {
        self.nll_loss_with::<Mean, S2, KT, G2, L2>(target)
    }

    /// `nll_loss_with` picks the reduction at the type level. NLL shares
    /// cross-entropy's reduction-shape rule (`None` keeps one loss per row),
    /// since it is exactly cross-entropy's second half.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let log_probs = Cpu.tensor([
    ///     [0.2f32.ln(), 0.3f32.ln(), 0.5f32.ln()],
    ///     [0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()],
    /// ])
    /// .unwrap();
    /// let target = Cpu.tensor([2u32, 0]).unwrap();
    /// // Unreduced: [-ln(0.5), -ln(0.6)], one loss per row.
    /// let per = log_probs.nll_loss_with::<NoneReduction, _, _, _, _>(&target).unwrap();
    /// let vals = per.to_vec1::<f32>().unwrap();
    /// assert!((vals[0] - -0.5f32.ln()).abs() < 1e-5);
    /// assert!((vals[1] - -0.6f32.ln()).abs() < 1e-5);
    /// ```
    pub fn nll_loss_with<
        R,
        S2: Shape + crate::shapes::DynShape,
        KT: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, KT, G2, Local, L2>,
    ) -> Result<Tensor<R::Output, B, K, G, Local, crate::shapes::RowMajor>>
    where
        R: ReductionMode + CrossEntropyReductionShape<S>,
        L: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        B: Execute<op::UnsqueezeExact>
            + Execute<op::Gather>
            + Execute<op::ReshapeExact>
            + Execute<op::Neg>
            + Execute<op::MeanAll>
            + Execute<op::SumAll>
            + crate::exec::Capabilities,
        <B as Execute<op::UnsqueezeExact>>::Output: Into<B::Storage<KT>>,
        <B as Execute<op::Gather>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Neg>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::MeanAll>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::SumAll>>::Output: Into<B::Storage<K>>,
    {
        let per = self.nll_per_element(target)?;
        let dtype = per._dtype.clone();
        let device = per._device.clone();
        let grad = per._grad.clone();
        let mut out_shape_dims: Vec<usize> = vec![];
        let storage = match R::as_enum() {
            Reduction::None => {
                out_shape_dims = per.shape_buf().as_ref().to_vec();
                per.inner
            }
            Reduction::Mean => per.mean_all()?.inner,
            Reduction::Sum => per.sum_all()?.inner,
        };
        let out_shape =
            shape_buf_from_dims::<R::Output>(OperationKind::Reduction, &out_shape_dims)?;
        Tensor::from_parts(storage, out_shape, dtype, device, grad)
    }

    /// Token-level cross-entropy over `[B, T, C]` logits and `[B, T]` targets.
    /// Uses the default `Mean` reduction.
    ///
    /// The rank-2 cross-entropy descriptor only admits `[B, C]` logits, so
    /// this flattens both operands over the leading axes to `[B*T, C]` /
    /// `[B*T]`, runs the rank-2 core, and (for `None`) restores the `[B, T]`
    /// geometry. There is no `ignore_index` convention to carry: the existing
    /// cross-entropy has none, so the 3D path matches it exactly.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let logits = Cpu.tensor([[[1.0f32, 2.0], [0.5, -0.5]]]).unwrap();
    /// let targets = Cpu.tensor([[1u32, 0]]).unwrap();
    /// let loss = logits.cross_entropy_token_loss(&targets).unwrap();
    /// assert!(loss.to_scalar::<f32>().unwrap().is_finite());
    /// ```
    pub fn cross_entropy_token_loss<
        S2: Shape + crate::shapes::DynShape,
        KT: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, KT, G2, Local, L2>,
    ) -> Result<crate::shapes::Dense<crate::shapes::Dyn, B, K, G, Local>>
    where
        L: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        B: Execute<op::CrossEntropyLoss> + Execute<op::ReshapeExact> + crate::exec::Capabilities,
        <B as Execute<op::CrossEntropyLoss>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>> + Into<B::Storage<KT>>,
    {
        self.cross_entropy_token_loss_with::<Mean, S2, KT, G2, L2>(target)
    }

    /// `cross_entropy_token_loss_with` picks the reduction at the type level.
    /// The output is dynamically shaped: a scalar for `Mean`/`Sum`, `[B, T]`
    /// for `None`.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let logits = Cpu.tensor([[[1.0f32, 2.0], [0.5, -0.5]]]).unwrap();
    /// let targets = Cpu.tensor([[1u32, 0]]).unwrap();
    /// // Token-CE with no reduction matches the flattened rank-2 CE row for row.
    /// let per_token = logits
    ///     .cross_entropy_token_loss_with::<NoneReduction, _, _, _, _>(&targets)
    ///     .unwrap();
    /// assert_eq!(per_token.dims().dims(), &[1, 2]);
    /// let flat_logits = logits.reshape(shape![2, 2]).unwrap();
    /// let flat_targets = targets.reshape(shape![2]).unwrap();
    /// let per_row = flat_logits
    ///     .cross_entropy_loss_with::<NoneReduction, _, _, _, _>(&flat_targets)
    ///     .unwrap();
    /// assert_eq!(
    ///     per_token.to_vec1::<f32>().unwrap(),
    ///     per_row.to_vec1::<f32>().unwrap()
    /// );
    /// ```
    pub fn cross_entropy_token_loss_with<
        R,
        S2: Shape + crate::shapes::DynShape,
        KT: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        L2: crate::shapes::Layout<S2>,
    >(
        &self,
        target: &Tensor<S2, B, KT, G2, Local, L2>,
    ) -> Result<crate::shapes::Dense<crate::shapes::Dyn, B, K, G, Local>>
    where
        R: ReductionMode,
        L: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        L2: crate::shapes::Restatable + crate::shapes::Layout<crate::shapes::Dyn>,
        B: Execute<op::CrossEntropyLoss> + Execute<op::ReshapeExact> + crate::exec::Capabilities,
        <B as Execute<op::CrossEntropyLoss>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<K>> + Into<B::Storage<KT>>,
    {
        let dims: Vec<usize> = self.dims().into();
        let [batch, seq, classes] = dims.as_slice() else {
            return Err(crate::err::Error::Shape(
                crate::shapes::error::ShapeError::RankMismatch {
                    operation: OperationKind::Reduction,
                    expected: crate::shapes::RankExpectation::Exactly(3),
                    actual: dims.len(),
                },
            ));
        };
        let (batch, seq, _classes) = (*batch, *seq, *classes);
        let target_dims: Vec<usize> = target.dims().into();
        if target_dims.as_slice() != [batch, seq] {
            let (axis, lhs, rhs) = if target_dims.first() != Some(&batch) {
                (0, batch, target_dims.first().copied().unwrap_or(0))
            } else {
                (1, seq, target_dims.get(1).copied().unwrap_or(0))
            };
            return Err(crate::err::Error::Shape(
                crate::shapes::error::ShapeError::DimensionMismatch {
                    operation: OperationKind::Reduction,
                    axis: crate::shapes::Axis::Index(axis),
                    lhs,
                    rhs,
                    constraint: crate::shapes::DimensionConstraint::Equal,
                },
            ));
        }
        // Flatten both operands over the leading axes with the existing
        // reshape machinery (grad-transparent, dtype-generic), then run the
        // rank-2 cross-entropy core the catalog already admits.
        let rows = batch.checked_mul(seq).ok_or_else(|| {
            crate::err::Error::Shape(crate::shapes::error::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Reshape,
                expression: "flattened batch-times-sequence product",
            })
        })?;
        let flat_pred =
            self.clone().into_dyn().reshape_infer(
                crate::shapes::InferShape::<crate::shapes::Dyn>::new(vec![Some(rows), None]),
            )?;
        let flat_target = target
            .clone()
            .into_dyn()
            .reshape_infer(crate::shapes::InferShape::<crate::shapes::Dyn>::new(vec![
                None,
            ]))?;
        let prediction = TensorHandle::from_storage::<B, K, Local>(&flat_pred.inner);
        let target_handle = TensorHandle::from_storage::<B, KT, Local>(&flat_target.inner);
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
        let storage: B::Storage<K> = inner.into();
        // `None` restores the [B, T] token geometry; Mean/Sum collapse to scalar.
        // The rank-2 core returns [rows] storage, so the `None` case wraps
        // with that exact shape first and reshapes to [B, T] through the
        // tracked reshape (grad-transparent like every other composition
        // step here); `from_parts` demands an exact shape match and would
        // refuse the direct [B, T] claim on [rows] storage.
        if R::as_enum() == Reduction::None {
            // Reshape the [rows] output storage back to [B, T] through the
            // tracked reshape (same grad-transparent step `reshape_infer`
            // takes, spelled out because the target geometry is fully
            // known); wrapping the [rows] storage with a [B, T] claim
            // directly would fail `from_parts`' exact-shape check.
            let out_buf =
                shape_buf_from_dims::<crate::shapes::Dyn>(OperationKind::Reduction, &[batch, seq])?;
            let out_shape = crate::shapes::ShapeValue::<crate::shapes::Dyn>::try_new(out_buf)
                .map_err(crate::err::Error::Shape)?;
            let input = TensorHandle::from_storage::<B, K, Local>(&storage);
            let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
            let inner = G::grad_mode(&self._grad)
                .restrict(|| {
                    dispatch::execute_shaped::<op::ReshapeExact, B, crate::shapes::Dyn>(
                        &context,
                        ShapeAttributes {
                            shape: out_shape.shape_buf().as_ref().to_vec(),
                        },
                        &[input],
                        &out_shape,
                    )
                })
                .map_err(crate::err::Error::from)?;
            let storage: B::Storage<K> = inner.into();
            return Tensor::from_parts(
                storage,
                out_shape.shape_buf().clone(),
                self._dtype.clone(),
                self._device.clone(),
                self._grad.clone(),
            );
        }
        let out_shape = shape_buf_from_dims::<crate::shapes::Dyn>(OperationKind::Reduction, &[])?;
        Tensor::from_parts(
            storage,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
