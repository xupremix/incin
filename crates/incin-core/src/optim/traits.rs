//! `Optimizer`/`ScaledOptimizer`, the generic update-rule vocabulary, plus
//! `OptimizerBackend`/`ValueClippingBackend`, the canonical execution
//! profile a backend implements to support them and the blanket impl over
//! the exact-op descriptors that satisfies it. Kept as one file even though
//! `super::group`'s `ParameterGroup` sits textually between them in the
//! original single-file layout, since neither trait pair needs anything
//! `ParameterGroup` defines.

use crate::autograd::Gradients;
use crate::err::Result;
use crate::tensor::backend::{AutogradBackend, VariableBackend};

/// Trait defining a generic optimization algorithm.
///
/// Implementors receive a reference to the [`Gradients`] computed from a backward pass
/// and apply the appropriate parameter update rule to all tracked variables.
pub trait Optimizer<B: VariableBackend + AutogradBackend> {
    /// Steps the optimizer using the given gradients, updating the tracked parameters.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()>;
}

/// Extension trait for optimizers that support mixed-precision loss scaling.
pub trait ScaledOptimizer<B: VariableBackend + AutogradBackend>: Optimizer<B> {
    /// Steps the optimizer with loss scaling support, unscaling gradients in-place and skipping
    /// updates on overflow.
    ///
    /// Returns `Ok(true)` if the step was applied (gradients were finite and unscaled).
    /// Returns `Ok(false)` if non-finite gradients were found (overflow), skipping the update.
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool>;
}

use crate::backend_authoring::{Capabilities, Execute};
use crate::dist::Local;
use crate::err::Error;
use crate::exec::request::TensorHandle;
use crate::exec::{
    catalog::{ClampAttributes, NoAttributes, ScalarAttributes, op},
    dispatch,
};
use crate::shapes::Dyn;
use crate::shapes::{ShapeBuf, ShapeValue};
use crate::tensor::dtype::DType;

/// The canonical execution profile required by generic optimizers.
///
/// Backend authors satisfy this profile by implementing the exact operation
/// descriptors listed in the blanket implementation below.
pub trait OptimizerBackend<K: DType>: VariableBackend {
    fn optimizer_add(lhs: &Self::Storage<K>, rhs: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_sub(lhs: &Self::Storage<K>, rhs: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_mul(lhs: &Self::Storage<K>, rhs: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_div(lhs: &Self::Storage<K>, rhs: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_sqrt(storage: &Self::Storage<K>) -> Result<Self::Storage<K>>;
    fn optimizer_mul_scalar(storage: &Self::Storage<K>, value: f64) -> Result<Self::Storage<K>>;
    fn optimizer_add_scalar(storage: &Self::Storage<K>, value: f64) -> Result<Self::Storage<K>>;
}

impl<B, K: DType> OptimizerBackend<K> for B
where
    B: VariableBackend
        + Capabilities
        + Execute<op::Add>
        + Execute<op::Sub>
        + Execute<op::Mul>
        + Execute<op::Div>
        + Execute<op::Sqrt>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sub>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Mul>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Div>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Sqrt>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<K>>,
{
    fn optimizer_add(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_binary::<op::Add, B, K>(lhs, rhs)
    }

    fn optimizer_sub(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_binary::<op::Sub, B, K>(lhs, rhs)
    }

    fn optimizer_mul(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_binary::<op::Mul, B, K>(lhs, rhs)
    }

    fn optimizer_div(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_binary::<op::Div, B, K>(lhs, rhs)
    }

    fn optimizer_sqrt(storage: &B::Storage<K>) -> Result<B::Storage<K>> {
        execute_unary::<op::Sqrt, B, K>(storage)
    }

    fn optimizer_mul_scalar(storage: &B::Storage<K>, value: f64) -> Result<B::Storage<K>> {
        execute_scalar::<op::MulScalar, B, K>(storage, value)
    }

    fn optimizer_add_scalar(storage: &B::Storage<K>, value: f64) -> Result<B::Storage<K>> {
        execute_scalar::<op::AddScalar, B, K>(storage, value)
    }
}

/// Per-element gradient clamping, as a capability separate from
/// [`OptimizerBackend`] rather than a required method on it.
///
/// `Execute<op::Clamp>` is a CPU-only descriptor today - CUDA, WGPU, and
/// Metal do not implement it. Making [`clip_grad_value`](crate::optim::clip_grad_value) a method on
/// `OptimizerBackend` itself would have added that bound to every backend's
/// existing, working conformance, breaking `Adam`/`SGD`/`AdamW` for every
/// non-CPU backend the moment this landed. This trait exists so a backend
/// that can already run `Adam` keeps being able to, whether or not it can
/// also clamp; `clip_grad_value` requires this trait specifically, not
/// `OptimizerBackend`.
pub trait ValueClippingBackend<K: DType>: OptimizerBackend<K> {
    /// Clamps every element of `storage` into `[min, max]`, independently of
    /// every other element - the per-element counterpart to the group-wide
    /// rescale [`clip_grad_norm`](crate::optim::clip_grad_norm) performs.
    fn optimizer_clamp(storage: &Self::Storage<K>, min: f64, max: f64) -> Result<Self::Storage<K>>;
}

impl<B, K: DType> ValueClippingBackend<K> for B
where
    B: OptimizerBackend<K> + Capabilities + Execute<op::Clamp>,
    <B as Execute<op::Clamp>>::Output: Into<B::Storage<K>>,
{
    fn optimizer_clamp(storage: &B::Storage<K>, min: f64, max: f64) -> Result<B::Storage<K>> {
        execute_clamp::<op::Clamp, B, K>(storage, min, max)
    }
}

fn execute_binary<O, B, K>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>
where
    O: crate::exec::catalog::Operation<Attributes = NoAttributes>,
    B: VariableBackend + Capabilities + Execute<O>,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let expected =
        ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&B::shape(lhs))).map_err(Error::Shape)?;
    let inputs = [
        TensorHandle::from_storage::<B, K, Local>(lhs),
        TensorHandle::from_storage::<B, K, Local>(rhs),
    ];
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(&context, NoAttributes, &inputs, &expected)
        .map(Into::into)
        .map_err(Error::from)
}

fn execute_unary<O, B, K>(storage: &B::Storage<K>) -> Result<B::Storage<K>>
where
    O: crate::exec::catalog::Operation<Attributes = NoAttributes>,
    B: VariableBackend + Capabilities + Execute<O>,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&B::shape(storage)))
        .map_err(Error::Shape)?;
    let input = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(&context, NoAttributes, &[input], &expected)
        .map(Into::into)
        .map_err(Error::from)
}

fn execute_scalar<O, B, K>(storage: &B::Storage<K>, value: f64) -> Result<B::Storage<K>>
where
    O: crate::exec::catalog::Operation<Attributes = ScalarAttributes>,
    B: VariableBackend + Capabilities + Execute<O>,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&B::shape(storage)))
        .map_err(Error::Shape)?;
    let input = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(&context, ScalarAttributes { value }, &[input], &expected)
        .map(Into::into)
        .map_err(Error::from)
}

fn execute_clamp<O, B, K>(storage: &B::Storage<K>, min: f64, max: f64) -> Result<B::Storage<K>>
where
    O: crate::exec::catalog::Operation<Attributes = ClampAttributes>,
    B: VariableBackend + Capabilities + Execute<O>,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&B::shape(storage)))
        .map_err(Error::Shape)?;
    let input = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(
        &context,
        ClampAttributes { min, max },
        &[input],
        &expected,
    )
    .map(Into::into)
    .map_err(Error::from)
}
