//! `clip_grad_norm`/`clip_grad_value`, the public gradient-clipping API
//! called between the backward pass and `Optimizer::step`.

use super::group::ParameterGroup;
use super::support::invalid_optimizer_config;
use super::traits::{OptimizerBackend, ValueClippingBackend};
use crate::autograd::Gradients;
use crate::err::Result;
use crate::tensor::backend::{AutogradBackend, HostReadback, VariableBackend};
use crate::tensor::dtype::ConstDType;

/// Rescales a parameter group's gradients so their global L2 norm is at most
/// `max_norm`, and returns the norm they had before rescaling.
///
/// This is the standard total-norm form: the norm is taken over the
/// concatenation of every gradient in the group, not per parameter, so the
/// direction of the update is preserved and only its length changes. A group
/// already under the threshold is left untouched and the returned norm is the
/// one it had.
///
/// Call it between the backward pass and [`Optimizer::step`](crate::optim::Optimizer::step):
///
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::optim::{ParameterGroup, clip_grad_norm};
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
/// let input = Tensor::<s![1, 4], DefaultBackend>::ones(())?.require_grad();
/// let mut gradients = model.forward(input)?.sum_all()?.backward()?;
///
/// let group = ParameterGroup::<DefaultBackend, f32>::from_module(&model)?;
/// let before = clip_grad_norm(&group, &mut gradients, 1.0)?;
/// assert!(before >= 0.0);
///
/// let mut optimizer = SGD::<DefaultBackend>::from_module(&model, 0.01)?;
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
///
/// # Errors
///
/// Returns an error when `max_norm` is not finite and positive, when a
/// gradient cannot be read back to the host, or when the backend refuses the
/// rescale.
pub fn clip_grad_norm<B, K>(
    params: &ParameterGroup<B, K>,
    grads: &mut Gradients<B>,
    max_norm: f64,
) -> Result<f64>
where
    B: VariableBackend + AutogradBackend + OptimizerBackend<K> + HostReadback,
    K: ConstDType,
{
    const OPERATION: &str = "clip_grad_norm";
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(invalid_optimizer_config(
            OPERATION,
            "the maximum norm must be finite and greater than zero",
        ));
    }

    // Two passes over the group, because the scale factor is a property of the
    // whole set: nothing can be rescaled until every gradient has been
    // measured. The first pass also collects the storage handles so the second
    // does not repeat the parameter lookup.
    let mut squared_total = 0.0f64;
    let mut present = alloc::vec::Vec::new();
    for (_, var) in params.iter() {
        let tensor = B::var_as_tensor::<K>(var)?;
        let Some(grad) = B::get_grad::<K>(&tensor, grads.as_backend())? else {
            continue;
        };
        for value in B::float_to_vec1::<K>(&grad)? {
            squared_total += value * value;
        }
        present.push((tensor, grad));
    }

    let total_norm = squared_total.sqrt();
    if !total_norm.is_finite() {
        return Err(invalid_optimizer_config(
            OPERATION,
            "the gradient norm is not finite, so no finite rescale exists. Inspect the \
             backward pass rather than clipping a NaN into range.",
        ));
    }
    if total_norm <= max_norm {
        return Ok(total_norm);
    }

    // The epsilon keeps the divisor away from zero. It cannot matter here -
    // this branch already established `total_norm > max_norm > 0` - but it
    // keeps the expression the same one every reference implementation writes,
    // which is worth more than the branch it would save.
    let scale = max_norm / (total_norm + 1e-6);
    for (tensor, grad) in present {
        let scaled = B::optimizer_mul_scalar(&grad, scale)?;
        B::set_grad::<K>(&tensor, grads.as_backend_mut(), scaled)?;
    }
    Ok(total_norm)
}

/// Clamps every element of every gradient in a parameter group into
/// `[-clip_value, clip_value]`, independently of every other element.
///
/// This is the per-element counterpart to [`clip_grad_norm`]'s group-wide
/// rescale: a gradient with one exploding element and otherwise-reasonable
/// ones is left with that one element flattened to the bound rather than
/// having its whole direction rescaled by the outlier. The two are not
/// interchangeable and neither dominates the other - `clip_grad_norm`
/// preserves the gradient's direction, this does not.
///
/// Call it between the backward pass and [`Optimizer::step`](crate::optim::Optimizer::step), exactly where
/// [`clip_grad_norm`] is called:
///
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::optim::{ParameterGroup, clip_grad_value};
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
/// let input = Tensor::<s![1, 4], DefaultBackend>::ones(())?.require_grad();
/// let mut gradients = model.forward(input)?.sum_all()?.backward()?;
///
/// let group = ParameterGroup::<DefaultBackend, f32>::from_module(&model)?;
/// clip_grad_value(&group, &mut gradients, 1.0)?;
///
/// let mut optimizer = SGD::<DefaultBackend>::from_module(&model, 0.01)?;
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
///
/// # Errors
///
/// Returns an error when `clip_value` is not finite and positive, or when
/// the backend refuses the clamp.
pub fn clip_grad_value<B, K>(
    params: &ParameterGroup<B, K>,
    grads: &mut Gradients<B>,
    clip_value: f64,
) -> Result<()>
where
    B: VariableBackend + AutogradBackend + ValueClippingBackend<K>,
    K: ConstDType,
{
    const OPERATION: &str = "clip_grad_value";
    if !clip_value.is_finite() || clip_value <= 0.0 {
        return Err(invalid_optimizer_config(
            OPERATION,
            "the clip value must be finite and greater than zero",
        ));
    }

    for (_, var) in params.iter() {
        let tensor = B::var_as_tensor::<K>(var)?;
        let Some(grad) = B::get_grad::<K>(&tensor, grads.as_backend())? else {
            continue;
        };
        let clamped = B::optimizer_clamp(&grad, -clip_value, clip_value)?;
        B::set_grad::<K>(&tensor, grads.as_backend_mut(), clamped)?;
    }
    Ok(())
}
