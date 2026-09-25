//! Per-tensor scaled casts into and out of FP8 (issue #94).
//!
//! An FP8 tensor without a scale is not usable: `e4m3` spans ±448, so values
//! must be scaled into range before the cast, and the scale travels with the
//! cast explicitly. There is no implicit path in: the
//! [`to_dtype_scaled`](crate::tensor::base::Tensor::to_dtype_scaled) method
//! divides by `scale` first (so `amax / scale <= 1` keeps every value
//! representable up to rounding), and
//! [`from_dtype_scaled`](crate::tensor::base::Tensor::from_dtype_scaled)
//! multiplies back.
//!
//! Both compose recorded operations (`div_scalar`/`mul_scalar` around
//! [`to_dtype`](crate::tensor::base::Tensor::to_dtype)), so gradients flow:
//! the `to_dtype` cast records the float-to-float identity edge (fp8 is
//! `DTypeKind::Float`, which is what admits that edge), and the scalar
//! multiplies record their own. No new catalog operation was needed.
//!
//! The scale policy itself (adaptive loss-scaling-style growth/backoff) is
//! device-bound work under issues #2/#94; the CPU slice here is the explicit
//! mechanism with a caller-held scale, e.g. derived from
//! `w.abs()?.max_all()?`.

use crate::backend_authoring::Backend;
use crate::backend_authoring::Execute;
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::Capabilities;
use crate::exec::catalog::op;
use crate::shapes::{Dense, DynShape, Layout, Shape};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::{DType, FloatDType};
use crate::tensor::grad::RequiresGrad;

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad, L: Layout<S>>
    Tensor<S, B, K, G, Local, L>
{
    /// Casts into an FP8 dtype with an explicit per-tensor scale.
    ///
    /// Computes `self / scale` in the input dtype, then casts to `T2`
    /// (`F8E4M3` or `F8E5M2`). The caller owns the scale: with
    /// `scale = amax`, every value lands in `[-1, 1]` and the fp8 cast only
    /// rounds, never saturates.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let w = Cpu.tensor([224.0f32, -112.0, 56.0]).unwrap();
    /// let w8 = w.to_dtype_scaled::<F8E4M3>(224.0).unwrap();
    /// assert_eq!(w8.dtype(), F8E4M3::descriptor(&Default::default()));
    /// ```
    pub fn to_dtype_scaled<T2>(&self, scale: f32) -> Result<Dense<S, B, T2, G, Local>>
    where
        K: FloatDType,
        T2: FloatDType,
        B: Execute<op::DivScalar> + Execute<op::ToDType> + Capabilities,
        <B as Execute<op::DivScalar>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::ToDType>>::Output: Into<B::Storage<T2>>,
    {
        let scaled = self.div_scalar(f64::from(scale))?;
        scaled.to_dtype::<T2>()
    }

    /// Casts out of an FP8 dtype back to a float, reapplying the scale.
    ///
    /// Computes `self.to_dtype::<T2>() * scale`: the inverse of
    /// [`to_dtype_scaled`](Self::to_dtype_scaled) for the same `scale`.
    /// Admitting an fp8 *source* is a capability-row decision (the cast
    /// target is an attribute the executor checks, the source is what the
    /// row admits), so on backends whose rows do not list fp8 yet this
    /// reports the row's typed refusal rather than casting.
    pub fn from_dtype_scaled<T2>(&self, scale: f32) -> Result<Dense<S, B, T2, G, Local>>
    where
        K: FloatDType,
        T2: FloatDType,
        B: Execute<op::ToDType> + Execute<op::MulScalar> + Capabilities,
        <B as Execute<op::ToDType>>::Output: Into<B::Storage<T2>>,
        <B as Execute<op::MulScalar>>::Output: Into<B::Storage<T2>>,
    {
        let widened = self.to_dtype::<T2>()?;
        widened.mul_scalar(f64::from(scale))
    }
}
