//! Block quantization: `quantize`/`dequantize` and the dtype admission bounds.
//!
//! Implements issue #93's Decisions 3-5 at the tensor surface:
//!
//! - [`Tensor::quantize`](crate::tensor::base::Tensor::quantize) compresses a
//!   float tensor into `Q8_0` blocks along its last axis;
//!   [`Tensor::dequantize`](crate::tensor::base::Tensor::dequantize) decodes
//!   blocks back to a float dtype. Both dispatch the catalog's
//!   `Quantize`/`Dequantize` descriptors, so validation, capability
//!   admission, and the straight-through gradient node are the backend's,
//!   exactly as for every other operation.
//! - Block divisibility (Decision 4) is proved twice: a `const { assert! }`
//!   reads [`Shape::STATIC_EXTENTS`](crate::shapes::Shape::STATIC_EXTENTS)
//!   and fails compilation when a statically known last-axis extent is not a
//!   multiple of the 32-element block, and a runtime check reports the same
//!   rule as a typed error for extents only known with a `Dyn` shape.
//! - Dtype admission (Decision 3) is the [`FloatCapable`]/[`QuantCapable`]
//!   bound pair: statically quantized (or integer) dtypes are refused at the
//!   call site, while `Dyn` admits the call so the operation catalog's
//!   runtime dtype check can refuse with a descriptive error instead.

use alloc::format;

use crate::backend_authoring::Backend;
use crate::backend_authoring::Execute;
use crate::dist::placement::Local;
use crate::err::{Error, Result};
use crate::exec::Capabilities;
use crate::exec::catalog::{QuantizationAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::Layout;
use crate::shapes::{Dense, Shape};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::{DType, FloatDType, Q8_0, QuantDType};
use crate::tensor::grad::RequiresGrad;

/// Elements per `Q8_0` block: the extent the block axis must divide evenly.
///
/// Mirrors [`StorageEncoding::block`](crate::tensor::dtype::StorageEncoding::block)`(32, 34, 2)` behind
/// [`Q8_0::DESCRIPTOR`] (32 `i8` values plus one `f16` scale per 34-byte
/// block). The CPU kernels chunk the flat buffer in units of this many
/// elements, so a last-axis extent that is a whole multiple of it is exactly
/// the condition under which flat chunking stays inside one row.
const BLOCK_LEN: usize = 32;

/// A dtype whose values an elementwise float operation can consume.
///
/// Admits every [`FloatDType`] (`f32`/`f64`/`f16`/`bf16`) and [`Dyn`](crate::shapes::Dyn),
/// whose descriptor is only checkable at runtime: the operation catalog's
/// `require_float` rule rejects quantized and integer metadata there with a
/// typed descriptor error, which is Decision 3's runtime half.
///
/// Block-quantized dtypes such as [`Q8_0`] deliberately do **not** implement
/// it. An elementwise float kernel has no block path - there is no per-element
/// `f32` to read out of a 34-byte block without decoding it first - so
/// `Q8_0` operands are refused at the call site with `E0277` rather than
/// reaching a kernel that could not run. The same applies to integer dtypes:
/// the float unary surface never admitted them at runtime either.
///
/// This is the static counterpart of the catalog's `SemanticProfile::UnaryFloat`
/// admission, and the bound is applied to that profile's tensor methods
/// (issue #93, Decisions 3 and 4). Custom dtype definitions that want to pass
/// through float kernels can implement this trait for their own type.
pub trait FloatCapable: DType {}

impl<T: FloatDType> FloatCapable for T {}
impl FloatCapable for crate::shapes::Dyn {}

/// A dtype whose storage is a block-quantized encoding.
///
/// Admits every [`QuantDType`] (currently [`Q8_0`]) and [`Dyn`](crate::shapes::Dyn),
/// whose descriptor is only checkable at runtime. `Dyn` operands reach the
/// operation catalog, which refuses anything that is not `Q8_0` with a typed
/// descriptor error ("dequantize requires q8_0 input and floating output
/// metadata") - Decision 3's runtime half for dtype facts.
///
/// Float and integer dtypes do **not** implement it: decoding blocks is only
/// meaningful when the input really holds blocks, so `dequantize` on an `f32`
/// operand fails with `E0277` naming this trait instead of reaching a kernel
/// that would reinterpret scalar bytes as block storage.
pub trait QuantCapable: DType {}

impl<T: QuantDType> QuantCapable for T {}
impl QuantCapable for crate::shapes::Dyn {}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, L: Layout<S>> Tensor<S, B, K, G, Local, L> {
    /// Compresses this float tensor into `Q8_0` blocks along `axis`.
    ///
    /// `Q8_0` stores 32 consecutive elements as one 34-byte block (32 `i8`
    /// values scaled by one `f16`), so the chosen axis - which must resolve to
    /// the *last* axis, `-1` or `rank - 1` - has to be a whole multiple of 32.
    /// Only the last axis can carry that layout: the kernel indexes the flat
    /// row-major buffer, and a last axis divisible by 32 is precisely the
    /// condition under which flat 32-element chunks never straddle two rows.
    /// Any other axis argument is refused with a typed error naming it.
    ///
    /// The divisibility proof has two halves (issue #93, Decision 4):
    ///
    /// - a statically known extent fails **compilation** at this call site via
    ///   a `const { assert! }` over [`Shape::STATIC_EXTENTS`], proved by the
    ///   compile-fail fixture `quantize_block_axis_not_divisible`;
    /// - an extent only known at runtime (a `Dyn` shape) fails with
    ///   [`Error::Msg`] naming the axis, the actual extent, and the required
    ///   multiple of 32.
    ///
    /// `K` must be [`FloatCapable`], so a statically `Q8_0` (or integer)
    /// operand is `E0277` at the call site while a `Dyn` operand proceeds to
    /// the catalog's runtime float check. The result carries `Q8_0`'s dtype,
    /// this tensor's shape/device/gradient marker, and the backend's
    /// straight-through gradient node for `quantize` (Decision 2, wired in the
    /// CPU executor). The operand must be contiguous; the capability row
    /// refuses a strided view because the kernel reads the block buffer
    /// directly. The CPU row narrows the input further to `f32` (its kernel
    /// matches on the `F32` buffer variant rather than converting), so an
    /// `f16`/`f64`/`bf16` operand compiles and passes descriptor validation
    /// but is refused at admission with a typed `UnsupportedReason` rather
    /// than a panic; CUDA admits every float storage dtype through its typed
    /// kernel entries.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// // 64 elements: two full Q8_0 blocks along the last (only) axis.
    /// let t = Tensor::<s![64], IncinBackend>::full(0.5f32, ()).unwrap();
    /// let q = t.quantize(-1).unwrap();
    /// let back = q.dequantize::<f32>().unwrap();
    /// let values = back.to_vec1::<f32>().unwrap();
    /// // The f16 block scale keeps the round-trip approximate, not exact.
    /// assert!(values.iter().all(|v| (v - 0.5).abs() < 1e-2));
    /// ```
    pub fn quantize(&self, axis: isize) -> Result<Dense<S, B, Q8_0, G, Local>>
    where
        K: FloatCapable,
        B: Execute<op::Quantize> + Capabilities,
        <B as Execute<op::Quantize>>::Output: Into<B::Storage<Q8_0>>,
    {
        // Issue #93, Decision 4: the static half of the block proof. A
        // statically known last-axis extent that is not a multiple of 32
        // fails here at monomorphization (E0080 at the call site) rather
        // than reaching the kernel. `Dyn`'s STATIC_EXTENTS is empty, so the
        // block below evaluates to nothing and the runtime check carries
        // the rule instead.
        const {
            let extents = S::STATIC_EXTENTS;
            if !extents.is_empty()
                && let Some(extent) = extents[extents.len() - 1]
            {
                assert!(
                    extent.is_multiple_of(BLOCK_LEN),
                    "the last axis of a Q8_0 block must be a multiple of 32"
                );
            }
        }

        let dims = self.shape_buf().clone();
        let rank = dims.rank();
        if rank == 0 {
            return Err(Error::Msg(
                "quantize: a rank-0 tensor has no axis to block over; \
                 the operand must have rank >= 1"
                    .into(),
            ));
        }
        let last = rank - 1;
        let resolved = if axis < 0 {
            rank as i128 + axis as i128
        } else {
            axis as i128
        };
        if !(0..rank as i128).contains(&resolved) {
            return Err(Error::Msg(format!(
                "quantize: axis {axis} is out of bounds for a rank-{rank} tensor"
            )));
        }
        let idx = resolved as usize;
        if idx != last {
            return Err(Error::Msg(format!(
                "quantize: axis {axis} resolves to axis {idx}, but Q8_0 blocks run \
                 along the last axis only; pass -1 (the last axis, extent rules below)"
            )));
        }
        let extent = dims.as_ref()[last];
        if !extent.is_multiple_of(BLOCK_LEN) {
            return Err(Error::Msg(format!(
                "quantize: axis {last} has extent {extent}, which is not a multiple \
                 of the Q8_0 block size {BLOCK_LEN}; the block axis is the last axis, \
                 so its extent must be a whole multiple of {BLOCK_LEN}"
            )));
        }

        let field = <Q8_0 as DType>::init(());
        let descriptor = <Q8_0 as DType>::descriptor(&field);
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::Quantize, B, S>(
                    &context,
                    QuantizationAttributes { dtype: descriptor },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            field,
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Decodes this `Q8_0` block tensor into a float tensor of `Kout`.
    ///
    /// The inverse of [`quantize`](Self::quantize): blocks are read flat and
    /// expanded to one float element each, so the result has the same shape
    /// as the operand. `K` must be [`QuantCapable`] (`Q8_0` or `Dyn`, whose
    /// runtime descriptor the catalog checks) and `Kout` any [`FloatDType`];
    /// the CPU executor narrows the output further (its kernel currently
    /// writes `f32` only) and refuses a non-`f32` target with a typed
    /// `UnsupportedReason` rather than a panic.
    ///
    /// The backend records the straight-through gradient node for
    /// `dequantize` (Decision 2), dual to [`quantize`](Self::quantize)'s, so
    /// `dequantize(quantize(x))` backward is exactly the identity.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![32], IncinBackend>::full(1.0f32, ()).unwrap();
    /// let q = t.quantize(-1).unwrap();
    /// let back = q.dequantize::<f32>().unwrap();
    /// assert_eq!(back.to_vec1::<f32>().unwrap().len(), 32);
    /// ```
    pub fn dequantize<Kout: FloatDType>(&self) -> Result<Dense<S, B, Kout, G, Local>>
    where
        K: QuantCapable,
        B: Execute<op::Dequantize> + Capabilities,
        <B as Execute<op::Dequantize>>::Output: Into<B::Storage<Kout>>,
    {
        let field = <Kout as DType>::init(());
        let descriptor = <Kout as DType>::descriptor(&field);
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::Dequantize, B, S>(
                    &context,
                    QuantizationAttributes { dtype: descriptor },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            field,
            self._device.clone(),
            self._grad.clone(),
        )
    }
}
