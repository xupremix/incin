//! Typed preallocated key/value cache for incremental decoding (issue #104).
//!
//! A decoder that generates one token at a time would otherwise re-run the
//! whole prefix through attention on every step. [`KvCache`] holds the keys
//! and values already computed for that prefix in a buffer whose shape is
//! `[batch, kv_heads, capacity, head_dim]`, so a step only projects its own
//! tokens and attends against the cache.
//!
//! # Why the shape is static and rank-4
//!
//! Capacity is part of the type, not a growable field: the buffer is allocated
//! once for the full decode window, and an append that would run past it fails
//! with [`Error::CacheCapacityExceeded`](crate::err::Error::CacheCapacityExceeded)
//! rather than reallocating. A `Dyn` shape would let the capacity drift away
//! from the type that promised it, so `S` must be a fully static rank-4 shape
//! (for example `s![batch, kv_heads, capacity, head_dim]`). Construction
//! rejects anything else with the violated geometry named.
//!
//! The cache is deliberately **not** a [`VisitState`](crate::nn::VisitState)
//! member of an attention module: generation owns it across calls, passes it
//! to
//! [`MultiHeadAttention::forward_with_cache`](crate::nn::MultiHeadAttention::forward_with_cache)
//! (which appends as it decodes) or, for cross-attention, fills it once with
//! [`CrossAttention::prefill_memory`](crate::nn::CrossAttention::prefill_memory)
//! and reads it by shared reference from
//! [`CrossAttention::forward_with_cache`](crate::nn::CrossAttention::forward_with_cache),
//! and drops it when decoding ends. `len` is runtime bookkeeping and is never
//! serialized; [`KvCache::reset`] starts a new sequence without touching the
//! allocation.
//!
//! # Example
//!
//! ```
//! # extern crate incin_core as incin;
//! # use incin::nn::KvCache;
//! # use incin::prelude::*;
//! # type Cpu = incin_backends::cpu::CpuBackendImpl;
//! # fn main() -> Result<()> {
//! // [batch=1, kv_heads=2, capacity=8, head_dim=4]
//! let mut cache = KvCache::<s![1, 2, 8, 4], Cpu, f32>::new(())?;
//! assert_eq!(cache.len(), 0);
//! assert_eq!(cache.capacity(), 8);
//!
//! let k = Tensor::<Dyn, Cpu>::zeros(vec![1, 2, 3, 4])?;
//! let v = Tensor::<Dyn, Cpu>::zeros(vec![1, 2, 3, 4])?;
//! cache.append(&k, &v)?;
//! assert_eq!(cache.len(), 3);
//!
//! let (ck, cv) = cache.kv()?;
//! assert_eq!(ck.dims().dims(), &[1, 2, 3, 4]);
//! assert_eq!(cv.dims().dims(), &[1, 2, 3, 4]);
//!
//! cache.reset();
//! assert_eq!(cache.len(), 0);
//! # Ok(())
//! # }
//! ```

use alloc::format;
use alloc::vec::Vec;

use crate::dist::Local;
use crate::err::{Error, ErrorMessage, Result};
use crate::exec::Capabilities;
use crate::exec::catalog::op;
use crate::nn::param::{Buffer, ParameterInit};
use crate::shapes::{Dyn, DynShape, Layout, Shape, ShapeBuf};
use crate::tensor::arg::TensorArgs;
use crate::tensor::arg_into::ArgInto;
use crate::tensor::backend::{Execute, SupportsDType, VariableBackend};
use crate::tensor::base::Tensor;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::{Grad, NoGrad};

/// Preallocated key/value cache with a static `[batch, kv_heads, capacity, head_dim]` shape.
///
/// See the [module documentation](self) for the capacity contract and why the
/// cache stays outside module state.
#[derive(Debug)]
pub struct KvCache<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType = f32> {
    keys: Buffer<S, B, K>,
    values: Buffer<S, B, K>,
    len: usize,
}

impl<S, B, K> Clone for KvCache<S, B, K>
where
    S: Shape,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
{
    /// Clones the underlying buffers (each clone shares the variable slot via
    /// the backend's `Var: Clone`); `len` is independent bookkeeping.
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            values: self.values.clone(),
            len: self.len,
        }
    }
}

/// Rejects any cache shape that is not fully static rank-4.
fn check_geometry<S: Shape>(operation: &'static str) -> Result<()> {
    let rank = S::RANK;
    let extents = S::STATIC_EXTENTS;
    let all_static = extents.len() == 4 && extents.iter().all(Option::is_some);
    if rank == Some(4) && all_static {
        return Ok(());
    }
    let described = match (rank, extents.is_empty()) {
        (Some(r), false) if extents.len() == r => alloc::format!("{extents:?}"),
        (Some(r), _) => alloc::format!("rank {r} with static extents {extents:?}"),
        (None, _) => alloc::format!("a non-static shape (rank {rank:?})"),
    };
    Err(Error::InvalidModuleState {
        operation,
        reason: ErrorMessage::new(format!(
            "KvCache requires a fully static rank-4 shape \
             [batch, kv_heads, capacity, head_dim]; got {described}"
        )),
    })
}

/// Zero-initializes one capacity-sized buffer from already-resolved parts.
fn buffer_zeros<S, B, K>(
    shape: ShapeBuf,
    dtype: K::Field,
    device: <B::Device as Device>::Field,
) -> Result<Buffer<S, B, K>>
where
    S: Shape,
    B: VariableBackend + SupportsDType<K> + ParameterInit<K>,
    K: DType,
{
    let var = B::execute_plan_raw(
        shape.as_ref(),
        &dtype,
        &device,
        crate::nn::init::InitPlan::Zeros,
    )?;
    Buffer::from_parts_checked(var, shape, dtype, device)
}

/// The `(keys, values)` pair returned by [`KvCache::kv`]: both `Dyn`-shaped,
/// `NoGrad`, rank-4 `[batch, kv_heads, len, head_dim]` views.
type KvPair<B, K> = (
    Tensor<Dyn, B, K, NoGrad, Local>,
    Tensor<Dyn, B, K, NoGrad, Local>,
);

/// Construction and geometry accessors (no backend op rows required).
impl<S, B, K> KvCache<S, B, K>
where
    S: Shape + DynShape,
    B: VariableBackend,
    K: DType,
{
    /// Number of tokens currently stored (the logical sequence length).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache holds no tokens.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Full cache geometry: `[batch, kv_heads, capacity, head_dim]`.
    #[must_use]
    pub fn dims(&self) -> Vec<usize> {
        self.keys.shape_dims()
    }

    /// Preallocated sequence capacity (dimension 2 of the cache shape).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.keys.shape_dims()[2]
    }

    /// Batch extent of the cache.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.keys.shape_dims()[0]
    }

    /// Key/value head count stored in the cache.
    #[must_use]
    pub fn kv_heads(&self) -> usize {
        self.keys.shape_dims()[1]
    }

    /// Per-head width stored in the cache.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.keys.shape_dims()[3]
    }

    /// Clears the logical length without touching the allocation.
    ///
    /// Bytes past the new `len` may still hold the previous sequence; only
    /// [`kv`](Self::kv) exposes stored content, and it stops at `len`.
    pub const fn reset(&mut self) {
        self.len = 0;
    }

    /// Views an entire capacity-sized buffer as a `Dyn` tensor.
    fn view_full(&self, buffer: &Buffer<S, B, K>) -> Result<Tensor<Dyn, B, K, NoGrad, Local>> {
        let tensor = buffer.as_tensor()?;
        let shape = tensor.shape_buf().clone();
        let dtype = tensor._dtype.clone();
        let device = tensor._device.clone();
        let grad = tensor._grad;
        let inner = tensor.inner;
        Tensor::try_from_storage(inner, shape, dtype, device, grad)
    }
}

/// Allocation (needs the parameter-initializer rows).
impl<S, B, K> KvCache<S, B, K>
where
    S: Shape + DynShape,
    B: VariableBackend + SupportsDType<K> + ParameterInit<K>,
    K: DType,
{
    /// Builds a zero-filled cache whose type fixes the geometry.
    ///
    /// For a fully static shape the constructor argument is `()`; dtype and
    /// device arguments follow the same `TensorArgs` rules as every other
    /// parameterized allocation.
    pub fn new<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
    {
        check_geometry::<S>("KvCache::new")?;
        let (shape, dtype, device, _) =
            <(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::construct(
                args.into_arg(),
            )?;
        let keys = buffer_zeros(shape.clone(), dtype.clone(), device.clone())?;
        let values = buffer_zeros(shape, dtype, device)?;
        Ok(Self {
            keys,
            values,
            len: 0,
        })
    }
}

/// Read-back and append (needs narrow + concat rows).
impl<S, B, K> KvCache<S, B, K>
where
    S: Shape + DynShape,
    B: VariableBackend + Capabilities + Execute<op::Narrow> + Execute<op::ConcatExact>,
    K: DType,
    <B as Execute<op::Narrow>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::ConcatExact>>::Output: Into<B::Storage<K>>,
{
    /// Logical keys/values `[batch, kv_heads, len, head_dim]`, never the
    /// capacity-sized slots past `len`.
    ///
    /// Stale bytes beyond `len` (including content left by a previous sequence
    /// after [`reset`](Self::reset)) are not part of this view.
    pub fn kv(&self) -> Result<KvPair<B, K>> {
        let keys = self
            .view_full(&self.keys)?
            .try_narrow(2isize, 0, self.len)?;
        let values = self
            .view_full(&self.values)?
            .try_narrow(2isize, 0, self.len)?;
        Ok((keys.forget_layout(), values.forget_layout()))
    }

    /// Appends one decode step's keys and values at axis 2.
    ///
    /// `keys`/`values` must be rank-4 `[batch, kv_heads, seq, head_dim]` with
    /// `NoGrad` tracking (detach projections before caching). Overflow raises
    /// [`Error::CacheCapacityExceeded`];
    /// a geometry mismatch raises [`Error::InvalidModuleState`].
    pub fn append<LK, LV>(
        &mut self,
        keys: &Tensor<Dyn, B, K, NoGrad, Local, LK>,
        values: &Tensor<Dyn, B, K, NoGrad, Local, LV>,
    ) -> Result<()>
    where
        LK: Layout<Dyn>,
        LV: Layout<Dyn>,
    {
        let k_dims = keys.shape_buf().as_ref().to_vec();
        let v_dims = values.shape_buf().as_ref().to_vec();
        let [batch, kv_heads, seq, head_dim] = k_dims[..] else {
            return Err(Error::InvalidModuleState {
                operation: "KvCache::append",
                reason: ErrorMessage::new(format!(
                    "expected rank-4 keys [batch, kv_heads, seq, head_dim], got {k_dims:?}"
                )),
            });
        };
        if v_dims != k_dims {
            return Err(Error::InvalidModuleState {
                operation: "KvCache::append",
                reason: ErrorMessage::new(format!(
                    "keys {k_dims:?} and values {v_dims:?} must have the same shape"
                )),
            });
        }
        let cache_dims = self.keys.shape_dims();
        if batch != cache_dims[0] || kv_heads != cache_dims[1] || head_dim != cache_dims[3] {
            return Err(Error::InvalidModuleState {
                operation: "KvCache::append",
                reason: ErrorMessage::new(format!(
                    "key/value shape {k_dims:?} does not match the cache \
                     [batch, kv_heads, capacity, head_dim] {cache_dims:?}"
                )),
            });
        }
        if seq == 0 {
            return Ok(());
        }
        let capacity = cache_dims[2];
        let start = self.len;
        let end = start + seq;
        if end > capacity {
            return Err(Error::CacheCapacityExceeded {
                operation: "KvCache::append",
                requested: end,
                capacity,
            });
        }

        let new_keys = self.stitch(&self.keys, keys, start, end, capacity)?;
        let new_values = self.stitch(&self.values, values, start, end, capacity)?;
        B::assign_var::<K>(&mut self.keys.inner, &new_keys.inner)?;
        B::assign_var::<K>(&mut self.values.inner, &new_values.inner)?;
        self.len = end;
        Ok(())
    }

    /// Rebuilds `prefix ++ new ++ tail` at full capacity for one buffer.
    ///
    /// The tail (slots past the new `len`) is preserved so `assign_var` can
    /// write a complete capacity-shaped candidate; those bytes are never
    /// exposed by [`kv`](Self::kv).
    fn stitch<LN: Layout<Dyn>>(
        &self,
        buffer: &Buffer<S, B, K>,
        new: &Tensor<Dyn, B, K, NoGrad, Local, LN>,
        start: usize,
        end: usize,
        capacity: usize,
    ) -> Result<Tensor<Dyn, B, K, NoGrad, Local>> {
        let full = self.view_full(buffer)?;
        let mut parts: Vec<Tensor<Dyn, B, K, NoGrad, Local>> = Vec::new();
        if start > 0 {
            parts.push(full.clone().try_narrow(2isize, 0, start)?.forget_layout());
        }
        parts.push(new.clone().forget_layout());
        if end < capacity {
            parts.push(
                full.try_narrow(2isize, end, capacity - end)?
                    .forget_layout(),
            );
        }
        let mut acc = parts
            .pop()
            .expect("stitch always has at least the new segment");
        while let Some(part) = parts.pop() {
            acc = part.concat(&acc, 2isize)?.forget_layout();
        }
        Ok(acc)
    }
}
