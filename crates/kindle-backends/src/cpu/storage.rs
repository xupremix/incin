//! `CpuBuffer`/`CpuStorage`: the `Arc`-backed, `TensorId`-tagged data
//! structure every later op builds on.
//!
//! `CpuStorage` flows as an immutable, cheaply-cloned value: the `Rc`
//! clone is cheap, and view operations (`reshape`/`transpose`/`broadcast_as`)
//! never allocate a new buffer when the source is already contiguous — they
//! construct new shape/stride/offset metadata sharing the same `Arc<CpuBuffer>`.
//! This is NOT the `CpuVar` mutation boundary (a separate later plan);
//! nothing in this file mutates a `CpuBuffer` in place.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use half::{bf16, f16};
use kindle_core::prelude::Error;
use kindle_core::prelude::Result;

use crate::cpu::stride;

/// A monotonic identity tag for a `CpuStorage` value.
///
/// Backed by a global `AtomicU64` counter (never derived from a pointer
/// address, per the anti-pattern of using `Rc` pointer identity — pointer
/// reuse after drop is a real, hard-to-reproduce bug class). Two
/// independently constructed `TensorId`s are never equal, even after many
/// calls to `next()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

/// `NEXT_TENSOR_ID`.
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(0);

impl TensorId {
    /// Allocate a fresh, never-before-seen `TensorId`.
    pub(crate) fn next() -> Self {
        // Ordering::Relaxed is sufficient: this counter is an identity
        // source, not a synchronization primitive guarding shared data.
        TensorId(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Dtype-tagged raw data buffer.
///
/// All 7 `DTypeId` variants are reserved as enum shape per the
/// project's "Resolving the Deferred Gray Areas" decision. Only `F32` needs
/// real arithmetic elsewhere in this phase; the other variants exist as
/// data-holding shapes since `storage.rs` itself doesn't perform arithmetic,
/// only shape/stride bookkeeping.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_0 {
    pub(crate) d: half::f16,
    pub(crate) qs: [i8; 32],
}

#[derive(Debug, Clone, PartialEq)]
/// Implementation of `CpuBuffer` for the respective backend..
pub enum CpuBuffer {
    /// `F32`.
    F32(Vec<f32>),
    /// `F64`.
    F64(Vec<f64>),
    /// `U8`.
    U8(Vec<u8>),
    /// `U32`.
    U32(Vec<u32>),
    /// `I64`.
    I64(Vec<i64>),
    /// `F16`.
    F16(Vec<f16>),
    /// `BF16`.
    BF16(Vec<bf16>),
    /// `Q8_0`.
    Q8_0(Vec<BlockQ8_0>),
}

impl CpuBuffer {
    /// Total number of scalar elements held by this buffer, regardless of
    /// dtype variant.
    pub fn len(&self) -> usize {
        match self {
            CpuBuffer::F32(v) => v.len(),
            CpuBuffer::F64(v) => v.len(),
            CpuBuffer::U8(v) => v.len(),
            CpuBuffer::U32(v) => v.len(),
            CpuBuffer::I64(v) => v.len(),
            CpuBuffer::F16(v) => v.len(),
            CpuBuffer::BF16(v) => v.len(),
            CpuBuffer::Q8_0(v) => v.len() * 32,
        }
    }

    /// True if this buffer holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read raw bytes of this buffer (useful for sending to GPU).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            match self {
                CpuBuffer::F32(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                }
                CpuBuffer::F64(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                }
                CpuBuffer::U8(v) => v.as_slice(),
                CpuBuffer::U32(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                }
                CpuBuffer::I64(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                }
                CpuBuffer::F16(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
                }
                CpuBuffer::BF16(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
                }
                CpuBuffer::Q8_0(v) => core::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * core::mem::size_of::<BlockQ8_0>(),
                ),
            }
        }
    }

    /// Build a new buffer with the same dtype variant as `self`, populated
    /// from `values` (one `f64` per element, converted to the variant's
    /// native element type).
    ///
    /// Used by elementwise ops (`elementwise_binary`/`elementwise_unary`) to
    /// preserve the operands' actual dtype instead of hardcoding `F32` for
    /// every result regardless of input dtype (that hardcoding was a silent
    /// precision-loss bug: an F64/F16/BF16 tensor would previously come back
    /// out of `add`/`mul`/`relu`/etc. downcast through f32 with no error).
    ///
    /// Panics if `self` is `Q8_0` — quantized buffers are never produced by
    /// elementwise float ops.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn from_f64_values(&self, values: Vec<f64>) -> CpuBuffer {
        match self {
            CpuBuffer::F32(_) => CpuBuffer::F32(values.into_iter().map(|v| v as f32).collect()),
            CpuBuffer::F64(_) => CpuBuffer::F64(values),
            CpuBuffer::U8(_) => CpuBuffer::U8(values.into_iter().map(|v| v as u8).collect()),
            CpuBuffer::U32(_) => CpuBuffer::U32(values.into_iter().map(|v| v as u32).collect()),
            CpuBuffer::I64(_) => CpuBuffer::I64(values.into_iter().map(|v| v as i64).collect()),
            CpuBuffer::F16(_) => {
                CpuBuffer::F16(values.into_iter().map(half::f16::from_f64).collect())
            }
            CpuBuffer::BF16(_) => {
                CpuBuffer::BF16(values.into_iter().map(half::bf16::from_f64).collect())
            }
            CpuBuffer::Q8_0(_) => {
                panic!("from_f64_values not supported on Q8_0 quantized buffer")
            }
        }
    }

    /// Read the scalar at flat buffer index `i` as an `f64`, regardless of
    /// dtype variant. Used by strided-index resolution in
    /// `CpuStorage::get`.
    fn get_f64(&self, i: usize) -> f64 {
        match self {
            CpuBuffer::F32(v) => v[i] as f64,
            CpuBuffer::F64(v) => v[i],
            CpuBuffer::U8(v) => v[i] as f64,
            CpuBuffer::U32(v) => v[i] as f64,
            CpuBuffer::I64(v) => v[i] as f64,
            CpuBuffer::F16(v) => v[i].to_f64(),
            CpuBuffer::BF16(v) => v[i].to_f64(),
            CpuBuffer::Q8_0(_) => {
                panic!("get_f64 not supported directly on Q8_0 quantized buffer")
            }
        }
    }
}

/// A strided, `Arc`-backed view into a `CpuBuffer`.
///
/// `CpuStorage` is `Clone`-able cheaply: cloning only clones the `Rc`
/// pointer (increments the strong count) plus small `Vec<usize>` shape/stride
/// metadata, never the underlying buffer contents.
#[derive(Debug, Clone)]
pub struct CpuStorage {
    pub(crate) buffer: Arc<CpuBuffer>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) id: TensorId,
}

impl CpuStorage {
    /// Build a `CpuStorage` from a contiguous (row-major) buffer and
    /// shape. Strides are computed via `stride::contiguous_strides`.
    pub fn from_contiguous(data: CpuBuffer, shape: Vec<usize>) -> Self {
        let strides = stride::contiguous_strides(&shape);
        CpuStorage {
            buffer: Arc::new(data),
            shape,
            strides,
            offset: 0,
            id: TensorId::next(),
        }
    }

    /// Build a fresh, contiguous, all-ones `CpuStorage` with the same
    /// shape and dtype variant as `other`. Used by `tape::backward()` to
    /// seed the loss tensor's gradient before walking the tape.
    pub fn ones_like(other: &CpuStorage) -> Self {
        let total: usize = other.shape.iter().product();

        let new_buffer = match &*other.buffer {
            CpuBuffer::F32(_) => CpuBuffer::F32(vec![1.0f32; total]),
            CpuBuffer::F64(_) => CpuBuffer::F64(vec![1.0f64; total]),
            CpuBuffer::U8(_) => CpuBuffer::U8(vec![1u8; total]),
            CpuBuffer::U32(_) => CpuBuffer::U32(vec![1u32; total]),
            CpuBuffer::I64(_) => CpuBuffer::I64(vec![1i64; total]),
            CpuBuffer::F16(_) => CpuBuffer::F16(vec![half::f16::from_f64(1.0); total]),
            CpuBuffer::BF16(_) => CpuBuffer::BF16(vec![half::bf16::from_f64(1.0); total]),
            CpuBuffer::Q8_0(_) => panic!("ones_like not supported on Q8_0 buffer"),
        };

        CpuStorage::from_contiguous(new_buffer, other.shape.clone())
    }

    /// Resolve a logical multi-index through `self.strides`/`self.offset`
    /// into the underlying `CpuBuffer`, returning the value as `f64`.
    ///
    /// This works correctly through a non-contiguous (e.g. transposed) view
    /// without requiring a prior call to `contiguous()`.
    pub fn get(&self, idx: &[usize]) -> f64 {
        debug_assert_eq!(idx.len(), self.shape.len());
        let mut flat = self.offset;
        for (i, s) in idx.iter().zip(self.strides.iter()) {
            flat += i * s;
        }
        self.buffer.get_f64(flat)
    }

    /// Reshape to `new_shape`, per Pattern 1: metadata-only (sharing the
    /// same `Arc<CpuBuffer>`, no allocation) when `self` is already
    /// contiguous; otherwise materializes a contiguous copy first, then
    /// recurses.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        if stride::is_contiguous(&self.shape, &self.strides) {
            Ok(CpuStorage {
                buffer: self.buffer.clone(),
                shape: new_shape.to_vec(),
                strides: stride::contiguous_strides(new_shape),
                offset: self.offset,
                id: TensorId::next(),
            })
        } else {
            let materialized = self.contiguous();
            materialized.reshape(new_shape)
        }
    }

    /// Swap dimensions `dim1`/`dim2`. Metadata-only: shares the same `Rc`,
    /// keeps `offset`, produces (generally) non-contiguous strides.
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        if dim1 >= self.shape.len() || dim2 >= self.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "transpose",
                expected: self.shape.clone(),
                got: vec![dim1, dim2],
                msg: format!(
                    "transpose dims ({dim1}, {dim2}) out of range for shape {:?}",
                    self.shape
                ),
            });
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim1, dim2);
        strides.swap(dim1, dim2);
        Ok(CpuStorage {
            buffer: self.buffer.clone(),
            shape,
            strides,
            offset: self.offset,
            id: TensorId::next(),
        })
    }

    /// Broadcast this storage to `target_shape`. Metadata-only: any
    /// newly-inserted leading dim or any dim expanding from `1` gets stride
    /// `0`; all other dims keep their existing stride. Buffer shared via
    /// `Rc` clone.
    pub fn broadcast_as(&self, target_shape: &[usize]) -> Result<Self> {
        // Validate compatibility (right-aligned numpy/Candle-style rules).
        stride::broadcast_shape(&self.shape, target_shape)?;

        let target_len = target_shape.len();
        let src_len = self.shape.len();
        let mut new_strides = vec![0usize; target_len];

        for i in 0..target_len {
            // Right-align: axis `i` in target corresponds to axis
            // `i - (target_len - src_len)` in source, if that's non-negative.
            if i >= target_len - src_len {
                let src_axis = i - (target_len - src_len);
                let src_dim = self.shape[src_axis];
                let tgt_dim = target_shape[i];
                if src_dim == tgt_dim {
                    new_strides[i] = self.strides[src_axis];
                } else if src_dim == 1 {
                    // Expanding a size-1 dim: stride 0.
                    new_strides[i] = 0;
                } else {
                    return Err(Error::ShapeMismatch {
                        op: "broadcast_as",
                        expected: target_shape.to_vec(),
                        got: self.shape.clone(),
                        msg: format!("cannot broadcast dim {src_dim} to {tgt_dim} at axis {i}"),
                    });
                }
            }
            // else: newly-inserted leading dim -> stride 0 (already set above).
        }

        Ok(CpuStorage {
            buffer: self.buffer.clone(),
            shape: target_shape.to_vec(),
            strides: new_strides,
            offset: self.offset,
            id: TensorId::next(),
        })
    }

    /// Narrow dimension `dim` to the half-open range `[start, start + len)`.
    /// Metadata-only: shares the same `Arc<CpuBuffer>`, keeps `strides`
    /// completely unchanged (this is the load-bearing O(1) correctness
    /// property — never recompute strides from `contiguous_strides`, since
    /// that would silently produce wrong results on an already-transposed
    /// or otherwise non-contiguous source view), and only adjusts `offset`
    /// (by `start * strides[dim]`) and `shape[dim]` (to `len`).
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        if dim >= self.shape.len() || start + len > self.shape[dim] {
            return Err(Error::ShapeMismatch {
                op: "narrow",
                expected: self.shape.clone(),
                got: vec![dim, start, len],
                msg: format!(
                    "narrow(dim={dim}, start={start}, len={len}) out of bounds for shape {:?}",
                    self.shape
                ),
            });
        }
        let mut shape = self.shape.clone();
        shape[dim] = len;
        Ok(CpuStorage {
            buffer: self.buffer.clone(),
            shape,
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[dim],
            id: TensorId::next(),
        })
    }

    /// Materialize a fresh, contiguous copy of this storage by walking the
    /// current shape/strides/offset and copying element-by-element in
    /// row-major order. Used only on the non-contiguous fallback path of
    /// `reshape`.
    pub(crate) fn contiguous(&self) -> Self {
        if stride::is_contiguous(&self.shape, &self.strides) {
            return self.clone();
        }

        let total: usize = self.shape.iter().product();
        let mut multi_idx = vec![0usize; self.shape.len()];

        macro_rules! materialize {
            ($variant:ident, $ty:ty) => {{
                let mut out: Vec<$ty> = Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(self.get(&multi_idx) as $ty);
                    increment_index(&mut multi_idx, &self.shape);
                }
                CpuBuffer::$variant(out)
            }};
        }

        let new_buffer = match &*self.buffer {
            CpuBuffer::F32(_) => materialize!(F32, f32),
            CpuBuffer::F64(_) => materialize!(F64, f64),
            CpuBuffer::U8(_) => materialize!(U8, u8),
            CpuBuffer::U32(_) => materialize!(U32, u32),
            CpuBuffer::I64(_) => materialize!(I64, i64),
            CpuBuffer::F16(_) => {
                let mut out: Vec<f16> = Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(f16::from_f64(self.get(&multi_idx)));
                    increment_index(&mut multi_idx, &self.shape);
                }
                CpuBuffer::F16(out)
            }
            CpuBuffer::BF16(_) => {
                let mut out: Vec<bf16> = Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(bf16::from_f64(self.get(&multi_idx)));
                    increment_index(&mut multi_idx, &self.shape);
                }
                CpuBuffer::BF16(out)
            }
            CpuBuffer::Q8_0(_) => panic!("materialize not supported on Q8_0 buffer"),
        };

        CpuStorage::from_contiguous(new_buffer, self.shape.clone())
    }
}

/// Increment a row-major multi-index in place (odometer-style), matching the
/// iteration order `contiguous_strides` assumes.
pub(crate) fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for i in (0..idx.len()).rev() {
        idx[i] += 1;
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
    }
}

/// Build a zero-filled, freshly-allocated, contiguous `CpuStorage` of
/// `original_shape` (dtype-matched to `values`), then copy `values`'s data
/// into the sub-region starting at `region_start` (one offset per axis).
/// Every position outside that sub-region is left exactly zero.
///
/// This is the shared zero-pad-scatter backward primitive for `narrow`/
/// `slice`: `grad_out` (shaped like the narrowed region) is scattered back
/// into a zero buffer shaped like the original (pre-narrow) tensor, at the
/// same offset the forward narrow started from. It is a module-level free
/// function (not a `CpuStorage` method) because it constructs a NEW
/// storage from two independent shape/value inputs, rather than adjusting
/// `self`'s own metadata.
pub(crate) fn scatter_into_zeros(
    original_shape: &[usize],
    region_start: &[usize],
    values: &CpuStorage,
) -> CpuStorage {
    let total: usize = original_shape.iter().product();
    let out_strides = stride::contiguous_strides(original_shape);
    let mut multi_idx = vec![0usize; values.shape.len()];
    let value_count: usize = values.shape.iter().product();

    macro_rules! scatter_variant {
        ($variant:ident, $ty:ty, $zero:expr) => {{
            let mut out: Vec<$ty> = vec![$zero; total];
            for _ in 0..value_count {
                let mut flat_dest = 0usize;
                for (axis, i) in multi_idx.iter().enumerate() {
                    flat_dest += (region_start[axis] + i) * out_strides[axis];
                }
                out[flat_dest] = values.get(&multi_idx) as $ty;
                increment_index(&mut multi_idx, &values.shape);
            }
            CpuBuffer::$variant(out)
        }};
    }

    let new_buffer = match &*values.buffer {
        CpuBuffer::F32(_) => scatter_variant!(F32, f32, 0.0f32),
        CpuBuffer::F64(_) => scatter_variant!(F64, f64, 0.0f64),
        CpuBuffer::U8(_) => scatter_variant!(U8, u8, 0u8),
        CpuBuffer::U32(_) => scatter_variant!(U32, u32, 0u32),
        CpuBuffer::I64(_) => scatter_variant!(I64, i64, 0i64),
        CpuBuffer::F16(_) => {
            let mut out: Vec<f16> = vec![f16::from_f64(0.0); total];
            for _ in 0..value_count {
                let mut flat_dest = 0usize;
                for (axis, i) in multi_idx.iter().enumerate() {
                    flat_dest += (region_start[axis] + i) * out_strides[axis];
                }
                out[flat_dest] = f16::from_f64(values.get(&multi_idx));
                increment_index(&mut multi_idx, &values.shape);
            }
            CpuBuffer::F16(out)
        }
        CpuBuffer::BF16(_) => {
            let mut out: Vec<bf16> = vec![bf16::from_f64(0.0); total];
            for _ in 0..value_count {
                let mut flat_dest = 0usize;
                for (axis, i) in multi_idx.iter().enumerate() {
                    flat_dest += (region_start[axis] + i) * out_strides[axis];
                }
                out[flat_dest] = bf16::from_f64(values.get(&multi_idx));
                increment_index(&mut multi_idx, &values.shape);
            }
            CpuBuffer::BF16(out)
        }
        CpuBuffer::Q8_0(_) => panic!("scatter_into_zeros not supported on Q8_0 buffer"),
    };

    CpuStorage::from_contiguous(new_buffer, original_shape.to_vec())
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    /// `storage_2x3`.
    fn storage_2x3() -> CpuStorage {
        CpuStorage::from_contiguous(
            CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
        )
    }

    #[test]
    /// `from_contiguous_has_expected_shape_and_strides`.
    fn from_contiguous_has_expected_shape_and_strides() {
        let s = storage_2x3();
        assert_eq!(s.shape, vec![2, 3]);
        assert_eq!(s.strides, stride::contiguous_strides(&[2, 3]));
    }

    #[test]
    /// `reshape_contiguous_shares_buffer_and_gets_new_id`.
    fn reshape_contiguous_shares_buffer_and_gets_new_id() {
        let s = storage_2x3();
        let strong_count_before = Arc::strong_count(&s.buffer);
        let r = s.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape, vec![3, 2]);
        assert!(Arc::ptr_eq(&s.buffer, &r.buffer));
        assert_eq!(Arc::strong_count(&s.buffer), strong_count_before + 1);
        assert_ne!(s.id, r.id);
    }

    #[test]
    /// `reshape_non_contiguous_materializes_then_reshapes`.
    fn reshape_non_contiguous_materializes_then_reshapes() {
        let s = storage_2x3();
        let t = s.transpose(0, 1).unwrap(); // [3,2], non-contiguous
        // Reshape a non-contiguous storage: must NOT share the original
        // buffer's Arc (a materialized copy is required).
        let r = t.reshape(&[6]).unwrap();
        assert!(!Arc::ptr_eq(&t.buffer, &r.buffer));
        assert_eq!(r.shape, vec![6]);
        // Values should match manual transposed-read order:
        // transposed [3,2] view of original [2,3] = [[1,4],[2,5],[3,6]]
        // flattened -> [1,4,2,5,3,6]
        if let CpuBuffer::F32(v) = &*r.buffer {
            assert_eq!(v, &vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        } else {
            panic!("expected F32 buffer");
        }
    }

    #[test]
    /// `transpose_shares_buffer_and_swaps_shape_strides`.
    fn transpose_shares_buffer_and_swaps_shape_strides() {
        let s = storage_2x3();
        let strong_count_before = Arc::strong_count(&s.buffer);
        let t = s.transpose(0, 1).unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        assert!(Arc::ptr_eq(&s.buffer, &t.buffer));
        assert_eq!(Arc::strong_count(&s.buffer), strong_count_before + 1);
        assert!(!stride::is_contiguous(&t.shape, &t.strides));
        assert_ne!(s.id, t.id);
    }

    #[test]
    /// `transposed_view_reads_correct_values_without_contiguous_call`.
    fn transposed_view_reads_correct_values_without_contiguous_call() {
        let s = storage_2x3(); // [[1,2,3],[4,5,6]]
        let t = s.transpose(0, 1).unwrap(); // [[1,4],[2,5],[3,6]]
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1]), 4.0);
        assert_eq!(t.get(&[1, 0]), 2.0);
        assert_eq!(t.get(&[1, 1]), 5.0);
        assert_eq!(t.get(&[2, 0]), 3.0);
        assert_eq!(t.get(&[2, 1]), 6.0);
    }

    #[test]
    /// `broadcast_as_expands_and_shares_buffer`.
    fn broadcast_as_expands_and_shares_buffer() {
        let s = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
        let strong_count_before = Arc::strong_count(&s.buffer);
        let b = s.broadcast_as(&[4, 3]).unwrap();
        assert_eq!(b.shape, vec![4, 3]);
        assert!(Arc::ptr_eq(&s.buffer, &b.buffer));
        assert_eq!(Arc::strong_count(&s.buffer), strong_count_before + 1);
        // Expanded axis (axis 0, size 1 -> 4) must have stride 0.
        assert_eq!(b.strides[0], 0);
        // Non-expanded axis keeps its original stride.
        assert_eq!(b.strides[1], s.strides[1]);
        assert_ne!(s.id, b.id);
        // Reading through the broadcast view produces correct values.
        for row in 0..4 {
            assert_eq!(b.get(&[row, 0]), 1.0);
            assert_eq!(b.get(&[row, 1]), 2.0);
            assert_eq!(b.get(&[row, 2]), 3.0);
        }
    }

    #[test]
    /// `narrow_contiguous_shares_buffer_and_slices_correct_values`.
    fn narrow_contiguous_shares_buffer_and_slices_correct_values() {
        // [3,2] storage: [[1,4],[2,5],[3,6]]
        let s = CpuStorage::from_contiguous(
            CpuBuffer::F32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
            vec![3, 2],
        );
        let strong_count_before = Arc::strong_count(&s.buffer);
        let n = s.narrow(0, 1, 1).unwrap();
        assert_eq!(n.shape, vec![1, 2]);
        assert!(Arc::ptr_eq(&s.buffer, &n.buffer));
        assert_eq!(Arc::strong_count(&s.buffer), strong_count_before + 1);
        assert_eq!(n.get(&[0, 0]), 2.0);
        assert_eq!(n.get(&[0, 1]), 5.0);
        assert_ne!(s.id, n.id);
    }

    #[test]
    /// `narrow_on_transposed_view_reads_correct_values_without_materializing`.
    fn narrow_on_transposed_view_reads_correct_values_without_materializing() {
        let s = storage_2x3(); // [[1,2,3],[4,5,6]]
        let t = s.transpose(0, 1).unwrap(); // [[1,4],[2,5],[3,6]], non-contiguous
        let n = t.narrow(0, 1, 1).unwrap(); // row 1 of the transposed view -> [2,5]
        // Proves no materialization occurred: the narrowed view still shares
        // the transposed view's own Arc<CpuBuffer>.
        assert!(Arc::ptr_eq(&t.buffer, &n.buffer));
        assert_eq!(n.shape, vec![1, 2]);
        assert_eq!(n.get(&[0, 0]), 2.0);
        assert_eq!(n.get(&[0, 1]), 5.0);
    }

    #[test]
    /// `narrow_out_of_bounds_length_errors`.
    fn narrow_out_of_bounds_length_errors() {
        let s = storage_2x3();
        let result = s.narrow(0, 1, 2); // start=1, len=2 -> needs shape[0] >= 3, but it's 2
        assert!(result.is_err());
    }

    #[test]
    /// `narrow_dim_out_of_range_errors`.
    fn narrow_dim_out_of_range_errors() {
        let s = storage_2x3();
        let result = s.narrow(5, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    /// `narrow_boundary_values_succeed`.
    fn narrow_boundary_values_succeed() {
        let s = storage_2x3(); // shape [2,3]
        // Full-length narrow (a no-op in effect).
        let full = s.narrow(1, 0, 3).unwrap();
        assert_eq!(full.shape, vec![2, 3]);
        assert_eq!(full.get(&[0, 0]), 1.0);
        assert_eq!(full.get(&[1, 2]), 6.0);

        // start + len == shape[dim] exactly.
        let edge = s.narrow(1, 1, 2).unwrap();
        assert_eq!(edge.shape, vec![2, 2]);
        assert_eq!(edge.get(&[0, 0]), 2.0);
        assert_eq!(edge.get(&[0, 1]), 3.0);
        assert_eq!(edge.get(&[1, 0]), 5.0);
        assert_eq!(edge.get(&[1, 1]), 6.0);
    }

    #[test]
    /// `tensor_id_never_repeats_across_many_calls`.
    fn tensor_id_never_repeats_across_many_calls() {
        let mut ids = hashbrown::HashSet::new();
        for _ in 0..1000 {
            let id = TensorId::next();
            assert!(ids.insert(id), "TensorId::next() produced a duplicate");
        }
    }

    #[test]
    /// `scatter_into_zeros_partial_overlap_writes_only_target_region`.
    fn scatter_into_zeros_partial_overlap_writes_only_target_region() {
        let values = CpuStorage::from_contiguous(CpuBuffer::F32(vec![7.0, 8.0, 9.0]), vec![1, 3]);
        let result = scatter_into_zeros(&[4, 3], &[1, 0], &values);
        assert_eq!(result.shape, vec![4, 3]);
        for col in 0..3 {
            assert_eq!(result.get(&[1, col]), values.get(&[0, col]));
        }
        for row in [0usize, 2, 3] {
            for col in 0..3 {
                assert_eq!(result.get(&[row, col]), 0.0);
            }
        }
    }

    #[test]
    /// `scatter_into_zeros_full_overlap_matches_values_exactly`.
    fn scatter_into_zeros_full_overlap_matches_values_exactly() {
        let values =
            CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
        let result = scatter_into_zeros(&[2, 2], &[0, 0], &values);
        assert_eq!(result.shape, vec![2, 2]);
        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(result.get(&[row, col]), values.get(&[row, col]));
            }
        }
    }

    #[test]
    /// `scatter_into_zeros_returns_fresh_buffer_not_sharing_values_rc`.
    fn scatter_into_zeros_returns_fresh_buffer_not_sharing_values_rc() {
        let values = CpuStorage::from_contiguous(CpuBuffer::F32(vec![7.0, 8.0, 9.0]), vec![1, 3]);
        let result = scatter_into_zeros(&[4, 3], &[1, 0], &values);
        assert!(!Arc::ptr_eq(&values.buffer, &result.buffer));
    }

    #[test]
    /// `scatter_into_zeros_1d_case`.
    fn scatter_into_zeros_1d_case() {
        let values = CpuStorage::from_contiguous(CpuBuffer::F32(vec![9.0, 10.0]), vec![2]);
        let result = scatter_into_zeros(&[5], &[2], &values);
        assert_eq!(result.shape, vec![5]);
        let expected = [0.0, 0.0, 9.0, 10.0, 0.0];
        for (i, exp) in expected.iter().enumerate() {
            assert_eq!(result.get(&[i]), *exp);
        }
    }
}
