//! `NativeBuffer`/`NativeStorage`: the `Arc`-backed, `TensorId`-tagged data
//! structure every later op builds on.
//!
//! `NativeStorage` flows as an immutable, cheaply-cloned value: the `Rc`
//! clone is cheap, and view operations (`reshape`/`transpose`/`broadcast_as`)
//! never allocate a new buffer when the source is already contiguous — they
//! construct new shape/stride/offset metadata sharing the same `Arc<NativeBuffer>`.
//! This is NOT the `NativeVar` mutation boundary (a separate later plan);
//! nothing in this file mutates a `NativeBuffer` in place.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use half::{bf16, f16};
use kindle_core::prelude::Error;
use kindle_core::prelude::Result;

use crate::stride;

/// A monotonic identity tag for a `NativeStorage` value.
///
/// Backed by a global `AtomicU64` counter (never derived from a pointer
/// address, per the anti-pattern of using `Rc` pointer identity — pointer
/// reuse after drop is a real, hard-to-reproduce bug class). Two
/// independently constructed `TensorId`s are never equal, even after many
/// calls to `next()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct TensorId(u64);

/// Auto-generated documentation for NEXT_TENSOR_ID.
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
/// All 7 `KindleDType` variants are reserved as enum shape per the
/// project's "Resolving the Deferred Gray Areas" decision. Only `F32` needs
/// real arithmetic elsewhere in this phase; the other variants exist as
/// data-holding shapes since `storage.rs` itself doesn't perform arithmetic,
/// only shape/stride bookkeeping.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_0 {
    pub(crate) d: half::f16,
    pub(crate) qs: [i8; 32],
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
/// Native CUDA device buffer.
pub struct NativeCudaBuffer {
    pub(crate) len: usize,
    pub(crate) data: alloc::sync::Arc<cudarc::driver::CudaSlice<u8>>,
    pub(crate) device: alloc::sync::Arc<cudarc::driver::CudaContext>,
    pub(crate) device_id: usize,
}

#[cfg(feature = "cuda")]
impl PartialEq for NativeCudaBuffer {
    /// Auto-generated documentation for eq.
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && alloc::sync::Arc::ptr_eq(&self.data, &other.data)
    }
}

#[cfg(feature = "cuda")]
impl Clone for NativeCudaBuffer {
    /// Auto-generated documentation for clone.
    fn clone(&self) -> Self {
        NativeCudaBuffer {
            len: self.len,
            data: self.data.clone(),
            device: self.device.clone(),
            device_id: self.device_id,
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[derive(Debug, Clone, PartialEq)]
/// Stub CUDA buffer for non-CUDA builds.
pub struct NativeCudaBuffer {
    pub(crate) len: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Metal device buffer.
pub struct NativeMetalBuffer {
    pub(crate) len: usize,
}

#[derive(Debug, Clone, PartialEq)]
/// Auto-generated documentation for NativeBuffer.
pub enum NativeBuffer {
    /// Auto-generated documentation for F32.
    F32(Vec<f32>),
    /// Auto-generated documentation for F64.
    F64(Vec<f64>),
    /// Auto-generated documentation for U8.
    U8(Vec<u8>),
    /// Auto-generated documentation for U32.
    U32(Vec<u32>),
    /// Auto-generated documentation for I64.
    I64(Vec<i64>),
    /// Auto-generated documentation for F16.
    F16(Vec<f16>),
    /// Auto-generated documentation for BF16.
    BF16(Vec<bf16>),
    /// Auto-generated documentation for Q8_0.
    Q8_0(Vec<BlockQ8_0>),
    /// Auto-generated documentation for Cuda.
    Cuda(NativeCudaBuffer),
    /// Auto-generated documentation for Metal.
    Metal(NativeMetalBuffer),
}

impl NativeBuffer {
    /// Total number of scalar elements held by this buffer, regardless of
    /// dtype variant.
    pub fn len(&self) -> usize {
        match self {
            NativeBuffer::F32(v) => v.len(),
            NativeBuffer::F64(v) => v.len(),
            NativeBuffer::U8(v) => v.len(),
            NativeBuffer::U32(v) => v.len(),
            NativeBuffer::I64(v) => v.len(),
            NativeBuffer::F16(v) => v.len(),
            NativeBuffer::BF16(v) => v.len(),
            NativeBuffer::Q8_0(v) => v.len() * 32,
            NativeBuffer::Cuda(b) => b.len,
            NativeBuffer::Metal(b) => b.len,
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
                NativeBuffer::F32(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                }
                NativeBuffer::F64(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                }
                NativeBuffer::U8(v) => v.as_slice(),
                NativeBuffer::U32(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                }
                NativeBuffer::I64(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8)
                }
                NativeBuffer::F16(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
                }
                NativeBuffer::BF16(v) => {
                    core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
                }
                NativeBuffer::Q8_0(v) => core::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * core::mem::size_of::<BlockQ8_0>(),
                ),
                NativeBuffer::Cuda(_) => panic!("as_bytes not supported on CUDA buffer"),
                NativeBuffer::Metal(_) => panic!("as_bytes not supported on Metal buffer"),
            }
        }
    }

    /// Read the scalar at flat buffer index `i` as an `f64`, regardless of
    /// dtype variant. Used by strided-index resolution in
    /// `NativeStorage::get`.
    fn get_f64(&self, i: usize) -> f64 {
        match self {
            NativeBuffer::F32(v) => v[i] as f64,
            NativeBuffer::F64(v) => v[i],
            NativeBuffer::U8(v) => v[i] as f64,
            NativeBuffer::U32(v) => v[i] as f64,
            NativeBuffer::I64(v) => v[i] as f64,
            NativeBuffer::F16(v) => v[i].to_f64(),
            NativeBuffer::BF16(v) => v[i].to_f64(),
            NativeBuffer::Q8_0(_) => {
                panic!("get_f64 not supported directly on Q8_0 quantized buffer")
            }
            NativeBuffer::Cuda(_) => panic!("get_f64 not supported directly on CUDA buffer"),
            NativeBuffer::Metal(_) => panic!("get_f64 not supported directly on Metal buffer"),
        }
    }
}

/// A strided, `Arc`-backed view into a `NativeBuffer`.
///
/// `NativeStorage` is `Clone`-able cheaply: cloning only clones the `Rc`
/// pointer (increments the strong count) plus small `Vec<usize>` shape/stride
/// metadata, never the underlying buffer contents.
#[derive(Debug, Clone)]
pub struct NativeStorage {
    pub(crate) buffer: Arc<NativeBuffer>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) id: TensorId,
}

impl NativeStorage {
    /// Build a `NativeStorage` from a contiguous (row-major) buffer and
    /// shape. Strides are computed via `stride::contiguous_strides`.
    pub fn from_contiguous(data: NativeBuffer, shape: Vec<usize>) -> Self {
        let strides = stride::contiguous_strides(&shape);
        NativeStorage {
            buffer: Arc::new(data),
            shape,
            strides,
            offset: 0,
            id: TensorId::next(),
        }
    }

    /// Build a fresh, contiguous, all-ones `NativeStorage` with the same
    /// shape and dtype variant as `other`. Used by `tape::backward()` to
    /// seed the loss tensor's gradient before walking the tape.
    pub fn ones_like(other: &NativeStorage) -> Self {
        let total: usize = other.shape.iter().product();

        #[cfg(feature = "cuda")]
        if let NativeBuffer::Cuda(b) = &*other.buffer {
            let stream = b.device.default_stream();
            let h_data = vec![1.0f32; total];
            let h_bytes: &[u8] = bytemuck::cast_slice(&h_data);
            let mut dev_data = stream.alloc_zeros::<u8>(total * 4).unwrap();
            stream.memcpy_htod(h_bytes, &mut dev_data).unwrap();
            let cuda_buf = NativeBuffer::Cuda(NativeCudaBuffer {
                len: total,
                data: Arc::new(dev_data),
                device: b.device.clone(),
                device_id: b.device_id,
            });
            return NativeStorage::from_contiguous(cuda_buf, other.shape.clone());
        }

        let new_buffer = match &*other.buffer {
            NativeBuffer::F32(_) => NativeBuffer::F32(vec![1.0f32; total]),
            NativeBuffer::F64(_) => NativeBuffer::F64(vec![1.0f64; total]),
            NativeBuffer::U8(_) => NativeBuffer::U8(vec![1u8; total]),
            NativeBuffer::U32(_) => NativeBuffer::U32(vec![1u32; total]),
            NativeBuffer::I64(_) => NativeBuffer::I64(vec![1i64; total]),
            NativeBuffer::F16(_) => NativeBuffer::F16(vec![half::f16::from_f64(1.0); total]),
            NativeBuffer::BF16(_) => NativeBuffer::BF16(vec![half::bf16::from_f64(1.0); total]),
            NativeBuffer::Q8_0(_) => panic!("ones_like not supported on Q8_0 buffer"),
            NativeBuffer::Cuda(_) => panic!("ones_like CUDA unreachable"),
            NativeBuffer::Metal(_) => panic!("ones_like not supported on Metal buffer"),
        };

        NativeStorage::from_contiguous(new_buffer, other.shape.clone())
    }


    /// Resolve a logical multi-index through `self.strides`/`self.offset`
    /// into the underlying `NativeBuffer`, returning the value as `f64`.
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
    /// same `Arc<NativeBuffer>`, no allocation) when `self` is already
    /// contiguous; otherwise materializes a contiguous copy first, then
    /// recurses.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        if stride::is_contiguous(&self.shape, &self.strides) {
            Ok(NativeStorage {
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
        Ok(NativeStorage {
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

        Ok(NativeStorage {
            buffer: self.buffer.clone(),
            shape: target_shape.to_vec(),
            strides: new_strides,
            offset: self.offset,
            id: TensorId::next(),
        })
    }

    /// Narrow dimension `dim` to the half-open range `[start, start + len)`.
    /// Metadata-only: shares the same `Arc<NativeBuffer>`, keeps `strides`
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
        Ok(NativeStorage {
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
    fn contiguous(&self) -> Self {
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
                NativeBuffer::$variant(out)
            }};
        }

        let new_buffer = match &*self.buffer {
            NativeBuffer::F32(_) => materialize!(F32, f32),
            NativeBuffer::F64(_) => materialize!(F64, f64),
            NativeBuffer::U8(_) => materialize!(U8, u8),
            NativeBuffer::U32(_) => materialize!(U32, u32),
            NativeBuffer::I64(_) => materialize!(I64, i64),
            NativeBuffer::F16(_) => {
                let mut out: Vec<f16> = Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(f16::from_f64(self.get(&multi_idx)));
                    increment_index(&mut multi_idx, &self.shape);
                }
                NativeBuffer::F16(out)
            }
            NativeBuffer::BF16(_) => {
                let mut out: Vec<bf16> = Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(bf16::from_f64(self.get(&multi_idx)));
                    increment_index(&mut multi_idx, &self.shape);
                }
                NativeBuffer::BF16(out)
            }
            NativeBuffer::Q8_0(_) => panic!("materialize not supported on Q8_0 buffer"),
            NativeBuffer::Cuda(_) => panic!("materialize not supported on CUDA buffer"),
            NativeBuffer::Metal(_) => panic!("materialize not supported on Metal buffer"),
        };

        NativeStorage::from_contiguous(new_buffer, self.shape.clone())
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

/// Build a zero-filled, freshly-allocated, contiguous `NativeStorage` of
/// `original_shape` (dtype-matched to `values`), then copy `values`'s data
/// into the sub-region starting at `region_start` (one offset per axis).
/// Every position outside that sub-region is left exactly zero.
///
/// This is the shared zero-pad-scatter backward primitive for `narrow`/
/// `slice`: `grad_out` (shaped like the narrowed region) is scattered back
/// into a zero buffer shaped like the original (pre-narrow) tensor, at the
/// same offset the forward narrow started from. It is a module-level free
/// function (not a `NativeStorage` method) because it constructs a NEW
/// storage from two independent shape/value inputs, rather than adjusting
/// `self`'s own metadata.
pub(crate) fn scatter_into_zeros(
    original_shape: &[usize],
    region_start: &[usize],
    values: &NativeStorage,
) -> NativeStorage {
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
            NativeBuffer::$variant(out)
        }};
    }

    let new_buffer = match &*values.buffer {
        NativeBuffer::F32(_) => scatter_variant!(F32, f32, 0.0f32),
        NativeBuffer::F64(_) => scatter_variant!(F64, f64, 0.0f64),
        NativeBuffer::U8(_) => scatter_variant!(U8, u8, 0u8),
        NativeBuffer::U32(_) => scatter_variant!(U32, u32, 0u32),
        NativeBuffer::I64(_) => scatter_variant!(I64, i64, 0i64),
        NativeBuffer::F16(_) => {
            let mut out: Vec<f16> = vec![f16::from_f64(0.0); total];
            for _ in 0..value_count {
                let mut flat_dest = 0usize;
                for (axis, i) in multi_idx.iter().enumerate() {
                    flat_dest += (region_start[axis] + i) * out_strides[axis];
                }
                out[flat_dest] = f16::from_f64(values.get(&multi_idx));
                increment_index(&mut multi_idx, &values.shape);
            }
            NativeBuffer::F16(out)
        }
        NativeBuffer::BF16(_) => {
            let mut out: Vec<bf16> = vec![bf16::from_f64(0.0); total];
            for _ in 0..value_count {
                let mut flat_dest = 0usize;
                for (axis, i) in multi_idx.iter().enumerate() {
                    flat_dest += (region_start[axis] + i) * out_strides[axis];
                }
                out[flat_dest] = bf16::from_f64(values.get(&multi_idx));
                increment_index(&mut multi_idx, &values.shape);
            }
            NativeBuffer::BF16(out)
        }
        NativeBuffer::Cuda(_) => panic!("scatter_into_zeros not supported on CUDA buffer"),
        NativeBuffer::Metal(_) => panic!("scatter_into_zeros not supported on Metal buffer"),
        NativeBuffer::Q8_0(_) => panic!("scatter_into_zeros not supported on Q8_0 buffer"),
    };

    NativeStorage::from_contiguous(new_buffer, original_shape.to_vec())
}

#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use super::*;

    /// Auto-generated documentation for storage_2x3.
    fn storage_2x3() -> NativeStorage {
        NativeStorage::from_contiguous(
            NativeBuffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
        )
    }

    #[test]
    /// Auto-generated documentation for from_contiguous_has_expected_shape_and_strides.
    fn from_contiguous_has_expected_shape_and_strides() {
        let s = storage_2x3();
        assert_eq!(s.shape, vec![2, 3]);
        assert_eq!(s.strides, stride::contiguous_strides(&[2, 3]));
    }

    #[test]
    /// Auto-generated documentation for reshape_contiguous_shares_buffer_and_gets_new_id.
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
    /// Auto-generated documentation for reshape_non_contiguous_materializes_then_reshapes.
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
        if let NativeBuffer::F32(v) = &*r.buffer {
            assert_eq!(v, &vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        } else {
            panic!("expected F32 buffer");
        }
    }

    #[test]
    /// Auto-generated documentation for transpose_shares_buffer_and_swaps_shape_strides.
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
    /// Auto-generated documentation for transposed_view_reads_correct_values_without_contiguous_call.
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
    /// Auto-generated documentation for broadcast_as_expands_and_shares_buffer.
    fn broadcast_as_expands_and_shares_buffer() {
        let s = NativeStorage::from_contiguous(NativeBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
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
    /// Auto-generated documentation for narrow_contiguous_shares_buffer_and_slices_correct_values.
    fn narrow_contiguous_shares_buffer_and_slices_correct_values() {
        // [3,2] storage: [[1,4],[2,5],[3,6]]
        let s = NativeStorage::from_contiguous(
            NativeBuffer::F32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
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
    /// Auto-generated documentation for narrow_on_transposed_view_reads_correct_values_without_materializing.
    fn narrow_on_transposed_view_reads_correct_values_without_materializing() {
        let s = storage_2x3(); // [[1,2,3],[4,5,6]]
        let t = s.transpose(0, 1).unwrap(); // [[1,4],[2,5],[3,6]], non-contiguous
        let n = t.narrow(0, 1, 1).unwrap(); // row 1 of the transposed view -> [2,5]
        // Proves no materialization occurred: the narrowed view still shares
        // the transposed view's own Arc<NativeBuffer>.
        assert!(Arc::ptr_eq(&t.buffer, &n.buffer));
        assert_eq!(n.shape, vec![1, 2]);
        assert_eq!(n.get(&[0, 0]), 2.0);
        assert_eq!(n.get(&[0, 1]), 5.0);
    }

    #[test]
    /// Auto-generated documentation for narrow_out_of_bounds_length_errors.
    fn narrow_out_of_bounds_length_errors() {
        let s = storage_2x3();
        let result = s.narrow(0, 1, 2); // start=1, len=2 -> needs shape[0] >= 3, but it's 2
        assert!(result.is_err());
    }

    #[test]
    /// Auto-generated documentation for narrow_dim_out_of_range_errors.
    fn narrow_dim_out_of_range_errors() {
        let s = storage_2x3();
        let result = s.narrow(5, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    /// Auto-generated documentation for narrow_boundary_values_succeed.
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
    /// Auto-generated documentation for tensor_id_never_repeats_across_many_calls.
    fn tensor_id_never_repeats_across_many_calls() {
        let mut ids = hashbrown::HashSet::new();
        for _ in 0..1000 {
            let id = TensorId::next();
            assert!(ids.insert(id), "TensorId::next() produced a duplicate");
        }
    }

    #[test]
    /// Auto-generated documentation for scatter_into_zeros_partial_overlap_writes_only_target_region.
    fn scatter_into_zeros_partial_overlap_writes_only_target_region() {
        let values =
            NativeStorage::from_contiguous(NativeBuffer::F32(vec![7.0, 8.0, 9.0]), vec![1, 3]);
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
    /// Auto-generated documentation for scatter_into_zeros_full_overlap_matches_values_exactly.
    fn scatter_into_zeros_full_overlap_matches_values_exactly() {
        let values =
            NativeStorage::from_contiguous(NativeBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
        let result = scatter_into_zeros(&[2, 2], &[0, 0], &values);
        assert_eq!(result.shape, vec![2, 2]);
        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(result.get(&[row, col]), values.get(&[row, col]));
            }
        }
    }

    #[test]
    /// Auto-generated documentation for scatter_into_zeros_returns_fresh_buffer_not_sharing_values_rc.
    fn scatter_into_zeros_returns_fresh_buffer_not_sharing_values_rc() {
        let values =
            NativeStorage::from_contiguous(NativeBuffer::F32(vec![7.0, 8.0, 9.0]), vec![1, 3]);
        let result = scatter_into_zeros(&[4, 3], &[1, 0], &values);
        assert!(!Arc::ptr_eq(&values.buffer, &result.buffer));
    }

    #[test]
    /// Auto-generated documentation for scatter_into_zeros_1d_case.
    fn scatter_into_zeros_1d_case() {
        let values = NativeStorage::from_contiguous(NativeBuffer::F32(vec![9.0, 10.0]), vec![2]);
        let result = scatter_into_zeros(&[5], &[2], &values);
        assert_eq!(result.shape, vec![5]);
        let expected = [0.0, 0.0, 9.0, 10.0, 0.0];
        for (i, exp) in expected.iter().enumerate() {
            assert_eq!(result.get(&[i]), *exp);
        }
    }
}
