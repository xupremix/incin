//! `NativeBuffer`/`NativeStorage`: the `Rc`-backed, `TensorId`-tagged data
//! structure every later op builds on.
//!
//! `NativeStorage` flows as an immutable, cheaply-cloned value: the `Rc`
//! clone is cheap, and view operations (`reshape`/`transpose`/`broadcast_as`)
//! never allocate a new buffer when the source is already contiguous — they
//! construct new shape/stride/offset metadata sharing the same `Rc<NativeBuffer>`.
//! This is NOT the `NativeVar` mutation boundary (a separate later plan);
//! nothing in this file mutates a `NativeBuffer` in place.

use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use half::{bf16, f16};
use kindle_core::err::Error;
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
pub struct TensorId(u64);

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(0);

impl TensorId {
    /// Allocate a fresh, never-before-seen `TensorId`.
    pub fn next() -> Self {
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
pub enum NativeBuffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
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
        }
    }

    /// True if this buffer holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        }
    }
}

/// A strided, `Rc`-backed view into a `NativeBuffer`.
///
/// `NativeStorage` is `Clone`-able cheaply: cloning only clones the `Rc`
/// pointer (increments the strong count) plus small `Vec<usize>` shape/stride
/// metadata, never the underlying buffer contents.
#[derive(Debug, Clone)]
pub struct NativeStorage {
    pub buffer: Rc<NativeBuffer>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub id: TensorId,
}

impl NativeStorage {
    /// Build a `NativeStorage` from a contiguous (row-major) buffer and
    /// shape. Strides are computed via `stride::contiguous_strides`.
    pub fn from_contiguous(data: NativeBuffer, shape: Vec<usize>) -> Self {
        let strides = stride::contiguous_strides(&shape);
        NativeStorage {
            buffer: Rc::new(data),
            shape,
            strides,
            offset: 0,
            id: TensorId::next(),
        }
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
    /// same `Rc<NativeBuffer>`, no allocation) when `self` is already
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
        };

        NativeStorage::from_contiguous(new_buffer, self.shape.clone())
    }
}

/// Increment a row-major multi-index in place (odometer-style), matching the
/// iteration order `contiguous_strides` assumes.
fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for i in (0..idx.len()).rev() {
        idx[i] += 1;
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn storage_2x3() -> NativeStorage {
        NativeStorage::from_contiguous(
            NativeBuffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
        )
    }

    #[test]
    fn from_contiguous_has_expected_shape_and_strides() {
        let s = storage_2x3();
        assert_eq!(s.shape, vec![2, 3]);
        assert_eq!(s.strides, stride::contiguous_strides(&[2, 3]));
    }

    #[test]
    fn reshape_contiguous_shares_buffer_and_gets_new_id() {
        let s = storage_2x3();
        let strong_count_before = Rc::strong_count(&s.buffer);
        let r = s.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape, vec![3, 2]);
        assert!(Rc::ptr_eq(&s.buffer, &r.buffer));
        assert_eq!(Rc::strong_count(&s.buffer), strong_count_before + 1);
        assert_ne!(s.id, r.id);
    }

    #[test]
    fn reshape_non_contiguous_materializes_then_reshapes() {
        let s = storage_2x3();
        let t = s.transpose(0, 1).unwrap(); // [3,2], non-contiguous
        // Reshape a non-contiguous storage: must NOT share the original
        // buffer's Rc (a materialized copy is required).
        let r = t.reshape(&[6]).unwrap();
        assert!(!Rc::ptr_eq(&t.buffer, &r.buffer));
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
    fn transpose_shares_buffer_and_swaps_shape_strides() {
        let s = storage_2x3();
        let strong_count_before = Rc::strong_count(&s.buffer);
        let t = s.transpose(0, 1).unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        assert!(Rc::ptr_eq(&s.buffer, &t.buffer));
        assert_eq!(Rc::strong_count(&s.buffer), strong_count_before + 1);
        assert!(!stride::is_contiguous(&t.shape, &t.strides));
        assert_ne!(s.id, t.id);
    }

    #[test]
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
    fn broadcast_as_expands_and_shares_buffer() {
        let s = NativeStorage::from_contiguous(NativeBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
        let strong_count_before = Rc::strong_count(&s.buffer);
        let b = s.broadcast_as(&[4, 3]).unwrap();
        assert_eq!(b.shape, vec![4, 3]);
        assert!(Rc::ptr_eq(&s.buffer, &b.buffer));
        assert_eq!(Rc::strong_count(&s.buffer), strong_count_before + 1);
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
    fn tensor_id_never_repeats_across_many_calls() {
        let mut ids = std::collections::HashSet::new();
        for _ in 0..1000 {
            let id = TensorId::next();
            assert!(ids.insert(id), "TensorId::next() produced a duplicate");
        }
    }
}
