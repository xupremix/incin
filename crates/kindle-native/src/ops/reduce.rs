//! `ReductionOps` for `NativeBackend<T, D>`: every method now has a real
//! implementation — `sum_all`/`mean_all`/`sum_dim`/`sum_keepdim` (Phase 1),
//! `mean_dim`/`mean_keepdim`/`max_dim`/`max_keepdim`/`min_dim`/`min_keepdim`/
//! `max_all`/`min_all` (Phase 2, gradcheck-verified backward), and
//! `argmax`/`argmin` (Phase 2, forward-only by structural design). Zero
//! remaining unsupported-backend-operation error stubs.
//!
//! ## Design Notes
//!
//! * `sum_all` / `mean_all` backward: the incoming scalar gradient must be
//!   *broadcast* back to every element of the original shape — the exact
//!   inverse of sum. This is NOT a call to `tape::unbroadcast` (which handles
//!   the opposite direction); instead, the backward closure fills a new
//!   contiguous storage with `grad_scalar / n` (for `mean_all`) or
//!   `grad_scalar` (for `sum_all`) repeated across the original shape.
//!
//! * `sum_dim` / `sum_keepdim` need real implementations even though
//!   PATTERNS.md marks them "stub acceptable" at the public-trait level,
//!   because `tape::unbroadcast` (Plan 02) depends on the same axis-reduce
//!   logic internally. Rather than making tape.rs's private helpers
//!   `pub(crate)` and introducing a dependency, this file carries its own
//!   `sum_axis_keepdim` / `sum_axis_squeeze` helpers — identical in logic to
//!   tape.rs's private versions, independent in scope, so that neither side
//!   regresses the other's tests.
//!
//! * `mean_dim` / `mean_keepdim` are thin wrappers over `sum_axis_squeeze` /
//!   `sum_axis_keepdim`, divided by axis length (forward and backward both).
//!
//! * `max_dim` / `min_dim` / `max_keepdim` / `min_keepdim` / `max_all` /
//!   `min_all` route gradient to exactly one winning element per output
//!   position via `max_axis_with_indices` / `min_axis_with_indices` (strict
//!   `>`/`<` comparison — first-encountered winner on ties, never splitting
//!   or duplicating gradient mass) and the shared `scatter_axis_grad`
//!   backward helper.
//!
//! * `argmax` / `argmin` are forward-only — `kindle-core`'s
//!   `Tensor::argmax`/`argmin` structurally force `G = NoGrad` on their
//!   output regardless of the input's own `G`, so neither method calls
//!   `tape::push` (the one deliberate exception to this file's
//!   every-other-method unconditional-push convention).
//!
//! * Any leftover unimplemented method (there are none as of Phase 2) would
//!   keep returning the typed unsupported-backend-operation error — never a
//!   silent `Ok(t.clone())` placeholder (T-01-15 mitigation).

use kindle_core::prelude::Error;
use kindle_core::prelude::{Backend, DType, ReductionOps, Result};

use crate::NativeBackend;
use crate::ops::elementwise::increment_index;
use crate::storage::{NativeBuffer, NativeStorage};
use crate::stride::contiguous_strides;
use crate::tape::{self, TapeEntry};

// ---------------------------------------------------------------------------
// Internal axis-reduce helpers (independent of tape.rs's private equivalents)
// ---------------------------------------------------------------------------

/// Compute the flat row-major index of `idx` within `shape`.
fn flatten_index(idx: &[usize], shape: &[usize]) -> usize {
    let strides = contiguous_strides(shape);
    idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

/// Inverse of `flatten_index`: recover the multi-index within `shape`
/// corresponding to flat row-major index `flat`. Used by `argmax`/`argmin`
/// to resolve a winning FLAT source index into its coordinate along the
/// reduced axis.
fn unflatten_index(flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = contiguous_strides(shape);
    let mut remaining = flat;
    let mut idx = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        if strides[i] == 0 {
            idx[i] = 0;
        } else {
            idx[i] = remaining / strides[i];
            remaining %= strides[i];
        }
    }
    idx
}

/// Sum-reduce `storage` over `axis`, *keeping* that axis as size 1
/// (e.g. `[4, 3]` over axis 0 → `[1, 3]`).
pub(crate) fn sum_axis_keepdim(storage: &NativeStorage, axis: usize) -> NativeStorage {
    let mut out_shape = storage.shape.clone();
    out_shape[axis] = 1;
    let total: usize = out_shape.iter().product();

    macro_rules! reduce_variant {
        ($variant:ident, $to_ty:expr) => {{
            let mut out = vec![Default::default(); total];
            let mut idx = vec![0usize; storage.shape.len()];
            let src_total: usize = storage.shape.iter().product();
            for _ in 0..src_total {
                let mut out_idx = idx.clone();
                out_idx[axis] = 0;
                let flat_out = flatten_index(&out_idx, &out_shape);
                out[flat_out] += $to_ty(storage.get(&idx));
                increment_index(&mut idx, &storage.shape);
            }
            NativeBuffer::$variant(out)
        }};
    }

    let new_buffer = match &*storage.buffer {
        NativeBuffer::F32(_) => reduce_variant!(F32, |v: f64| v as f32),
        NativeBuffer::F64(_) => reduce_variant!(F64, |v: f64| v),
        NativeBuffer::U8(_) => reduce_variant!(U8, |v: f64| v as u8),
        NativeBuffer::U32(_) => reduce_variant!(U32, |v: f64| v as u32),
        NativeBuffer::I64(_) => reduce_variant!(I64, |v: f64| v as i64),
        NativeBuffer::F16(_) => reduce_variant!(F16, |v: f64| half::f16::from_f64(v)),
        NativeBuffer::BF16(_) => reduce_variant!(BF16, |v: f64| half::bf16::from_f64(v)),
        NativeBuffer::Cuda(_) => {
            #[cfg(feature = "cuda")]
            return crate::ops::cuda_reduce::launch_reduce_op("sum_axis_keepdim", "0.0", "acc = acc + val", storage, axis, true).unwrap();
            #[cfg(not(feature = "cuda"))]
            panic!("CUDA not enabled");
        },
        NativeBuffer::Metal(_) => panic!("sum_axis_keepdim not supported on Metal buffer"),
        NativeBuffer::Q8_0(_) => panic!("sum_axis_keepdim not supported on Q8_0 buffer"),
    };

    NativeStorage::from_contiguous(new_buffer, out_shape)
}

/// Sum-reduce `storage` over `axis`, *removing* that axis from the shape
/// entirely (e.g. `[4, 3]` over axis 0 → `[3]`).
pub(crate) fn sum_axis_squeeze(storage: &NativeStorage, axis: usize) -> NativeStorage {
    let reduced = sum_axis_keepdim(storage, axis);
    let mut new_shape = reduced.shape.clone();
    new_shape.remove(axis);
    // Squeezing a size-1 keepdim result is a pure metadata reshape (no data
    // movement) since the output is already contiguous.
    reduced
        .reshape(&new_shape)
        .expect("squeeze reshape of size-1 keepdim result cannot fail (same element count)")
}

/// Build a contiguous `NativeStorage` of `shape` where every element equals
/// `scalar_value`, matching the dtype variant of `like`. Used by `sum_all` and
/// `mean_all` backward closures to broadcast the incoming scalar gradient back
/// to the full original shape.
fn fill_like(like: &NativeStorage, shape: &[usize], scalar_value: f64) -> NativeStorage {
    let total: usize = shape.iter().product();
    let new_buffer = match &*like.buffer {
        NativeBuffer::F32(_) => NativeBuffer::F32(vec![scalar_value as f32; total]),
        NativeBuffer::F64(_) => NativeBuffer::F64(vec![scalar_value; total]),
        NativeBuffer::U8(_) => NativeBuffer::U8(vec![scalar_value as u8; total]),
        NativeBuffer::U32(_) => NativeBuffer::U32(vec![scalar_value as u32; total]),
        NativeBuffer::I64(_) => NativeBuffer::I64(vec![scalar_value as i64; total]),
        NativeBuffer::F16(_) => NativeBuffer::F16(vec![half::f16::from_f64(scalar_value); total]),
        NativeBuffer::BF16(_) => {
            NativeBuffer::BF16(vec![half::bf16::from_f64(scalar_value); total])
        }
        NativeBuffer::Cuda(b) => {
            #[cfg(feature = "cuda")]
            {
                let device_id = b.device_id;
                let stream = b.device.default_stream();
                let h_data = vec![scalar_value as f32; total];
                let h_bytes: &[u8] = bytemuck::cast_slice(&h_data);
                let mut dev_data = stream.alloc_zeros::<u8>(total * 4).unwrap();
                stream.memcpy_htod(h_bytes, &mut dev_data).unwrap();
                let cuda_buf = NativeBuffer::Cuda(crate::storage::NativeCudaBuffer {
                    len: total,
                    data: alloc::sync::Arc::new(dev_data),
                    device: b.device.clone(),
                    device_id,
                });
                return NativeStorage::from_contiguous(cuda_buf, shape.to_vec());
            }
            #[cfg(not(feature = "cuda"))]
            panic!("CUDA not enabled");
        }
        NativeBuffer::Metal(_) => panic!("fill_like not supported on Metal buffer"),
        NativeBuffer::Q8_0(_) => panic!("fill_like not supported on Q8_0 buffer"),
    };
    NativeStorage::from_contiguous(new_buffer, shape.to_vec())
}

/// Reduce along `axis`, tracking the WINNING flat-index-into-source at each
/// output position (needed for backward's gradient-routing scatter).
/// Ties: strict `>` (not `>=`) naturally picks first-encountered winner
/// during forward iteration (Pitfall 3 mitigation, T-02-07).
fn max_axis_with_indices(storage: &NativeStorage, axis: usize) -> (NativeStorage, Vec<usize>) {
    #[cfg(feature = "cuda")]
    if matches!(&*storage.buffer, NativeBuffer::Cuda(_)) {
        return crate::ops::cuda_reduce::launch_reduce_with_indices_host(
            "max_axis_with_indices", "-INFINITY",
            "if (val > best_val) { best_val = val; best_idx = i; }",
            storage, axis, true
        ).unwrap();
    }

    let mut out_shape = storage.shape.clone();
    out_shape[axis] = 1;
    let out_total: usize = out_shape.iter().product();
    let mut best_val = vec![f64::NEG_INFINITY; out_total];
    let mut best_flat_src_idx = vec![0usize; out_total];

    let mut idx = vec![0usize; storage.shape.len()];
    let src_total: usize = storage.shape.iter().product();
    for _ in 0..src_total {
        let mut out_idx = idx.clone();
        out_idx[axis] = 0;
        let flat_out = flatten_index(&out_idx, &out_shape);
        let v = storage.get(&idx);
        if v > best_val[flat_out] {
            best_val[flat_out] = v;
            best_flat_src_idx[flat_out] = flatten_index(&idx, &storage.shape);
        }
        increment_index(&mut idx, &storage.shape);
    }
    let out = NativeStorage::from_contiguous(
        NativeBuffer::F32(best_val.iter().map(|&v| v as f32).collect()),
        out_shape,
    );
    (out, best_flat_src_idx)
}

/// Mirror of `max_axis_with_indices`, seeded with `f64::INFINITY` and a
/// strict `<` comparison — same first-encountered-winner convention.
fn min_axis_with_indices(storage: &NativeStorage, axis: usize) -> (NativeStorage, Vec<usize>) {
    #[cfg(feature = "cuda")]
    if matches!(&*storage.buffer, NativeBuffer::Cuda(_)) {
        return crate::ops::cuda_reduce::launch_reduce_with_indices_host(
            "min_axis_with_indices", "INFINITY",
            "if (val < best_val) { best_val = val; best_idx = i; }",
            storage, axis, true
        ).unwrap();
    }

    let mut out_shape = storage.shape.clone();
    out_shape[axis] = 1;
    let out_total: usize = out_shape.iter().product();
    let mut best_val = vec![f64::INFINITY; out_total];
    let mut best_flat_src_idx = vec![0usize; out_total];

    let mut idx = vec![0usize; storage.shape.len()];
    let src_total: usize = storage.shape.iter().product();
    for _ in 0..src_total {
        let mut out_idx = idx.clone();
        out_idx[axis] = 0;
        let flat_out = flatten_index(&out_idx, &out_shape);
        let v = storage.get(&idx);
        if v < best_val[flat_out] {
            best_val[flat_out] = v;
            best_flat_src_idx[flat_out] = flatten_index(&idx, &storage.shape);
        }
        increment_index(&mut idx, &storage.shape);
    }
    let out = NativeStorage::from_contiguous(
        NativeBuffer::F32(best_val.iter().map(|&v| v as f32).collect()),
        out_shape,
    );
    (out, best_flat_src_idx)
}

/// Backward helper shared by `max_dim`/`min_dim`/`max_keepdim`/`min_keepdim`:
/// build a zero-filled buffer sized to `original_shape`, then scatter
/// `grad_out`'s per-output-position value into ONLY the recorded winning
/// flat index for that position (T-02-08 mitigation — reuses `flatten_index`
/// rather than hand-deriving a new index-resolution formula).
fn scatter_axis_grad(
    grad_out: &NativeStorage,
    winning_flat_src_idx: &[usize],
    original_shape: &[usize],
) -> NativeStorage {
    let total: usize = original_shape.iter().product();
    let mut vals = vec![0.0f32; total];
    let out_total: usize = grad_out.shape.iter().product();
    let mut out_idx = vec![0usize; grad_out.shape.len()];
    for flat_out in 0..out_total {
        let g = grad_out.get(&out_idx);
        vals[winning_flat_src_idx[flat_out]] = g as f32;
        increment_index(&mut out_idx, &grad_out.shape);
    }
    NativeStorage::from_contiguous(NativeBuffer::F32(vals), original_shape.to_vec())
}

// ---------------------------------------------------------------------------
// ReductionOps impl
// ---------------------------------------------------------------------------

impl<T: DType, D: kindle_core::prelude::Device> ReductionOps<Self> for NativeBackend<T, D> {
    /// Sum every element of `t` into a single-element scalar storage (shape
    /// `[]`). Pushes a `TapeEntry` whose backward broadcasts the incoming
    /// scalar gradient uniformly back across `t`'s original shape.
    fn sum_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        #[cfg(feature = "cuda")]
        if matches!(&*t.buffer, NativeBuffer::Cuda(_)) {
            let mut acc = t.clone();
            for axis in (0..t.shape.len()).rev() {
                acc = crate::ops::cuda_reduce::launch_reduce_op(
                    "sum_all", "0.0", "acc = acc + val", &acc, 0, false
                ).unwrap();
            }
            let t_clone = t.clone();
            let (t_id, out_id) = (t.id, acc.id);
            tape::push(TapeEntry {
                output_id: out_id,
                input_ids: vec![t_id],
                backward: Box::new(move |grad_out: &NativeStorage| {
                    let scalar_grad = if let NativeBuffer::Cuda(b) = &*grad_out.buffer {
                        let stream = b.device.default_stream();
                        let mut h = vec![0f32; 1];
                        unsafe { let dev_f32 = b.data.transmute::<f32>(1).unwrap(); let v = stream.clone_dtoh(&dev_f32).unwrap(); h[0] = v[0]; }
                        h[0] as f64
                    } else { grad_out.get(&vec![0usize; grad_out.shape.len()]) };
                    vec![fill_like(&t_clone, &t_clone.shape, scalar_grad)]
                }),
            });
            return Ok(acc);
        }

        let total: usize = t.shape.iter().product();
        let mut idx = vec![0usize; t.shape.len()];
        let mut sum = 0f64;
        for _ in 0..total {
            sum += t.get(&idx);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(vec![sum as f32]), vec![]);

        let original_shape = t.shape.clone();
        let t_clone = t.clone(); // dtype reference for fill_like
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                // grad_out is a scalar []; broadcast it to every element of
                // the original shape (the backward of sum is "distribute
                // everywhere").
                let scalar_grad = grad_out.get(&vec![0usize; grad_out.shape.len()]);
                vec![fill_like(&t_clone, &original_shape, scalar_grad)]
            }),
        });

        Ok(out)
    }

    /// Mean of every element of `t`. Backward scales the incoming scalar
    /// gradient by `1/n` before broadcasting back to the original shape.
    fn mean_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        #[cfg(feature = "cuda")]
        if matches!(&*t.buffer, NativeBuffer::Cuda(_)) {
            let total = t.shape.iter().product::<usize>();
            let n = total as f64;
            let mut acc = t.clone();
            for _ in 0..t.shape.len() {
                acc = crate::ops::cuda_reduce::launch_reduce_op(
                    "mean_all_sum", "0.0", "acc = acc + val", &acc, 0, false
                ).unwrap();
            }
            // divide by n using elementwise scalar op on GPU
            let inv_n = 1.0 / n.max(1.0);
            let h_n = vec![inv_n as f32];
            let t_clone = t.clone();
            let (t_id, out_id) = (t.id, acc.id);
            tape::push(TapeEntry {
                output_id: out_id,
                input_ids: vec![t_id],
                backward: Box::new(move |grad_out: &NativeStorage| {
                    let scalar_grad = if let NativeBuffer::Cuda(b) = &*grad_out.buffer {
                        let stream = b.device.default_stream();
                        let mut h = vec![0f32; 1];
                        unsafe { let dev_f32 = b.data.transmute::<f32>(1).unwrap(); let v = stream.clone_dtoh(&dev_f32).unwrap(); h[0] = v[0]; }
                        h[0] as f64
                    } else { grad_out.get(&vec![0usize; grad_out.shape.len()]) };
                    let scaled = if n > 0.0 { scalar_grad / n } else { 0.0 };
                    vec![fill_like(&t_clone, &t_clone.shape, scaled)]
                }),
            });
            return Ok(acc);
        }

        let total: usize = t.shape.iter().product();
        let mut idx = vec![0usize; t.shape.len()];
        let mut sum = 0f64;
        for _ in 0..total {
            sum += t.get(&idx);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let mean = if total > 0 { sum / total as f64 } else { 0.0 };
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(vec![mean as f32]), vec![]);

        let original_shape = t.shape.clone();
        let t_clone = t.clone();
        let n = total as f64;
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let scalar_grad = grad_out.get(&vec![0usize; grad_out.shape.len()]);
                // d(mean)/d(x_i) = 1/n for each element.
                let scaled = if n > 0.0 { scalar_grad / n } else { 0.0 };
                vec![fill_like(&t_clone, &original_shape, scaled)]
            }),
        });

        Ok(out)
    }

    /// Maximum over every element of `t`, as a scalar (shape `[]`).
    /// Independent flat iteration (mirrors `sum_all`'s structure, NOT a
    /// reshape-then-axis-reduce composition, per RESEARCH.md Open Question 2).
    /// Backward scatters the incoming scalar gradient to ONLY the single
    /// global winning flat index, zero everywhere else.
    fn max_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut idx = vec![0usize; t.shape.len()];
        let mut best_val = f64::NEG_INFINITY;
        let mut best_flat_idx = 0usize;
        for flat in 0..total {
            let v = t.get(&idx);
            if v > best_val {
                best_val = v;
                best_flat_idx = flat;
            }
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(vec![best_val as f32]), vec![]);

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let scalar_grad = grad_out.get(&vec![0usize; grad_out.shape.len()]);
                let total: usize = original_shape.iter().product();
                let mut vals = vec![0.0f32; total];
                vals[best_flat_idx] = scalar_grad as f32;
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(vals),
                    original_shape.clone(),
                )]
            }),
        });

        Ok(out)
    }

    /// Minimum over every element of `t`, as a scalar (shape `[]`). Mirror of
    /// `max_all` with strict `<` comparison.
    fn min_all<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut idx = vec![0usize; t.shape.len()];
        let mut best_val = f64::INFINITY;
        let mut best_flat_idx = 0usize;
        for flat in 0..total {
            let v = t.get(&idx);
            if v < best_val {
                best_val = v;
                best_flat_idx = flat;
            }
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(vec![best_val as f32]), vec![]);

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let scalar_grad = grad_out.get(&vec![0usize; grad_out.shape.len()]);
                let total: usize = original_shape.iter().product();
                let mut vals = vec![0.0f32; total];
                vals[best_flat_idx] = scalar_grad as f32;
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(vals),
                    original_shape.clone(),
                )]
            }),
        });

        Ok(out)
    }

    /// Sum over `dim`, removing that axis from the output shape.
    /// (e.g. `[2, 3]` over dim 0 → `[3]`)
    fn sum_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "sum_dim",
                expected: t.shape.clone(),
                got: vec![dim],
                msg: format!("sum_dim: axis {dim} out of range for shape {:?}", t.shape),
            });
        }
        let out = sum_axis_squeeze(t, dim);

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                // Backward of sum_dim (squeeze): reinsert the axis with size 1,
                // then broadcast back to the original shape.
                let mut keepdim_shape = grad_out.shape.clone();
                keepdim_shape.insert(dim, 1);
                let keepdim = grad_out
                    .reshape(&keepdim_shape)
                    .expect("sum_dim backward: reinserting squeezed axis cannot fail");
                let expanded = keepdim
                    .broadcast_as(&original_shape)
                    .expect("sum_dim backward: broadcast to original shape cannot fail");
                // Materialize the broadcast view (walk all elements) so the
                // gradient is a concrete contiguous tensor, not a strided view
                // that upstream accumulation might mis-sum.
                let total: usize = original_shape.iter().product();
                let mut idx = vec![0usize; original_shape.len()];
                let mut vals = Vec::with_capacity(total);
                for _ in 0..total {
                    vals.push(expanded.get(&idx) as f32);
                    increment_index(&mut idx, &original_shape);
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(vals),
                    original_shape.clone(),
                )]
            }),
        });

        Ok(out)
    }

    /// Sum over `dim`, keeping that axis as size 1.
    /// (e.g. `[2, 3]` over dim 0 → `[1, 3]`)
    fn sum_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "sum_keepdim",
                expected: t.shape.clone(),
                got: vec![dim],
                msg: format!(
                    "sum_keepdim: axis {dim} out of range for shape {:?}",
                    t.shape
                ),
            });
        }
        let out = sum_axis_keepdim(t, dim);

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                // Backward of sum_keepdim: broadcast the keepdim gradient
                // (which already has size 1 on `dim`) back to the original
                // shape, then materialize it.
                let expanded = grad_out
                    .broadcast_as(&original_shape)
                    .expect("sum_keepdim backward: broadcast to original shape cannot fail");
                let total: usize = original_shape.iter().product();
                let mut idx = vec![0usize; original_shape.len()];
                let mut vals = Vec::with_capacity(total);
                for _ in 0..total {
                    vals.push(expanded.get(&idx) as f32);
                    increment_index(&mut idx, &original_shape);
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(vals),
                    original_shape.clone(),
                )]
            }),
        });

        Ok(out)
    }

    /// Mean over `dim`, removing that axis from the output shape.
    /// Thin wrapper over `sum_axis_squeeze`, divided by the axis length.
    /// (e.g. `[2, 3]` over dim 0 → `[3]`, each value = column sum / 2)
    fn mean_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "mean_dim",
                expected: t.shape.clone(),
                got: vec![dim],
                msg: format!("mean_dim: axis {dim} out of range for shape {:?}", t.shape),
            });
        }
        let axis_len = t.shape[dim] as f64;
        let summed = sum_axis_squeeze(t, dim);
        let out_shape = summed.shape.clone();
        let total: usize = out_shape.iter().product();
        let mut idx = vec![0usize; out_shape.len()];
        let mut vals = Vec::with_capacity(total);
        for _ in 0..total {
            vals.push((summed.get(&idx) / axis_len) as f32);
            increment_index(&mut idx, &out_shape);
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(vals), out_shape);

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                // Backward of mean_dim (squeeze): reinsert the axis with size
                // 1, broadcast back to the original shape, then scale every
                // materialized value by 1/axis_len (mirrors mean_all's 1/n
                // relationship to sum_all).
                let mut keepdim_shape = grad_out.shape.clone();
                keepdim_shape.insert(dim, 1);
                let keepdim = grad_out
                    .reshape(&keepdim_shape)
                    .expect("mean_dim backward: reinserting squeezed axis cannot fail");
                let expanded = keepdim
                    .broadcast_as(&original_shape)
                    .expect("mean_dim backward: broadcast to original shape cannot fail");
                let total: usize = original_shape.iter().product();
                let mut idx = vec![0usize; original_shape.len()];
                let mut vals = Vec::with_capacity(total);
                for _ in 0..total {
                    vals.push((expanded.get(&idx) / axis_len) as f32);
                    increment_index(&mut idx, &original_shape);
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(vals),
                    original_shape.clone(),
                )]
            }),
        });

        Ok(out)
    }

    /// Mean over `dim`, keeping that axis as size 1.
    /// Thin wrapper over `sum_axis_keepdim`, divided by the axis length.
    /// (e.g. `[2, 3]` over dim 0 → `[1, 3]`)
    fn mean_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "mean_keepdim",
                expected: t.shape.clone(),
                got: vec![dim],
                msg: format!(
                    "mean_keepdim: axis {dim} out of range for shape {:?}",
                    t.shape
                ),
            });
        }
        let axis_len = t.shape[dim] as f64;
        let summed = sum_axis_keepdim(t, dim);
        let out_shape = summed.shape.clone();
        let total: usize = out_shape.iter().product();
        let mut idx = vec![0usize; out_shape.len()];
        let mut vals = Vec::with_capacity(total);
        for _ in 0..total {
            vals.push((summed.get(&idx) / axis_len) as f32);
            increment_index(&mut idx, &out_shape);
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(vals), out_shape);

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                // Backward of mean_keepdim: broadcast the keepdim gradient
                // (already size 1 on `dim`) back to the original shape, then
                // scale by 1/axis_len.
                let expanded = grad_out
                    .broadcast_as(&original_shape)
                    .expect("mean_keepdim backward: broadcast to original shape cannot fail");
                let total: usize = original_shape.iter().product();
                let mut idx = vec![0usize; original_shape.len()];
                let mut vals = Vec::with_capacity(total);
                for _ in 0..total {
                    vals.push((expanded.get(&idx) / axis_len) as f32);
                    increment_index(&mut idx, &original_shape);
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(vals),
                    original_shape.clone(),
                )]
            }),
        });

        Ok(out)
    }

    /// Maximum over `dim`, removing that axis from the output shape.
    /// Backward routes gradient to exactly one winning element per output
    /// position (T-02-07/T-02-08 mitigations).
    fn max_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "max_dim",
                expected: t.shape.clone(),
                got: vec![dim],
                msg: format!("max_dim: axis {dim} out of range for shape {:?}", t.shape),
            });
        }
        let (keepdim_out, winning_flat_src_idx) = max_axis_with_indices(t, dim);
        let mut squeeze_shape = keepdim_out.shape.clone();
        squeeze_shape.remove(dim);
        let out = keepdim_out
            .reshape(&squeeze_shape)
            .expect("max_dim: squeeze reshape of size-1 keepdim result cannot fail");

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![scatter_axis_grad(
                    grad_out,
                    &winning_flat_src_idx,
                    &original_shape,
                )]
            }),
        });

        Ok(out)
    }

    /// Maximum over `dim`, keeping that axis as size 1.
    fn max_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "max_keepdim",
                expected: t.shape.clone(),
                got: vec![dim],
                msg: format!(
                    "max_keepdim: axis {dim} out of range for shape {:?}",
                    t.shape
                ),
            });
        }
        let (out, winning_flat_src_idx) = max_axis_with_indices(t, dim);

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![scatter_axis_grad(
                    grad_out,
                    &winning_flat_src_idx,
                    &original_shape,
                )]
            }),
        });

        Ok(out)
    }

    /// Minimum over `dim`, removing that axis from the output shape. Mirror
    /// of `max_dim` using `min_axis_with_indices`.
    fn min_dim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "min_dim",
                expected: t.shape.clone(),
                got: vec![dim],
                msg: format!("min_dim: axis {dim} out of range for shape {:?}", t.shape),
            });
        }
        let (keepdim_out, winning_flat_src_idx) = min_axis_with_indices(t, dim);
        let mut squeeze_shape = keepdim_out.shape.clone();
        squeeze_shape.remove(dim);
        let out = keepdim_out
            .reshape(&squeeze_shape)
            .expect("min_dim: squeeze reshape of size-1 keepdim result cannot fail");

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![scatter_axis_grad(
                    grad_out,
                    &winning_flat_src_idx,
                    &original_shape,
                )]
            }),
        });

        Ok(out)
    }

    /// Minimum over `dim`, keeping that axis as size 1. Mirror of
    /// `max_keepdim` using `min_axis_with_indices`.
    fn min_keepdim<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "min_keepdim",
                expected: t.shape.clone(),
                got: vec![dim],
                msg: format!(
                    "min_keepdim: axis {dim} out of range for shape {:?}",
                    t.shape
                ),
            });
        }
        let (out, winning_flat_src_idx) = min_axis_with_indices(t, dim);

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![scatter_axis_grad(
                    grad_out,
                    &winning_flat_src_idx,
                    &original_shape,
                )]
            }),
        });

        Ok(out)
    }

    /// Index of the maximum element. `Some(d)`: per-axis, axis removed from
    /// the output shape (mirrors `max_dim`'s squeeze shape). `None`: fully
    /// flattened, returns a scalar (shape `[]`) holding the single winning
    /// flat index. Forward-only — `kindle-core`'s `Tensor::argmax`
    /// structurally forces `G = NoGrad` on the output regardless of the
    /// input's own `G`, so this deliberately never calls `tape::push`
    /// (T-02-09 mitigation; the one exception to this file's
    /// every-other-method unconditional-push convention).
    fn argmax<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        match dim {
            Some(d) => {
                if d >= t.shape.len() {
                    return Err(Error::ShapeMismatch {
                        op: "argmax",
                        expected: t.shape.clone(),
                        got: vec![d],
                        msg: format!("argmax: axis {d} out of range for shape {:?}", t.shape),
                    });
                }
                let (_, winning_flat_src_idx) = max_axis_with_indices(t, d);
                let mut out_shape = t.shape.clone();
                out_shape[d] = 1;
                // Convert each winning FLAT source index into its coordinate
                // along `d` (the axis-position the winner occupied), not the
                // flat index itself.
                let idx_vals: Vec<i64> = winning_flat_src_idx
                    .iter()
                    .map(|&flat_src| {
                        let multi = unflatten_index(flat_src, &t.shape);
                        multi[d] as i64
                    })
                    .collect();
                let keepdim_out =
                    NativeStorage::from_contiguous(NativeBuffer::I64(idx_vals), out_shape);
                let mut squeeze_shape = keepdim_out.shape.clone();
                squeeze_shape.remove(d);
                Ok(keepdim_out
                    .reshape(&squeeze_shape)
                    .expect("argmax: squeeze reshape of size-1 keepdim result cannot fail"))
            }
            None => {
                let total: usize = t.shape.iter().product();
                let mut idx = vec![0usize; t.shape.len()];
                let mut best_val = f64::NEG_INFINITY;
                let mut best_flat_idx = 0i64;
                for flat in 0..total {
                    let v = t.get(&idx);
                    if v > best_val {
                        best_val = v;
                        best_flat_idx = flat as i64;
                    }
                    if !t.shape.is_empty() {
                        increment_index(&mut idx, &t.shape);
                    }
                }
                Ok(NativeStorage::from_contiguous(
                    NativeBuffer::I64(vec![best_flat_idx]),
                    vec![],
                ))
            }
        }
    }

    /// Index of the minimum element. Mirror of `argmax` using
    /// `min_axis_with_indices`. Forward-only, no `tape::push` (T-02-09).
    fn argmin<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        match dim {
            Some(d) => {
                if d >= t.shape.len() {
                    return Err(Error::ShapeMismatch {
                        op: "argmin",
                        expected: t.shape.clone(),
                        got: vec![d],
                        msg: format!("argmin: axis {d} out of range for shape {:?}", t.shape),
                    });
                }
                let (_, winning_flat_src_idx) = min_axis_with_indices(t, d);
                let mut out_shape = t.shape.clone();
                out_shape[d] = 1;
                let idx_vals: Vec<i64> = winning_flat_src_idx
                    .iter()
                    .map(|&flat_src| {
                        let multi = unflatten_index(flat_src, &t.shape);
                        multi[d] as i64
                    })
                    .collect();
                let keepdim_out =
                    NativeStorage::from_contiguous(NativeBuffer::I64(idx_vals), out_shape);
                let mut squeeze_shape = keepdim_out.shape.clone();
                squeeze_shape.remove(d);
                Ok(keepdim_out
                    .reshape(&squeeze_shape)
                    .expect("argmin: squeeze reshape of size-1 keepdim result cannot fail"))
            }
            None => {
                let total: usize = t.shape.iter().product();
                let mut idx = vec![0usize; t.shape.len()];
                let mut best_val = f64::INFINITY;
                let mut best_flat_idx = 0i64;
                for flat in 0..total {
                    let v = t.get(&idx);
                    if v < best_val {
                        best_val = v;
                        best_flat_idx = flat as i64;
                    }
                    if !t.shape.is_empty() {
                        increment_index(&mut idx, &t.shape);
                    }
                }
                Ok(NativeStorage::from_contiguous(
                    NativeBuffer::I64(vec![best_flat_idx]),
                    vec![],
                ))
            }
        }
    }

    /// Auto-generated documentation for topk.
    fn topk<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(<Self as Backend>::Storage<K>, <Self as Backend>::Storage<KInt>)> {
        let shape = &t.shape;
        if dim >= shape.len() {
            return Err(Error::ShapeMismatch {
                op: "topk",
                expected: shape.clone(),
                got: vec![dim],
                msg: format!("topk: axis {} out of range", dim),
            });
        }
        let k = k.min(shape[dim]);
        let mut out_shape = shape.clone();
        out_shape[dim] = k;

        let mut base_shape = shape.clone();
        base_shape[dim] = 1;
        let n_slices = base_shape.iter().product::<usize>();

        let mut out_vals = vec![0.0f32; out_shape.iter().product()];
        let mut out_indices = vec![0u32; out_shape.iter().product()];

        for i in 0..n_slices {
            let mut rem = i;
            let mut coords = vec![0usize; shape.len()];
            for dd in (0..shape.len()).rev() {
                coords[dd] = rem % base_shape[dd];
                rem /= base_shape[dd];
            }

            let mut slice_vals = Vec::with_capacity(shape[dim]);
            for j in 0..shape[dim] {
                coords[dim] = j;
                slice_vals.push((t.get(&coords), j as u32));
            }
            if largest {
                slice_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            } else {
                slice_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            }

            let mut out_coords = coords.clone();
            for j in 0..k {
                out_coords[dim] = j;
                let flat = flatten_index(&out_coords, &out_shape);
                out_vals[flat] = slice_vals[j].0 as f32;
                out_indices[flat] = slice_vals[j].1;
            }
        }
        Ok((
            NativeStorage::from_contiguous(NativeBuffer::F32(out_vals), out_shape.clone()),
            NativeStorage::from_contiguous(NativeBuffer::U32(out_indices), out_shape),
        ))
    }

    /// Auto-generated documentation for argsort.
    fn argsort<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let shape = &t.shape;
        if dim >= shape.len() {
            return Err(Error::ShapeMismatch {
                op: "argsort",
                expected: shape.clone(),
                got: vec![dim],
                msg: format!("argsort: axis {} out of range", dim),
            });
        }
        let mut base_shape = shape.clone();
        base_shape[dim] = 1;
        let n_slices = base_shape.iter().product::<usize>();
        let mut out = vec![0u32; shape.iter().product()];

        for i in 0..n_slices {
            let mut rem = i;
            let mut coords = vec![0usize; shape.len()];
            for dd in (0..shape.len()).rev() {
                coords[dd] = rem % base_shape[dd];
                rem /= base_shape[dd];
            }

            let mut slice_vals = Vec::with_capacity(shape[dim]);
            for k in 0..shape[dim] {
                coords[dim] = k;
                slice_vals.push((t.get(&coords), k as u32));
            }
            if descending {
                slice_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            } else {
                slice_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            }
            for k in 0..shape[dim] {
                coords[dim] = k;
                let flat = flatten_index(&coords, shape);
                out[flat] = slice_vals[k].1;
            }
        }
        Ok(NativeStorage::from_contiguous(NativeBuffer::U32(out), shape.clone()))
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use super::*;
    use crate::gradcheck::gradcheck;
    use crate::tape;

    /// Auto-generated documentation for B.
    type B = NativeBackend<f32, kindle_core::prelude::Cpu>;

    /// Auto-generated documentation for matrix.
    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![rows, cols])
    }

    /// Auto-generated documentation for vector.
    fn vector(v: Vec<f32>) -> NativeStorage {
        let len = v.len();
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![len])
    }

    /// Auto-generated documentation for f32_vec.
    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    // --- sum_all ---

    #[test]
    /// Auto-generated documentation for sum_all_on_2x3_returns_correct_scalar.
    fn sum_all_on_2x3_returns_correct_scalar() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::sum_all::<f32>(&t).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new()); // scalar shape []
        assert_eq!(out.get(&[]), 21.0);
    }

    #[test]
    /// Auto-generated documentation for sum_all_backward_distributes_grad_uniformly.
    fn sum_all_backward_distributes_grad_uniformly() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::sum_all::<f32>(&t).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have gradient");
        assert_eq!(g.shape, vec![2, 3]);
        // sum_all backward: every element receives grad_scalar = 1.0 (ones_like seed)
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    // --- mean_all ---

    #[test]
    /// Auto-generated documentation for mean_all_on_2x3_returns_correct_scalar.
    fn mean_all_on_2x3_returns_correct_scalar() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::mean_all::<f32>(&t).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
        // mean = 21/6 = 3.5
        let v = out.get(&[]);
        assert!((v - 3.5).abs() < 1e-5, "mean_all expected 3.5, got {v}");
    }

    #[test]
    /// Auto-generated documentation for mean_all_backward_distributes_grad_scaled_by_1_over_n.
    fn mean_all_backward_distributes_grad_scaled_by_1_over_n() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::mean_all::<f32>(&t).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have gradient");
        assert_eq!(g.shape, vec![2, 3]);
        // d(mean)/d(x_i) = 1/6; incoming grad = 1.0 → each element gets 1/6
        for &v in f32_vec(g).iter() {
            assert!(
                (v - 1.0 / 6.0).abs() < 1e-5,
                "mean_all backward: expected 1/6, got {v}"
            );
        }
    }

    // --- sum_dim ---

    #[test]
    /// Auto-generated documentation for sum_dim_removes_axis_0_on_2x3.
    fn sum_dim_removes_axis_0_on_2x3() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::sum_dim::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![3]);
        // col sums: 1+4=5, 2+5=7, 3+6=9
        assert_eq!(f32_vec(&out), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    /// Auto-generated documentation for sum_dim_removes_axis_1_on_2x3.
    fn sum_dim_removes_axis_1_on_2x3() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::sum_dim::<f32>(&t, 1).unwrap();
        assert_eq!(out.shape, vec![2]);
        // row sums: 1+2+3=6, 4+5+6=15
        assert_eq!(f32_vec(&out), vec![6.0, 15.0]);
    }

    #[test]
    /// Auto-generated documentation for sum_dim_backward_broadcasts_grad_back_to_original_shape.
    fn sum_dim_backward_broadcasts_grad_back_to_original_shape() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::sum_dim::<f32>(&t, 0).unwrap(); // shape [3]
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have gradient");
        assert_eq!(g.shape, vec![2, 3]);
        // ones_like(out) = [1,1,1] broadcast back to [2,3] = ones
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    // --- sum_keepdim ---

    #[test]
    /// Auto-generated documentation for sum_keepdim_retains_axis_0_on_2x3.
    fn sum_keepdim_retains_axis_0_on_2x3() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::sum_keepdim::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(f32_vec(&out), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    /// Auto-generated documentation for sum_keepdim_backward_broadcasts_grad_to_original_shape.
    fn sum_keepdim_backward_broadcasts_grad_to_original_shape() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::sum_keepdim::<f32>(&t, 0).unwrap(); // shape [1, 3]
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have gradient");
        assert_eq!(g.shape, vec![2, 3]);
        // ones_like([1,3]) broadcast to [2,3] = ones
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    // --- sum_all backward with non-trivial incoming gradient (tape chain) ---

    #[test]
    /// Auto-generated documentation for sum_all_backward_scales_by_incoming_gradient.
    fn sum_all_backward_scales_by_incoming_gradient() {
        // Build a small graph: out = sum_all(t), then seed with grad = 2.0
        // instead of 1.0 by composing with a scalar mul.
        // Simplest approach: verify via a custom tape entry.
        let t = vector(vec![1.0, 2.0, 3.0]);
        let sum_out = B::sum_all::<f32>(&t).unwrap();
        // Manually build a loss = 2.0 * sum_out by pushing a tape entry
        let loss = NativeStorage::from_contiguous(NativeBuffer::F32(vec![0.0f32]), vec![]);
        let (sum_id, loss_id) = (sum_out.id, loss.id);
        tape::push(TapeEntry {
            output_id: loss_id,
            input_ids: vec![sum_id],
            backward: Box::new(|_grad_out: &NativeStorage| {
                // d(2 * sum_out) / d(sum_out) = 2
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(vec![2.0f32]),
                    vec![],
                )]
            }),
        });
        let grads = tape::backward(&loss).unwrap();
        let g = grads.get(t.id).expect("t should have gradient");
        assert_eq!(g.shape, vec![3]);
        // Each element's gradient = 2.0 (scalar grad) * 1 (sum backward factor) = 2.0
        assert_eq!(f32_vec(g), vec![2.0, 2.0, 2.0]);
    }

    // --- mean_dim / mean_keepdim ---

    #[test]
    /// Auto-generated documentation for mean_dim_column_means_on_2x3.
    fn mean_dim_column_means_on_2x3() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::mean_dim::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![3]);
        let vals = f32_vec(&out);
        for (v, expected) in vals.iter().zip([2.5, 3.5, 4.5].iter()) {
            assert!((v - expected).abs() < 1e-5, "got {v}, expected {expected}");
        }
    }

    #[test]
    /// Auto-generated documentation for mean_keepdim_column_means_on_2x3.
    fn mean_keepdim_column_means_on_2x3() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = B::mean_keepdim::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
        let vals = f32_vec(&out);
        for (v, expected) in vals.iter().zip([2.5, 3.5, 4.5].iter()) {
            assert!((v - expected).abs() < 1e-5, "got {v}, expected {expected}");
        }
    }

    #[test]
    /// Auto-generated documentation for mean_dim_gradcheck_dim0.
    fn mean_dim_gradcheck_dim0() {
        let x = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            let reduced = B::mean_dim::<f32>(&inputs[0], 0).unwrap();
            B::sum_all::<f32>(&reduced).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "mean_dim gradcheck max relative error too high: {max_rel_err}"
        );
    }

    #[test]
    /// Auto-generated documentation for mean_keepdim_gradcheck_dim1.
    fn mean_keepdim_gradcheck_dim1() {
        let x = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            let reduced = B::mean_keepdim::<f32>(&inputs[0], 1).unwrap();
            B::sum_all::<f32>(&reduced).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "mean_keepdim gradcheck max relative error too high: {max_rel_err}"
        );
    }

    // --- max_dim / min_dim / max_keepdim / min_keepdim / max_all / min_all ---

    #[test]
    /// Auto-generated documentation for max_dim_column_maxima_on_2x3.
    fn max_dim_column_maxima_on_2x3() {
        let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let out = B::max_dim::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![3]);
        assert_eq!(f32_vec(&out), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    /// Auto-generated documentation for max_keepdim_column_maxima_on_2x3.
    fn max_keepdim_column_maxima_on_2x3() {
        let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let out = B::max_keepdim::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(f32_vec(&out), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    /// Auto-generated documentation for min_dim_column_minima_on_2x3.
    fn min_dim_column_minima_on_2x3() {
        let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let out = B::min_dim::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![3]);
        assert_eq!(f32_vec(&out), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    /// Auto-generated documentation for min_keepdim_column_minima_on_2x3.
    fn min_keepdim_column_minima_on_2x3() {
        let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let out = B::min_keepdim::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(f32_vec(&out), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    /// Auto-generated documentation for max_all_and_min_all_on_flat_vector.
    fn max_all_and_min_all_on_flat_vector() {
        let t = vector(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let max_out = B::max_all::<f32>(&t).unwrap();
        assert_eq!(max_out.shape, Vec::<usize>::new());
        assert_eq!(max_out.get(&[]), 6.0);

        let min_out = B::min_all::<f32>(&t).unwrap();
        assert_eq!(min_out.shape, Vec::<usize>::new());
        assert_eq!(min_out.get(&[]), 1.0);
    }

    #[test]
    /// Auto-generated documentation for max_dim_gradcheck_all_distinct_values.
    fn max_dim_gradcheck_all_distinct_values() {
        let x = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            let reduced = B::max_dim::<f32>(&inputs[0], 0).unwrap();
            B::sum_all::<f32>(&reduced).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "max_dim gradcheck max relative error too high: {max_rel_err}"
        );
    }

    /// Tie case (Pitfall 3 / T-02-07): column 0 has a tie between two equal
    /// maxima (`2.0`, `2.0`). The winning column's summed backward gradient
    /// must equal exactly `1.0` (the incoming seed gradient from
    /// `sum_all`'s ones-seed), NOT `2.0`, which would indicate the naive
    /// "scatter to every `==` position" bug.
    #[test]
    fn max_dim_backward_routes_gradient_to_exactly_one_winner_on_tie() {
        // Matrix [2,2]: column 0 = [2.0, 2.0] (tie), column 1 = [1.0, 3.0].
        let t = matrix(vec![2.0, 1.0, 2.0, 3.0], 2, 2);
        let out = B::max_dim::<f32>(&t, 0).unwrap();
        assert_eq!(f32_vec(&out), vec![2.0, 3.0]);

        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have gradient");
        assert_eq!(g.shape, vec![2, 2]);
        let vals = f32_vec(g);
        // Column 0 (indices 0 and 2 in row-major [2,2]) gradient total must
        // be exactly 1.0, split across exactly one of the two tied rows.
        let col0_total = vals[0] + vals[2];
        assert!(
            (col0_total - 1.0).abs() < 1e-6,
            "tie-case column 0 gradient total should be 1.0, got {col0_total}"
        );
        // Exactly one of the two tied positions receives the full 1.0.
        assert!(
            (vals[0] - 1.0).abs() < 1e-6 && vals[2].abs() < 1e-6
                || vals[0].abs() < 1e-6 && (vals[2] - 1.0).abs() < 1e-6,
            "expected exactly one winner in tied column 0, got vals[0]={}, vals[2]={}",
            vals[0],
            vals[2]
        );
    }

    // --- argmax / argmin ---

    /// Auto-generated documentation for i64_vec.
    fn i64_vec(s: &NativeStorage) -> Vec<i64> {
        match &*s.buffer {
            NativeBuffer::I64(v) => v.clone(),
            _ => panic!("expected I64 buffer"),
        }
    }

    #[test]
    /// Auto-generated documentation for argmax_dim0_returns_row_index_of_column_max.
    fn argmax_dim0_returns_row_index_of_column_max() {
        let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let out = B::argmax::<f32, i64>(&t, Some(0)).unwrap();
        assert_eq!(out.shape, vec![3]);
        // col0 max is row1's 4 -> idx 1; col1 max is row0's 5 -> idx 0;
        // col2 max is row1's 6 -> idx 1.
        assert_eq!(i64_vec(&out), vec![1, 0, 1]);
    }

    #[test]
    /// Auto-generated documentation for argmax_dim_none_returns_scalar_flat_index_of_global_max.
    fn argmax_dim_none_returns_scalar_flat_index_of_global_max() {
        let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let out = B::argmax::<f32, i64>(&t, None).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
        // global max 6.0 is at flat index 5.
        assert_eq!(i64_vec(&out), vec![5]);
    }

    #[test]
    /// Auto-generated documentation for argmin_dim0_and_dim_none_mirror_argmax.
    fn argmin_dim0_and_dim_none_mirror_argmax() {
        let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let out_dim0 = B::argmin::<f32, i64>(&t, Some(0)).unwrap();
        assert_eq!(out_dim0.shape, vec![3]);
        // col0 min is row0's 1 -> idx 0; col1 min is row1's 2 -> idx 1;
        // col2 min is row0's 3 -> idx 0.
        assert_eq!(i64_vec(&out_dim0), vec![0, 1, 0]);

        let out_none = B::argmin::<f32, i64>(&t, None).unwrap();
        assert_eq!(out_none.shape, Vec::<usize>::new());
        // global min 1.0 is at flat index 0.
        assert_eq!(i64_vec(&out_none), vec![0]);
    }

    /// argmax/argmin must push NO TapeEntry (structural NoGrad, T-02-09):
    /// calling them, then immediately running `tape::backward` on an
    /// unrelated small graph, must succeed cleanly with no interference
    /// from a spurious entry either method might have left behind.
    #[test]
    fn argmax_argmin_do_not_push_tape_entries() {
        let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
        let _ = B::argmax::<f32, i64>(&t, Some(0)).unwrap();
        let _ = B::argmax::<f32, i64>(&t, None).unwrap();
        let _ = B::argmin::<f32, i64>(&t, Some(0)).unwrap();
        let _ = B::argmin::<f32, i64>(&t, None).unwrap();

        // Build and run an unrelated small graph immediately after; if
        // argmax/argmin had pushed spurious TapeEntry values, this
        // unrelated backward() would either panic or produce a corrupted
        // gradient for `unrelated`.
        let unrelated = vector(vec![10.0, 20.0, 30.0]);
        let sum_out = B::sum_all::<f32>(&unrelated).unwrap();
        let grads = tape::backward(&sum_out).unwrap();
        let g = grads
            .get(unrelated.id)
            .expect("unrelated should have gradient");
        assert_eq!(f32_vec(g), vec![1.0, 1.0, 1.0]);
    }
}
