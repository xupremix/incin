//! The autograd tape: unconditional per-op recording (D-05), reverse-walk
//! gradient accumulation with drain-on-return (D-06), and the shared
//! `unbroadcast` helper (NATBACK-06).
//!
//! This tape is deliberately independent from `kindle-core`'s
//! `tensor::tracing` module (`TRACING_GRAPH`, used for ONNX export) — D-04.
//! The two thread-locals never reference each other; this file must not
//! import or reference `TRACING_GRAPH` anywhere.
//!
//! The single highest-risk correctness surface here is
//! `backward()`'s accumulation loop: a tensor read by two downstream ops
//! MUST have its gradient contributions summed (`entry().and_modify()`),
//! never overwritten (a bare `.insert()`) — this is NATBACK-05's literal
//! correctness gate.

use alloc::collections::BTreeMap;
use core::cell::RefCell;

use kindle_core::prelude::Result;

use crate::storage::{NativeBuffer, NativeStorage, TensorId};

/// A boxed backward closure: receives the accumulated gradient for a
/// `TapeEntry`'s `output_id` and returns one gradient per `input_id`, in
/// the same order.
pub type BackwardFn = Box<dyn Fn(&NativeStorage) -> Vec<NativeStorage> + Send + Sync>;

/// One recorded operation: the output it produced, the inputs it consumed,
/// and a boxed backward closure mapping an accumulated output-gradient to
/// one gradient per input (same order as `input_ids`).
///
/// Per D-05, `push()` records every op unconditionally — the backend has no
/// visibility into whether the surrounding `Tensor<..., G>`'s `G` is `Grad`
/// or `NoGrad`.
pub struct TapeEntry {
    /// Auto-generated documentation for output_id.
    pub output_id: TensorId,
    /// Auto-generated documentation for input_ids.
    pub input_ids: Vec<TensorId>,
    /// Auto-generated documentation for backward.
    pub backward: BackwardFn,
}

/// The backend's gradient container: `Backend::Grads` in a later plan's
/// `lib.rs` impl. Wraps a plain `BTreeMap` keyed by `TensorId`.
pub struct NativeGrads {
    /// Auto-generated documentation for grads.
    pub grads: BTreeMap<TensorId, NativeStorage>,
}

impl NativeGrads {
    /// Look up the accumulated gradient for a given tensor id, if any.
    pub fn get(&self, id: TensorId) -> Option<&NativeStorage> {
        self.grads.get(&id)
    }
}

thread_local! {
    /// Auto-generated documentation for TAPE.
    static TAPE: RefCell<Vec<TapeEntry>> = RefCell::new(Vec::new());
}

/// Push a `TapeEntry` onto the thread-local tape, unconditionally (D-05).
pub fn push(entry: TapeEntry) {
    TAPE.with(|t| t.borrow_mut().push(entry));
}

/// Number of entries currently on the tape. Exposed for tests proving the
/// tape drains fully between `backward()` calls (D-06).
#[cfg(test)]
fn len() -> usize {
    TAPE.with(|t| t.borrow().len())
}

/// Walk the tape backward, seeding `loss`'s gradient with ones, accumulating
/// (never overwriting) contributions for reused tensors, and draining the
/// tape before returning (D-06).
///
/// Algorithm (RESEARCH.md Pattern 3):
/// 1. Seed `grads[loss.id] = ones_like(loss)`.
/// 2. Drain the tape via `mem::take` BEFORE walking it (D-06) — this must
///    happen before any entry is invoked, not just before returning.
/// 3. Walk the drained entries in reverse insertion order. For each entry,
///    look up the accumulated gradient for `output_id`; if absent, `continue`
///    (an unreached branch, not an error). Otherwise invoke the backward
///    closure and accumulate each resulting gradient via
///    `entry(id).and_modify(sum).or_insert(new)` — never a bare `.insert()`.
pub fn backward(loss: &NativeStorage) -> Result<NativeGrads> {
    let mut grads: BTreeMap<TensorId, NativeStorage> = BTreeMap::new();
    grads.insert(loss.id, NativeStorage::ones_like(loss));

    // Drain BEFORE walking (D-06) — mirrors tracing.rs's extract_graph()
    // mem::take idiom, but on an independent thread-local (D-04).
    let entries = TAPE.with(|t| core::mem::take(&mut *t.borrow_mut()));

    for entry in entries.into_iter().rev() {
        let Some(grad_out) = grads.get(&entry.output_id).cloned() else {
            continue;
        };
        let input_grads = (entry.backward)(&grad_out);
        for (input_id, g) in entry.input_ids.into_iter().zip(input_grads) {
            grads
                .entry(input_id)
                .and_modify(|acc| *acc = add_native_storage(acc, &g))
                .or_insert(g);
            //           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // NEVER a bare `.insert()` here — that's the literal
            // NATBACK-05 overwrite bug this loop must not reintroduce.
        }
    }

    Ok(NativeGrads { grads })
}

/// Helper to check if a tensor contains NaN or Infinity
fn check_nan(storage: &NativeStorage, id: TensorId) {
    let has_nan = match &*storage.buffer {
        NativeBuffer::F32(v) => v.iter().any(|x| x.is_nan() || x.is_infinite()),
        NativeBuffer::F64(v) => v.iter().any(|x| x.is_nan() || x.is_infinite()),
        NativeBuffer::F16(v) => v.iter().any(|x| x.is_nan() || x.is_infinite()),
        NativeBuffer::BF16(v) => v.iter().any(|x| x.is_nan() || x.is_infinite()),
        _ => false,
    };
    if has_nan {
        panic!("NaN or Infinity detected in gradient for TensorId {:?}", id);
    }
}

/// Same as `backward()`, but aggressively validates every intermediate gradient
/// for NaN or Infinity values, panicking immediately to pinpoint the exact operation.
pub fn backward_with_nan_check(loss: &NativeStorage) -> Result<NativeGrads> {
    let mut grads: BTreeMap<TensorId, NativeStorage> = BTreeMap::new();
    grads.insert(loss.id, NativeStorage::ones_like(loss));

    let entries = TAPE.with(|t| core::mem::take(&mut *t.borrow_mut()));

    for entry in entries.into_iter().rev() {
        let Some(grad_out) = grads.get(&entry.output_id).cloned() else {
            continue;
        };
        let input_grads = (entry.backward)(&grad_out);
        for (input_id, g) in entry.input_ids.into_iter().zip(input_grads) {
            check_nan(&g, input_id);
            grads
                .entry(input_id)
                .and_modify(|acc| {
                    *acc = add_native_storage(acc, &g);
                    check_nan(acc, input_id);
                })
                .or_insert(g);
        }
    }

    Ok(NativeGrads { grads })
}

/// Elementwise sum of two ALREADY-shape-matching gradients.
///
/// This is intentionally NOT the public `NumericOps::add` a later plan
/// implements (which must broadcast) — tape-internal accumulation only ever
/// sums two gradients that have already been shape-matched to their target
/// via `unbroadcast`, so no broadcast logic is needed here.
fn add_native_storage(a: &NativeStorage, b: &NativeStorage) -> NativeStorage {
    debug_assert_eq!(
        a.shape, b.shape,
        "tape accumulation requires matching shapes"
    );

    macro_rules! add_variant {
        ($variant:ident, $a_vec:expr, $b_vec:expr) => {{
            let total: usize = a.shape.iter().product();
            let mut out = Vec::with_capacity(total);
            let mut idx = vec![0usize; a.shape.len()];
            for _ in 0..total {
                out.push($a_vec(a, &idx) + $b_vec(b, &idx));
                increment_index(&mut idx, &a.shape);
            }
            NativeBuffer::$variant(out)
        }};
    }

    let new_buffer = match (&*a.buffer, &*b.buffer) {
        (NativeBuffer::F32(_), NativeBuffer::F32(_)) => {
            add_variant!(
                F32,
                |s: &NativeStorage, i: &[usize]| s.get(i) as f32,
                |s: &NativeStorage, i: &[usize]| s.get(i) as f32
            )
        }
        (NativeBuffer::F64(_), NativeBuffer::F64(_)) => {
            add_variant!(
                F64,
                |s: &NativeStorage, i: &[usize]| s.get(i),
                |s: &NativeStorage, i: &[usize]| s.get(i)
            )
        }
        _ => {
            // Only float dtypes are exercised by this phase's gradients;
            // fall back to an f32 sum for any other matching-variant pair
            // (I64/U8/U32 gradients are not a Phase 1 concern).
            add_variant!(
                F32,
                |s: &NativeStorage, i: &[usize]| s.get(i) as f32,
                |s: &NativeStorage, i: &[usize]| s.get(i) as f32
            )
        }
    };

    NativeStorage::from_contiguous(new_buffer, a.shape.clone())
}

/// Right-align `grad.shape` and `target_shape`; sum-reduce over any leading
/// axis present in `grad.shape` but absent from `target_shape` (squeezing
/// it away entirely), then sum-reduce (with keepdim) over any axis where
/// `target_shape` has size 1 but `grad`'s corresponding axis is `>1`.
///
/// A no-op (returns a clone) when `grad.shape == target_shape`.
pub fn unbroadcast(grad: &NativeStorage, target_shape: &[usize]) -> Result<NativeStorage> {
    if grad.shape == target_shape {
        return Ok(grad.clone());
    }

    let ndim_diff = grad.shape.len().saturating_sub(target_shape.len());
    let mut result = grad.clone();

    // Sum over any leading dims that don't exist in target_shape at all —
    // squeeze the axis away entirely (drop it from the shape).
    for _ in 0..ndim_diff {
        result = sum_dim_squeeze(&result, 0);
    }

    // Sum (with keepdim) over any axis where target_shape has size 1 but
    // result's corresponding axis is >1.
    for (i, &t_dim) in target_shape.iter().enumerate() {
        if t_dim == 1 && result.shape[i] != 1 {
            result = sum_dim_keepdim(&result, i);
        }
    }

    Ok(result)
}

/// Sum-reduce `storage` over `axis`, removing that axis from the shape
/// entirely (e.g. `[4,3]` reduced over axis 0 -> `[3]`).
fn sum_dim_squeeze(storage: &NativeStorage, axis: usize) -> NativeStorage {
    let reduced = sum_dim_keepdim(storage, axis);
    let mut new_shape = reduced.shape.clone();
    new_shape.remove(axis);
    // Squeezing a size-1 axis out of an already-contiguous buffer is a pure
    // metadata reshape (no data movement) since the buffer already excludes
    // that axis's stride contribution once its size is 1.
    reduced
        .reshape(&new_shape)
        .expect("squeeze reshape of size-1 axis cannot fail")
}

/// Sum-reduce `storage` over `axis`, keeping the axis present with size 1
/// (e.g. `[4,3]` reduced over axis 0 with keepdim -> `[1,3]`).
fn sum_dim_keepdim(storage: &NativeStorage, axis: usize) -> NativeStorage {
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
        NativeBuffer::Cuda(_) => panic!("sum_dim_keepdim not supported on CUDA buffer"),
        NativeBuffer::Metal(_) => panic!("sum_dim_keepdim not supported on Metal buffer"),
        NativeBuffer::Q8_0(_) => panic!("sum_dim_keepdim not supported on Q8_0 buffer"),
    };

    NativeStorage::from_contiguous(new_buffer, out_shape)
}

/// Compute the flat row-major index of `idx` within `shape`.
fn flatten_index(idx: &[usize], shape: &[usize]) -> usize {
    let strides = crate::stride::contiguous_strides(shape);
    idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

/// Increment a row-major multi-index in place (odometer-style), matching
/// `storage.rs`'s own `increment_index` iteration order.
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
/// Auto-generated documentation for tests.
mod tests {
    use super::*;
    use crate::storage::NativeBuffer;

    /// Auto-generated documentation for scalar.
    fn scalar(v: f32) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(vec![v]), vec![])
    }

    /// Auto-generated documentation for vector.
    fn vector(v: Vec<f32>) -> NativeStorage {
        let len = v.len();
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![len])
    }

    /// Auto-generated documentation for matrix.
    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![rows, cols])
    }

    // --- unbroadcast standalone tests (NATBACK-06) ---

    #[test]
    /// Auto-generated documentation for unbroadcast_bias_vector_b_n_to_n.
    fn unbroadcast_bias_vector_b_n_to_n() {
        // grad shape [4,3] (B=4, N=3), summed back to [3] (bias vector case).
        let grad = matrix(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            4,
            3,
        );
        let result = unbroadcast(&grad, &[3]).unwrap();
        assert_eq!(result.shape, vec![3]);
        // Column sums: col0 = 1+4+7+10=22, col1 = 2+5+8+11=26, col2 = 3+6+9+12=30
        assert_eq!(result.get(&[0]), 22.0);
        assert_eq!(result.get(&[1]), 26.0);
        assert_eq!(result.get(&[2]), 30.0);
    }

    #[test]
    /// Auto-generated documentation for unbroadcast_scalar_target_sums_all_axes.
    fn unbroadcast_scalar_target_sums_all_axes() {
        // grad shape [4,3], forward-broadcast from a scalar `[]`, summed
        // back to `[]` (scalar case).
        let grad = matrix(vec![1.0; 12], 4, 3);
        let result = unbroadcast(&grad, &[]).unwrap();
        assert_eq!(result.shape, Vec::<usize>::new());
        assert_eq!(result.get(&[]), 12.0);
    }

    #[test]
    /// Auto-generated documentation for unbroadcast_same_shape_is_noop.
    fn unbroadcast_same_shape_is_noop() {
        let grad = vector(vec![1.0, 2.0, 3.0]);
        let result = unbroadcast(&grad, &[3]).unwrap();
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.get(&[0]), 1.0);
        assert_eq!(result.get(&[1]), 2.0);
        assert_eq!(result.get(&[2]), 3.0);
    }

    // --- tape accumulation tests (NATBACK-05) ---

    #[test]
    /// Auto-generated documentation for backward_seeds_loss_gradient_with_ones.
    fn backward_seeds_loss_gradient_with_ones() {
        let loss = scalar(5.0);
        let grads = backward(&loss).unwrap();
        let g = grads.get(loss.id).unwrap();
        assert_eq!(g.get(&[]), 1.0);
    }

    #[test]
    /// Auto-generated documentation for backward_accumulates_not_overwrites_on_tensor_reuse.
    fn backward_accumulates_not_overwrites_on_tensor_reuse() {
        // Hand-built two-op graph: a single input tensor `x` is consumed
        // twice (mirrors `x.add(&x)`-shaped reuse). Two independent
        // TapeEntry values both list the same input_id; the resulting
        // gradient must equal the SUM of both consumers' contributions,
        // not a last-write-wins overwrite.
        let x = vector(vec![1.0, 2.0, 3.0]);
        let out1 = vector(vec![10.0, 20.0, 30.0]);
        let out2 = vector(vec![100.0, 200.0, 300.0]);

        let x_id = x.id;
        let out1_id = out1.id;
        let out2_id = out2.id;

        // Consumer 1: backward multiplies incoming grad by 2 (contribution A).
        push(TapeEntry {
            output_id: out1_id,
            input_ids: vec![x_id],
            backward: Box::new(|grad_out: &NativeStorage| {
                // Contribution A: grad_out * 2 element-wise (manual, no ops.rs yet)
                let data: Vec<f32> = (0..grad_out.shape[0])
                    .map(|i| (grad_out.get(&[i]) * 2.0) as f32)
                    .collect();
                vec![vector(data)]
            }),
        });

        // Consumer 2: backward multiplies incoming grad by 3 (contribution B).
        push(TapeEntry {
            output_id: out2_id,
            input_ids: vec![x_id],
            backward: Box::new(|grad_out: &NativeStorage| {
                let data: Vec<f32> = (0..grad_out.shape[0])
                    .map(|i| (grad_out.get(&[i]) * 3.0) as f32)
                    .collect();
                vec![vector(data)]
            }),
        });

        // Seed both outputs' gradients directly by driving backward() from
        // a fabricated "loss" that both out1 and out2 feed into equally
        // (grad_out for each = ones, since backward() seeds loss.id = ones
        // and we want independent, known incoming grads for each entry).
        // Simplest correct approach: call backward() with out1 as "loss"
        // is insufficient (only seeds out1). Instead, seed manually via two
        // separate hand-verified single-consumer computations, then compare
        // against the real two-consumer tape walk below.

        // Expected contribution from consumer 1 alone: ones_like(out1) * 2 = [2,2,2]
        // Expected contribution from consumer 2 alone: ones_like(out2) * 3 = [3,3,3]
        // Expected SUM at x: [5,5,5]

        // Drive backward() using a combined loss whose gradient seeds both
        // out1_id and out2_id with ones: we push one more entry that maps a
        // synthetic "total_loss" to both out1 and out2 with an identity-ish
        // backward (returns ones for each, matching backward()'s own
        // ones_like seeding contract for a fan-out sum node).
        let total_loss = scalar(0.0);
        let total_loss_id = total_loss.id;
        push(TapeEntry {
            output_id: total_loss_id,
            input_ids: vec![out1_id, out2_id],
            backward: Box::new(|_grad_out: &NativeStorage| {
                vec![vector(vec![1.0, 1.0, 1.0]), vector(vec![1.0, 1.0, 1.0])]
            }),
        });

        let grads = backward(&total_loss).unwrap();
        let g = grads
            .get(x_id)
            .expect("x should have an accumulated gradient");
        assert_eq!(g.shape, vec![3]);
        // SUM of both consumers' contributions: [2,2,2] + [3,3,3] = [5,5,5]
        assert_eq!(g.get(&[0]), 5.0);
        assert_eq!(g.get(&[1]), 5.0);
        assert_eq!(g.get(&[2]), 5.0);
    }

    #[test]
    /// Auto-generated documentation for backward_drains_tape_and_second_call_is_not_contaminated.
    fn backward_drains_tape_and_second_call_is_not_contaminated() {
        // First independent small graph.
        let x1 = scalar(1.0);
        let out1 = scalar(2.0);
        push(TapeEntry {
            output_id: out1.id,
            input_ids: vec![x1.id],
            backward: Box::new(|grad_out: &NativeStorage| {
                vec![scalar((grad_out.get(&[]) * 10.0) as f32)]
            }),
        });
        let grads1 = backward(&out1).unwrap();
        assert_eq!(grads1.get(x1.id).unwrap().get(&[]), 10.0);

        // Tape must be empty immediately after backward() returns.
        assert_eq!(len(), 0);

        // Second, independent small graph — must not see any entry from
        // the first call, and must not be contaminated by grads1's map.
        let x2 = scalar(1.0);
        let out2 = scalar(2.0);
        push(TapeEntry {
            output_id: out2.id,
            input_ids: vec![x2.id],
            backward: Box::new(|grad_out: &NativeStorage| {
                vec![scalar((grad_out.get(&[]) * 100.0) as f32)]
            }),
        });
        let grads2 = backward(&out2).unwrap();
        assert_eq!(grads2.get(x2.id).unwrap().get(&[]), 100.0);

        // grads2 must not contain x1's id (proves no cross-call leakage).
        assert!(grads2.get(x1.id).is_none());
        assert_eq!(len(), 0);
    }

    #[test]
    /// Auto-generated documentation for tape_len_is_zero_immediately_after_any_backward_call.
    fn tape_len_is_zero_immediately_after_any_backward_call() {
        let x = scalar(1.0);
        let out = scalar(2.0);
        push(TapeEntry {
            output_id: out.id,
            input_ids: vec![x.id],
            backward: Box::new(|grad_out: &NativeStorage| vec![grad_out.clone()]),
        });
        let _ = backward(&out).unwrap();
        assert_eq!(len(), 0);
    }

    #[test]
    #[should_panic(expected = "NaN or Infinity detected in gradient")]
    /// Auto-generated documentation for backward_with_nan_check_panics_on_nan.
    fn backward_with_nan_check_panics_on_nan() {
        let x = scalar(1.0);
        let out = scalar(2.0);
        push(TapeEntry {
            output_id: out.id,
            input_ids: vec![x.id],
            backward: Box::new(|_grad_out: &NativeStorage| vec![scalar(f32::NAN)]),
        });
        let _ = backward_with_nan_check(&out);
    }
}
