//! The autograd tape: unconditional per-op recording (D-05), reverse-walk
//! gradient accumulation with drain-on-return (D-06), and the shared
//! `unbroadcast` helper (CPUBACK-06).
//!
//! This tape is deliberately independent from `incin-core`'s
//! `tensor::tracing` module (`TRACING_GRAPH`, used for ONNX export) — D-04.
//! The two thread-locals never reference each other; this file must not
//! import or reference `TRACING_GRAPH` anywhere.
//!
//! The single highest-risk correctness surface here is
//! `backward()`'s accumulation loop: a tensor read by two downstream ops
//! MUST have its gradient contributions summed (`entry().and_modify()`),
//! never overwritten (a bare `.insert()`) — this is CPUBACK-05's literal
//! correctness gate.

use alloc::collections::BTreeMap;
use core::cell::RefCell;

use incin_core::prelude::Result;

use crate::cpu::storage::{CpuBuffer, CpuStorage, TensorId};

// A thread-local backward-call counter for telemetry step tracking.
#[cfg(feature = "telemetry")]
thread_local! {
    static BACKWARD_STEP: RefCell<usize> = const { RefCell::new(0) };
}

/// A boxed backward closure: receives the accumulated gradient for a
/// `TapeEntry`'s `output_id` and returns one gradient per `input_id`, in
/// the same order.
pub(crate) type BackwardFn = Box<dyn Fn(&CpuStorage) -> Vec<CpuStorage> + Send + Sync>;

/// One recorded operation: the output it produced, the inputs it consumed,
/// and a boxed backward closure mapping an accumulated output-gradient to
/// one gradient per input (same order as `input_ids`).
///
/// Per D-05, `push()` records every op unconditionally — the backend has no
/// visibility into whether the surrounding `Tensor<..., G>`'s `G` is `Grad`
/// or `NoGrad`.
pub(crate) struct TapeEntry {
    /// `output_id`.
    pub(crate) output_id: TensorId,
    /// `input_ids`.
    pub(crate) input_ids: Vec<TensorId>,
    /// `backward`.
    pub(crate) backward: BackwardFn,
}

/// The backend's gradient container: `Backend::Grads` in a later plan's
/// `lib.rs` impl. Wraps a plain `BTreeMap` keyed by `TensorId`.
pub struct CpuGrads {
    // Private per B-3 (.agents/API_DESIGN.md "pub(crate) is default"): use
    // `.get(id)` — downstream crates shouldn't inspect/mutate the internal
    // BTreeMap beyond the intended query API.
    pub(crate) grads: BTreeMap<TensorId, CpuStorage>,
}

impl CpuGrads {
    /// Look up the accumulated gradient for a given tensor id, if any.
    pub fn get(&self, id: TensorId) -> Option<&CpuStorage> {
        self.grads.get(&id)
    }
}

thread_local! {
    /// `TAPE`.
    static TAPE: RefCell<Vec<TapeEntry>> = const { RefCell::new(Vec::new()) };
}

/// Push a `TapeEntry` onto the thread-local tape, unconditionally (D-05).
pub(crate) fn push(entry: TapeEntry) {
    TAPE.with(|t| t.borrow_mut().push(entry));
    // Emit a scalar tracking tape depth when telemetry is enabled.
    #[cfg(feature = "telemetry")]
    {
        let depth = TAPE.with(|t| t.borrow().len()) as f64;
        let step = BACKWARD_STEP.with(|s| *s.borrow());
        crate::telemetry::emit_scalar(step, "tape/depth", depth);
    }
}

/// Number of entries currently on the tape. Exposed for tests proving the
/// tape drains fully between `backward()` calls (D-06).
#[cfg(test)]
pub(crate) fn len() -> usize {
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
pub fn backward(loss: &CpuStorage) -> Result<CpuGrads> {
    let mut grads: BTreeMap<TensorId, CpuStorage> = BTreeMap::new();
    grads.insert(loss.id, CpuStorage::ones_like(loss));

    // Drain BEFORE walking (D-06) — mirrors tracing.rs's extract_graph()
    // mem::take idiom, but on an independent thread-local (D-04).
    let entries = TAPE.with(|t| core::mem::take(&mut *t.borrow_mut()));
    #[cfg(feature = "telemetry")]
    let n_ops = entries.len();

    for entry in entries.into_iter().rev() {
        let Some(grad_out) = grads.get(&entry.output_id).cloned() else {
            continue;
        };
        let input_grads = (entry.backward)(&grad_out);
        for (input_id, g) in entry.input_ids.into_iter().zip(input_grads) {
            grads
                .entry(input_id)
                .and_modify(|acc| *acc = add_cpu_storage(acc, &g))
                .or_insert(g);
            //           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // NEVER a bare `.insert()` here — that's the literal
            // CPUBACK-05 overwrite bug this loop must not reintroduce.
        }
    }

    #[cfg(feature = "telemetry")]
    {
        let step = BACKWARD_STEP.with(|s| {
            let cur = *s.borrow();
            *s.borrow_mut() += 1;
            cur
        });
        emit_backward_telemetry(step, n_ops);
    }

    Ok(CpuGrads { grads })
}

/// Emit telemetry post-backward when the feature is enabled.
#[cfg(feature = "telemetry")]
fn emit_backward_telemetry(step: usize, n_ops: usize) {
    crate::telemetry::emit_scalar(step, "tape/ops", n_ops as f64);
    // Snapshot the current tracing graph and ship it to incin-viz.
    #[cfg(feature = "std")]
    {
        if let Some(g) = incin_core::prelude::tracing_graph_snapshot() {
            crate::telemetry::emit_graph_snapshot(g);
        }
    }
}

/// Helper to check if a tensor contains NaN or Infinity
fn check_nan(storage: &CpuStorage, id: TensorId) {
    let has_nan = match &*storage.buffer {
        CpuBuffer::F32(v) => v.iter().any(|x| x.is_nan() || x.is_infinite()),
        CpuBuffer::F64(v) => v.iter().any(|x| x.is_nan() || x.is_infinite()),
        CpuBuffer::F16(v) => v.iter().any(|x| x.is_nan() || x.is_infinite()),
        CpuBuffer::BF16(v) => v.iter().any(|x| x.is_nan() || x.is_infinite()),
        _ => false,
    };
    if has_nan {
        panic!("NaN or Infinity detected in gradient for TensorId {:?}", id);
    }
}

/// Same as `backward()`, but aggressively validates every intermediate gradient
/// for NaN or Infinity values, panicking immediately to pinpoint the exact operation.
pub fn backward_with_nan_check(loss: &CpuStorage) -> Result<CpuGrads> {
    let mut grads: BTreeMap<TensorId, CpuStorage> = BTreeMap::new();
    grads.insert(loss.id, CpuStorage::ones_like(loss));

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
                    *acc = add_cpu_storage(acc, &g);
                    check_nan(acc, input_id);
                })
                .or_insert(g);
        }
    }

    Ok(CpuGrads { grads })
}

/// Elementwise sum of two ALREADY-shape-matching gradients.
///
/// This is intentionally NOT the public `NumericOps::add` a later plan
/// implements (which must broadcast) — tape-internal accumulation only ever
/// sums two gradients that have already been shape-matched to their target
/// via `unbroadcast`, so no broadcast logic is needed here.
fn add_cpu_storage(a: &CpuStorage, b: &CpuStorage) -> CpuStorage {
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
            CpuBuffer::$variant(out)
        }};
    }

    let new_buffer = match (&*a.buffer, &*b.buffer) {
        (CpuBuffer::F32(_), CpuBuffer::F32(_)) => {
            add_variant!(
                F32,
                |s: &CpuStorage, i: &[usize]| s.get(i) as f32,
                |s: &CpuStorage, i: &[usize]| s.get(i) as f32
            )
        }
        (CpuBuffer::F64(_), CpuBuffer::F64(_)) => {
            add_variant!(
                F64,
                |s: &CpuStorage, i: &[usize]| s.get(i),
                |s: &CpuStorage, i: &[usize]| s.get(i)
            )
        }
        _ => {
            add_variant!(
                F32,
                |s: &CpuStorage, i: &[usize]| s.get(i) as f32,
                |s: &CpuStorage, i: &[usize]| s.get(i) as f32
            )
        }
    };

    CpuStorage::from_contiguous(new_buffer, a.shape.to_vec())
}

/// Right-align `grad.shape` and `target_shape`; sum-reduce over any leading
/// axis present in `grad.shape` but absent from `target_shape` (squeezing
/// it away entirely), then sum-reduce (with keepdim) over any axis where
/// `target_shape` has size 1 but `grad`'s corresponding axis is `>1`.
///
/// A no-op (returns a clone) when `grad.shape.dims() == target_shape`.
pub(crate) fn unbroadcast(grad: &CpuStorage, target_shape: &[usize]) -> Result<CpuStorage> {
    if grad.shape.dims() == target_shape {
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
fn sum_dim_squeeze(storage: &CpuStorage, axis: usize) -> CpuStorage {
    let reduced = sum_dim_keepdim(storage, axis);
    let mut new_shape = reduced.shape.to_vec();
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
fn sum_dim_keepdim(storage: &CpuStorage, axis: usize) -> CpuStorage {
    let mut out_shape = storage.shape.to_vec();
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
            CpuBuffer::$variant(out)
        }};
    }

    let new_buffer = match &*storage.buffer {
        CpuBuffer::F32(_) => reduce_variant!(F32, |v: f64| v as f32),
        CpuBuffer::F64(_) => reduce_variant!(F64, |v: f64| v),
        CpuBuffer::U8(_) => reduce_variant!(U8, |v: f64| v as u8),
        CpuBuffer::U32(_) => reduce_variant!(U32, |v: f64| v as u32),
        CpuBuffer::I64(_) => reduce_variant!(I64, |v: f64| v as i64),
        CpuBuffer::F16(_) => reduce_variant!(F16, |v: f64| half::f16::from_f64(v)),
        CpuBuffer::BF16(_) => reduce_variant!(BF16, |v: f64| half::bf16::from_f64(v)),

        CpuBuffer::Q8_0(_) => panic!("sum_dim_keepdim not supported on Q8_0 buffer"),
    };

    CpuStorage::from_contiguous(new_buffer, out_shape)
}

/// Compute the flat row-major index of `idx` within `shape`.
fn flatten_index(idx: &[usize], shape: &[usize]) -> usize {
    let strides = crate::cpu::stride::contiguous_strides(shape);
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
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::storage::CpuBuffer;

    /// `scalar`.
    fn scalar(v: f32) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(vec![v]), vec![])
    }

    /// `vector`.
    fn vector(v: Vec<f32>) -> CpuStorage {
        let len = v.len();
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![len])
    }

    /// `matrix`.
    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![rows, cols])
    }

    // --- unbroadcast standalone tests (CPUBACK-06) ---

    #[test]
    /// `unbroadcast_bias_vector_b_n_to_n`.
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
    /// `unbroadcast_scalar_target_sums_all_axes`.
    fn unbroadcast_scalar_target_sums_all_axes() {
        // grad shape [4,3], forward-broadcast from a scalar `[]`, summed
        // back to `[]` (scalar case).
        let grad = matrix(vec![1.0; 12], 4, 3);
        let result = unbroadcast(&grad, &[]).unwrap();
        assert_eq!(result.shape, Vec::<usize>::new());
        assert_eq!(result.get(&[]), 12.0);
    }

    #[test]
    /// `unbroadcast_same_shape_is_noop`.
    fn unbroadcast_same_shape_is_noop() {
        let grad = vector(vec![1.0, 2.0, 3.0]);
        let result = unbroadcast(&grad, &[3]).unwrap();
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.get(&[0]), 1.0);
        assert_eq!(result.get(&[1]), 2.0);
        assert_eq!(result.get(&[2]), 3.0);
    }

    // --- tape accumulation tests (CPUBACK-05) ---

    #[test]
    /// `backward_seeds_loss_gradient_with_ones`.
    fn backward_seeds_loss_gradient_with_ones() {
        let loss = scalar(5.0);
        let grads = backward(&loss).unwrap();
        let g = grads.get(loss.id).unwrap();
        assert_eq!(g.get(&[]), 1.0);
    }

    #[test]
    /// `backward_accumulates_not_overwrites_on_tensor_reuse`.
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
            backward: Box::new(|grad_out: &CpuStorage| {
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
            backward: Box::new(|grad_out: &CpuStorage| {
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
            backward: Box::new(|_grad_out: &CpuStorage| {
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
    /// `backward_drains_tape_and_second_call_is_not_contaminated`.
    fn backward_drains_tape_and_second_call_is_not_contaminated() {
        // First independent small graph.
        let x1 = scalar(1.0);
        let out1 = scalar(2.0);
        push(TapeEntry {
            output_id: out1.id,
            input_ids: vec![x1.id],
            backward: Box::new(|grad_out: &CpuStorage| {
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
            backward: Box::new(|grad_out: &CpuStorage| {
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
    /// `tape_len_is_zero_immediately_after_any_backward_call`.
    fn tape_len_is_zero_immediately_after_any_backward_call() {
        let x = scalar(1.0);
        let out = scalar(2.0);
        push(TapeEntry {
            output_id: out.id,
            input_ids: vec![x.id],
            backward: Box::new(|grad_out: &CpuStorage| vec![grad_out.clone()]),
        });
        let _ = backward(&out).unwrap();
        assert_eq!(len(), 0);
    }

    #[test]
    #[should_panic(expected = "NaN or Infinity detected in gradient")]
    /// `backward_with_nan_check_panics_on_nan`.
    fn backward_with_nan_check_panics_on_nan() {
        let x = scalar(1.0);
        let out = scalar(2.0);
        push(TapeEntry {
            output_id: out.id,
            input_ids: vec![x.id],
            backward: Box::new(|_grad_out: &CpuStorage| vec![scalar(f32::NAN)]),
        });
        let _ = backward_with_nan_check(&out);
    }
}
