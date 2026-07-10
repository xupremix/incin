//! Naive, stride-aware, unbatched-2D-only `matmul` for `NativeBackend`.
//!
//! Per CONTEXT.md's explicit Phase 1 scope decision, batch-broadcast matmul
//! (>2D operands) is out of scope here (deferred to Phase 3 / NATBACK-07).
//! The inner loop reads through each operand's own shape/strides/offset via
//! `NativeStorage::get` directly — it never forces a `.contiguous()`
//! materialization first, so a transposed (non-contiguous) view produces
//! correct values without a hidden copy (NATBACK-02 / Pitfall 3).
//!
//! This file only contributes a plain function, `matmul_impl`, rather than
//! its own `impl TensorOps` block: Rust does not allow two separate `impl
//! TensorOps<..> for NativeBackend<..>` blocks for the same trait+type
//! across two files, so `ops/shape_ops.rs`'s single `TensorOps` impl block
//! calls into `matmul_impl` for its `matmul` method.

use kindle_core::err::Error;
use kindle_core::prelude::Result;

use crate::storage::{NativeBuffer, NativeStorage};
use crate::stride;
use crate::tape::{self, TapeEntry};

/// Swap the two axes of a 2D `NativeStorage` (thin wrapper over
/// `NativeStorage::transpose(0, 1)`, reused by the backward closure so the
/// gradient composition is built from already-tested primitives rather than
/// a bespoke derivation).
fn transpose_2d(t: &NativeStorage) -> NativeStorage {
    t.transpose(0, 1)
        .expect("2D transpose of a 2D matmul operand cannot fail")
}

/// Swap ONLY the last two axes of an N-D (`N >= 2`) `NativeStorage`, leaving
/// every leading batch axis untouched. Generalizes `transpose_2d` to the
/// batched case; both are thin wrappers over the same
/// `NativeStorage::transpose(dim1, dim2)` primitive.
pub(crate) fn transpose_last2(t: &NativeStorage) -> NativeStorage {
    let r = t.shape.len();
    t.transpose(r - 2, r - 1)
        .expect("transpose of the last two axes of a rank>=2 tensor cannot fail")
}

/// Batched matmul: broadcasts both operands' batch dims (every axis except
/// the trailing 2), flattens to `[batch, M, K]`/`[batch, K, N]`, and loops
/// calling `matmul_impl` per batch index (D-01: naive, no rayon).
///
/// Handles the unbatched case too (`lhs.shape.len() == 2 && rhs.shape.len()
/// == 2`) as the degenerate `batch_total == 1` case — ONE uniform code path
/// for all batch ranks, no `<=3D` vs `>3D` special-casing.
///
/// This function ONLY implements the forward computation (Task 1). Task 2
/// adds this op's own hand-composed top-level `TapeEntry` (using
/// `transpose_last2` + recursive `batched_matmul_impl` calls +
/// `tape::unbroadcast`), layered on top of this forward result.
pub(crate) fn batched_matmul_impl(
    lhs: &NativeStorage,
    rhs: &NativeStorage,
) -> Result<NativeStorage> {
    let (l_rank, r_rank) = (lhs.shape.len(), rhs.shape.len());
    if l_rank < 2 || r_rank < 2 {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: vec![2],
            got: vec![l_rank, r_rank],
            msg: format!(
                "batched matmul requires both operands to have rank >= 2; got lhs.shape={:?}, rhs.shape={:?}",
                lhs.shape, rhs.shape
            ),
        });
    }

    let (m, k_lhs) = (lhs.shape[l_rank - 2], lhs.shape[l_rank - 1]);
    let (k_rhs, n) = (rhs.shape[r_rank - 2], rhs.shape[r_rank - 1]);
    if k_lhs != k_rhs {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: vec![k_lhs],
            got: vec![k_rhs],
            msg: format!(
                "matmul inner dims must match: lhs.shape={:?} (K={k_lhs}), rhs.shape={:?} (K={k_rhs})",
                lhs.shape, rhs.shape
            ),
        });
    }

    // Batch dims = every axis except the trailing 2, right-aligned per
    // stride::broadcast_shape's existing NumPy-style rule (REUSED, not
    // reimplemented).
    let lhs_batch = &lhs.shape[..l_rank - 2];
    let rhs_batch = &rhs.shape[..r_rank - 2];
    let out_batch = stride::broadcast_shape(lhs_batch, rhs_batch)?;

    let mut lhs_target = out_batch.clone();
    lhs_target.extend_from_slice(&[m, k_lhs]);
    let mut rhs_target = out_batch.clone();
    rhs_target.extend_from_slice(&[k_rhs, n]);

    let lhs_b = lhs.broadcast_as(&lhs_target)?;
    let rhs_b = rhs.broadcast_as(&rhs_target)?;

    // `Iterator::product()` over an empty `out_batch` (the unbatched,
    // no-batch-dims-at-all case) already correctly yields `1` (empty
    // product) with no `.max(1)` guard needed — and a genuine size-0 batch
    // axis correctly yields `batch_total == 0` via the same plain product,
    // so this does NOT conflate a size-0 axis with the unbatched case
    // (Pitfall 6).
    let batch_total: usize = out_batch.iter().product();

    let lhs_flat = lhs_b.reshape(&[batch_total, m, k_lhs])?;
    let rhs_flat = rhs_b.reshape(&[batch_total, k_rhs, n])?;

    let mut out_data: Vec<f32> = Vec::with_capacity(batch_total * m * n);
    for b in 0..batch_total {
        let lhs_slice = lhs_flat.narrow(0, b, 1)?.reshape(&[m, k_lhs])?;
        let rhs_slice = rhs_flat.narrow(0, b, 1)?.reshape(&[k_rhs, n])?;
        let out_slice = matmul_impl(&lhs_slice, &rhs_slice)?;
        for mi in 0..m {
            for ni in 0..n {
                out_data.push(out_slice.get(&[mi, ni]) as f32);
            }
        }
    }

    let mut out_shape = out_batch;
    out_shape.extend_from_slice(&[m, n]);

    Ok(NativeStorage::from_contiguous(
        NativeBuffer::F32(out_data),
        out_shape,
    ))
}

/// Naive triple-nested-loop 2D matmul: `lhs` (`[M,K]`) @ `rhs` (`[K,N]`) ->
/// `[M,N]`. Reads through each operand's own strides/offset (via
/// `NativeStorage::get`), so a transposed (non-contiguous) operand is
/// handled correctly without an implicit contiguous materialization.
///
/// Pushes a `TapeEntry` whose backward closure computes
/// `grad_lhs = grad_out @ rhs^T` and `grad_rhs = lhs^T @ grad_out`, composed
/// by recursing into `matmul_impl` itself plus `transpose_2d` — not a
/// bespoke hand-derived kernel.
pub(crate) fn matmul_impl(lhs: &NativeStorage, rhs: &NativeStorage) -> Result<NativeStorage> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 || lhs.shape[1] != rhs.shape[0] {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            expected: vec![lhs.shape[0], rhs.shape.first().copied().unwrap_or(0)],
            got: rhs.shape.clone(),
            msg: format!(
                "matmul requires unbatched 2D operands with lhs.shape[1] == rhs.shape[0]; got lhs={:?}, rhs={:?}",
                lhs.shape, rhs.shape
            ),
        });
    }

    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let n = rhs.shape[1];

    let mut out = Vec::with_capacity(m * n);
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f64;
            for ki in 0..k {
                acc += lhs.get(&[mi, ki]) * rhs.get(&[ki, ni]);
            }
            out.push(acc as f32);
        }
    }

    let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), vec![m, n]);

    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &NativeStorage| {
            let grad_lhs = matmul_impl(grad_out, &transpose_2d(&rhs_capture))
                .expect("grad_lhs = grad_out @ rhs^T cannot fail (shapes proven compatible)");
            let grad_rhs = matmul_impl(&transpose_2d(&lhs_capture), grad_out)
                .expect("grad_rhs = lhs^T @ grad_out cannot fail (shapes proven compatible)");
            vec![grad_lhs, grad_rhs]
        }),
    });

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![rows, cols])
    }

    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    #[test]
    fn matmul_forward_hand_computed_2x3_times_3x4() {
        // lhs = [[1,2,3],[4,5,6]] (2x3)
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        // rhs = [[7,8,9,10],[11,12,13,14],[15,16,17,18]] (3x4)
        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let out = matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        // Row 0: [1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18]
        //      = [7+22+45, 8+24+48, 9+26+51, 10+28+54] = [74, 80, 86, 92]
        // Row 1: [4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18]
        //      = [28+55+90, 32+60+96, 36+65+102, 40+70+108] = [173, 188, 203, 218]
        assert_eq!(
            f32_vec(&out),
            vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
        );
    }

    #[test]
    fn matmul_forward_transposed_lhs_view_is_correct_without_materializing() {
        // Original storage is [3,2] = [[1,4],[2,5],[3,6]]; transpose(0,1)
        // gives a non-contiguous [2,3] view = [[1,2,3],[4,5,6]] (same
        // logical values as the previous test's `lhs`), read directly
        // through strides (no .contiguous() call in matmul_impl itself).
        let original = matrix(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2);
        let lhs = original.transpose(0, 1).unwrap(); // [2,3], non-contiguous
        assert!(!crate::stride::is_contiguous(&lhs.shape, &lhs.strides));

        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let out = matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        assert_eq!(
            f32_vec(&out),
            vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
        );
    }

    #[test]
    fn matmul_backward_matches_hand_computed_gradients() {
        // lhs [2,3], rhs [3,4] as above; grad_out is a synthetic [2,4] all-ones-ish
        // matrix with distinct values so the composition is unambiguous.
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let out = matmul_impl(&lhs, &rhs).unwrap();

        let grads = tape::backward(&out).unwrap();
        let lhs_grad = grads.get(lhs.id).expect("lhs should have a gradient");
        let rhs_grad = grads.get(rhs.id).expect("rhs should have a gradient");

        // grad_out = ones_like(out) = [2,4] all ones.
        // grad_lhs = grad_out @ rhs^T : [2,4] @ [4,3] -> [2,3]
        // rhs^T rows are rhs's columns: col0=[7,11,15], col1=[8,12,16],
        // col2=[9,13,17], col3=[10,14,18]. Each output row of grad_lhs is the
        // sum of rhs^T's rows (since grad_out row is all ones):
        // sum over rhs^T's 4 rows (its columns as rows) = per-column sums of rhs:
        // col0: 7+11+15=33, col1: 8+12+16=36, col2: 9+13+17=39, col3: 10+14+18=42
        // Wait: rhs^T is [4,3] (rows = rhs's 4 columns transposed to rows length 3).
        // rhs^T row i = rhs's column i as a length-3 vector: [rhs[0][i], rhs[1][i], rhs[2][i]]
        // grad_lhs[m][k] = sum_n grad_out[m][n] * rhs^T[n][k] = sum_n rhs[k][n] (since grad_out=1)
        //               = sum over n of rhs[k][n] = row-sum of rhs's row k.
        // rhs row 0 = [7,8,9,10] sum=34; row1=[11,12,13,14] sum=50; row2=[15,16,17,18] sum=66
        assert_eq!(lhs_grad.shape, vec![2, 3]);
        assert_eq!(f32_vec(lhs_grad), vec![34.0, 50.0, 66.0, 34.0, 50.0, 66.0]);

        // grad_rhs = lhs^T @ grad_out : [3,2] @ [2,4] -> [3,4]
        // grad_rhs[k][n] = sum_m lhs^T[k][m] * grad_out[m][n] = sum_m lhs[m][k] (since grad_out=1)
        //               = column-sum of lhs's column k.
        // lhs col0 = [1,4] sum=5; col1=[2,5] sum=7; col2=[3,6] sum=9
        assert_eq!(rhs_grad.shape, vec![3, 4]);
        assert_eq!(
            f32_vec(rhs_grad),
            vec![5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0]
        );
    }

    #[test]
    fn matmul_shape_incompatible_returns_err_not_panic() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = matrix(vec![0.0; 20], 4, 5);
        let result = matmul_impl(&lhs, &rhs);
        assert!(result.is_err());
    }

    fn tensor(v: Vec<f32>, shape: Vec<usize>) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), shape)
    }

    /// Test 1 (unbatched, degenerate case): `batched_matmul_impl` on a
    /// `[2,3]`/`[3,4]` pair (both rank 2, `batch_total == 1` degenerate case
    /// flowing through the SAME code path as any batched call) produces
    /// identical values to `matmul_impl` on the same inputs.
    #[test]
    fn batched_matmul_unbatched_degenerate_matches_matmul_impl() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let expected = matmul_impl(&lhs, &rhs).unwrap();
        let out = batched_matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, expected.shape);
        assert_eq!(f32_vec(&out), f32_vec(&expected));
    }

    /// Test 2 (equal-batch): `[2,3,4]`/`[2,4,5]` operands produce shape
    /// `[2,3,5]` matching a hand-computed per-batch-slice reference (2
    /// independent `[3,4]@[4,5]` matmuls).
    #[test]
    fn batched_matmul_equal_batch_matches_per_slice_reference() {
        // Batch 0: lhs = [[1..12]] reshaped [3,4], rhs = [1..20] reshaped [4,5]
        let lhs_b0: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let lhs_b1: Vec<f32> = (13..=24).map(|x| x as f32).collect();
        let rhs_b0: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        let rhs_b1: Vec<f32> = (21..=40).map(|x| x as f32).collect();

        let mut lhs_data = lhs_b0.clone();
        lhs_data.extend(lhs_b1.clone());
        let mut rhs_data = rhs_b0.clone();
        rhs_data.extend(rhs_b1.clone());

        let lhs = tensor(lhs_data, vec![2, 3, 4]);
        let rhs = tensor(rhs_data, vec![2, 4, 5]);

        let out = batched_matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3, 5]);

        let ref0 = matmul_impl(&matrix(lhs_b0, 3, 4), &matrix(rhs_b0, 4, 5)).unwrap();
        let ref1 = matmul_impl(&matrix(lhs_b1, 3, 4), &matrix(rhs_b1, 4, 5)).unwrap();

        let out_data = f32_vec(&out);
        assert_eq!(&out_data[0..15], &f32_vec(&ref0)[..]);
        assert_eq!(&out_data[15..30], &f32_vec(&ref1)[..]);
    }

    /// Test 3 (batch-broadcast-left): `[1,3,4]`/`[2,4,5]` operands produce
    /// shape `[2,3,5]`, with the `[1,...]` operand's single batch slice
    /// correctly reused for both output batch indices.
    #[test]
    fn batched_matmul_batch_broadcast_left_reuses_single_slice() {
        let lhs_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let rhs_b0: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        let rhs_b1: Vec<f32> = (21..=40).map(|x| x as f32).collect();
        let mut rhs_data = rhs_b0.clone();
        rhs_data.extend(rhs_b1.clone());

        let lhs = tensor(lhs_data.clone(), vec![1, 3, 4]);
        let rhs = tensor(rhs_data, vec![2, 4, 5]);

        let out = batched_matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3, 5]);

        let ref0 = matmul_impl(&matrix(lhs_data.clone(), 3, 4), &matrix(rhs_b0, 4, 5)).unwrap();
        let ref1 = matmul_impl(&matrix(lhs_data, 3, 4), &matrix(rhs_b1, 4, 5)).unwrap();

        let out_data = f32_vec(&out);
        assert_eq!(&out_data[0..15], &f32_vec(&ref0)[..]);
        assert_eq!(&out_data[15..30], &f32_vec(&ref1)[..]);
    }

    /// Test 4 (batch-broadcast-right): `[2,3,4]`/`[1,4,5]` mirrors Test 3
    /// with the broadcast on the other operand.
    #[test]
    fn batched_matmul_batch_broadcast_right_reuses_single_slice() {
        let lhs_b0: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let lhs_b1: Vec<f32> = (13..=24).map(|x| x as f32).collect();
        let rhs_data: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        let mut lhs_data = lhs_b0.clone();
        lhs_data.extend(lhs_b1.clone());

        let lhs = tensor(lhs_data, vec![2, 3, 4]);
        let rhs = tensor(rhs_data.clone(), vec![1, 4, 5]);

        let out = batched_matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3, 5]);

        let ref0 = matmul_impl(&matrix(lhs_b0, 3, 4), &matrix(rhs_data.clone(), 4, 5)).unwrap();
        let ref1 = matmul_impl(&matrix(lhs_b1, 3, 4), &matrix(rhs_data, 4, 5)).unwrap();

        let out_data = f32_vec(&out);
        assert_eq!(&out_data[0..15], &f32_vec(&ref0)[..]);
        assert_eq!(&out_data[15..30], &f32_vec(&ref1)[..]);
    }

    /// Test 5 (>3D): `[2,2,3,4]`/`[2,2,4,5]` (rank 4, two batch dims)
    /// produces shape `[2,2,3,5]` matching a hand-computed reference for at
    /// least one specific batch index (batch index (1,1), i.e. flattened
    /// batch index 3).
    #[test]
    fn batched_matmul_rank4_matches_reference_at_one_batch_index() {
        let total_batches = 4; // 2*2
        let mut lhs_data = Vec::new();
        let mut rhs_data = Vec::new();
        let mut lhs_slices = Vec::new();
        let mut rhs_slices = Vec::new();
        for b in 0..total_batches {
            let l: Vec<f32> = (0..12).map(|x| (x + b * 100) as f32).collect();
            let r: Vec<f32> = (0..20).map(|x| (x + b * 100) as f32).collect();
            lhs_data.extend(l.clone());
            rhs_data.extend(r.clone());
            lhs_slices.push(l);
            rhs_slices.push(r);
        }

        let lhs = tensor(lhs_data, vec![2, 2, 3, 4]);
        let rhs = tensor(rhs_data, vec![2, 2, 4, 5]);

        let out = batched_matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 2, 3, 5]);

        // Flattened batch index 3 corresponds to (1,1).
        let batch_idx = 3;
        let reference = matmul_impl(
            &matrix(lhs_slices[batch_idx].clone(), 3, 4),
            &matrix(rhs_slices[batch_idx].clone(), 4, 5),
        )
        .unwrap();
        let out_data = f32_vec(&out);
        let start = batch_idx * 15;
        assert_eq!(&out_data[start..start + 15], &f32_vec(&reference)[..]);
    }

    /// Test 6 (>3D with batch-dim broadcast): a rank-3 operand (`[1,3,4]`)
    /// broadcasting against a rank-4 operand (`[2,1,4,5]`) via
    /// `stride::broadcast_shape`'s existing leading-dim-insertion rule,
    /// producing the correctly-broadcast `[2,1,3,5]` output shape.
    #[test]
    fn batched_matmul_rank3_broadcasts_against_rank4_leading_dim_insertion() {
        let lhs_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let rhs_data: Vec<f32> = (1..=40).map(|x| x as f32).collect();

        let lhs = tensor(lhs_data, vec![1, 3, 4]);
        let rhs = tensor(rhs_data, vec![2, 1, 4, 5]);

        let out = batched_matmul_impl(&lhs, &rhs).unwrap();
        // lhs_batch = [1] right-aligned against rhs_batch = [2,1] ->
        // broadcast_shape([1], [2,1]) = [2,1] (leading dim inserted for lhs).
        assert_eq!(out.shape, vec![2, 1, 3, 5]);
    }

    /// Test 7 (Pitfall 6, size-0 batch): a `[0,3,4]`/`[0,4,5]` pair produces
    /// an empty (`[0,3,5]`) output without panicking.
    #[test]
    fn batched_matmul_size_zero_batch_produces_empty_output_without_panic() {
        let lhs = tensor(vec![], vec![0, 3, 4]);
        let rhs = tensor(vec![], vec![0, 4, 5]);
        let out = batched_matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![0, 3, 5]);
        assert_eq!(f32_vec(&out).len(), 0);
    }

    /// Test 8 (Pitfall 6, size-1 batch NOT unwrapped): a `[1,3,4]` operand
    /// batched against a `[5,4,6]` operand produces a `[5,3,6]` output (the
    /// size-1 batch dim is broadcast, not silently treated as
    /// unbatched-rank-2).
    #[test]
    fn batched_matmul_size_one_batch_is_broadcast_not_unwrapped() {
        let lhs_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let rhs_data: Vec<f32> = (1..=120).map(|x| x as f32).collect();

        let lhs = tensor(lhs_data, vec![1, 3, 4]);
        let rhs = tensor(rhs_data, vec![5, 4, 6]);

        let out = batched_matmul_impl(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![5, 3, 6]);
        assert_eq!(f32_vec(&out).len(), 5 * 3 * 6);
    }
}
