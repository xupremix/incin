use super::*;
use incin_core::tensor::dtype::DTypeId;

/// `matrix`.
fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> CpuStorage {
    CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![rows, cols])
}

/// `f32_vec`.
fn f32_vec(s: &CpuStorage) -> Vec<f32> {
    match &*s.buffer {
        CpuBuffer::F32(v) => v.clone(),
        _ => panic!("expected F32 buffer"),
    }
}

#[test]
/// `matmul_forward_hand_computed_2x3_times_3x4`.
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
/// `matmul_forward_transposed_lhs_view_is_correct_without_materializing`.
fn matmul_forward_transposed_lhs_view_is_correct_without_materializing() {
    // Original storage is [3,2] = [[1,4],[2,5],[3,6]]; transpose(0,1)
    // gives a non-contiguous [2,3] view = [[1,2,3],[4,5,6]] (same
    // logical values as the previous test's `lhs`), read directly
    // through strides (no .contiguous() call in matmul_impl itself).
    let original = matrix(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2);
    let lhs = original.transpose(0, 1).unwrap(); // [2,3], non-contiguous
    assert!(!crate::cpu::stride::is_contiguous(&lhs.shape, &lhs.strides));

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
/// `matmul_backward_matches_hand_computed_gradients`.
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
    // rhs^T is [4,3] (rows = rhs's 4 columns transposed to rows length 3).
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
/// `matmul_shape_incompatible_returns_err_not_panic`.
fn matmul_shape_incompatible_returns_err_not_panic() {
    let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let rhs = matrix(vec![0.0; 20], 4, 5);
    let result = matmul_impl(&lhs, &rhs);
    assert!(result.is_err());
}

/// `tensor`.
fn tensor(v: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
    CpuStorage::from_contiguous(CpuBuffer::F32(v), shape)
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

// --- Task 2: batched matmul backward (gradcheck) ---

use crate::cpu::gradcheck::gradcheck;

/// Wraps `batched_matmul_impl` with `sum_all` so `gradcheck` (which
/// requires a scalar-output op) can drive it.
fn batched_matmul_sum_op(inputs: &[CpuStorage]) -> CpuStorage {
    let out = batched_matmul_impl(&inputs[0], &inputs[1]).unwrap();
    crate::cpu::ops::reduce::sum_all(&out).unwrap()
}

/// Test 1: gradcheck on `batched_matmul_impl` for the UNBATCHED
/// degenerate case (`[2,3]`/`[3,4]`) reports `max_relative_error < 1e-2`.
///
/// Uses small-magnitude values (not the 1..18 range used by the
/// hand-computed forward/backward tests above): `sum_all` over the full
/// batch*M*N output accumulates enough terms that larger-magnitude
/// inputs push the f32 finite-difference numerator into
/// catastrophic-cancellation noise at `eps=1e-4` (observed empirically:
/// values up to 18 produced ~5% relative error purely from f32
/// subtraction rounding, not a gradient bug — confirmed by the
/// analytic gradient exactly matching the hand-computed reference in
/// `batched_matmul_gradcheck_*`'s sibling forward/backward tests above).
#[test]
fn batched_matmul_gradcheck_unbatched_degenerate() {
    let lhs = matrix(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2, 3);
    let rhs = matrix(
        vec![0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        3,
        4,
    );
    let max_rel_err = gradcheck(batched_matmul_sum_op, &[lhs, rhs], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gradcheck max relative error too high: {max_rel_err}"
    );
}

/// Test 2: gradcheck on `batched_matmul_impl` for the EQUAL-BATCH case
/// (`[2,3,4]`/`[2,4,5]`) reports `max_relative_error < 1e-2`.
#[test]
fn batched_matmul_gradcheck_equal_batch() {
    let lhs_data: Vec<f32> = (1..=24).map(|x| x as f32 * 0.01).collect();
    let rhs_data: Vec<f32> = (1..=40).map(|x| x as f32 * 0.01).collect();
    let lhs = tensor(lhs_data, vec![2, 3, 4]);
    let rhs = tensor(rhs_data, vec![2, 4, 5]);
    let max_rel_err = gradcheck(batched_matmul_sum_op, &[lhs, rhs], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gradcheck max relative error too high: {max_rel_err}"
    );
}

/// Test 3: gradcheck on `batched_matmul_impl` for the
/// BATCH-BROADCAST-LEFT case (`[1,3,4]`/`[2,4,5]`) reports
/// `max_relative_error < 1e-2`, AND `grad_lhs`'s shape equals the
/// operand's OWN original `[1,3,4]` shape (proving `unbroadcast`
/// correctly reduced the broadcast-expanded `[2,3,4]`-shaped
/// intermediate gradient back down, not left at the broadcast shape).
#[test]
fn batched_matmul_gradcheck_batch_broadcast_left() {
    let lhs_data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.01).collect();
    let rhs_data: Vec<f32> = (1..=40).map(|x| x as f32 * 0.01).collect();
    let lhs = tensor(lhs_data, vec![1, 3, 4]);
    let rhs = tensor(rhs_data, vec![2, 4, 5]);
    let (lhs_id, rhs_id) = (lhs.id, rhs.id);

    let max_rel_err = gradcheck(batched_matmul_sum_op, &[lhs.clone(), rhs.clone()], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gradcheck max relative error too high: {max_rel_err}"
    );

    // Re-run once more, outside gradcheck's internal tape usage, to
    // directly inspect grad_lhs's shape after a real backward() walk.
    let out = batched_matmul_impl(&lhs, &rhs).unwrap();
    let sum = crate::cpu::ops::reduce::sum_all(&out).unwrap();
    let grads = tape::backward(&sum).unwrap();
    let grad_lhs = grads.get(lhs_id).expect("grad_lhs should exist");
    let _ = rhs_id;
    assert_eq!(grad_lhs.shape, vec![1, 3, 4]);
}

/// Test 4: gradcheck on `batched_matmul_impl` for the
/// BATCH-BROADCAST-RIGHT case (`[2,3,4]`/`[1,4,5]`) mirrors Test 3 for
/// `grad_rhs`.
#[test]
fn batched_matmul_gradcheck_batch_broadcast_right() {
    let lhs_data: Vec<f32> = (1..=24).map(|x| x as f32 * 0.01).collect();
    let rhs_data: Vec<f32> = (1..=20).map(|x| x as f32 * 0.01).collect();
    let lhs = tensor(lhs_data, vec![2, 3, 4]);
    let rhs = tensor(rhs_data, vec![1, 4, 5]);
    let rhs_id = rhs.id;

    let max_rel_err = gradcheck(batched_matmul_sum_op, &[lhs.clone(), rhs.clone()], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gradcheck max relative error too high: {max_rel_err}"
    );

    let out = batched_matmul_impl(&lhs, &rhs).unwrap();
    let sum = crate::cpu::ops::reduce::sum_all(&out).unwrap();
    let grads = tape::backward(&sum).unwrap();
    let grad_rhs = grads.get(rhs_id).expect("grad_rhs should exist");
    assert_eq!(grad_rhs.shape, vec![1, 4, 5]);
}

/// Test 5: gradcheck on `batched_matmul_impl` for a `>3D` case
/// (`[2,2,3,4]`/`[2,2,4,5]`) reports `max_relative_error < 1e-2`.
#[test]
fn batched_matmul_gradcheck_rank4() {
    let lhs_data: Vec<f32> = (1..=48).map(|x| x as f32 * 0.002).collect();
    let rhs_data: Vec<f32> = (1..=80).map(|x| x as f32 * 0.002).collect();
    let lhs = tensor(lhs_data, vec![2, 2, 3, 4]);
    let rhs = tensor(rhs_data, vec![2, 2, 4, 5]);
    let max_rel_err = gradcheck(batched_matmul_sum_op, &[lhs, rhs], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gradcheck max relative error too high: {max_rel_err}"
    );
}

// --- PRF-002: kernel isolation, layout coverage, and tape accounting ---

/// Run one `[m,k] @ [k,n]` product through `gemm`'s real dispatch and
/// through `scalar_gemm` directly, and return both. Comparing the two is
/// how every layout below is checked without a second reference
/// implementation: `scalar_gemm` is the kernel that is always correct,
/// and `gemm` is whichever specialization this build and this layout
/// selected.
fn dispatched_and_scalar(
    m: usize,
    k: usize,
    n: usize,
    lhs: &CpuStorage,
    rhs: &CpuStorage,
) -> (Vec<f32>, Vec<f32>) {
    let (lhs_view, rhs_view) = (MatrixView::trailing(lhs), MatrixView::trailing(rhs));
    let mut dispatched = vec![0f32; m * n];
    gemm(m, k, n, lhs_view, rhs_view, &mut dispatched);
    let mut scalar = vec![0f32; m * n];
    scalar_gemm(m, k, n, lhs_view, rhs_view, &mut scalar);
    (dispatched, scalar)
}

/// Whatever kernel this build selects has to agree with `scalar_gemm` on
/// every layout the views can describe: both operands contiguous, either
/// one transposed, both transposed, and a non-zero view offset.
///
/// The tolerance is relative and loose enough for reassociation, because
/// that is the only difference any of these kernels is allowed to have.
#[test]
fn every_layout_agrees_with_the_kernel_that_is_always_correct() {
    let base_lhs: Vec<f32> = (1..=24).map(|x| x as f32 * 0.25).collect();
    let base_rhs: Vec<f32> = (1..=30).map(|x| x as f32 * 0.125).collect();

    let contiguous_lhs = matrix(base_lhs.clone(), 4, 6);
    let contiguous_rhs = matrix(base_rhs.clone(), 6, 5);
    // A [6,4] buffer read as a transposed [4,6] view: same logical matrix,
    // a column stride of 6 instead of 1.
    let transposed_lhs = matrix(base_lhs.clone(), 6, 4).transpose(0, 1).unwrap();
    let transposed_rhs = matrix(base_rhs.clone(), 5, 6).transpose(0, 1).unwrap();
    // A view that starts partway into its buffer, so `offset` is not zero.
    let offset_rhs = matrix(
        (0..36).map(|x| x as f32 * 0.125).collect::<Vec<f32>>(),
        6,
        6,
    )
    .narrow(1, 1, 5)
    .unwrap();

    let cases: [(&str, &CpuStorage, &CpuStorage); 5] = [
        ("both contiguous", &contiguous_lhs, &contiguous_rhs),
        ("transposed lhs", &transposed_lhs, &contiguous_rhs),
        ("transposed rhs", &contiguous_lhs, &transposed_rhs),
        ("both transposed", &transposed_lhs, &transposed_rhs),
        ("offset rhs view", &contiguous_lhs, &offset_rhs),
    ];

    for (name, lhs, rhs) in cases {
        let (dispatched, scalar) = dispatched_and_scalar(4, 6, 5, lhs, rhs);
        for (index, (got, want)) in dispatched.iter().zip(&scalar).enumerate() {
            let tolerance = 1e-4 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "{name}: element {index} was {got}, scalar kernel says {want}"
            );
        }
    }
}

/// The same agreement at a size large enough to cross `cpu-blas`'s
/// crossover, so a build with that feature on is actually exercising the
/// blocked kernel here rather than repeating the test above.
#[test]
fn a_large_product_agrees_with_the_kernel_that_is_always_correct() {
    let (m, k, n) = (96, 80, 72);
    let lhs = matrix(
        (0..m * k)
            .map(|x| ((x % 17) as f32 - 8.0) * 0.125)
            .collect(),
        m,
        k,
    );
    let rhs = matrix(
        (0..k * n)
            .map(|x| ((x % 23) as f32 - 11.0) * 0.0625)
            .collect(),
        k,
        n,
    );

    let (dispatched, scalar) = dispatched_and_scalar(m, k, n, &lhs, &rhs);
    for (index, (got, want)) in dispatched.iter().zip(&scalar).enumerate() {
        let tolerance = 1e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "element {index} was {got}, scalar kernel says {want}"
        );
    }
}

/// A contiguous non-`f32` matmul used to return all zeros: the
/// row-streaming branch was chosen on stride alone and then bailed out on
/// finding a buffer it could not read, leaving the zeroed output as the
/// answer. Values, not just the shape, are asserted here.
///
/// It then returned the right values wearing the wrong label, because
/// every kernel wrote `f32`. The dtype is asserted for the same reason the
/// values are: an operand's dtype surviving its own matmul is the property
/// `scaled_dot_product_attention` was silently losing.
#[test]
fn a_contiguous_f64_matmul_computes_values_and_keeps_its_dtype() {
    let lhs = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
    let identity =
        CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0, 0.0, 0.0, 1.0]), vec![2, 2]);

    let out = matmul_impl(&lhs, &identity).unwrap();
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.dtype, DTypeId::F64.descriptor());
    assert_eq!(
        (0..4)
            .map(|i| out.get(&[i / 2, i % 2]))
            .collect::<Vec<f64>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

/// The batched path splits on dtype separately from the unbatched one, so
/// it is asserted separately rather than assumed to follow.
#[test]
fn a_batched_non_f32_matmul_keeps_its_dtype() {
    let lhs = CpuStorage::from_contiguous(
        CpuBuffer::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        vec![2, 2, 2],
    );
    let rhs = CpuStorage::from_contiguous(
        CpuBuffer::F64(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        vec![2, 2, 2],
    );

    let out = batched_matmul_impl(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 2, 2]);
    assert_eq!(out.dtype, DTypeId::F64.descriptor());
    // Both slices multiply by the identity, so the operand comes back.
    assert_eq!(out.get(&[1, 1, 1]), 8.0);
    assert_eq!(out.get(&[0, 1, 0]), 3.0);
}

/// A batched matmul records exactly one tape entry, whatever the batch
/// count. It used to record one per batch slice plus one, because the
/// batch loop called the tape-recording unbatched entry point; those
/// extra entries were unreachable during a backward walk but still held
/// every intermediate slice alive.
#[test]
fn a_batched_matmul_records_one_tape_entry_per_call_not_one_per_slice() {
    let lhs = tensor((1..=96).map(|x| x as f32 * 0.01).collect(), vec![8, 3, 4]);
    let rhs = tensor((1..=160).map(|x| x as f32 * 0.01).collect(), vec![8, 4, 5]);

    let before = tape::depth();
    let out = batched_matmul_impl(&lhs, &rhs).unwrap();
    assert_eq!(
        tape::depth() - before,
        1,
        "an 8-slice batched matmul recorded more than one entry"
    );

    // And the walk itself must not leave anything behind either: the
    // backward closure recurses into the forward kernel, which no longer
    // records. Entries pushed during a walk are dead weight, because
    // `backward` drains the tape before it starts.
    let sum = crate::cpu::ops::reduce::sum_all(&out).unwrap();
    let grads = tape::backward(&sum).unwrap();
    assert!(grads.get(lhs.id).is_some());
    assert_eq!(
        tape::depth(),
        0,
        "backward left entries on a tape it had already drained"
    );
}

/// The blocked kernel has to actually be the one running when `cpu-blas`
/// is on, and has to actually decline below its crossover. Without this,
/// the agreement tests above would still pass on a build where
/// `blocked_gemm` silently returned `false` for every input.
#[cfg(feature = "cpu-blas")]
#[test]
fn the_blocked_kernel_takes_large_products_and_declines_small_ones() {
    let big_lhs = matrix(vec![0.5; 96 * 80], 96, 80);
    let big_rhs = matrix(vec![0.25; 80 * 72], 80, 72);
    let mut out = vec![0f32; 96 * 72];
    assert!(blocked_gemm(
        96,
        80,
        72,
        MatrixView::trailing(&big_lhs),
        MatrixView::trailing(&big_rhs),
        &mut out,
    ));

    let small_lhs = matrix(vec![0.5; 8 * 8], 8, 8);
    let small_rhs = matrix(vec![0.25; 8 * 8], 8, 8);
    let mut out = vec![0f32; 64];
    assert!(!blocked_gemm(
        8,
        8,
        8,
        MatrixView::trailing(&small_lhs),
        MatrixView::trailing(&small_rhs),
        &mut out,
    ));

    // And it declines a dtype it cannot read, rather than reinterpreting it.
    let f64_lhs = CpuStorage::from_contiguous(CpuBuffer::F64(vec![0.5; 96 * 80]), vec![96, 80]);
    let mut out = vec![0f32; 96 * 72];
    assert!(!blocked_gemm(
        96,
        80,
        72,
        MatrixView::trailing(&f64_lhs),
        MatrixView::trailing(&big_rhs),
        &mut out,
    ));
}

/// A batch-broadcast operand that is also non-contiguous is the case the
/// previous implementation had to materialize: expanding `[1,3,4]` to
/// `[6,3,4]` and reshaping copied the operand six times, and a transposed
/// operand made the copy unavoidable. The plan reads it in place, so this
/// asserts the values are still right.
#[test]
fn a_broadcast_non_contiguous_operand_matches_its_per_slice_reference() {
    // [1,4,3] transposed on its last two axes is a non-contiguous [1,3,4].
    let lhs_source = tensor((1..=12).map(|x| x as f32).collect(), vec![1, 4, 3]);
    let lhs = transpose_last2(&lhs_source);
    assert!(!crate::cpu::stride::is_contiguous(&lhs.shape, &lhs.strides));
    assert_eq!(lhs.shape, vec![1, 3, 4]);

    let rhs = tensor((1..=120).map(|x| x as f32).collect(), vec![6, 4, 5]);

    let out = batched_matmul_impl(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![6, 3, 5]);

    // The single lhs slice, materialized once, against each rhs slice.
    let lhs_flat: Vec<f32> = (0..3)
        .flat_map(|row| (0..4).map(move |col| (row, col)))
        .map(|(row, col)| lhs.get(&[0, row, col]) as f32)
        .collect();
    let out_data = f32_vec(&out);
    for slice in 0..6 {
        let rhs_slice: Vec<f32> = (0..20).map(|x| (slice * 20 + x + 1) as f32).collect();
        let reference =
            matmul_impl(&matrix(lhs_flat.clone(), 3, 4), &matrix(rhs_slice, 4, 5)).unwrap();
        assert_eq!(
            &out_data[slice * 15..slice * 15 + 15],
            &f32_vec(&reference)[..],
            "batch slice {slice}"
        );
    }
}

/// An operand whose batch axes are broadcast on both sides at once, which
/// is where a batch index that ignores the coalescing the plan performs
/// would go wrong. `[2,1,3,4]` against `[1,5,4,6]` broadcasts each
/// operand along the axis the other one owns.
#[test]
fn batch_axes_broadcast_on_both_sides_index_the_right_slices() {
    let lhs = tensor((1..=24).map(|x| x as f32 * 0.5).collect(), vec![2, 1, 3, 4]);
    let rhs = tensor(
        (1..=120).map(|x| x as f32 * 0.25).collect(),
        vec![1, 5, 4, 6],
    );

    let out = batched_matmul_impl(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 5, 3, 6]);

    let out_data = f32_vec(&out);
    for outer in 0..2 {
        let lhs_slice: Vec<f32> = (0..12).map(|x| (outer * 12 + x + 1) as f32 * 0.5).collect();
        for inner in 0..5 {
            let rhs_slice: Vec<f32> = (0..24)
                .map(|x| (inner * 24 + x + 1) as f32 * 0.25)
                .collect();
            let reference =
                matmul_impl(&matrix(lhs_slice.clone(), 3, 4), &matrix(rhs_slice, 4, 6)).unwrap();
            let start = (outer * 5 + inner) * 18;
            assert_eq!(
                &out_data[start..start + 18],
                &f32_vec(&reference)[..],
                "slice ({outer}, {inner})"
            );
        }
    }
}
