use super::*;
use crate::cpu::gradcheck::{F32_STEP, GRAD_TOL, gradcheck};
use crate::cpu::tape;

/// `matrix`.
fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> CpuStorage {
    CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![rows, cols])
}

/// `vector`.
fn vector(v: Vec<f32>) -> CpuStorage {
    let len = v.len();
    CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![len])
}

/// `f32_vec`.
fn f32_vec(s: &CpuStorage) -> Vec<f32> {
    match &*s.buffer {
        CpuBuffer::F32(v) => v.clone(),
        _ => panic!("expected F32 buffer"),
    }
}

// --- sum_all ---

#[test]
/// `sum_all_on_2x3_returns_correct_scalar`.
fn sum_all_on_2x3_returns_correct_scalar() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::sum_all(&t).unwrap();
    assert_eq!(out.shape, Vec::<usize>::new()); // scalar shape []
    assert_eq!(out.get(&[]), 21.0);
}

#[test]
/// `sum_all_backward_distributes_grad_uniformly`.
fn sum_all_backward_distributes_grad_uniformly() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::sum_all(&t).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have gradient");
    assert_eq!(g.shape, vec![2, 3]);
    // sum_all backward: every element receives grad_scalar = 1.0 (ones_like seed)
    assert_eq!(f32_vec(g), vec![1.0; 6]);
}

// --- prod_all / prod_dim ---

#[test]
/// `prod_all_on_2x3_returns_correct_scalar`.
fn prod_all_on_2x3_returns_correct_scalar() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::prod_all(&t).unwrap();
    assert_eq!(out.shape, Vec::<usize>::new());
    assert_eq!(out.get(&[]), 720.0);
}

#[test]
/// `prod_all_keeps_the_operand_dtype`.
fn prod_all_keeps_the_operand_dtype() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0, 2.0, 3.0, 4.0]), vec![4]);
    let out = crate::cpu::ops::reduce::prod_all(&t).unwrap();
    assert_eq!(out.dtype, DTypeId::F64.descriptor());
    assert_eq!(out.get(&[]), 24.0);
}

#[test]
/// `prod_dim_multiplies_along_the_named_axis_only`.
fn prod_dim_multiplies_along_the_named_axis_only() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::prod_dim(&t, 1).unwrap();
    assert_eq!(out.shape, vec![2]);
    assert_eq!(out.get(&[0]), 6.0); // 1*2*3
    assert_eq!(out.get(&[1]), 120.0); // 4*5*6
}

// --- mean_all ---

#[test]
/// `mean_all_on_2x3_returns_correct_scalar`.
fn mean_all_on_2x3_returns_correct_scalar() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::mean_all(&t).unwrap();
    assert_eq!(out.shape, Vec::<usize>::new());
    // mean = 21/6 = 3.5
    let v = out.get(&[]);
    assert!((v - 3.5).abs() < 1e-5, "mean_all expected 3.5, got {v}");
}

#[test]
/// `mean_all_backward_distributes_grad_scaled_by_1_over_n`.
fn mean_all_backward_distributes_grad_scaled_by_1_over_n() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::mean_all(&t).unwrap();
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
/// `sum_dim_removes_axis_0_on_2x3`.
fn sum_dim_removes_axis_0_on_2x3() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::sum_dim(&t, 0).unwrap();
    assert_eq!(out.shape, vec![3]);
    // col sums: 1+4=5, 2+5=7, 3+6=9
    assert_eq!(f32_vec(&out), vec![5.0, 7.0, 9.0]);
}

#[test]
/// `sum_dim_removes_axis_1_on_2x3`.
fn sum_dim_removes_axis_1_on_2x3() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::sum_dim(&t, 1).unwrap();
    assert_eq!(out.shape, vec![2]);
    // row sums: 1+2+3=6, 4+5+6=15
    assert_eq!(f32_vec(&out), vec![6.0, 15.0]);
}

#[test]
/// `sum_dim_backward_broadcasts_grad_back_to_original_shape`.
fn sum_dim_backward_broadcasts_grad_back_to_original_shape() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::sum_dim(&t, 0).unwrap(); // shape [3]
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have gradient");
    assert_eq!(g.shape, vec![2, 3]);
    // ones_like(out) = [1,1,1] broadcast back to [2,3] = ones
    assert_eq!(f32_vec(g), vec![1.0; 6]);
}

// --- sum_keepdim ---

#[test]
/// `sum_keepdim_retains_axis_0_on_2x3`.
fn sum_keepdim_retains_axis_0_on_2x3() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::sum_keepdim(&t, 0).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    assert_eq!(f32_vec(&out), vec![5.0, 7.0, 9.0]);
}

#[test]
/// `sum_keepdim_backward_broadcasts_grad_to_original_shape`.
fn sum_keepdim_backward_broadcasts_grad_to_original_shape() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::sum_keepdim(&t, 0).unwrap(); // shape [1, 3]
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have gradient");
    assert_eq!(g.shape, vec![2, 3]);
    // ones_like([1,3]) broadcast to [2,3] = ones
    assert_eq!(f32_vec(g), vec![1.0; 6]);
}

// --- sum_all backward with non-trivial incoming gradient (tape chain) ---

#[test]
/// `sum_all_backward_scales_by_incoming_gradient`.
fn sum_all_backward_scales_by_incoming_gradient() {
    // Build a small graph: out = sum_all(t), then seed with grad = 2.0
    // instead of 1.0 by composing with a scalar mul.
    // Simplest approach: verify via a custom tape entry.
    let t = vector(vec![1.0, 2.0, 3.0]);
    let sum_out = crate::cpu::ops::reduce::sum_all(&t).unwrap();
    // Manually build a loss = 2.0 * sum_out by pushing a tape entry
    let loss = CpuStorage::from_contiguous(CpuBuffer::F32(vec![0.0f32]), vec![]);
    let (sum_id, loss_id) = (sum_out.id, loss.id);
    tape::push_with(|| TapeEntry {
        output_id: loss_id,
        input_ids: vec![sum_id],
        backward: Box::new(|_grad_out: &CpuStorage| {
            Ok(
                // d(2 * sum_out) / d(sum_out) = 2
                vec![CpuStorage::from_contiguous(
                    CpuBuffer::F32(vec![2.0f32]),
                    vec![],
                )],
            )
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
/// `mean_dim_column_means_on_2x3`.
fn mean_dim_column_means_on_2x3() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::mean_dim(&t, 0).unwrap();
    assert_eq!(out.shape, vec![3]);
    let vals = f32_vec(&out);
    for (v, expected) in vals.iter().zip([2.5, 3.5, 4.5].iter()) {
        assert!((v - expected).abs() < 1e-5, "got {v}, expected {expected}");
    }
}

#[test]
/// `mean_keepdim_column_means_on_2x3`.
fn mean_keepdim_column_means_on_2x3() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::mean_keepdim(&t, 0).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    let vals = f32_vec(&out);
    for (v, expected) in vals.iter().zip([2.5, 3.5, 4.5].iter()) {
        assert!((v - expected).abs() < 1e-5, "got {v}, expected {expected}");
    }
}

#[test]
/// `mean_dim_gradcheck_dim0`.
fn mean_dim_gradcheck_dim0() {
    let x = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let reduced = crate::cpu::ops::reduce::mean_dim(&inputs[0], 0).unwrap();
        crate::cpu::ops::reduce::sum_all(&reduced).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], F32_STEP);
    assert!(
        max_rel_err < GRAD_TOL,
        "mean_dim gradcheck max relative error too high: {max_rel_err}"
    );
}

#[test]
/// `mean_keepdim_gradcheck_dim1`.
fn mean_keepdim_gradcheck_dim1() {
    let x = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let reduced = crate::cpu::ops::reduce::mean_keepdim(&inputs[0], 1).unwrap();
        crate::cpu::ops::reduce::sum_all(&reduced).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], F32_STEP);
    assert!(
        max_rel_err < GRAD_TOL,
        "mean_keepdim gradcheck max relative error too high: {max_rel_err}"
    );
}

// --- max_dim / min_dim / max_keepdim / min_keepdim / max_all / min_all ---

#[test]
/// `max_dim_column_maxima_on_2x3`.
fn max_dim_column_maxima_on_2x3() {
    let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::max_dim(&t, 0).unwrap();
    assert_eq!(out.shape, vec![3]);
    assert_eq!(f32_vec(&out), vec![4.0, 5.0, 6.0]);
}

#[test]
/// `max_keepdim_column_maxima_on_2x3`.
fn max_keepdim_column_maxima_on_2x3() {
    let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::max_keepdim(&t, 0).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    assert_eq!(f32_vec(&out), vec![4.0, 5.0, 6.0]);
}

#[test]
/// `min_dim_column_minima_on_2x3`.
fn min_dim_column_minima_on_2x3() {
    let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::min_dim(&t, 0).unwrap();
    assert_eq!(out.shape, vec![3]);
    assert_eq!(f32_vec(&out), vec![1.0, 2.0, 3.0]);
}

#[test]
/// `min_keepdim_column_minima_on_2x3`.
fn min_keepdim_column_minima_on_2x3() {
    let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::min_keepdim(&t, 0).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    assert_eq!(f32_vec(&out), vec![1.0, 2.0, 3.0]);
}

#[test]
/// `max_all_and_min_all_on_flat_vector`.
fn max_all_and_min_all_on_flat_vector() {
    let t = vector(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
    let max_out = crate::cpu::ops::reduce::max_all(&t).unwrap();
    assert_eq!(max_out.shape, Vec::<usize>::new());
    assert_eq!(max_out.get(&[]), 6.0);

    let min_out = crate::cpu::ops::reduce::min_all(&t).unwrap();
    assert_eq!(min_out.shape, Vec::<usize>::new());
    assert_eq!(min_out.get(&[]), 1.0);
}

#[test]
/// `max_dim_gradcheck_all_distinct_values`.
fn max_dim_gradcheck_all_distinct_values() {
    let x = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let reduced = crate::cpu::ops::reduce::max_dim(&inputs[0], 0).unwrap();
        crate::cpu::ops::reduce::sum_all(&reduced).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], F32_STEP);
    assert!(
        max_rel_err < GRAD_TOL,
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
    let out = crate::cpu::ops::reduce::max_dim(&t, 0).unwrap();
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

/// `i64_vec`.
fn i64_vec(s: &CpuStorage) -> Vec<i64> {
    match &*s.buffer {
        CpuBuffer::I64(v) => v.clone(),
        _ => panic!("expected I64 buffer"),
    }
}

#[test]
/// `argmax_dim0_returns_row_index_of_column_max`.
fn argmax_dim0_returns_row_index_of_column_max() {
    let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::argmax::<i64>(&t, Some(0)).unwrap();
    assert_eq!(out.shape, vec![3]);
    // col0 max is row1's 4 -> idx 1; col1 max is row0's 5 -> idx 0;
    // col2 max is row1's 6 -> idx 1.
    assert_eq!(i64_vec(&out), vec![1, 0, 1]);
}

#[test]
/// `argmax_dim_none_returns_scalar_flat_index_of_global_max`.
fn argmax_dim_none_returns_scalar_flat_index_of_global_max() {
    let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::argmax::<i64>(&t, None).unwrap();
    assert_eq!(out.shape, Vec::<usize>::new());
    // global max 6.0 is at flat index 5.
    assert_eq!(i64_vec(&out), vec![5]);
}

#[test]
/// `argmin_dim0_and_dim_none_mirror_argmax`.
fn argmin_dim0_and_dim_none_mirror_argmax() {
    let t = matrix(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let out_dim0 = crate::cpu::ops::reduce::argmin::<i64>(&t, Some(0)).unwrap();
    assert_eq!(out_dim0.shape, vec![3]);
    // col0 min is row0's 1 -> idx 0; col1 min is row1's 2 -> idx 1;
    // col2 min is row0's 3 -> idx 0.
    assert_eq!(i64_vec(&out_dim0), vec![0, 1, 0]);

    let out_none = crate::cpu::ops::reduce::argmin::<i64>(&t, None).unwrap();
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
    let _ = crate::cpu::ops::reduce::argmax::<i64>(&t, Some(0)).unwrap();
    let _ = crate::cpu::ops::reduce::argmax::<i64>(&t, None).unwrap();
    let _ = crate::cpu::ops::reduce::argmin::<i64>(&t, Some(0)).unwrap();
    let _ = crate::cpu::ops::reduce::argmin::<i64>(&t, None).unwrap();

    // Build and run an unrelated small graph immediately after; if
    // argmax/argmin had pushed spurious TapeEntry values, this
    // unrelated backward() would either panic or produce a corrupted
    // gradient for `unrelated`.
    let unrelated = vector(vec![10.0, 20.0, 30.0]);
    let sum_out = crate::cpu::ops::reduce::sum_all(&unrelated).unwrap();
    let grads = tape::backward(&sum_out).unwrap();
    let g = grads
        .get(unrelated.id)
        .expect("unrelated should have gradient");
    assert_eq!(f32_vec(g), vec![1.0, 1.0, 1.0]);
}

// --- cumsum ---

#[test]
/// `cumsum_scans_along_the_named_axis`.
fn cumsum_scans_along_the_named_axis() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = crate::cpu::ops::reduce::cumsum(&t, 1).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    // Row 0: [1, 1+2, 1+2+3] = [1, 3, 6]; row 1: [4, 9, 15].
    assert_eq!(f32_vec(&out), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
/// `cumsum_backward_receives_the_suffix_sum_of_the_incoming_gradient`.
fn cumsum_backward_receives_the_suffix_sum_of_the_incoming_gradient() {
    // The scan's Jacobian is lower-triangular ones, so the cotangent is its
    // transpose: grad_in[d] = sum_{k >= d} grad_out[k]. Seeding with ones
    // gives position d the count of elements at or after it.
    let t = vector(vec![1.0, 2.0, 3.0, 4.0]);
    let scanned = crate::cpu::ops::reduce::cumsum(&t, 0).unwrap();
    let loss = crate::cpu::ops::reduce::sum_all(&scanned).unwrap();
    let grads = tape::backward(&loss).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(g.shape, vec![4]);
    assert_eq!(f32_vec(g), vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
/// `cumsum_gradcheck_matches_finite_differences_on_both_axes`.
fn cumsum_gradcheck_matches_finite_differences_on_both_axes() {
    let x = matrix(vec![0.5, -1.0, 2.0, 1.5, -0.5, 0.25], 2, 3);
    let operands = [x];
    for dim in [0usize, 1] {
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let scanned = crate::cpu::ops::reduce::cumsum(&inputs[0], dim).unwrap();
            crate::cpu::ops::reduce::sum_all(&scanned).unwrap()
        };
        let max_rel_err = gradcheck(op, &operands, F32_STEP);
        assert!(
            max_rel_err < GRAD_TOL,
            "cumsum(dim={dim}) gradcheck too high: {max_rel_err}"
        );
    }
}

#[test]
/// `cumsum_rejects_an_out_of_range_axis_instead_of_panicking`.
fn cumsum_rejects_an_out_of_range_axis_instead_of_panicking() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let err = crate::cpu::ops::reduce::cumsum(&t, 2).unwrap_err();
    assert!(
        matches!(err, Error::ShapeMismatch { op: "cumsum", .. }),
        "expected a ShapeMismatch naming cumsum, got: {err:?}"
    );
}

// --- prod backward (the catalog declares Reduction gradients Defined) ---

#[test]
/// `prod_all_backward_divides_by_each_operand_element`.
fn prod_all_backward_divides_by_each_operand_element() {
    // y = prod(x) = 24; d(y)/d(x_i) = y / x_i. With seed 1 the gradient is
    // [24/2, 24/4, ...] i.e. the product of every element except x_i.
    let t = vector(vec![2.0, 4.0, 3.0]);
    let out = crate::cpu::ops::reduce::prod_all(&t).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(g.shape, vec![3]);
    // excluding 2 -> 12; excluding 4 -> 6; excluding 3 -> 8.
    assert_eq!(f32_vec(g), vec![12.0, 6.0, 8.0]);
}

#[test]
/// `prod_all_with_a_single_zero_routes_the_whole_gradient_through_it`.
fn prod_all_with_a_single_zero_routes_the_whole_gradient_through_it() {
    // One zero: only the zero's position has a non-zero derivative, equal to
    // the product of the remaining (non-zero) elements.
    let t = vector(vec![2.0, 0.0, 5.0]);
    let out = crate::cpu::ops::reduce::prod_all(&t).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(f32_vec(g), vec![0.0, 10.0, 0.0]);
}

#[test]
/// `prod_all_with_multiple_zeros_has_an_all_zero_gradient`.
fn prod_all_with_multiple_zeros_has_an_all_zero_gradient() {
    let t = vector(vec![2.0, 0.0, 5.0, 0.0]);
    let out = crate::cpu::ops::reduce::prod_all(&t).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(f32_vec(g), vec![0.0; 4]);
}

#[test]
/// `prod_all_gradcheck_matches_finite_differences`.
fn prod_all_gradcheck_matches_finite_differences() {
    // The product is multilinear in its operands, so the central difference
    // is exact here even across the zero.
    let x = vector(vec![1.5, -2.0, 0.0, 3.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        crate::cpu::ops::reduce::prod_all(&inputs[0]).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], F32_STEP);
    assert!(max_rel_err < GRAD_TOL, "prod_all gradcheck too high: {max_rel_err}");
}

#[test]
/// `prod_dim_backward_scales_each_axis_slice_independently`.
fn prod_dim_backward_scales_each_axis_slice_independently() {
    // Per output slice along dim 1: grad_i = prod(slice) / slice[i].
    let t = matrix(vec![1.0, 2.0, 3.0, 2.0, 1.0, 5.0], 2, 3);
    let out = crate::cpu::ops::reduce::prod_dim(&t, 1).unwrap(); // [6, 10]
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    // Row 0: 6/[1,2,3] = [6,3,2]; row 1: 10/[2,1,5] = [5,10,2].
    assert_eq!(f32_vec(g), vec![6.0, 3.0, 2.0, 5.0, 10.0, 2.0]);
}

#[test]
/// `prod_dim_gradcheck_matches_finite_differences`.
fn prod_dim_gradcheck_matches_finite_differences() {
    let x = matrix(vec![0.5, -1.0, 2.0, 1.5, -0.5, 4.0], 2, 3);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let reduced = crate::cpu::ops::reduce::prod_dim(&inputs[0], 1).unwrap();
        crate::cpu::ops::reduce::sum_all(&reduced).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], F32_STEP);
    assert!(max_rel_err < GRAD_TOL, "prod_dim gradcheck too high: {max_rel_err}");
}
