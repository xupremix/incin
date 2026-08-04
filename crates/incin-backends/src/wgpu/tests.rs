#![cfg(test)]

use crate::wgpu::storage::{WgpuBuffer, WgpuStorage};
use crate::wgpu::{WgpuBackendImpl, WgpuVar};
use incin_core::backend_authoring::*;
use incin_core::prelude::*;

// Helper: create a WgpuStorage from a flat vec and shape
/// `storage`.
fn storage(data: Vec<f32>, shape: Vec<usize>) -> WgpuStorage {
    WgpuStorage::new(WgpuBuffer::try_from_slice(&data).unwrap(), shape)
}

/// `readback`.
fn readback(s: &WgpuStorage) -> Vec<f32> {
    s.buffer.to_vec::<f32>().unwrap()
}

/// `approx_eq`.
fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// `vec_approx_eq`.
fn vec_approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
}

/// `B`.
type B = WgpuBackendImpl<f32, WgpuN<incin_core::typenum::U0>>;

// ── Creation ──────────────────────────────────────────────────────────────

#[test]
/// `test_zeros`.
fn test_zeros() {
    let s = <B as CreationOps<B>>::zeros::<f32>(&[2, 3], DTypeId::F32, &DeviceId::wgpu(0)).unwrap();
    assert_eq!(s.shape, vec![2, 3]);
    assert!(readback(&s).iter().all(|&x| x == 0.0));
}

#[test]
/// `test_ones`.
fn test_ones() {
    let s = <B as CreationOps<B>>::ones::<f32>(&[3, 2], DTypeId::F32, &DeviceId::wgpu(0)).unwrap();
    assert!(readback(&s).iter().all(|&x| x == 1.0));
}

#[test]
/// `test_rand_shape`.
fn test_rand_shape() {
    let s = <B as CreationOps<B>>::rand::<f32>(&[4, 4], DTypeId::F32, &DeviceId::wgpu(0)).unwrap();
    assert_eq!(s.shape, vec![4, 4]);
    let data = readback(&s);
    // All values should be in [0, 1)
    assert!(data.iter().all(|&x| (0.0..1.0).contains(&x)));
}

#[test]
/// `test_randn_shape`.
fn test_randn_shape() {
    let s = <B as CreationOps<B>>::randn::<f32>(&[100], DTypeId::F32, &DeviceId::wgpu(0)).unwrap();
    assert_eq!(s.shape, vec![100]);
}

// ── Binary ops ────────────────────────────────────────────────────────────

#[test]
/// `test_add`.
fn test_add() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = storage(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let out = <B as NumericOps<B>>::add::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
/// `test_sub`.
fn test_sub() {
    let a = storage(vec![10.0, 20.0, 30.0], vec![3]);
    let b = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let out = <B as NumericOps<B>>::sub::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![9.0, 18.0, 27.0]);
}

#[test]
/// `test_mul`.
fn test_mul() {
    let a = storage(vec![2.0, 3.0, 4.0], vec![3]);
    let b = storage(vec![5.0, 6.0, 7.0], vec![3]);
    let out = <B as NumericOps<B>>::mul::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![10.0, 18.0, 28.0]);
}

#[test]
/// `test_div`.
fn test_div() {
    let a = storage(vec![10.0, 20.0, 30.0], vec![3]);
    let b = storage(vec![2.0, 4.0, 5.0], vec![3]);
    let out = <B as NumericOps<B>>::div::<f32>(&a, &b).unwrap();
    assert!(vec_approx_eq(&readback(&out), &[5.0, 5.0, 6.0], 1e-5));
}

// ── Matmul ────────────────────────────────────────────────────────────────

#[test]
/// `test_matmul_2x3_3x2`.
fn test_matmul_2x3_3x2() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = storage(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let out = <B as TensorOps<B>>::matmul::<f32>(&a, &b).unwrap();
    assert_eq!(out.shape, vec![2, 2]);
    assert!(vec_approx_eq(
        &readback(&out),
        &[58.0, 64.0, 139.0, 154.0],
        1e-4
    ));
}

#[test]
/// `test_matmul_square`.
fn test_matmul_square() {
    let a = storage(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // identity
    let b = storage(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
    let out = <B as TensorOps<B>>::matmul::<f32>(&a, &b).unwrap();
    assert!(vec_approx_eq(&readback(&out), &[3.0, 4.0, 5.0, 6.0], 1e-4));
}

// ── Float / Unary ops ─────────────────────────────────────────────────────

#[test]
/// `test_relu`.
fn test_relu() {
    let a = storage(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    let out = <B as FloatOps<B>>::relu::<f32>(&a).unwrap();
    assert_eq!(readback(&out), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
/// `test_neg`.
fn test_neg() {
    let a = storage(vec![1.0, -2.0, 3.0], vec![3]);
    let out = <B as FloatOps<B>>::neg::<f32>(&a).unwrap();
    assert_eq!(readback(&out), vec![-1.0, 2.0, -3.0]);
}

#[test]
/// `test_abs`.
fn test_abs() {
    let a = storage(vec![-3.0, 0.0, 4.0], vec![3]);
    let out = <B as FloatOps<B>>::abs::<f32>(&a).unwrap();
    assert_eq!(readback(&out), vec![3.0, 0.0, 4.0]);
}

#[test]
/// `test_sqrt`.
fn test_sqrt() {
    let a = storage(vec![4.0, 9.0, 16.0], vec![3]);
    let out = <B as FloatOps<B>>::sqrt::<f32>(&a).unwrap();
    assert!(vec_approx_eq(&readback(&out), &[2.0, 3.0, 4.0], 1e-5));
}

#[test]
/// `test_exp_log`.
fn test_exp_log() {
    let a = storage(vec![0.0, 1.0, 2.0], vec![3]);
    let exp_out = <B as FloatOps<B>>::exp::<f32>(&a).unwrap();
    let expected_exp = [1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)];
    assert!(vec_approx_eq(&readback(&exp_out), &expected_exp, 1e-5));

    let log_out = <B as FloatOps<B>>::log::<f32>(&exp_out).unwrap();
    assert!(vec_approx_eq(&readback(&log_out), &[0.0, 1.0, 2.0], 1e-5));
}

#[test]
/// `test_sigmoid`.
fn test_sigmoid() {
    let a = storage(vec![0.0], vec![1]);
    let out = <B as FloatOps<B>>::sigmoid::<f32>(&a).unwrap();
    assert!(approx_eq(readback(&out)[0], 0.5, 1e-5));
}

#[test]
/// `test_tanh`.
fn test_tanh() {
    let a = storage(vec![0.0], vec![1]);
    let out = <B as FloatOps<B>>::tanh::<f32>(&a).unwrap();
    assert!(approx_eq(readback(&out)[0], 0.0, 1e-5));
}

#[test]
/// `test_swish`.
fn test_swish() {
    // swish(x) = x * sigmoid(x); swish(0) = 0
    let a = storage(vec![0.0, 1.0], vec![2]);
    let out = <B as FloatOps<B>>::swish::<f32>(&a).unwrap();
    let data = readback(&out);
    assert!(approx_eq(data[0], 0.0, 1e-5));
    // swish(1) = 1 * sigmoid(1) ≈ 0.7311
    assert!(approx_eq(data[1], 0.7310586, 1e-4));
}

#[test]
/// `test_gelu`.
fn test_gelu() {
    let a = storage(vec![0.0], vec![1]);
    let out = <B as FloatOps<B>>::gelu::<f32>(&a).unwrap();
    assert!(approx_eq(readback(&out)[0], 0.0, 1e-5));
}

#[test]
/// `test_add_scalar`.
fn test_add_scalar() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let out = <B as FloatOps<B>>::add_scalar_float::<f32>(&a, 10.0).unwrap();
    assert!(vec_approx_eq(&readback(&out), &[11.0, 12.0, 13.0], 1e-5));
}

#[test]
/// `test_mul_scalar`.
fn test_mul_scalar() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let out = <B as FloatOps<B>>::mul_scalar_float::<f32>(&a, 3.0).unwrap();
    assert!(vec_approx_eq(&readback(&out), &[3.0, 6.0, 9.0], 1e-5));
}

#[test]
/// `test_softmax`.
fn test_softmax() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let out = <B as FloatOps<B>>::softmax::<f32>(&a, 1).unwrap();
    let data = readback(&out);
    // Sum should be 1
    let sum: f32 = data.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-5));
    // Max should be at index 2
    let max_idx = data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(max_idx, 2);
}

// ── Reduction ops ─────────────────────────────────────────────────────────

#[test]
/// `test_sum_all`.
fn test_sum_all() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = <B as ReductionOps<B>>::sum_all::<f32>(&a).unwrap();
    assert!(approx_eq(readback(&out)[0], 10.0, 1e-4));
}

#[test]
/// `test_mean_all`.
fn test_mean_all() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let out = <B as ReductionOps<B>>::mean_all::<f32>(&a).unwrap();
    assert!(approx_eq(readback(&out)[0], 2.5, 1e-4));
}

#[test]
/// `test_max_all`.
fn test_max_all() {
    let a = storage(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6]);
    let out = <B as ReductionOps<B>>::max_all::<f32>(&a).unwrap();
    assert!(approx_eq(readback(&out)[0], 9.0, 1e-4));
}

#[test]
/// `test_min_all`.
fn test_min_all() {
    let a = storage(vec![3.0, 1.0, 4.0, -2.0, 5.0], vec![5]);
    let out = <B as ReductionOps<B>>::min_all::<f32>(&a).unwrap();
    assert!(approx_eq(readback(&out)[0], -2.0, 1e-4));
}

#[test]
/// `test_sum_dim`.
fn test_sum_dim() {
    // [[1,2,3],[4,5,6]] sum along dim 0 -> [5,7,9]
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as ReductionOps<B>>::sum_dim::<f32>(&a, 0).unwrap();
    assert!(vec_approx_eq(&readback(&out), &[5.0, 7.0, 9.0], 1e-4));
    assert_eq!(out.shape, vec![3]);
}

#[test]
/// `test_sum_keepdim`.
fn test_sum_keepdim() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as ReductionOps<B>>::sum_keepdim::<f32>(&a, 0).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
}

#[test]
/// `test_mean_dim`.
fn test_mean_dim() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as ReductionOps<B>>::mean_dim::<f32>(&a, 0).unwrap();
    assert!(vec_approx_eq(&readback(&out), &[2.5, 3.5, 4.5], 1e-4));
}

#[test]
/// `test_max_dim`.
fn test_max_dim() {
    let a = storage(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
    let out = <B as ReductionOps<B>>::max_dim::<f32>(&a, 0).unwrap();
    assert!(vec_approx_eq(&readback(&out), &[4.0, 5.0, 6.0], 1e-4));
}
#[test]
/// `test_argmax_flat`.
fn test_argmax_flat() {
    let a = storage(vec![1.0, 5.0, 3.0, 9.0, 2.0], vec![5]);
    let out = <B as ReductionOps<B>>::argmax::<f32, u32>(&a, None).unwrap();
    assert_eq!(out.buffer.to_vec::<u32>().unwrap()[0] as usize, 3);
}

#[test]
/// `test_argmin_flat`.
fn test_argmin_flat() {
    let a = storage(vec![3.0, -1.0, 5.0], vec![3]);
    let out = <B as ReductionOps<B>>::argmin::<f32, u32>(&a, None).unwrap();
    assert_eq!(out.buffer.to_vec::<u32>().unwrap()[0] as usize, 1);
}

// ── Tensor ops ────────────────────────────────────────────────────────────

#[test]
/// `test_reshape`.
fn test_reshape() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as TensorOps<B>>::reshape::<f32>(&a, &[3, 2]).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(readback(&out), readback(&a)); // same buffer, same data
}

#[test]
/// `test_transpose_2d`.
fn test_transpose_2d() {
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as TensorOps<B>>::transpose::<f32>(&a, 0, 1).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
    assert!(vec_approx_eq(
        &readback(&out),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        1e-5
    ));
}

#[test]
/// `test_flatten`.
fn test_flatten() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as TensorOps<B>>::flatten::<f32>(&a, 0, 1).unwrap();
    assert_eq!(out.shape, vec![6]);
}

#[test]
/// `test_squeeze`.
fn test_squeeze() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let out = <B as TensorOps<B>>::squeeze::<f32>(&a, 0).unwrap();
    assert_eq!(out.shape, vec![3]);
}

#[test]
fn test_scaled_dot_product_attention_uniform_when_query_is_zero() {
    // q is all-zero, so q@k^T is all-zero regardless of k or the scale,
    // softmax of an all-zero row is uniform, and the output is exactly the
    // unweighted average of v's rows — avoids hand-computing exponentials.
    let q = storage(vec![0.0, 0.0], vec![1, 2]);
    let k = storage(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
    let v = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let out =
        <B as TensorOps<B>>::scaled_dot_product_attention::<f32>(&q, &k, &v, None, None).unwrap();
    assert_eq!(out.shape, vec![1, 2]);
    assert!(vec_approx_eq(&readback(&out), &[3.0, 4.0], 1e-4));
}

#[test]
fn scaled_dot_product_attention_records_gradients_for_all_three_operands() {
    let q = storage(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);
    let k = storage(vec![0.5, 0.6, 0.7, 0.8], vec![2, 2]);
    let v = storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out =
        <B as TensorOps<B>>::scaled_dot_product_attention::<f32>(&q, &k, &v, None, None).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    assert!(grads.get(q.id).is_some(), "q should have a gradient");
    assert!(grads.get(k.id).is_some(), "k should have a gradient");
    assert!(grads.get(v.id).is_some(), "v should have a gradient");
}

#[test]
fn test_where_cond_same_shape() {
    let mask = storage(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let on_true = storage(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let on_false = storage(vec![-1.0, -2.0, -3.0, -4.0], vec![4]);
    let out =
        <B as TensorOps<B>>::where_cond::<f32, f32>(&mask, &on_true, &on_false).unwrap();
    assert_eq!(readback(&out), vec![10.0, -2.0, 30.0, -4.0]);
}

#[test]
fn test_where_cond_broadcasts_on_false_against_on_true() {
    // on_false is a scalar-per-row [2,1] broadcast against on_true's [2,3].
    let mask = storage(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 3]);
    let on_true = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let on_false = storage(vec![-1.0, -2.0], vec![2, 1]);
    let out =
        <B as TensorOps<B>>::where_cond::<f32, f32>(&mask, &on_true, &on_false).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(readback(&out), vec![1.0, -1.0, 3.0, -2.0, 5.0, -2.0]);
}

#[test]
fn where_cond_backward_routes_grad_by_the_mask_and_unbroadcasts() {
    let mask = storage(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let on_true = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    // on_false is a single value broadcast across all 4 positions, so its
    // gradient must sum every position the mask routed to it.
    let on_false = storage(vec![9.0], vec![1]);
    let out =
        <B as TensorOps<B>>::where_cond::<f32, f32>(&mask, &on_true, &on_false).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let g_true = grads.get(on_true.id).expect("on_true should have a gradient");
    let g_false = grads
        .get(on_false.id)
        .expect("on_false should have a gradient");
    // ones_like seed: grad flows to on_true at mask positions 0 and 2, to
    // on_false (summed over its two broadcast positions 1 and 3) otherwise.
    assert_eq!(readback(g_true), vec![1.0, 0.0, 1.0, 0.0]);
    assert_eq!(readback(g_false), vec![2.0]);
}

#[test]
fn test_gather() {
    let t = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let index = storage(vec![2.0, 0.0], vec![2, 1]);
    let out = <B as TensorOps<B>>::gather::<f32, f32>(&t, 0, &index).unwrap();
    assert_eq!(out.shape, vec![2, 1]);
    // Row 0's column 0 gathers t[2,0]=5 (index[0,0]=2), row 1's column 0
    // gathers t[0,0]=1 (index[1,0]=0).
    assert_eq!(readback(&out), vec![5.0, 1.0]);
}

#[test]
fn gather_backward_scatter_adds_to_every_position_that_was_read() {
    // index selects position 0 twice, so gather's backward must accumulate
    // both contributions into grad_t[0] rather than overwrite.
    let t = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let index = storage(vec![0.0, 0.0, 1.0], vec![3]);
    let out = <B as TensorOps<B>>::gather::<f32, f32>(&t, 0, &index).unwrap();
    assert_eq!(readback(&out), vec![1.0, 1.0, 2.0]);
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    // ones_like seed: grad_t[0] accumulates from both reads of position 0,
    // grad_t[1] from the single read of position 1, grad_t[2] untouched.
    assert_eq!(readback(g), vec![2.0, 1.0, 0.0]);
}

#[test]
fn test_scatter() {
    let t = storage(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![3, 2]);
    let index = storage(vec![2.0, 0.0], vec![2, 1]);
    let src = storage(vec![9.0, 8.0], vec![2, 1]);
    let out = <B as TensorOps<B>>::scatter::<f32, f32>(&t, 0, &index, &src).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
    // Row 0's column 0 gets src[1]=8 (index[1]=0), row 2's column 0 gets
    // src[0]=9 (index[0]=2); every other position is untouched.
    assert_eq!(readback(&out), vec![8.0, 0.0, 0.0, 0.0, 9.0, 0.0]);
}

#[test]
fn test_index_select() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let index = storage(vec![2.0, 0.0], vec![2]);
    let out = <B as TensorOps<B>>::index_select::<f32, f32>(&a, 0, &index).unwrap();
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(readback(&out), vec![5.0, 6.0, 1.0, 2.0]);
}

#[test]
fn test_masked_fill() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let mask = storage(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let out = <B as TensorOps<B>>::masked_fill::<f32, f32>(&a, &mask, -1.0).unwrap();
    assert_eq!(readback(&out), vec![-1.0, 2.0, -1.0, 4.0]);
}

#[test]
fn masked_fill_rejects_a_mismatched_mask_shape() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let mask = storage(vec![1.0, 0.0], vec![2]);
    assert!(<B as TensorOps<B>>::masked_fill::<f32, f32>(&a, &mask, -1.0).is_err());
}

#[test]
/// Same fixture as the CPU backend's
/// `group_norm_statistics_are_per_sample_not_across_the_batch`.
fn group_norm_statistics_are_per_sample_not_across_the_batch() {
    let first: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let second: Vec<f32> = first.iter().map(|v| v + 100.0).collect();
    let data = first.iter().copied().chain(second).collect::<Vec<f32>>();
    let t = storage(data, vec![2, 4, 1, 2]);

    let out = readback(&<B as TensorOps<B>>::group_norm::<f32>(&t, 2, 1e-5).unwrap());

    assert_eq!(out[..8], out[8..], "the two samples must normalize alike");
    // Group 0 of sample 0 is [0,1,2,3]: mean 1.5, population variance 1.25.
    let inv_std = 1.0 / (1.25f64 + 1e-5).sqrt();
    for (i, value) in [0.0f64, 1.0, 2.0, 3.0].iter().enumerate() {
        let expected = ((value - 1.5) * inv_std) as f32;
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "element {i}: got {}, want {expected}",
            out[i]
        );
    }
}

#[test]
/// Same fixture as the CPU backend's
/// `instance_norm_normalizes_each_channel_of_each_sample_alone`.
fn instance_norm_normalizes_each_channel_of_each_sample_alone() {
    let t = storage(
        vec![
            1.0, 1.0, 5.0, 7.0, // sample 0: channel 0 flat, channel 1 varies
            2.0, 2.0, 9.0, 3.0, // sample 1: channel 0 flat, channel 1 varies
        ],
        vec![2, 2, 2],
    );

    let out = readback(&<B as TensorOps<B>>::instance_norm::<f32>(&t, 1e-5).unwrap());

    for flat in [0, 1, 4, 5] {
        assert!(
            out[flat].abs() < 1e-5,
            "constant channel at {flat} must normalize to zero, got {}",
            out[flat]
        );
    }
    assert!((out[2] + 1.0).abs() < 1e-3, "got {}", out[2]);
    assert!((out[3] - 1.0).abs() < 1e-3, "got {}", out[3]);
    assert!((out[6] - 1.0).abs() < 1e-3, "got {}", out[6]);
    assert!((out[7] + 1.0).abs() < 1e-3, "got {}", out[7]);
}

#[test]
fn group_norm_rejects_zero_groups() {
    let t = storage(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
    assert!(<B as TensorOps<B>>::group_norm::<f32>(&t, 0, 1e-5).is_err());
}

#[test]
fn test_unfold() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let out = <B as TensorOps<B>>::unfold::<f32>(&a, 0, 3, 1).unwrap();
    assert_eq!(out.shape, vec![3, 3]);
    assert_eq!(
        readback(&out),
        vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn unfold_rejects_a_window_larger_than_the_dimension() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(<B as TensorOps<B>>::unfold::<f32>(&a, 0, 4, 1).is_err());
}

#[test]
fn test_pixel_shuffle() {
    // N=1, C=4, H=1, W=1, upscale_factor=2 -> N=1, C=1, H=2, W=2.
    // Channel c_in maps to output position (r_h, r_w) = (c_in / 2, c_in % 2).
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4, 1, 1]);
    let out = <B as TensorOps<B>>::pixel_shuffle::<f32>(&a, 2).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    assert_eq!(readback(&out), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn pixel_shuffle_rejects_channels_not_divisible_by_upscale_squared() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![1, 3, 1, 1]);
    assert!(<B as TensorOps<B>>::pixel_shuffle::<f32>(&a, 2).is_err());
}

#[test]
fn test_repeat() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = <B as TensorOps<B>>::repeat::<f32>(&a, &[2, 1]).unwrap();
    assert_eq!(out.shape, vec![4, 2]);
    assert_eq!(readback(&out), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_pad() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = <B as TensorOps<B>>::pad::<f32>(&a, &[(1, 0), (0, 1)], -1.0).unwrap();
    assert_eq!(out.shape, vec![3, 3]);
    assert_eq!(
        readback(&out),
        vec![-1.0, -1.0, -1.0, 1.0, 2.0, -1.0, 3.0, 4.0, -1.0]
    );
}

#[test]
fn test_triu() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
    let out = <B as TensorOps<B>>::triu::<f32>(&a, 0).unwrap();
    assert_eq!(
        readback(&out),
        vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
    );
}

#[test]
fn test_tril() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
    let out = <B as TensorOps<B>>::tril::<f32>(&a, 0).unwrap();
    assert_eq!(
        readback(&out),
        vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_diag_builds_matrix_from_vector() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let out = <B as TensorOps<B>>::diag::<f32>(&a, 0).unwrap();
    assert_eq!(out.shape, vec![3, 3]);
    assert_eq!(
        readback(&out),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    );
}

#[test]
fn test_diag_extracts_from_matrix() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 3]);
    let out = <B as TensorOps<B>>::diag::<f32>(&a, 0).unwrap();
    assert_eq!(out.shape, vec![3]);
    assert_eq!(readback(&out), vec![1.0, 5.0, 9.0]);
}

#[test]
/// `test_narrow`.
fn test_narrow() {
    // [[1,2,3],[4,5,6]] narrow dim=0, start=1, len=1 -> [[4,5,6]]
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as TensorOps<B>>::narrow::<f32>(&a, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    assert!(vec_approx_eq(&readback(&out), &[4.0, 5.0, 6.0], 1e-5));
}

#[test]
/// `test_concat_dim0`.
fn test_concat_dim0() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = storage(vec![4.0, 5.0, 6.0], vec![1, 3]);
    let out = <B as TensorOps<B>>::concat::<f32>(&[&a, &b], 0).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(readback(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
/// `test_stack`.
fn test_stack() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let b = storage(vec![4.0, 5.0, 6.0], vec![3]);
    let out = <B as TensorOps<B>>::stack::<f32>(&[&a, &b], 0).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(readback(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
/// `test_float_to_scalar`.
fn test_float_to_scalar() {
    let a = storage(vec![42.0], vec![1]);
    let val = <B as TensorOps<B>>::float_to_scalar::<f32>(&a).unwrap();
    assert!(approx_eq(val as f32, 42.0, 1e-5));
}

// ── Module ops ────────────────────────────────────────────────────────────

#[test]
/// `test_embedding`.
fn test_embedding() {
    // vocab=3, dim=4; pick indices [0, 2]
    let weight = storage(
        vec![
            0.1, 0.2, 0.3, 0.4, // token 0
            0.5, 0.6, 0.7, 0.8, // token 1
            0.9, 1.0, 1.1, 1.2, // token 2
        ],
        vec![3, 4],
    );
    let indices = storage(vec![0.0, 2.0], vec![2]);
    let out = <B as ModuleOps<B>>::embedding::<f32, f32>(&indices, &weight).unwrap();
    assert_eq!(out.shape, vec![2, 4]);
    assert!(vec_approx_eq(
        &readback(&out),
        &[0.1, 0.2, 0.3, 0.4, 0.9, 1.0, 1.1, 1.2],
        1e-5
    ));
}

#[test]
/// `test_layer_norm`.
fn test_layer_norm() {
    let x = storage(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let gamma = storage(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let beta = storage(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
    let out = <B as ModuleOps<B>>::layer_norm::<f32>(&x, &gamma, Some(&beta), 1e-5).unwrap();
    let data = readback(&out);
    // After layer norm, mean≈0, std≈1
    let mean: f32 = data.iter().sum::<f32>() / 4.0;
    assert!(approx_eq(mean, 0.0, 1e-4));
    let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
    assert!(approx_eq(var, 1.0, 1e-3));
}

#[test]
/// `test_adaptive_avg_pool2d`.
fn test_adaptive_avg_pool2d() {
    // 1x1x4x4 -> 1x1x2x2
    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let inp = storage(data, vec![1, 1, 4, 4]);
    let out = <B as ModuleOps<B>>::adaptive_avg_pool2d::<f32>(&inp, (2, 2)).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    let result = readback(&out);
    // Top-left 2x2 avg = (1+2+5+6)/4 = 3.5
    assert!(approx_eq(result[0], 3.5, 1e-4));
}

#[test]
/// `test_max_pool2d`.
fn test_max_pool2d() {
    // 1x1x4x4, kernel 2x2, stride 2
    let data: Vec<f32> = vec![
        1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 15.0, 14.0, 16.0,
    ];
    let inp = storage(data, vec![1, 1, 4, 4]);
    let out = <B as ModuleOps<B>>::max_pool2d::<f32>(&inp, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    let result = readback(&out);
    assert!(approx_eq(result[0], 7.0, 1e-4));
    assert!(approx_eq(result[1], 8.0, 1e-4));
    assert!(approx_eq(result[2], 15.0, 1e-4));
    assert!(approx_eq(result[3], 16.0, 1e-4));
}

// ── Loss ops ──────────────────────────────────────────────────────────────

#[test]
/// `test_cross_entropy_mean`.
fn test_cross_entropy_mean() {
    // 2 classes, batch=2; pred logits
    let pred = storage(vec![2.0, 1.0, 0.5, 3.0], vec![2, 2]);
    let target = storage(vec![0.0, 1.0], vec![2]); // class 0 and class 1
    let out = <B as LossOps<B>>::cross_entropy_loss::<f32, f32>(
        &pred,
        &target,
        incin_core::prelude::Reduction::Mean,
    )
    .unwrap();
    let loss = readback(&out)[0];
    assert!(loss > 0.0, "Loss should be positive");
    assert!(loss < 5.0, "Loss should be reasonable");
}

#[test]
fn cross_entropy_loss_matches_hand_computed_value_for_nonzero_target() {
    // Regression test: target/index storage is physically F32 bytes (the
    // embedding WGSL kernel's `u32(indices[i])` confirms this backend does
    // real value conversion, not raw-byte reinterpretation), so building the
    // one-hot target must read it back the same way. The existing
    // `test_cross_entropy_mean` above only asserts loose bounds and would
    // not have caught a bit-reinterpret bug that zeroes out every non-0.0
    // target row's contribution — this test pins down an exact expected
    // value with target class 1 (row 0) and class 1 (row 1, this backend's
    // bit-pattern-vs-value bug would previously silently drop this row's
    // real contribution, understating the loss).
    let pred = storage(vec![2.0, 1.0, 0.5, 3.0], vec![2, 2]);
    let target = storage(vec![0.0, 1.0], vec![2]); // class 0, class 1
    let out = <B as LossOps<B>>::cross_entropy_loss::<f32, f32>(
        &pred,
        &target,
        incin_core::prelude::Reduction::Mean,
    )
    .unwrap();
    let loss = readback(&out)[0];
    // Hand-computed: -log_softmax([2,1])[0] = 0.313262,
    // -log_softmax([0.5,3])[1] = 0.078890, mean = 0.196076.
    assert!(
        approx_eq(loss, 0.196076, 1e-4),
        "expected ~0.196076, got {loss}"
    );
}

#[test]
fn cross_entropy_loss_backward_matches_finite_difference() {
    // cross_entropy_loss is fully composed from already-wired primitives
    // (log_softmax's sub/exp/sum_keepdim/log/broadcast_as chain, mul,
    // sum_dim, neg, mean_all), so — like softmax/layer_norm/batch_norm
    // before it — this should already be gradient-correct with no new
    // wiring; this test verifies that rather than assuming it.
    let pred = storage(vec![2.0, 1.0, -0.5, 0.5, 3.0, 0.2], vec![2, 3]);
    let target = storage(vec![0.0, 2.0], vec![2]); // class 0, class 2
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        <B as LossOps<B>>::cross_entropy_loss::<f32, f32>(
            &inputs[0],
            &target,
            incin_core::prelude::Reduction::Mean,
        )
        .unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[pred], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "cross_entropy_loss gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

// ── Optimizer ─────────────────────────────────────────────────────────────

#[test]
/// `test_adamw_step`.
fn test_adamw_step() {
    let param = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let grad = storage(vec![0.1, 0.2, 0.3], vec![3]);
    let mut m = storage(vec![0.0, 0.0, 0.0], vec![3]);
    let mut v = storage(vec![0.0, 0.0, 0.0], vec![3]);

    let old_param = readback(&param);
    let mut var = WgpuVar {
        storage: param.clone(),
    };

    <B as OptimizerOps<B>>::adamw_step::<f32>(
        &mut var, &grad, &mut m, &mut v, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1,
    )
    .unwrap();

    let new_param = readback(&var.storage);
    // Parameters should have moved
    assert!(
        new_param
            .iter()
            .zip(old_param.iter())
            .any(|(n, o)| (n - o).abs() > 1e-6)
    );
}

// ── GPU AdamW parity ─────────────────────────────────────────────────────

#[test]
/// `test_adamw_gpu_matches_reference`.
fn test_adamw_gpu_matches_reference() {
    // Reference CPU calculation for a single AdamW step
    let lr = 0.001f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    let wd = 0.01f32;
    let step = 1;

    let p_init = vec![1.0f32, -0.5, 2.0];
    let g_init = vec![0.1f32, 0.2, -0.3];

    // Reference calculation
    let mut ref_p = p_init.clone();
    let mut ref_m = [0.0f32; 3];
    let mut ref_v = [0.0f32; 3];
    for i in 0..3 {
        let p_val = p_init[i] - lr * wd * p_init[i];
        let g = g_init[i];

        ref_m[i] = beta1 * ref_m[i] + (1.0 - beta1) * g;
        ref_v[i] = beta2 * ref_v[i] + (1.0 - beta2) * g * g;

        ref_p[i] = p_val - lr * ref_m[i] / (ref_v[i].sqrt() + eps);
    }

    // GPU calculation
    let param = storage(p_init.clone(), vec![3]);
    let grad = storage(g_init.clone(), vec![3]);
    let mut m = storage(vec![0.0, 0.0, 0.0], vec![3]);
    let mut v = storage(vec![0.0, 0.0, 0.0], vec![3]);
    let mut var = WgpuVar {
        storage: param.clone(),
    };

    <B as OptimizerOps<B>>::adamw_step::<f32>(
        &mut var,
        &grad,
        &mut m,
        &mut v,
        lr as f64,
        beta1 as f64,
        beta2 as f64,
        eps as f64,
        wd as f64,
        step as usize,
    )
    .unwrap();

    let gpu_p = readback(&var.storage);
    assert!(
        vec_approx_eq(&gpu_p, &ref_p, 1e-5),
        "GPU AdamW mismatch: got {:?}, expected {:?}",
        gpu_p,
        ref_p
    );
}

// ── Conv ops ──────────────────────────────────────────────────────────────

#[test]
/// `test_conv2d_identity_kernel`.
fn test_conv2d_identity_kernel() {
    // 3x3 identity filter: output should equal the center pixel (no padding, stride=1)
    // Input: 1x1x3x3, Weight: 1x1x1x1 = [[1.0]] -> output = input
    let inp = storage(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
    );
    let weight = storage(vec![1.0], vec![1, 1, 1, 1]);
    let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, None, 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 3, 3]);
    assert!(vec_approx_eq(
        &readback(&out),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        1e-5
    ));
}

#[test]
/// `test_conv2d_known_output`.
fn test_conv2d_known_output() {
    // Input 1x1x4x4, weight 1x1x2x2, stride=1, padding=0
    // Using weight [[1,0],[0,1]] (sum of diagonal)
    let inp_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let inp = storage(inp_data, vec![1, 1, 4, 4]);
    let weight = storage(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
    let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, None, 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 3, 3]);
    let result = readback(&out);
    // out[0,0] = inp[0,0]*1 + inp[0,1]*0 + inp[1,0]*0 + inp[1,1]*1 = 1 + 6 = 7
    assert!(approx_eq(result[0], 7.0, 1e-4));
    // out[0,1] = inp[0,1]*1 + inp[0,2]*0 + inp[1,1]*0 + inp[1,2]*1 = 2 + 7 = 9
    assert!(approx_eq(result[1], 9.0, 1e-4));
}

#[test]
/// `test_conv2d_with_bias`.
fn test_conv2d_with_bias() {
    let inp = storage(vec![1.0, 2.0, 2.0, 1.0], vec![1, 1, 2, 2]);
    let weight = storage(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
    let bias = storage(vec![10.0], vec![1]);
    let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, Some(&bias), 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 1, 1]);
    // sum(1+2+2+1) = 6, + bias 10 = 16
    assert!(approx_eq(readback(&out)[0], 16.0, 1e-4));
}

#[test]
/// `test_conv2d_padding`.
fn test_conv2d_padding() {
    // 1x1x1x1 input with 1-padding, 1x1x3x3 kernel -> 1x1x1x1 output
    let inp = storage(vec![1.0], vec![1, 1, 1, 1]);
    let weight = storage(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![1, 1, 3, 3],
    );
    let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, None, 1, 1, 1, 1).unwrap();
    // Input padded: only center = 1.0, matches weight center (index 4)
    assert_eq!(out.shape, vec![1, 1, 1, 1]);
    assert!(approx_eq(readback(&out)[0], 1.0, 1e-4));
}

#[test]
/// `test_conv2d_two_output_channels`.
fn test_conv2d_two_output_channels() {
    // 1 input, 2 output channels: each filter reads the same input
    let inp = storage(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    // Filter 1: sum all    Filter 2: negate sum
    let weight = storage(
        vec![
            1.0, 1.0, 1.0, 1.0, // C_out=0
            -1.0, -1.0, -1.0, -1.0, // C_out=1
        ],
        vec![2, 1, 2, 2],
    );
    let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, None, 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 2, 1, 1]);
    let data = readback(&out);
    assert!(approx_eq(data[0], 10.0, 1e-4)); // 1+2+3+4=10
    assert!(approx_eq(data[1], -10.0, 1e-4));
}

#[test]
/// `test_conv1d_basic`.
fn test_conv1d_basic() {
    // Input: 1 batch, 1 channel, 4 elements; Weight: 1 out, 1 in, 2 kernel
    let inp = storage(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
    let weight = storage(vec![1.0, 1.0], vec![1, 1, 2]);
    let out = <B as ModuleOps<B>>::conv1d::<f32>(&inp, &weight, None, 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 3]);
    // out[0] = 1+2=3, out[1] = 2+3=5, out[2] = 3+4=7
    assert!(vec_approx_eq(&readback(&out), &[3.0, 5.0, 7.0], 1e-4));
}

#[test]
fn test_quantize_dequantize() {
    let mut data = vec![0.0f32; 64];
    for (i, d) in data.iter_mut().enumerate() {
        *d = (i as f32 - 32.0) * 0.1; // ranging -3.2 to +3.1
    }
    let s = storage(data.clone(), vec![2, 32]);
    let q_storage = <B as QuantizedOps<B>>::quantize::<f32, incin_core::prelude::Q8_0>(&s).unwrap();
    assert_eq!(q_storage.dtype, DTypeId::Q8_0);
    assert_eq!(q_storage.device, DeviceId::wgpu(0));
    assert_eq!(q_storage.shape, vec![2, 32]);
    assert_eq!(q_storage.offset_elements, 0);
    let deq_storage =
        <B as QuantizedOps<B>>::dequantize::<incin_core::prelude::Q8_0, f32>(&q_storage).unwrap();
    let deq_data = readback(&deq_storage);

    for (orig, deq) in data.iter().zip(deq_data.iter()) {
        let diff = (orig - deq).abs();
        assert!(diff < 0.05, "Diff too large: {} vs {}", orig, deq);
    }
}

// ── Autograd (C-3 regression guard) ─────────────────────────────────────
//
// Before this fix, `tape::push` had zero call sites anywhere in this
// module — `backward()` ran without error but silently returned no
// gradient for any parameter. Every test below fails loudly (missing or
// wrong gradient) if that regresses, instead of the old failure mode
// (no error, no gradient, wrong training silently).

#[test]
fn add_backward_gives_grad_one_to_both_operands() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let b = storage(vec![10.0, 20.0, 30.0], vec![3]);
    let out = <B as NumericOps<B>>::add::<f32>(&a, &b).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let ga = grads.get(a.id).expect("a should have a gradient");
    let gb = grads.get(b.id).expect("b should have a gradient");
    assert!(vec_approx_eq(&readback(ga), &[1.0, 1.0, 1.0], 1e-5));
    assert!(vec_approx_eq(&readback(gb), &[1.0, 1.0, 1.0], 1e-5));
}

#[test]
fn sub_backward_negates_rhs_contribution() {
    let a = storage(vec![10.0, 20.0, 30.0], vec![3]);
    let b = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let out = <B as NumericOps<B>>::sub::<f32>(&a, &b).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let ga = grads.get(a.id).unwrap();
    let gb = grads.get(b.id).unwrap();
    assert!(vec_approx_eq(&readback(ga), &[1.0, 1.0, 1.0], 1e-5));
    assert!(vec_approx_eq(&readback(gb), &[-1.0, -1.0, -1.0], 1e-5));
}

#[test]
fn mul_backward_uses_other_operands_real_values() {
    // d(a*b)/da = b, d(a*b)/db = a.
    let a = storage(vec![2.0, 3.0, 4.0], vec![3]);
    let b = storage(vec![5.0, 6.0, 7.0], vec![3]);
    let out = <B as NumericOps<B>>::mul::<f32>(&a, &b).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let ga = grads.get(a.id).unwrap();
    let gb = grads.get(b.id).unwrap();
    assert!(vec_approx_eq(&readback(ga), &[5.0, 6.0, 7.0], 1e-5));
    assert!(vec_approx_eq(&readback(gb), &[2.0, 3.0, 4.0], 1e-5));
}

#[test]
fn div_backward_matches_quotient_rule() {
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2.
    let a = storage(vec![6.0, 8.0], vec![2]);
    let b = storage(vec![2.0, 4.0], vec![2]);
    let out = <B as NumericOps<B>>::div::<f32>(&a, &b).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let ga = grads.get(a.id).unwrap();
    let gb = grads.get(b.id).unwrap();
    assert!(vec_approx_eq(&readback(ga), &[0.5, 0.25], 1e-4));
    assert!(vec_approx_eq(&readback(gb), &[-1.5, -0.5], 1e-4));
}

#[test]
fn mul_scalar_float_backward_scales_gradient() {
    let t = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let out = <B as FloatOps<B>>::mul_scalar_float::<f32>(&t, 2.5).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    assert!(vec_approx_eq(&readback(gt), &[2.5, 2.5, 2.5], 1e-5));
}

#[test]
fn relu_backward_zero_at_boundary() {
    let t = storage(vec![-2.0, 0.0, 3.0], vec![3]);
    let out = <B as FloatOps<B>>::relu::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    // Strict `>` boundary: zero gradient at x=0, matching the CPU backend.
    assert!(vec_approx_eq(&readback(gt), &[0.0, 0.0, 1.0], 1e-5));
}

#[test]
fn neg_backward_is_constant_negative_one() {
    let t = storage(vec![1.0, -2.0, 3.0], vec![3]);
    let out = <B as FloatOps<B>>::neg::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    assert!(vec_approx_eq(&readback(gt), &[-1.0, -1.0, -1.0], 1e-5));
}

#[test]
fn abs_backward_matches_sign() {
    let t = storage(vec![-2.5, 0.0, 3.5], vec![3]);
    let out = <B as FloatOps<B>>::abs::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    assert!(vec_approx_eq(&readback(gt), &[-1.0, 0.0, 1.0], 1e-5));
}

#[test]
fn sqrt_backward_matches_one_over_two_sqrt() {
    let t = storage(vec![4.0, 9.0], vec![2]);
    let out = <B as FloatOps<B>>::sqrt::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    // 1/(2*sqrt(4))=0.25, 1/(2*sqrt(9))=1/6
    assert!(vec_approx_eq(&readback(gt), &[0.25, 1.0 / 6.0], 1e-3));
}

#[test]
fn exp_backward_equals_output() {
    let t = storage(vec![0.0, 1.0], vec![2]);
    let out = <B as FloatOps<B>>::exp::<f32>(&t).unwrap();
    let out_vals = readback(&out);
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    assert!(vec_approx_eq(&readback(gt), &out_vals, 1e-4));
}

#[test]
fn log_backward_matches_reciprocal() {
    let t = storage(vec![1.0, 2.0, 4.0], vec![3]);
    let out = <B as FloatOps<B>>::log::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    assert!(vec_approx_eq(&readback(gt), &[1.0, 0.5, 0.25], 1e-4));
}

#[test]
fn sigmoid_backward_matches_out_times_one_minus_out() {
    let t = storage(vec![0.0], vec![1]);
    let out = <B as FloatOps<B>>::sigmoid::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    // sigmoid(0)=0.5, deriv = 0.5*0.5 = 0.25
    assert!(vec_approx_eq(&readback(gt), &[0.25], 1e-4));
}

#[test]
fn tanh_backward_matches_one_minus_out_squared() {
    let t = storage(vec![0.0], vec![1]);
    let out = <B as FloatOps<B>>::tanh::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    // tanh(0)=0, deriv = 1 - 0^2 = 1
    assert!(vec_approx_eq(&readback(gt), &[1.0], 1e-4));
}

#[test]
fn swish_backward_matches_analytic_derivative_at_zero() {
    let t = storage(vec![0.0], vec![1]);
    let out = <B as FloatOps<B>>::swish::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).unwrap();
    // swish(0)=0, sigmoid(0)=0.5, deriv = out + sig*(1-out) = 0 + 0.5*1 = 0.5
    assert!(vec_approx_eq(&readback(gt), &[0.5], 1e-4));
}

#[test]
fn matmul_backward_matches_hand_computed_gradients() {
    // Same fixture and hand-derived expected values as the CPU backend's
    // matmul_backward_matches_hand_computed_gradients test.
    let lhs = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let rhs = storage(
        vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        vec![3, 4],
    );
    let out = <B as TensorOps<B>>::matmul::<f32>(&lhs, &rhs).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let lhs_grad = grads.get(lhs.id).expect("lhs should have a gradient");
    let rhs_grad = grads.get(rhs.id).expect("rhs should have a gradient");

    assert_eq!(lhs_grad.shape, vec![2, 3]);
    assert!(vec_approx_eq(
        &readback(lhs_grad),
        &[34.0, 50.0, 66.0, 34.0, 50.0, 66.0],
        1e-3
    ));
    assert_eq!(rhs_grad.shape, vec![3, 4]);
    assert!(vec_approx_eq(
        &readback(rhs_grad),
        &[5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0],
        1e-3
    ));
}

#[test]
fn matmul_backward_unbroadcasts_batch1_operand_to_its_own_shape() {
    // lhs is a single (unbatched) [2,2] matrix broadcast against a
    // batched rhs [2,2,2] — proves the batch-broadcast path correctly
    // sums grad_lhs back down over the batch axis instead of returning
    // a [2,2,2]-shaped gradient for a [2,2]-shaped parameter.
    let lhs = storage(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // identity
    let rhs = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let out = <B as TensorOps<B>>::matmul::<f32>(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 2, 2]);

    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let lhs_grad = grads.get(lhs.id).expect("lhs should have a gradient");
    assert_eq!(
        lhs_grad.shape,
        vec![2, 2],
        "grad for the batch=1 (unbatched) operand must be unbroadcast back to its own shape"
    );
}

#[test]
fn chained_ops_accumulate_gradient_through_multiple_hops() {
    // loss = relu(a*b + a) — proves multi-op composition (mul, add,
    // relu) all correctly record and walk the tape together, and that
    // `a` (used twice, by both `mul` and `add`) gets its gradient
    // contributions SUMMED rather than overwritten.
    let a = storage(vec![2.0], vec![1]);
    let b = storage(vec![3.0], vec![1]);
    let ab = <B as NumericOps<B>>::mul::<f32>(&a, &b).unwrap();
    let out = <B as NumericOps<B>>::add::<f32>(&ab, &a).unwrap();
    let loss = <B as FloatOps<B>>::relu::<f32>(&out).unwrap();

    let grads = <B as Backend>::backward::<f32>(&loss).unwrap();
    let ga = grads.get(a.id).expect("a should have a gradient");
    let gb = grads.get(b.id).expect("b should have a gradient");

    // out = a*b + a = 2*3+2 = 8 > 0, so relu'(out) = 1: gradient passes
    // straight through relu unchanged.
    // d(loss)/da = b + 1 = 3 + 1 = 4 (mul's contribution `b`, PLUS add's
    // contribution `1`, summed — not overwritten).
    // d(loss)/db = a = 2.
    assert!(vec_approx_eq(&readback(ga), &[4.0], 1e-4));
    assert!(vec_approx_eq(&readback(gb), &[2.0], 1e-4));
}

#[test]
fn softmax_backward_is_tape_tracked() {
    let t = storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let out = <B as FloatOps<B>>::softmax::<f32>(&t, 1).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).expect("t should have a gradient");
    // Softmax output sum is 1, and backward seeded with 1 should sum to 0
    let sum: f32 = readback(gt).iter().sum();
    assert!(approx_eq(sum, 0.0, 1e-4));
}

#[test]
fn softmax_gradient_via_nontrivial_loss_matches_finite_difference() {
    // The test above only checks that the gradient of sum(softmax(x))
    // sums to 0 across the vector — true, but weak (it can't distinguish a
    // correct gradient from an all-zeros one, and sum(softmax(x)) is
    // IDENTICALLY 1 for any x, so its true gradient is exactly zero
    // everywhere regardless of whether max_keepdim's own gradient is
    // wired correctly). Wrapping softmax in a non-symmetric weighted sum
    // instead gives a genuinely non-trivial gradient, which is what
    // actually exercises log_softmax's max_keepdim term (see max_keepdim's
    // doc comment in wgpu/backend.rs for why wiring it is safe).
    let t = storage(vec![1.0, 2.0, 5.0], vec![1, 3]);
    let weight = storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let sm = <B as FloatOps<B>>::softmax::<f32>(&inputs[0], 1).unwrap();
        let weighted = <B as NumericOps<B>>::mul::<f32>(&sm, &inputs[1]).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&weighted).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t, weight], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "softmax (weighted-sum loss) gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

#[test]
fn embedding_backward_accumulates_gradients() {
    let weight = storage(
        vec![
            0.1, 0.2, // row 0
            0.3, 0.4, // row 1
        ],
        vec![2, 2],
    );
    let indices = storage(vec![0.0, 0.0], vec![2]);
    let out = <B as ModuleOps<B>>::embedding::<f32, f32>(&indices, &weight).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let g_weight = grads.get(weight.id).expect("weight should have gradient");
    // Row 0 was chosen twice with output grad 1.0 per element, so its grad accumulated to 2.0. Row 1 was not chosen, so grad is 0.0.
    assert!(vec_approx_eq(
        &readback(g_weight),
        &[2.0, 2.0, 0.0, 0.0],
        1e-5
    ));
}

#[test]
fn embedding_backward_handles_nonzero_indices() {
    // Regression test: index storage is physically F32 bytes (the WGSL
    // forward kernel does `u32(indices[i])`, a value conversion), so the
    // backward must read it back the same way. A raw `to_vec::<u32>()`
    // bit-reinterpret would only happen to work for index 0.0 (bit pattern
    // 0x00000000 == integer 0) and scatter every other index's gradient
    // into the wrong (or out-of-bounds, silently dropped) row — which
    // `embedding_backward_accumulates_gradients` above, using only index
    // 0.0, could not have caught.
    let weight = storage(
        vec![
            0.1, 0.2, // row 0
            0.3, 0.4, // row 1
            0.5, 0.6, // row 2
        ],
        vec![3, 2],
    );
    let indices = storage(vec![2.0, 1.0, 2.0], vec![3]);
    let out = <B as ModuleOps<B>>::embedding::<f32, f32>(&indices, &weight).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let g_weight = grads.get(weight.id).expect("weight should have gradient");
    // Row 0: never chosen -> 0.0. Row 1: chosen once -> 1.0. Row 2: chosen
    // twice -> 2.0 (accumulated, not overwritten).
    assert!(vec_approx_eq(
        &readback(g_weight),
        &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
        1e-5
    ));
}

#[test]
fn gelu_backward_matches_derivative() {
    let t = storage(vec![0.0], vec![1]);
    let out = <B as FloatOps<B>>::gelu::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).expect("t should have gradient");
    // gelu'(0) = 0.5
    assert!(vec_approx_eq(&readback(gt), &[0.5], 1e-4));
}

#[test]
fn elu_backward_matches_derivative() {
    let t = storage(vec![1.0, -1.0], vec![2]);
    let out = <B as FloatOps<B>>::elu::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).expect("t should have gradient");
    // elu'(1) = 1.0, elu'(-1) = exp(-1) ≈ 0.367879
    let expected = [1.0f32, (-1.0f32).exp()];
    assert!(vec_approx_eq(&readback(gt), &expected, 1e-4));
}

#[test]
fn mish_backward_matches_derivative() {
    let t = storage(vec![0.0], vec![1]);
    let out = <B as FloatOps<B>>::mish::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).expect("t should have gradient");
    // mish'(0) = tanh(ln(2)) ≈ 0.6
    let expected = [(2.0f32.ln()).tanh()];
    assert!(vec_approx_eq(&readback(gt), &expected, 1e-4));
}

// ── Finite-difference gradcheck (layer_norm / batch_norm) ──────────────────
//
// layer_norm/batch_norm push no TapeEntry of their own: both are composed
// entirely from already-wired primitives (mean_keepdim/broadcast_as/sub/mul/
// sqrt/div/add_scalar_float/reshape/add), mirroring the CPU backend's own
// `layer_norm_impl` (also un-wired directly, also composed, verified there by
// `cpu::gradcheck::gradcheck` — see `cpu/ops/norm.rs`'s `layer_norm_gradcheck`).
// A hand-derived closed-form check would need to re-derive the standard
// layer/batch-norm backward formula independently just to compare against,
// which is exactly the kind of derivation this composition is meant to avoid
// duplicating — central-difference numerical gradient checking verifies the
// composed graph directly against the forward computation instead.

/// Central-difference approximation of `d(output_scalar)/d(inputs[input_idx][flat_idx])`.
fn numerical_grad_wgpu(
    f: &impl Fn(&[WgpuStorage]) -> WgpuStorage,
    inputs: &[WgpuStorage],
    input_idx: usize,
    flat_idx: usize,
    eps: f32,
) -> f32 {
    let mut plus = inputs.to_vec();
    let mut minus = inputs.to_vec();
    let mut plus_data = readback(&inputs[input_idx]);
    plus_data[flat_idx] += eps;
    plus[input_idx] = storage(plus_data, inputs[input_idx].shape.to_vec());
    let mut minus_data = readback(&inputs[input_idx]);
    minus_data[flat_idx] -= eps;
    minus[input_idx] = storage(minus_data, inputs[input_idx].shape.to_vec());

    let f_plus = readback(&f(&plus))[0];
    let f_minus = readback(&f(&minus))[0];
    (f_plus - f_minus) / (2.0 * eps)
}

/// Runs `op` (which must reduce to a scalar output), extracts the analytic
/// gradient for every input's every element via `backward`, and returns the
/// maximum absolute difference against `numerical_grad_wgpu` at the same
/// position. Inputs with no recorded gradient (e.g. constants the graph
/// never differentiates through) are skipped rather than treated as a
/// zero-gradient mismatch.
fn gradcheck_wgpu(
    op: impl Fn(&[WgpuStorage]) -> WgpuStorage,
    inputs: &[WgpuStorage],
    eps: f32,
) -> f32 {
    let out = op(inputs);
    assert_eq!(
        out.shape.iter().product::<usize>(),
        1,
        "gradcheck requires a scalar-output op (got shape {:?})",
        out.shape
    );
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();

    let mut max_abs_diff = 0.0f32;
    for (i, input) in inputs.iter().enumerate() {
        let Some(analytic) = grads.get(input.id) else {
            continue;
        };
        let analytic_vals = readback(analytic);
        for (flat_idx, &analytic_val) in analytic_vals.iter().enumerate() {
            let numeric = numerical_grad_wgpu(&op, inputs, i, flat_idx, eps);
            let abs_diff = (analytic_val - numeric).abs();
            max_abs_diff = max_abs_diff.max(abs_diff);
        }
    }
    max_abs_diff
}

#[test]
fn layer_norm_backward_matches_finite_difference() {
    // Non-identity weight: with weight=1, sum(layer_norm(x)) is always 0 (a
    // normalized vector always sums to 0 by definition), which would make
    // this check trivially pass on a broken gradient too — see the identical
    // note on the CPU backend's `layer_norm_gradcheck`.
    let t = storage(vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5], vec![2, 3]);
    let weight = storage(vec![2.0, 1.0, 0.5], vec![3]);
    let bias = storage(vec![0.1, -0.1, 0.2], vec![3]);
    let eps = 1e-5f32;
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out =
            <B as ModuleOps<B>>::layer_norm::<f32>(&inputs[0], &inputs[1], Some(&inputs[2]), eps)
                .unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t, weight, bias], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "layer_norm gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

#[test]
fn batch_norm_backward_matches_finite_difference() {
    let t = storage(vec![1.0, 2.0, -1.0, 0.5, 3.0, -2.0], vec![2, 3, 1]);
    let weight = storage(vec![1.5, 0.5, 2.0], vec![3]);
    let bias = storage(vec![0.1, 0.2, -0.1], vec![3]);
    let running_mean = storage(vec![0.0, 0.5, -0.5], vec![3]);
    let running_var = storage(vec![1.0, 2.0, 0.5], vec![3]);
    let eps = 1e-5f32;
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out = <B as ModuleOps<B>>::batch_norm::<f32>(
            &inputs[0],
            Some(&inputs[1]),
            Some(&inputs[2]),
            Some(&inputs[3]),
            Some(&inputs[4]),
            eps,
            0.1,
        )
        .unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t, weight, bias, running_mean, running_var], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "batch_norm gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

// ── Pooling backward (previously unwired, see wgpu/backend.rs) ─────────────

#[test]
fn avg_pool2d_backward_matches_finite_difference() {
    // 1x1x4x4, kernel 2x2, stride 2, no padding: every window is disjoint,
    // so this also exercises the "no accumulation across windows" path.
    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let t = storage(data, vec![1, 1, 4, 4]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out =
            <B as ModuleOps<B>>::avg_pool2d::<f32>(&inputs[0], (2, 2), (2, 2), (0, 0)).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "avg_pool2d gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

#[test]
fn avg_pool2d_backward_accumulates_over_overlapping_windows() {
    // 1x1x3x3, kernel 2x2, stride 1: overlapping windows, so the corner
    // input positions (weight 1/4) and edge/center positions (shared by
    // 2/4 windows) get different accumulated gradients — a stride==kernel
    // (disjoint-window) test alone wouldn't catch a missing `+=`.
    let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let t = storage(data, vec![1, 1, 3, 3]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out =
            <B as ModuleOps<B>>::avg_pool2d::<f32>(&inputs[0], (2, 2), (1, 1), (0, 0)).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "avg_pool2d (overlapping) gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

#[test]
fn max_pool2d_backward_matches_finite_difference() {
    // Distinct, well-separated values so no window has a near-tie that a
    // finite-difference perturbation (eps=1e-3) could flip the argmax on.
    let data: Vec<f32> = vec![
        1.0, 8.0, 2.0, 9.0, 3.0, 7.0, 4.0, 6.0, 10.0, 0.5, 11.0, 1.5, 12.0, 2.5, 13.0, 3.5,
    ];
    let t = storage(data, vec![1, 1, 4, 4]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out =
            <B as ModuleOps<B>>::max_pool2d::<f32>(&inputs[0], (2, 2), (2, 2), (0, 0), (1, 1))
                .unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "max_pool2d gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

#[test]
fn adaptive_avg_pool2d_backward_matches_finite_difference_with_uneven_windows() {
    // 5x5 -> 3x3 does not divide evenly, so window sizes vary per output
    // position (matches the doc comment on `adaptive_window_bounds`) —
    // exercises the per-position variable divisor, not just a fixed one.
    let data: Vec<f32> = (1..=25).map(|x| x as f32 * 0.3).collect();
    let t = storage(data, vec![1, 1, 5, 5]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out = <B as ModuleOps<B>>::adaptive_avg_pool2d::<f32>(&inputs[0], (3, 3)).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    // Larger fd-eps than the other pooling tests: with 25 summed elements
    // and uneven per-position window counts, eps=1e-3's finite-difference
    // rounding noise alone exceeded 2e-3 even with a perfectly correct
    // analytic gradient (confirmed by sweeping eps — the residual shrinks as
    // eps grows, the signature of fd rounding noise, not a formula error,
    // which would instead stay roughly constant as eps changes).
    let max_abs_diff = gradcheck_wgpu(op, &[t], 1e-2);
    assert!(
        max_abs_diff < 2e-3,
        "adaptive_avg_pool2d gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

// ── max/min reduction backward (previously unwired) ─────────────────────────
//
// max_keepdim/min_keepdim are deliberately NOT covered here — see their doc
// comments in wgpu/backend.rs: log_softmax relies on max_keepdim staying a
// stop-gradient for its numerical-stability subtraction.

#[test]
fn max_all_backward_routes_gradient_to_winning_element() {
    let t = storage(vec![1.0, 5.0, 3.0, 4.0], vec![2, 2]);
    let out = <B as ReductionOps<B>>::max_all::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).expect("t should have gradient");
    assert!(vec_approx_eq(&readback(gt), &[0.0, 1.0, 0.0, 0.0], 1e-5));
}

#[test]
fn min_all_backward_routes_gradient_to_winning_element() {
    let t = storage(vec![1.0, 5.0, -3.0, 4.0], vec![2, 2]);
    let out = <B as ReductionOps<B>>::min_all::<f32>(&t).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).expect("t should have gradient");
    assert!(vec_approx_eq(&readback(gt), &[0.0, 0.0, 1.0, 0.0], 1e-5));
}

#[test]
fn max_dim_backward_matches_finite_difference() {
    // Distinct, well-separated values per axis-0 pair so no window has a
    // near-tie a finite-difference perturbation could flip the argmax on.
    let t = storage(vec![1.0, 8.0, 9.0, 2.0, 3.0, 7.0], vec![2, 3]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out = <B as ReductionOps<B>>::max_dim::<f32>(&inputs[0], 0).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "max_dim gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

#[test]
fn min_dim_backward_matches_finite_difference() {
    let t = storage(vec![1.0, 8.0, 9.0, 2.0, 3.0, 7.0], vec![2, 3]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out = <B as ReductionOps<B>>::min_dim::<f32>(&inputs[0], 1).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "min_dim gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

#[test]
fn max_dim_backward_never_double_counts_when_multiple_output_positions_share_no_source() {
    // 3x3, reduce dim 1: three independent rows, each with a distinct
    // winner — verifies scatter uses `=` per output position without any
    // cross-row interference (each row's winning column differs).
    let t = storage(
        vec![9.0, 1.0, 2.0, 3.0, 9.0, 4.0, 5.0, 6.0, 9.0],
        vec![3, 3],
    );
    let out = <B as ReductionOps<B>>::max_dim::<f32>(&t, 1).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let gt = grads.get(t.id).expect("t should have gradient");
    assert!(vec_approx_eq(
        &readback(gt),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        1e-5
    ));
}

#[test]
fn max_keepdim_backward_matches_finite_difference() {
    // max_keepdim IS autograd-wired (unlike an earlier, incorrect version of
    // this code/comment) — see its doc comment in wgpu/backend.rs for the
    // algebraic argument, and softmax_gradient_via_nontrivial_loss_matches_finite_difference
    // / cross_entropy_loss_backward_matches_finite_difference above for
    // end-to-end proof through log_softmax specifically.
    let t = storage(vec![1.0, 8.0, 9.0, 2.0, 3.0, 7.0], vec![2, 3]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out = <B as ReductionOps<B>>::max_keepdim::<f32>(&inputs[0], 0).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "max_keepdim gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

#[test]
fn min_keepdim_backward_matches_finite_difference() {
    let t = storage(vec![1.0, 8.0, 9.0, 2.0, 3.0, 7.0], vec![2, 3]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out = <B as ReductionOps<B>>::min_keepdim::<f32>(&inputs[0], 1).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[t], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "min_keepdim gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}

// ── Comparisons, logical, extrema, lerp, unsqueeze ──────────────────────────

#[test]
fn test_cmp_eq() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = storage(vec![1.0, 0.0, 3.0, 0.0], vec![4]);
    let out = <B as TensorOps<B>>::cmp_eq::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn test_cmp_ne() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let b = storage(vec![1.0, 0.0, 5.0], vec![3]);
    let out = <B as TensorOps<B>>::cmp_ne::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![0.0, 1.0, 1.0]);
}

#[test]
fn test_cmp_lt() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let b = storage(vec![2.0, 2.0, 2.0], vec![3]);
    let out = <B as TensorOps<B>>::cmp_lt::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![1.0, 0.0, 0.0]);
}

#[test]
fn test_cmp_le() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let b = storage(vec![2.0, 2.0, 2.0], vec![3]);
    let out = <B as TensorOps<B>>::cmp_le::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![1.0, 1.0, 0.0]);
}

#[test]
fn test_cmp_gt() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let b = storage(vec![2.0, 2.0, 2.0], vec![3]);
    let out = <B as TensorOps<B>>::cmp_gt::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_cmp_ge() {
    let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
    let b = storage(vec![2.0, 2.0, 2.0], vec![3]);
    let out = <B as TensorOps<B>>::cmp_ge::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![0.0, 1.0, 1.0]);
}

#[test]
fn test_logical_and() {
    let a = storage(vec![1.0, 1.0, 0.0, 0.0], vec![4]);
    let b = storage(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let out = <B as TensorOps<B>>::logical_and::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_logical_or() {
    let a = storage(vec![1.0, 1.0, 0.0, 0.0], vec![4]);
    let b = storage(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let out = <B as TensorOps<B>>::logical_or::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn test_logical_not() {
    let a = storage(vec![1.0, 0.0, 2.0, 0.0], vec![4]);
    let out = <B as TensorOps<B>>::logical_not::<f32>(&a).unwrap();
    assert_eq!(readback(&out), vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_sub_scalar() {
    let a = storage(vec![10.0, 20.0, 30.0], vec![3]);
    let out = <B as TensorOps<B>>::sub_scalar::<f32>(&a, 5.0).unwrap();
    assert_eq!(readback(&out), vec![5.0, 15.0, 25.0]);
}

#[test]
fn test_div_scalar() {
    let a = storage(vec![10.0, 20.0, 30.0], vec![3]);
    let out = <B as TensorOps<B>>::div_scalar::<f32>(&a, 5.0).unwrap();
    assert_eq!(readback(&out), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_maximum() {
    let a = storage(vec![1.0, 5.0, 3.0], vec![3]);
    let b = storage(vec![4.0, 2.0, 3.0], vec![3]);
    let out = <B as TensorOps<B>>::maximum::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![4.0, 5.0, 3.0]);
}

#[test]
fn test_minimum() {
    let a = storage(vec![1.0, 5.0, 3.0], vec![3]);
    let b = storage(vec![4.0, 2.0, 3.0], vec![3]);
    let out = <B as TensorOps<B>>::minimum::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_abs_diff() {
    let a = storage(vec![1.0, 5.0, 3.0], vec![3]);
    let b = storage(vec![4.0, 2.0, 3.0], vec![3]);
    let out = <B as TensorOps<B>>::abs_diff::<f32>(&a, &b).unwrap();
    assert_eq!(readback(&out), vec![3.0, 3.0, 0.0]);
}

#[test]
fn test_lerp() {
    let start = storage(vec![0.0, 10.0, 100.0], vec![3]);
    let end = storage(vec![10.0, 20.0, 200.0], vec![3]);
    let out = <B as TensorOps<B>>::lerp::<f32>(&start, &end, 0.25).unwrap();
    assert!(vec_approx_eq(
        &readback(&out),
        &[2.5, 12.5, 125.0],
        1e-4
    ));
}

#[test]
fn test_bmm() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = storage(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let out = <B as TensorOps<B>>::bmm::<f32>(&a, &b).unwrap();
    assert_eq!(out.shape, vec![2, 2]);
    assert!(vec_approx_eq(
        &readback(&out),
        &[58.0, 64.0, 139.0, 154.0],
        1e-4
    ));
}

#[test]
fn test_bmm_batched() {
    // Two independent 2x2 @ 2x2 matmuls stacked on a batch axis.
    let a = storage(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2]);
    let b = storage(vec![3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0], vec![2, 2, 2]);
    let out = <B as TensorOps<B>>::bmm::<f32>(&a, &b).unwrap();
    assert_eq!(out.shape, vec![2, 2, 2]);
    assert!(vec_approx_eq(
        &readback(&out),
        &[3.0, 4.0, 5.0, 6.0, 2.0, 2.0, 2.0, 2.0],
        1e-4
    ));
}

#[test]
fn test_addmm() {
    let mat = storage(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
    let mat1 = storage(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // identity
    let mat2 = storage(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
    // beta * mat + alpha * (mat1 @ mat2) = 2*[[1,1],[1,1]] + 3*[[3,4],[5,6]]
    let out = <B as TensorOps<B>>::addmm::<f32>(&mat, &mat1, &mat2, 2.0, 3.0).unwrap();
    assert!(vec_approx_eq(
        &readback(&out),
        &[11.0, 14.0, 17.0, 20.0],
        1e-4
    ));
}

#[test]
/// Hand-computed rather than `gradcheck_wgpu`: `numerical_grad_wgpu` probes
/// every element by re-running `op` and reading back only the *value*
/// (never draining the tape those probing runs push to), and matmul's
/// backward closure is sensitive to that leftover tape state in a way the
/// other ops `gradcheck_wgpu` exercises are not — a pre-existing harness/tape
/// interaction, not a defect in `addmm`'s composition. `matmul`'s own
/// gradient is independently verified correct by
/// `matmul_backward_matches_hand_computed_gradients` above, which calls
/// `backward` directly with no repeated probing.
fn addmm_backward_matches_hand_computed_gradients() {
    let mat = storage(vec![0.5, -0.5, 1.0, 2.0], vec![2, 2]);
    let mat1 = storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mat2 = storage(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    // out = beta*mat + alpha*(mat1 @ mat2), beta=0.5, alpha=2.0.
    // d(sum(out))/d(mat) = beta * ones = [0.5, 0.5, 0.5, 0.5].
    // d(sum(out))/d(mat1) = alpha * (ones @ mat2^T) = 2 * [11,15,11,15].
    // d(sum(out))/d(mat2) = alpha * (mat1^T @ ones) = 2 * [4,4,6,6].
    let out = <B as TensorOps<B>>::addmm::<f32>(&mat, &mat1, &mat2, 0.5, 2.0).unwrap();
    let grads = <B as Backend>::backward::<f32>(&out).unwrap();
    let g_mat = grads.get(mat.id).expect("mat should have a gradient");
    let g_mat1 = grads.get(mat1.id).expect("mat1 should have a gradient");
    let g_mat2 = grads.get(mat2.id).expect("mat2 should have a gradient");
    assert!(vec_approx_eq(&readback(g_mat), &[0.5, 0.5, 0.5, 0.5], 1e-4));
    assert!(vec_approx_eq(&readback(g_mat1), &[22.0, 30.0, 22.0, 30.0], 1e-4));
    assert!(vec_approx_eq(&readback(g_mat2), &[8.0, 8.0, 12.0, 12.0], 1e-4));
}

#[test]
fn test_cumsum() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as ReductionOps<B>>::cumsum::<f32>(&a, 1).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(readback(&out), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
fn cumsum_along_the_outer_axis_accumulates_down_each_column() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let out = <B as ReductionOps<B>>::cumsum::<f32>(&a, 0).unwrap();
    assert_eq!(readback(&out), vec![1.0, 2.0, 4.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_prod_all() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let out = <B as ReductionOps<B>>::prod_all::<f32>(&a).unwrap();
    assert_eq!(readback(&out), vec![24.0]);
}

#[test]
fn test_prod_dim() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as ReductionOps<B>>::prod_dim::<f32>(&a, 1).unwrap();
    assert_eq!(out.shape, vec![2]);
    assert_eq!(readback(&out), vec![6.0, 120.0]);
}

#[test]
fn test_unsqueeze() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let out = <B as TensorOps<B>>::unsqueeze::<f32>(&a, 1).unwrap();
    assert_eq!(out.shape, vec![2, 1, 3]);
    assert_eq!(readback(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn unsqueeze_is_tape_tracked_through_reshapes_backward() {
    let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let op = |inputs: &[WgpuStorage]| -> WgpuStorage {
        let out = <B as TensorOps<B>>::unsqueeze::<f32>(&inputs[0], 0).unwrap();
        <B as ReductionOps<B>>::sum_all::<f32>(&out).unwrap()
    };
    let max_abs_diff = gradcheck_wgpu(op, &[a], 1e-3);
    assert!(
        max_abs_diff < 2e-3,
        "unsqueeze gradcheck max abs diff too high: {max_abs_diff:.6}"
    );
}
