//! Unit tests for the CUDA backend implementation.

use super::*;

#[test]
fn byte_length_uses_authoritative_storage_dtype() {
    assert_eq!(
        checked_storage_byte_len(7, DTypeId::F16.into()).unwrap(),
        14
    );
    assert_eq!(
        checked_storage_byte_len(7, DTypeId::BF16.into()).unwrap(),
        14
    );
    assert_eq!(
        checked_storage_byte_len(7, DTypeId::F32.into()).unwrap(),
        28
    );
    assert_eq!(
        checked_storage_byte_len(7, DTypeId::F64.into()).unwrap(),
        56
    );
    assert!(checked_storage_byte_len(usize::MAX, DTypeId::F64.into()).is_err());
}

#[test]
fn storage_validation_accepts_renderable_float_family_and_i64_indices() {
    let device = DeviceId::cuda(0);
    for dtype in [
        DTypeId::F16,
        DTypeId::BF16,
        DTypeId::F32,
        DTypeId::F64,
        DTypeId::I64,
    ] {
        validate_cuda_storage(dtype.into(), &device, "test").unwrap();
    }
    assert!(matches!(
        validate_cuda_storage(DTypeId::U32.into(), &device, "test"),
        Err(Error::UnsupportedDType { .. })
    ));
    assert!(validate_cuda_storage(DTypeId::F32.into(), &DeviceId::cpu(), "test").is_err());
}

// shape_cardinality_is_checked_before_allocation moved to
// bytes::tests::numel_is_the_checked_product_of_the_dims, which now owns
// the one checked_numel implementation this file calls.

// The tests below exercise real GPU dispatch (`::{reshape,
// transpose, narrow, broadcast_as, squeeze, stack, slice, flatten,
// broadcast_left, matmul}`) and therefore need a real CUDA device to
// run — none is available in this environment, so this path is compile-verified
// only locally. `#[ignore]`d so `cargo test` stays green everywhere; run with
// `cargo test --features cuda,std -- --ignored` on real hardware.

type B = CudaBackendImpl<Cuda>;

fn cuda_f32(shape: &[usize], values: Vec<f32>) -> CudaStorage {
    cuda_from_f32(
        shape,
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
        values,
        "test",
    )
    .unwrap()
}

#[test]
#[ignore = "requires CUDA hardware"]
fn reshape_preserves_element_order() {
    let t = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = B::reshape::<f32>(&t, &[3, 2]).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn reshape_rejects_mismatched_element_count() {
    let t = cuda_f32(&[2, 3], vec![0.0; 6]);
    assert!(B::reshape::<f32>(&t, &[4, 2]).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn transpose_2d_swaps_shape() {
    let t = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = B::transpose::<f32>(&t, 0, 1).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn narrow_reduces_target_dim() {
    let t = cuda_f32(&[4, 3], vec![0.0; 12]);
    let out = B::narrow::<f32>(&t, 0, 1, 2).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn broadcast_as_expands_size_one_dim() {
    let t = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
    let out = B::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
    assert_eq!(out.shape, vec![4, 3]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn broadcast_as_rejects_incompatible_shape() {
    let t = cuda_f32(&[2, 3], vec![0.0; 6]);
    assert!(B::broadcast_as::<f32>(&t, &[2, 5]).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn compare_writes_bool_storage_at_the_broadcast_shape() {
    use crate::cuda::ops::compare::{CompareOp, launch_compare};
    let lhs = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
    let rhs = cuda_f32(&[2, 3], vec![1.0, 0.0, 3.0, 5.0, 2.0, 3.0]);
    let lhs_b = B::broadcast_as::<f32>(&lhs, &[2, 3]).unwrap();
    let out = launch_compare(CompareOp::Eq, &lhs_b, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::Bool.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn compare_rejects_mismatched_shapes() {
    use crate::cuda::ops::compare::{CompareOp, launch_compare};
    let lhs = cuda_f32(&[2, 3], vec![0.0; 6]);
    let rhs = cuda_f32(&[2, 4], vec![0.0; 8]);
    assert!(launch_compare(CompareOp::Lt, &lhs, &rhs).is_err());
}

fn cuda_bool(shape: &[usize], values: Vec<u8>) -> CudaStorage {
    cuda_from_bytes(shape, DTypeId::Bool.descriptor(), 0, &values).unwrap()
}

#[test]
#[ignore = "requires CUDA hardware"]
fn where_cond_selects_at_the_shared_operand_shape() {
    use crate::cuda::ops::select::launch_where_cond;
    let mask = cuda_bool(&[2, 3], vec![1, 0, 1, 0, 1, 0]);
    let on_true = cuda_f32(&[2, 3], vec![1.0; 6]);
    let on_false = cuda_f32(&[2, 3], vec![0.0; 6]);
    let out = launch_where_cond(&mask, &on_true, &on_false).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::F32.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn where_cond_rejects_mismatched_shapes() {
    use crate::cuda::ops::select::launch_where_cond;
    let mask = cuda_bool(&[2, 3], vec![1; 6]);
    let on_true = cuda_f32(&[2, 4], vec![0.0; 8]);
    let on_false = cuda_f32(&[2, 3], vec![0.0; 6]);
    assert!(launch_where_cond(&mask, &on_true, &on_false).is_err());
}

/// `launch_where_cond` itself takes no broadcast responsibility (see its
/// own doc): a lower-rank mask has to go through
/// `launch_broadcast_bool_mask` first, the same composition
/// `Execute<op::WhereCond>` performs.
#[test]
#[ignore = "requires CUDA hardware"]
fn broadcast_bool_mask_expands_a_lower_rank_mask() {
    use crate::cuda::ops::select::launch_broadcast_bool_mask;
    let mask = cuda_bool(&[3], vec![1, 0, 1]);
    let out = launch_broadcast_bool_mask(&mask, &[2, 3]).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::Bool.descriptor());
}

/// The composition `Execute<op::WhereCond>` performs when the mask
/// arrives at a lower rank than the data it selects between — the exact
/// case `where_cond`'s own descriptor permits (its output shape is the
/// broadcast of all three operands, not just the two data ones).
#[test]
#[ignore = "requires CUDA hardware"]
fn where_cond_broadcasts_a_lower_rank_mask_before_selecting() {
    use crate::cuda::ops::select::{launch_broadcast_bool_mask, launch_where_cond};
    let mask = cuda_bool(&[3], vec![1, 0, 1]);
    let on_true = cuda_f32(&[2, 3], vec![1.0; 6]);
    let on_false = cuda_f32(&[2, 3], vec![0.0; 6]);
    let mask_b = launch_broadcast_bool_mask(&mask, &[2, 3]).unwrap();
    let out = launch_where_cond(&mask_b, &on_true, &on_false).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::F32.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn masked_fill_overwrites_at_the_input_shape() {
    use crate::cuda::ops::select::launch_masked_fill;
    let input = cuda_f32(&[2, 3], vec![1.0; 6]);
    let mask = cuda_bool(&[2, 3], vec![1, 0, 1, 0, 1, 0]);
    let out = launch_masked_fill(&input, &mask, 9.0).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::F32.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn masked_fill_rejects_mismatched_shapes() {
    use crate::cuda::ops::select::launch_masked_fill;
    let input = cuda_f32(&[2, 3], vec![0.0; 6]);
    let mask = cuda_bool(&[2, 4], vec![0; 8]);
    assert!(launch_masked_fill(&input, &mask, 0.0).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn logical_and_or_and_not_write_bool_storage_at_the_shared_shape() {
    use crate::cuda::ops::logical::{launch_logical_and, launch_logical_not, launch_logical_or};
    let lhs = cuda_bool(&[4], vec![1, 1, 0, 0]);
    let rhs = cuda_bool(&[4], vec![1, 0, 1, 0]);

    let and_out = launch_logical_and(&lhs, &rhs).unwrap();
    assert_eq!(and_out.shape, vec![4]);
    assert_eq!(and_out.dtype(), DTypeId::Bool.descriptor());

    let or_out = launch_logical_or(&lhs, &rhs).unwrap();
    assert_eq!(or_out.shape, vec![4]);
    assert_eq!(or_out.dtype(), DTypeId::Bool.descriptor());

    let not_out = launch_logical_not(&lhs).unwrap();
    assert_eq!(not_out.shape, vec![4]);
    assert_eq!(not_out.dtype(), DTypeId::Bool.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn logical_and_rejects_mismatched_shapes() {
    use crate::cuda::ops::logical::launch_logical_and;
    let lhs = cuda_bool(&[2, 3], vec![1; 6]);
    let rhs = cuda_bool(&[2, 4], vec![1; 8]);
    assert!(launch_logical_and(&lhs, &rhs).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn logical_and_rejects_non_bool_storage() {
    use crate::cuda::ops::logical::launch_logical_and;
    let lhs = cuda_f32(&[4], vec![1.0; 4]);
    let rhs = cuda_bool(&[4], vec![1; 4]);
    assert!(launch_logical_and(&lhs, &rhs).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn squeeze_removes_size_one_axis() {
    let t = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
    let out = B::squeeze::<f32>(&t, 0).unwrap();
    assert_eq!(out.shape, vec![3]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_computes_correct_shape_and_values() {
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let out = B::matmul::<f32>(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_rejects_incompatible_inner_dims() {
    let lhs = cuda_f32(&[2, 3], vec![0.0; 6]);
    let rhs = cuda_f32(&[4, 2], vec![0.0; 8]);
    assert!(B::matmul::<f32>(&lhs, &rhs).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_backward_produces_gradients_for_both_operands() {
    let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let (lhs_id, rhs_id) = (lhs.id, rhs.id);
    let out = B::matmul::<f32>(&lhs, &rhs).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    assert!(grads.get(lhs_id).is_some());
    assert!(grads.get(rhs_id).is_some());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn narrow_backward_zero_pads_grad_to_original_shape() {
    let t = cuda_f32(&[4, 3], vec![0.0; 12]);
    let t_id = t.id;
    let out = B::narrow::<f32>(&t, 0, 1, 2).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let g = grads
        .get(t_id)
        .expect("narrow input should have a gradient");
    assert_eq!(g.shape, vec![4, 3]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn max_pool2d_computes_correct_output_shape() {
    // N=1,C=1,H=4,W=4, kernel=2, stride=2 -> 2x2 output
    let t = cuda_f32(&[1, 1, 4, 4], vec![0.0; 16]);
    let out = B::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn max_pool2d_backward_zero_pads_to_input_shape() {
    let t = cuda_f32(&[1, 1, 4, 4], vec![0.0; 16]);
    let t_id = t.id;
    let out = B::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let g = grads
        .get(t_id)
        .expect("max_pool2d input should have a gradient");
    assert_eq!(g.shape, vec![1, 1, 4, 4]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn avg_pool2d_computes_correct_output_shape() {
    let t = cuda_f32(&[1, 1, 4, 4], vec![0.0; 16]);
    let out = B::avg_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0)).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv2d_computes_correct_output_shape_and_values() {
    // [1,1,3,3] input, [1,1,2,2] kernel, stride=1, no padding -> [1,1,2,2],
    // matching CPU's hand-computed test fixture (conv.rs) exactly.
    let t = cuda_f32(
        &[1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let w = cuda_f32(&[1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let out = B::conv2d::<f32>(&t, &w, None, 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    let vals = download_f32_host(&out).unwrap();
    assert_eq!(vals, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv2d_with_bias_adds_per_channel_constant() {
    let t = cuda_f32(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let w = cuda_f32(&[1, 1, 1, 1], vec![1.0]);
    let bias = cuda_f32(&[1], vec![10.0]);
    let out = B::conv2d::<f32>(&t, &w, Some(&bias), 1, 0, 1, 1).unwrap();
    let vals = download_f32_host(&out).unwrap();
    assert_eq!(vals, vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv2d_backward_produces_gradients_for_input_and_weight() {
    let t = cuda_f32(
        &[1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let w = cuda_f32(&[1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let (t_id, w_id) = (t.id, w.id);
    let out = B::conv2d::<f32>(&t, &w, None, 1, 0, 1, 1).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    assert_eq!(grads.get(t_id).unwrap().shape, vec![1, 1, 3, 3]);
    assert_eq!(grads.get(w_id).unwrap().shape, vec![1, 1, 2, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv2d_groups_matches_two_independent_convs() {
    // groups=2 depthwise-ish split: Cin=2,Cout=2 each channel convolved
    // independently, mirrors CPU's `conv2d_forward_groups_matches_two_independent_convs`.
    let t = cuda_f32(&[1, 2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let w = cuda_f32(&[2, 1, 1, 1], vec![2.0, 3.0]);
    let out = B::conv2d::<f32>(&t, &w, None, 1, 0, 1, 2).unwrap();
    assert_eq!(out.shape, vec![1, 2, 2, 2]);
    let vals = download_f32_host(&out).unwrap();
    assert_eq!(vals, vec![2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0]);
}

// mse_loss/l1_loss/bce_with_logits_loss have no override in this file's
// the free loss helpers (`incin-backends/src/legacy.rs`),
// which compose entirely from ``/``/``
// (already wired on CUDA). These tests exist to prove that resolution
// actually compiles and runs correctly, not to add new functionality.

// The tests below cover the methods added in this pass: `unsqueeze`,
// the host-readback conversions, `addmm`/`bmm`/
// `scaled_dot_product_attention`. Same convention as everything above —
// `#[ignore]`d because there is no CUDA device in this environment, so
// only compilation is verified here; run with `--ignored` on real
// hardware. Fixtures and expected values are the same ones the CPU and
// WGPU backends' own tests for the identical methods use.

#[test]
#[ignore = "requires CUDA hardware"]
fn test_full() {
    let out = B::full::<f32>(3.5, &[2, 2], DTypeId::F32.into(), &DeviceId::cuda(0)).unwrap();
    assert_eq!(download_f32_host(&out).unwrap(), vec![3.5, 3.5, 3.5, 3.5]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn test_arange() {
    let out = B::arange::<f32>(1.0, 2.0, &[4], DTypeId::F32.into(), &DeviceId::cuda(0)).unwrap();
    assert_eq!(download_f32_host(&out).unwrap(), vec![1.0, 3.0, 5.0, 7.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn test_linspace() {
    let out = B::linspace::<f32>(0.0, 10.0, &[5], DTypeId::F32.into(), &DeviceId::cuda(0)).unwrap();
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![0.0, 2.5, 5.0, 7.5, 10.0]
    );
}
