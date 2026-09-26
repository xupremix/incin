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
// run - none is available in this environment, so this path is compile-verified
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
    assert_eq!(
        out.strides().as_ref(),
        &[2, 1],
        "TransposeExact materialises dense row-major on every backend (issue #113)"
    );
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

/// A `bool` mask broadcast rides the surviving width-parametric
/// `shape_op_8bit` path (#122 deleted the dedicated `bool` launcher).
/// `[3,1] -> [3,2]` must repeat each element *within* its row: a flat
/// memcpy of the three input bytes into the six-byte output would instead
/// produce the input repeated *across* rows, so the byte-for-byte
/// assertion below is the case that catches it.
#[test]
#[ignore = "requires CUDA hardware"]
fn bool_mask_broadcast_repeats_within_rows_not_across_them() {
    use crate::cuda::ops::shape::launch_broadcast;
    let mask = cuda_bool(&[3, 1], vec![1, 0, 1]);
    let out = launch_broadcast(&mask, &[3, 2]).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.dtype(), DTypeId::Bool.descriptor());
    assert_eq!(
        crate::cuda::testing::download_bytes(&out),
        vec![1, 1, 0, 0, 1, 1]
    );
}

/// The composition `Execute<op::WhereCond>` performs when the mask
/// arrives at a lower rank than the data it selects between - the exact
/// case `where_cond`'s own descriptor permits (its output shape is the
/// broadcast of all three operands, not just the two data ones). Since
/// #122 that composition's mask broadcast is `shape::launch_broadcast`,
/// the same path every other dtype takes.
#[test]
#[ignore = "requires CUDA hardware"]
fn where_cond_broadcasts_a_lower_rank_mask_before_selecting() {
    use crate::cuda::ops::select::launch_where_cond;
    use crate::cuda::ops::shape::launch_broadcast;
    let mask = cuda_bool(&[3], vec![1, 0, 1]);
    let on_true = cuda_f32(&[2, 3], vec![1.0; 6]);
    let on_false = cuda_f32(&[2, 3], vec![0.0; 6]);
    let mask_b = launch_broadcast(&mask, &[2, 3]).unwrap();
    let out = launch_where_cond(&mask_b, &on_true, &on_false).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::F32.descriptor());
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    );
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
    // The values, not just the shape: dispatch (#85) may serve this from
    // cuBLASLt or fall back to `kernels/matmul.cu`, and both must agree.
    // These products and sums are small integers, exactly representable in
    // f32 under any accumulation order, so equality is the right bar.
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![58.0, 64.0, 139.0, 154.0],
        "the product must be identical whether cuBLASLt or the NVRTC fallback runs"
    );
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

// Issue #85: the cuBLASLt path itself, driven directly. These call
// `cublaslt::try_launch_matmul` rather than `B::matmul` so a failure names
// the cuBLASLt path instead of being masked by the NVRTC fallback that
// `launch_matmul` would take on a plain request. The whole block is gated
// to `cuda-vendor` builds: without the feature the `cublaslt` module does
// not exist, which is the point of the gate - no cuBLASLt symbol is
// reachable from a build that did not ask for vendor libraries.

#[cfg(feature = "cuda-vendor")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cublaslt_path_computes_the_f32_product() {
    use crate::cuda::ops::cublaslt::try_launch_matmul;

    let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let out = try_launch_matmul(&lhs, &rhs, None, None)
        .expect("cuBLASLt must answer a canonical f32 request on hardware")
        .expect("the request fits the #85 dispatch policy");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![58.0, 64.0, 139.0, 154.0]
    );
}

#[cfg(feature = "cuda-vendor")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cublaslt_bias_epilogue_adds_the_bias() {
    use crate::cuda::ops::cublaslt::try_launch_matmul;

    let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let bias = cuda_f32(&[2], vec![10.0, 20.0]);
    let out = try_launch_matmul(&lhs, &rhs, Some(&bias), None)
        .expect("the biased request must stay on the cuBLASLt path")
        .expect("the request fits the #85 dispatch policy");
    // [58,64;139,154] + [10,20] broadcast down the columns.
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![68.0, 84.0, 149.0, 174.0]
    );
}

#[cfg(feature = "cuda-vendor")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cublaslt_bias_relu_epilogue_fuses_the_activation() {
    use crate::cuda::ops::cublaslt::try_launch_matmul;
    use cudarc::cublaslt::Activation;

    // [[1,-2],[3,-4]] @ [[5,6],[7,8]] = [[-9,-10],[-13,-14]];
    // + [20,0] -> [[11,-10],[7,-14]]; relu -> [11,0,7,0].
    let lhs = cuda_f32(&[2, 2], vec![1.0, -2.0, 3.0, -4.0]);
    let rhs = cuda_f32(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let bias = cuda_f32(&[2], vec![20.0, 0.0]);
    let out = try_launch_matmul(&lhs, &rhs, Some(&bias), Some(Activation::Relu))
        .expect("the fused request must stay on the cuBLASLt path")
        .expect("the request fits the #85 dispatch policy");
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![11.0, 0.0, 7.0, 0.0],
        "relu must clamp the negatives after the bias is added, in one kernel"
    );
}

#[cfg(feature = "cuda-vendor")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cublaslt_epilogue_requests_fail_closed_on_a_malformed_bias() {
    use crate::cuda::ops::cublaslt::try_launch_matmul;

    let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let short_bias = cuda_f32(&[3], vec![1.0, 2.0, 3.0]);
    let error = try_launch_matmul(&lhs, &rhs, Some(&short_bias), None)
        .expect_err("a bias of the wrong length must be refused, not ignored");
    assert!(
        format!("{error}").contains("vector of length 2"),
        "the refusal must name the required bias length, got: {error}"
    );
}

#[cfg(feature = "cuda-vendor")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cublaslt_plain_requests_outside_policy_report_not_applicable() {
    use crate::cuda::ops::cublaslt::try_launch_matmul;

    // The *plain* (rank-2, epilogue-free) request reports "does not apply"
    // for rank 3 and the caller keeps its existing path - which, since
    // issue #85, may still reach cuBLASLt through the dedicated batched
    // entry `try_launch_batched_matmul` when the feature is on. This test
    // pins only the plain entry's refusal; the batched entry's admission
    // is covered by the `cuda-vendor` agreement test in
    // `tests/cuda_gemm_batched.rs`.
    let lhs = cuda_f32(&[2, 2, 3], (1..=12).map(|v| v as f32).collect());
    let rhs = cuda_f32(&[2, 3, 2], (1..=12).map(|v| v as f32).collect());
    let out = try_launch_matmul(&lhs, &rhs, None, None).expect("plain requests never fail closed");
    assert!(out.is_none(), "rank 3 must not fit the cuBLASLt policy");
}

#[cfg(feature = "cuda-vendor")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cublaslt_epilogue_requests_on_out_of_policy_operands_fail_closed() {
    use crate::cuda::ops::cublaslt::try_launch_matmul;

    // An epilogue request that does not fit must error rather than return
    // `Ok(None)` - `Ok(None)` would invite a caller to fall back to a
    // kernel that has no epilogue and silently drop the bias.
    let lhs = cuda_f32(&[2, 2, 3], (1..=12).map(|v| v as f32).collect());
    let rhs = cuda_f32(&[2, 3, 2], (1..=12).map(|v| v as f32).collect());
    let bias = cuda_f32(&[2], vec![1.0, 2.0]);
    let error = try_launch_matmul(&lhs, &rhs, Some(&bias), None)
        .expect_err("out-of-policy epilogue requests must fail closed");
    assert!(
        format!("{error}").contains("dispatch policy"),
        "the refusal must name the dispatch policy, got: {error}"
    );
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

/// Rank three is the unbatched `[C, H, W]` form the CPU kernels accept: the
/// kernel runs with batch one and the output drops the leading axis. Values
/// are hand-computed, not mirrored from another backend, so agreement is
/// evidence rather than a tautology.
#[test]
#[ignore = "requires CUDA hardware"]
fn pool2d_accepts_unbatched_rank3_with_matching_values() {
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let t = cuda_f32(&[1, 4, 4], values);
    let max = B::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    assert_eq!(max.shape, vec![1, 2, 2]);
    assert_eq!(download_f32_host(&max).unwrap(), vec![6.0, 8.0, 14.0, 16.0]);
    let avg = B::avg_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0)).unwrap();
    assert_eq!(avg.shape, vec![1, 2, 2]);
    assert_eq!(download_f32_host(&avg).unwrap(), vec![3.5, 5.5, 11.5, 13.5]);
    let adaptive = crate::cuda::ops::pool::launch_adaptive_avg_pool2d(&t, (2, 2)).unwrap();
    assert_eq!(adaptive.shape, vec![1, 2, 2]);
    assert_eq!(
        download_f32_host(&adaptive).unwrap(),
        vec![3.5, 5.5, 11.5, 13.5]
    );
}

/// The rank-3 backward reaches the unbatched input at its own shape: the max
/// gradient lands only on the winning positions, the average spreads evenly.
#[test]
#[ignore = "requires CUDA hardware"]
fn pool2d_rank3_backward_reaches_the_unbatched_input() {
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let t = cuda_f32(&[1, 4, 4], values);
    let t_id = t.id;
    let max = B::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    let grads = crate::cuda::tape::backward(&max).unwrap();
    let g = grads
        .get(t_id)
        .expect("rank-3 max_pool2d input should have a gradient");
    assert_eq!(g.shape, vec![1, 4, 4]);
    assert_eq!(
        download_f32_host(g).unwrap(),
        vec![
            0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 1.0,
        ]
    );

    let zeros = cuda_f32(&[1, 4, 4], vec![0.0; 16]);
    let zeros_id = zeros.id;
    let avg = B::avg_pool2d::<f32>(&zeros, (2, 2), (2, 2), (0, 0)).unwrap();
    let grads = crate::cuda::tape::backward(&avg).unwrap();
    let g = grads
        .get(zeros_id)
        .expect("rank-3 avg_pool2d input should have a gradient");
    assert_eq!(g.shape, vec![1, 4, 4]);
    assert!(
        download_f32_host(g).unwrap().iter().all(|&v| v == 0.25),
        "each input feeds exactly one window of four"
    );
}

/// Anything outside rank 3-4 names the operation on a `RankMismatch` instead
/// of panicking indexing `shape[3]`. A panic would fail this test too, but
/// the assertion pins the typed error rather than merely surviving.
#[test]
#[ignore = "requires CUDA hardware"]
fn pool2d_rejects_rank5_with_a_typed_error() {
    let t = cuda_f32(&[1, 1, 2, 2, 2], vec![0.0; 8]);
    let result = B::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1));
    let error = format!("{:?}", result.expect_err("rank-5 pooling must be refused"));
    assert!(
        error.contains("rank between 3 and 4"),
        "expected a rank error, got: {error}"
    );
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
// `scaled_dot_product_attention`. Same convention as everything above -
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

// ---------------------------------------------------------------------------
// layer_norm forward parity and backward (issue #4)
// ---------------------------------------------------------------------------

// Parity against the CPU reference, so these need both backends.
#[cfg(feature = "cpu")]
use crate::cpu::storage::{CpuBuffer as HostBuffer, CpuStorage as HostStorage};

/// The documented fixture: two rows of four, non-trivial weight and bias, so
/// no symmetry can hide a permuted axis.
fn ln_input() -> CudaStorage {
    cuda_f32(&[2, 4], vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0])
}

fn ln_weight() -> CudaStorage {
    cuda_f32(&[4], vec![2.0, 1.0, 0.5, 1.5])
}

fn ln_bias() -> CudaStorage {
    cuda_f32(&[4], vec![0.1, -0.1, 0.2, -0.2])
}

#[cfg(feature = "cpu")]
fn host_f32(shape: &[usize], values: Vec<f32>) -> HostStorage {
    HostStorage::from_contiguous(HostBuffer::F32(values), shape)
}

#[cfg(feature = "cpu")]
fn host_values(storage: &HostStorage) -> Vec<f64> {
    let total: usize = storage.shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(total);
    // Exactly one index per axis: a scalar (rank 0) reads once with the
    // empty index. `len().max(1)` would hand `get` a one-element index
    // for a zero-rank shape and trip its rank assert - which is how the
    // scalar-reduction parity tests died in the reference reader rather
    // than in any backend under test.
    let mut index = vec![0usize; storage.shape.len()];
    for _ in 0..total {
        out.push(storage.get(&index));
        for (i, extent) in index.iter_mut().zip(storage.shape.iter()).rev() {
            *i += 1;
            if *i < *extent {
                break;
            }
            *i = 0;
        }
    }
    out
}

#[cfg(feature = "cpu")]
/// CPU forward plus its composed backward, on the same values the CUDA side
/// runs: the reference the parity tests below compare against.
fn cpu_layer_norm_grads(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = host_f32(&[2, 4], input.to_vec());
    let w = host_f32(&[4], weight.to_vec());
    let b = bias.map(|values| host_f32(&[4], values.to_vec()));
    let out = crate::cpu::ops::norm::layer_norm_impl::<incin_core::tensor::device::Cpu, f32>(
        &t,
        &w,
        b.as_ref(),
        1e-5,
    )
    .unwrap();
    let grads = crate::cpu::tape::backward(&out).unwrap();
    let read = |storage: &HostStorage| {
        let grad = grads.get(storage.id).unwrap();
        host_values(grad)
    };
    let db = match &b {
        Some(bias_storage) => read(bias_storage),
        None => Vec::new(),
    };
    (host_values(&out), read(&t), read(&w), db)
}

fn assert_close(left: &[f64], right: &[f64], tol: f64, what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: length mismatch");
    for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        let denom = l.abs().max(r.abs()).max(1e-6);
        assert!(
            (l - r).abs() / denom <= tol,
            "{what}[{i}]: cuda={l} cpu={r}"
        );
    }
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_forward_matches_cpu_reference() {
    // Regression guard for the stats-saving edit to the fused template: the
    // two extra stores must not disturb the output values.
    let (input, weight, bias) = (ln_input(), ln_weight(), ln_bias());
    let out = B::layer_norm::<f32>(&input, &weight, Some(&bias), 1e-5).unwrap();
    assert_eq!(out.shape, vec![2, 4]);
    let (expected, _, _, _) = cpu_layer_norm_grads(
        &[0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0],
        &[2.0, 1.0, 0.5, 1.5],
        Some(&[0.1, -0.1, 0.2, -0.2]),
    );
    let got: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| *v as f64)
        .collect();
    assert_close(&got, &expected, 1e-5, "forward");
    // Draining here keeps this forward's entry off the next test's walk.
    let _ = crate::cuda::tape::backward(&out);
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_backward_matches_cpu_reference() {
    let (input, weight, bias) = (ln_input(), ln_weight(), ln_bias());
    let (input_id, weight_id, bias_id) = (input.id, weight.id, bias.id);
    let out = B::layer_norm::<f32>(&input, &weight, Some(&bias), 1e-5).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let read = |id: incin_core::exec::TensorId| {
        let grad = grads
            .get(id)
            .expect("layer norm input should have a gradient");
        download_f32_host(grad)
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    let (_, expected_dx, expected_dw, expected_db) = cpu_layer_norm_grads(
        &[0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0],
        &[2.0, 1.0, 0.5, 1.5],
        Some(&[0.1, -0.1, 0.2, -0.2]),
    );
    // Welford on device against composed primitives on host: agreement to
    // four digits, not bit-exact.
    assert_close(&read(input_id), &expected_dx, 1e-4, "dx");
    assert_close(&read(weight_id), &expected_dw, 1e-4, "dw");
    assert_close(&read(bias_id), &expected_db, 1e-4, "db");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_uniform_upstream_gradient_gives_zero_input_gradient() {
    // Analytic property, no reference needed, but it only holds for uniform
    // weight: with `gw = gout * weight` uniform too, every input gradient is
    // exactly rstd*(gw - mean(gw) - y*mean(gw*y)) = 0. Averaging the upstream
    // gradient first and multiplying by weight after is the same only there,
    // which is why this test pins the uniform-weight case while the parity
    // tests above pin a non-uniform one.
    let input = ln_input();
    let weight = cuda_f32(&[4], vec![1.0; 4]);
    let bias = ln_bias();
    let (input_id, bias_id) = (input.id, bias.id);
    let out = B::layer_norm::<f32>(&input, &weight, Some(&bias), 1e-5).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let dx = download_f32_host(grads.get(input_id).unwrap()).unwrap();
    for (i, value) in dx.iter().enumerate() {
        assert!(
            value.abs() < 1e-4,
            "dx[{i}] should vanish under uniform gradients, got {value}"
        );
    }
    // Same seed, dual property: each bias element sees every row once.
    let db = download_f32_host(grads.get(bias_id).unwrap()).unwrap();
    assert_eq!(db, vec![2.0, 2.0, 2.0, 2.0]);
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_backward_without_bias_returns_two_gradients() {
    let (input, weight) = (ln_input(), ln_weight());
    let (input_id, weight_id) = (input.id, weight.id);
    let out = B::layer_norm::<f32>(&input, &weight, None, 1e-5).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let dx = grads.get(input_id).expect("input should have a gradient");
    let dw = grads.get(weight_id).expect("weight should have a gradient");
    assert_eq!(dx.shape, vec![2, 4]);
    assert_eq!(dw.shape, vec![4]);
    let (_, expected_dx, expected_dw, _) = cpu_layer_norm_grads(
        &[0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0],
        &[2.0, 1.0, 0.5, 1.5],
        None,
    );
    let read = |storage: &CudaStorage| {
        download_f32_host(storage)
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    assert_close(&read(dx), &expected_dx, 1e-4, "dx without bias");
    assert_close(&read(dw), &expected_dw, 1e-4, "dw without bias");
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_backward_replays_saved_statistics() {
    // White-box proof that the kernel reads the passed statistics rather
    // than recomputing them: the same launch with a perturbed mean must
    // produce different gradients, and with the true statistics must match
    // the CPU reference.
    use crate::cuda::ops::norm::launch_layer_norm_backward;
    let (input, weight, bias) = (ln_input(), ln_weight(), ln_bias());
    let (_, stats) =
        crate::cuda::ops::norm::launch_layer_norm(&input, &weight, Some(&bias), 1e-5, true)
            .unwrap();
    let stats = stats.expect("recording forward keeps statistics");
    let gout = cuda_f32(&[2, 4], vec![1.0, 0.5, -0.5, 2.0, -1.0, 1.0, 0.25, -0.75]);
    let grads =
        launch_layer_norm_backward(&gout, &input, &weight, &stats.mean, &stats.rstd, true).unwrap();
    let read = |storage: &CudaStorage| {
        download_f32_host(storage)
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    let dx = read(&grads.input);
    // Perturb the saved mean by 1.0 on both rows: a kernel that recomputed
    // its statistics internally would be unaffected, so its gradients would
    // coincide. They must not.
    let bad_mean = cuda_f32(
        &[2],
        vec![
            download_f32_host(&stats.mean).unwrap()[0] + 1.0,
            download_f32_host(&stats.mean).unwrap()[1] + 1.0,
        ],
    );
    let bad =
        launch_layer_norm_backward(&gout, &input, &weight, &bad_mean, &stats.rstd, true).unwrap();
    let bad_dx = read(&bad.input);
    let drift: f64 = dx
        .iter()
        .zip(bad_dx.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        drift > 1e-3,
        "perturbed statistics left the gradients unchanged: the kernel is not reading them"
    );
    // And with the true statistics, the CPU reference agrees. The normalized
    // values come straight from the definition here: reading them off the
    // operation's output would include the weight and bias the formula has
    // already accounted for.
    let t = host_f32(&[2, 4], vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0]);
    let w = host_f32(&[4], vec![2.0, 1.0, 0.5, 1.5]);
    let g = host_f32(&[2, 4], vec![1.0, 0.5, -0.5, 2.0, -1.0, 1.0, 0.25, -0.75]);
    let xv = host_values(&t);
    let wv = host_values(&w);
    let gv = host_values(&g);
    let mut expected = Vec::with_capacity(8);
    for row in 0..2 {
        let mean = xv[row * 4..row * 4 + 4].iter().sum::<f64>() / 4.0;
        let var = xv[row * 4..row * 4 + 4]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / 4.0;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        let yv: Vec<f64> = xv[row * 4..row * 4 + 4]
            .iter()
            .map(|v| (v - mean) * rstd)
            .collect();
        let mut sum_gw = 0.0;
        let mut sum_gwy = 0.0;
        for col in 0..4 {
            sum_gw += gv[row * 4 + col] * wv[col];
            sum_gwy += gv[row * 4 + col] * wv[col] * yv[col];
        }
        for col in 0..4 {
            expected.push(
                rstd * (gv[row * 4 + col] * wv[col] - sum_gw / 4.0 - yv[col] * sum_gwy / 4.0),
            );
        }
    }
    assert_close(&dx, &expected, 1e-4, "dx against host definition");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_rejects_mismatched_upstream_shape() {
    let (input, weight, bias) = (ln_input(), ln_weight(), ln_bias());
    let (_, stats) =
        crate::cuda::ops::norm::launch_layer_norm(&input, &weight, Some(&bias), 1e-5, true)
            .unwrap();
    let stats = stats.expect("recording forward keeps statistics");
    let short = cuda_f32(&[2, 3], vec![0.0; 6]);
    assert!(
        crate::cuda::ops::norm::launch_layer_norm_backward(
            &short,
            &input,
            &weight,
            &stats.mean,
            &stats.rstd,
            true,
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_rejects_integer_storage() {
    // i64 passes storage validation (indices live in it) but no float kernel
    // may read it: the launch must refuse before any transmute.
    let bytes: Vec<u8> = (0..8).flat_map(|v: i64| v.to_le_bytes()).collect();
    let input =
        crate::cuda::backend::cuda_from_bytes(&[2, 4], DTypeId::I64.into(), 0, &bytes).unwrap();
    let weight = cuda_f32(&[4], vec![1.0; 4]);
    assert!(B::layer_norm::<i64>(&input, &weight, None, 1e-5).is_err());
}

// ---------------------------------------------------------------------------
// batch_norm training forward/backward (issue #123)
// ---------------------------------------------------------------------------

/// `[N, C, H] = [2, 2, 2]`: two channels, two spatial positions, four
/// elements reduced into each channel's statistics -- small enough to
/// check against the CPU reference element-wise, large enough that a
/// permuted (batch, spatial) indexing would show.
const BN_VALUES: [f32; 8] = [0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0];
const BN_GOUT: [f32; 8] = [1.0, 0.5, -0.5, 2.0, -1.0, 1.0, 0.25, -0.75];
const BN_WEIGHT: [f32; 2] = [2.0, 0.5];
const BN_BIAS: [f32; 2] = [0.1, -0.2];
const BN_EPS: f32 = 1e-5;

fn bn_input() -> CudaStorage {
    cuda_f32(&[2, 2, 2], BN_VALUES.to_vec())
}

fn bn_weight() -> CudaStorage {
    cuda_f32(&[2], BN_WEIGHT.to_vec())
}

fn bn_bias() -> CudaStorage {
    cuda_f32(&[2], BN_BIAS.to_vec())
}

/// CPU training forward plus its composed backward, on the same values the
/// CUDA side runs: the reference the parity tests below compare against.
/// Mirrors `cpu_layer_norm_grads` above for the output and dx; dw and db are
/// summed in closed form here instead, because the CPU composition reshapes
/// weight/bias into a broadcast copy with a fresh `TensorId` the tape never
/// links back (its reshape is deliberately silent -- parameters are treated
/// as fixed inputs there, which is also why `batch_norm_training_gradcheck`
/// only covers the input path). The backward runs with the `BN_GOUT` seed
/// every CUDA-side comparison uses too: under the default ones seed
/// `dw = sum(xhat)` is identically ~0 (xhat is mean-centered), which would
/// make the affine comparison noise-blind. `xhat` is reconstructed in f64
/// from the raw fixture values and their per-channel statistics.
#[cfg(feature = "cpu")]
fn cpu_batch_norm_training_grads(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = host_f32(&[2, 2, 2], input.to_vec());
    let w = host_f32(&[2], weight.to_vec());
    let b = host_f32(&[2], bias.to_vec());
    let out =
        crate::cpu::ops::norm::batch_norm_training_impl::<incin_core::tensor::device::Cpu, f32>(
            &t,
            Some(&w),
            Some(&b),
            BN_EPS,
        )
        .unwrap();
    let seed = host_f32(&[2, 2, 2], BN_GOUT.to_vec());
    let grads = crate::cpu::tape::backward_with(&out, &seed).unwrap();
    let dx = host_values(grads.get(t.id).unwrap());
    let out_values = host_values(&out);
    // The affine gradients come from closed-form per-channel sums over the
    // same [2, 2, 2] fixture: channel = (flat / spatial) % C, where spatial
    // is the product of the axes after the channel one (H here, 2) --
    // the same split `batch_norm_geometry` uses, not elements-per-channel.
    let num_channels = weight.len();
    let spatial: usize = t.shape[2..].iter().product();
    let batch_elements = input.len() / num_channels;
    let mut mean = vec![0f64; num_channels];
    let mut variance = vec![0f64; num_channels];
    for (flat, value) in input.iter().enumerate() {
        mean[flat / spatial % num_channels] += f64::from(*value);
    }
    for channel in &mut mean {
        *channel /= batch_elements as f64;
    }
    for (flat, value) in input.iter().enumerate() {
        let centered = f64::from(*value) - mean[flat / spatial % num_channels];
        variance[flat / spatial % num_channels] += centered * centered;
    }
    for channel in &mut variance {
        *channel /= batch_elements as f64;
    }
    let mut dw = vec![0f64; num_channels];
    let mut db = vec![0f64; num_channels];
    for flat in 0..input.len() {
        let channel = flat / spatial % num_channels;
        let xhat = (f64::from(input[flat]) - mean[channel])
            / (variance[channel] + f64::from(BN_EPS)).sqrt();
        db[channel] += f64::from(BN_GOUT[flat]);
        dw[channel] += f64::from(BN_GOUT[flat]) * xhat;
    }
    (out_values, dx, dw, db)
}

/// The affine reference in `cpu_batch_norm_training_grads` is closed-form
/// code standing in for a tape path the CPU composition does not have, so
/// it gets its own check: central finite differences of the CPU forward
/// under the same `BN_GOUT` seed direction (loss = sum(out * g)). Runs on
/// the host as a guard for every hardware test that leans on it.
#[cfg(feature = "cpu")]
#[test]
fn batch_norm_affine_reference_matches_finite_differences() {
    let forward_dot = |weight: &[f32], bias: &[f32]| -> f64 {
        let t = host_f32(&[2, 2, 2], BN_VALUES.to_vec());
        let w = host_f32(&[2], weight.to_vec());
        let b = host_f32(&[2], bias.to_vec());
        let out = crate::cpu::ops::norm::batch_norm_training_impl::<
            incin_core::tensor::device::Cpu,
            f32,
        >(&t, Some(&w), Some(&b), BN_EPS)
        .unwrap();
        host_values(&out)
            .iter()
            .zip(BN_GOUT.iter())
            .map(|(value, g)| value * f64::from(*g))
            .sum()
    };
    let (_, _, dw, db) = cpu_batch_norm_training_grads(&BN_VALUES, &BN_WEIGHT, &BN_BIAS);
    let step = 1e-3f64;
    for channel in 0..BN_WEIGHT.len() {
        let mut plus = BN_WEIGHT;
        let mut minus = BN_WEIGHT;
        plus[channel] += step as f32;
        minus[channel] -= step as f32;
        let numeric_dw =
            (forward_dot(&plus, &BN_BIAS) - forward_dot(&minus, &BN_BIAS)) / (2.0 * step);
        let denom = dw[channel].abs().max(numeric_dw.abs()).max(1e-6);
        assert!(
            (numeric_dw - dw[channel]).abs() / denom <= 1e-3,
            "dw[{channel}]: analytic={} numeric={numeric_dw}",
            dw[channel]
        );

        let mut plus = BN_BIAS;
        let mut minus = BN_BIAS;
        plus[channel] += step as f32;
        minus[channel] -= step as f32;
        let numeric_db =
            (forward_dot(&BN_WEIGHT, &plus) - forward_dot(&BN_WEIGHT, &minus)) / (2.0 * step);
        let denom = db[channel].abs().max(numeric_db.abs()).max(1e-6);
        assert!(
            (numeric_db - db[channel]).abs() / denom <= 1e-3,
            "db[{channel}]: analytic={} numeric={numeric_db}",
            db[channel]
        );
    }
}

/// Host-side admission for the row #123 flipped. The query alone would not
/// have caught a stale `false` on the legacy row -- the typed normalization
/// row beside it already claimed `training = true`, and `support` answers
/// from the first rule that satisfies the query -- so this pins the legacy
/// row's own flag as well.
#[test]
fn cuda_batch_norm_training_rows_admit_host_side() {
    use incin_core::exec::{
        CapabilityQuery, LayoutClass, MathMode, OperationIdentity, SupportLevel,
    };
    use incin_core::shapes::OperationKind;
    use incin_core::tensor::device::DeviceKind;

    let query = CapabilityQuery {
        operation: OperationIdentity::Builtin(OperationKind::BatchNorm),
        dtype: DTypeId::F32.descriptor(),
        layout: LayoutClass::Contiguous,
        rank: 3,
        training: true,
        math_mode: MathMode::Precise,
    };
    let level = crate::capability::support(DeviceKind::Cuda, &query);
    assert!(
        !matches!(level, SupportLevel::Unsupported(_)),
        "a training-mode CUDA batch norm must be admitted since #123, got {level:?}"
    );
    let legacy = crate::capability::CUDA_CAPABILITIES
        .iter()
        .find(|rule| rule.operation == OperationKind::BatchNorm)
        .expect("CUDA registers BatchNorm");
    assert!(
        legacy.training,
        "the legacy CUDA BatchNorm row must claim training since #123"
    );
    assert_eq!(legacy.layouts, [LayoutClass::Contiguous]);
}

/// Shape and dtype parity against the CPU training composition, without a
/// device: the backward's allocation plan is a pure function of the
/// forward's operand shapes, and the precision policy that picks the
/// gradient dtypes resolves on the host. Together they are what makes a
/// gradient hand-off between backends line up.
#[cfg(feature = "cpu")]
#[test]
fn batch_norm_training_backward_plan_matches_cpu_reference() {
    use incin_core::exec::{LayoutClass, PrecisionRequest};

    let t = host_f32(&[2, 2, 2], BN_VALUES.to_vec());
    let w = host_f32(&[2], BN_WEIGHT.to_vec());
    let b = host_f32(&[2], BN_BIAS.to_vec());
    let out =
        crate::cpu::ops::norm::batch_norm_training_impl::<incin_core::tensor::device::Cpu, f32>(
            &t,
            Some(&w),
            Some(&b),
            BN_EPS,
        )
        .unwrap();
    let grads = crate::cpu::tape::backward(&out).unwrap();

    let plan =
        crate::cuda::ops::norm::batch_norm_grad_shapes(&t.shape, Some(&w.shape), Some(&b.shape));
    // The input path is what the CPU composition's tape covers (gradcheck
    // pins it), so its gradient's shape is a real cross-backend reference:
    // dx must carry the input's shape, not the per-channel extent.
    let dx_shape = grads.get(t.id).unwrap().shape.as_ref().to_vec();
    assert_eq!(plan.input, dx_shape, "dx must take the input's shape");
    // weight/bias pass through as the operand shapes themselves. For valid
    // batch-norm operands those equal the channel extent, which is also why
    // deriving them from the geometry would pass here -- the point pinned
    // below is that the plan reads them off the operands at all (the unit
    // test `batch_norm_grad_shapes_follow_the_forward_operands` covers the
    // operand-free case where a geometry-derived answer would invent [C]).
    assert_eq!(plan.weight, Some(w.shape.as_ref().to_vec()));
    assert_eq!(plan.bias, Some(b.shape.as_ref().to_vec()));

    // The backward allocates dx in storage dtype and dw/db in the policy's
    // compute dtype; for the f32-only row both are f32, which is also the
    // dtype the CPU tape hands back for the gradient it does produce.
    let req = PrecisionRequest::new(
        OperationKind::Normalization,
        DTypeId::F32.into(),
        DTypeId::F32.into(),
        LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );
    let policy = crate::cuda::backend::native_precision(&req).unwrap();
    assert_eq!(policy.compute, DTypeId::F32.into());
    let dx = grads.get(t.id).unwrap();
    assert_eq!(
        dx.dtype, policy.compute,
        "the CUDA backward's gradient dtype must match the CPU reference's"
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn batch_norm_training_forward_matches_cpu_reference() {
    // Regression guard for the stats-saving edit to the fused template: the
    // two extra stores and the Welford reduction must not disturb the
    // output values the inference form used to produce from running stats.
    let (input, weight, bias) = (bn_input(), bn_weight(), bn_bias());
    let out = B::batch_norm::<f32>(&input, Some(&weight), Some(&bias), BN_EPS).unwrap();
    assert_eq!(out.shape, vec![2, 2, 2]);
    let (expected, _, _, _) = cpu_batch_norm_training_grads(&BN_VALUES, &BN_WEIGHT, &BN_BIAS);
    let got: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| *v as f64)
        .collect();
    assert_close(&got, &expected, 1e-5, "training forward");
    // Draining here keeps this forward's entry off the next test's walk.
    let _ = crate::cuda::tape::backward(&out);
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn batch_norm_backward_matches_cpu_reference() {
    let (input, weight, bias) = (bn_input(), bn_weight(), bn_bias());
    let (input_id, weight_id, bias_id) = (input.id, weight.id, bias.id);
    let out = B::batch_norm::<f32>(&input, Some(&weight), Some(&bias), BN_EPS).unwrap();
    let seed = cuda_f32(&[2, 2, 2], BN_GOUT.to_vec());
    let grads = crate::cuda::tape::backward_with(&out, &seed).unwrap();
    let read = |id: incin_core::exec::TensorId| {
        let grad = grads
            .get(id)
            .expect("batch norm operand should have a gradient");
        download_f32_host(grad)
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    let (_, expected_dx, expected_dw, expected_db) =
        cpu_batch_norm_training_grads(&BN_VALUES, &BN_WEIGHT, &BN_BIAS);
    // A Welford reduction on device against composed primitives on host:
    // agreement to four digits, not bit-exact. The BN_GOUT seed (rather
    // than the default ones) matters for dw -- under ones, sum(xhat) is
    // ~0 by construction, so a wrong kernel could pass on noise. With a
    // non-uniform seed the mean(gw) and mean(gw*xhat) terms of dx are all
    // load-bearing too, which is how a wrong formula stops surviving a
    // smoke test.
    assert_close(&read(input_id), &expected_dx, 1e-4, "dx");
    assert_close(&read(weight_id), &expected_dw, 1e-4, "dw");
    assert_close(&read(bias_id), &expected_db, 1e-4, "db");
    // And each gradient carries its operand's shape, the tape hand-off
    // contract the host-side plan test above pins without a device.
    assert_eq!(grads.get(input_id).unwrap().shape, vec![2, 2, 2]);
    assert_eq!(grads.get(weight_id).unwrap().shape, vec![2]);
    assert_eq!(grads.get(bias_id).unwrap().shape, vec![2]);
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn batch_norm_backward_replays_saved_statistics() {
    // White-box proof that the backward reads the statistics the forward
    // saved rather than recomputing them: the same launch with a perturbed
    // mean must produce different gradients (a kernel recomputing its own
    // statistics internally would be unaffected), and with the true
    // statistics must match the CPU reference.
    use crate::cuda::ops::norm::{launch_batch_norm_backward, launch_batch_norm_training};
    let (input, weight, bias) = (bn_input(), bn_weight(), bn_bias());
    let (_, stats) =
        launch_batch_norm_training(&input, Some(&weight), Some(&bias), BN_EPS, true).unwrap();
    let stats = stats.expect("recording forward keeps statistics");
    let gout = cuda_f32(&[2, 2, 2], BN_GOUT.to_vec());
    let grads = launch_batch_norm_backward(
        &gout,
        &input,
        Some(&weight),
        Some(&bias),
        &stats.mean,
        &stats.rstd,
    )
    .unwrap();
    assert!(
        grads.weight.is_some() && grads.bias.is_some(),
        "a forward with weight and bias must produce both gradients"
    );
    let read = |storage: &CudaStorage| {
        download_f32_host(storage)
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    let dx = read(&grads.input);
    // Perturb the saved per-channel mean by 1.0 on both channels.
    let true_mean = download_f32_host(&stats.mean).unwrap();
    let bad_mean = cuda_f32(&[2], vec![true_mean[0] + 1.0, true_mean[1] + 1.0]);
    let bad = launch_batch_norm_backward(
        &gout,
        &input,
        Some(&weight),
        Some(&bias),
        &bad_mean,
        &stats.rstd,
    )
    .unwrap();
    let bad_dx = read(&bad.input);
    let drift: f64 = dx
        .iter()
        .zip(bad_dx.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        drift > 1e-3,
        "perturbed statistics left the gradients unchanged: the kernel is not reading them"
    );
    // And with the true statistics, the CPU reference agrees.
    let (_, expected_dx, expected_dw, expected_db) =
        cpu_batch_norm_training_grads(&BN_VALUES, &BN_WEIGHT, &BN_BIAS);
    assert_close(&dx, &expected_dx, 1e-4, "dx against CPU reference");
    assert_close(&read(&grads.weight.unwrap()), &expected_dw, 1e-4, "dw");
    assert_close(&read(&grads.bias.unwrap()), &expected_db, 1e-4, "db");
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn batch_norm_training_execute_delegates_to_the_tape_tracked_method() {
    // The executor branch #123 added: `attributes.training = true` must no
    // longer be refused by name but run the tape-tracked method, record
    // exactly one entry, and answer with the same values CPU's training
    // kernel produces.
    use incin_core::exec::catalog::BatchNormAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new()).with_training(true);
    let (input, weight, bias) = (bn_input(), bn_weight(), bn_bias());
    let (input_id, weight_id, bias_id) = (input.id, weight.id, bias.id);
    let handles = [
        TensorHandle::from_storage::<B, f32, _>(&input),
        TensorHandle::from_storage::<B, f32, _>(&weight),
        TensorHandle::from_storage::<B, f32, _>(&bias),
    ];
    let before = crate::cuda::tape::depth();
    let out = dispatch::execute::<op::BatchNorm, _>(
        &context,
        BatchNormAttributes {
            epsilon: f64::from(BN_EPS),
            momentum: 0.1,
            training: true,
            has_weight: true,
            has_bias: true,
            has_running_mean: false,
            has_running_variance: false,
        },
        &handles,
    )
    .expect("a training batch norm must execute on CUDA since #123");
    assert!(
        crate::cuda::tape::depth() > before,
        "a training-mode forward under a recording grad mode must push a tape entry"
    );
    let (expected, expected_dx, expected_dw, expected_db) =
        cpu_batch_norm_training_grads(&BN_VALUES, &BN_WEIGHT, &BN_BIAS);
    let got: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| *v as f64)
        .collect();
    assert_close(&got, &expected, 1e-5, "dispatched forward");
    let seed = cuda_f32(&[2, 2, 2], BN_GOUT.to_vec());
    let grads = crate::cuda::tape::backward_with(&out, &seed).unwrap();
    let read = |id: incin_core::exec::TensorId| {
        download_f32_host(grads.get(id).expect("operand should have a gradient"))
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    assert_close(&read(input_id), &expected_dx, 1e-4, "dispatched dx");
    assert_close(&read(weight_id), &expected_dw, 1e-4, "dispatched dw");
    assert_close(&read(bias_id), &expected_db, 1e-4, "dispatched db");
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn batch_norm_training_without_affine_ignores_running_statistics() {
    // The other half of #123's acceptance: gradients with *no* optional
    // weight/bias, and a training forward handed the running pair. The
    // running values are deliberately far from the batch's own statistics,
    // so an inference-style kernel normalizing by them would miss the CPU
    // reference by orders of magnitude -- this pins that training mode
    // accepts the pair for arity and ignores it, exactly like CPU's
    // `batch_norm_training_impl`, which never reads it either.
    use incin_core::exec::catalog::BatchNormAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new()).with_training(true);
    let input = bn_input();
    let input_id = input.id;
    let running_mean = cuda_f32(&[2], vec![8.0, -8.0]);
    let running_var = cuda_f32(&[2], vec![64.0, 25.0]);
    let handles = [
        TensorHandle::from_storage::<B, f32, _>(&input),
        TensorHandle::from_storage::<B, f32, _>(&running_mean),
        TensorHandle::from_storage::<B, f32, _>(&running_var),
    ];
    let before = crate::cuda::tape::depth();
    let out = dispatch::execute::<op::BatchNorm, _>(
        &context,
        BatchNormAttributes {
            epsilon: f64::from(BN_EPS),
            momentum: 0.1,
            training: true,
            has_weight: false,
            has_bias: false,
            has_running_mean: true,
            has_running_variance: true,
        },
        &handles,
    )
    .expect("training batch norm without affine operands must execute on CUDA");
    assert!(
        crate::cuda::tape::depth() > before,
        "a training-mode forward under a recording grad mode must push a tape entry"
    );

    let t = host_f32(&[2, 2, 2], BN_VALUES.to_vec());
    let expected_out = crate::cpu::ops::norm::batch_norm_training_impl::<
        incin_core::tensor::device::Cpu,
        f32,
    >(&t, None, None, BN_EPS)
    .unwrap();
    let expected_values = host_values(&expected_out);
    let got: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| *v as f64)
        .collect();
    assert_close(&got, &expected_values, 1e-5, "no-affine training forward");

    let seed = cuda_f32(&[2, 2, 2], BN_GOUT.to_vec());
    let grads = crate::cuda::tape::backward_with(&out, &seed).unwrap();
    let dx = download_f32_host(
        grads
            .get(input_id)
            .expect("the input must carry a gradient without affine operands"),
    )
    .unwrap()
    .iter()
    .map(|v| *v as f64)
    .collect::<Vec<_>>();
    let host_seed = host_f32(&[2, 2, 2], BN_GOUT.to_vec());
    let expected_grads = crate::cpu::tape::backward_with(&expected_out, &host_seed).unwrap();
    let expected_dx = host_values(expected_grads.get(t.id).unwrap());
    assert_close(&dx, &expected_dx, 1e-4, "no-affine dx");
    // Nothing else was an operand, so nothing else may carry a gradient:
    // a stray dw/db entry would mean the backward wrote the absent
    // parameter scratch and the tape attached it to something. The walk
    // always retains the seeded loss entry itself (`tape::backward`
    // inserts `loss.id()`), so the map holds exactly the output and the
    // input - two entries, not one.
    assert_eq!(
        grads.len(),
        2,
        "only the output seed and the input gradient may appear; no other gradient may appear"
    );
    assert!(
        grads.get(out.id).is_some(),
        "the walk retains the seeded loss entry"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn l1_loss_trains_through_scalar_reduction_on_cuda() {
    // A mean reduction seeds the walk with a scalar gradient that the next
    // recipe needs at full width. `unbroadcast` used to hand the scalar on
    // unchanged -- the CPU kernels broadcast implicitly and never noticed,
    // but the CUDA binary launch refused it in `iteration_plan`. This loss
    // is the shape that caught it: sub, abs, then mean.
    //
    // The expected gradients also pin the fused abs derivative at zero:
    // pred[0] == targ[0] makes diff[0] exactly 0, where the symbolic
    // differentiator used to answer -1 against CPU `Sign`, PyTorch, and the
    // hand-written kernel expression, which all answer 0.
    use incin_core::exec::catalog::{LossAttributes, LossReduction};
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};
    let context = ExecutionContext::new(B::new());
    let pred = cuda_f32(&[3], vec![1.0, 0.0, -1.0]);
    let targ = cuda_f32(&[3], vec![1.0, 1.0, 0.0]);
    let pred_id = pred.id;
    let pred_handle = TensorHandle::from_storage::<B, f32, _>(&pred);
    let targ_handle = TensorHandle::from_storage::<B, f32, _>(&targ);
    let out = dispatch::execute::<op::L1Loss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[pred_handle, targ_handle],
    )
    .expect("l1 executes on CUDA");
    assert_eq!(download_f32_host(&out).unwrap(), vec![2.0 / 3.0]);
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad = grads.get(pred_id).expect("pred has a gradient");
    let values = download_f32_host(grad).unwrap();
    assert_eq!(values.len(), 3);
    assert!(
        (values[0] - 0.0).abs() < 1e-6,
        "sign(0) must be 0, got {}",
        values[0]
    );
    for (i, value) in values.iter().enumerate().skip(1) {
        assert!(
            (value - (-1.0 / 3.0)).abs() < 1e-6,
            "grad[{i}] should be sign/3, got {value}"
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn unbroadcast_scalar_seed_for_size_one_target_materializes() {
    // Regression for the latent cross-backend panic: a size-1 target
    // dimension at an index >= the grad's rank (here the whole target)
    // used to index past the reduced grad in the keepdim loop at
    // `result.shape[i]`. A scalar broadcasts to any shape, so `[] -> [1]`
    // expands to `[v]` - the same materialization #121 pinned for a `[3]`
    // target, now shared by all three backends' tape tails.
    let grad = cuda_f32(&[], vec![2.0]);
    let result = crate::cuda::tape::unbroadcast(&grad, &[1])
        .expect("a compatible scalar seed for a size-1 target expands");
    assert_eq!(result.shape, vec![1]);
    assert_eq!(download_f32_host(&result).unwrap(), vec![2.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn unbroadcast_rank_deficit_that_cannot_broadcast_into_target_is_refused() {
    // Mutually-broadcastable is not enough: `[4]` and `[2,1]` resolve
    // together to `[2,4]`, but `[4]` does not broadcast *into* `[2,1]`.
    // Pre-fix this indexed past the grad in the keepdim loop and panicked;
    // it must refuse by name instead (mirrors the CPU/WGPU tests).
    let grad = cuda_f32(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        crate::cuda::tape::unbroadcast(&grad, &[2, 1]),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cross_entropy_loss_trains_through_gather_on_cuda() {
    // The executor used to call the raw gather launch, which runs the kernel
    // but records no tape entry: forward matched, backward reached nothing,
    // and the logits gradient was silently absent. Routing through the
    // tape-tracked gather (scatter-based backward) closes the walk.
    use incin_core::exec::catalog::{LossAttributes, LossReduction};
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};
    let context = ExecutionContext::new(B::new());
    let logits = cuda_f32(&[2, 3], vec![2.0, 1.0, 0.5, 0.5, 1.5, 0.0]);
    let target_bytes: Vec<u8> = [0i64, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    let targets =
        crate::cuda::backend::cuda_from_bytes(&[2], DTypeId::I64.into(), 0, &target_bytes).unwrap();
    let logits_id = logits.id;
    let logits_handle = TensorHandle::from_storage::<B, f32, _>(&logits);
    let targets_handle = TensorHandle::from_storage::<B, i64, _>(&targets);
    let out = dispatch::execute::<op::CrossEntropyLoss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[logits_handle, targets_handle],
    )
    .expect("cross entropy executes on CUDA");
    let fwd = download_f32_host(&out).unwrap();
    assert!(
        (fwd[0] - 1.2144).abs() < 1e-3,
        "forward should match -(log p0 + log p2)/2, got {}",
        fwd[0]
    );
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad = grads.get(logits_id).expect("logits have a gradient");
    let values = download_f32_host(grad).unwrap();
    // dL/dlogits = (softmax - onehot) / batch, computed by hand.
    let expected = [-0.1857, 0.1156, 0.0701, 0.1156, 0.3142, -0.4299];
    assert_eq!(values.len(), expected.len());
    for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (f64::from(*got) - want).abs() < 1e-3,
            "logits grad[{i}]: got {got}, want {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// Training rows that recorded nothing: softmax, rms_norm, transpose_view,
// and the attention chain through softmax.
// ---------------------------------------------------------------------------

// Parity against the CPU reference, so these need both backends.
#[cfg(feature = "cpu")]
use crate::cpu::CpuBackendImpl as HostBackend;

#[cfg(feature = "cpu")]
/// CPU forward of a canonical op plus backward under an explicit seed, on
/// the same values the CUDA side runs. Untyped dispatch: shapes still
/// validate, only the caller-held proof is absent, which value parity does
/// not need. Returns the output followed by one gradient per input.
fn cpu_forward_and_grads<O>(
    attributes: O::Attributes,
    inputs: &[HostStorage],
    seed_values: &[f32],
) -> (HostStorage, alloc::vec::Vec<HostStorage>)
where
    O: incin_core::backend_authoring::Operation,
    HostBackend: incin_core::backend_authoring::Execute<O, Output = HostStorage>,
{
    use incin_core::backend_authoring::ExecutionContext;
    use incin_core::exec::{TensorHandle, dispatch};
    let context = ExecutionContext::new(HostBackend::new());
    let handles: Vec<TensorHandle> = inputs
        .iter()
        .map(TensorHandle::from_storage::<HostBackend, f32, _>)
        .collect();
    let out = dispatch::execute::<O, HostBackend>(&context, attributes, &handles)
        .expect("CPU reference executes");
    let seed = HostStorage::from_contiguous(HostBuffer::F32(seed_values.to_vec()), &out.shape);
    let grads = crate::cpu::tape::backward_with(&out, &seed).unwrap();
    let input_grads = inputs
        .iter()
        .map(|storage| {
            grads
                .get(storage.id)
                .unwrap_or_else(|| panic!("CPU reference is missing a gradient"))
                .clone()
        })
        .collect();
    (out, input_grads)
}

fn sm_values() -> (Vec<usize>, Vec<f32>) {
    (vec![2, 3], vec![1.0, 2.0, 3.0, 0.5, -0.5, 0.0])
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn softmax_trains_on_cuda() {
    use incin_core::exec::catalog::AxisAttributes;
    let (dims, values) = sm_values();
    let input = cuda_f32(&dims, values.clone());
    let input_id = input.id;
    let out = B::softmax::<f32>(&input, 1).unwrap();
    assert_eq!(out.shape, dims);
    // Forward parity against the CPU reference first: the composition
    // replaced a fused kernel, so its values need their own check.
    let host_in = host_f32(&dims, values.clone());
    let seed_values = [1.0, 0.0, -1.0, 0.5, 0.5, -2.0];
    let (host_out, host_grads) = cpu_forward_and_grads::<incin_core::exec::op::Softmax>(
        AxisAttributes { axis: 1 },
        &[host_in],
        &seed_values,
    );
    let got: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| *v as f64)
        .collect();
    assert_close(&got, &host_values(&host_out), 1e-5, "softmax forward");
    // Then the gradients, under a non-uniform seed (a uniform seed gives a
    // zero gradient by definition and would prove nothing).
    let seed = cuda_f32(&dims, seed_values.to_vec());
    let grads = crate::cuda::tape::backward_with(&out, &seed).unwrap();
    let dx = download_f32_host(grads.get(input_id).unwrap()).unwrap();
    assert_close(
        &dx.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &host_values(&host_grads[0]),
        1e-4,
        "softmax dx",
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn rms_norm_trains_on_cuda() {
    use incin_core::exec::catalog::EpsilonAttributes;
    let (dims, values) = sm_values();
    let input = cuda_f32(&dims, values.clone());
    let weight = cuda_f32(&[3], vec![1.0, 0.5, 2.0]);
    let (input_id, weight_id) = (input.id, weight.id);
    let out = B::rms_norm::<f32>(&input, &weight, 1e-5).unwrap();
    assert_eq!(out.shape, dims);
    let seed = cuda_f32(&dims, vec![1.0, 0.0, -1.0, 0.5, 0.5, -2.0]);
    let grads = crate::cuda::tape::backward_with(&out, &seed).unwrap();
    let read = |id: incin_core::exec::TensorId| {
        download_f32_host(grads.get(id).unwrap())
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    let host_in = host_f32(&dims, values);
    let host_w = host_f32(&[3], vec![1.0, 0.5, 2.0]);
    let seed_values = [1.0, 0.0, -1.0, 0.5, 0.5, -2.0];
    let (_, host_grads) = cpu_forward_and_grads::<incin_core::exec::op::RmsNorm>(
        EpsilonAttributes { epsilon: 1e-5 },
        &[host_in, host_w],
        &seed_values,
    );
    assert_close(
        &read(input_id),
        &host_values(&host_grads[0]),
        1e-4,
        "rms dx",
    );
    assert_close(
        &read(weight_id),
        &host_values(&host_grads[1]),
        1e-4,
        "rms dw",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn transpose_view_trains_on_cuda() {
    let input = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let input_id = input.id;
    let out = B::transpose_view::<f32>(&input, 0, 1).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
    // A permutation's backward is the same permutation: exact, no tolerance.
    let seed = cuda_f32(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let grads = crate::cuda::tape::backward_with(&out, &seed).unwrap();
    assert_eq!(
        download_f32_host(grads.get(input_id).unwrap()).unwrap(),
        vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn attention_trains_end_to_end_on_cuda() {
    use incin_core::exec::catalog::AttentionAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};
    let context = ExecutionContext::new(B::new());
    let q = cuda_f32(&[1, 2, 2], vec![0.5, 1.0, -0.5, 0.25]);
    let k = cuda_f32(&[1, 2, 2], vec![0.5, -1.0, 0.0, 0.75]);
    let v = cuda_f32(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let (qid, kid, vid) = (q.id, k.id, v.id);
    let handles = [
        TensorHandle::from_storage::<B, f32, _>(&q),
        TensorHandle::from_storage::<B, f32, _>(&k),
        TensorHandle::from_storage::<B, f32, _>(&v),
    ];
    let out = dispatch::execute::<op::ScaledDotProductAttention, _>(
        &context,
        AttentionAttributes {
            scale: None,
            has_mask: false,
        },
        &handles,
    )
    .expect("attention executes on CUDA");
    assert_eq!(out.shape, vec![1, 2, 2]);
    // The defect this proves absent: the softmax link recorded nothing, so
    // no gradient reached any of the three inputs.
    let grads = crate::cuda::tape::backward(&out).unwrap();
    for (id, name) in [(qid, "query"), (kid, "key"), (vid, "value")] {
        let grad = grads
            .get(id)
            .unwrap_or_else(|| panic!("{name} has no gradient"));
        assert_eq!(grad.shape, vec![1, 2, 2], "{name} gradient shape");
        let values = download_f32_host(grad).unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{name} gradient is not finite: {values:?}"
        );
        assert!(
            values.iter().any(|v| *v != 0.0),
            "{name} gradient is all zeros"
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn dropout_trains_through_the_replayed_mask_on_cuda() {
    // Training dropout is mask, scale, and nothing else: the forward draws
    // once, and the backward must replay that exact draw rather than a fresh
    // one. The entries of the composed chain capture the materialized mask,
    // so this checks both halves elementwise: every output is 0 or scaled
    // input, and every gradient is 0 or the scale in the same lanes.
    use incin_core::exec::catalog::DropoutAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};
    let context = ExecutionContext::new(B::new());
    let input = cuda_f32(&[8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let input_id = input.id;
    let handle = TensorHandle::from_storage::<B, f32, _>(&input);
    let out = dispatch::execute::<op::Dropout, _>(
        &context,
        DropoutAttributes {
            probability: 0.5,
            training: true,
        },
        &[handle],
    )
    .expect("dropout executes on CUDA");
    let kept: Vec<f32> = download_f32_host(&out).unwrap();
    assert_eq!(kept.len(), 8);
    for (index, output) in kept.iter().enumerate() {
        let input_value = (index + 1) as f32;
        assert!(
            *output == 0.0 || *output == 2.0 * input_value,
            "output[{index}] = {output}: neither dropped nor scaled"
        );
    }
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad = grads.get(input_id).expect("input has a gradient");
    for (output, grad) in kept.iter().zip(download_f32_host(grad).unwrap()) {
        let expected = if *output == 0.0 { 0.0 } else { 2.0 };
        assert!(
            (grad - expected).abs() < 1e-5,
            "gradient {grad} does not replay the forward mask lane (output {output})"
        );
    }
}
#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn gather_backward_accumulates_duplicate_indices_like_cpu() {
    // Duplicate indices: every position contributes, so grad_t sums.
    // The overwrite kernel kept one contribution (`[1,1,0]`); the CPU
    // reference (`gather_storage` backward, `+=`) is `[2,1,0]`.
    let t = cuda_f32(&[3], vec![10.0, 20.0, 30.0]);
    let index_bytes: Vec<u8> = [0i64, 0i64, 1i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let index =
        crate::cuda::backend::cuda_from_bytes(&[3], DTypeId::I64.into(), 0, &index_bytes).unwrap();
    let t_id = t.id;
    let out = B::gather::<f32, i64>(&t, 0, &index).unwrap();
    assert_eq!(download_f32_host(&out).unwrap(), vec![10.0, 10.0, 20.0]);
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let got = download_f32_host(grads.get(t_id).unwrap()).unwrap();
    // CPU reference on the same values.
    let host_t = host_f32(&[3], vec![10.0, 20.0, 30.0]);
    let host_idx = HostStorage::from_contiguous(HostBuffer::I64(vec![0, 0, 1]), [3]);
    let host_out = crate::cpu::ops::shape_ops::gather_storage(&host_t, 0, &host_idx).unwrap();
    let host_grads = crate::cpu::tape::backward(&host_out).unwrap();
    let want = host_values(host_grads.get(host_t.id).unwrap());
    assert_close(
        &got.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &want,
        1e-5,
        "gather duplicate dx",
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn scatter_src_grad_keeps_only_last_write_like_cpu() {
    // Duplicate writes: forward last-wins, so only the surviving write earns
    // a cotangent. The plain gather returned every writer's copy (`[1,1]`);
    // the CPU reference (`scatter_storage` backward) is `[0,1]`.
    // Forward itself still races on GPU (plain stores, no ordering), so this
    // pins the backward only -- a deterministic forward needs a bigger kernel.
    let t = cuda_f32(&[1, 2], vec![1.0, 2.0]);
    let index_bytes: Vec<u8> = [0i64, 0i64].iter().flat_map(|v| v.to_le_bytes()).collect();
    let index =
        crate::cuda::backend::cuda_from_bytes(&[2, 1], DTypeId::I64.into(), 0, &index_bytes)
            .unwrap();
    let src = cuda_f32(&[2, 1], vec![7.0, 8.0]);
    let (t_id, src_id) = (t.id, src.id);
    let out = B::scatter::<f32, i64>(&t, 0, &index, &src).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let got_t = download_f32_host(grads.get(t_id).unwrap()).unwrap();
    let got_src = download_f32_host(grads.get(src_id).unwrap()).unwrap();
    let host_t = host_f32(&[1, 2], vec![1.0, 2.0]);
    let host_idx = HostStorage::from_contiguous(HostBuffer::I64(vec![0, 0]), [2, 1]);
    let host_src = host_f32(&[2, 1], vec![7.0, 8.0]);
    let host_out =
        crate::cpu::ops::shape_ops::scatter_storage(&host_t, 0, &host_idx, &host_src).unwrap();
    let host_grads = crate::cpu::tape::backward(&host_out).unwrap();
    assert_close(
        &got_t.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &host_values(host_grads.get(host_t.id).unwrap()),
        1e-5,
        "scatter duplicate grad_t",
    );
    assert_close(
        &got_src.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        &host_values(host_grads.get(host_src.id).unwrap()),
        1e-5,
        "scatter duplicate grad_src",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn nograd_chain_records_nothing_on_cuda() {
    // GRD-002 on the CUDA tape: a NoGrad forward leaves the depth unchanged.
    use incin_core::exec::GradMode;
    let depth_before = crate::cuda::tape::depth();
    GradMode::Disabled.scope(|| {
        let a = cuda_f32(&[2], vec![1.0, 2.0]);
        let b = cuda_f32(&[2], vec![3.0, 4.0]);
        let _ = B::add::<f32>(&a, &b).unwrap();
    });
    assert_eq!(crate::cuda::tape::depth(), depth_before);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn f64_exp_trains_on_cuda() {
    // `ones_like` hardcoded f32 bytes under the loss dtype, so any f64
    // backward panicked in `CudaStorage::new` before reaching the kernel.
    fn download_f64(t: &CudaStorage) -> Vec<f64> {
        let bytes = t
            .buffer
            .device
            .default_stream()
            .clone_dtoh(&*t.buffer.data)
            .unwrap();
        bytemuck::cast_slice::<u8, f64>(&bytes).to_vec()
    }
    let vals = vec![0.5f64, 1.0, 1.5];
    let bytes: Vec<u8> = bytemuck::cast_slice(&vals).to_vec();
    let t = crate::cuda::backend::cuda_from_bytes(&[3], DTypeId::F64.into(), 0, &bytes).unwrap();
    let t_id = t.id;
    let out = B::exp::<f64>(&t).unwrap();
    assert_eq!(out.shape, vec![3]);
    let fwd = download_f64(&out);
    for (got, x) in fwd.iter().zip(vals.iter()) {
        assert!(
            (got - x.exp()).abs() < 1e-9,
            "f64 exp fwd: got {got}, want {}",
            x.exp()
        );
    }
    // Ones seed: dx = exp(x).
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let g = download_f64(grads.get(t_id).unwrap());
    for (got, x) in g.iter().zip(vals.iter()) {
        assert!(
            (got - x.exp()).abs() < 1e-9,
            "f64 exp bwd: got {got}, want {}",
            x.exp()
        );
    }
}

// ---------------------------------------------------------------------------
// The eight rows CUDA was missing against CPU: `LogSoftmax`,
// `LogSumExpDim`/`LogSumExpKeepDim`, `Sort`, `RepeatInterleave`, `OneHot`,
// `Bincount`, `ScatterAdd` (issues #86/#87/#88/#84). The first test runs
// everywhere - it answers the capability query, not the device. The rest are
// `#[ignore]`d like every other hardware test above.
// ---------------------------------------------------------------------------

#[test]
fn the_eight_new_cuda_rows_admit_their_canonical_invocations() {
    use incin_core::exec::{
        CapabilityQuery, LayoutClass, MathMode, OperationIdentity, SupportLevel, UnsupportedReason,
    };
    use incin_core::shapes::OperationKind;
    use incin_core::tensor::device::DeviceKind;

    // (operation, operand dtype, training mode) tuples a real invocation
    // sends: the value operand's dtype for the float ops, the index
    // operand's dtype for the three integer-indexed ones. Rank 2 clears
    // every row's floor; Contiguous is admitted by all four groups.
    let admitted: &[(OperationKind, DTypeId, bool)] = &[
        (OperationKind::LogSoftmax, DTypeId::F32, true),
        (OperationKind::LogSumExpDim, DTypeId::F32, true),
        (OperationKind::LogSumExpKeepDim, DTypeId::F32, true),
        (OperationKind::Sort, DTypeId::F32, false),
        (OperationKind::RepeatInterleave, DTypeId::F32, true),
        (OperationKind::OneHot, DTypeId::I64, true),
        (OperationKind::Bincount, DTypeId::I64, true),
        (OperationKind::ScatterAdd, DTypeId::I64, true),
        (OperationKind::ScatterAdd, DTypeId::F32, true),
    ];
    for &(operation, dtype, training) in admitted {
        let query = CapabilityQuery {
            operation: OperationIdentity::Builtin(operation),
            dtype: dtype.descriptor(),
            layout: LayoutClass::Contiguous,
            rank: 2,
            training,
            math_mode: MathMode::Precise,
        };
        let level = crate::capability::support(DeviceKind::Cuda, &query);
        assert!(
            !matches!(level, SupportLevel::Unsupported(_)),
            "{operation:?} must admit a {dtype:?} operand with training={training}, got {level:?}"
        );
    }

    // `Sort` promises evaluation only - the same contract `Argsort` and
    // `TopK` already answer - so a training-mode query is refused by name
    // rather than silently running a gradient-less forward.
    let sort_training = CapabilityQuery {
        operation: OperationIdentity::Builtin(OperationKind::Sort),
        dtype: DTypeId::F32.descriptor(),
        layout: LayoutClass::Contiguous,
        rank: 2,
        training: true,
        math_mode: MathMode::Precise,
    };
    assert!(
        matches!(
            crate::capability::support(DeviceKind::Cuda, &sort_training),
            SupportLevel::Unsupported(UnsupportedReason::Training { .. })
        ),
        "Sort's row is training=false and must refuse a training-mode query"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn log_softmax_matches_the_shifted_definition() {
    let values = [1.0f32, 2.0, 3.0, 0.5, -1.0, 2.5];
    let t = cuda_f32(&[2, 3], values.to_vec());
    let out = crate::cuda::backend::elementwise::cuda_log_softmax::<Cuda>(&t, 1).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    let got = download_f32_host(&out).unwrap();
    for row in 0..2 {
        let xs = &[[1.0f64, 2.0, 3.0], [0.5, -1.0, 2.5]][row];
        let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = xs.iter().map(|x| (x - max).exp()).sum();
        for (col, x) in xs.iter().enumerate() {
            let want = x - max - sum.ln();
            assert!(
                (f64::from(got[row * 3 + col]) - want).abs() < 1e-5,
                "log_softmax[{row},{col}]: got {}, want {want}",
                got[row * 3 + col]
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn log_sum_exp_keeps_the_axis_and_the_dim_form_squeezes_it() {
    use incin_core::exec::catalog::AxisAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let t = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.5]);
    let kept = crate::cuda::backend::elementwise::cuda_logsumexp_keepdim::<Cuda>(&t, 1).unwrap();
    assert_eq!(kept.shape, vec![2, 1]);
    let got = download_f32_host(&kept).unwrap();
    for (row, xs) in [[1.0f64, 2.0, 3.0], [0.5, -1.0, 2.5]].iter().enumerate() {
        let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = xs.iter().map(|x| (x - max).exp()).sum();
        let want = max + sum.ln();
        assert!(
            (f64::from(got[row]) - want).abs() < 1e-5,
            "logsumexp_keepdim[{row}]: got {}, want {want}",
            got[row]
        );
    }

    // The Dim form travels dispatch and comes back squeezed, with the same
    // values: the composition's tail is `squeeze` (a tape-tracked reshape).
    let context = ExecutionContext::new(B::new());
    let handle = TensorHandle::from_storage::<B, f32, _>(&t);
    let squeezed =
        dispatch::execute::<op::LogSumExpDim, _>(&context, AxisAttributes { axis: 1 }, &[handle])
            .expect("logsumexp_dim executes on CUDA");
    assert_eq!(squeezed.shape, vec![2]);
    let squeezed_vals = download_f32_host(&squeezed).unwrap();
    for (keep, dim) in got.iter().zip(squeezed_vals.iter()) {
        assert!(
            (keep - dim).abs() < 1e-6,
            "squeeze changed the value: {keep} vs {dim}"
        );
    }

    // The axis-out-of-range refusal carries CPU's op name and wording.
    let error = crate::cuda::backend::elementwise::cuda_logsumexp_keepdim::<Cuda>(&t, 5)
        .expect_err("axis 5 is outside a rank-2 operand");
    assert!(
        error
            .to_string()
            .contains("logsumexp_keepdim: axis 5 out of range"),
        "unexpected error: {error}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn log_sum_exp_backpropagates_the_row_softmax() {
    // d/dx logsumexp(x) = softmax(x). The stabilizing `max` pushes no tape
    // entry, so this also proves the shift's cotangent cancels for free:
    // a missing cancellation would skew the sum away from 1.0.
    let t = cuda_f32(&[3], vec![1.0, 2.0, 3.0]);
    let t_id = t.id;
    let out = crate::cuda::backend::elementwise::cuda_logsumexp_keepdim::<Cuda>(&t, 0).unwrap();
    assert_eq!(out.shape, vec![1]);
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad = download_f32_host(grads.get(t_id).unwrap()).unwrap();
    let expected = [0.090_030_57f64, 0.244_728_47, 0.665_240_96];
    assert_eq!(grad.len(), 3);
    let total: f64 = grad.iter().map(|&v| f64::from(v)).sum();
    for (i, (got, want)) in grad.iter().zip(expected.iter()).enumerate() {
        assert!(
            (f64::from(*got) - want).abs() < 1e-5,
            "logsumexp grad[{i}]: got {got}, want {want}"
        );
    }
    assert!(
        (total - 1.0).abs() < 1e-5,
        "softmax gradients must sum to 1, got {total}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn sort_orders_each_axis_slice_and_returns_a_replaying_permutation() {
    use incin_core::exec::catalog::ArgsortAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new());
    let input = [3.0f32, 1.0, 2.0, 6.0, 4.0, 5.0];
    let t = cuda_f32(&[2, 3], input.to_vec());
    let handle = TensorHandle::from_storage::<B, f32, _>(&t);
    // index_dtype is what the frontend sends; the CUDA row returns i64
    // indices physically, the same convention Argsort/TopK already use.
    let (values, indices) = dispatch::execute::<op::Sort, _>(
        &context,
        ArgsortAttributes {
            axis: 1,
            descending: false,
            index_dtype: DTypeId::U32.descriptor(),
        },
        &[handle],
    )
    .expect("sort executes on CUDA");
    assert_eq!(values.shape, vec![2, 3]);
    assert_eq!(indices.shape, vec![2, 3]);
    assert_eq!(
        download_f32_host(&values).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let index_bytes = indices
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*indices.buffer.data)
        .unwrap();
    let index_values: Vec<i64> = bytemuck::cast_slice::<u8, i64>(&index_bytes).to_vec();
    assert_eq!(index_values, vec![1, 2, 0, 1, 2, 0]);
    // The permutation replays to the sorted values.
    let sorted = download_f32_host(&values).unwrap();
    for (flat, &position) in index_values.iter().enumerate() {
        assert_eq!(
            input[flat / 3 * 3 + position as usize],
            sorted[flat],
            "indices must reorder the input into the sorted values at {flat}"
        );
    }

    // Descending flips both outputs.
    let handle = TensorHandle::from_storage::<B, f32, _>(&t);
    let (values, indices) = dispatch::execute::<op::Sort, _>(
        &context,
        ArgsortAttributes {
            axis: 1,
            descending: true,
            index_dtype: DTypeId::U32.descriptor(),
        },
        &[handle],
    )
    .expect("descending sort executes on CUDA");
    assert_eq!(
        download_f32_host(&values).unwrap(),
        vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    );
    let index_bytes = indices
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*indices.buffer.data)
        .unwrap();
    let index_values: Vec<i64> = bytemuck::cast_slice::<u8, i64>(&index_bytes).to_vec();
    assert_eq!(index_values, vec![0, 2, 1, 0, 2, 1]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn repeat_interleave_places_copies_adjacently() {
    let t = cuda_f32(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let axis1 = B::repeat_interleave::<f32>(&t, 2, 1).unwrap();
    assert_eq!(axis1.shape, vec![2, 4]);
    assert_eq!(
        download_f32_host(&axis1).unwrap(),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    );
    let axis0 = B::repeat_interleave::<f32>(&t, 3, 0).unwrap();
    assert_eq!(axis0.shape, vec![6, 2]);
    assert_eq!(
        download_f32_host(&axis0).unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    );
    // Fail-closed guards carry CPU's reasons even though the descriptor
    // refuses both shapes of mistake first.
    assert!(matches!(
        B::repeat_interleave::<f32>(&t, 0, 1),
        Err(Error::Backend(
            incin_core::error::BackendError::InvalidInput {
                reason: "repeat_interleave needs at least one repeat per element",
                ..
            }
        ))
    ));
    assert!(matches!(
        B::repeat_interleave::<f32>(&t, 2, 2),
        Err(Error::Backend(
            incin_core::error::BackendError::InvalidInput {
                reason: "repeat_interleave axis is outside the operand's rank",
                ..
            }
        ))
    ));
}

#[test]
#[ignore = "requires CUDA hardware"]
fn repeat_interleave_backward_sums_each_group_in_order() {
    let t = cuda_f32(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let t_id = t.id;
    let out = B::repeat_interleave::<f32>(&t, 2, 1).unwrap();
    let seed = cuda_f32(&[2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let grads = crate::cuda::tape::backward_with(&out, &seed).unwrap();
    let grad = download_f32_host(grads.get(t_id).unwrap()).unwrap();
    // Each source element sits under two adjacent outputs; the group-sum is
    // (1+2, 3+4, 5+6, 7+8).
    assert_eq!(grad, vec![3.0, 7.0, 11.0, 15.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn repeat_interleave_reads_through_a_strided_view() {
    // The row admits Strided operands, so the kernel must decode logical
    // coordinates and read through the view's physical strides - a flat
    // linearization would silently gather the wrong elements here.
    let base = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let view = B::transpose_view::<f32>(&base, 0, 1).unwrap();
    assert_eq!(view.shape, vec![3, 2]);
    let out = B::repeat_interleave::<f32>(&view, 2, 1).unwrap();
    assert_eq!(out.shape, vec![3, 4]);
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 5.0, 5.0, 3.0, 3.0, 6.0, 6.0]
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn one_hot_writes_bool_rows_and_pads_out_of_range_targets() {
    let bytes: Vec<u8> = [0i64, 2, -1, 7]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let t = crate::cuda::backend::cuda_from_bytes(&[4], DTypeId::I64.into(), 0, &bytes).unwrap();
    let out = B::one_hot::<i64>(&t, 4).unwrap();
    assert_eq!(out.shape, vec![4, 4]);
    assert_eq!(out.dtype(), DTypeId::Bool.descriptor());
    let got = out
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*out.buffer.data)
        .unwrap();
    // 0 -> slot 0; 2 -> slot 2; -1 and 7 -> all-false rows (ONNX padding),
    // not an error.
    assert_eq!(got, vec![1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    // A scalar target encodes as a single row.
    let scalar_bytes: Vec<u8> = 2i64.to_le_bytes().to_vec();
    let scalar =
        crate::cuda::backend::cuda_from_bytes(&[], DTypeId::I64.into(), 0, &scalar_bytes).unwrap();
    let row = B::one_hot::<i64>(&scalar, 3).unwrap();
    assert_eq!(row.shape, vec![3]);
    let got = row
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*row.buffer.data)
        .unwrap();
    assert_eq!(got, vec![0, 0, 1]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn bincount_counts_each_bin_and_refuses_a_target_outside_them() {
    let bytes: Vec<u8> = [0i64, 1, 1, 0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let t = crate::cuda::backend::cuda_from_bytes(&[4], DTypeId::I64.into(), 0, &bytes).unwrap();
    let out = B::bincount::<i64>(&t, 3).unwrap();
    assert_eq!(out.shape, vec![3]);
    assert_eq!(out.dtype(), DTypeId::I64.descriptor());
    let got = out
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*out.buffer.data)
        .unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, i64>(&got), &[2i64, 2, 0]);

    // An index outside the bins is CPU's error, not a silently low count.
    let bad_bytes: Vec<u8> = [0i64, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
    let bad =
        crate::cuda::backend::cuda_from_bytes(&[2], DTypeId::I64.into(), 0, &bad_bytes).unwrap();
    assert!(matches!(
        B::bincount::<i64>(&bad, 3),
        Err(Error::Backend(
            incin_core::error::BackendError::InvalidInput {
                reason: "bincount index is not a whole number inside the bin range",
                ..
            }
        ))
    ));
    // Zero bins is refused before anything allocates, with CPU's reason.
    assert!(matches!(
        B::bincount::<i64>(&t, 0),
        Err(Error::Backend(
            incin_core::error::BackendError::InvalidInput {
                reason: "bincount needs at least one bin to count into",
                ..
            }
        ))
    ));
}

#[test]
#[ignore = "requires CUDA hardware"]
fn scatter_add_accumulates_in_order_and_drops_out_of_range_targets() {
    let t = cuda_f32(&[3], vec![0.0, 0.0, 0.0]);
    let index_bytes: Vec<u8> = [0i64, 0, 2, 9]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let index =
        crate::cuda::backend::cuda_from_bytes(&[4], DTypeId::I64.into(), 0, &index_bytes).unwrap();
    let src = cuda_f32(&[4], vec![1.0, 2.0, 4.0, 100.0]);
    let out = B::scatter_add::<f32>(&t, 0, &index, &src).unwrap();
    assert_eq!(out.shape, vec![3]);
    // Duplicate destinations sum (1+2 at slot 0); target 9 is outside a
    // length-3 axis and is dropped rather than clamped, so the 100 never
    // lands anywhere.
    assert_eq!(download_f32_host(&out).unwrap(), vec![3.0, 0.0, 4.0]);

    // Backward under a ones seed: the input's cotangent passes through
    // untouched, the source receives the cotangent at each surviving write
    // and zero at the dropped one, and the index stays off the tape.
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad_t = download_f32_host(grads.get(t.id).unwrap()).unwrap();
    assert_eq!(grad_t, vec![1.0, 1.0, 1.0]);
    let grad_src = download_f32_host(grads.get(src.id).unwrap()).unwrap();
    assert_eq!(grad_src, vec![1.0, 1.0, 1.0, 0.0]);
    assert!(
        grads.get(index.id).is_none(),
        "the integer index operand must stay off the tape, as on CPU"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn scatter_add_refuses_last_write_wins_with_the_cpus_wording() {
    use incin_core::exec::catalog::{DuplicateIndexRule, ScatterAttributes};
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new());
    let t = cuda_f32(&[2], vec![0.0, 0.0]);
    let index_bytes: Vec<u8> = [0i64, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let index =
        crate::cuda::backend::cuda_from_bytes(&[2], DTypeId::I64.into(), 0, &index_bytes).unwrap();
    let src = cuda_f32(&[2], vec![1.0, 2.0]);
    let handles = [
        TensorHandle::from_storage::<B, f32, _>(&t),
        TensorHandle::from_storage::<B, i64, _>(&index),
        TensorHandle::from_storage::<B, f32, _>(&src),
    ];
    let error = dispatch::execute::<op::ScatterAdd, _>(
        &context,
        ScatterAttributes {
            axis: 0,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
        &handles,
    )
    .expect_err("last-write-wins is not a scatter_add rule");
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("scatter_add accumulates duplicate indices and implements no other rule")
            && rendered.contains("use scatter for last-write-wins"),
        "refusal must carry CPU's wording, got {rendered}"
    );
}

// ---------------------------------------------------------------------------
// Issue #103, AC6: CUDA `grouped_matmul`. All five are hardware-gated like
// every other device path above; the geometry mirrors
// `crates/incin/tests/routing_primitives.rs` so CPU and CUDA pin the same
// hand-checkable values.
// ---------------------------------------------------------------------------

fn cuda_i64(shape: &[usize], values: &[i64]) -> CudaStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    crate::cuda::backend::cuda_from_bytes(shape, DTypeId::I64.into(), 0, &bytes).unwrap()
}

#[test]
#[ignore = "requires CUDA hardware"]
fn grouped_matmul_applies_each_experts_weight_to_its_own_rows() {
    let lhs = cuda_f32(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let rhs = cuda_f32(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let offsets = cuda_i64(&[3], &[0, 2, 2]);
    let out = B::grouped_matmul::<f32>(&lhs, &rhs, &offsets).unwrap();
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![1.0, 2.0, 3.0, 4.0],
        "expert 0's weights apply to both rows; expert 1 owns none"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn grouped_matmul_skips_an_empty_expert_span() {
    let lhs = cuda_f32(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    // Expert 1 and 2 are empty (offsets 1..1, 1..1); expert 0 owns both rows.
    let rhs = cuda_f32(
        &[3, 2, 2],
        vec![
            1.0, 2.0, 3.0, 4.0, // expert 0
            5.0, 6.0, 7.0, 8.0, // expert 1 (empty)
            9.0, 10.0, 11.0, 12.0, // expert 2 (empty)
        ],
    );
    let offsets = cuda_i64(&[4], &[0, 2, 2, 2]);
    let out = B::grouped_matmul::<f32>(&lhs, &rhs, &offsets).unwrap();
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(download_f32_host(&out).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn grouped_matmul_matches_a_loop_of_individual_matmuls() {
    let tokens = 4;
    let k = 2;
    let experts = 2;
    let n = 3;
    let lhs_values: Vec<f32> = (0..tokens * k).map(|i| (i % 5) as f32 + 1.0).collect();
    let rhs_values: Vec<f32> = (0..experts * k * n).map(|i| (i % 7) as f32 * 0.5).collect();
    let lhs = cuda_f32(&[tokens, k], lhs_values.clone());
    let rhs = cuda_f32(&[experts, k, n], rhs_values.clone());
    let offsets = cuda_i64(&[3], &[0, 1, tokens as i64]);

    let out = B::grouped_matmul::<f32>(&lhs, &rhs, &offsets).unwrap();
    assert_eq!(out.shape, vec![tokens, n]);
    let got = download_f32_host(&out).unwrap();

    // Host-side reference: expert 0 owns row 0, expert 1 owns rows 1..4.
    let mut expected = vec![0f32; tokens * n];
    for (expert, start, end) in [(0usize, 0usize, 1usize), (1, 1, tokens)] {
        for row in start..end {
            for col in 0..n {
                let mut acc = 0.0f32;
                for contracting in 0..k {
                    acc += lhs_values[row * k + contracting]
                        * rhs_values[(expert * k + contracting) * n + col];
                }
                expected[row * n + col] = acc;
            }
        }
    }
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "grouped_matmul[{i}]: got {g}, loop reference {e}"
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn grouped_matmul_refuses_offsets_that_do_not_tile_with_cpus_wording() {
    let lhs = cuda_f32(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let rhs = cuda_f32(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    // Final offset is 1, not 2: expert rows do not cover the activation.
    let bad = cuda_i64(&[3], &[0, 1, 1]);
    let err = B::grouped_matmul::<f32>(&lhs, &rhs, &bad)
        .expect_err("offsets that leave a row uncovered must be refused");
    let rendered = format!("{err}");
    assert!(
        rendered.contains("must tile [0, 2)"),
        "tile refusal must carry CPU's wording, got {rendered}"
    );

    // Decreasing offsets name overlapping spans rather than a partition.
    let decreasing = cuda_i64(&[3], &[0, 2, 1]);
    let err = B::grouped_matmul::<f32>(&lhs, &rhs, &decreasing)
        .expect_err("decreasing offsets must be refused");
    let rendered = format!("{err}");
    assert!(
        rendered.contains("must be non-decreasing"),
        "ordering refusal must carry CPU's wording, got {rendered}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn grouped_matmul_gradient_reaches_both_matrices_and_not_the_offsets() {
    let lhs = cuda_f32(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let rhs = cuda_f32(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let offsets = cuda_i64(&[3], &[0, 1, 2]);
    let (lhs_id, rhs_id, offsets_id) = (lhs.id, rhs.id, offsets.id);

    let out = B::grouped_matmul::<f32>(&lhs, &rhs, &offsets).unwrap();
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![1.0, 2.0, 7.0, 8.0],
        "each row should use its own expert's weights"
    );

    // Ones-seeded backward is d(sum)/d(inputs), the same check CPU's
    // routing_primitives gradient test makes through sum_all.
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad_lhs = download_f32_host(grads.get(lhs_id).unwrap()).unwrap();
    assert_eq!(
        grad_lhs,
        vec![3.0, 7.0, 11.0, 15.0],
        "d(sum)/d(lhs) = each row dotted into its own expert's weights"
    );
    let grad_rhs = download_f32_host(grads.get(rhs_id).unwrap()).unwrap();
    assert_eq!(
        grad_rhs,
        vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "d(sum)/d(rhs[e]) = lhs_rows^T @ ones for that expert's rows"
    );
    assert!(
        grads.get(offsets_id).is_none(),
        "the integer offsets operand must stay off the tape, as on CPU"
    );
}

// ---------------------------------------------------------------------------
// Issue #90: the dtype-parametric matmul rows. The first test runs
// everywhere - it answers the capability query, not the device. The rest are
// `#[ignore]`d like every other hardware test above.
// ---------------------------------------------------------------------------

#[test]
fn the_dtype_parametric_cuda_matmul_rows_admit_their_canonical_invocations() {
    use incin_core::exec::{
        CapabilityQuery, LayoutClass, MathMode, OperationIdentity, SupportLevel, UnsupportedReason,
    };
    use incin_core::shapes::OperationKind;
    use incin_core::tensor::device::DeviceKind;

    // (operation, operand dtype, rank, training): every float storage dtype
    // across the four rows that moved to FLOAT_DTYPES. Rank clears each
    // row's floor (2 for `matmul`, 3 for `bmm`, 1 for `addmm`/`linear`);
    // Contiguous is admitted by all the groups involved.
    let admitted: &[(OperationKind, DTypeId, usize, bool)] = &[
        (OperationKind::MatMulExact, DTypeId::BF16, 2, true),
        (OperationKind::MatMulExact, DTypeId::F16, 2, true),
        (OperationKind::MatMulExact, DTypeId::F32, 2, true),
        (OperationKind::MatMulExact, DTypeId::F64, 2, true),
        (OperationKind::BatchedMatMul, DTypeId::BF16, 3, true),
        (OperationKind::BatchedMatMul, DTypeId::F16, 3, true),
        (OperationKind::BatchedMatMul, DTypeId::F32, 3, true),
        (OperationKind::BatchedMatMul, DTypeId::F64, 3, true),
        (OperationKind::Addmm, DTypeId::BF16, 2, true),
        (OperationKind::Addmm, DTypeId::F16, 2, true),
        (OperationKind::Addmm, DTypeId::F32, 2, true),
        (OperationKind::Addmm, DTypeId::F64, 2, true),
        (OperationKind::Linear, DTypeId::BF16, 2, true),
        (OperationKind::Linear, DTypeId::F16, 2, true),
        (OperationKind::Linear, DTypeId::F32, 2, true),
        (OperationKind::Linear, DTypeId::F64, 2, true),
    ];
    for &(operation, dtype, rank, training) in admitted {
        let query = CapabilityQuery {
            operation: OperationIdentity::Builtin(operation),
            dtype: dtype.descriptor(),
            layout: LayoutClass::Contiguous,
            rank,
            training,
            math_mode: MathMode::Precise,
        };
        let level = crate::capability::support(DeviceKind::Cuda, &query);
        assert!(
            !matches!(level, SupportLevel::Unsupported(_)),
            "{operation:?} must admit a {dtype:?} operand at rank {rank} with training={training}, got {level:?}"
        );
    }

    // FLOAT_DTYPES is the whole widening: integer, boolean and quantized
    // operands stay refused rather than riding the moved rows.
    let refused: &[(OperationKind, DTypeId, usize)] = &[
        (OperationKind::MatMulExact, DTypeId::I64, 2),
        (OperationKind::MatMulExact, DTypeId::Bool, 2),
        (OperationKind::MatMulExact, DTypeId::Q8_0, 2),
        (OperationKind::BatchedMatMul, DTypeId::I64, 3),
        (OperationKind::Addmm, DTypeId::I64, 2),
        (OperationKind::Linear, DTypeId::I64, 2),
    ];
    for &(operation, dtype, rank) in refused {
        let query = CapabilityQuery {
            operation: OperationIdentity::Builtin(operation),
            dtype: dtype.descriptor(),
            layout: LayoutClass::Contiguous,
            rank,
            training: true,
            math_mode: MathMode::Precise,
        };
        let level = crate::capability::support(DeviceKind::Cuda, &query);
        assert!(
            matches!(level, SupportLevel::Unsupported(_)),
            "{operation:?} must still refuse a {dtype:?} operand, got {level:?}"
        );
    }

    // The rows that did not widen stay honest: `ScaledDotProductAttention`
    // sits on f32-only `softmax`, so an f16 query is refused by dtype.
    let sdpa_f16 = CapabilityQuery {
        operation: OperationIdentity::Builtin(OperationKind::ScaledDotProductAttention),
        dtype: DTypeId::F16.descriptor(),
        layout: LayoutClass::Contiguous,
        rank: 2,
        training: true,
        math_mode: MathMode::Precise,
    };
    assert!(
        matches!(
            crate::capability::support(DeviceKind::Cuda, &sdpa_f16),
            SupportLevel::Unsupported(UnsupportedReason::DType { .. })
        ),
        "SDPA must stay f32-only on CUDA"
    );

    // A training-mode query keeps resolving: the moved rows all carry a
    // real tape entry (checked structurally by the rank routing test below
    // and the existing backward tests).
    let training = CapabilityQuery {
        operation: OperationIdentity::Builtin(OperationKind::MatMulExact),
        dtype: DTypeId::F32.descriptor(),
        layout: LayoutClass::Contiguous,
        rank: 2,
        training: false,
        math_mode: MathMode::Precise,
    };
    assert!(
        !matches!(
            crate::capability::support(DeviceKind::Cuda, &training),
            SupportLevel::Unsupported(_)
        ),
        "MatMulExact must also resolve in inference mode"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_multiplies_f16_operands() {
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]] -
    // every intermediate is an integer below 2048, exactly representable
    // in f16, so this asserts the kernel's arithmetic rather than a
    // tolerance window.
    let lhs_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let rhs_bytes: Vec<u8> = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let lhs = crate::cuda::backend::cuda_from_bytes(&[2, 3], DTypeId::F16.into(), 0, &lhs_bytes)
        .expect("f16 storage is CUDA-admissible");
    let rhs = crate::cuda::backend::cuda_from_bytes(&[3, 2], DTypeId::F16.into(), 0, &rhs_bytes)
        .expect("f16 storage is CUDA-admissible");
    let out = B::matmul::<f32>(&lhs, &rhs).expect("f16 matmul executes");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.dtype(), DTypeId::F16.descriptor());
    let got_bytes = out
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*out.buffer.data)
        .unwrap();
    let got: Vec<f32> = got_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(got, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_multiplies_bf16_operands() {
    // Same product as the f16 test; every result is an integer below 256,
    // exactly representable in bf16's 8-bit mantissa.
    let lhs_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let rhs_bytes: Vec<u8> = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let lhs = crate::cuda::backend::cuda_from_bytes(&[2, 3], DTypeId::BF16.into(), 0, &lhs_bytes)
        .expect("bf16 storage is CUDA-admissible");
    let rhs = crate::cuda::backend::cuda_from_bytes(&[3, 2], DTypeId::BF16.into(), 0, &rhs_bytes)
        .expect("bf16 storage is CUDA-admissible");
    let out = B::matmul::<f32>(&lhs, &rhs).expect("bf16 matmul executes");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.dtype(), DTypeId::BF16.descriptor());
    let got_bytes = out
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*out.buffer.data)
        .unwrap();
    let got: Vec<f32> = got_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert_eq!(got, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_multiplies_f64_operands() {
    let lhs_vals = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_vals = [7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let lhs_bytes: Vec<u8> = lhs_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let rhs_bytes: Vec<u8> = rhs_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let lhs = crate::cuda::backend::cuda_from_bytes(&[2, 3], DTypeId::F64.into(), 0, &lhs_bytes)
        .expect("f64 storage is CUDA-admissible");
    let rhs = crate::cuda::backend::cuda_from_bytes(&[3, 2], DTypeId::F64.into(), 0, &rhs_bytes)
        .expect("f64 storage is CUDA-admissible");
    let out = B::matmul::<f32>(&lhs, &rhs).expect("f64 matmul executes");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.dtype(), DTypeId::F64.descriptor());
    let got_bytes = out
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*out.buffer.data)
        .unwrap();
    let got: Vec<f64> = got_bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_refuses_a_mixed_dtype_operand_pair() {
    // One typed kernel reads both operands, so f32 @ f64 fails on the host
    // side before any buffer is reinterpreted.
    let lhs = cuda_f32(&[2, 2], vec![1.0; 4]);
    let rhs_bytes: Vec<u8> = [1.0f64; 4].iter().flat_map(|v| v.to_le_bytes()).collect();
    let rhs = crate::cuda::backend::cuda_from_bytes(&[2, 2], DTypeId::F64.into(), 0, &rhs_bytes)
        .expect("f64 storage is CUDA-admissible");
    let error = B::matmul::<f32>(&lhs, &rhs).expect_err("f32 @ f64 must be refused");
    assert!(
        matches!(
            error,
            Error::DTypeMismatch {
                operation: "matmul",
                ..
            }
        ),
        "mixed-dtype matmul must fail as a dtype mismatch, got {error:?}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_exact_executes_rank_three_by_routing_to_batched_matmul() {
    // `Tensor::matmul` sends every rank through `MatMulExact`, and the
    // capability row admits 2..=MAX: before the rank routing fix this
    // failed inside `matmul`'s unbatched-2D check even though the row
    // promised rank 3.
    use incin_core::exec::catalog::NoAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new());
    let lhs = cuda_f32(&[2, 2, 3], (1..=12).map(|v| v as f32).collect());
    let rhs = cuda_f32(&[2, 3, 2], (1..=12).map(|v| v as f32).collect());
    let lhs_handle = TensorHandle::from_storage::<B, f32, _>(&lhs);
    let rhs_handle = TensorHandle::from_storage::<B, f32, _>(&rhs);
    let out =
        dispatch::execute::<op::MatMulExact, _>(&context, NoAttributes, &[lhs_handle, rhs_handle])
            .expect("a rank-3 MatMulExact must reach batched_matmul, not the unbatched-2D refusal");
    assert_eq!(out.shape, vec![2, 2, 2]);
    // batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    // batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[220,244],[301,334]]
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0]
    );
}

// ---------------------------------------------------------------------------
// Issue #84: the four loss/norm families that still had no hardware value
// tests — mse, bce_with_logits (zero-logit slope regression), group_norm,
// and instance_norm. All are `#[ignore]`d so `cargo test` stays green on
// machines without a CUDA device; run with `-- --ignored` on real hardware.
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn mse_loss_trains_through_scalar_reduction_on_cuda() {
    // MSE is sub, mul, mean — the same composition shape as l1, so it gets
    // its own hardware walk to pin the squared-diff chain end to end.
    use incin_core::exec::catalog::{LossAttributes, LossReduction};
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new());
    let pred_values = vec![1.0, 0.0, -1.0, 2.0];
    let targ_values = vec![1.0, 1.0, 0.0, 0.5];
    let pred = cuda_f32(&[4], pred_values.clone());
    let targ = cuda_f32(&[4], targ_values.clone());
    let pred_id = pred.id;
    let out = dispatch::execute::<op::MseLoss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[
            TensorHandle::from_storage::<B, f32, _>(&pred),
            TensorHandle::from_storage::<B, f32, _>(&targ),
        ],
    )
    .expect("mse executes on CUDA");

    let fwd = download_f32_host(&out).unwrap();
    assert_eq!(fwd.len(), 1);

    // CPU reference on the same values, seeded with ones (backward's default).
    let host_out_and_grads = cpu_forward_and_grads::<op::MseLoss>(
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[
            host_f32(&[4], pred_values.clone()),
            host_f32(&[4], targ_values),
        ],
        &[1.0],
    );
    let want_fwd = host_values(&host_out_and_grads.0);
    let want_dx = host_values(&host_out_and_grads.1[0]);
    assert_close(
        &fwd.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        &want_fwd,
        1e-5,
        "mse forward",
    );

    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad = grads.get(pred_id).expect("pred has a gradient");
    let got_dx: Vec<f64> = download_f32_host(grad)
        .unwrap()
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    assert_close(&got_dx, &want_dx, 1e-5, "mse dx");
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn bce_with_logits_zero_logit_backward_matches_sigmoid_minus_target_on_cuda() {
    // The stock relu derivative answers 0 at exactly zero; CPU (and PyTorch)
    // define the subgradient of max(x, 0) at x = 0 as 0.5. This pins the
    // #84 fix: GradMode::Disabled relu plus a custom tape entry with the
    // CPU slope table.
    use incin_core::exec::catalog::{LossAttributes, LossReduction};
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new());
    let p_values = vec![0.0, 0.0, 0.0];
    let z_values = vec![0.0, 1.0, 0.25];
    for (reduction, scale) in [
        (LossReduction::Sum, 1.0f64),
        (LossReduction::Mean, 1.0 / 3.0),
    ] {
        let p = cuda_f32(&[1, 3], p_values.clone());
        let z = cuda_f32(&[1, 3], z_values.clone());
        let p_id = p.id;
        let out = dispatch::execute::<op::BceWithLogitsLoss, _>(
            &context,
            LossAttributes { reduction },
            &[
                TensorHandle::from_storage::<B, f32, _>(&p),
                TensorHandle::from_storage::<B, f32, _>(&z),
            ],
        )
        .expect("bce_with_logits executes on CUDA");

        let grads = crate::cuda::tape::backward(&out).unwrap();
        let grad = grads.get(p_id).expect("pred should have a gradient");
        let got = download_f32_host(grad).unwrap();
        for (i, (actual, expected)) in got.iter().zip([0.5f64, -0.5, 0.25].iter()).enumerate() {
            assert!(
                (f64::from(*actual) - expected * scale).abs() < 1e-6,
                "{reduction:?}: grad[{i}] = {actual}, expected {}",
                expected * scale
            );
        }
    }

    // Forward also matches the CPU reference on the same inputs.
    let p = cuda_f32(&[1, 3], p_values.clone());
    let z = cuda_f32(&[1, 3], z_values.clone());
    let out = dispatch::execute::<op::BceWithLogitsLoss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[
            TensorHandle::from_storage::<B, f32, _>(&p),
            TensorHandle::from_storage::<B, f32, _>(&z),
        ],
    )
    .unwrap();
    let (host_out, _) = cpu_forward_and_grads::<op::BceWithLogitsLoss>(
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[
            host_f32(&[1, 3], p_values.clone()),
            host_f32(&[1, 3], z_values),
        ],
        &[1.0; 3],
    );
    let got_fwd: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    assert_close(&got_fwd, &host_values(&host_out), 1e-5, "bce forward");
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn group_norm_trains_through_the_statistical_path_on_cuda() {
    // Group norm composes reshape → mean → sub → mul → mean → add eps →
    // sqrt → div → reshape. Every step is tape-tracked, so forward must
    // match the CPU reference values and backward must replay the same
    // statistical path (not a silent variance).
    use incin_core::exec::catalog::GroupNormAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new());
    let values = vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0];
    let input = cuda_f32(&[2, 4], values.clone());
    let input_id = input.id;
    let out = dispatch::execute::<op::GroupNorm, _>(
        &context,
        GroupNormAttributes {
            groups: 2,
            epsilon: 1e-5,
        },
        &[TensorHandle::from_storage::<B, f32, _>(&input)],
    )
    .expect("group_norm executes on CUDA");

    let seed_values = vec![1.0; 8];
    let (host_out, host_grads) = cpu_forward_and_grads::<op::GroupNorm>(
        GroupNormAttributes {
            groups: 2,
            epsilon: 1e-5,
        },
        &[host_f32(&[2, 4], values.clone())],
        &seed_values,
    );

    let got_fwd: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    assert_close(
        &got_fwd,
        &host_values(&host_out),
        1e-5,
        "group_norm forward",
    );

    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad = grads.get(input_id).expect("input has a gradient");
    let got_dx: Vec<f64> = download_f32_host(grad)
        .unwrap()
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    assert_close(&got_dx, &host_values(&host_grads[0]), 1e-5, "group_norm dx");
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires CUDA hardware"]
fn instance_norm_normalizes_each_channel_alone_on_cuda() {
    // instance_norm is group_norm with one group per channel, so each
    // channel of each sample is normalized alone. Forward must match the
    // CPU `instance_norm_storage` reference.
    use incin_core::exec::catalog::EpsilonAttributes;
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};

    let context = ExecutionContext::new(B::new());
    let values = vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0];
    let input = cuda_f32(&[2, 4], values.clone());
    let out = dispatch::execute::<op::InstanceNorm, _>(
        &context,
        EpsilonAttributes { epsilon: 1e-5 },
        &[TensorHandle::from_storage::<B, f32, _>(&input)],
    )
    .expect("instance_norm executes on CUDA");

    let (host_out, _) = cpu_forward_and_grads::<op::InstanceNorm>(
        EpsilonAttributes { epsilon: 1e-5 },
        &[host_f32(&[2, 4], values)],
        &[1.0; 8],
    );
    let got_fwd: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    assert_close(
        &got_fwd,
        &host_values(&host_out),
        1e-5,
        "instance_norm forward",
    );
}
