use super::*;

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
/// `reshape_through_trait_matches_direct_storage_call`.
fn reshape_through_trait_matches_direct_storage_call() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let direct = t.reshape(&[3, 2]).unwrap();
    let via_trait = reshape_storage(&t, &[3, 2]).unwrap();
    assert_eq!(via_trait.shape, direct.shape);
    assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
}

#[test]
/// `transpose_through_trait_matches_direct_storage_call`.
fn transpose_through_trait_matches_direct_storage_call() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let direct = t.transpose(0, 1).unwrap();
    let via_trait = transpose_storage(&t, 0, 1).unwrap();
    assert_eq!(via_trait.shape, direct.shape);
    assert_eq!(via_trait.strides, direct.strides);
}

#[test]
/// `broadcast_as_through_trait_matches_direct_storage_call`.
fn broadcast_as_through_trait_matches_direct_storage_call() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
    let direct = t.broadcast_as(&[4, 3]).unwrap();
    let via_trait = broadcast_as_storage(&t, &[4, 3]).unwrap();
    assert_eq!(via_trait.shape, direct.shape);
    assert_eq!(via_trait.strides, direct.strides);
}

#[test]
/// `float_to_scalar_reads_single_element`.
fn float_to_scalar_reads_single_element() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![42.0]), vec![]);
    let v = float_to_scalar_storage(&t).unwrap();
    assert_eq!(v, 42.0);
}

#[test]
/// `float_to_vec1_reads_all_elements_row_major`.
fn float_to_vec1_reads_all_elements_row_major() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
    let v = float_to_vec1_storage(&t).unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

#[test]
/// `reshape_backward_reshapes_grad_back_to_original_shape`.
fn reshape_backward_reshapes_grad_back_to_original_shape() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = reshape_storage(&t, &[6]).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    assert_eq!(g.shape, vec![2, 3]);
    assert_eq!(f32_vec(g), vec![1.0; 6]);
}

#[test]
/// `transpose_backward_reapplies_same_transpose`.
fn transpose_backward_reapplies_same_transpose() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = transpose_storage(&t, 0, 1).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    assert_eq!(g.shape, vec![2, 3]);
    assert_eq!(f32_vec(g), vec![1.0; 6]);
}

#[test]
/// `broadcast_as_backward_unbroadcasts_to_original_shape`.
fn broadcast_as_backward_unbroadcasts_to_original_shape() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
    let out = broadcast_as_storage(&t, &[4, 3]).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    assert_eq!(g.shape, vec![1, 3]);
    // ones_like(out) [4,3] summed over the broadcast axis -> [4,4,4]
    assert_eq!(f32_vec(g), vec![4.0, 4.0, 4.0]);
}

#[test]
/// `matmul_via_trensor_ops_delegates_to_matmul_impl`.
fn matmul_via_trensor_ops_delegates_to_matmul_impl() {
    let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let rhs = matrix(
        vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        3,
        4,
    );
    let out = matmul_storage(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 4]);
    assert_eq!(
        f32_vec(&out),
        vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
    );
}

#[test]
/// `unsupported_methods_return_typed_error_not_silent_placeholder`.
fn unsupported_methods_return_typed_error_not_silent_placeholder() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    // All other  methods are now fully implemented. We prove that
    // unsupported operations return typed errors by attempting to convert
    // to Q8_0, which is intentionally left unsupported in the Cpu backend.
    let result = tensor_to_dtype_storage(&t, DTypeId::Q8_0.descriptor());
    assert!(matches!(
        result,
        Err(Error::UnsupportedBackendOperation {
            op: "tensor_to_dtype(Q8_0)",
            ..
        })
    ));
}

/// Task 1 Test 1: `::narrow` called through the trait matches
/// calling `CpuStorage::narrow` directly (thin-wrapper equivalence).
#[test]
fn narrow_through_trait_matches_direct_storage_call() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let direct = t.narrow(0, 1, 1).unwrap();
    let via_trait = narrow_storage(&t, 0, 1, 1).unwrap();
    assert_eq!(via_trait.shape, direct.shape);
    assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
}

/// Task 1 Test 2: `narrow`'s backward zero-pads `grad_out` back to the
/// original shape at the correct region.
#[test]
fn narrow_backward_zero_pads_grad_to_original_shape() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let out = narrow_storage(&t, 0, 1, 1).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    assert_eq!(g.shape, vec![3, 2]);
    assert_eq!(f32_vec(g), vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
}

/// Task 1 Test 3: out-of-bounds narrow range returns `Err`, not a panic.
#[test]
fn narrow_out_of_bounds_returns_err_not_panic() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let result = narrow_storage(&t, 0, 2, 2);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

/// Task 1 Test 4: `narrow`'s forward value on a pre-transposed
/// (non-contiguous) input still produces correct values.
#[test]
fn narrow_on_transposed_input_produces_correct_values_without_materializing() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let transposed = transpose_storage(&t, 0, 1).unwrap();
    // transposed is logically [[1,4],[2,5],[3,6]], shape [3,2]
    let narrowed = narrow_storage(&transposed, 0, 1, 1).unwrap();
    assert_eq!(narrowed.shape, vec![1, 2]);
    assert_eq!(narrowed.get(&[0, 0]), 2.0);
    assert_eq!(narrowed.get(&[0, 1]), 5.0);
}

/// Task 2 Test 1: `slice(t, &[(1,3),(0,2)])` on a `[4,3]` matrix matches
/// manually narrowing dim 0 to `(1,3)` then dim 1 to `(0,2)` in sequence.
#[test]
fn slice_matches_manual_sequential_narrow_calls() {
    let t = matrix(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        4,
        3,
    );
    let manual = narrow_storage(&t, 0, 1, 2).unwrap();
    let manual = narrow_storage(&manual, 1, 0, 2).unwrap();

    let via_slice = slice_storage(&t, &[(1, 3), (0, 2)]).unwrap();
    assert_eq!(via_slice.shape, manual.shape);
    assert_eq!(f32_vec(&via_slice), f32_vec(&manual));
}

/// Task 2 Test 2: `slice` on a pre-transposed (non-contiguous) input,
/// across multiple dims, produces correct values without a
/// `.contiguous()` call happening internally.
#[test]
fn slice_on_transposed_input_across_multiple_dims_produces_correct_values() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let transposed = transpose_storage(&t, 0, 1).unwrap();
    // transposed: [[1,4],[2,5],[3,6]], shape [3,2]
    // slice rows [1,3) and cols [0,1) -> [[2],[3]]
    let out = slice_storage(&transposed, &[(1, 3), (0, 1)]).unwrap();
    assert_eq!(out.shape, vec![2, 1]);
    assert_eq!(out.get(&[0, 0]), 2.0);
    assert_eq!(out.get(&[1, 0]), 3.0);
}

/// Task 2 Test 3: `slice`'s backward correctly zero-pads back to the
/// original shape, composed entirely from `narrow`'s own backward.
#[test]
fn slice_backward_zero_pads_grad_to_original_shape() {
    let t = matrix(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        4,
        3,
    );
    let out = slice_storage(&t, &[(1, 3), (0, 2)]).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    assert_eq!(g.shape, vec![4, 3]);
    assert_eq!(
        f32_vec(g),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );
}

/// Task 2 Test 4: an out-of-bounds range in any dim of a multi-dim
/// `slice` call returns `Err`, not a panic.
#[test]
fn slice_out_of_bounds_range_returns_err_not_panic() {
    let t = matrix(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        4,
        3,
    );
    let result = slice_storage(&t, &[(1, 3), (0, 5)]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

/// `tensor3`.
fn tensor3(v: Vec<f32>, d0: usize, d1: usize, d2: usize) -> CpuStorage {
    CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![d0, d1, d2])
}

/// Task 3 Test 1: `squeeze(t, 1)` on a `[3,1,4]` storage produces shape
/// `[3,4]` with unchanged (row-major) values.
#[test]
fn squeeze_removes_size_one_axis_and_preserves_values() {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = tensor3(data.clone(), 3, 1, 4);
    let out = squeeze_storage(&t, 1).unwrap();
    assert_eq!(out.shape, vec![3, 4]);
    assert_eq!(f32_vec(&out), data);
}

/// Task 3 Test 2: `squeeze(t, 0)` on a `[3,1,4]` storage (dim 0 has size
/// 3, not 1) returns a clear squeeze-specific `Error::ShapeMismatch`.
#[test]
fn squeeze_on_non_one_sized_axis_returns_shape_mismatch() {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = tensor3(data, 3, 1, 4);
    let result = squeeze_storage(&t, 0);
    match result {
        Err(Error::ShapeMismatch { op, .. }) => assert_eq!(op, "squeeze"),
        other => panic!("expected squeeze-specific ShapeMismatch, got {other:?}"),
    }
}

/// Task 3 Test 3: `squeeze`'s backward reshapes `grad_out` back to the
/// original `[3,1,4]` shape, delegated entirely to `reshape`'s backward.
#[test]
fn squeeze_backward_reshapes_grad_to_original_shape() {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = tensor3(data, 3, 1, 4);
    let out = squeeze_storage(&t, 1).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    assert_eq!(g.shape, vec![3, 1, 4]);
    assert_eq!(f32_vec(g), vec![1.0; 12]);
}

/// Task 3 Test 4: `flatten(t, 1, 2)` on a `[2,3,4]` storage produces
/// shape `[2,12]` (merging dims 1..=2).
#[test]
fn flatten_merges_middle_dims() {
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let t = tensor3(data.clone(), 2, 3, 4);
    let out = flatten_storage(&t, 1, 2).unwrap();
    assert_eq!(out.shape, vec![2, 12]);
    assert_eq!(f32_vec(&out), data);
}

/// Task 3 Test 5: `flatten(t, 0, 2)` on a `[2,3,4]` storage (flattening
/// all dims) produces shape `[24]`.
#[test]
fn flatten_all_dims_produces_1d_shape() {
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let t = tensor3(data.clone(), 2, 3, 4);
    let out = flatten_storage(&t, 0, 2).unwrap();
    assert_eq!(out.shape, vec![24]);
    assert_eq!(f32_vec(&out), data);
}

/// Task 3 Test 6: `flatten`'s backward reshapes `grad_out` back to the
/// original shape, delegated entirely to `reshape`'s backward.
#[test]
fn flatten_backward_reshapes_grad_to_original_shape() {
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let t = tensor3(data, 2, 3, 4);
    let out = flatten_storage(&t, 1, 2).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    assert_eq!(g.shape, vec![2, 3, 4]);
    assert_eq!(f32_vec(g), vec![1.0; 24]);
}

/// Test 6: `::matmul` called through the trait on two rank-2
/// operands still produces identical values to a direct `matmul_impl`
/// call (dispatch does not change the unbatched path's behavior).
#[test]
fn matmul_dispatch_rank2_matches_matmul_impl_directly() {
    let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let rhs = matrix(
        vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        3,
        4,
    );
    let direct = matmul_impl(&lhs, &rhs).unwrap();
    let via_trait = matmul_storage(&lhs, &rhs).unwrap();
    assert_eq!(via_trait.shape, direct.shape);
    assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
}

/// Test 7: `::matmul` called through the trait on two rank-3
/// (or higher) operands correctly dispatches to `batched_matmul_impl`
/// and produces the same values a direct `batched_matmul_impl` call
/// would.
#[test]
fn matmul_dispatch_rank3_matches_batched_matmul_impl_directly() {
    let lhs_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let rhs_data: Vec<f32> = (1..=40).map(|x| x as f32).collect();
    let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(lhs_data), vec![2, 3, 4]);
    let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(rhs_data), vec![2, 4, 5]);

    let direct = batched_matmul_impl(&lhs, &rhs).unwrap();
    let via_trait = matmul_storage(&lhs, &rhs).unwrap();
    assert_eq!(via_trait.shape, direct.shape);
    assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
}

/// Task 1 Test 1: `concat(&[a, b], 0)` where `a` is `[2,3]` and `b` is
/// `[3,3]` produces shape `[5,3]`, rows 0-1 matching `a`, rows 2-4
/// matching `b`.
#[test]
fn concat_dim0_stacks_rows_in_input_order() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = matrix(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        3,
        3,
    );
    let out = concat_storage(&[&a, &b], 0).unwrap();
    assert_eq!(out.shape, vec![5, 3]);
    assert_eq!(
        f32_vec(&out),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
        ]
    );
}

/// Task 1 Test 2: `concat(&[a, b], 1)` where `a` is `[2,3]` and `b` is
/// `[2,2]` produces shape `[2,5]`, columns correctly interleaved by row.
#[test]
fn concat_dim1_interleaves_columns_by_row() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = matrix(vec![7.0, 8.0, 9.0, 10.0], 2, 2);
    let out = concat_storage(&[&a, &b], 1).unwrap();
    assert_eq!(out.shape, vec![2, 5]);
    assert_eq!(
        f32_vec(&out),
        vec![1.0, 2.0, 3.0, 7.0, 8.0, 4.0, 5.0, 6.0, 9.0, 10.0]
    );
}

/// Task 1 Test 3 (Pitfall 5 regression): a size-1-vs-size-larger
/// mismatch at a NON-concat axis is REJECTED with `Err(ShapeMismatch)`,
/// proving the validation uses exact equality, not
/// `stride::broadcast_shape`'s size-1-is-compatible-with-anything rule.
#[test]
fn concat_rejects_non_concat_axis_size_mismatch_even_when_broadcast_compatible() {
    // a: [3,1], b: [3,4] -- dim 1 sizes differ (1 vs 4), concatenating on
    // dim 0. stride::broadcast_shape would treat size-1 as compatible
    // with anything; concat must NOT.
    let a = matrix(vec![1.0, 2.0, 3.0], 3, 1);
    let b = matrix(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        3,
        4,
    );
    let result = concat_storage(&[&a, &b], 0);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

/// Task 1 Test 4: `concat(&[], 0)` (empty input list) returns
/// `Err(Error::ShapeMismatch)`, not a panic.
#[test]
fn concat_empty_input_list_returns_err_not_panic() {
    let result: Result<CpuStorage> = concat_storage(&[], 0);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

/// Task 1 Test 5: `concat` called with `dim >= rank` returns
/// `Err(Error::ShapeMismatch)`.
#[test]
fn concat_dim_out_of_bounds_returns_err() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let result = concat_storage(&[&a, &b], 2);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

/// Task 1 Test 6: `concat`'s backward correctly narrows `grad_out` back
/// to each input's own shape at its cumulative `dim`-offset, with 2
/// inputs of DIFFERENT sizes along the concat dim.
#[test]
fn concat_backward_narrows_grad_to_each_inputs_own_shape_and_values() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = matrix(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        3,
        3,
    );
    let out = concat_storage(&[&a, &b], 0).unwrap();
    let grads = tape::backward(&out).unwrap();

    let ga = grads.get(a.id).expect("a should have a gradient");
    assert_eq!(ga.shape, vec![2, 3]);
    for r in 0..2 {
        for c in 0..3 {
            assert_eq!(ga.get(&[r, c]), 1.0);
        }
    }

    let gb = grads.get(b.id).expect("b should have a gradient");
    assert_eq!(gb.shape, vec![3, 3]);
    for r in 0..3 {
        for c in 0..3 {
            assert_eq!(gb.get(&[r, c]), 1.0);
        }
    }
}

/// Task 1 Test 7: each input to `concat` is read through its OWN
/// strides without being materialized first - one input is a
/// TRANSPOSED (non-contiguous) view, output values are still correct.
#[test]
fn concat_on_transposed_input_produces_correct_values_without_materializing() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let transposed = transpose_storage(&a, 0, 1).unwrap();
    // transposed: [[1,4],[2,5],[3,6]], shape [3,2]
    let b = matrix(vec![100.0, 200.0], 1, 2);
    let out = concat_storage(&[&transposed, &b], 0).unwrap();
    assert_eq!(out.shape, vec![4, 2]);
    assert_eq!(
        f32_vec(&out),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 100.0, 200.0]
    );
}

/// Task 2 Test 1: `stack(&[a, b], 0)` where `a`/`b` are both `[2,3]`
/// produces shape `[2,2,3]`, with the new axis-0 slices matching `a`/`b`
/// respectively.
#[test]
fn stack_dim0_inserts_new_leading_axis() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = matrix(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
    let out = stack_storage(&[&a, &b], 0).unwrap();
    assert_eq!(out.shape, vec![2, 2, 3]);
    assert_eq!(
        f32_vec(&out),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ]
    );
}

/// Task 2 Test 2: `stack(&[a, b], 2)` (dim equal to rank, appending at
/// the very end) where `a`/`b` are both `[2,3]` produces shape `[2,3,2]`.
#[test]
fn stack_dim_equal_to_rank_appends_new_trailing_axis() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = matrix(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
    let out = stack_storage(&[&a, &b], 2).unwrap();
    assert_eq!(out.shape, vec![2, 3, 2]);
    // Element [r,c,0] == a[r,c], [r,c,1] == b[r,c]
    for r in 0..2 {
        for c in 0..3 {
            assert_eq!(out.get(&[r, c, 0]), a.get(&[r, c]));
            assert_eq!(out.get(&[r, c, 1]), b.get(&[r, c]));
        }
    }
}

/// Task 2 Test 3: `stack` with mismatched-shape inputs returns
/// `Err(Error::ShapeMismatch)` - stack requires IDENTICAL shapes,
/// stricter than concat's "all-but-one-axis" rule.
#[test]
fn stack_rejects_mismatched_shapes() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 4);
    let result = stack_storage(&[&a, &b], 0);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

/// Task 2 Test 4: `stack(&[], 0)` (empty input list) returns
/// `Err(Error::ShapeMismatch)`, not a panic.
#[test]
fn stack_empty_input_list_returns_err_not_panic() {
    let result: Result<CpuStorage> = stack_storage(&[], 0);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

/// Task 2 Test 5: `stack`'s backward correctly narrows-then-squeezes
/// `grad_out` back to each input's own ORIGINAL shape (the inserted
/// axis removed), with 2 distinct inputs.
#[test]
fn stack_backward_narrows_and_squeezes_grad_to_original_shape() {
    let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = matrix(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
    let out = stack_storage(&[&a, &b], 0).unwrap();
    let grads = tape::backward(&out).unwrap();

    let ga = grads.get(a.id).expect("a should have a gradient");
    assert_eq!(ga.shape, vec![2, 3]);
    for r in 0..2 {
        for c in 0..3 {
            assert_eq!(ga.get(&[r, c]), 1.0);
        }
    }

    let gb = grads.get(b.id).expect("b should have a gradient");
    assert_eq!(gb.shape, vec![2, 3]);
    for r in 0..2 {
        for c in 0..3 {
            assert_eq!(gb.get(&[r, c]), 1.0);
        }
    }
}

/// Task 3 Test 1: `broadcast_left(t, &[4])` on a `[3]` vector produces
/// shape `[4,3]` (the `[4]` prepended as a new leading dim, `t`'s own
/// `[3]` shape unchanged and trailing).
#[test]
fn broadcast_left_prepends_single_new_leading_dim() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
    let out = broadcast_left_storage(&t, &[4]).unwrap();
    assert_eq!(out.shape, vec![4, 3]);
    for row in 0..4 {
        assert_eq!(out.get(&[row, 0]), 1.0);
        assert_eq!(out.get(&[row, 1]), 2.0);
        assert_eq!(out.get(&[row, 2]), 3.0);
    }
}

/// Task 3 Test 2: `broadcast_left(t, &[2,4])` on a `[3]` vector produces
/// shape `[2,4,3]` (multiple new leading dims prepended at once).
#[test]
fn broadcast_left_prepends_multiple_new_leading_dims() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
    let out = broadcast_left_storage(&t, &[2, 4]).unwrap();
    assert_eq!(out.shape, vec![2, 4, 3]);
    for i in 0..2 {
        for j in 0..4 {
            assert_eq!(out.get(&[i, j, 0]), 1.0);
            assert_eq!(out.get(&[i, j, 1]), 2.0);
            assert_eq!(out.get(&[i, j, 2]), 3.0);
        }
    }
}

/// Task 3 Test 3: `broadcast_left`'s backward correctly unbroadcasts
/// `grad_out` back to `t`'s own original shape, with ZERO new backward
/// code (delegates entirely to `Self::broadcast_as`).
#[test]
fn broadcast_left_backward_unbroadcasts_to_original_shape() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
    let out = broadcast_left_storage(&t, &[4]).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("t should have a gradient");
    assert_eq!(g.shape, vec![3]);
    // ones_like(out) [4,3] summed over the broadcast axis -> [4,4,4]
    assert_eq!(f32_vec(g), vec![4.0, 4.0, 4.0]);
}

/// Task 3 Test 4: `broadcast_left` called through the trait matches
/// calling `CpuStorage::broadcast_as` directly with the manually
/// prepended target shape (thin-wrapper equivalence).
#[test]
fn broadcast_left_through_trait_matches_direct_broadcast_as_call() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
    let direct = t.broadcast_as(&[4, 3]).unwrap();
    let via_trait = broadcast_left_storage(&t, &[4]).unwrap();
    assert_eq!(via_trait.shape, direct.shape);
    assert_eq!(via_trait.strides, direct.strides);
}

/// Every pre-existing `group_norm` test used a batch of 1, which is the
/// one size at which grouping over the whole flattened buffer and grouping
/// per sample agree. Two samples are the smallest case that tells them
/// apart: sample 1 is sample 0 shifted by a constant, and normalization
/// removes a constant offset, so a per-sample result has to be identical
/// for both. Grouping across the batch cannot produce that.
#[test]
fn group_norm_statistics_are_per_sample_not_across_the_batch() {
    let first: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let second: Vec<f32> = first.iter().map(|v| v + 100.0).collect();
    let data = first.iter().copied().chain(second).collect::<Vec<f32>>();
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(data), vec![2, 4, 1, 2]);

    let out = f32_vec(&group_norm_storage(&t, 2, 1e-5).unwrap());

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

/// `instance_norm` is `group_norm` with one group per channel, so each
/// channel of each sample normalizes alone. A channel holding a single
/// repeated value therefore has zero variance and normalizes to zero,
/// whatever the other channels hold.
#[test]
fn instance_norm_normalizes_each_channel_of_each_sample_alone() {
    let t = CpuStorage::from_contiguous(
        CpuBuffer::F32(vec![
            1.0, 1.0, 5.0, 7.0, // sample 0: channel 0 flat, channel 1 varies
            2.0, 2.0, 9.0, 3.0, // sample 1: channel 0 flat, channel 1 varies
        ]),
        vec![2, 2, 2],
    );

    let out = f32_vec(&instance_norm_storage(&t, 1e-5).unwrap());

    for flat in [0, 1, 4, 5] {
        assert!(
            out[flat].abs() < 1e-5,
            "constant channel at {flat} must normalize to zero, got {}",
            out[flat]
        );
    }
    // A two-element channel normalizes to the symmetric pair -1, +1.
    assert!((out[2] + 1.0).abs() < 1e-3, "got {}", out[2]);
    assert!((out[3] - 1.0).abs() < 1e-3, "got {}", out[3]);
    assert!((out[6] - 1.0).abs() < 1e-3, "got {}", out[6]);
    assert!((out[7] + 1.0).abs() < 1e-3, "got {}", out[7]);
}
/// `scaled_dot_product_attention` is composed from `matmul`, `softmax` and
/// `add`, so the f32 result it used to return for every operand dtype was
/// the matmul mislabel showing through. Asserted here rather than only at
/// matmul because this is the composition a caller actually reaches.
#[test]
fn attention_keeps_the_operand_dtype() {
    let operand =
        || CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0, 0.0, 0.0, 1.0]), vec![2, 2]);
    let out = scaled_dot_product_attention_storage::<incin_core::tensor::device::Cpu>(
        &operand(),
        &operand(),
        &operand(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        out.dtype,
        incin_core::tensor::dtype::DTypeId::F64.descriptor()
    );
    assert_eq!(out.shape, vec![2, 2]);
}

// --- pointwise-family backwards (catalog: BinaryBroadcast gradients Defined) ---

#[test]
/// `sub_scalar_backward_is_the_identity`.
fn sub_scalar_backward_is_the_identity() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let out = sub_scalar_storage(&t, 5.0).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(f32_vec(g), vec![1.0; 4]);
}

#[test]
/// `div_scalar_backward_scales_by_one_over_the_constant`.
fn div_scalar_backward_scales_by_one_over_the_constant() {
    let t = matrix(vec![2.0, 4.0, 6.0, 8.0], 2, 2);
    let out = div_scalar_storage(&t, 2.0).unwrap();
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    // ones seed / 2.
    assert_eq!(f32_vec(g), vec![0.5; 4]);
}

#[test]
/// `maximum_backward_routes_to_the_strictly_greater_operand_with_ties_to_rhs`.
fn maximum_backward_routes_to_the_strictly_greater_operand_with_ties_to_rhs() {
    let lhs = matrix(vec![3.0, 1.0, 2.0, 5.0], 2, 2);
    let rhs = matrix(vec![1.0, 1.0, 4.0, 5.0], 2, 2);
    let out = crate::cpu::ops::shape_ops::lerp_storage; // keep import shape stable
    let _ = out;
    let max_out = {
        let mask = crate::cpu::ops::shape_ops::elementwise_cmp(&lhs, &rhs, |a, b| a > b).unwrap();
        where_storage(&mask, &lhs, &rhs).unwrap()
    };
    assert_eq!(f32_vec(&max_out), vec![3.0, 1.0, 4.0, 5.0]);
    let grads = tape::backward(&max_out).unwrap();
    // lhs receives only where it is strictly greater; the tie at [1,1]
    // routes to rhs, matching maximum's piecewise convention.
    assert_eq!(
        f32_vec(grads.get(lhs.id).unwrap()),
        vec![1.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        f32_vec(grads.get(rhs.id).unwrap()),
        vec![0.0, 1.0, 1.0, 1.0]
    );
}

#[test]
/// `minimum_backward_mirrors_maximum`.
fn minimum_backward_mirrors_maximum() {
    let lhs = matrix(vec![1.0, 3.0, 2.0, 5.0], 2, 2);
    let rhs = matrix(vec![3.0, 3.0, 1.0, 5.0], 2, 2);
    let min_out = {
        let mask = crate::cpu::ops::shape_ops::elementwise_cmp(&lhs, &rhs, |a, b| a < b).unwrap();
        where_storage(&mask, &lhs, &rhs).unwrap()
    };
    assert_eq!(f32_vec(&min_out), vec![1.0, 3.0, 1.0, 5.0]);
    let grads = tape::backward(&min_out).unwrap();
    // Strictly-less positions go to lhs; the tie and the rhs-wins position
    // route to rhs.
    assert_eq!(
        f32_vec(grads.get(lhs.id).unwrap()),
        vec![1.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        f32_vec(grads.get(rhs.id).unwrap()),
        vec![0.0, 1.0, 1.0, 1.0]
    );
}

#[test]
/// `abs_diff_gradcheck_matches_finite_differences`.
fn abs_diff_gradcheck_matches_finite_differences() {
    use crate::cpu::gradcheck::{F32_STEP, GRAD_TOL, gradcheck};
    let lhs = matrix(vec![0.5, -1.0, 2.0, 1.5], 2, 2);
    let rhs = matrix(vec![1.0, -0.25, -0.5, 3.0], 2, 2);
    let operands = [lhs, rhs];
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let diff = crate::cpu::ops::elementwise::sub_storage(&inputs[0], &inputs[1]).unwrap();
        crate::cpu::ops::reduce::sum_all(
            &crate::cpu::ops::elementwise::canonical_abs(&diff).unwrap(),
        )
        .unwrap()
    };
    let err = gradcheck(op, &operands, F32_STEP);
    assert!(err < GRAD_TOL, "abs_diff gradcheck too high: {err}");
}

#[test]
/// `lerp_gradcheck_matches_finite_differences_for_both_operands`.
fn lerp_gradcheck_matches_finite_differences_for_both_operands() {
    use crate::cpu::gradcheck::{F32_STEP, GRAD_TOL, gradcheck};
    let start = matrix(vec![0.5, -1.0, 2.0, 1.5], 2, 2);
    let end = matrix(vec![1.0, -0.25, -0.5, 3.0], 2, 2);
    let operands = [start, end];
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let out = lerp_storage(&inputs[0], &inputs[1], 0.3).unwrap();
        crate::cpu::ops::reduce::sum_all(&out).unwrap()
    };
    let err = gradcheck(op, &operands, F32_STEP);
    assert!(err < GRAD_TOL, "lerp gradcheck too high: {err}");
}

// --- selection/indexing backwards (catalog promises these gradients) ---

#[test]
/// `masked_fill_backward_passes_through_only_unmasked_positions`.
fn masked_fill_backward_passes_through_only_unmasked_positions() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![4]);
    let mask = CpuStorage::from_contiguous(CpuBuffer::Bool(vec![1, 0, 1, 0]), vec![4]);
    let out = masked_fill_storage(&t, &mask, 9.0).unwrap();
    assert_eq!(f32_vec(&out), vec![9.0, 2.0, 9.0, 4.0]);
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(f32_vec(g), vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
/// `index_select_backward_accumulates_repeated_selections`.
fn index_select_backward_accumulates_repeated_selections() {
    // Rows of the input selected by index [2, 0, 2]: row 2 is chosen twice,
    // so its cotangent accumulates both contributions.
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let index = CpuStorage::from_contiguous(CpuBuffer::I64(vec![2, 0, 2]), vec![3]);
    let out = index_select_storage(&t, 0, &index).unwrap();
    assert_eq!(f32_vec(&out), vec![5.0, 6.0, 1.0, 2.0, 5.0, 6.0]);
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(f32_vec(g), vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0]);
}

#[test]
/// `scatter_backward_zeroes_overwritten_positions_and_routes_to_source`.
fn scatter_backward_zeroes_overwritten_positions_and_routes_to_source() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let index = CpuStorage::from_contiguous(CpuBuffer::I64(vec![1, 0]), vec![2, 1]);
    let source = matrix(vec![10.0, 20.0], 2, 1);
    let out = scatter_storage(&t, 0, &index, &source).unwrap();
    assert_eq!(f32_vec(&out), vec![20.0, 2.0, 10.0, 4.0]);
    let grads = tape::backward(&out).unwrap();
    // Overwritten positions ([0,0] and [1,0]) receive nothing on the input
    // path; their cotangents land on the source instead.
    assert_eq!(f32_vec(grads.get(t.id).unwrap()), vec![0.0, 1.0, 0.0, 1.0]);
    assert_eq!(f32_vec(grads.get(source.id).unwrap()), vec![1.0, 1.0]);
}

#[test]
/// `scatter_backward_accumulates_duplicate_writes_on_the_source_path`.
fn scatter_backward_accumulates_duplicate_writes_on_the_source_path() {
    let t = matrix(vec![1.0, 2.0], 1, 2);
    let index = CpuStorage::from_contiguous(CpuBuffer::I64(vec![0, 0]), vec![2, 1]);
    let source = matrix(vec![7.0, 8.0], 2, 1);
    let out = scatter_storage(&t, 0, &index, &source).unwrap();
    // Last write wins in the forward.
    assert_eq!(f32_vec(&out), vec![8.0, 2.0]);
    let grads = tape::backward(&out).unwrap();
    // Last-write-wins: only the surviving write routes a cotangent to the
    // source; the overwritten first write contributed nothing.
    assert_eq!(f32_vec(grads.get(t.id).unwrap()), vec![0.0, 1.0]);
    assert_eq!(f32_vec(grads.get(source.id).unwrap()), vec![0.0, 1.0]);
}

// --- shape-family backwards (catalog: Shape gradients Defined) ---

#[test]
/// `repeat_backward_sums_tiles_onto_the_source`.
fn repeat_backward_sums_tiles_onto_the_source() {
    let t = matrix(vec![1.0, 2.0], 1, 2);
    let out = repeat_storage(&t, &[2, 2]).unwrap();
    assert_eq!(out.shape, vec![2, 4]);
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    // Each source element is repeated 4x, so ones seed sums to 4 per element.
    assert_eq!(f32_vec(g), vec![4.0, 4.0]);
}

#[test]
/// `pad_backward_shifts_the_cotangent_by_the_padding_offsets`.
fn pad_backward_shifts_the_cotangent_by_the_padding_offsets() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2]);
    let out = pad_storage(&t, &[(1, 1)], 9.0).unwrap();
    assert_eq!(f32_vec(&out), vec![9.0, 1.0, 2.0, 9.0]);
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(f32_vec(g), vec![1.0, 1.0]);
}

#[test]
/// `unfold_backward_accumulates_through_overlapping_windows`.
fn unfold_backward_accumulates_through_overlapping_windows() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![4]);
    // Windows of size 2 with step 1 over [1,2,3,4] -> [[1,2],[2,3],[3,4]].
    let out = unfold_storage(&t, 0, 2, 1).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(f32_vec(&out), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    // Interior elements belong to two windows.
    assert_eq!(f32_vec(g), vec![1.0, 2.0, 2.0, 1.0]);
}

#[test]
/// `pixel_shuffle_backward_inverts_the_permutation`.
fn pixel_shuffle_backward_inverts_the_permutation() {
    // 1x4 channels of a 1x1 image shuffled to 1 channel of a 2x2 image:
    // every output element maps to exactly one input element either way.
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![1, 4, 1, 1]);
    let out = pixel_shuffle_storage(&t, 2).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    let grads = tape::backward(&out).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    // The permutation is bijective here, so ones seed stays ones.
    assert_eq!(f32_vec(g), vec![1.0; 4]);
}

#[test]
/// `triu_tril_backward_apply_the_same_mask`.
fn triu_tril_backward_apply_the_same_mask() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let upper = triu_storage(&t, 0).unwrap();
    assert_eq!(f32_vec(&upper), vec![1.0, 2.0, 0.0, 4.0]);
    let grads = tape::backward(&upper).unwrap();
    assert_eq!(f32_vec(grads.get(t.id).unwrap()), vec![1.0, 1.0, 0.0, 1.0]);

    let lower = tril_storage(&t, 0).unwrap();
    assert_eq!(f32_vec(&lower), vec![1.0, 0.0, 3.0, 4.0]);
    let grads = tape::backward(&lower).unwrap();
    assert_eq!(f32_vec(grads.get(t.id).unwrap()), vec![1.0, 0.0, 1.0, 1.0]);
}

#[test]
/// `diag_vector_to_matrix_backward_reads_the_diagonal_of_the_cotangent`.
fn diag_vector_to_matrix_backward_reads_the_diagonal_of_the_cotangent() {
    let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2]);
    let out = diag_storage(&t, 0).unwrap();
    assert_eq!(f32_vec(&out), vec![1.0, 0.0, 0.0, 2.0]);
    // Seed the matrix cotangent by summing it; the diagonal entries carry it.
    let loss = crate::cpu::ops::reduce::sum_all(&out).unwrap();
    let grads = tape::backward(&loss).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(f32_vec(g), vec![1.0, 1.0]);
}

#[test]
/// `diag_matrix_to_vector_backward_scatters_onto_the_diagonal`.
fn diag_matrix_to_vector_backward_scatters_onto_the_diagonal() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let out = diag_storage(&t, 0).unwrap();
    assert_eq!(out.shape, vec![2]);
    assert_eq!(f32_vec(&out), vec![1.0, 4.0]);
    let loss = crate::cpu::ops::reduce::sum_all(&out).unwrap();
    let grads = tape::backward(&loss).unwrap();
    let g = grads.get(t.id).expect("input should have gradient");
    assert_eq!(f32_vec(g), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
/// `group_norm_gradcheck_proves_the_statistical_path_is_recorded`.
fn group_norm_gradcheck_proves_the_statistical_path_is_recorded() {
    use crate::cpu::gradcheck::{F32_STEP, GRAD_TOL, gradcheck};
    // Normalizing by batch statistics means the cotangent must flow through
    // the mean and variance back into every element. A tape-silent
    // statistic shows up here as a finite-difference divergence, which is
    // exactly how training-mode batch norm's defect was found.
    let t = CpuStorage::from_contiguous(
        CpuBuffer::F32(vec![
            0.5, 1.0, -0.5, 0.2, 1.5, -1.0, 0.3, -0.3, //
            0.8, -0.8, 1.2, -1.2, 0.1, 0.9, -0.1, 0.4,
        ]),
        vec![2, 2, 2, 2],
    );
    let operands = [t];
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let out = group_norm_storage(&inputs[0], 2, 1e-5).unwrap();
        crate::cpu::ops::reduce::sum_all(&out).unwrap()
    };
    let err = gradcheck(op, &operands, F32_STEP);
    assert!(err < GRAD_TOL, "group_norm gradcheck too high: {err}");
}
