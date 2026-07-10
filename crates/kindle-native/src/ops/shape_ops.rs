//! `TensorOps` for `NativeBackend<T, D>`: real `reshape`/`transpose`/
//! `broadcast_as`/`matmul`/`float_to_scalar`/`float_to_vec1`; every other
//! method is a typed stub returning `Error::UnsupportedBackendOperation`.
//!
//! This is the single `impl TensorOps<..> for NativeBackend<..>` block for
//! the whole crate — `matmul`'s method body delegates to
//! `ops::matmul::matmul_impl` (see that file's module doc for why the naive
//! loop lives in its own file as a plain function rather than its own impl
//! block). `reshape`/`transpose`/`broadcast_as` are thin wrappers over
//! `NativeStorage`'s own already-O(1) view methods (Plan 01) — they do not
//! duplicate that logic, only add tape tracking (D-05: every op is a graph
//! node, unconditionally recorded).

use kindle_core::err::Error;
use kindle_core::prelude::{Backend, DType, KindleDType, Result, TensorOps};

use crate::NativeBackend;
use crate::ops::matmul::{batched_matmul_impl, matmul_impl};
use crate::storage::NativeStorage;
use crate::tape::{self, TapeEntry};

impl<T: DType, D: kindle_core::prelude::Device> TensorOps<Self> for NativeBackend<T, D> {
    fn reshape<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = t.reshape(shape)?;

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![
                    grad_out
                        .reshape(&original_shape)
                        .expect("reshape backward: grad_out reshape to original shape cannot fail (same element count)"),
                ]
            }),
        });
        Ok(out)
    }

    fn transpose<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = t.transpose(dim1, dim2)?;

        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            // Transposing the same two axes twice is idempotent, so the
            // backward closure is the same transpose applied to grad_out.
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![
                    grad_out
                        .transpose(dim1, dim2)
                        .expect("transpose backward: re-applying the same transpose cannot fail"),
                ]
            }),
        });
        Ok(out)
    }

    fn broadcast_as<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = t.broadcast_as(shape)?;

        let original_shape = t.shape.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![
                    tape::unbroadcast(grad_out, &original_shape)
                        .expect("broadcast_as backward: unbroadcast to original shape"),
                ]
            }),
        });
        Ok(out)
    }

    fn matmul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if lhs.shape.len() == 2 && rhs.shape.len() == 2 {
            matmul_impl(lhs, rhs)
        } else {
            batched_matmul_impl(lhs, rhs)
        }
    }

    fn narrow<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = t.narrow(dim, start, len)?;

        let original_shape = t.shape.clone();
        let mut region_start = vec![0usize; original_shape.len()];
        region_start[dim] = start;
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![crate::storage::scatter_into_zeros(
                    &original_shape,
                    &region_start,
                    grad_out,
                )]
            }),
        });
        Ok(out)
    }

    fn squeeze<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() || t.shape[dim] != 1 {
            return Err(Error::ShapeMismatch {
                op: "squeeze",
                expected: vec![1],
                got: t.shape.clone(),
                msg: format!(
                    "squeeze requires axis {dim} to have size 1, got size {} in shape {:?}",
                    t.shape.get(dim).copied().unwrap_or(0),
                    t.shape
                ),
            });
        }

        let mut target_shape = t.shape.clone();
        target_shape.remove(dim);
        Self::reshape::<K>(t, &target_shape)
    }

    fn stack<K: DType>(
        _t: &[&<Self as Backend>::Storage<K>],
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "stack",
            backend: "Native",
        })
    }

    fn concat<K: DType>(
        _t: &[&<Self as Backend>::Storage<K>],
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "concat",
            backend: "Native",
        })
    }

    fn slice<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut out = t.clone();
        for (dim, &(start, end)) in ranges.iter().enumerate() {
            out = Self::narrow::<K>(&out, dim, start, end - start)?;
        }
        Ok(out)
    }

    fn flatten<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if start_dim > end_dim || end_dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "flatten",
                expected: t.shape.clone(),
                got: vec![start_dim, end_dim],
                msg: format!(
                    "flatten(start_dim={start_dim}, end_dim={end_dim}) out of bounds for shape {:?}",
                    t.shape
                ),
            });
        }

        let merged: usize = t.shape[start_dim..=end_dim].iter().product();
        let mut target_shape = t.shape[..start_dim].to_vec();
        target_shape.push(merged);
        target_shape.extend_from_slice(&t.shape[end_dim + 1..]);

        Self::reshape::<K>(t, &target_shape)
    }

    fn broadcast_left<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "broadcast_left",
            backend: "Native",
        })
    }

    fn float_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<f64> {
        if t.shape.iter().product::<usize>() != 1 {
            return Err(Error::ShapeMismatch {
                op: "float_to_scalar",
                expected: vec![1],
                got: t.shape.clone(),
                msg: "float_to_scalar requires a single-element tensor".to_string(),
            });
        }
        Ok(t.get(&vec![0usize; t.shape.len()]))
    }

    fn float_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<std::vec::Vec<f64>> {
        if t.shape.len() != 1 {
            return Err(Error::UnsupportedBackendOperation {
                op: "float_to_vec1",
                backend: "Native",
            });
        }
        Ok((0..t.shape[0]).map(|i| t.get(&[i])).collect())
    }

    fn int_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<i64> {
        Err(Error::UnsupportedBackendOperation {
            op: "int_to_scalar",
            backend: "Native",
        })
    }

    fn int_to_vec1<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<std::vec::Vec<i64>> {
        Err(Error::UnsupportedBackendOperation {
            op: "int_to_vec1",
            backend: "Native",
        })
    }

    fn tensor_to_dtype<K: DType, K2: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dtype: KindleDType,
    ) -> Result<<Self as Backend>::Storage<K2>> {
        Err(Error::UnsupportedBackendOperation {
            op: "tensor_to_dtype",
            backend: "Native",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::NativeBuffer;

    type TestBackend = NativeBackend<f32, kindle_core::prelude::Cpu>;

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
    fn reshape_through_trait_matches_direct_storage_call() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let direct = t.reshape(&[3, 2]).unwrap();
        let via_trait = TestBackend::reshape::<f32>(&t, &[3, 2]).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }

    #[test]
    fn transpose_through_trait_matches_direct_storage_call() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let direct = t.transpose(0, 1).unwrap();
        let via_trait = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(via_trait.strides, direct.strides);
    }

    #[test]
    fn broadcast_as_through_trait_matches_direct_storage_call() {
        let t = NativeStorage::from_contiguous(NativeBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
        let direct = t.broadcast_as(&[4, 3]).unwrap();
        let via_trait = TestBackend::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(via_trait.strides, direct.strides);
    }

    #[test]
    fn float_to_scalar_reads_single_element() {
        let t = NativeStorage::from_contiguous(NativeBuffer::F32(vec![42.0]), vec![]);
        let v = TestBackend::float_to_scalar::<f32>(&t).unwrap();
        assert_eq!(v, 42.0);
    }

    #[test]
    fn float_to_vec1_reads_all_elements_row_major() {
        let t = NativeStorage::from_contiguous(NativeBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
        let v = TestBackend::float_to_vec1::<f32>(&t).unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn reshape_backward_reshapes_grad_back_to_original_shape() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = TestBackend::reshape::<f32>(&t, &[6]).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![2, 3]);
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    #[test]
    fn transpose_backward_reapplies_same_transpose() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![2, 3]);
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    #[test]
    fn broadcast_as_backward_unbroadcasts_to_original_shape() {
        let t = NativeStorage::from_contiguous(NativeBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
        let out = TestBackend::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![1, 3]);
        // ones_like(out) [4,3] summed over the broadcast axis -> [4,4,4]
        assert_eq!(f32_vec(g), vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn matmul_via_trensor_ops_delegates_to_matmul_impl() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = matrix(
            vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
            3,
            4,
        );
        let out = TestBackend::matmul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        assert_eq!(
            f32_vec(&out),
            vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
        );
    }

    #[test]
    fn unsupported_methods_return_typed_error_not_silent_placeholder() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = TestBackend::stack::<f32>(&[&t], 0);
        assert!(matches!(
            result,
            Err(Error::UnsupportedBackendOperation { op: "stack", .. })
        ));
    }

    /// Task 1 Test 1: `TensorOps::narrow` called through the trait matches
    /// calling `NativeStorage::narrow` directly (thin-wrapper equivalence).
    #[test]
    fn narrow_through_trait_matches_direct_storage_call() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let direct = t.narrow(0, 1, 1).unwrap();
        let via_trait = TestBackend::narrow::<f32>(&t, 0, 1, 1).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }

    /// Task 1 Test 2: `narrow`'s backward zero-pads `grad_out` back to the
    /// original shape at the correct region.
    #[test]
    fn narrow_backward_zero_pads_grad_to_original_shape() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let out = TestBackend::narrow::<f32>(&t, 0, 1, 1).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![3, 2]);
        assert_eq!(f32_vec(g), vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    /// Task 1 Test 3: out-of-bounds narrow range returns `Err`, not a panic.
    #[test]
    fn narrow_out_of_bounds_returns_err_not_panic() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let result = TestBackend::narrow::<f32>(&t, 0, 2, 2);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 1 Test 4: `narrow`'s forward value on a pre-transposed
    /// (non-contiguous) input still produces correct values.
    #[test]
    fn narrow_on_transposed_input_produces_correct_values_without_materializing() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let transposed = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        // transposed is logically [[1,4],[2,5],[3,6]], shape [3,2]
        let narrowed = TestBackend::narrow::<f32>(&transposed, 0, 1, 1).unwrap();
        assert_eq!(narrowed.shape, vec![1, 2]);
        assert_eq!(narrowed.get(&[0, 0]), 2.0);
        assert_eq!(narrowed.get(&[0, 1]), 5.0);
    }

    /// Task 2 Test 1: `slice(t, &[(1,3),(0,2)])` on a `[4,3]` matrix matches
    /// manually narrowing dim 0 to `(1,3)` then dim 1 to `(0,2)` in sequence.
    #[test]
    fn slice_matches_manual_sequential_narrow_calls() {
        let t = matrix(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            4,
            3,
        );
        let manual = TestBackend::narrow::<f32>(&t, 0, 1, 2).unwrap();
        let manual = TestBackend::narrow::<f32>(&manual, 1, 0, 2).unwrap();

        let via_slice = TestBackend::slice::<f32>(&t, &[(1, 3), (0, 2)]).unwrap();
        assert_eq!(via_slice.shape, manual.shape);
        assert_eq!(f32_vec(&via_slice), f32_vec(&manual));
    }

    /// Task 2 Test 2: `slice` on a pre-transposed (non-contiguous) input,
    /// across multiple dims, produces correct values without a
    /// `.contiguous()` call happening internally.
    #[test]
    fn slice_on_transposed_input_across_multiple_dims_produces_correct_values() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let transposed = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        // transposed: [[1,4],[2,5],[3,6]], shape [3,2]
        // slice rows [1,3) and cols [0,1) -> [[2],[3]]
        let out = TestBackend::slice::<f32>(&transposed, &[(1, 3), (0, 1)]).unwrap();
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.get(&[0, 0]), 2.0);
        assert_eq!(out.get(&[1, 0]), 3.0);
    }

    /// Task 2 Test 3: `slice`'s backward correctly zero-pads back to the
    /// original shape, composed entirely from `narrow`'s own backward.
    #[test]
    fn slice_backward_zero_pads_grad_to_original_shape() {
        let t = matrix(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            4,
            3,
        );
        let out = TestBackend::slice::<f32>(&t, &[(1, 3), (0, 2)]).unwrap();
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
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            4,
            3,
        );
        let result = TestBackend::slice::<f32>(&t, &[(1, 3), (0, 5)]);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    fn tensor3(v: Vec<f32>, d0: usize, d1: usize, d2: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![d0, d1, d2])
    }

    /// Task 3 Test 1: `squeeze(t, 1)` on a `[3,1,4]` storage produces shape
    /// `[3,4]` with unchanged (row-major) values.
    #[test]
    fn squeeze_removes_size_one_axis_and_preserves_values() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let t = tensor3(data.clone(), 3, 1, 4);
        let out = TestBackend::squeeze::<f32>(&t, 1).unwrap();
        assert_eq!(out.shape, vec![3, 4]);
        assert_eq!(f32_vec(&out), data);
    }

    /// Task 3 Test 2: `squeeze(t, 0)` on a `[3,1,4]` storage (dim 0 has size
    /// 3, not 1) returns a clear squeeze-specific `Error::ShapeMismatch`.
    #[test]
    fn squeeze_on_non_one_sized_axis_returns_shape_mismatch() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let t = tensor3(data, 3, 1, 4);
        let result = TestBackend::squeeze::<f32>(&t, 0);
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
        let out = TestBackend::squeeze::<f32>(&t, 1).unwrap();
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
        let out = TestBackend::flatten::<f32>(&t, 1, 2).unwrap();
        assert_eq!(out.shape, vec![2, 12]);
        assert_eq!(f32_vec(&out), data);
    }

    /// Task 3 Test 5: `flatten(t, 0, 2)` on a `[2,3,4]` storage (flattening
    /// all dims) produces shape `[24]`.
    #[test]
    fn flatten_all_dims_produces_1d_shape() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let t = tensor3(data.clone(), 2, 3, 4);
        let out = TestBackend::flatten::<f32>(&t, 0, 2).unwrap();
        assert_eq!(out.shape, vec![24]);
        assert_eq!(f32_vec(&out), data);
    }

    /// Task 3 Test 6: `flatten`'s backward reshapes `grad_out` back to the
    /// original shape, delegated entirely to `reshape`'s backward.
    #[test]
    fn flatten_backward_reshapes_grad_to_original_shape() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let t = tensor3(data, 2, 3, 4);
        let out = TestBackend::flatten::<f32>(&t, 1, 2).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![2, 3, 4]);
        assert_eq!(f32_vec(g), vec![1.0; 24]);
    }

    /// Test 6: `TensorOps::matmul` called through the trait on two rank-2
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
        let via_trait = TestBackend::matmul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }

    /// Test 7: `TensorOps::matmul` called through the trait on two rank-3
    /// (or higher) operands correctly dispatches to `batched_matmul_impl`
    /// and produces the same values a direct `batched_matmul_impl` call
    /// would.
    #[test]
    fn matmul_dispatch_rank3_matches_batched_matmul_impl_directly() {
        let lhs_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let rhs_data: Vec<f32> = (1..=40).map(|x| x as f32).collect();
        let lhs = NativeStorage::from_contiguous(NativeBuffer::F32(lhs_data), vec![2, 3, 4]);
        let rhs = NativeStorage::from_contiguous(NativeBuffer::F32(rhs_data), vec![2, 4, 5]);

        let direct = batched_matmul_impl(&lhs, &rhs).unwrap();
        let via_trait = TestBackend::matmul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }
}
