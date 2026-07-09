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
use crate::ops::matmul::matmul_impl;
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
        matmul_impl(lhs, rhs)
    }

    fn narrow<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
        _start: usize,
        _len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "narrow",
            backend: "Native",
        })
    }

    fn squeeze<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "squeeze",
            backend: "Native",
        })
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
        _t: &<Self as Backend>::Storage<K>,
        _ranges: &[(usize, usize)],
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "slice",
            backend: "Native",
        })
    }

    fn flatten<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _start_dim: usize,
        _end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "flatten",
            backend: "Native",
        })
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
        let result = TestBackend::narrow::<f32>(&t, 0, 0, 1);
        assert!(matches!(
            result,
            Err(Error::UnsupportedBackendOperation { op: "narrow", .. })
        ));
    }
}
