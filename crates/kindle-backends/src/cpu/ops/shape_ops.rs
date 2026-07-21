//! `TensorOps` for `CpuBackend<T, D>`: real `reshape`/`transpose`/
//! `broadcast_as`/`matmul`/`float_to_scalar`/`float_to_vec1`; every other
//! method is a typed stub returning `Error::UnsupportedBackendOperation`.
//!
//! This is the single `impl TensorOps<..> for CpuBackend<..>` block for
//! the whole crate — `matmul`'s method body delegates to
//! `ops::matmul::matmul_impl` (see that file's module doc for why the naive
//! loop lives in its own file as a plain function rather than its own impl
//! block). `reshape`/`transpose`/`broadcast_as` are thin wrappers over
//! `CpuStorage`'s own already-O(1) view methods (Plan 01) — they do not
//! duplicate that logic, only add tape tracking (D-05: every op is a graph
//! node, unconditionally recorded).

use kindle_core::prelude::Error;
use kindle_core::prelude::{Backend, DType, KindleDType, Result, TensorOps};

use crate::cpu::CpuBackend;
use crate::cpu::ops::matmul::{batched_matmul_impl, matmul_impl};
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::tape::{self, TapeEntry};

impl<T: DType, D: kindle_core::prelude::Device> TensorOps<Self> for CpuBackend<T, D> {
    /// `reshape`.
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
            backward: Box::new(move |grad_out: &CpuStorage| {
                vec![
                    grad_out
                        .reshape(&original_shape)
                        .expect("reshape backward: grad_out reshape to original shape cannot fail (same element count)"),
                ]
            }),
        });
        Ok(out)
    }

    /// `transpose`.
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
            backward: Box::new(move |grad_out: &CpuStorage| {
                vec![
                    grad_out
                        .transpose(dim1, dim2)
                        .expect("transpose backward: re-applying the same transpose cannot fail"),
                ]
            }),
        });
        Ok(out)
    }

    /// `broadcast_as`.
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
            backward: Box::new(move |grad_out: &CpuStorage| {
                vec![
                    tape::unbroadcast(grad_out, &original_shape)
                        .expect("broadcast_as backward: unbroadcast to original shape"),
                ]
            }),
        });
        Ok(out)
    }

    /// `matmul`.
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

    /// `narrow`.
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
            backward: Box::new(move |grad_out: &CpuStorage| {
                vec![crate::cpu::storage::scatter_into_zeros(
                    &original_shape,
                    &region_start,
                    grad_out,
                )]
            }),
        });
        Ok(out)
    }

    /// `squeeze`.
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

    /// `stack`.
    fn stack<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::ShapeMismatch {
                op: "stack",
                expected: vec![],
                got: vec![],
                msg: alloc::string::String::from("stack requires at least one input tensor"),
            });
        }

        let rank = tensors[0].shape.len();
        if dim > rank {
            return Err(Error::ShapeMismatch {
                op: "stack",
                expected: tensors[0].shape.clone(),
                got: vec![dim],
                msg: format!(
                    "stack dim {dim} out of range for rank-{rank} shape {:?} (dim may equal rank to append at the end)",
                    tensors[0].shape
                ),
            });
        }

        for t in tensors.iter().skip(1) {
            if t.shape != tensors[0].shape {
                return Err(Error::ShapeMismatch {
                    op: "stack",
                    expected: tensors[0].shape.clone(),
                    got: t.shape.clone(),
                    msg: format!(
                        "stack requires every input to have an IDENTICAL shape; expected {:?}, got {:?}",
                        tensors[0].shape, t.shape
                    ),
                });
            }
        }

        // Unsqueeze each input by reshaping to a target shape with a new
        // size-1 axis spliced in at `dim` (the TensorOps trait has no
        // dedicated `unsqueeze` method), then delegate to Self::concat —
        // this composition needs zero new backward code: reshape's and
        // concat's own tape entries compose correctly on their own.
        let mut unsqueezed = Vec::with_capacity(tensors.len());
        for t in tensors.iter() {
            let mut target_shape = t.shape.clone();
            target_shape.insert(dim, 1);
            unsqueezed.push(Self::reshape::<K>(t, &target_shape)?);
        }

        let refs: Vec<&<Self as Backend>::Storage<K>> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    /// `concat`.
    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::ShapeMismatch {
                op: "concat",
                expected: vec![],
                got: vec![],
                msg: alloc::string::String::from("concat requires at least one input tensor"),
            });
        }

        let rank = tensors[0].shape.len();
        if dim >= rank {
            return Err(Error::ShapeMismatch {
                op: "concat",
                expected: tensors[0].shape.clone(),
                got: vec![dim],
                msg: format!(
                    "concat dim {dim} out of range for rank-{rank} shape {:?}",
                    tensors[0].shape
                ),
            });
        }

        for t in tensors.iter().skip(1) {
            if t.shape.len() != rank {
                return Err(Error::ShapeMismatch {
                    op: "concat",
                    expected: tensors[0].shape.clone(),
                    got: t.shape.clone(),
                    msg: format!(
                        "concat requires every input to have the same rank; expected rank {rank}, got shape {:?}",
                        t.shape
                    ),
                });
            }
            // Every axis EXCEPT `dim` must match EXACTLY — never
            // broadcast-compatible (Pitfall 5: a size-1-vs-larger mismatch
            // here must be REJECTED, not silently accepted the way
            // stride::broadcast_shape would treat it).
            for (axis, (&a, &b)) in tensors[0].shape.iter().zip(t.shape.iter()).enumerate() {
                if axis != dim && a != b {
                    return Err(Error::ShapeMismatch {
                        op: "concat",
                        expected: tensors[0].shape.clone(),
                        got: t.shape.clone(),
                        msg: format!(
                            "concat requires exact equality on every non-concat axis; axis {axis} has size {a} vs {b}"
                        ),
                    });
                }
            }
        }

        let mut out_shape = tensors[0].shape.clone();
        out_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();
        let out_strides = crate::cpu::stride::contiguous_strides(&out_shape);
        let total: usize = out_shape.iter().product();

        // Cumulative offset of each input along `dim`, needed by both the
        // forward copy and the backward narrow-based scatter.
        let mut cumulative_offsets = Vec::with_capacity(tensors.len());
        let mut running = 0usize;
        for t in tensors.iter() {
            cumulative_offsets.push(running);
            running += t.shape[dim];
        }

        macro_rules! concat_variant {
            ($variant:ident, $ty:ty) => {{
                let mut out: Vec<$ty> = vec![Default::default(); total];
                for (t, &offset) in tensors.iter().zip(cumulative_offsets.iter()) {
                    // Read this input through ITS OWN strides directly — no
                    // prior `.contiguous()` materialization.
                    let value_count: usize = t.shape.iter().product();
                    let mut multi_idx = vec![0usize; t.shape.len()];
                    for _ in 0..value_count {
                        let mut flat_dest = 0usize;
                        for (axis, &i) in multi_idx.iter().enumerate() {
                            let dest_i = if axis == dim { i + offset } else { i };
                            flat_dest += dest_i * out_strides[axis];
                        }
                        out[flat_dest] = t.get(&multi_idx) as $ty;
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::$variant(out)
            }};
        }

        let new_buffer = match &*tensors[0].buffer {
            CpuBuffer::F32(_) => concat_variant!(F32, f32),
            CpuBuffer::F64(_) => concat_variant!(F64, f64),
            CpuBuffer::U8(_) => concat_variant!(U8, u8),
            CpuBuffer::U32(_) => concat_variant!(U32, u32),
            CpuBuffer::I64(_) => concat_variant!(I64, i64),
            CpuBuffer::F16(_) => {
                let mut out: Vec<half::f16> = vec![half::f16::from_f64(0.0); total];
                for (t, &offset) in tensors.iter().zip(cumulative_offsets.iter()) {
                    let value_count: usize = t.shape.iter().product();
                    let mut multi_idx = vec![0usize; t.shape.len()];
                    for _ in 0..value_count {
                        let mut flat_dest = 0usize;
                        for (axis, &i) in multi_idx.iter().enumerate() {
                            let dest_i = if axis == dim { i + offset } else { i };
                            flat_dest += dest_i * out_strides[axis];
                        }
                        out[flat_dest] = half::f16::from_f64(t.get(&multi_idx));
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::F16(out)
            }
            CpuBuffer::BF16(_) => {
                let mut out: Vec<half::bf16> = vec![half::bf16::from_f64(0.0); total];
                for (t, &offset) in tensors.iter().zip(cumulative_offsets.iter()) {
                    let value_count: usize = t.shape.iter().product();
                    let mut multi_idx = vec![0usize; t.shape.len()];
                    for _ in 0..value_count {
                        let mut flat_dest = 0usize;
                        for (axis, &i) in multi_idx.iter().enumerate() {
                            let dest_i = if axis == dim { i + offset } else { i };
                            flat_dest += dest_i * out_strides[axis];
                        }
                        out[flat_dest] = half::bf16::from_f64(t.get(&multi_idx));
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::BF16(out)
            }
            CpuBuffer::Q8_0(_) => panic!("concat not supported on Q8_0 buffer"),
        };

        let out = CpuStorage::from_contiguous(new_buffer, out_shape);

        let out_id = out.id;
        let input_ids: Vec<_> = tensors.iter().map(|t| t.id).collect();
        let input_dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape[dim]).collect();
        let offsets = cumulative_offsets.clone();
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids,
            backward: Box::new(move |grad_out: &CpuStorage| {
                offsets
                    .iter()
                    .zip(input_dim_sizes.iter())
                    .map(|(&offset, &len)| {
                        grad_out
                            .narrow(dim, offset, len)
                            .expect("concat backward: narrow of grad_out at a valid cumulative offset cannot fail")
                    })
                    .collect()
            }),
        });

        Ok(out)
    }

    /// `slice`.
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

    /// `flatten`.
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

    /// `broadcast_left`.
    fn broadcast_left<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut target_shape = shape.to_vec();
        target_shape.extend_from_slice(&t.shape);
        Self::broadcast_as::<K>(t, &target_shape)
    }

    /// `float_to_scalar`.
    fn float_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<f64> {
        if t.shape.iter().product::<usize>() != 1 {
            return Err(Error::ShapeMismatch {
                op: "float_to_scalar",
                expected: vec![1],
                got: t.shape.clone(),
                msg: alloc::string::String::from(
                    "float_to_scalar requires a single-element tensor",
                ),
            });
        }
        Ok(t.get(&vec![0usize; t.shape.len()]))
    }

    /// `float_to_vec1`.
    fn float_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<alloc::vec::Vec<f64>> {
        let total: usize = t.shape.iter().product();
        let mut out = alloc::vec::Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push(t.get(&idx));
            if !t.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &t.shape);
            }
        }
        Ok(out)
    }

    /// `int_to_scalar`.
    fn int_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<i64> {
        if t.shape.iter().product::<usize>() != 1 {
            return Err(Error::ShapeMismatch {
                op: "int_to_scalar",
                expected: vec![1],
                got: t.shape.clone(),
                msg: alloc::string::String::from("int_to_scalar requires a single-element tensor"),
            });
        }
        Ok(t.get(&vec![0usize; t.shape.len()]) as i64)
    }

    /// `int_to_vec1`.
    fn int_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<alloc::vec::Vec<i64>> {
        let total: usize = t.shape.iter().product();
        let mut out = alloc::vec::Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push(t.get(&idx) as i64);
            if !t.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &t.shape);
            }
        }
        Ok(out)
    }

    /// `tensor_to_dtype`.
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &<Self as Backend>::Storage<K>,
        dtype: KindleDType,
    ) -> Result<<Self as Backend>::Storage<K2>> {
        let total: usize = t.shape.iter().product();
        let mut multi_idx = vec![0usize; t.shape.len()];

        macro_rules! convert_variant {
            ($variant:ident, $ty:ty) => {{
                let mut out: alloc::vec::Vec<$ty> = alloc::vec::Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(t.get(&multi_idx) as $ty);
                    if !t.shape.is_empty() {
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::$variant(out)
            }};
        }

        let new_buffer = match dtype {
            KindleDType::F32 => convert_variant!(F32, f32),
            KindleDType::F64 => convert_variant!(F64, f64),
            KindleDType::U8 => convert_variant!(U8, u8),
            KindleDType::U32 => convert_variant!(U32, u32),
            KindleDType::I64 => convert_variant!(I64, i64),
            KindleDType::F16 => {
                let mut out: alloc::vec::Vec<half::f16> = alloc::vec::Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(half::f16::from_f64(t.get(&multi_idx)));
                    if !t.shape.is_empty() {
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::F16(out)
            }
            KindleDType::BF16 => {
                let mut out: alloc::vec::Vec<half::bf16> = alloc::vec::Vec::with_capacity(total);
                for _ in 0..total {
                    out.push(half::bf16::from_f64(t.get(&multi_idx)));
                    if !t.shape.is_empty() {
                        crate::cpu::storage::increment_index(&mut multi_idx, &t.shape);
                    }
                }
                CpuBuffer::BF16(out)
            }
            KindleDType::Q8_0 => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "tensor_to_dtype(Q8_0)",
                    backend: "Cpu",
                });
            }
            _ => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "tensor_to_dtype(unknown)",
                    backend: "Cpu",
                });
            }
        };

        Ok(CpuStorage::from_contiguous(new_buffer, t.shape.clone()))
    }
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    /// `TestBackend`.
    type TestBackend = CpuBackend<f32, kindle_core::prelude::Cpu>;

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
        let via_trait = TestBackend::reshape::<f32>(&t, &[3, 2]).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(f32_vec(&via_trait), f32_vec(&direct));
    }

    #[test]
    /// `transpose_through_trait_matches_direct_storage_call`.
    fn transpose_through_trait_matches_direct_storage_call() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let direct = t.transpose(0, 1).unwrap();
        let via_trait = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(via_trait.strides, direct.strides);
    }

    #[test]
    /// `broadcast_as_through_trait_matches_direct_storage_call`.
    fn broadcast_as_through_trait_matches_direct_storage_call() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
        let direct = t.broadcast_as(&[4, 3]).unwrap();
        let via_trait = TestBackend::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(via_trait.strides, direct.strides);
    }

    #[test]
    /// `float_to_scalar_reads_single_element`.
    fn float_to_scalar_reads_single_element() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![42.0]), vec![]);
        let v = TestBackend::float_to_scalar::<f32>(&t).unwrap();
        assert_eq!(v, 42.0);
    }

    #[test]
    /// `float_to_vec1_reads_all_elements_row_major`.
    fn float_to_vec1_reads_all_elements_row_major() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![3]);
        let v = TestBackend::float_to_vec1::<f32>(&t).unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    /// `reshape_backward_reshapes_grad_back_to_original_shape`.
    fn reshape_backward_reshapes_grad_back_to_original_shape() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = TestBackend::reshape::<f32>(&t, &[6]).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![2, 3]);
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    #[test]
    /// `transpose_backward_reapplies_same_transpose`.
    fn transpose_backward_reapplies_same_transpose() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = TestBackend::transpose::<f32>(&t, 0, 1).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(t.id).expect("t should have a gradient");
        assert_eq!(g.shape, vec![2, 3]);
        assert_eq!(f32_vec(g), vec![1.0; 6]);
    }

    #[test]
    /// `broadcast_as_backward_unbroadcasts_to_original_shape`.
    fn broadcast_as_backward_unbroadcasts_to_original_shape() {
        let t = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0]), vec![1, 3]);
        let out = TestBackend::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
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
        let out = TestBackend::matmul::<f32>(&lhs, &rhs).unwrap();
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
        // All other TensorOps methods are now fully implemented. We prove that
        // unsupported operations return typed errors by attempting to convert
        // to Q8_0, which is intentionally left unsupported in the Cpu backend.
        let result = TestBackend::tensor_to_dtype::<f32, f32>(&t, KindleDType::Q8_0);
        assert!(matches!(
            result,
            Err(Error::UnsupportedBackendOperation {
                op: "tensor_to_dtype(Q8_0)",
                ..
            })
        ));
    }

    /// Task 1 Test 1: `TensorOps::narrow` called through the trait matches
    /// calling `CpuStorage::narrow` directly (thin-wrapper equivalence).
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
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
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
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
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
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            4,
            3,
        );
        let result = TestBackend::slice::<f32>(&t, &[(1, 3), (0, 5)]);
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
        let lhs = CpuStorage::from_contiguous(CpuBuffer::F32(lhs_data), vec![2, 3, 4]);
        let rhs = CpuStorage::from_contiguous(CpuBuffer::F32(rhs_data), vec![2, 4, 5]);

        let direct = batched_matmul_impl(&lhs, &rhs).unwrap();
        let via_trait = TestBackend::matmul::<f32>(&lhs, &rhs).unwrap();
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
        let out = TestBackend::concat::<f32>(&[&a, &b], 0).unwrap();
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
        let out = TestBackend::concat::<f32>(&[&a, &b], 1).unwrap();
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
        let result = TestBackend::concat::<f32>(&[&a, &b], 0);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 1 Test 4: `concat(&[], 0)` (empty input list) returns
    /// `Err(Error::ShapeMismatch)`, not a panic.
    #[test]
    fn concat_empty_input_list_returns_err_not_panic() {
        let result: Result<CpuStorage> = TestBackend::concat::<f32>(&[], 0);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 1 Test 5: `concat` called with `dim >= rank` returns
    /// `Err(Error::ShapeMismatch)`.
    #[test]
    fn concat_dim_out_of_bounds_returns_err() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = TestBackend::concat::<f32>(&[&a, &b], 2);
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
        let out = TestBackend::concat::<f32>(&[&a, &b], 0).unwrap();
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
    /// strides without being materialized first — one input is a
    /// TRANSPOSED (non-contiguous) view, output values are still correct.
    #[test]
    fn concat_on_transposed_input_produces_correct_values_without_materializing() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let transposed = TestBackend::transpose::<f32>(&a, 0, 1).unwrap();
        // transposed: [[1,4],[2,5],[3,6]], shape [3,2]
        let b = matrix(vec![100.0, 200.0], 1, 2);
        let out = TestBackend::concat::<f32>(&[&transposed, &b], 0).unwrap();
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
        let out = TestBackend::stack::<f32>(&[&a, &b], 0).unwrap();
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
        let out = TestBackend::stack::<f32>(&[&a, &b], 2).unwrap();
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
    /// `Err(Error::ShapeMismatch)` — stack requires IDENTICAL shapes,
    /// stricter than concat's "all-but-one-axis" rule.
    #[test]
    fn stack_rejects_mismatched_shapes() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 4);
        let result = TestBackend::stack::<f32>(&[&a, &b], 0);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 2 Test 4: `stack(&[], 0)` (empty input list) returns
    /// `Err(Error::ShapeMismatch)`, not a panic.
    #[test]
    fn stack_empty_input_list_returns_err_not_panic() {
        let result: Result<CpuStorage> = TestBackend::stack::<f32>(&[], 0);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    /// Task 2 Test 5: `stack`'s backward correctly narrows-then-squeezes
    /// `grad_out` back to each input's own ORIGINAL shape (the inserted
    /// axis removed), with 2 distinct inputs.
    #[test]
    fn stack_backward_narrows_and_squeezes_grad_to_original_shape() {
        let a = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = matrix(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
        let out = TestBackend::stack::<f32>(&[&a, &b], 0).unwrap();
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
        let out = TestBackend::broadcast_left::<f32>(&t, &[4]).unwrap();
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
        let out = TestBackend::broadcast_left::<f32>(&t, &[2, 4]).unwrap();
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
        let out = TestBackend::broadcast_left::<f32>(&t, &[4]).unwrap();
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
        let via_trait = TestBackend::broadcast_left::<f32>(&t, &[4]).unwrap();
        assert_eq!(via_trait.shape, direct.shape);
        assert_eq!(via_trait.strides, direct.strides);
    }
}
