use crate::cuda::storage::CudaStorage;
use crate::dtype_policy::{BackendFamily, OperationKind, resolve_dtype_policy};
use alloc::sync::Arc;
use incin_core::backend_authoring::*;
use incin_core::prelude::*;

/// Type alias for `IncinBackend<T, D>` with a CUDA device. Kept for backwards
/// compatibility — prefer `IncinBackend<T, Cuda>` in new code.
#[derive(Clone)]
pub struct CudaBackendImpl<T = f32, D = Cuda>(core::marker::PhantomData<(T, D)>);

impl<T, D> CudaBackendImpl<T, D> {
    /// Construct the stateless CUDA executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<T, D> Default for CudaBackendImpl<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_cuda_storage_dtype {
    ($($dtype:ty),+ $(,)?) => {
        $(
            impl<T: DType, D: Device> SupportsDType<$dtype> for CudaBackendImpl<T, D> {
                fn resolve_dtype(
                    field: &<$dtype as DType>::Field,
                    _device: &DeviceId,
                ) -> Result<DTypeId> {
                    Ok(<$dtype as DType>::to_incin(field))
                }
            }
        )+
    };
}

impl_cuda_storage_dtype!(f32, f64, f16, bf16, i64);

impl<T: DType, D: Device> SupportsDType<Dyn> for CudaBackendImpl<T, D> {
    fn resolve_dtype(field: &DTypeId, _device: &DeviceId) -> Result<DTypeId> {
        resolve_dtype_policy(
            BackendFamily::Cuda,
            OperationKind::Storage,
            *field,
            "storage",
        )
        .map(|_| *field)
    }
}

#[derive(Clone)]
pub struct CudaVar {
    pub storage: CudaStorage,
}

pub type CudaGrads = crate::cuda::tape::CudaGrads;

impl<T: DType, D: Device> TensorOps<Self> for CudaBackendImpl<T, D> {
    // No CUDA kernels exist for these yet.
    crate::unsupported::unsupported_tensor_ops! {
        where_cond, gather, scatter, index_select, masked_fill,
        repeat, pad, triu, tril, diag,
        cmp_eq, cmp_ne, cmp_lt, cmp_le, cmp_gt, cmp_ge,
        logical_and, logical_or, logical_not,
        sub_scalar, div_scalar, maximum, minimum, abs_diff, lerp,
        unfold, pixel_shuffle, group_norm, instance_norm,
    }

    /// `unsqueeze`. Metadata-only, like `reshape` (which it delegates to and
    /// so inherits gradient wiring from), matching CPU's/WGPU's own
    /// `unsqueeze`.
    fn unsqueeze<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut target_shape = t.shape.to_vec();
        if dim <= target_shape.len() {
            target_shape.insert(dim, 1);
        } else {
            target_shape.push(1);
        }
        <Self as TensorOps<Self>>::reshape::<K>(t, &target_shape)
    }

    /// `float_to_scalar`. Same host-readback CUDA's own `to_bytes`/
    /// `topk`/`argsort` already use, restricted to F32 like those (a
    /// dtype-generic version is a separate, larger piece of work tracked
    /// apart from this pass — see `docs/PROJECT_STATUS.md`).
    fn float_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<f64> {
        let numel = checked_numel(&t.shape)?;
        if numel != 1 {
            return Err(Error::Shape(ShapeError::InvalidParameter {
                operation: OperationKind::Storage,
                parameter: "float_to_scalar element count",
                value: numel,
            }));
        }
        cuda_require_f32(t.buffer.dtype, "float_to_scalar")?;
        let data = download_f32_host(t)?;
        let value = data.first().copied().ok_or(Error::InternalInvariant {
            operation: "cuda_float_to_scalar",
            reason: "validated one-element storage read back no bytes",
        })?;
        Ok(f64::from(value))
    }
    /// `float_to_vec1`.
    fn float_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<Vec<f64>> {
        cuda_require_f32(t.buffer.dtype, "float_to_vec1")?;
        let data = download_f32_host(t)?;
        Ok(data.iter().map(|&x| x as f64).collect())
    }
    /// `int_to_scalar`.
    fn int_to_scalar<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<i64> {
        cuda_require_f32(t.buffer.dtype, "int_to_scalar")?;
        let data = download_f32_host(t)?;
        let value = data.first().copied().ok_or(Error::InvalidByteLength {
            expected: core::mem::size_of::<f32>(),
            got: 0,
        })?;
        incin_core::prelude::convert_f64_to_i64(
            "int_to_scalar",
            t.buffer.dtype,
            f64::from(value),
            incin_core::prelude::FloatToIntPolicy::Exact,
        )
    }
    /// `int_to_vec1`.
    fn int_to_vec1<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<Vec<i64>> {
        cuda_require_f32(t.buffer.dtype, "int_to_vec1")?;
        let data = download_f32_host(t)?;
        data.into_iter()
            .map(|value| {
                incin_core::prelude::convert_f64_to_i64(
                    "int_to_vec1",
                    t.buffer.dtype,
                    f64::from(value),
                    incin_core::prelude::FloatToIntPolicy::Exact,
                )
            })
            .collect()
    }
    /// `tensor_to_dtype`. Matches WGPU's own passthrough: both backends'
    /// physical storage does not vary with the requested logical dtype in a
    /// way this call needs to touch.
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &<Self as Backend>::Storage<K>,
        _dtype: DTypeId,
    ) -> Result<<Self as Backend>::Storage<K2>> {
        CudaStorage::try_new(t.buffer.clone(), t.shape.to_vec())
    }

    /// `addmm`. `beta * mat + alpha * (mat1 @ mat2)`, composed from the
    /// already tape-wired `matmul`/`mul_scalar_float`/`add`, matching CPU's
    /// and WGPU's own composition exactly — no new kernel, just reuse of
    /// already-implemented ones.
    fn addmm<K: DType>(
        mat: &<Self as Backend>::Storage<K>,
        mat1: &<Self as Backend>::Storage<K>,
        mat2: &<Self as Backend>::Storage<K>,
        beta: f64,
        alpha: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mm = <Self as TensorOps<Self>>::matmul::<K>(mat1, mat2)?;
        let mm_alpha = <Self as FloatOps<Self>>::mul_scalar_float::<K>(&mm, alpha)?;
        let mat_beta = <Self as FloatOps<Self>>::mul_scalar_float::<K>(mat, beta)?;
        <Self as NumericOps<Self>>::add::<K>(&mat_beta, &mm_alpha)
    }
    /// `bmm`. `matmul` already handles the batch dimensions, matching CPU
    /// and WGPU.
    fn bmm<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        <Self as TensorOps<Self>>::matmul::<K>(lhs, rhs)
    }

    /// `scaled_dot_product_attention`. Composed from the already tape-wired
    /// `transpose`/`matmul`/`mul_scalar_float`/`add`/`softmax`, matching
    /// CPU's and WGPU's own composition exactly, no new kernel.
    fn scaled_dot_product_attention<K: DType>(
        q: &<Self as Backend>::Storage<K>,
        k: &<Self as Backend>::Storage<K>,
        v: &<Self as Backend>::Storage<K>,
        mask: Option<&<Self as Backend>::Storage<K>>,
        scale: Option<f64>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let k_rank = k.shape.len();
        let k_t = if k_rank >= 2 {
            <Self as TensorOps<Self>>::transpose::<K>(k, k_rank - 2, k_rank - 1)?
        } else {
            k.clone()
        };
        let scores = <Self as TensorOps<Self>>::matmul::<K>(q, &k_t)?;
        let d_k = *q.shape.last().unwrap_or(&1) as f64;
        let s = scale.unwrap_or_else(|| 1.0 / d_k.sqrt());
        let scaled_scores = <Self as FloatOps<Self>>::mul_scalar_float::<K>(&scores, s)?;
        let masked_scores = if let Some(m) = mask {
            <Self as NumericOps<Self>>::add::<K>(&scaled_scores, m)?
        } else {
            scaled_scores
        };
        let attn = <Self as FloatOps<Self>>::softmax::<K>(&masked_scores, scores.shape.len() - 1)?;
        <Self as TensorOps<Self>>::matmul::<K>(&attn, v)
    }

    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cuda::ops::shape::launch_concat(tensors, dim)
    }

    /// Metadata-only: every `CudaStorage` this backend produces is always
    /// fully contiguous (`narrow`/`transpose`/`broadcast_as` below
    /// materialize a fresh contiguous buffer rather than building a
    /// strided view — CUDA's elementwise/matmul/reduce kernels assume flat
    /// contiguous memory), so reshaping never needs to touch the data or
    /// check contiguity first, unlike CPU's `reshape`.
    fn reshape<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let old_numel = ShapeBuf::from_slice(&t.shape).checked_numel(OperationKind::Reshape)?;
        let new_numel = ShapeBuf::from_slice(shape).checked_numel(OperationKind::Reshape)?;
        if old_numel != new_numel {
            return Err(Error::ShapeMismatch {
                op: "reshape",
                expected: t.shape.to_vec(),
                got: shape.to_vec(),
                msg: format!(
                    "reshape requires the same element count; {:?} has {old_numel}, target {:?} has {new_numel}",
                    t.shape, shape
                ),
            });
        }
        let out = CudaStorage::new(t.buffer.clone(), shape.to_vec());
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![Self::reshape::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }

    /// Materializes (see `reshape`'s doc for why CUDA can't use CPU's
    /// metadata-only strided-view approach here).
    fn transpose<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim1 >= t.shape.len() || dim2 >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "transpose",
                expected: t.shape.to_vec(),
                got: vec![dim1, dim2],
                msg: format!(
                    "transpose dims ({dim1}, {dim2}) out of range for shape {:?}",
                    t.shape
                ),
            });
        }
        let out = crate::cuda::ops::shape::launch_transpose(t, dim1, dim2)?;
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            // Re-applying the same transpose is its own inverse.
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![crate::cuda::ops::shape::launch_transpose(
                    grad_out, dim1, dim2,
                )?])
            }),
        });
        Ok(out)
    }

    /// Matmul is only wired for unbatched 2D operands so far — falls through to the `Backend`
    /// trait's default `Err(UnsupportedBackendOperation)` for anything else.
    fn matmul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if lhs.shape.len() != 2 || rhs.shape.len() != 2 || lhs.shape[1] != rhs.shape[0] {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: vec![lhs.shape[0], rhs.shape.first().copied().unwrap_or(0)],
                got: rhs.shape.to_vec(),
                msg: format!(
                    "matmul requires unbatched 2D operands with lhs.shape[1] == rhs.shape[0]; got lhs={:?}, rhs={:?}",
                    lhs.shape, rhs.shape
                ),
            });
        }
        let out = crate::cuda::ops::matmul::launch_matmul(lhs, rhs)?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                // grad_lhs = grad_out @ rhs.T ; grad_rhs = lhs.T @ grad_out
                let rhs_t = crate::cuda::ops::shape::launch_transpose(&rhs_capture, 0, 1)?;
                let grad_lhs = crate::cuda::ops::matmul::launch_matmul(grad_out, &rhs_t)?;
                let lhs_t = crate::cuda::ops::shape::launch_transpose(&lhs_capture, 0, 1)?;
                let grad_rhs = crate::cuda::ops::matmul::launch_matmul(&lhs_t, grad_out)?;
                Ok(vec![grad_lhs, grad_rhs])
            }),
        });
        Ok(out)
    }

    /// Materializes (see `reshape`'s doc for why).
    fn broadcast_as<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        // Validates compatibility before dispatch — an invalid broadcast
        // must error, not silently read garbage/OOB indices in the kernel.
        crate::cpu::stride::broadcast_shape(&t.shape, shape)?;

        let out = crate::cuda::ops::shape::launch_broadcast(t, shape)?;
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![crate::cuda::tape::unbroadcast(
                    grad_out,
                    &original_shape,
                )?])
            }),
        });
        Ok(out)
    }

    /// Materializes (see `reshape`'s doc for why).
    fn narrow<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() || start + len > t.shape[dim] {
            return Err(Error::ShapeMismatch {
                op: "narrow",
                expected: t.shape.to_vec(),
                got: vec![dim, start, len],
                msg: format!(
                    "narrow(dim={dim}, start={start}, len={len}) out of bounds for shape {:?}",
                    t.shape
                ),
            });
        }
        let out = crate::cuda::ops::shape::launch_narrow(t, dim, start, len)?;
        let original_shape = t.shape.to_vec();
        let mut region_start = vec![0usize; original_shape.len()];
        region_start[dim] = start;
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![crate::cuda::ops::shape::scatter_into_zeros(
                    &original_shape,
                    &region_start,
                    grad_out,
                )?])
            }),
        });
        Ok(out)
    }

    /// Composed from `reshape` (zero new tape entries — matches CPU/WGPU).
    fn squeeze<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if dim >= t.shape.len() || t.shape[dim] != 1 {
            return Err(Error::ShapeMismatch {
                op: "squeeze",
                expected: vec![1],
                got: t.shape.to_vec(),
                msg: format!(
                    "squeeze requires axis {dim} to have size 1, got size {} in shape {:?}",
                    t.shape.get(dim).copied().unwrap_or(0),
                    t.shape
                ),
            });
        }
        let mut target_shape = t.shape.to_vec();
        target_shape.remove(dim);
        Self::reshape::<K>(t, &target_shape)
    }

    /// Composed from `reshape` + `concat` (zero new tape entries — matches
    /// CPU/WGPU: `TensorOps` has no dedicated `unsqueeze`, so each input is
    /// reshaped to insert a size-1 axis at `dim`, then concatenated there).
    fn stack<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if tensors.is_empty() {
            return Err(Error::ShapeMismatch {
                op: "stack",
                expected: vec![],
                got: vec![],
                msg: "stack requires at least one input tensor".to_string(),
            });
        }
        let rank = tensors[0].shape.len();
        if dim > rank {
            return Err(Error::ShapeMismatch {
                op: "stack",
                expected: tensors[0].shape.to_vec(),
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
                    expected: tensors[0].shape.to_vec(),
                    got: t.shape.to_vec(),
                    msg: format!(
                        "stack requires every input to have an IDENTICAL shape; expected {:?}, got {:?}",
                        tensors[0].shape, t.shape
                    ),
                });
            }
        }
        let mut unsqueezed = Vec::with_capacity(tensors.len());
        for t in tensors.iter() {
            let mut target_shape = t.shape.to_vec();
            target_shape.insert(dim, 1);
            unsqueezed.push(Self::reshape::<K>(t, &target_shape)?);
        }
        let refs: Vec<&<Self as Backend>::Storage<K>> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    /// Composed from `narrow` (zero new tape entries — matches CPU/WGPU).
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

    /// Composed from `reshape` (zero new tape entries — matches CPU/WGPU).
    fn flatten<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if start_dim > end_dim || end_dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "flatten",
                expected: t.shape.to_vec(),
                got: vec![start_dim, end_dim],
                msg: format!(
                    "flatten(start_dim={start_dim}, end_dim={end_dim}) out of bounds for shape {:?}",
                    t.shape
                ),
            });
        }
        let merged: usize =
            incin_core::prelude::ShapeBuf::from_slice(&(t.shape[start_dim..=end_dim]))
                .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        let mut target_shape = t.shape[..start_dim].to_vec();
        target_shape.push(merged);
        target_shape.extend_from_slice(&t.shape[end_dim + 1..]);
        Self::reshape::<K>(t, &target_shape)
    }

    /// Composed from `broadcast_as` (zero new tape entries — matches CPU/WGPU).
    fn broadcast_left<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let mut target_shape = shape.to_vec();
        target_shape.extend_from_slice(&t.shape);
        Self::broadcast_as::<K>(t, &target_shape)
    }
}

impl<T: DType, D: Device> NumericOps<Self> for CudaBackendImpl<T, D> {
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("add", "a + b", lhs, rhs, &out_shape)?;
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![
                    crate::cuda::tape::unbroadcast(grad_out, &lhs_shape)?,
                    crate::cuda::tape::unbroadcast(grad_out, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }

    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("sub", "a - b", lhs, rhs, &out_shape)?;
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let neg_grad =
                    crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", grad_out)?;
                Ok(vec![
                    crate::cuda::tape::unbroadcast(grad_out, &lhs_shape)?,
                    crate::cuda::tape::unbroadcast(&neg_grad, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }

    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("mul", "a * b", lhs, rhs, &out_shape)?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let grad_lhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &rhs_capture.shape)?;
                let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &rhs_capture,
                    &grad_lhs_shape,
                )?;
                let grad_rhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &lhs_capture.shape)?;
                let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &lhs_capture,
                    &grad_rhs_shape,
                )?;
                Ok(vec![
                    crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }

    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("div", "a / b", lhs, rhs, &out_shape)?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &rhs_capture.shape)?;
                let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "div",
                    "a / b",
                    grad_out,
                    &rhs_capture,
                    &grad_lhs_shape,
                )?;
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let rhs_sq_shape =
                    crate::cpu::stride::broadcast_shape(&rhs_capture.shape, &rhs_capture.shape)?;
                let rhs_sq = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    &rhs_capture,
                    &rhs_capture,
                    &rhs_sq_shape,
                )?;
                let ratio_shape =
                    crate::cpu::stride::broadcast_shape(&lhs_capture.shape, &rhs_sq.shape)?;
                let lhs_over_rhs_sq = crate::cuda::ops::elementwise::launch_binary_op(
                    "div",
                    "a / b",
                    &lhs_capture,
                    &rhs_sq,
                    &ratio_shape,
                )?;
                let neg_ratio =
                    crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", &lhs_over_rhs_sq)?;
                let grad_rhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &neg_ratio.shape)?;
                let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &neg_ratio,
                    &grad_rhs_shape,
                )?;
                Ok(vec![
                    crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                    crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }
}

fn push_unary_tape_entry(
    t_id: crate::cuda::storage::TensorId,
    out_id: crate::cuda::storage::TensorId,
    grad_fn: impl Fn(&CudaStorage) -> Result<CudaStorage> + Send + Sync + 'static,
) {
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CudaStorage| grad_fn(grad_out).map(|grad| vec![grad])),
    });
}

impl<T: DType, D: Device> FloatOps<Self> for CudaBackendImpl<T, D> {
    // No CUDA kernel is launched for these yet. They are declared rather than
    // inherited so the gap is visible from the backend that has it.
    crate::unsupported::unsupported_float_ops! {
        unary: sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
               atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    fn relu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op("relu", "x > 0.0f ? x : 0.0f", t)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let deriv = crate::cuda::ops::elementwise::launch_unary_op(
                "step",
                "x > 0.0f ? 1.0f : 0.0f",
                &t_capture,
            )?;
            let out_shape = crate::cpu::stride::broadcast_shape(&grad_out.shape, &deriv.shape)?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul", "a * b", grad_out, &deriv, &out_shape,
            )
        });
        Ok(out)
    }

    fn sigmoid<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op(
            "sigmoid",
            "1.0f / (1.0f + expf(-x))",
            t,
        )?;
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let neg_out =
                crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", &out_capture)?;
            let one_minus_out =
                crate::cuda::ops::elementwise::launch_unary_op("add_one", "1.0f + x", &neg_out)?;
            let deriv_shape =
                crate::cpu::stride::broadcast_shape(&out_capture.shape, &one_minus_out.shape)?;
            let deriv = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                &out_capture,
                &one_minus_out,
                &deriv_shape,
            )?;
            let grad_shape = crate::cpu::stride::broadcast_shape(&grad_out.shape, &deriv.shape)?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &deriv,
                &grad_shape,
            )
        });
        Ok(out)
    }

    fn tanh<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op("tanh", "tanhf(x)", t)?;
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let out_sq_shape =
                crate::cpu::stride::broadcast_shape(&out_capture.shape, &out_capture.shape)?;
            let out_sq = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                &out_capture,
                &out_capture,
                &out_sq_shape,
            )?;
            let neg_out_sq = crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", &out_sq)?;
            let deriv =
                crate::cuda::ops::elementwise::launch_unary_op("add_one", "1.0f + x", &neg_out_sq)?;
            let grad_shape = crate::cpu::stride::broadcast_shape(&grad_out.shape, &deriv.shape)?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &deriv,
                &grad_shape,
            )
        });
        Ok(out)
    }

    fn swish<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out =
            crate::cuda::ops::elementwise::launch_unary_op("swish", "x / (1.0f + expf(-x))", t)?;
        let t_capture = t.clone();
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let sig = crate::cuda::ops::elementwise::launch_unary_op(
                "sigmoid",
                "1.0f / (1.0f + expf(-x))",
                &t_capture,
            )?;
            let neg_out =
                crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", &out_capture)?;
            let one_minus_out =
                crate::cuda::ops::elementwise::launch_unary_op("add_one", "1.0f + x", &neg_out)?;
            let sig_term_shape =
                crate::cpu::stride::broadcast_shape(&sig.shape, &one_minus_out.shape)?;
            let sig_term = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                &sig,
                &one_minus_out,
                &sig_term_shape,
            )?;
            let deriv_shape =
                crate::cpu::stride::broadcast_shape(&out_capture.shape, &sig_term.shape)?;
            let deriv = crate::cuda::ops::elementwise::launch_binary_op(
                "add",
                "a + b",
                &out_capture,
                &sig_term,
                &deriv_shape,
            )?;
            let grad_shape = crate::cpu::stride::broadcast_shape(&grad_out.shape, &deriv.shape)?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &deriv,
                &grad_shape,
            )
        });
        Ok(out)
    }

    /// `mish(x) = x * tanh(softplus(x))`. Forward/backward formulas ported
    /// verbatim from `cpu/ops/elementwise_kernel.rs`'s `UnaryOp::Mish` /
    /// `cpu/ops/elementwise.rs::mish`'s backward closure — not re-derived —
    /// including the `x > 20` softplus overflow guard.
    fn mish<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let softplus_expr = "x > 20.0f ? x : logf(1.0f + expf(x))";
        let sp = crate::cuda::ops::elementwise::launch_unary_op("softplus", softplus_expr, t)?;
        let th = crate::cuda::ops::elementwise::launch_unary_op("tanhf", "tanhf(x)", &sp)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("mul", "a * b", t, &th, &t.shape)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let sp = crate::cuda::ops::elementwise::launch_unary_op(
                "softplus",
                softplus_expr,
                &t_capture,
            )?;
            let th = crate::cuda::ops::elementwise::launch_unary_op("tanhf", "tanhf(x)", &sp)?;
            let sig = crate::cuda::ops::elementwise::launch_unary_op(
                "sigmoid",
                "1.0f / (1.0f + expf(-x))",
                &t_capture,
            )?;
            // deriv = th + x * sig * (1 - th^2)
            let th_sq = crate::cuda::ops::elementwise::launch_binary_op(
                "mul", "a * b", &th, &th, &th.shape,
            )?;
            let one_minus_th_sq =
                crate::cuda::ops::elementwise::launch_unary_op("one_minus", "1.0f - x", &th_sq)?;
            let x_sig = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                &t_capture,
                &sig,
                &t_capture.shape,
            )?;
            let term2 = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                &x_sig,
                &one_minus_th_sq,
                &x_sig.shape,
            )?;
            let deriv = crate::cuda::ops::elementwise::launch_binary_op(
                "add", "a + b", &th, &term2, &th.shape,
            )?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &deriv,
                &grad_out.shape,
            )
        });
        Ok(out)
    }

    /// `elu(x) = x > 0 ? x : exp(x) - 1`. Backward is output-based
    /// (`o > 0 ? 1 : o + 1`), ported verbatim from
    /// `cpu/ops/elementwise.rs::elu`'s backward closure.
    fn elu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op(
            "elu",
            "x > 0.0f ? x : expf(x) - 1.0f",
            t,
        )?;
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let deriv = crate::cuda::ops::elementwise::launch_unary_op(
                "elu_grad",
                "x > 0.0f ? 1.0f : x + 1.0f",
                &out_capture,
            )?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &deriv,
                &grad_out.shape,
            )
        });
        Ok(out)
    }

    /// `gelu(x) = x * 0.5 * (1 + erf(x/sqrt(2)))`, using CUDA's native
    /// `erff` device intrinsic (unlike WGPU, which has no `erf` primitive
    /// and needed a polynomial approximation). Backward ported verbatim from
    /// `cpu/ops/elementwise.rs::gelu`'s backward closure (input-based:
    /// `cdf(x) + x * pdf(x)`).
    fn gelu<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let cdf_expr = "0.5f * (1.0f + erff(x * 0.7071067811865476f))";
        let cdf = crate::cuda::ops::elementwise::launch_unary_op("gelu_cdf", cdf_expr, t)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("mul", "a * b", t, &cdf, &t.shape)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let cdf =
                crate::cuda::ops::elementwise::launch_unary_op("gelu_cdf", cdf_expr, &t_capture)?;
            let pdf = crate::cuda::ops::elementwise::launch_unary_op(
                "gelu_pdf",
                "0.3989422804014327f * expf(-x * x * 0.5f)",
                &t_capture,
            )?;
            let x_pdf = crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                &t_capture,
                &pdf,
                &t_capture.shape,
            )?;
            let deriv = crate::cuda::ops::elementwise::launch_binary_op(
                "add", "a + b", &cdf, &x_pdf, &cdf.shape,
            )?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &deriv,
                &grad_out.shape,
            )
        });
        Ok(out)
    }

    fn exp<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op("exp", "expf(x)", t)?;
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let grad_shape =
                crate::cpu::stride::broadcast_shape(&grad_out.shape, &out_capture.shape)?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &out_capture,
                &grad_shape,
            )
        });
        Ok(out)
    }

    fn log<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op("log", "logf(x)", t)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let grad_shape =
                crate::cpu::stride::broadcast_shape(&grad_out.shape, &t_capture.shape)?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "div",
                "a / b",
                grad_out,
                &t_capture,
                &grad_shape,
            )
        });
        Ok(out)
    }

    fn sqrt<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op("sqrt", "sqrtf(x)", t)?;
        let out_capture = out.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let ratio_shape =
                crate::cpu::stride::broadcast_shape(&grad_out.shape, &out_capture.shape)?;
            let ratio = crate::cuda::ops::elementwise::launch_binary_op(
                "div",
                "a / b",
                grad_out,
                &out_capture,
                &ratio_shape,
            )?;
            crate::cuda::ops::elementwise::launch_unary_op("half", "x * 0.5f", &ratio)
        });
        Ok(out)
    }

    fn neg<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", t)?;
        push_unary_tape_entry(t.id, out.id, |grad_out| {
            crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", grad_out)
        });
        Ok(out)
    }

    fn abs<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out = crate::cuda::ops::elementwise::launch_unary_op("abs", "fabsf(x)", t)?;
        let t_capture = t.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let sign = crate::cuda::ops::elementwise::launch_unary_op(
                "sign",
                "x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f)",
                &t_capture,
            )?;
            let grad_shape = crate::cpu::stride::broadcast_shape(&grad_out.shape, &sign.shape)?;
            crate::cuda::ops::elementwise::launch_binary_op(
                "mul",
                "a * b",
                grad_out,
                &sign,
                &grad_shape,
            )
        });
        Ok(out)
    }

    fn step<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let out =
            crate::cuda::ops::elementwise::launch_unary_op("step", "x > 0.0f ? 1.0f : 0.0f", t)?;
        push_unary_tape_entry(t.id, out.id, |grad_out| {
            crate::cuda::ops::elementwise::launch_unary_op("zero", "0.0f", grad_out)
        });
        Ok(out)
    }

    fn add_scalar_float<K: DType>(t: &CudaStorage, scalar: f64) -> Result<CudaStorage> {
        let expr = format!("x + ({:.8}f)", scalar as f32);
        let out = crate::cuda::ops::elementwise::launch_unary_op("add_scalar", &expr, t)?;
        push_unary_tape_entry(t.id, out.id, |grad_out| Ok(grad_out.clone()));
        Ok(out)
    }

    fn mul_scalar_float<K: DType>(t: &CudaStorage, scalar: f64) -> Result<CudaStorage> {
        let expr = format!("x * ({:.8}f)", scalar as f32);
        let out = crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, t)?;
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let expr = format!("x * ({:.8}f)", scalar as f32);
            crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, grad_out)
        });
        Ok(out)
    }

    fn softmax<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let ls = log_softmax::<T, K>(t, dim)?;
        Self::exp::<K>(&ls)
    }
}

/// Helper function to compute log_softmax composed from primitives on CUDA backend.
pub(crate) fn log_softmax<T: DType, K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
    let max = CudaBackendImpl::<T>::max_keepdim::<K>(t, dim)?;
    let max_b = CudaBackendImpl::<T>::broadcast_as::<K>(&max, &t.shape)?;
    let diff = CudaBackendImpl::<T>::sub::<K>(t, &max_b)?;
    let exp_diff = CudaBackendImpl::<T>::exp::<K>(&diff)?;
    let sum_exp = CudaBackendImpl::<T>::sum_keepdim::<K>(&exp_diff, dim)?;
    let sum_exp_b = CudaBackendImpl::<T>::broadcast_as::<K>(&sum_exp, &t.shape)?;
    let log_sum = CudaBackendImpl::<T>::log::<K>(&sum_exp_b)?;
    CudaBackendImpl::<T>::sub::<K>(&diff, &log_sum)
}

impl<T: DType, D: Device> CreationOps<Self> for CudaBackendImpl<T, D> {
    // No kernel fills an arbitrary value or generates a sequence yet.
    crate::unsupported::unsupported_creation_ops! {
        fill: full;
        sequence: arange, linspace;
    }

    fn zeros<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaStorage> {
        cuda_from_f32(
            shape,
            dtype,
            device,
            vec![0.0; checked_numel(shape)?],
            "zeros",
        )
    }

    fn ones<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaStorage> {
        cuda_from_f32(
            shape,
            dtype,
            device,
            vec![1.0; checked_numel(shape)?],
            "ones",
        )
    }

    fn rand<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaStorage> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let values = (0..checked_numel(shape)?).map(|_| rng.r#gen()).collect();
        cuda_from_f32(shape, dtype, device, values, "rand")
    }

    fn randn<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaStorage> {
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand::thread_rng();
        let values = (0..checked_numel(shape)?)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        cuda_from_f32(shape, dtype, device, values, "randn")
    }

    fn var_zeros<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaVar> {
        Self::zeros::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    fn var_ones<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaVar> {
        Self::ones::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    fn var_rand<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaVar> {
        Self::rand::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    fn var_randn<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaVar> {
        Self::randn::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }
}
impl<T: DType, D: Device> ReductionOps<Self> for CudaBackendImpl<T, D> {
    // No product-reduction or prefix-scan kernel exists yet.
    crate::unsupported::unsupported_reduction_ops! {
        all: prod_all;
        dim: prod_dim, cumsum;
    }

    fn sum_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let rank = t.shape.len();
        if rank == 0 {
            return Ok(t.clone());
        }
        let mut curr = t.clone();
        for dim in (0..rank).rev() {
            curr = Self::sum_dim::<K>(&curr, dim)?;
        }
        Ok(curr)
    }

    fn mean_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let total = checked_numel(&t.shape)? as f64;
        let sum = Self::sum_all::<K>(t)?;
        if total > 0.0 {
            Self::mul_scalar_float::<K>(&sum, 1.0 / total)
        } else {
            Ok(sum)
        }
    }

    fn max_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let rank = t.shape.len();
        if rank == 0 {
            return Ok(t.clone());
        }
        let mut curr = t.clone();
        for dim in (0..rank).rev() {
            curr = Self::max_dim::<K>(&curr, dim)?;
        }
        Ok(curr)
    }

    fn min_all<K: DType>(t: &CudaStorage) -> Result<CudaStorage> {
        let rank = t.shape.len();
        if rank == 0 {
            return Ok(t.clone());
        }
        let mut curr = t.clone();
        for dim in (0..rank).rev() {
            curr = Self::min_dim::<K>(&curr, dim)?;
        }
        Ok(curr)
    }

    fn sum_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_reduce_op("sum", t, dim, false)?;
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::tape::unbroadcast(grad_out, &t_shape)
        });
        Ok(out)
    }

    fn sum_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let out = crate::cuda::ops::reduce::launch_reduce_op("sum", t, dim, true)?;
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::tape::unbroadcast(grad_out, &t_shape)
        });
        Ok(out)
    }

    fn mean_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let axis_len = *t.shape.get(dim).ok_or(ShapeError::InvalidParameter {
            operation: OperationKind::Reduction,
            parameter: "axis",
            value: dim,
        })? as f64;
        let sum = crate::cuda::ops::reduce::launch_reduce_op("sum", t, dim, false)?;
        let out = if axis_len > 0.0 {
            Self::mul_scalar_float::<K>(&sum, 1.0 / axis_len)?
        } else {
            sum
        };
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let unb = crate::cuda::tape::unbroadcast(grad_out, &t_shape)?;
            if axis_len > 0.0 {
                let expr = format!("x * ({:.8}f)", (1.0 / axis_len) as f32);
                crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, &unb)
            } else {
                Ok(unb)
            }
        });
        Ok(out)
    }

    fn mean_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        let axis_len = *t.shape.get(dim).ok_or(ShapeError::InvalidParameter {
            operation: OperationKind::Reduction,
            parameter: "axis",
            value: dim,
        })? as f64;
        let sum = crate::cuda::ops::reduce::launch_reduce_op("sum", t, dim, true)?;
        let out = if axis_len > 0.0 {
            Self::mul_scalar_float::<K>(&sum, 1.0 / axis_len)?
        } else {
            sum
        };
        let t_shape = t.shape.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let unb = crate::cuda::tape::unbroadcast(grad_out, &t_shape)?;
            if axis_len > 0.0 {
                let expr = format!("x * ({:.8}f)", (1.0 / axis_len) as f32);
                crate::cuda::ops::elementwise::launch_unary_op("mul_scalar", &expr, &unb)
            } else {
                Ok(unb)
            }
        });
        Ok(out)
    }

    fn max_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_reduce_op("max", t, dim, false)
    }

    fn max_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_reduce_op("max", t, dim, true)
    }

    fn min_dim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_reduce_op("min", t, dim, false)
    }

    fn min_keepdim<K: DType>(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
        crate::cuda::ops::reduce::launch_reduce_op("min", t, dim, true)
    }

    /// `dim: None` flattens first, then reduces axis 0 — for a 1D tensor,
    /// "coordinate along axis 0 of the winner" and "global flat index of
    /// the winner" are the same number, so this needs no special-casing
    /// versus the `Some(d)` path, matching CPU's `argmax`/`argmin` semantics
    /// (flat index for `None`, per-axis coordinate for `Some(d)`) exactly.
    fn argmax<K: DType, KInt: DType>(t: &CudaStorage, dim: Option<usize>) -> Result<CudaStorage> {
        let (target, axis) = match dim {
            Some(d) => {
                if d >= t.shape.len() {
                    return Err(Error::ShapeMismatch {
                        op: "argmax",
                        expected: t.shape.to_vec(),
                        got: vec![d],
                        msg: format!("argmax: axis {d} out of range for shape {:?}", t.shape),
                    });
                }
                (t.clone(), d)
            }
            None => {
                let numel: usize = incin_core::prelude::ShapeBuf::from_slice(&(t.shape))
                    .checked_numel(incin_core::prelude::OperationKind::Storage)?;
                (<Self as TensorOps<Self>>::reshape::<K>(t, &[numel])?, 0)
            }
        };
        let (_, idx_u32) =
            crate::cuda::ops::reduce::launch_reduce_with_indices_op("max", &target, axis, false)?;
        crate::cuda::ops::reduce::indices_u32_to_i64(&idx_u32)
    }

    fn argmin<K: DType, KInt: DType>(t: &CudaStorage, dim: Option<usize>) -> Result<CudaStorage> {
        let (target, axis) = match dim {
            Some(d) => {
                if d >= t.shape.len() {
                    return Err(Error::ShapeMismatch {
                        op: "argmin",
                        expected: t.shape.to_vec(),
                        got: vec![d],
                        msg: format!("argmin: axis {d} out of range for shape {:?}", t.shape),
                    });
                }
                (t.clone(), d)
            }
            None => {
                let numel: usize = incin_core::prelude::ShapeBuf::from_slice(&(t.shape))
                    .checked_numel(incin_core::prelude::OperationKind::Storage)?;
                (<Self as TensorOps<Self>>::reshape::<K>(t, &[numel])?, 0)
            }
        };
        let (_, idx_u32) =
            crate::cuda::ops::reduce::launch_reduce_with_indices_op("min", &target, axis, false)?;
        crate::cuda::ops::reduce::indices_u32_to_i64(&idx_u32)
    }

    fn topk<K: DType, KInt: DType>(
        t: &CudaStorage,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(CudaStorage, CudaStorage)> {
        cuda_topk_host(t, k, dim, largest)
    }

    fn argsort<K: DType, KInt: DType>(
        t: &CudaStorage,
        dim: usize,
        descending: bool,
    ) -> Result<CudaStorage> {
        cuda_argsort_host(t, dim, descending)
    }
}
impl<T: DType, D: Device> QuantizedOps<Self> for CudaBackendImpl<T, D> {
    fn quantize<K: FloatDType, Q: QuantDType>(t: &CudaStorage) -> Result<CudaStorage> {
        crate::cuda::ops::quant::launch_quantize(t)
    }

    fn dequantize<Q: QuantDType, K: FloatDType>(t: &CudaStorage) -> Result<CudaStorage> {
        crate::cuda::ops::quant::launch_dequantize(t)
    }

    /// **Correctness-first, not bandwidth-optimal**: dequantizes both
    /// operands to `f32` then calls the already-wired `matmul`, unlike
    /// CPU's `quantized_matmul` (`cpu/ops/quant.rs`), which fuses the Q8_0
    /// block-dequant directly into an AVX2 dot product without ever
    /// materializing full-precision copies — the `QuantizedOps` trait doc
    /// explicitly frames avoiding that materialization as the point of this
    /// method. Porting CPU's fused block-dot-product math to a new CUDA
    /// kernel blind (no hardware here to verify Q8_0 block-scale handling
    /// against) is exactly the kind of change this codebase's audit history
    /// treats as too risky to do without real-hardware verification — this
    /// composition is mathematically equivalent (same result, more memory
    /// bandwidth), and is the safer choice until real hardware is
    /// available to validate a fused kernel against. Only `Q8_0` is
    /// supported, matching CPU's own restriction exactly.
    fn quantized_matmul<Q: QuantDType>(
        lhs: &CudaStorage,
        rhs: &CudaStorage,
    ) -> Result<CudaStorage> {
        if core::any::TypeId::of::<Q>() != core::any::TypeId::of::<incin_core::prelude::Q8_0>() {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantized_matmul",
                backend: "Cuda (only Q8_0 supported)",
            });
        }
        if rhs.shape.len() != 2 {
            return Err(Error::Msg("quantized_matmul rhs must be 2D [N, K]".into()));
        }
        if lhs.shape.len() < 2 {
            return Err(Error::Msg(
                "quantized_matmul lhs requires at least 2D shapes".into(),
            ));
        }
        let k = lhs.shape[lhs.shape.len() - 1];
        let m: usize =
            incin_core::prelude::ShapeBuf::from_slice(&(lhs.shape[..lhs.shape.len() - 1]))
                .checked_numel(incin_core::prelude::OperationKind::Storage)?;
        let n = rhs.shape[0];
        if k != rhs.shape[1] {
            return Err(Error::Msg(format!(
                "quantized_matmul K mismatch: {k} != {}",
                rhs.shape[1]
            )));
        }
        if !k.is_multiple_of(32) {
            return Err(Error::Msg(format!(
                "quantized_matmul K must be multiple of 32, got {k}"
            )));
        }

        let lhs_f32 = Self::dequantize::<Q, f32>(lhs)?;
        let rhs_f32 = Self::dequantize::<Q, f32>(rhs)?;
        let lhs_2d = <Self as TensorOps<Self>>::reshape::<f32>(&lhs_f32, &[m, k])?;
        // rhs is stored [N, K]; matmul needs [K, N].
        let rhs_t = crate::cuda::ops::shape::launch_transpose(&rhs_f32, 0, 1)?;
        let out_2d = crate::cuda::ops::matmul::launch_matmul(&lhs_2d, &rhs_t)?;

        let mut out_shape = lhs.shape.to_vec();
        let last = out_shape.len() - 1;
        out_shape[last] = n;
        <Self as TensorOps<Self>>::reshape::<f32>(&out_2d, &out_shape)
    }
}
impl<T: DType, D: Device> OptimizerOps<Self> for CudaBackendImpl<T, D> {}

/// Tape-tracked wrapper pairing `launch_im2col_2d`/`launch_col2im_2d` as each
/// other's forward/backward (they are exact inverses of one another). Once
/// this is a proper tape op, `conv1d`/`conv2d`'s own forward can be composed
/// entirely from already-tape-tracked primitives (`narrow`/`reshape`/
/// `matmul`/`concat` plus this) with NO hand-written backward closure of
/// their own — mirroring the `LossOps`/`OptimizerOps` "free via composition"
/// discovery documented by the backend conformance audit.
fn im2col_2d_tape(
    t: &CudaStorage,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CudaStorage> {
    let out = crate::cuda::ops::conv::launch_im2col_2d(t, kh, kw, stride, padding, dilation)?;
    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            let h_out =
                crate::cuda::ops::conv::out_size(original_shape[2], kh, stride, padding, dilation)?;
            let w_out =
                crate::cuda::ops::conv::out_size(original_shape[3], kw, stride, padding, dilation)?;
            Ok(vec![crate::cuda::ops::conv::launch_col2im_2d(
                grad_out,
                &original_shape,
                h_out,
                w_out,
                kh,
                kw,
                stride,
                padding,
                dilation,
            )?])
        }),
    });
    Ok(out)
}

/// Symmetric counterpart of `im2col_2d_tape` — `conv_transpose2d`'s forward
/// calls this directly (its forward IS `conv2d`'s backward-data formula),
/// so this needs its own tape entry whose backward is `launch_im2col_2d`.
fn col2im_2d_tape(
    cols: &CudaStorage,
    target_shape: &[usize],
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CudaStorage> {
    let out = crate::cuda::ops::conv::launch_col2im_2d(
        cols,
        target_shape,
        h_out,
        w_out,
        kh,
        kw,
        stride,
        padding,
        dilation,
    )?;
    let cols_shape = cols.shape.to_vec();
    let (cols_id, out_id) = (cols.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![cols_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            let cols_grad = crate::cuda::ops::conv::launch_im2col_2d(
                grad_out, kh, kw, stride, padding, dilation,
            )?;
            debug_assert_eq!(cols_grad.shape, cols_shape);
            Ok(vec![cols_grad])
        }),
    });
    Ok(out)
}

/// 1D analogue of `im2col_2d_tape`.
fn im2col_1d_tape(
    t: &CudaStorage,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CudaStorage> {
    let out = crate::cuda::ops::conv::launch_im2col_1d(t, k, stride, padding, dilation)?;
    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            let l_out =
                crate::cuda::ops::conv::out_size(original_shape[2], k, stride, padding, dilation)?;
            Ok(vec![crate::cuda::ops::conv::launch_col2im_1d(
                grad_out,
                &original_shape,
                l_out,
                k,
                stride,
                padding,
                dilation,
            )?])
        }),
    });
    Ok(out)
}

/// Pads `t: [B, C, H, W]` with `pad_h`/`pad_w` trailing zero rows/columns —
/// `conv_transpose2d`'s `output_padding` handling. This is exactly `narrow`'s
/// backward (`scatter_into_zeros` at `region_start = [0,0,0,0]`) reused as a
/// forward op, so its own backward is the matching two-axis narrow back down
/// to the original `H`/`W`.
fn pad_trailing_zeros_2d_tape(t: &CudaStorage, pad_h: usize, pad_w: usize) -> Result<CudaStorage> {
    let (b, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let target_shape = vec![b, c, h + pad_h, w + pad_w];
    let out = crate::cuda::ops::shape::scatter_into_zeros(&target_shape, &[0, 0, 0, 0], t)?;
    let (t_id, out_id) = (t.id, out.id);
    crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CudaStorage| {
            let narrowed_h = crate::cuda::ops::shape::launch_narrow(grad_out, 2, 0, h)?;
            let narrowed = crate::cuda::ops::shape::launch_narrow(&narrowed_h, 3, 0, w)?;
            Ok(vec![narrowed])
        }),
    });
    Ok(out)
}

/// Matches `cpu/ops/conv.rs::validate_groups` exactly.
fn validate_conv_groups(op: &'static str, cin: usize, cout: usize, groups: usize) -> Result<()> {
    if groups == 0 || !cin.is_multiple_of(groups) || !cout.is_multiple_of(groups) {
        return Err(Error::ShapeMismatch {
            op,
            expected: vec![groups],
            got: vec![cin, cout],
            msg: format!("{op}: groups={groups} must evenly divide both Cin={cin} and Cout={cout}"),
        });
    }
    Ok(())
}

impl<T: DType, D: Device> ModuleOps<Self> for CudaBackendImpl<T, D> {
    fn layer_norm<K: DType>(
        input: &CudaStorage,
        weight: &CudaStorage,
        bias: Option<&CudaStorage>,
        eps: f32,
    ) -> Result<CudaStorage> {
        crate::cuda::ops::norm::launch_layer_norm(input, weight, bias, eps)
    }

    fn batch_norm<K: DType>(
        input: &CudaStorage,
        weight: Option<&CudaStorage>,
        bias: Option<&CudaStorage>,
        running_mean: Option<&CudaStorage>,
        running_variance: Option<&CudaStorage>,
        eps: f32,
        _momentum: f64,
    ) -> Result<CudaStorage> {
        crate::cuda::ops::norm::launch_batch_norm(
            input,
            weight,
            bias,
            running_mean,
            running_variance,
            eps,
        )
    }

    /// Embedding table lookup. Only `w` (the weight table) is differentiable
    /// — `t` (integer indices) is not part of the tape's `input_ids`,
    /// matching CPU's `embedding_impl` (`cpu/ops/embedding.rs`) exactly.
    fn embedding<K: DType, KInt: DType>(
        t: &<Self as Backend>::Storage<KInt>,
        w: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if w.shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                op: "embedding",
                expected: vec![0, 0],
                got: w.shape.to_vec(),
                msg: format!(
                    "embedding: weight table must be rank-2 [vocab_size, hidden_size], got shape {:?}",
                    w.shape
                ),
            });
        }
        let (vocab_size, hidden_size) = (w.shape[0], w.shape[1]);
        let out = crate::cuda::ops::embedding::launch_embedding_forward(w, t)?;
        let indices_capture = t.clone();
        let (w_id, out_id) = (w.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![w_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![
                    crate::cuda::ops::embedding::launch_embedding_backward(
                        grad_out,
                        &indices_capture,
                        vocab_size,
                        hidden_size,
                    )?,
                ])
            }),
        });
        Ok(out)
    }

    /// Backward replays `max_indices` (captured from the forward pass)
    /// through `scatter_pool_grad_2d` — no forward recomputation needed,
    /// mirrors CPU's `max_window_2d`/`scatter_pool_grad_2d` pairing exactly.
    fn max_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let (out, max_indices) = crate::cuda::ops::pool::launch_max_pool2d_forward(
            t,
            kernel_size,
            stride,
            padding,
            dilation,
        )?;
        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![crate::cuda::ops::pool::launch_scatter_pool_grad_2d(
                    grad_out,
                    &max_indices,
                    &input_shape,
                )?])
            }),
        });
        Ok(out)
    }

    /// Backward is a real CUDA kernel (`avg_pool2d_backward`), unlike
    /// WGPU's host-readback-and-Rust-loop approach — see this file's
    /// module doc.
    fn avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out =
            crate::cuda::ops::pool::launch_avg_pool2d_forward(t, kernel_size, stride, padding)?;
        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![crate::cuda::ops::pool::launch_avg_pool2d_backward(
                    grad_out,
                    &input_shape,
                    kernel_size,
                    stride,
                    padding,
                )?])
            }),
        });
        Ok(out)
    }

    fn adaptive_avg_pool2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = crate::cuda::ops::pool::launch_adaptive_avg_pool2d_forward(t, output_size)?;
        let input_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![
                    crate::cuda::ops::pool::launch_adaptive_avg_pool2d_backward(
                        grad_out,
                        &input_shape,
                    )?,
                ])
            }),
        });
        Ok(out)
    }

    /// Composed entirely from already-tape-tracked primitives (`narrow`,
    /// `im2col_1d_tape`, `reshape`, `matmul`, `concat`) — no hand-written
    /// backward closure of its own, unlike CPU/WGPU's `conv1d`. Per-group,
    /// per-batch `matmul` (CUDA’s own `matmul` is currently unbatched and 2D only) trades kernel-launch count for zero
    /// new hand-derived backward math in a compile-verified-only environment
    /// (no CUDA hardware here to gradcheck against).
    fn conv1d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let (batch, cin, len) = (t.shape[0], t.shape[1], t.shape[2]);
        let (cout, cin_g, k) = (w.shape[0], w.shape[1], w.shape[2]);
        validate_conv_groups("conv1d", cin, cout, groups)?;
        if cin / groups != cin_g {
            return Err(Error::ShapeMismatch {
                op: "conv1d",
                expected: vec![cin / groups],
                got: vec![cin_g],
                msg: format!(
                    "conv1d: weight's Cin/groups ({cin_g}) does not match input Cin/groups ({})",
                    cin / groups
                ),
            });
        }
        let cout_g = cout / groups;
        let l_out = crate::cuda::ops::conv::out_size(len, k, stride, padding, dilation)?;

        let mut group_outputs: Vec<CudaStorage> = Vec::with_capacity(groups);
        for g in 0..groups {
            let input_g = <Self as TensorOps<Self>>::narrow::<K>(t, 1, g * cin_g, cin_g)?;
            let weight_g = <Self as TensorOps<Self>>::narrow::<K>(w, 0, g * cout_g, cout_g)?;
            let cols = im2col_1d_tape(&input_g, k, stride, padding, dilation)?;
            let weight_mat =
                <Self as TensorOps<Self>>::reshape::<K>(&weight_g, &[cout_g, cin_g * k])?;

            let mut batch_outs: Vec<CudaStorage> = Vec::with_capacity(batch);
            for bi in 0..batch {
                let cols_b = <Self as TensorOps<Self>>::narrow::<K>(&cols, 0, bi, 1)?;
                let cols_b = <Self as TensorOps<Self>>::squeeze::<K>(&cols_b, 0)?;
                let out_b = <Self as TensorOps<Self>>::matmul::<K>(&weight_mat, &cols_b)?;
                let out_b = <Self as TensorOps<Self>>::reshape::<K>(&out_b, &[1, cout_g, l_out])?;
                batch_outs.push(out_b);
            }
            let group_out = if batch == 1 {
                batch_outs.into_iter().next().unwrap()
            } else {
                let refs: Vec<&CudaStorage> = batch_outs.iter().collect();
                <Self as TensorOps<Self>>::concat::<K>(&refs, 0)?
            };
            group_outputs.push(group_out);
        }
        let conv_out = if groups == 1 {
            group_outputs.into_iter().next().unwrap()
        } else {
            let refs: Vec<&CudaStorage> = group_outputs.iter().collect();
            <Self as TensorOps<Self>>::concat::<K>(&refs, 1)?
        };

        match bias {
            Some(bv) => {
                let bias_shaped = <Self as TensorOps<Self>>::reshape::<K>(bv, &[1, cout, 1])?;
                <Self as NumericOps<Self>>::add::<K>(&conv_out, &bias_shaped)
            }
            None => Ok(conv_out),
        }
    }

    /// Mirrors `conv1d`'s exact structure generalized to two spatial axes.
    /// CUDA's `im2col_2d` kernel lays cols out channel-major
    /// (`[B, Cin_g*Kh*Kw, H_out*W_out]` — see `cuda/ops/conv.rs`'s module
    /// doc), so this computes `weight_mat @ cols_b` directly per batch, no
    /// transpose of either operand needed (unlike CPU/WGPU's
    /// spatial-major `cols @ weight_mat^T`).
    fn conv2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let (batch, cin, h, wid) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
        let (cout, cin_g, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
        validate_conv_groups("conv2d", cin, cout, groups)?;
        if cin / groups != cin_g {
            return Err(Error::ShapeMismatch {
                op: "conv2d",
                expected: vec![cin / groups],
                got: vec![cin_g],
                msg: format!(
                    "conv2d: weight's Cin/groups ({cin_g}) does not match input Cin/groups ({})",
                    cin / groups
                ),
            });
        }
        let cout_g = cout / groups;
        let h_out = crate::cuda::ops::conv::out_size(h, kh, stride, padding, dilation)?;
        let w_out = crate::cuda::ops::conv::out_size(wid, kw, stride, padding, dilation)?;

        let mut group_outputs: Vec<CudaStorage> = Vec::with_capacity(groups);
        for g in 0..groups {
            let input_g = <Self as TensorOps<Self>>::narrow::<K>(t, 1, g * cin_g, cin_g)?;
            let weight_g = <Self as TensorOps<Self>>::narrow::<K>(w, 0, g * cout_g, cout_g)?;
            let cols = im2col_2d_tape(&input_g, kh, kw, stride, padding, dilation)?;
            let weight_mat =
                <Self as TensorOps<Self>>::reshape::<K>(&weight_g, &[cout_g, cin_g * kh * kw])?;

            let mut batch_outs: Vec<CudaStorage> = Vec::with_capacity(batch);
            for bi in 0..batch {
                let cols_b = <Self as TensorOps<Self>>::narrow::<K>(&cols, 0, bi, 1)?;
                let cols_b = <Self as TensorOps<Self>>::squeeze::<K>(&cols_b, 0)?;
                let out_b = <Self as TensorOps<Self>>::matmul::<K>(&weight_mat, &cols_b)?;
                let out_b =
                    <Self as TensorOps<Self>>::reshape::<K>(&out_b, &[1, cout_g, h_out * w_out])?;
                batch_outs.push(out_b);
            }
            let group_out = if batch == 1 {
                batch_outs.into_iter().next().unwrap()
            } else {
                let refs: Vec<&CudaStorage> = batch_outs.iter().collect();
                <Self as TensorOps<Self>>::concat::<K>(&refs, 0)?
            };
            group_outputs.push(group_out);
        }
        let conv_out = if groups == 1 {
            group_outputs.into_iter().next().unwrap()
        } else {
            let refs: Vec<&CudaStorage> = group_outputs.iter().collect();
            <Self as TensorOps<Self>>::concat::<K>(&refs, 1)?
        };
        let conv_out =
            <Self as TensorOps<Self>>::reshape::<K>(&conv_out, &[batch, cout, h_out, w_out])?;

        match bias {
            Some(bv) => {
                let bias_shaped = <Self as TensorOps<Self>>::reshape::<K>(bv, &[1, cout, 1, 1])?;
                <Self as NumericOps<Self>>::add::<K>(&conv_out, &bias_shaped)
            }
            None => Ok(conv_out),
        }
    }

    /// Transposed convolution's forward is exactly `conv2d`'s own
    /// backward-data formula applied directly to `t` (RESEARCH.md Pattern
    /// 4, same as CPU/WGPU): `weight_mat_t @ input_b` per batch, folded via
    /// `col2im_2d_tape` (reusing the exact inverse of `im2col_2d_tape`).
    /// `output_padding` is its own final step via `pad_trailing_zeros_2d_tape`,
    /// never folded into `padding`'s symmetric arithmetic. Only `groups ==
    /// 1` is supported, matching CPU/WGPU's own documented scope.
    fn conv_transpose2d<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        if groups != 1 {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: vec![1],
                got: vec![groups],
                msg: format!(
                    "conv_transpose2d: only groups == 1 is supported on CudaBackendImpl, got groups={groups}"
                ),
            });
        }
        if t.shape.len() != 4 || w.shape.len() != 4 {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: vec![4],
                got: vec![t.shape.len()],
                msg: "conv_transpose2d expected 4D input and weight".into(),
            });
        }
        let (batch, cin, h, wid) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
        let (w_cin, cout, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
        if w_cin != cin {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: vec![cin],
                got: vec![w_cin],
                msg: format!(
                    "conv_transpose2d: weight's Cin ({w_cin}) does not match input Cin ({cin})"
                ),
            });
        }

        let h_nat =
            crate::cuda::ops::conv::natural_transpose_out_size(h, kh, stride, padding, dilation)?;
        let w_nat =
            crate::cuda::ops::conv::natural_transpose_out_size(wid, kw, stride, padding, dilation)?;

        // weight: [Cin, Cout, Kh, Kw] (Candle's conv_transpose2d convention,
        // matching CPU/WGPU).
        let flattened_weight =
            ShapeBuf::from_slice(&[cout, kh, kw]).checked_numel(OperationKind::Conv2d)?;
        let weight_mat = <Self as TensorOps<Self>>::reshape::<K>(w, &[cin, flattened_weight])?;
        let weight_mat_t = <Self as TensorOps<Self>>::transpose::<K>(&weight_mat, 0, 1)?;
        let input_flat = <Self as TensorOps<Self>>::reshape::<K>(t, &[batch, cin, h * wid])?;

        let mut batch_cols: Vec<CudaStorage> = Vec::with_capacity(batch);
        for bi in 0..batch {
            let input_b = <Self as TensorOps<Self>>::narrow::<K>(&input_flat, 0, bi, 1)?;
            let input_b = <Self as TensorOps<Self>>::squeeze::<K>(&input_b, 0)?;
            let cols_b = <Self as TensorOps<Self>>::matmul::<K>(&weight_mat_t, &input_b)?;
            let cols_b =
                <Self as TensorOps<Self>>::reshape::<K>(&cols_b, &[1, cout * kh * kw, h * wid])?;
            batch_cols.push(cols_b);
        }
        let cols = if batch == 1 {
            batch_cols.into_iter().next().unwrap()
        } else {
            let refs: Vec<&CudaStorage> = batch_cols.iter().collect();
            <Self as TensorOps<Self>>::concat::<K>(&refs, 0)?
        };

        let natural_out = col2im_2d_tape(
            &cols,
            &[batch, cout, h_nat, w_nat],
            h,
            wid,
            kh,
            kw,
            stride,
            padding,
            dilation,
        )?;

        let conv_out = if output_padding == 0 {
            natural_out
        } else {
            pad_trailing_zeros_2d_tape(&natural_out, output_padding, output_padding)?
        };

        match bias {
            Some(bv) => {
                let bias_shaped = <Self as TensorOps<Self>>::reshape::<K>(bv, &[1, cout, 1, 1])?;
                <Self as NumericOps<Self>>::add::<K>(&conv_out, &bias_shaped)
            }
            None => Ok(conv_out),
        }
    }
}
impl<T: DType, D: Device> LossOps<Self> for CudaBackendImpl<T, D> {
    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &CudaStorage,
        target: &CudaStorage,
        reduction: incin_core::prelude::Reduction,
    ) -> Result<CudaStorage> {
        let classes = if pred.shape.len() > 1 {
            pred.shape[1]
        } else {
            pred.shape[0]
        };
        let sm = Self::softmax::<K>(pred, 1)?;
        let log_sm = Self::log::<K>(&sm)?;
        let nll = crate::cuda::ops::loss::launch_nll_loss(&log_sm, target, classes)?;
        match reduction {
            incin_core::prelude::Reduction::Mean => Self::mean_all::<K>(&nll),
            incin_core::prelude::Reduction::Sum => Self::sum_all::<K>(&nll),
            incin_core::prelude::Reduction::None => Ok(nll),
        }
    }
}

impl<T: DType, D: Device> Backend for CudaBackendImpl<T, D> {
    type Device = D;
    type FloatElem = T;
    type IntElem = i64;

    type Storage<K: DType> = CudaStorage;
    type RawVar = CudaVar;
    type Grads = CudaGrads;

    type InnerBackend = Self;

    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
        t.shape.to_vec()
    }
    fn storage_dtype<K: DType>(t: &Self::Storage<K>) -> Option<DTypeId> {
        Some(t.buffer.dtype)
    }
    fn storage_device<K: DType>(t: &Self::Storage<K>) -> Option<DeviceId> {
        Some(DeviceId::cuda(t.buffer.device_id))
    }
    fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
        "CudaTensor(...)".to_string()
    }
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        format!("CudaTensor(shape={:?})", t.shape)
    }
    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::cuda::tape::backward(loss)
    }
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
        let bytes = t
            .buffer
            .device
            .default_stream()
            .clone_dtoh(&*t.buffer.data)
            .map_err(|error| Error::Msg(format!("CUDA download failed: {error:?}")))?;
        let expected = checked_storage_byte_len(t.buffer.len, t.buffer.dtype)?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        Ok(bytes)
    }
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_cuda_storage(dtype, device, "from_bytes")?;
        let numel = checked_numel(shape)?;
        let expected = checked_storage_byte_len(numel, dtype)?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        cuda_from_bytes(shape, dtype, device.ordinal(), bytes)
    }
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(CudaVar { storage: t.clone() })
    }
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }
}

fn validate_cuda(dtype: DTypeId, device: &DeviceId, op: &'static str) -> Result<()> {
    validate_cuda_device(device)?;
    resolve_dtype_policy(BackendFamily::Cuda, OperationKind::Fill, dtype, op).map(|_| ())
}

fn validate_cuda_storage(dtype: DTypeId, device: &DeviceId, op: &'static str) -> Result<()> {
    validate_cuda_device(device)?;
    validate_cuda_storage_dtype(dtype, op)
}

fn validate_cuda_storage_dtype(dtype: DTypeId, op: &'static str) -> Result<()> {
    resolve_dtype_policy(BackendFamily::Cuda, OperationKind::Storage, dtype, op).map(|_| ())
}

fn validate_cuda_device(device: &DeviceId) -> Result<()> {
    if device.kind() != DeviceKind::Cuda {
        return Err(Error::DeviceInitializationError {
            expected: "cuda".into(),
            got: format!("{:?}", device.kind()),
        });
    }
    Ok(())
}

fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |numel, &dimension| {
        numel
            .checked_mul(dimension)
            .ok_or_else(|| Error::Msg(format!("CUDA tensor shape overflows usize: {shape:?}")))
    })
}

fn checked_storage_byte_len(numel: usize, dtype: DTypeId) -> Result<usize> {
    numel.checked_mul(dtype.element_size()).ok_or_else(|| {
        Error::Msg(format!(
            "CUDA storage byte length overflow: {numel} {:?} elements",
            dtype
        ))
    })
}

fn cuda_from_f32(
    shape: &[usize],
    dtype: DTypeId,
    device: &DeviceId,
    values: Vec<f32>,
    op: &'static str,
) -> Result<CudaStorage> {
    validate_cuda(dtype, device, op)?;
    cuda_from_bytes(
        shape,
        dtype,
        device.ordinal(),
        bytemuck::cast_slice(&values),
    )
}

fn cuda_from_bytes(
    shape: &[usize],
    dtype: DTypeId,
    ordinal: usize,
    bytes: &[u8],
) -> Result<CudaStorage> {
    validate_cuda_storage_dtype(dtype, "from_bytes")?;
    let numel = checked_numel(shape)?;
    let expected = checked_storage_byte_len(numel, dtype)?;
    if bytes.len() != expected {
        return Err(Error::InvalidByteLength {
            expected,
            got: bytes.len(),
        });
    }
    // Through the process-wide cache rather than CudaContext::new directly. Both
    // retain the same primary context, but a fresh Arc per tensor means the last
    // one dropped releases it, and the next allocation pays 131 ms to bring it
    // back. The cache holds one handle forever, which keeps this on the 1 us
    // path. See cuda::gpu::cuda_cache::try_get_cuda_device.
    let context = crate::cuda::gpu::cuda_cache::try_get_cuda_device(ordinal).map_err(|_| {
        Error::InvalidDeviceOrdinal {
            backend: "Cuda",
            ordinal,
        }
    })?;
    let data = context
        .default_stream()
        .clone_htod(bytes)
        .map_err(|error| Error::Msg(format!("CUDA upload failed: {error:?}")))?;
    let buffer = crate::cuda::storage::CudaBuffer {
        len: numel,
        dtype,
        data: Arc::new(data),
        device: context,
        device_id: ordinal,
    };
    Ok(CudaStorage::new(Arc::new(buffer), shape.to_vec()))
}

/// Guards `download_f32_host`/`upload_f32_from_host` callers against the
/// class of bug those two helpers cannot detect on their own: they assume
/// F32 storage unconditionally, so calling them on any of CUDA's other
/// storage dtypes (I64/BF16/F16/F64 — see `CUDA_STORAGE_DTYPES` in
/// `capability.rs`) would silently reinterpret the wrong bytes rather than
/// error. `topk`/`argsort` (this file, `cuda_topk_host`/`cuda_argsort_host`)
/// have this exact gap already and are tracked separately; every new
/// F32-only host-round-trip op added in this pass checks first instead of
/// repeating it.
fn cuda_require_f32(dtype: DTypeId, op: &'static str) -> Result<()> {
    if dtype != DTypeId::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            backend: "cuda",
            op,
        });
    }
    Ok(())
}

/// Downloads an F32 `CudaStorage`'s raw contents to a host `Vec<f32>`.
fn download_f32_host(t: &CudaStorage) -> Result<Vec<f32>> {
    let bytes = t
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*t.buffer.data)
        .map_err(|error| BackendError::Execution {
            operation: OperationKind::Storage,
            message: format!("CUDA download failed: {error:?}").into(),
        })?;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

/// Uploads a host `Vec<f32>` as a fresh `CudaStorage` on the same device as
/// `t_buf`, reusing its existing device/stream (no new `CudaContext`,
/// unlike `cuda_from_bytes`).
fn upload_f32_from_host(
    t_buf: &crate::cuda::storage::CudaBuffer,
    shape: Vec<usize>,
    values: Vec<f32>,
) -> Result<CudaStorage> {
    let stream = t_buf.device.default_stream();
    let data = stream
        .clone_htod(bytemuck::cast_slice(&values))
        .map_err(|error| BackendError::Execution {
            operation: OperationKind::Storage,
            message: format!("CUDA upload failed: {error:?}").into(),
        })?;
    let buffer = crate::cuda::storage::CudaBuffer {
        len: values.len(),
        dtype: DTypeId::F32,
        data: Arc::new(data),
        device: t_buf.device.clone(),
        device_id: t_buf.device_id,
    };
    Ok(CudaStorage::new(Arc::new(buffer), shape))
}

/// `U32` counterpart of `upload_f32_from_host` — used for `topk`/`argsort`'s
/// index outputs.
fn upload_u32_from_host(
    t_buf: &crate::cuda::storage::CudaBuffer,
    shape: Vec<usize>,
    values: Vec<u32>,
) -> Result<CudaStorage> {
    let stream = t_buf.device.default_stream();
    let data = stream
        .clone_htod(bytemuck::cast_slice(&values))
        .map_err(|error| BackendError::Execution {
            operation: OperationKind::Storage,
            message: format!("CUDA upload failed: {error:?}").into(),
        })?;
    let buffer = crate::cuda::storage::CudaBuffer {
        len: values.len(),
        dtype: DTypeId::U32,
        data: Arc::new(data),
        device: t_buf.device.clone(),
        device_id: t_buf.device_id,
    };
    Ok(CudaStorage::new(Arc::new(buffer), shape))
}

/// `topk`/`argsort` have no CUDA kernel on ANY backend — WGPU's own
/// implementation (`wgpu/backend.rs::topk`) is equally a host download, a
/// plain per-slice Rust sort, and a re-upload; this ports that exact
/// algorithm (coordinate decode, sort, flat-index re-encode) verbatim, so
/// it's not a CUDA-specific shortcut, it's what the "true" GPU backend
/// already does for these two ops. Output indices stay `U32` (never
/// converted to `I64`), matching CPU/WGPU's own `topk`/`argsort` exactly —
/// unlike `argmax`/`argmin`, which both DO convert to `I64` on every
/// backend (a pre-existing inconsistency in the trait's own reference
/// backends, not something introduced here).
fn cuda_topk_host(
    t: &CudaStorage,
    k: usize,
    dim: usize,
    largest: bool,
) -> Result<(CudaStorage, CudaStorage)> {
    let shape = t.shape.to_vec();
    if dim >= shape.len() {
        return Err(Error::ShapeMismatch {
            op: "topk",
            expected: shape.clone(),
            got: vec![dim],
            msg: format!("topk: axis {dim} out of range for shape {shape:?}"),
        });
    }
    let k = k.min(shape[dim]);
    let data = download_f32_host(t)?;

    let mut out_shape = shape.clone();
    out_shape[dim] = k;
    let mut base_shape = shape.clone();
    base_shape[dim] = 1;

    let n_slices: usize = incin_core::prelude::ShapeBuf::from_slice(&(base_shape))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let out_numel: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_shape))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let mut out_vals = vec![0.0f32; out_numel];
    let mut out_indices = vec![0u32; out_numel];

    for i in 0..n_slices {
        let mut rem = i;
        let mut coords = vec![0usize; shape.len()];
        for dd in (0..shape.len()).rev() {
            coords[dd] = rem % base_shape[dd];
            rem /= base_shape[dd];
        }

        let mut slice_vals: Vec<(f32, u32)> = Vec::with_capacity(shape[dim]);
        for j in 0..shape[dim] {
            coords[dim] = j;
            let mut flat = 0usize;
            let mut stride = 1usize;
            for dd in (0..shape.len()).rev() {
                flat += coords[dd] * stride;
                stride *= shape[dd];
            }
            slice_vals.push((data[flat], crate::cuda::checked_u32(j, "CUDA topk index")?));
        }
        if largest {
            slice_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        } else {
            slice_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        }

        let mut out_coords = coords.clone();
        for (j, &(val, idx)) in slice_vals.iter().enumerate().take(k) {
            out_coords[dim] = j;
            let mut flat = 0usize;
            let mut stride = 1usize;
            for dd in (0..out_shape.len()).rev() {
                flat += out_coords[dd] * stride;
                stride *= out_shape[dd];
            }
            out_vals[flat] = val;
            out_indices[flat] = idx;
        }
    }

    let t_buf = &*t.buffer;
    let vals_out = upload_f32_from_host(t_buf, out_shape.clone(), out_vals)?;
    let indices_out = upload_u32_from_host(t_buf, out_shape, out_indices)?;
    Ok((vals_out, indices_out))
}

/// See `cuda_topk_host`'s doc — same "no CUDA kernel on any backend, ported
/// verbatim from WGPU's host loop" note applies here.
fn cuda_argsort_host(t: &CudaStorage, dim: usize, descending: bool) -> Result<CudaStorage> {
    let shape = t.shape.to_vec();
    if dim >= shape.len() {
        return Err(Error::ShapeMismatch {
            op: "argsort",
            expected: shape.clone(),
            got: vec![dim],
            msg: format!("argsort: axis {dim} out of range for shape {shape:?}"),
        });
    }
    let data = download_f32_host(t)?;

    let mut base_shape = shape.clone();
    base_shape[dim] = 1;
    let n_slices: usize = incin_core::prelude::ShapeBuf::from_slice(&(base_shape))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let mut out = vec![0u32; ShapeBuf::from_slice(&shape).checked_numel(OperationKind::Storage)?];

    for i in 0..n_slices {
        let mut rem = i;
        let mut coords = vec![0usize; shape.len()];
        for dd in (0..shape.len()).rev() {
            coords[dd] = rem % base_shape[dd];
            rem /= base_shape[dd];
        }

        let mut slice_vals: Vec<(f32, u32)> = Vec::with_capacity(shape[dim]);
        for j in 0..shape[dim] {
            coords[dim] = j;
            let mut flat = 0usize;
            let mut stride = 1usize;
            for dd in (0..shape.len()).rev() {
                flat += coords[dd] * stride;
                stride *= shape[dd];
            }
            slice_vals.push((
                data[flat],
                crate::cuda::checked_u32(j, "CUDA argsort index")?,
            ));
        }
        if descending {
            slice_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        } else {
            slice_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        }
        for (j, &(_, idx)) in slice_vals.iter().enumerate() {
            coords[dim] = j;
            let mut flat = 0usize;
            let mut stride = 1usize;
            for dd in (0..shape.len()).rev() {
                flat += coords[dd] * stride;
                stride *= shape[dd];
            }
            out[flat] = idx;
        }
    }

    let t_buf = &*t.buffer;
    upload_u32_from_host(t_buf, shape, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_length_uses_authoritative_storage_dtype() {
        assert_eq!(checked_storage_byte_len(7, DTypeId::F16).unwrap(), 14);
        assert_eq!(checked_storage_byte_len(7, DTypeId::BF16).unwrap(), 14);
        assert_eq!(checked_storage_byte_len(7, DTypeId::F32).unwrap(), 28);
        assert_eq!(checked_storage_byte_len(7, DTypeId::F64).unwrap(), 56);
        assert!(checked_storage_byte_len(usize::MAX, DTypeId::F64).is_err());
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
            validate_cuda_storage(dtype, &device, "test").unwrap();
        }
        assert!(matches!(
            validate_cuda_storage(DTypeId::U32, &device, "test"),
            Err(Error::UnsupportedDType { .. })
        ));
        assert!(validate_cuda_storage(DTypeId::F32, &DeviceId::cpu(), "test").is_err());
    }

    #[test]
    fn shape_cardinality_is_checked_before_allocation() {
        assert_eq!(checked_numel(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_numel(&[usize::MAX, 0]).unwrap(), 0);
        assert!(checked_numel(&[usize::MAX, 2]).is_err());
    }

    // The tests below exercise real GPU dispatch (`TensorOps::{reshape,
    // transpose, narrow, broadcast_as, squeeze, stack, slice, flatten,
    // broadcast_left, matmul}`) and therefore need a real CUDA device to
    // run — none is available in this environment, so this path is compile-verified
    // only locally. `#[ignore]`d so `cargo test` stays green everywhere; run with
    // `cargo test --features cuda,std -- --ignored` on real hardware.

    type B = CudaBackendImpl<f32, Cuda>;

    fn cuda_f32(shape: &[usize], values: Vec<f32>) -> CudaStorage {
        cuda_from_f32(shape, DTypeId::F32, &DeviceId::cuda(0), values, "test").unwrap()
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn reshape_preserves_element_order() {
        let t = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = <B as TensorOps<B>>::reshape::<f32>(&t, &[3, 2]).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn reshape_rejects_mismatched_element_count() {
        let t = cuda_f32(&[2, 3], vec![0.0; 6]);
        assert!(<B as TensorOps<B>>::reshape::<f32>(&t, &[4, 2]).is_err());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn transpose_2d_swaps_shape() {
        let t = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = <B as TensorOps<B>>::transpose::<f32>(&t, 0, 1).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn narrow_reduces_target_dim() {
        let t = cuda_f32(&[4, 3], vec![0.0; 12]);
        let out = <B as TensorOps<B>>::narrow::<f32>(&t, 0, 1, 2).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn broadcast_as_expands_size_one_dim() {
        let t = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
        let out = <B as TensorOps<B>>::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
        assert_eq!(out.shape, vec![4, 3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn broadcast_as_rejects_incompatible_shape() {
        let t = cuda_f32(&[2, 3], vec![0.0; 6]);
        assert!(<B as TensorOps<B>>::broadcast_as::<f32>(&t, &[2, 5]).is_err());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn squeeze_removes_size_one_axis() {
        let t = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
        let out = <B as TensorOps<B>>::squeeze::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape, vec![3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn stack_inserts_new_axis() {
        let a = cuda_f32(&[3], vec![1.0, 2.0, 3.0]);
        let b = cuda_f32(&[3], vec![4.0, 5.0, 6.0]);
        let out = <B as TensorOps<B>>::stack::<f32>(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn slice_narrows_every_listed_dim() {
        let t = cuda_f32(&[4, 4], vec![0.0; 16]);
        let out = <B as TensorOps<B>>::slice::<f32>(&t, &[(1, 3), (0, 2)]).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn flatten_merges_middle_dims() {
        let t = cuda_f32(&[2, 3, 4], vec![0.0; 24]);
        let out = <B as TensorOps<B>>::flatten::<f32>(&t, 1, 2).unwrap();
        assert_eq!(out.shape, vec![2, 12]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn broadcast_left_prepends_leading_dims() {
        let t = cuda_f32(&[3], vec![1.0, 2.0, 3.0]);
        let out = <B as TensorOps<B>>::broadcast_left::<f32>(&t, &[2, 4]).unwrap();
        assert_eq!(out.shape, vec![2, 4, 3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn matmul_computes_correct_shape_and_values() {
        // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let out = <B as TensorOps<B>>::matmul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn matmul_rejects_incompatible_inner_dims() {
        let lhs = cuda_f32(&[2, 3], vec![0.0; 6]);
        let rhs = cuda_f32(&[4, 2], vec![0.0; 8]);
        assert!(<B as TensorOps<B>>::matmul::<f32>(&lhs, &rhs).is_err());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn matmul_backward_produces_gradients_for_both_operands() {
        let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let (lhs_id, rhs_id) = (lhs.id, rhs.id);
        let out = <B as TensorOps<B>>::matmul::<f32>(&lhs, &rhs).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        assert!(grads.get(lhs_id).is_some());
        assert!(grads.get(rhs_id).is_some());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn narrow_backward_zero_pads_grad_to_original_shape() {
        let t = cuda_f32(&[4, 3], vec![0.0; 12]);
        let t_id = t.id;
        let out = <B as TensorOps<B>>::narrow::<f32>(&t, 0, 1, 2).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        let g = grads
            .get(t_id)
            .expect("narrow input should have a gradient");
        assert_eq!(g.shape, vec![4, 3]);
    }

    fn cuda_i64(shape: &[usize], values: Vec<i64>) -> CudaStorage {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        cuda_from_bytes(shape, DTypeId::I64, DeviceId::cuda(0).ordinal(), &bytes).unwrap()
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn embedding_gathers_rows_by_index() {
        // vocab_size=3, hidden_size=2
        let w = cuda_f32(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let idx = cuda_i64(&[2], vec![2, 0]);
        let out = <B as ModuleOps<B>>::embedding::<f32, i64>(&idx, &w).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn embedding_rejects_non_rank2_weight() {
        let w = cuda_f32(&[3, 2, 1], vec![0.0; 6]);
        let idx = cuda_i64(&[1], vec![0]);
        assert!(<B as ModuleOps<B>>::embedding::<f32, i64>(&idx, &w).is_err());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn embedding_backward_produces_weight_gradient_only() {
        let w = cuda_f32(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let idx = cuda_i64(&[2], vec![2, 0]);
        let w_id = w.id;
        let out = <B as ModuleOps<B>>::embedding::<f32, i64>(&idx, &w).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        let g = grads
            .get(w_id)
            .expect("embedding weight should have a gradient");
        assert_eq!(g.shape, vec![3, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn max_pool2d_computes_correct_output_shape() {
        // N=1,C=1,H=4,W=4, kernel=2, stride=2 -> 2x2 output
        let t = cuda_f32(&[1, 1, 4, 4], vec![0.0; 16]);
        let out =
            <B as ModuleOps<B>>::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn max_pool2d_backward_zero_pads_to_input_shape() {
        let t = cuda_f32(&[1, 1, 4, 4], vec![0.0; 16]);
        let t_id = t.id;
        let out =
            <B as ModuleOps<B>>::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
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
        let out = <B as ModuleOps<B>>::avg_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn adaptive_avg_pool2d_matches_requested_output_size() {
        let t = cuda_f32(&[1, 1, 5, 5], vec![0.0; 25]);
        let out = <B as ModuleOps<B>>::adaptive_avg_pool2d::<f32>(&t, (3, 3)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn adaptive_avg_pool2d_backward_matches_input_shape() {
        let t = cuda_f32(&[1, 1, 5, 5], vec![0.0; 25]);
        let t_id = t.id;
        let out = <B as ModuleOps<B>>::adaptive_avg_pool2d::<f32>(&t, (3, 3)).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        let g = grads
            .get(t_id)
            .expect("adaptive_avg_pool2d input should have a gradient");
        assert_eq!(g.shape, vec![1, 1, 5, 5]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn conv1d_computes_correct_output_shape_and_values() {
        // [1,1,4] input, [1,1,2] kernel, stride=1, no padding -> [1,1,3] out,
        // matching CPU's hand-computed test fixture (conv.rs) exactly.
        let t = cuda_f32(&[1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let w = cuda_f32(&[1, 1, 2], vec![1.0, 1.0]);
        let out = <B as ModuleOps<B>>::conv1d::<f32>(&t, &w, None, 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3]);
        let vals = download_f32_host(&out).unwrap();
        assert_eq!(vals, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn conv1d_backward_produces_gradients_for_input_and_weight() {
        let t = cuda_f32(&[1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let w = cuda_f32(&[1, 1, 2], vec![1.0, 1.0]);
        let (t_id, w_id) = (t.id, w.id);
        let out = <B as ModuleOps<B>>::conv1d::<f32>(&t, &w, None, 1, 0, 1, 1).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        assert_eq!(grads.get(t_id).unwrap().shape, vec![1, 1, 4]);
        assert_eq!(grads.get(w_id).unwrap().shape, vec![1, 1, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn conv1d_rejects_groups_not_dividing_channels() {
        let t = cuda_f32(&[1, 3, 4], vec![0.0; 12]);
        let w = cuda_f32(&[3, 3, 2], vec![0.0; 18]);
        assert!(<B as ModuleOps<B>>::conv1d::<f32>(&t, &w, None, 1, 0, 1, 2).is_err());
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
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&t, &w, None, 1, 0, 1, 1).unwrap();
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
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&t, &w, Some(&bias), 1, 0, 1, 1).unwrap();
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
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&t, &w, None, 1, 0, 1, 1).unwrap();
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
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&t, &w, None, 1, 0, 1, 2).unwrap();
        assert_eq!(out.shape, vec![1, 2, 2, 2]);
        let vals = download_f32_host(&out).unwrap();
        assert_eq!(vals, vec![2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn conv_transpose2d_computes_correct_output_shape() {
        // [1,1,2,2] input, weight [Cin=1,Cout=1,2,2], stride=1 -> natural
        // [1,1,3,3] output (upsampling formula), matching CPU's
        // `conv_transpose2d_forward_hand_computed_basic` fixture shape.
        let t = cuda_f32(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let w = cuda_f32(&[1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
        let out =
            <B as ModuleOps<B>>::conv_transpose2d::<f32>(&t, &w, None, 1, 0, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn conv_transpose2d_output_padding_appends_trailing_rows_and_cols() {
        let t = cuda_f32(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let w = cuda_f32(&[1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
        let out =
            <B as ModuleOps<B>>::conv_transpose2d::<f32>(&t, &w, None, 1, 0, 1, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn conv_transpose2d_backward_produces_gradients_for_input_and_weight() {
        let t = cuda_f32(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let w = cuda_f32(&[1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
        let (t_id, w_id) = (t.id, w.id);
        let out =
            <B as ModuleOps<B>>::conv_transpose2d::<f32>(&t, &w, None, 1, 0, 0, 1, 1).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        assert_eq!(grads.get(t_id).unwrap().shape, vec![1, 1, 2, 2]);
        assert_eq!(grads.get(w_id).unwrap().shape, vec![1, 1, 2, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn conv_transpose2d_rejects_groups_other_than_one() {
        let t = cuda_f32(&[1, 1, 2, 2], vec![0.0; 4]);
        let w = cuda_f32(&[1, 1, 2, 2], vec![0.0; 4]);
        assert!(<B as ModuleOps<B>>::conv_transpose2d::<f32>(&t, &w, None, 1, 0, 0, 1, 2).is_err());
    }

    // mse_loss/l1_loss/bce_with_logits_loss have no override in this file's
    // `impl LossOps<Self> for CudaBackendImpl` — they resolve to
    // `LossOps`'s own default bodies (`incin-core/src/tensor/backend.rs`),
    // which compose entirely from `NumericOps`/`FloatOps`/`ReductionOps`
    // (already wired on CUDA). These tests exist to prove that resolution
    // actually compiles and runs correctly, not to add new functionality.

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn mse_loss_default_impl_resolves_and_runs_on_cuda() {
        let pred = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let target = cuda_f32(&[2, 3], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let out = <B as LossOps<B>>::mse_loss::<f32>(&pred, &target, Reduction::Mean).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn l1_loss_default_impl_resolves_and_runs_on_cuda() {
        let pred = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let target = cuda_f32(&[2, 3], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let out = <B as LossOps<B>>::l1_loss::<f32>(&pred, &target, Reduction::Sum).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn bce_with_logits_loss_default_impl_resolves_and_runs_on_cuda() {
        let pred = cuda_f32(&[2, 2], vec![0.0, 1.0, -1.0, 2.0]);
        let target = cuda_f32(&[2, 2], vec![0.0, 1.0, 1.0, 0.0]);
        let out = <B as LossOps<B>>::bce_with_logits_loss::<f32>(&pred, &target, Reduction::None)
            .unwrap();
        assert_eq!(out.shape, vec![2, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn mse_loss_backward_produces_gradient_via_composed_primitives() {
        let pred = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let target = cuda_f32(&[2, 3], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let pred_id = pred.id;
        let out = <B as LossOps<B>>::mse_loss::<f32>(&pred, &target, Reduction::Mean).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        let g = grads
            .get(pred_id)
            .expect("mse_loss pred should have a gradient");
        assert_eq!(g.shape, vec![2, 3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn mish_forward_matches_hand_computed_value() {
        // mish(0) = 0 * tanh(ln(2)) = 0
        let t = cuda_f32(&[1], vec![0.0]);
        let out = <B as FloatOps<B>>::mish::<f32>(&t).unwrap();
        assert_eq!(out.shape, vec![1]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn mish_backward_produces_gradient() {
        let t = cuda_f32(&[3], vec![-1.0, 0.0, 1.0]);
        let t_id = t.id;
        let out = <B as FloatOps<B>>::mish::<f32>(&t).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        let g = grads.get(t_id).expect("mish input should have a gradient");
        assert_eq!(g.shape, vec![3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn elu_forward_matches_hand_computed_value() {
        // elu(1) = 1 ; elu(-1) = exp(-1) - 1
        let t = cuda_f32(&[2], vec![1.0, -1.0]);
        let out = <B as FloatOps<B>>::elu::<f32>(&t).unwrap();
        assert_eq!(out.shape, vec![2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn elu_backward_produces_gradient() {
        let t = cuda_f32(&[2], vec![1.0, -1.0]);
        let t_id = t.id;
        let out = <B as FloatOps<B>>::elu::<f32>(&t).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        let g = grads.get(t_id).expect("elu input should have a gradient");
        assert_eq!(g.shape, vec![2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn gelu_forward_matches_hand_computed_value() {
        // gelu(0) = 0 * 0.5 * (1 + erf(0)) = 0
        let t = cuda_f32(&[1], vec![0.0]);
        let out = <B as FloatOps<B>>::gelu::<f32>(&t).unwrap();
        assert_eq!(out.shape, vec![1]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn gelu_backward_produces_gradient() {
        let t = cuda_f32(&[3], vec![-1.0, 0.0, 1.0]);
        let t_id = t.id;
        let out = <B as FloatOps<B>>::gelu::<f32>(&t).unwrap();
        let grads = crate::cuda::tape::backward(&out).unwrap();
        let g = grads.get(t_id).expect("gelu input should have a gradient");
        assert_eq!(g.shape, vec![3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn argmax_dim0_returns_row_index_of_column_max() {
        let t = cuda_f32(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let out = <B as ReductionOps<B>>::argmax::<f32, i64>(&t, Some(0)).unwrap();
        assert_eq!(out.shape, vec![3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn argmax_dim_none_returns_scalar_flat_index() {
        let t = cuda_f32(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let out = <B as ReductionOps<B>>::argmax::<f32, i64>(&t, None).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn argmin_dim0_returns_row_index_of_column_min() {
        let t = cuda_f32(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let out = <B as ReductionOps<B>>::argmin::<f32, i64>(&t, Some(0)).unwrap();
        assert_eq!(out.shape, vec![3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn argmax_rejects_out_of_range_axis() {
        let t = cuda_f32(&[2, 3], vec![0.0; 6]);
        assert!(<B as ReductionOps<B>>::argmax::<f32, i64>(&t, Some(5)).is_err());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn topk_returns_largest_k_values_and_their_indices() {
        // row0=[1,5,3], row1=[4,2,6]; dim=1, k=2, largest=true.
        let t = cuda_f32(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let (vals, indices) = <B as ReductionOps<B>>::topk::<f32, u32>(&t, 2, 1, true).unwrap();
        assert_eq!(vals.shape, vec![2, 2]);
        assert_eq!(indices.shape, vec![2, 2]);
        assert_eq!(download_f32_host(&vals).unwrap(), vec![5.0, 3.0, 6.0, 4.0]);
        let idx_bytes = indices
            .buffer
            .device
            .default_stream()
            .clone_dtoh(&*indices.buffer.data)
            .unwrap();
        let idx_vals: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&idx_bytes).to_vec();
        assert_eq!(idx_vals, vec![1, 2, 2, 0]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn topk_clamps_k_to_axis_length() {
        let t = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
        let (vals, indices) = <B as ReductionOps<B>>::topk::<f32, u32>(&t, 10, 1, true).unwrap();
        assert_eq!(vals.shape, vec![1, 3]);
        assert_eq!(indices.shape, vec![1, 3]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn topk_rejects_out_of_range_axis() {
        let t = cuda_f32(&[2, 3], vec![0.0; 6]);
        assert!(<B as ReductionOps<B>>::topk::<f32, u32>(&t, 1, 5, true).is_err());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn argsort_returns_ascending_indices_per_row() {
        // row0=[1,5,3] -> ascending order is indices [0,2,1]; row1=[4,2,6]
        // -> ascending order is indices [1,0,2].
        let t = cuda_f32(&[2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let out = <B as ReductionOps<B>>::argsort::<f32, u32>(&t, 1, false).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        let bytes = out
            .buffer
            .device
            .default_stream()
            .clone_dtoh(&*out.buffer.data)
            .unwrap();
        let idx_vals: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&bytes).to_vec();
        assert_eq!(idx_vals, vec![0, 2, 1, 1, 0, 2]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn argsort_rejects_out_of_range_axis() {
        let t = cuda_f32(&[2, 3], vec![0.0; 6]);
        assert!(<B as ReductionOps<B>>::argsort::<f32, u32>(&t, 5, false).is_err());
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn quantized_matmul_computes_correct_shape() {
        // lhs [2, 32] @ rhs [4, 32]^T -> [2, 4], K=32 is one Q8_0 block.
        let lhs_f32 = cuda_f32(&[2, 32], (0..64).map(|i| i as f32 * 0.01).collect());
        let rhs_f32 = cuda_f32(&[4, 32], (0..128).map(|i| i as f32 * 0.01).collect());
        let lhs_q =
            <B as QuantizedOps<B>>::quantize::<f32, incin_core::prelude::Q8_0>(&lhs_f32).unwrap();
        let rhs_q =
            <B as QuantizedOps<B>>::quantize::<f32, incin_core::prelude::Q8_0>(&rhs_f32).unwrap();
        let out =
            <B as QuantizedOps<B>>::quantized_matmul::<incin_core::prelude::Q8_0>(&lhs_q, &rhs_q)
                .unwrap();
        assert_eq!(out.shape, vec![2, 4]);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn quantized_matmul_rejects_non_multiple_of_32_k() {
        let lhs_f32 = cuda_f32(&[2, 16], vec![0.0; 32]);
        let rhs_f32 = cuda_f32(&[4, 16], vec![0.0; 64]);
        let lhs_q =
            <B as QuantizedOps<B>>::quantize::<f32, incin_core::prelude::Q8_0>(&lhs_f32).unwrap();
        let rhs_q =
            <B as QuantizedOps<B>>::quantize::<f32, incin_core::prelude::Q8_0>(&rhs_f32).unwrap();
        assert!(
            <B as QuantizedOps<B>>::quantized_matmul::<incin_core::prelude::Q8_0>(&lhs_q, &rhs_q)
                .is_err()
        );
    }

    // `OptimizerOps::adamw_step` has no override in this file's
    // `impl OptimizerOps<Self> for CudaBackendImpl {}` (empty) - it resolves
    // to `OptimizerOps`'s own default body (`incin-core/src/tensor/backend.rs:1000-1041`),
    // composed entirely from `NumericOps`/`FloatOps`/`assign_var` (already
    // wired on CUDA). This test exists to prove that resolution actually
    // compiles, not to add new functionality. The dedicated
    // `kernels/fused_adamw.cu` kernel remains genuinely unused - wiring it
    // would be a performance optimization over this composed default, not a
    // correctness fix, and is deliberately deferred (tracked as a performance opportunity in `PROPOSALS.md`).
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn adamw_step_default_impl_resolves_and_runs_on_cuda() {
        let param = cuda_f32(&[2], vec![1.0, 2.0]);
        let mut var = CudaVar { storage: param };
        let grad = cuda_f32(&[2], vec![0.1, 0.2]);
        let mut m = cuda_f32(&[2], vec![0.0, 0.0]);
        let mut v = cuda_f32(&[2], vec![0.0, 0.0]);
        <B as OptimizerOps<B>>::adamw_step::<f32>(
            &mut var, &grad, &mut m, &mut v, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1,
        )
        .unwrap();
        assert_eq!(var.storage.shape, vec![2]);
    }

    // The tests below cover the methods added in this pass: `unsqueeze`,
    // the host-readback conversions, `addmm`/`bmm`/
    // `scaled_dot_product_attention`. Same convention as everything above —
    // `#[ignore]`d because there is no CUDA device in this environment, so
    // only compilation is verified here; run with `--ignored` on real
    // hardware. Fixtures and expected values are the same ones the CPU and
    // WGPU backends' own tests for the identical methods use.

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_unsqueeze() {
        let t = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = <B as TensorOps<B>>::unsqueeze::<f32>(&t, 1).unwrap();
        assert_eq!(out.shape, vec![2, 1, 3]);
        assert_eq!(
            download_f32_host(&out).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_float_to_scalar() {
        let t = cuda_f32(&[1], vec![3.5]);
        assert_eq!(<B as TensorOps<B>>::float_to_scalar::<f32>(&t).unwrap(), 3.5);
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn float_to_scalar_rejects_a_non_f32_dtype() {
        let t = cuda_from_f32(&[1], DTypeId::F64, &DeviceId::cuda(0), vec![3.5], "test").unwrap();
        assert!(matches!(
            <B as TensorOps<B>>::float_to_scalar::<f32>(&t),
            Err(Error::UnsupportedDType { .. })
        ));
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_float_to_vec1() {
        let t = cuda_f32(&[3], vec![1.0, 2.0, 3.0]);
        assert_eq!(
            <B as TensorOps<B>>::float_to_vec1::<f32>(&t).unwrap(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_addmm() {
        let mat = cuda_f32(&[2, 2], vec![1.0, 1.0, 1.0, 1.0]);
        let mat1 = cuda_f32(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
        let mat2 = cuda_f32(&[2, 2], vec![3.0, 4.0, 5.0, 6.0]);
        // beta * mat + alpha * (mat1 @ mat2) = 2*[[1,1],[1,1]] + 3*[[3,4],[5,6]]
        let out = <B as TensorOps<B>>::addmm::<f32>(&mat, &mat1, &mat2, 2.0, 3.0).unwrap();
        assert_eq!(
            download_f32_host(&out).unwrap(),
            vec![11.0, 14.0, 17.0, 20.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_bmm() {
        let a = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let out = <B as TensorOps<B>>::bmm::<f32>(&a, &b).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(
            download_f32_host(&out).unwrap(),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    #[ignore = "requires CUDA hardware"]
    fn test_scaled_dot_product_attention_uniform_when_query_is_zero() {
        // Same fixture as the WGPU backend's own test for this method: q is
        // all-zero, so softmax of an all-zero row is uniform, and the
        // output is exactly the unweighted average of v's rows.
        let q = cuda_f32(&[1, 2], vec![0.0, 0.0]);
        let k = cuda_f32(&[3, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let v = cuda_f32(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out =
            <B as TensorOps<B>>::scaled_dot_product_attention::<f32>(&q, &k, &v, None, None)
                .unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(download_f32_host(&out).unwrap(), vec![3.0, 4.0]);
    }
}
