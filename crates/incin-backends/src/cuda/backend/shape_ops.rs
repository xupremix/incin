//! Structural and shape-changing CUDA operations: concat, reshape,
//! transpose, matmul, and the other movement/layout kernels.

#![allow(dead_code)]

use super::*;
use crate::cuda::storage::CudaBuffer;
use incin_core::shapes::OperationKind;

impl<D: Device> CudaBackendImpl<D> {
    /// Join operands along an existing axis. Backward splits the incoming
    /// gradient back into one segment per operand, at the same offsets the
    /// forward join used, via `narrow` - the same decomposition CPU's
    /// `concat_storage` backward relies on.
    pub(crate) fn concat<K: DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let storage_refs: Vec<&CudaStorage> = tensors.iter().map(|&t| t as &CudaStorage).collect();
        let out = crate::cuda::ops::shape::launch_concat(&storage_refs, dim)?;
        let input_ids: Vec<_> = storage_refs.iter().map(|t| t.id).collect();
        let segments: Vec<usize> = storage_refs
            .iter()
            .map(|t| t.shape.get(dim).copied().unwrap_or(0))
            .collect();
        let out_id = out.id;
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids,
            backward: Box::new(move |grad_out: &CudaStorage| {
                let mut start = 0usize;
                let mut grads = Vec::with_capacity(segments.len());
                for &len in &segments {
                    grads.push(crate::cuda::ops::shape::launch_narrow(
                        grad_out, dim, start, len,
                    )?);
                    start += len;
                }
                Ok(grads)
            }),
        });
        Ok(out)
    }

    /// Composed from `reshape` (zero new tape entries - matches `squeeze`).
    pub(crate) fn unsqueeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
        let mut target_shape = t.shape.to_vec();
        if dim <= target_shape.len() {
            target_shape.insert(dim, 1);
        } else {
            target_shape.push(1);
        }
        Self::reshape::<K>(t, &target_shape)
    }

    /// Collapse the inclusive axis range `[start_dim, end_dim]` into one
    /// axis. Composed from `reshape`, like `squeeze`/`unsqueeze`.
    pub(crate) fn flatten<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
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
        let merged = checked_numel(&t.shape[start_dim..=end_dim])?;
        let mut target_shape = t.shape[..start_dim].to_vec();
        target_shape.push(merged);
        target_shape.extend_from_slice(&t.shape[end_dim + 1..]);
        Self::reshape::<K>(t, &target_shape)
    }

    /// Prepend `shape` to the operand's own shape and broadcast into it.
    /// Composed from `broadcast_as` (zero new tape entries).
    pub(crate) fn broadcast_left<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
        let mut target_shape = shape.to_vec();
        target_shape.extend_from_slice(&t.shape);
        Self::broadcast_as::<K>(t, &target_shape)
    }

    /// Take a half-open `[start, end)` window per axis, one `narrow` at a
    /// time. Every `narrow` already pushes its own correct tape entry, so
    /// the composite's backward is the tape replay over them, not new
    /// hand-derived math - mirrors the `RmsNorm`/`Softmax` composition.
    pub(crate) fn slice<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mut current: <Self as StorageBackend>::Storage<K> = {
            let t: &CudaStorage = t;
            t.clone()
        };
        // One `narrow` per range, unconditionally - same shape as CPU's own
        // `slice_storage`. A range list longer than the operand's rank is
        // not special-cased here either: it reaches `narrow`'s own
        // `dim >= t.shape.len()` check and fails loudly there, on the axis
        // that is actually out of bounds, rather than being caught early by
        // a bespoke message that could drift from `narrow`'s.
        for (axis, &(start, end)) in ranges.iter().enumerate() {
            let len = end.checked_sub(start).ok_or_else(|| {
                Error::Msg(format!(
                    "slice range on axis {axis} has end {end} before start {start}"
                ))
            })?;
            current = Self::narrow::<K>(&current, axis, start, len)?;
        }
        Ok(current)
    }

    /// Join operands along a new axis: `unsqueeze` each at `dim`, then
    /// `concat`. Zero new tape entries of its own - both steps already push
    /// their own.
    pub(crate) fn stack<K: DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let unsqueezed: Vec<<Self as StorageBackend>::Storage<K>> = tensors
            .iter()
            .map(|t| Self::unsqueeze::<K>(t, dim))
            .collect::<Result<_>>()?;
        let refs: Vec<&<Self as StorageBackend>::Storage<K>> = unsqueezed.iter().collect();
        Self::concat::<K>(&refs, dim)
    }

    /// Cut `axis` into consecutive pieces of `piece` elements each, the last
    /// one shorter if `extent` does not divide evenly. One `narrow` per
    /// piece, each already tape-tracked - mirrors CPU's own
    /// `consecutive_pieces`, which both `chunk` and `split` share.
    fn consecutive_pieces<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
        piece: usize,
    ) -> Result<Vec<<Self as StorageBackend>::Storage<K>>> {
        let t: &CudaStorage = t;
        let Some(&extent) = t.shape.get(axis) else {
            return Err(Error::Msg(
                "the split axis is outside the operand rank".into(),
            ));
        };
        if piece == 0 {
            return Err(Error::Msg(
                "a piece of length zero would never advance".into(),
            ));
        }
        let mut pieces = Vec::with_capacity(extent.div_ceil(piece));
        let mut start = 0;
        while start < extent {
            let length = (extent - start).min(piece);
            pieces.push(Self::narrow::<K>(t, axis, start, length)?);
            start += length;
        }
        Ok(pieces)
    }

    /// Cut `axis` into `chunks` roughly-equal consecutive pieces, rounding
    /// the piece size up so a request for more chunks than the axis can
    /// supply yields fewer pieces than asked rather than empty ones.
    pub(crate) fn chunk<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
        chunks: usize,
    ) -> Result<Vec<<Self as StorageBackend>::Storage<K>>> {
        let extent = {
            let storage: &CudaStorage = t;
            *storage
                .shape
                .get(axis)
                .ok_or_else(|| Error::Msg("the chunk axis is outside the operand rank".into()))?
        };
        if chunks == 0 {
            return Err(Error::Msg(
                "a chunk count of zero divides into nothing".into(),
            ));
        }
        let piece = extent.div_ceil(chunks);
        Self::consecutive_pieces::<K>(t, axis, piece)
    }

    /// Cut `axis` into consecutive pieces of exactly `split_size` elements,
    /// the last one shorter if it does not divide evenly.
    pub(crate) fn split<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
        split_size: usize,
    ) -> Result<Vec<<Self as StorageBackend>::Storage<K>>> {
        Self::consecutive_pieces::<K>(t, axis, split_size)
    }

    /// Metadata-only: every `CudaStorage` this backend produces is always
    /// fully contiguous (`narrow`/`transpose`/`broadcast_as` below
    /// materialize a fresh contiguous buffer rather than building a
    /// strided view - CUDA's elementwise/matmul/reduce kernels assume flat
    /// contiguous memory), so reshaping never needs to touch the data or
    /// check contiguity first, unlike CPU's `reshape`.
    pub(crate) fn reshape<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
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
    pub(crate) fn transpose<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
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

    /// Metadata-only transpose sharing the input buffer. Its backward is the
    /// same permutation applied to the upstream gradient, which is its own
    /// inverse exactly like the materializing transpose above -- except the
    /// recipe materializes rather than viewing again. A strided gradient
    /// would read back flat and break accumulation downstream, so correctness
    /// buys one copy here while the forward keeps its zero-copy win.
    pub(crate) fn transpose_view<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
        let out = crate::cuda::ops::shape::launch_transpose_view(t, dim1, dim2)?;
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                Ok(vec![crate::cuda::ops::shape::launch_transpose(
                    grad_out, dim1, dim2,
                )?])
            }),
        });
        Ok(out)
    }

    /// Matmul is only wired for unbatched 2D operands so far - falls through to the `Backend`
    /// trait's default `Err(UnsupportedBackendOperation)` for anything else.
    pub(crate) fn matmul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let (lhs, rhs): (&CudaStorage, &CudaStorage) = (lhs, rhs);
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

    /// General N-D matrix product with NumPy-style batch broadcasting,
    /// matching CPU's `matmul_storage` dispatcher. CUDA has no batched-GEMM
    /// kernel, so unlike CPU's dedicated `batched_gemm`, the batch case here
    /// is composed: broadcast both operands to the common batch shape,
    /// flatten the batch axes into one, run 2D `matmul` per batch slice via
    /// `narrow`, then `concat`/reshape the pieces back. Every step is
    /// already tape-tracked, so the composite's backward is the tape replay
    /// over them rather than new hand-derived math.
    pub(crate) fn batched_matmul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let (lhs, rhs): (&CudaStorage, &CudaStorage) = (lhs, rhs);
        let (lhs_rank, rhs_rank) = (lhs.shape.len(), rhs.shape.len());
        if lhs_rank < 2 || rhs_rank < 2 {
            return Err(Error::ShapeMismatch {
                op: "batched_matmul",
                expected: vec![2],
                got: vec![lhs_rank, rhs_rank],
                msg: format!(
                    "batched matmul requires both operands to have rank >= 2; got lhs.shape={:?}, rhs.shape={:?}",
                    lhs.shape, rhs.shape
                ),
            });
        }
        let (m, lhs_k) = (lhs.shape[lhs_rank - 2], lhs.shape[lhs_rank - 1]);
        let (rhs_k, n) = (rhs.shape[rhs_rank - 2], rhs.shape[rhs_rank - 1]);
        if lhs_k != rhs_k {
            return Err(Error::ShapeMismatch {
                op: "batched_matmul",
                expected: vec![lhs_k],
                got: vec![rhs_k],
                msg: format!(
                    "matmul inner dims must match: lhs.shape={:?} (K={lhs_k}), rhs.shape={:?} (K={rhs_k})",
                    lhs.shape, rhs.shape
                ),
            });
        }
        // The unbatched case is the overwhelming common one, and going
        // straight to the dedicated 2D kernel avoids paying the composed
        // path's reshape/narrow/concat overhead for it.
        if lhs_rank == 2 && rhs_rank == 2 {
            return Self::matmul::<K>(lhs, rhs);
        }

        let lhs_batch = &lhs.shape[..lhs_rank - 2];
        let rhs_batch = &rhs.shape[..rhs_rank - 2];
        let out_batch = crate::layout::broadcast_shape(lhs_batch, rhs_batch)?;
        let batch_total: usize = out_batch.iter().product();
        let mut out_shape = out_batch.clone();
        out_shape.extend_from_slice(&[m, n]);

        let mut lhs_target = out_batch.clone();
        lhs_target.extend_from_slice(&[m, lhs_k]);
        let mut rhs_target = out_batch.clone();
        rhs_target.extend_from_slice(&[rhs_k, n]);

        let lhs_wide = if lhs.shape == lhs_target {
            lhs.clone()
        } else {
            Self::broadcast_as::<K>(lhs, &lhs_target)?
        };
        let rhs_wide = if rhs.shape == rhs_target {
            rhs.clone()
        } else {
            Self::broadcast_as::<K>(rhs, &rhs_target)?
        };

        let lhs_flat = Self::reshape::<K>(&lhs_wide, &[batch_total, m, lhs_k])?;
        let rhs_flat = Self::reshape::<K>(&rhs_wide, &[batch_total, rhs_k, n])?;

        if batch_total == 0 {
            // No batch slice to run `matmul` over; a zero-element reshape of
            // either flattened operand is exact, not a placeholder - see
            // `IterationPlan::binary`'s own zero-vs-unbatched distinction,
            // which `crate::layout::broadcast_shape` already carries here.
            return Self::reshape::<K>(&lhs_flat, &out_shape);
        }

        let mut slices: Vec<<Self as StorageBackend>::Storage<K>> = Vec::with_capacity(batch_total);
        for index in 0..batch_total {
            let lhs_slice = Self::narrow::<K>(&lhs_flat, 0, index, 1)?;
            let rhs_slice = Self::narrow::<K>(&rhs_flat, 0, index, 1)?;
            let lhs_2d = Self::reshape::<K>(&lhs_slice, &[m, lhs_k])?;
            let rhs_2d = Self::reshape::<K>(&rhs_slice, &[rhs_k, n])?;
            let product = Self::matmul::<K>(&lhs_2d, &rhs_2d)?;
            slices.push(Self::reshape::<K>(&product, &[1, m, n])?);
        }
        let refs: Vec<&<Self as StorageBackend>::Storage<K>> = slices.iter().collect();
        let stacked = Self::concat::<K>(&refs, 0)?;
        Self::reshape::<K>(&stacked, &out_shape)
    }

    /// Materializes (see `reshape`'s doc for why).
    pub(crate) fn broadcast_as<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
        // Validates compatibility before dispatch - an invalid broadcast
        // must error, not silently read garbage/OOB indices in the kernel.
        crate::layout::broadcast_shape(&t.shape, shape)?;

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
    pub(crate) fn narrow<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
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

    /// Composed from `reshape` (zero new tape entries - matches CPU/WGPU).
    pub(crate) fn squeeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t: &CudaStorage = t;
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

    pub(crate) fn gather<K: DType, KInt: DType>(
        t: &CudaStorage,
        dim: usize,
        index: &CudaStorage,
    ) -> Result<CudaStorage> {
        let out = crate::cuda::ops::shape::launch_gather(t, dim, index)?;
        let t_capture = t.clone();
        let index_capture = index.clone();
        let (t_id, out_id) = (t.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let zero_base = CudaStorage::new(
                    Arc::new(CudaBuffer {
                        len: t_capture.buffer.len,
                        dtype: t_capture.buffer.dtype,
                        data: Arc::new(
                            t_capture
                                .buffer
                                .device
                                .default_stream()
                                .alloc_zeros::<u8>(crate::bytes::byte_len(
                                    DTypeId::F32,
                                    t_capture.buffer.len,
                                    OperationKind::Storage,
                                )?)
                                .map_err(|e| Error::Msg(format!("{e:?}")))?,
                        ),
                        device: t_capture.buffer.device.clone(),
                        device_id: t_capture.buffer.device_id,
                    }),
                    t_capture.shape.to_vec(),
                );
                // Scatter-add, not scatter-overwrite: every index position
                // contributes its cotangent, accumulating on duplicates to
                // match CPU's `gather_storage` backward (`+=`). The plain
                // scatter kept only one contribution there.
                let grad_input = crate::cuda::ops::shape::launch_scatter_add(
                    &zero_base,
                    dim,
                    &index_capture,
                    grad_out,
                )?;
                Ok(vec![grad_input])
            }),
        });
        Ok(out)
    }

    pub(crate) fn scatter<K: DType, KInt: DType>(
        t: &CudaStorage,
        dim: usize,
        index: &CudaStorage,
        src: &CudaStorage,
    ) -> Result<CudaStorage> {
        let out = crate::cuda::ops::shape::launch_scatter(t, dim, index, src)?;
        // Backward mirrors CPU's `scatter_storage`: the input keeps its
        // cotangent everywhere EXCEPT the positions a write overwrote (those
        // land on the source instead), and the source receives the output
        // cotangent only through the LAST write to each destination, because
        // the forward's last-write-wins rule means earlier writes to the same
        // position contributed nothing. The integer index operand is off the
        // tape by construction, same as CPU.
        let (index_capture, src_capture) = (index.clone(), src.clone());
        let (t_id, src_id, out_id) = (t.id, src.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id, src_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                // Scattering zeros over the same index zeroes exactly the
                // overwritten positions however many writes collided there,
                // which is the whole of the input path; no bookkeeping needed.
                let zeros = Self::zeros::<K>(
                    &src_capture.shape,
                    DTypeId::F32.descriptor(),
                    &DeviceId::cuda(src_capture.buffer.device_id),
                )?;
                let grad_t =
                    crate::cuda::ops::shape::launch_scatter(grad_out, dim, &index_capture, &zeros)?;
                // Source path gathers the output cotangent, then keeps only
                // the last write per destination (row-major, like CPU).
                // Unique indices are exact; duplicates previously returned
                // every writer's copy (`[1,1]` for a `[0,1]` reference).
                // Note the forward itself still races with duplicates -- the
                // kernel stores with plain writes, so which value wins is
                // nondeterministic on GPU while CPU keeps the last row-major
                // write. A deterministic forward needs a bigger kernel (see
                // follow-ups); this makes the backward match CPU semantics.
                let grad_src = crate::cuda::ops::shape::launch_scatter_src_grad(
                    grad_out,
                    dim,
                    &index_capture,
                )?;
                Ok(vec![grad_t, grad_src])
            }),
        });
        Ok(out)
    }

    pub(crate) fn diag<K: DType>(t: &CudaStorage, diagonal: i64) -> Result<CudaStorage> {
        crate::cuda::ops::shape::launch_diag(t, diagonal as i32)
    }

    pub(crate) fn pad<K: DType>(
        t: &CudaStorage,
        padding: &[(usize, usize)],
        value: f64,
    ) -> Result<CudaStorage> {
        let out = crate::cuda::ops::shape::launch_pad(t, padding, value)?;
        let t_shape = t.shape.clone();
        let padding_capture = padding.to_vec();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            let mut curr = grad_out.clone();
            for (axis, &(before, _after)) in padding_capture.iter().enumerate() {
                let len = t_shape[axis];
                curr = crate::cuda::ops::shape::launch_narrow(&curr, axis, before, len)?;
            }
            Ok(curr)
        });
        Ok(out)
    }

    pub(crate) fn repeat<K: DType>(t: &CudaStorage, repeats: &[usize]) -> Result<CudaStorage> {
        let out = crate::cuda::ops::shape::launch_repeat(t, repeats)?;
        let t_shape = t.shape.clone();
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::tape::unbroadcast(grad_out, &t_shape)
        });
        Ok(out)
    }

    pub(crate) fn tril<K: DType>(t: &CudaStorage, diagonal: i64) -> Result<CudaStorage> {
        let out = crate::cuda::ops::shape::launch_tril(t, diagonal as i32)?;
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::ops::shape::launch_tril(grad_out, diagonal as i32)
        });
        Ok(out)
    }

    pub(crate) fn triu<K: DType>(t: &CudaStorage, diagonal: i64) -> Result<CudaStorage> {
        let out = crate::cuda::ops::shape::launch_triu(t, diagonal as i32)?;
        push_unary_tape_entry(t.id, out.id, move |grad_out| {
            crate::cuda::ops::shape::launch_triu(grad_out, diagonal as i32)
        });
        Ok(out)
    }

    pub(crate) fn embedding<K: DType, KInt: DType>(
        weight: &CudaStorage,
        indices: &CudaStorage,
    ) -> Result<CudaStorage> {
        let out = crate::cuda::ops::shape::launch_embedding(weight, indices)?;
        let indices_capture = indices.clone();
        let (vocab_size, hidden_size) = (weight.shape[0], weight.shape[1]);
        let (weight_id, out_id) = (weight.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![weight_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let grad_weight = crate::cuda::ops::shape::launch_embedding_backward(
                    grad_out,
                    &indices_capture,
                    vocab_size,
                    hidden_size,
                )?;
                Ok(vec![grad_weight])
            }),
        });
        Ok(out)
    }
}

pub(crate) fn cuda_reshape_storage(t: &CudaStorage, shape: &[usize]) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::reshape::<f32>(t, shape)
}

pub(crate) fn cuda_transpose_storage(t: &CudaStorage, d1: usize, d2: usize) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::transpose::<f32>(t, d1, d2)
}

pub(crate) fn cuda_matmul_storage(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::matmul::<f32>(lhs, rhs)
}

pub(crate) fn cuda_broadcast_as_storage(t: &CudaStorage, shape: &[usize]) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::broadcast_as::<f32>(t, shape)
}

pub(crate) fn cuda_narrow_storage(
    t: &CudaStorage,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::narrow::<f32>(t, dim, start, len)
}

pub(crate) fn cuda_unsqueeze_storage(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::unsqueeze::<f32>(t, dim)
}

pub(crate) fn cuda_squeeze_storage(t: &CudaStorage, dim: usize) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::squeeze::<f32>(t, dim)
}

pub(crate) fn cuda_concat_storage(t: &[&CudaStorage], dim: usize) -> Result<CudaStorage> {
    CudaBackendImpl::<Cuda>::concat::<f32>(t, dim)
}
