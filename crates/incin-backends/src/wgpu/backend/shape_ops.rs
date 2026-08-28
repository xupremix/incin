//! Structural and shape-changing WGPU operations: matmul, reshape,
//! transpose, and broadcast.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
//   (reshape, transpose, matmul, narrow, flatten, squeeze, stack, concat, etc.)
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> WgpuBackendImpl<D> {
    /// `matmul`.
    pub(crate) fn matmul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if lhs.shape.len() < 2 || rhs.shape.len() < 2 {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: vec![2],
                got: vec![lhs.shape.len(), rhs.shape.len()],
                msg: "matmul requires at least 2D inputs".to_string(),
            });
        }

        let lhs_rank = lhs.shape.len();
        let rhs_rank = rhs.shape.len();

        let m = lhs.shape[lhs_rank - 2];
        let k = lhs.shape[lhs_rank - 1];
        let n = rhs.shape[rhs_rank - 1];

        if k != rhs.shape[rhs_rank - 2] {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs.shape.to_vec(),
                got: rhs.shape.to_vec(),
                msg: "matmul inner dims must match".to_string(),
            });
        }

        // Compute batch dims
        let lhs_batch = ShapeBuf::from_slice(&lhs.shape[..lhs_rank - 2])
            .checked_numel(OperationKind::MatMul)?;
        let rhs_batch = ShapeBuf::from_slice(&rhs.shape[..rhs_rank - 2])
            .checked_numel(OperationKind::MatMul)?;

        let batch = core::cmp::max(lhs_batch, rhs_batch);
        if lhs_batch != 1 && rhs_batch != 1 && lhs_batch != rhs_batch {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs.shape.to_vec(),
                got: rhs.shape.to_vec(),
                msg: "matmul batch dims incompatible".to_string(),
            });
        }

        let lhs_stride_b = if lhs_batch == 1 {
            0
        } else {
            m.checked_mul(k).ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::MatMul,
                expression: "WGPU matmul lhs batch stride",
            })?
        };
        let rhs_stride_b = if rhs_batch == 1 {
            0
        } else {
            k.checked_mul(n).ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::MatMul,
                expression: "WGPU matmul rhs batch stride",
            })?
        };

        // Output shape matches the larger batched input
        let mut out_shape = if lhs_batch > 1 {
            lhs.shape[..lhs_rank - 2].to_vec()
        } else {
            rhs.shape[..rhs_rank - 2].to_vec()
        };
        if out_shape.is_empty() && batch > 1 {
            out_shape.push(batch);
        }
        out_shape.push(m);
        out_shape.push(n);

        let state = crate::wgpu::device::get_device_state();
        let shader = include_str!("../shaders/matmul.wgsl");
        let pipeline = crate::wgpu::pipeline::get_or_create_pipeline("matmul", shader, "main");

        let out_n = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::MatMul)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::MatMul)?;
        let [
            m_u32,
            k_u32,
            n_u32,
            batch_u32,
            lhs_stride_u32,
            rhs_stride_u32,
        ] = checked_u32_array(
            [m, k, n, batch, lhs_stride_b, rhs_stride_b],
            "WGPU matmul kernel parameter",
        )?;
        let shape_data = [
            m_u32,
            k_u32,
            n_u32,
            batch_u32,
            lhs_stride_u32,
            rhs_stride_u32,
        ];
        let shape_buf = WgpuBuffer::from_slice(&shape_data);

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = state
            .device
            .create_bind_group(&::wgpu::BindGroupDescriptor {
                label: Some("Matmul BG"),
                layout: &bgl,
                entries: &[
                    ::wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lhs.buffer.buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.buffer.buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.buffer.as_entire_binding(),
                    },
                    ::wgpu::BindGroupEntry {
                        binding: 3,
                        resource: shape_buf.buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = state
            .device
            .create_command_encoder(&::wgpu::CommandEncoderDescriptor {
                label: Some("Matmul"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&::wgpu::ComputePassDescriptor {
                label: Some("Matmul"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(n_u32.div_ceil(16), m_u32.div_ceil(16), batch_u32);
        }
        state.queue.submit(core::iter::once(encoder.finish()));
        let out = WgpuStorage::new(out_buf, out_shape);

        // Backward: grad_lhs = grad_out @ rhs^T, grad_rhs = lhs^T @ grad_out,
        // composed from Self::matmul + Self::transpose recursion (mirrors the
        // CPU backend's batched_matmul_impl exactly) rather than a bespoke
        // kernel. Self::matmul already broadcasts a batch=1 operand against
        // the other's batch shape internally (lhs_stride_b/rhs_stride_b=0
        // above), so `grad_out @ rhs^T`/`lhs^T @ grad_out` naturally come out
        // at the OUTPUT batch shape; `unbroadcast` then reduces back down to
        // each operand's own original (possibly batch=1) shape.
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &WgpuStorage| {
                let rhs_rank = rhs_capture.shape.len();
                let rhs_t = Self::transpose::<K>(&rhs_capture, rhs_rank - 2, rhs_rank - 1)?;
                let grad_lhs_full = Self::matmul::<K>(grad_out, &rhs_t)?;

                let lhs_rank = lhs_capture.shape.len();
                let lhs_t = Self::transpose::<K>(&lhs_capture, lhs_rank - 2, lhs_rank - 1)?;
                let grad_rhs_full = Self::matmul::<K>(&lhs_t, grad_out)?;

                Ok(vec![
                    crate::wgpu::tape::unbroadcast(&grad_lhs_full, &lhs_shape)?,
                    crate::wgpu::tape::unbroadcast(&grad_rhs_full, &rhs_shape)?,
                ])
            }),
        });
        Ok(out)
    }

    /// `reshape`.
    pub(crate) fn reshape<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if num_elements(&t.shape)? != num_elements(shape)? {
            return Err(Error::ShapeMismatch {
                op: "reshape",
                expected: t.shape.to_vec(),
                got: shape.to_vec(),
                msg: "total elements must match".to_string(),
            });
        }
        let out = WgpuStorage::new(t.buffer.clone(), shape.to_vec());
        let original_shape = t.shape.to_vec();
        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::reshape::<K>(grad_out, &original_shape)?])
            }),
        });
        Ok(out)
    }

    /// Drop an axis of extent 1.
    ///
    /// A view, not a move: the elements are already in the right order, so
    /// this is `reshape` with the axis removed from the shape. Composed the
    /// same way CUDA composes it, and it inherits `reshape`'s tape entry
    /// rather than pushing one of its own.
    ///
    /// Refusing a non-unit axis is the point of the check: silently keeping
    /// an axis the caller asked to remove would hand back a tensor of a
    /// different rank than the one they wrote down.
    pub(crate) fn squeeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if dim >= t.shape.len() || t.shape[dim] != 1 {
            return Err(Error::ShapeMismatch {
                op: "squeeze",
                expected: alloc::vec![1],
                got: t.shape.to_vec(),
                msg: alloc::format!(
                    "squeeze requires axis {dim} to have size 1, got size {} in shape {:?}",
                    t.shape.get(dim).copied().unwrap_or(0),
                    t.shape
                ),
            });
        }
        let mut target = t.shape.to_vec();
        target.remove(dim);
        Self::reshape::<K>(t, &target)
    }

    /// Insert an axis of extent 1 at `dim`.
    ///
    /// The inverse of [`squeeze`](Self::squeeze), and a view for the same
    /// reason. `dim == rank` appends rather than failing, matching CUDA.
    pub(crate) fn unsqueeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mut target = t.shape.to_vec();
        if dim <= target.len() {
            target.insert(dim, 1);
        } else {
            target.push(1);
        }
        Self::reshape::<K>(t, &target)
    }

    /// Collapse the inclusive axis range `[start_dim, end_dim]` into one axis.
    ///
    /// Composed from `reshape`, like the two above. The bounds check is what
    /// keeps a reversed or out-of-range range from producing a plausible
    /// wrong shape instead of an error.
    pub(crate) fn flatten<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if start_dim > end_dim || end_dim >= t.shape.len() {
            return Err(Error::ShapeMismatch {
                op: "flatten",
                expected: t.shape.to_vec(),
                got: alloc::vec![start_dim, end_dim],
                msg: alloc::format!(
                    "flatten(start_dim={start_dim}, end_dim={end_dim}) out of bounds for shape {:?}",
                    t.shape
                ),
            });
        }
        let collapsed: usize = t.shape[start_dim..=end_dim].iter().product();
        let mut target = t.shape[..start_dim].to_vec();
        target.push(collapsed);
        target.extend_from_slice(&t.shape[end_dim + 1..]);
        Self::reshape::<K>(t, &target)
    }

    /// `transpose`.
    pub(crate) fn transpose<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let shape = &t.shape;
        let mut new_shape = shape.to_vec();
        new_shape.swap(dim1, dim2);

        let out_n = checked_u32(
            num_elements(&new_shape)?,
            "WGPU transpose output element count",
        )?;
        let out_buf = WgpuBuffer::new_zeros(t.buffer.size);

        let mut aux = (0..shape.len()).collect::<Vec<_>>();
        aux.swap(dim1, dim2);

        let params = dispatch::prepare_shape_params(
            2, // op_mode = transpose
            out_n, &new_shape, shape, &aux,
        )?;

        dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
        let out = WgpuStorage::new(out_buf, new_shape);

        let (t_id, out_id) = (t.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                Ok(vec![Self::transpose::<K>(grad_out, dim1, dim2)?])
            }),
        });
        Ok(out)
    }

    pub(crate) fn broadcast_as<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        broadcast_storage(t, shape)
    }
}
