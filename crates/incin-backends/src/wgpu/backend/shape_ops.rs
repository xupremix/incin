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

    /// `narrow(axis, start, length)`: extract `length` entries from `axis`
    /// beginning at `start`. Composed as `shape.wgsl` mode 0 (slice), with the
    /// per-axis start in `aux` and zeros everywhere else.
    ///
    /// Backward is a zeroed buffer plus one `shape.wgsl` mode 1 (paste) that
    /// scatters the cotangent back to the window — the exact inverse of the
    /// forward slice, matching CPU's `narrow_storage`.
    pub(crate) fn narrow<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
        start: usize,
        length: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let rank = t.shape.len();
        if axis >= rank {
            return Err(Error::ShapeMismatch {
                op: "narrow",
                expected: t.shape.to_vec(),
                got: alloc::vec![axis],
                msg: "narrow axis out of bounds".to_string(),
            });
        }
        let dim = t.shape[axis];
        if start.saturating_add(length) > dim || length == 0 {
            return Err(Error::ShapeMismatch {
                op: "narrow",
                expected: alloc::vec![dim],
                got: alloc::vec![start, length],
                msg: "narrow window is empty or out of bounds".to_string(),
            });
        }
        let mut out_shape = t.shape.to_vec();
        out_shape[axis] = length;
        let out_n = num_elements(&out_shape)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;
        let mut aux = alloc::vec![0usize; rank];
        aux[axis] = start;
        let params = dispatch::prepare_shape_params(
            0,
            checked_u32(out_n, "narrow out")?,
            &out_shape,
            &t.shape,
            &aux,
        )?;
        dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
        let out = WgpuStorage::new(out_buf, out_shape);

        let (t_id, out_id) = (t.id, out.id);
        let in_shape = t.shape.to_vec();
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let n_in = num_elements(&in_shape)?;
                let g_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, n_in, OperationKind::Storage)?;
                let mut g_aux = alloc::vec![0usize; in_shape.len()];
                g_aux[axis] = start;
                let g_params = dispatch::prepare_shape_params(
                    1,
                    checked_u32(n_in, "narrow grad")?,
                    &in_shape,
                    &grad_out.shape,
                    &g_aux,
                )?;
                dispatch::dispatch_shape(&grad_out.buffer, &g_buf, &g_params);
                Ok(vec![WgpuStorage::new(g_buf, in_shape.clone())])
            }),
        });
        Ok(out)
    }

    /// `slice(ranges)`: the general form of [`narrow`](Self::narrow) over a
    /// per-axis window. One `shape.wgsl` mode-0 launch; backward is one
    /// mode-1 paste into a zeroed buffer, the inverse of the forward.
    pub(crate) fn slice_exact<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let rank = t.shape.len();
        if ranges.len() != rank {
            return Err(Error::ShapeMismatch {
                op: "slice_exact",
                expected: t.shape.to_vec(),
                got: ranges.iter().map(|r| r.1 - r.0).collect(),
                msg: "slice ranks must match".to_string(),
            });
        }
        let mut out_shape = alloc::vec::Vec::with_capacity(rank);
        let mut starts = alloc::vec![0usize; rank];
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start > end || end > t.shape[i] || start == end {
                return Err(Error::ShapeMismatch {
                    op: "slice_exact",
                    expected: alloc::vec![t.shape[i]],
                    got: alloc::vec![start, end],
                    msg: "slice range invalid or empty".to_string(),
                });
            }
            starts[i] = start;
            out_shape.push(end - start);
        }
        let out_n = num_elements(&out_shape)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;
        let params = dispatch::prepare_shape_params(
            0,
            checked_u32(out_n, "slice out")?,
            &out_shape,
            &t.shape,
            &starts,
        )?;
        dispatch::dispatch_shape(&t.buffer, &out_buf, &params);
        let out = WgpuStorage::new(out_buf, out_shape);

        let (t_id, out_id) = (t.id, out.id);
        let in_shape = t.shape.to_vec();
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let n_in = num_elements(&in_shape)?;
                let g_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, n_in, OperationKind::Storage)?;
                let g_params = dispatch::prepare_shape_params(
                    1,
                    checked_u32(n_in, "slice grad")?,
                    &in_shape,
                    &grad_out.shape,
                    &starts,
                )?;
                dispatch::dispatch_shape(&grad_out.buffer, &g_buf, &g_params);
                Ok(vec![WgpuStorage::new(g_buf, in_shape.clone())])
            }),
        });
        Ok(out)
    }

    /// `concat(inputs, axis)`: one zero-filled output plus one `shape.wgsl`
    /// mode-1 (paste) launch per input, each pasting its operand into the
    /// output at the running offset along `axis`. Matches CPU's
    /// `concat_storage` exactly (no `broadcast_as` promotion — operands must
    /// agree on every other axis).
    pub(crate) fn concat_exact<K: DType>(
        inputs: &[&<Self as StorageBackend>::Storage<K>],
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if inputs.is_empty() {
            return Err(Error::ShapeMismatch {
                op: "concat_exact",
                expected: alloc::vec![],
                got: alloc::vec![],
                msg: "concat expects at least one operand".to_string(),
            });
        }
        let rank = inputs[0].shape.len();
        if axis >= rank {
            return Err(Error::ShapeMismatch {
                op: "concat_exact",
                expected: inputs[0].shape.to_vec(),
                got: alloc::vec![axis],
                msg: "concat axis out of bounds".to_string(),
            });
        }
        let mut out_shape = inputs[0].shape.to_vec();
        out_shape[axis] = 0;
        for t in inputs {
            if t.shape.len() != rank {
                return Err(Error::ShapeMismatch {
                    op: "concat_exact",
                    expected: inputs[0].shape.to_vec(),
                    got: t.shape.to_vec(),
                    msg: "concat operands must share rank".to_string(),
                });
            }
            for d in 0..rank {
                if d != axis && t.shape[d] != inputs[0].shape[d] {
                    return Err(Error::ShapeMismatch {
                        op: "concat_exact",
                        expected: inputs[0].shape.to_vec(),
                        got: t.shape.to_vec(),
                        msg: "concat operands must agree off the concat axis".to_string(),
                    });
                }
            }
            out_shape[axis] += t.shape[axis];
        }
        let out_n = num_elements(&out_shape)?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;
        let out = WgpuStorage::new(out_buf, out_shape.clone());

        let mut offset = 0usize;
        for t in inputs {
            let mut start = alloc::vec![0usize; rank];
            start[axis] = offset;
            let n_in = num_elements(&t.shape)?;
            let params = dispatch::prepare_shape_params(
                1,
                checked_u32(n_in, "concat paste")?,
                &out_shape,
                &t.shape,
                &start,
            )?;
            dispatch::dispatch_shape(&t.buffer, &out.buffer, &params);
            offset += t.shape[axis];
        }

        let out_id = out.id;
        let input_ids = inputs.iter().map(|t| t.id).collect::<alloc::vec::Vec<_>>();
        let captured: alloc::vec::Vec<_> = inputs.iter().map(|t| (*t).clone()).collect();
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids,
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut grads = alloc::vec::Vec::with_capacity(captured.len());
                let mut offset = 0usize;
                for t in &captured {
                    grads.push(Self::narrow::<K>(grad_out, axis, offset, t.shape[axis])?);
                    offset += t.shape[axis];
                }
                Ok(grads)
            }),
        });
        Ok(out)
    }

    /// `stack(inputs, axis)`: insert a new axis at `axis`, each operand
    /// unsqueezed to that rank, then [`concat_exact`](Self::concat_exact).
    /// Mirrors CPU's `stack_storage` = unsqueeze + concat.
    pub(crate) fn stack_exact<K: DType>(
        inputs: &[&<Self as StorageBackend>::Storage<K>],
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if inputs.is_empty() {
            return Err(Error::ShapeMismatch {
                op: "stack_exact",
                expected: alloc::vec![],
                got: alloc::vec![],
                msg: "stack expects at least one operand".to_string(),
            });
        }
        if axis > inputs[0].shape.len() {
            return Err(Error::ShapeMismatch {
                op: "stack_exact",
                expected: inputs[0].shape.to_vec(),
                got: alloc::vec![axis],
                msg: "stack axis out of bounds".to_string(),
            });
        }
        let mut projected: alloc::vec::Vec<<Self as StorageBackend>::Storage<K>> =
            alloc::vec::Vec::with_capacity(inputs.len());
        for t in inputs {
            projected.push(Self::unsqueeze::<K>(t, axis)?);
        }
        let refs: alloc::vec::Vec<&_> = projected.iter().collect();
        Self::concat_exact::<K>(&refs, axis)
    }

    /// Keep the same half CPU's `triangular_storage` keeps: upper is
    /// `col >= row + offset` (`diag >= offset`), lower is `col <= row + offset`.
    /// Rank one is treated as the first row of an implicit matrix (`row = 0`,
    /// `col = index`), exactly as CPU does when `rank < 2`.
    fn triangular_keep(row: i64, col: i64, offset: i64, upper: bool) -> bool {
        let diag = col - row;
        if upper {
            diag >= offset
        } else {
            diag <= offset
        }
    }

    /// `tril`/`triu`: zero every entry on the wrong side of the `offset`-th
    /// diagonal of a rank 1 or 2 operand. Composed host-side over one download
    /// and one upload, matching CPU's `triangular_storage` loop index-for-index;
    /// no capability claim is involved in the mask itself. Backward reapplies
    /// the same keep rule to the cotangent, which is the identity CPU proves
    /// ("zeroing is its own transpose").
    fn triangular<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        offset: i64,
        upper: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let rank = t.shape.len();
        if rank == 0 || rank > 2 {
            return Err(Error::ShapeMismatch {
                op: if upper { "triu" } else { "tril" },
                expected: alloc::vec![1, 2],
                got: alloc::vec![rank],
                msg: "tril/triu accept rank 1 or 2".to_string(),
            });
        }
        let data = t.buffer.to_vec::<f32>()?;
        let mut masked = alloc::vec![0.0f32; data.len()];
        if rank == 1 {
            for (col, &value) in data.iter().enumerate() {
                if Self::triangular_keep(0, col as i64, offset, upper) {
                    masked[col] = value;
                }
            }
        } else {
            let cols = t.shape[1];
            for (idx, &value) in data.iter().enumerate() {
                let row = (idx / cols) as i64;
                let col = (idx % cols) as i64;
                if Self::triangular_keep(row, col, offset, upper) {
                    masked[idx] = value;
                }
            }
        }
        let out = WgpuStorage::new(WgpuBuffer::from_slice(&masked), t.shape.to_vec());

        let (t_id, out_id) = (t.id, out.id);
        let shape = t.shape.to_vec();
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: alloc::vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let mut g = grad_out.buffer.to_vec::<f32>()?;
                if shape.len() == 1 {
                    for (col, item) in g.iter_mut().enumerate() {
                        if !Self::triangular_keep(0, col as i64, offset, upper) {
                            *item = 0.0;
                        }
                    }
                } else {
                    let cols = shape[1];
                    for (idx, item) in g.iter_mut().enumerate() {
                        let row = (idx / cols) as i64;
                        let col = (idx % cols) as i64;
                        if !Self::triangular_keep(row, col, offset, upper) {
                            *item = 0.0;
                        }
                    }
                }
                Ok(alloc::vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&g),
                    grad_out.shape.to_vec()
                )])
            }),
        });
        Ok(out)
    }

    pub(crate) fn tril<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        offset: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::triangular::<K>(t, offset, false)
    }

    pub(crate) fn triu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        offset: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::triangular::<K>(t, offset, true)
    }

    /// `dot`: elementwise product then an all-sum, exactly CPU's composition.
    pub(crate) fn dot<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let product = Self::mul::<K>(lhs, rhs)?;
        Self::sum_all::<K>(&product)
    }

    /// `bmm`: batched (or plain 2-D) matmul under its own name, so the row
    /// inherits matmul's constraint and its tape.
    pub(crate) fn batched_matmul<K: DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::matmul::<K>(lhs, rhs)
    }

    /// `addmm(mat, lhs, rhs)`: `beta * mat + alpha * (lhs @ rhs)`, matching
    /// CPU's `addmm_storage` step for step (product, scale by alpha, scale mat
    /// by beta, add).
    pub(crate) fn addmm<K: DType>(
        mat: &<Self as StorageBackend>::Storage<K>,
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
        alpha: f64,
        beta: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let product = Self::matmul::<K>(lhs, rhs)?;
        let scaled_product = Self::mul_scalar_float::<K>(&product, alpha)?;
        let scaled_mat = Self::mul_scalar_float::<K>(mat, beta)?;
        Self::add::<K>(&scaled_mat, &scaled_product)
    }

    /// `linear(input, weight, bias?)`: promote a rank-one input to a single
    /// row, `input @ weight^T`, optionally add the bias (WGPU's binary path
    /// broadcasts), then drop the promoted row again — CPU's recipe, with every
    /// step a taped primitive.
    pub(crate) fn linear<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let unbatched = input.shape.len() == 1;
        let promoted;
        let rows = if unbatched {
            promoted = Self::reshape::<K>(input, &[1, input.shape[0]])?;
            &promoted
        } else {
            input
        };
        let transposed = Self::transpose::<K>(weight, 0, 1)?;
        let product = Self::matmul::<K>(rows, &transposed)?;
        let projected = match bias {
            None => product,
            Some(bias) => Self::add::<K>(&product, bias)?,
        };
        if unbatched {
            Self::reshape::<K>(&projected, &projected.shape[1..])
        } else {
            Ok(projected)
        }
    }
}
