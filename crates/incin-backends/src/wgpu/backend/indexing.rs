//! WGPU indexing ops: `embedding`/`gather`/`index_select` host-walks and the
//! `masked_fill`/`where_cond` GPU selection kernel.
//!
//! The three gather-family operations download their operands, compute on the
//! host exactly as `cpu::ops::{embedding, shape_ops::select}` do, and upload
//! the `f32` result — the same host-walk pattern `triangular`/`pad`/`repeat`
//! already use. Their tape recipes are copied from CPU walk-for-walk: ONE
//! `TapeEntry` per forward, `input_ids = vec![data.id]` only (the integer
//! index operand is off the tape), and a scatter-add backward that
//! accumulates repeated selections rather than overwriting them.
//!
//! `masked_fill`/`where_cond` take the GPU path: a new `select.wgsl` with two
//! modes, fed by a pre-broadcast mask (raw, no tape — a `bool` mask has
//! nowhere to send a gradient) and pre-broadcast values (tape-recording, for
//! the same reason CUDA's `Execute<WhereCond>` records them).

use super::*;
use crate::descriptor_bind::{invalid, kernel_error};

// ─────────────────────────────────────────────────────────────────────────────
// Host-walk helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Download an integer index operand as `i64` values, whatever physical
/// integer width it was uploaded with.
///
/// Refuses any non-integer dtype by name rather than reinterpreting bits:
/// a float or `bool` index has no honest conversion to a row address.
fn index_values(t: &WgpuStorage) -> Result<Vec<i64>> {
    match t.dtype.builtin_id() {
        Some(DTypeId::I64) => t.buffer.to_vec::<i64>(),
        Some(DTypeId::U32) => Ok(t
            .buffer
            .to_vec::<u32>()?
            .into_iter()
            .map(i64::from)
            .collect()),
        Some(DTypeId::U8) => Ok(t
            .buffer
            .to_vec::<u8>()?
            .into_iter()
            .map(i64::from)
            .collect()),
        _ => Err(Error::UnsupportedDType {
            dtype: t.dtype,
            backend: "Wgpu",
            op: "index_values",
        }),
    }
}

/// Refuse any dtype that is not `f32` for a value (non-index) operand.
pub(crate) fn require_f32(t: &WgpuStorage, op: &'static str) -> Result<()> {
    if t.dtype == DTypeId::F32.descriptor() {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            dtype: t.dtype,
            backend: "Wgpu",
            op,
        })
    }
}

/// Convert a downloaded index value to a row/position address, refusing
/// negatives and out-of-range values by name (CPU's `embedding_impl` and
/// `gather_storage` both fail closed here).
fn checked_address(
    raw: i64,
    bound: usize,
    operation: OperationKind,
    parameter: &'static str,
) -> Result<usize> {
    let address = usize::try_from(raw).map_err(|_| Error::InvalidConversion {
        operation: parameter,
        from: DTypeId::I64.descriptor(),
        to: DTypeId::U32.descriptor(),
        reason: incin_core::error::ConversionFailure::OutOfRange,
    })?;
    if address >= bound {
        return Err(incin_core::shapes::ShapeError::InvalidParameter {
            operation,
            parameter: "index",
            value: address,
        }
        .into());
    }
    Ok(address)
}

/// Row-major contiguous strides for a host walk (same as `shape_ops`).
fn host_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Odometer increment over `shape` (wraps like CPU's `increment_index`).
fn host_increment(idx: &mut [usize], shape: &[usize]) {
    for axis in (0..shape.len()).rev() {
        idx[axis] += 1;
        if idx[axis] < shape[axis] {
            return;
        }
        idx[axis] = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// embedding
// ─────────────────────────────────────────────────────────────────────────────

impl<D: Device> WgpuBackendImpl<D> {
    /// Gather rows of `w` (shape `[vocab_size, hidden_size]`) addressed by
    /// the integer indices in `t` (any rank), producing
    /// `t.shape ++ [hidden_size]`.
    ///
    /// Forward and backward are `cpu::ops::embedding::embedding_impl` walk
    /// for walk: ONE `TapeEntry` with `input_ids = vec![w.id]` only, and a
    /// scatter-add that accumulates repeated indices.
    pub(crate) fn embedding(indices: &WgpuStorage, weight: &WgpuStorage) -> Result<WgpuStorage> {
        require_f32(weight, "embedding_weight")?;
        if weight.shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                op: "embedding",
                expected: vec![0, 0],
                got: weight.shape.to_vec(),
                msg: alloc::format!(
                    "embedding: weight table must be rank-2 [vocab_size, hidden_size], got shape {:?}",
                    weight.shape
                ),
            });
        }
        let vocab_size = weight.shape[0];
        let hidden_size = weight.shape[1];

        let rows = weight.buffer.to_vec::<f32>()?;
        let raw_ids = index_values(indices)?;
        let total_indices = raw_ids.len();

        let mut row_indices: Vec<usize> = Vec::with_capacity(total_indices);
        let mut out_vals: Vec<f32> = Vec::with_capacity(total_indices * hidden_size);
        for &raw in &raw_ids {
            let row =
                checked_address(raw, vocab_size, OperationKind::Embedding, "embedding_index")?;
            out_vals.extend_from_slice(&rows[row * hidden_size..(row + 1) * hidden_size]);
            row_indices.push(row);
        }

        let mut out_shape = indices.shape.to_vec();
        out_shape.push(hidden_size);
        let out = WgpuStorage::new(WgpuBuffer::from_slice(&out_vals), out_shape);

        let w_total = weight.shape.iter().product::<usize>();
        let t_shape = indices.shape.to_vec();
        let w_shape = weight.shape.to_vec();
        let (w_id, out_id) = (weight.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![w_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let grad_strides = host_strides(&grad_out.shape);
                let mut grad_w = vec![0.0f32; w_total];
                let mut leading_idx = vec![0usize; t_shape.len()];
                for &row in &row_indices {
                    for h in 0..hidden_size {
                        let mut full_idx = leading_idx.clone();
                        full_idx.push(h);
                        let flat: usize = full_idx
                            .iter()
                            .zip(grad_strides.iter())
                            .map(|(&i, &s)| i * s)
                            .sum();
                        grad_w[row * hidden_size + h] += grad_data[flat];
                    }
                    if !t_shape.is_empty() {
                        host_increment(&mut leading_idx, &t_shape);
                    }
                }
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad_w),
                    w_shape.clone(),
                )])
            }),
        });
        Ok(out)
    }

    /// `gather(input, dim, index)`: CPU's `gather_storage` walk for walk.
    pub(crate) fn gather(
        input: &WgpuStorage,
        dim: usize,
        index: &WgpuStorage,
    ) -> Result<WgpuStorage> {
        require_f32(input, "gather_input")?;
        let data = input.buffer.to_vec::<f32>()?;
        let idx_vals = index_values(index)?;
        let out_shape = index.shape.to_vec();
        let out_strides = host_strides(&out_shape);
        let in_strides = host_strides(&input.shape);

        let mut out_vals = Vec::with_capacity(out_shape.iter().product::<usize>());
        let mut out_idx = vec![0usize; out_shape.len()];
        let total = out_shape.iter().product::<usize>();
        for _ in 0..total {
            // CPU's `gather_storage` reads `index.get(&idx)` — the full
            // multi-dim coordinate — not `idx[dim]`. Flatten `out_idx` over
            // `out_shape` to reach the same entry of `idx_vals`.
            let flat_idx: usize = out_idx
                .iter()
                .zip(out_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            let target = checked_address(
                idx_vals[flat_idx],
                input.shape[dim],
                OperationKind::Gather,
                "gather_index",
            )?;
            let mut src_idx = out_idx.clone();
            src_idx[dim] = target;
            let flat: usize = src_idx
                .iter()
                .zip(in_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            out_vals.push(data[flat]);
            if !out_shape.is_empty() {
                host_increment(&mut out_idx, &out_shape);
            }
        }
        let out = WgpuStorage::new(WgpuBuffer::from_slice(&out_vals), out_shape);

        let t_cap_shape = input.shape.to_vec();
        let index_shape = index.shape.to_vec();
        let captured_idx = idx_vals.clone();
        let (t_id, out_id) = (input.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let t_total = t_cap_shape.iter().product::<usize>();
                let mut grad_t = vec![0.0f32; t_total];
                let t_strides = host_strides(&t_cap_shape);
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let grad_strides = host_strides(&grad_out.shape);
                let mut idx = vec![0usize; index_shape.len()];
                for (flat_out, &raw_target) in captured_idx.iter().enumerate() {
                    // Reconstruct `idx` from `flat_out` over `index_shape`.
                    let mut rem = flat_out;
                    for axis in (0..index_shape.len()).rev() {
                        idx[axis] = rem % index_shape[axis];
                        rem /= index_shape[axis];
                    }
                    let target = raw_target as usize;
                    let mut src_idx = idx.clone();
                    src_idx[dim] = target;
                    let flat_dst: usize = src_idx
                        .iter()
                        .zip(t_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    let flat_g: usize = idx
                        .iter()
                        .zip(grad_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    if flat_dst < grad_t.len() {
                        grad_t[flat_dst] += grad_data[flat_g];
                    }
                }
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad_t),
                    t_cap_shape.clone(),
                )])
            }),
        });
        Ok(out)
    }

    /// `index_select(input, dim, index)`: CPU's `index_select_storage`
    /// walk for walk — output shape is `input` with `dim` replaced by
    /// `index.len()`, backward is the same scatter-add as `gather`.
    pub(crate) fn index_select(
        input: &WgpuStorage,
        dim: usize,
        index: &WgpuStorage,
    ) -> Result<WgpuStorage> {
        require_f32(input, "index_select_input")?;
        let data = input.buffer.to_vec::<f32>()?;
        let idx_vals = index_values(index)?;
        let count = idx_vals.len();

        let mut out_shape = input.shape.to_vec();
        out_shape[dim] = count;
        let in_strides = host_strides(&input.shape);
        let total = out_shape.iter().product::<usize>();
        let mut out_vals = Vec::with_capacity(total);
        let mut out_idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let selected = checked_address(
                idx_vals[out_idx[dim]],
                input.shape[dim],
                OperationKind::IndexSelect,
                "index_select_index",
            )?;
            let mut src_idx = out_idx.clone();
            src_idx[dim] = selected;
            let flat: usize = src_idx
                .iter()
                .zip(in_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            out_vals.push(data[flat]);
            if !out_shape.is_empty() {
                host_increment(&mut out_idx, &out_shape);
            }
        }
        let out = WgpuStorage::new(WgpuBuffer::from_slice(&out_vals), out_shape);

        let t_cap_shape = input.shape.to_vec();
        let captured_idx = idx_vals.clone();
        let (t_id, out_id) = (input.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let t_total = t_cap_shape.iter().product::<usize>();
                let mut grad_t = vec![0.0f32; t_total];
                let t_strides = host_strides(&t_cap_shape);
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                for (flat_g, &g) in grad_data.iter().enumerate() {
                    let mut rem = flat_g;
                    let mut gidx = vec![0usize; grad_out.shape.len()];
                    for axis in (0..grad_out.shape.len()).rev() {
                        gidx[axis] = rem % grad_out.shape[axis];
                        rem /= grad_out.shape[axis];
                    }
                    let selected = captured_idx[gidx[dim]] as usize;
                    let mut src_idx = gidx.clone();
                    src_idx[dim] = selected;
                    let flat_dst: usize = src_idx
                        .iter()
                        .zip(t_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    if flat_dst < grad_t.len() {
                        grad_t[flat_dst] += g;
                    }
                }
                Ok(vec![WgpuStorage::new(
                    WgpuBuffer::from_slice(&grad_t),
                    t_cap_shape.clone(),
                )])
            }),
        });
        Ok(out)
    }

    /// `masked_fill(input, mask, value)` through `select.wgsl` mode 1.
    ///
    /// The mask is pre-broadcast to the input's geometry with the *raw*
    /// no-tape form (`broadcast_storage_raw`): a `bool` mask has nowhere to
    /// send a gradient, the same reason CUDA's `Execute<MaskedFill>` uses
    /// `launch_broadcast` rather than `broadcast_as`. The forward admits a
    /// mask that would *enlarge* the input by name, matching CPU's
    /// `admit_broadcast_operand` and `validated.rs` verbatim (#100).
    pub(crate) fn masked_fill(
        input: &WgpuStorage,
        mask: &WgpuStorage,
        value: f64,
    ) -> Result<WgpuStorage> {
        require_f32(input, "masked_fill")?;
        if mask.dtype != DTypeId::Bool.descriptor() {
            return Err(Error::UnsupportedDType {
                dtype: mask.dtype,
                backend: "Wgpu",
                op: "masked_fill_mask",
            });
        }
        // #100: the mask must broadcast *into* the input without enlarging it.
        let broadcasted =
            crate::layout::broadcast_shape(&mask.shape, &input.shape).map_err(|_| {
                Error::ShapeMismatch {
                    op: "masked_fill",
                    expected: input.shape.to_vec(),
                    got: mask.shape.to_vec(),
                    msg: "mask must broadcast to the input shape".into(),
                }
            })?;
        if input.shape != broadcasted {
            return Err(Error::ShapeMismatch {
                op: "masked_fill",
                expected: input.shape.to_vec(),
                got: mask.shape.to_vec(),
                msg: "mask must broadcast to the input shape".into(),
            });
        }
        let mask_b = if mask.shape == input.shape {
            mask.clone()
        } else {
            broadcast_storage_raw(mask, &input.shape)?
        };

        let n = checked_u32(num_elements(&input.shape)?, "masked_fill element count")?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, n as usize, OperationKind::Storage)?;
        let params = [1u32, n, (value as f32).to_bits()];
        // `b` is unused by mode 1; bind the input again so the layout is fixed.
        dispatch::dispatch_select(
            &mask_b.buffer,
            &input.buffer,
            &input.buffer,
            &out_buf,
            &params,
        );
        let out = WgpuStorage::new(out_buf, input.shape.to_vec());

        let mask_capture = mask_b.clone();
        let (t_id, out_id) = (input.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let n = checked_u32(
                    num_elements(&grad_out.shape)?,
                    "masked_fill backward element count",
                )?;
                let g_buf =
                    WgpuBuffer::new_zeros_for(DTypeId::F32, n as usize, OperationKind::Storage)?;
                let params = [1u32, n, 0.0f32.to_bits()];
                dispatch::dispatch_select(
                    &mask_capture.buffer,
                    &grad_out.buffer,
                    &grad_out.buffer,
                    &g_buf,
                    &params,
                );
                Ok(vec![WgpuStorage::new(g_buf, grad_out.shape.to_vec())])
            }),
        });
        Ok(out)
    }

    /// `where_cond(mask, on_true, on_false)` through `select.wgsl` mode 0.
    ///
    /// Follows CUDA's `Execute<WhereCond>` composition exactly: fold the
    /// output shape as `broadcast(mask, broadcast(on_true, on_false))`,
    /// pre-broadcast the mask with the raw no-tape form and the values with
    /// the tape-recording `broadcast_storage`, then push ONE entry whose
    /// `input_ids` are the *broadcasted* value ids (so the generic walk
    /// continues through whatever entries those broadcasts pushed) and whose
    /// backward reuses the forward kernel as
    /// `where(mask, grad, zeros)` / `where(mask, zeros, grad)`.
    pub(crate) fn where_cond(
        mask: &WgpuStorage,
        on_true: &WgpuStorage,
        on_false: &WgpuStorage,
    ) -> Result<WgpuStorage> {
        require_f32(on_true, "where_cond_on_true")?;
        require_f32(on_false, "where_cond_on_false")?;
        if mask.dtype != DTypeId::Bool.descriptor() {
            return Err(Error::UnsupportedDType {
                dtype: mask.dtype,
                backend: "Wgpu",
                op: "where_cond_mask",
            });
        }
        let base_shape = crate::layout::broadcast_shape(&on_true.shape, &on_false.shape)
            .map_err(|e| Error::Msg(alloc::format!("where_cond: {e}")))?;
        let out_shape = crate::layout::broadcast_shape(&mask.shape, &base_shape)
            .map_err(|e| Error::Msg(alloc::format!("where_cond: {e}")))?;

        let mask_b = if mask.shape == out_shape {
            mask.clone()
        } else {
            broadcast_storage_raw(mask, &out_shape)?
        };
        let true_b = if on_true.shape == out_shape {
            on_true.clone()
        } else {
            broadcast_storage(on_true, &out_shape)?
        };
        let false_b = if on_false.shape == out_shape {
            on_false.clone()
        } else {
            broadcast_storage(on_false, &out_shape)?
        };

        let n = checked_u32(num_elements(&out_shape)?, "where_cond element count")?;
        let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, n as usize, OperationKind::Storage)?;
        let params = [0u32, n, 0u32];
        dispatch::dispatch_select(
            &mask_b.buffer,
            &true_b.buffer,
            &false_b.buffer,
            &out_buf,
            &params,
        );
        let out = WgpuStorage::new(out_buf, out_shape.clone());

        let mask_capture = mask_b.clone();
        let shape_capture = out_shape.clone();
        let (true_id, false_id, out_id) = (true_b.id, false_b.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![true_id, false_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let n = checked_u32(
                    num_elements(&shape_capture)?,
                    "where_cond backward element count",
                )?;
                let zeros =
                    WgpuBuffer::new_zeros_for(DTypeId::F32, n as usize, OperationKind::Storage)?;
                let zeros = WgpuStorage::new(zeros, shape_capture.clone());
                let params = [0u32, n, 0u32];
                let g_true_buf =
                    WgpuBuffer::new_zeros_for(DTypeId::F32, n as usize, OperationKind::Storage)?;
                dispatch::dispatch_select(
                    &mask_capture.buffer,
                    &grad_out.buffer,
                    &zeros.buffer,
                    &g_true_buf,
                    &params,
                );
                let g_false_buf =
                    WgpuBuffer::new_zeros_for(DTypeId::F32, n as usize, OperationKind::Storage)?;
                dispatch::dispatch_select(
                    &mask_capture.buffer,
                    &zeros.buffer,
                    &grad_out.buffer,
                    &g_false_buf,
                    &params,
                );
                Ok(vec![
                    WgpuStorage::new(g_true_buf, shape_capture.clone()),
                    WgpuStorage::new(g_false_buf, shape_capture.clone()),
                ])
            }),
        });
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute impls
// ─────────────────────────────────────────────────────────────────────────────

impl<D: Device> Execute<op::EmbeddingExact> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::EmbeddingExact, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::EmbeddingExact;
        let [indices, weight] = request.inputs else {
            return Err(invalid(
                operation,
                "embedding expects an index tensor and a weight table",
            ));
        };
        let indices = indices
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "indices is not WGPU storage"))?;
        let weight = weight
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "weight is not WGPU storage"))?;
        WgpuBackendImpl::<D>::embedding(indices, weight)
            .map_err(|e| kernel_error("Wgpu", operation, e))
    }
}

impl<D: Device> Execute<op::Gather> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Gather, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::Gather;
        let [input, index] = request.inputs else {
            return Err(invalid(operation, "gather expects 2 inputs"));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "input is not WGPU storage"))?;
        let index = index
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "index is not WGPU storage"))?;
        let axis = request.operation.descriptor().attributes().axis;
        WgpuBackendImpl::<D>::gather(input, axis, index)
            .map_err(|e| kernel_error("Wgpu", operation, e))
    }
}

impl<D: Device> Execute<op::IndexSelect> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::IndexSelect, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::IndexSelect;
        let [input, index] = request.inputs else {
            return Err(invalid(operation, "index_select expects 2 inputs"));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "input is not WGPU storage"))?;
        let index = index
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "index is not WGPU storage"))?;
        let axis = request.operation.descriptor().attributes().axis;
        WgpuBackendImpl::<D>::index_select(input, axis, index)
            .map_err(|e| kernel_error("Wgpu", operation, e))
    }
}

impl<D: Device> Execute<op::MaskedFill> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MaskedFill, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::MaskedFill;
        let [input, mask] = request.inputs else {
            return Err(invalid(
                operation,
                "masked_fill expects exactly two operands",
            ));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "input is not WGPU storage"))?;
        let mask = mask
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "mask is not WGPU storage"))?;
        let value = request.operation.descriptor().attributes().value;
        WgpuBackendImpl::<D>::masked_fill(input, mask, value)
            .map_err(|e| kernel_error("Wgpu", operation, e))
    }
}

impl<D: Device> Execute<op::WhereCond> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::WhereCond, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::WhereCond;
        let [mask, on_true, on_false] = request.inputs else {
            return Err(invalid(
                operation,
                "where_cond expects exactly three operands",
            ));
        };
        let mask = mask
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "mask is not WGPU storage"))?;
        let on_true = on_true
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "on_true is not WGPU storage"))?;
        let on_false = on_false
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "on_false is not WGPU storage"))?;
        WgpuBackendImpl::<D>::where_cond(mask, on_true, on_false)
            .map_err(|e| kernel_error("Wgpu", operation, e))
    }
}
