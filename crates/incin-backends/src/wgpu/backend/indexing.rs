//! WGPU indexing ops: `embedding`/`gather`/`index_select`/`diag`/`scatter`/
//! `scatter_add`/`one_hot`/`bincount` host-walks and the `masked_fill`/`where_cond` GPU selection
//! kernel.
//!
//! The gather-family operations download their operands, compute on the
//! host exactly as `cpu::ops::{embedding, shape_ops::select}` do, and upload
//! the `f32` result — the same host-walk pattern `triangular`/`pad`/`repeat`
//! already use. Their tape recipes are copied from CPU walk-for-walk: ONE
//! `TapeEntry` per forward, `input_ids = vec![data.id]` only (the integer
//! index operand is off the tape), and a scatter-add backward that
//! accumulates repeated selections rather than overwriting them.
//! `diag` walks `cpu::ops::shape_ops::triangular`'s diagonal forms the same
//! way; `scatter`/`scatter_add` walk `scatter_storage`/`scatter_add_storage`
//! with both value operands on the tape.
//!
//! `masked_fill`/`where_cond` take the GPU path: a new `select.wgsl` with two
//! modes, fed by a pre-broadcast mask (raw, no tape — a `bool` mask has
//! nowhere to send a gradient) and pre-broadcast values (tape-recording, for
//! the same reason CUDA's `Execute<WhereCond>` records them).

use super::*;
use crate::descriptor_bind::{invalid, kernel_error};
use incin_core::exec::catalog::DuplicateIndexRule;

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

/// Out-of-range check for a `scatter` index value, mirroring CUDA's
/// `launch_scatter` error flag: any value outside `[0, bound)` is a typed
/// [`BackendError::InvalidInput`] refusal, not a silent drop.
///
/// CPU's `scatter_storage` and Metal's `scatter` both drop these writes
/// quietly (`flat_dest < out_data.len()`), but this slice follows CUDA:
/// a scatter that names a position outside the operand is a caller bug,
/// and clamping or dropping would hand back an answer that silently
/// omitted a contribution. (`scatter_add` is the deliberate exception —
/// CUDA's ordered kernel and CPU both drop there, so this backend drops
/// there too. See [`WgpuBackendImpl::scatter_add`].)
fn scatter_address(raw: i64, bound: usize, operation: OperationKind) -> Result<usize> {
    match usize::try_from(raw) {
        Ok(address) if address < bound => Ok(address),
        _ => Err(Error::Backend(BackendError::InvalidInput {
            operation,
            reason: "index out of bounds",
        })),
    }
}

/// [`kernel_error`], but an already-typed [`Error::Backend`] is returned
/// as-is — the WGPU twin of CUDA's `index_kernel_error` (`cuda/executor.rs`).
///
/// The scatter pair raises [`BackendError::InvalidInput`] when an index
/// value falls outside the axis it names: a refusal of the operand, not a
/// device failure. Handing that to [`kernel_error`] would flatten it into
/// `Execution` with the `InvalidInput` text buried in the message, so a
/// caller matching on the typed variant would see "the backend executed
/// something and it failed" for an input rejected before any element was
/// read. Stays local to the new index executors below rather than changing
/// the shared helper.
fn index_kernel_error(operation: OperationKind, error: Error) -> BackendError {
    match error {
        Error::Backend(inner) => inner,
        other => kernel_error("Wgpu", operation, other),
    }
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

    /// `one_hot(indices, depth)`: append a `depth` axis, setting position
    /// `indices[i]` to `1.0` and the rest to `0.0` — CPU's `one_hot_storage`
    /// walk for walk.
    ///
    /// The result is `Bool` storage over the physical `f32` 0.0/1.0 encoding
    /// `wgpu/storage.rs::physical_element_bytes` documents, so it feeds the
    /// `masked_fill`/`where_cond` mask path without a conversion. Indices
    /// outside `0..depth` write no `1` (CPU's ONNX skip: the value names no
    /// slot, so no slot is set, and a padding index flows through instead of
    /// aborting the batch). Forward-only: no tape entry, the same
    /// `GradientRule::None` family `nonzero`/`bincount` belong to.
    pub(crate) fn one_hot(indices: &WgpuStorage, depth: usize) -> Result<WgpuStorage> {
        if depth == 0 {
            return Err(Error::Msg("one_hot: depth must be at least one".into()));
        }
        let idx_vals = index_values(indices)?;
        let total = num_elements(&indices.shape)?;
        let out_elements =
            total
                .checked_mul(depth)
                .ok_or(ShapeError::ArithmeticOverflow {
                    operation: OperationKind::OneHot,
                    expression: "one_hot output element count",
                })?;
        let mut out_shape = indices.shape.to_vec();
        out_shape.push(depth);
        let mut out = alloc::vec![0.0f32; out_elements];
        for (flat, &raw) in idx_vals.iter().enumerate() {
            if raw >= 0 && (raw as usize) < depth {
                out[flat * depth + raw as usize] = 1.0;
            }
        }
        Ok(WgpuStorage::new_with_dtype(
            WgpuBuffer::from_slice(&out),
            out_shape,
            DTypeId::Bool.descriptor(),
        ))
    }

    /// `bincount(indices, bins)`: count how many times each of `bins` slots
    /// appears in the integer index operand — CPU's `bincount_storage` walk
    /// for walk, through the same `index_values` accessor `one_hot` uses.
    ///
    /// The output shape `[bins]` is static (named by the attribute, not the
    /// values), which is what separates this slice from `nonzero`'s
    /// data-dependent extent. An index outside `0..bins` is a typed
    /// `InvalidInput` refusal, not a silent skip: a quietly-low count makes
    /// every offset derived from it wrong (CPU's
    /// `bincount_storage` errors for the same reason; `one_hot`'s ONNX skip
    /// does not apply here). Forward-only: no tape entry, the same
    /// `GradientRule::None` family `one_hot`/`nonzero` belong to.
    pub(crate) fn bincount(indices: &WgpuStorage, bins: usize) -> Result<WgpuStorage> {
        if bins == 0 {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Bincount,
                reason: "bincount needs at least one bin to count into",
            }));
        }
        let idx_vals = index_values(indices)?;
        let mut counts = alloc::vec![0i64; bins];
        for &raw in &idx_vals {
            // `index_values` already refused non-integer dtypes by name, so
            // every value here is whole; only the range remains to check.
            match usize::try_from(raw) {
                Ok(address) if address < bins => counts[address] += 1,
                _ => {
                    return Err(Error::Backend(BackendError::InvalidInput {
                        operation: OperationKind::Bincount,
                        reason: "bincount index is not a whole number inside the bin range",
                    }));
                }
            }
        }
        Ok(WgpuStorage::new_with_dtype(
            WgpuBuffer::from_slice(&counts),
            alloc::vec![bins],
            DTypeId::I64.descriptor(),
        ))
    }

    /// `diag(input, diagonal)`: CPU's `diag_storage` walk for walk.
    ///
    /// Rank 1 builds the diagonal matrix (`[n]` -> `[n + |k|, n + |k|]`,
    /// entry `i` at `(row, col)` with the same `k >= 0` row/col split CPU
    /// uses); rank 2 extracts the `k`-th diagonal as a vector, reading the
    /// first trailing matrix the way CPU's `diagonal_vector` does. Ranks
    /// outside 1–2 are refused like `tril`/`triu` are — the descriptor's
    /// `DiagonalAttributes` already rejects them before dispatch, so this
    /// is defense-in-depth with the same `ShapeMismatch` shape. Backward
    /// mirrors CPU in both directions: construct's adjoint is the extract
    /// with the same offset, extract's adjoint scatters the cotangent back
    /// onto the diagonal positions. ONE `TapeEntry`, `input_ids = vec![id]`.
    pub(crate) fn diag(input: &WgpuStorage, diagonal: i64) -> Result<WgpuStorage> {
        require_f32(input, "diag_input")?;
        let rank = input.shape.len();
        if rank != 1 && rank != 2 {
            return Err(Error::ShapeMismatch {
                op: "diag",
                expected: vec![1, 2],
                got: vec![rank],
                msg: "diag accepts rank 1 or 2".into(),
            });
        }
        let data = input.buffer.to_vec::<f32>()?;
        if rank == 1 {
            let n = input.shape[0];
            let k_abs = diagonal.unsigned_abs() as usize;
            let out_dim =
                n.checked_add(k_abs)
                    .ok_or(ShapeError::ArithmeticOverflow {
                        operation: OperationKind::Diag,
                        expression: "diagonal output dimension",
                    })?;
            let out_total =
                out_dim
                    .checked_mul(out_dim)
                    .ok_or(ShapeError::ArithmeticOverflow {
                        operation: OperationKind::Diag,
                        expression: "diagonal output element count",
                    })?;
            let mut out_vals = vec![0.0f32; out_total];
            for (i, &value) in data.iter().enumerate() {
                let row = if diagonal >= 0 { i } else { i + k_abs };
                let col = if diagonal >= 0 { i + k_abs } else { i };
                if row < out_dim && col < out_dim {
                    out_vals[row * out_dim + col] = value;
                }
            }
            let out = WgpuStorage::new(WgpuBuffer::from_slice(&out_vals), vec![out_dim, out_dim]);

            let (t_id, out_id) = (input.id, out.id);
            crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
                output_id: out_id,
                input_ids: vec![t_id],
                backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                    let grad_data = grad_out.buffer.to_vec::<f32>()?;
                    let mut vals = Vec::with_capacity(n);
                    for i in 0..n {
                        let row = if diagonal >= 0 { i } else { i + k_abs };
                        let col = if diagonal >= 0 { i + k_abs } else { i };
                        vals.push(if row < out_dim && col < out_dim {
                            grad_data[row * out_dim + col]
                        } else {
                            0.0
                        });
                    }
                    Ok(vec![WgpuStorage::new(
                        WgpuBuffer::from_slice(&vals),
                        vec![n],
                    )])
                }),
            });
            Ok(out)
        } else {
            let (rows, cols) = (input.shape[0], input.shape[1]);
            let mut values: Vec<f32> = Vec::new();
            let mut positions: Vec<(usize, usize)> = Vec::new();
            for row in 0..rows {
                let Some(col) = (row as i64).checked_add(diagonal).filter(|&col| col >= 0)
                else {
                    continue;
                };
                let col = col as usize;
                if col < cols {
                    values.push(data[row * cols + col]);
                    positions.push((row, col));
                }
            }
            let len = values.len();
            let out = WgpuStorage::new(WgpuBuffer::from_slice(&values), vec![len]);

            let original_shape = input.shape.to_vec();
            let (t_id, out_id) = (input.id, out.id);
            crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
                output_id: out_id,
                input_ids: vec![t_id],
                backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                    let grad_data = grad_out.buffer.to_vec::<f32>()?;
                    let total = original_shape.iter().product::<usize>();
                    let mut grads = vec![0.0f32; total];
                    let strides = host_strides(&original_shape);
                    for (i, &(row, col)) in positions.iter().enumerate() {
                        let flat: usize = [row, col]
                            .iter()
                            .zip(strides.iter())
                            .map(|(&coordinate, &stride)| coordinate * stride)
                            .sum();
                        grads[flat] += grad_data[i];
                    }
                    Ok(vec![WgpuStorage::new(
                        WgpuBuffer::from_slice(&grads),
                        original_shape.clone(),
                    )])
                }),
            });
            Ok(out)
        }
    }

    /// `scatter(input, dim, index, source)`: CPU's `scatter_storage` walk
    /// for walk — last-write-wins in row-major index order, output shaped
    /// like the input.
    ///
    /// Backward mirrors CPU's exactly: the input keeps its cotangent
    /// everywhere EXCEPT the positions a write overwrote, and the source
    /// receives the output cotangent only through the LAST write to each
    /// destination (earlier writes to the same position contributed
    /// nothing). ONE `TapeEntry` with `input_ids = vec![t.id, src.id]`;
    /// the integer index operand is off the tape.
    ///
    /// Two deliberate divergences from the CPU/Metal host walks, both
    /// mirroring CUDA's `launch_scatter` instead: an index value outside
    /// `[0, input.shape[dim])` is a typed `InvalidInput` refusal (see
    /// [`scatter_address`]), and rank/shape mismatches among the three
    /// operands are `InvalidInput` too (CUDA's launch prologue wording),
    /// rather than panics or silent misreads.
    pub(crate) fn scatter(
        input: &WgpuStorage,
        dim: usize,
        index: &WgpuStorage,
        source: &WgpuStorage,
    ) -> Result<WgpuStorage> {
        require_f32(input, "scatter_input")?;
        require_f32(source, "scatter_source")?;
        let rank = input.shape.len();
        if dim >= rank {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Scatter,
                reason: "scatter axis is outside the operand's rank",
            }));
        }
        if index.shape.len() != rank || source.shape.len() != rank {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Scatter,
                reason: "scatter index and source ranks must match the input rank",
            }));
        }
        if index.shape != source.shape {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Scatter,
                reason: "scatter expects the index and source operands to share a shape",
            }));
        }
        let bound = input.shape[dim];
        let data = input.buffer.to_vec::<f32>()?;
        let src_data = source.buffer.to_vec::<f32>()?;
        let idx_vals = index_values(index)?;
        let in_strides = host_strides(&input.shape);
        let idx_strides = host_strides(&index.shape);

        let mut out_data = data;
        // Every write as `(flat_dest, src_coord)` in row-major order, so
        // the backward can recover the surviving (last) write per
        // destination exactly as CPU's `written_flat_dest` does.
        let mut written: Vec<(usize, Vec<usize>)> = Vec::with_capacity(idx_vals.len());
        let mut idx = vec![0usize; rank];
        let total = index.shape.iter().product::<usize>();
        for _ in 0..total {
            let flat: usize = idx
                .iter()
                .zip(idx_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            let target = scatter_address(idx_vals[flat], bound, OperationKind::Scatter)?;
            let mut dest_idx = idx.clone();
            dest_idx[dim] = target;
            let flat_dest: usize = dest_idx
                .iter()
                .zip(in_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            out_data[flat_dest] = src_data[flat];
            written.push((flat_dest, idx.clone()));
            if !index.shape.is_empty() {
                host_increment(&mut idx, &index.shape);
            }
        }
        let out = WgpuStorage::new(WgpuBuffer::from_slice(&out_data), input.shape.to_vec());

        let in_shape = input.shape.to_vec();
        let src_shape = source.shape.to_vec();
        let in_total = in_shape.iter().product::<usize>();
        let src_total = src_shape.iter().product::<usize>();
        let src_strides = host_strides(&src_shape);
        let mut last_write_of_dest: alloc::collections::BTreeMap<usize, usize> =
            alloc::collections::BTreeMap::new();
        for (position, &(flat_dest, _)) in written.iter().enumerate() {
            last_write_of_dest.insert(flat_dest, position);
        }
        let surviving: Vec<(usize, Vec<usize>)> = last_write_of_dest
            .into_iter()
            .filter_map(|(flat_dest, position)| {
                written
                    .get(position)
                    .map(|(_, src_idx)| (flat_dest, src_idx.clone()))
            })
            .collect();
        let mut overwritten = vec![false; in_total];
        for &(flat_dest, _) in &surviving {
            overwritten[flat_dest] = true;
        }
        let (t_id, source_id, out_id) = (input.id, source.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id, source_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let mut grad_t = vec![0.0f32; in_total];
                for (i, item) in grad_t.iter_mut().enumerate() {
                    if !overwritten[i] {
                        *item = grad_data[i];
                    }
                }
                let mut grad_source = vec![0.0f32; src_total];
                for (flat_dest, src_coord) in &surviving {
                    let flat_src: usize = src_coord
                        .iter()
                        .zip(src_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    grad_source[flat_src] += grad_data[*flat_dest];
                }
                Ok(vec![
                    WgpuStorage::new(WgpuBuffer::from_slice(&grad_t), in_shape.clone()),
                    WgpuStorage::new(WgpuBuffer::from_slice(&grad_source), src_shape.clone()),
                ])
            }),
        });
        Ok(out)
    }

    /// `scatter_add(input, dim, index, source)`: CPU's
    /// `scatter_add_storage` walk for walk — colliding writes sum in
    /// row-major index order rather than last-write-wins.
    ///
    /// No WGSL kernel and no float atomics are involved, which is what
    /// makes this slice feasible under #91's no-float-atomics constraint:
    /// the accumulation is an ordinary sequential `+=` over downloaded
    /// `f32`, in exactly the row-major order the catalog's determinism
    /// contract names. A parallel WGSL scatter-add would need
    /// `atomicAdd` on floats (unavailable) and would still race on
    /// duplicate indices the way CUDA's atomic kernel did before its
    /// ordered two-pass replacement; the host walk *is* the ordered
    /// path, with one rounding per contribution like CPU.
    ///
    /// Out-of-range destinations are dropped rather than refused —
    /// mirroring CUDA's ordered kernel and CPU, not [`scatter_address`]:
    /// the typed `InvalidInput` refusal belongs to the overwriting
    /// `scatter`, whose dropped write would silently change which value
    /// survives. Backward mirrors CPU: the input's cotangent passes
    /// through untouched (addition displaces nothing) and every recorded
    /// write takes the output cotangent at its destination. ONE
    /// `TapeEntry`, index off-tape.
    pub(crate) fn scatter_add(
        input: &WgpuStorage,
        dim: usize,
        index: &WgpuStorage,
        source: &WgpuStorage,
    ) -> Result<WgpuStorage> {
        require_f32(input, "scatter_add_input")?;
        require_f32(source, "scatter_add_source")?;
        let rank = input.shape.len();
        if dim >= rank {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::ScatterAdd,
                reason: "scatter_add axis is outside the operand's rank",
            }));
        }
        if index.shape.len() != rank || source.shape.len() != rank {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::ScatterAdd,
                reason: "scatter_add index and source ranks must match the input rank",
            }));
        }
        if index.shape != source.shape {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::ScatterAdd,
                reason: "scatter_add expects the index and source operands to share a shape",
            }));
        }
        let bound = input.shape[dim];
        let data = input.buffer.to_vec::<f32>()?;
        let src_data = source.buffer.to_vec::<f32>()?;
        let idx_vals = index_values(index)?;
        let in_strides = host_strides(&input.shape);
        let idx_strides = host_strides(&index.shape);

        let mut out_data = data;
        // Every surviving write, in order — unlike `scatter` there is no
        // survivor to identify later: every contribution sums, so each
        // recorded write earned its cotangent.
        let mut writes: Vec<(usize, Vec<usize>)> = Vec::with_capacity(idx_vals.len());
        let mut idx = vec![0usize; rank];
        let total = index.shape.iter().product::<usize>();
        for _ in 0..total {
            let flat: usize = idx
                .iter()
                .zip(idx_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            let raw = idx_vals[flat];
            // Out-of-range destinations are dropped rather than refused:
            // CUDA's ordered kernel and CPU both drop here (clamping would
            // silently add a contribution to a row the caller never named).
            let target = match usize::try_from(raw) {
                Ok(address) if address < bound => Some(address),
                _ => None,
            };
            if let Some(target) = target {
                let mut dest_idx = idx.clone();
                dest_idx[dim] = target;
                let flat_dest: usize = dest_idx
                    .iter()
                    .zip(in_strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum();
                out_data[flat_dest] += src_data[flat];
                writes.push((flat_dest, idx.clone()));
            }
            if !index.shape.is_empty() {
                host_increment(&mut idx, &index.shape);
            }
        }
        let out = WgpuStorage::new(WgpuBuffer::from_slice(&out_data), input.shape.to_vec());

        let in_shape = input.shape.to_vec();
        let src_shape = source.shape.to_vec();
        let in_total = in_shape.iter().product::<usize>();
        let src_total = src_shape.iter().product::<usize>();
        let src_strides = host_strides(&src_shape);
        let (t_id, source_id, out_id) = (input.id, source.id, out.id);
        crate::wgpu::tape::push_with(|| crate::wgpu::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id, source_id],
            backward: alloc::boxed::Box::new(move |grad_out: &WgpuStorage| {
                let grad_data = grad_out.buffer.to_vec::<f32>()?;
                let mut grad_t = vec![0.0f32; in_total];
                for (i, item) in grad_t.iter_mut().enumerate() {
                    *item = grad_data[i];
                }
                let mut grad_source = vec![0.0f32; src_total];
                for (flat_dest, src_coord) in &writes {
                    let flat_src: usize = src_coord
                        .iter()
                        .zip(src_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    grad_source[flat_src] += grad_data[*flat_dest];
                }
                Ok(vec![
                    WgpuStorage::new(WgpuBuffer::from_slice(&grad_t), in_shape.clone()),
                    WgpuStorage::new(WgpuBuffer::from_slice(&grad_source), src_shape.clone()),
                ])
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
        )?;
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
                )?;
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
        )?;
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
                )?;
                let g_false_buf =
                    WgpuBuffer::new_zeros_for(DTypeId::F32, n as usize, OperationKind::Storage)?;
                dispatch::dispatch_select(
                    &mask_capture.buffer,
                    &zeros.buffer,
                    &grad_out.buffer,
                    &g_false_buf,
                    &params,
                )?;
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

/// `one_hot`: index tensor plus `depth` — bool-mask host walk above.
/// Forward-only (no tape entry), matching CPU/Metal/CUDA.
impl<D: Device> Execute<op::OneHot> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::OneHot, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::OneHot;
        let [indices] = request.inputs else {
            return Err(invalid(operation, "one_hot expects exactly 1 input"));
        };
        let indices = indices
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "indices is not WGPU storage"))?;
        let depth = request.operation.descriptor().attributes().depth;
        WgpuBackendImpl::<D>::one_hot(indices, depth)
            .map_err(|e| kernel_error("Wgpu", operation, e))
    }
}

/// `bincount`: index tensor plus `bins` — i64-histogram host walk above.
/// Forward-only (no tape entry), matching CPU/CUDA.
impl<D: Device> Execute<op::Bincount> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Bincount, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::Bincount;
        let [indices] = request.inputs else {
            return Err(invalid(operation, "bincount expects exactly 1 input"));
        };
        let indices = indices
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "indices is not WGPU storage"))?;
        let bins = request.operation.descriptor().attributes().bins;
        WgpuBackendImpl::<D>::bincount(indices, bins)
            .map_err(|e| index_kernel_error(operation, e))
    }
}

impl<D: Device> Execute<op::Diag> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Diag, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::Diag;
        let [input] = request.inputs else {
            return Err(invalid(operation, "diag expects exactly 1 input"));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "input is not WGPU storage"))?;
        let diagonal = request.operation.descriptor().attributes().offset;
        WgpuBackendImpl::<D>::diag(input, diagonal)
            .map_err(|e| index_kernel_error(operation, e))
    }
}

impl<D: Device> Execute<op::Scatter> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Scatter, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::Scatter;
        let [input, index, source] = request.inputs else {
            return Err(invalid(operation, "scatter expects 3 inputs"));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "input is not WGPU storage"))?;
        let index = index
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "index is not WGPU storage"))?;
        let source = source
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "source is not WGPU storage"))?;
        let attributes = request.operation.descriptor().attributes();
        if attributes.duplicate_indices == DuplicateIndexRule::Reject {
            return Err(invalid(
                operation,
                "this backend applies last-write-wins and cannot reject duplicate indices",
            ));
        }
        WgpuBackendImpl::<D>::scatter(input, attributes.axis, index, source)
            .map_err(|e| index_kernel_error(operation, e))
    }
}

impl<D: Device> Execute<op::ScatterAdd> for WgpuBackendImpl<D> {
    type Output = WgpuStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::ScatterAdd, Self>,
    ) -> core::result::Result<WgpuStorage, BackendError> {
        let operation = OperationKind::ScatterAdd;
        let [input, index, source] = request.inputs else {
            return Err(invalid(operation, "scatter_add expects 3 inputs"));
        };
        let input = input
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "input is not WGPU storage"))?;
        let index = index
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "index is not WGPU storage"))?;
        let source = source
            .downcast_ref::<WgpuStorage>()
            .ok_or_else(|| invalid(operation, "source is not WGPU storage"))?;
        let attributes = request.operation.descriptor().attributes();
        if attributes.duplicate_indices != DuplicateIndexRule::Accumulate {
            return Err(invalid(
                operation,
                "scatter_add accumulates duplicate indices and implements no other rule; \
                 use scatter for last-write-wins",
            ));
        }
        WgpuBackendImpl::<D>::scatter_add(input, attributes.axis, index, source)
            .map_err(|e| index_kernel_error(operation, e))
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
