//! Metal indexing ops: `embedding`, `gather` and `index_select` host-walks.
//!
//! The three gather-family operations download their operands, compute on the
//! host exactly as `cpu::ops::{embedding, shape_ops::select}` do, and upload
//! the `f32` result — the same host-walk pattern `layout.rs` already uses.
//! Their tape recipes are copied from CPU/WGPU walk-for-walk: ONE
//! `TapeEntry` per forward, `input_ids = vec![data.id]` only (the integer
//! index operand is off the tape), and a scatter-add backward that
//! accumulates repeated selections rather than overwriting them.
//!
//! `masked_fill`/`where_cond` are deliberately absent: Metal's storage
//! validator refuses `bool` (`validate_metal_storage_dtype`), so a mask
//! tensor cannot exist on this backend, and no capability group in lane
//! carries an `F32_AND_BOOL` row. See the `logical = []` note in
//! `metal_descriptor_operations!`.

use incin_core::error::{Error, Result};
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DTypeId;

use super::backend::{MetalBackendImpl, storage_from_f32};
use super::storage::MetalStorage;

/// Row-major element count of a dims slice, as `backend.rs` spells it.
fn numel(dims: &[usize]) -> Result<usize> {
    Ok(ShapeBuf::from_slice(dims).checked_numel(OperationKind::Storage)?)
}

/// Download an integer index operand as `i64` values.
///
/// Only `i64` is creatable on Metal (`validate_metal_storage_dtype` refuses
/// `u8`/`u32`), so the other widths of `INDEX_AND_F32_DTYPES` are a vacuous
/// part of the row — storage creation fails closed before admission. Any
/// non-integer dtype is refused by name rather than reinterpreted as bits.
fn index_values(t: &MetalStorage) -> Result<Vec<i64>> {
    match t.metadata().dtype().builtin_id() {
        Some(DTypeId::I64) => Ok(bytemuck::cast_slice::<u8, i64>(t.as_bytes()?).to_vec()),
        _ => Err(Error::UnsupportedDType {
            dtype: t.metadata().dtype(),
            backend: "Metal",
            op: "index_values",
        }),
    }
}

/// Refuse any dtype that is not `f32` for a value (non-index) operand.
fn require_f32(t: &MetalStorage, op: &'static str) -> Result<()> {
    if t.metadata().dtype() == DTypeId::F32.descriptor() {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            dtype: t.metadata().dtype(),
            backend: "Metal",
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

/// Row-major contiguous strides for a host walk (same as `layout.rs`).
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

impl<D: Device> MetalBackendImpl<D> {
    /// Gather rows of `w` (shape `[vocab_size, hidden_size]`) addressed by
    /// the integer indices in `t` (any rank), producing
    /// `t.shape ++ [hidden_size]`.
    ///
    /// Forward and backward are `cpu::ops::embedding::embedding_impl` walk
    /// for walk: ONE `TapeEntry` with `input_ids = vec![w.id]` only, and a
    /// scatter-add that accumulates repeated indices.
    pub(crate) fn embedding(indices: &MetalStorage, weight: &MetalStorage) -> Result<MetalStorage> {
        require_f32(weight, "embedding_weight")?;
        let w_dims = weight.shape();
        if w_dims.len() != 2 {
            return Err(Error::ShapeMismatch {
                op: "embedding",
                expected: vec![0, 0],
                got: w_dims.to_vec(),
                msg: format!(
                    "embedding: weight table must be rank-2 [vocab_size, hidden_size], got shape {w_dims:?}"
                ),
            });
        }
        let vocab_size = w_dims[0];
        let hidden_size = w_dims[1];

        let weight_bytes = weight.as_bytes()?;
        let rows: &[f32] = bytemuck::cast_slice(weight_bytes);
        let raw_ids = index_values(indices)?;

        let mut row_indices: Vec<usize> = Vec::with_capacity(raw_ids.len());
        let mut out_vals: Vec<f32> = Vec::with_capacity(raw_ids.len() * hidden_size);
        for &raw in &raw_ids {
            let row =
                checked_address(raw, vocab_size, OperationKind::Embedding, "embedding_index")?;
            out_vals.extend_from_slice(&rows[row * hidden_size..(row + 1) * hidden_size]);
            row_indices.push(row);
        }

        let mut out_shape = indices.shape().to_vec();
        out_shape.push(hidden_size);
        let out = storage_from_f32(&out_vals, &out_shape, weight)?;

        let w_total: usize = w_dims.iter().product();
        let t_shape = indices.shape().to_vec();
        let w_shape = w_dims.to_vec();
        let weight_like = weight.clone();
        let (w_id, out_id) = (weight.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![w_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let grad_bytes = grad_out.as_bytes()?;
                let grad_data: &[f32] = bytemuck::cast_slice(grad_bytes);
                let grad_strides = host_strides(grad_out.shape());
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
                Ok(vec![storage_from_f32(&grad_w, &w_shape, &weight_like)?])
            }),
        });
        Ok(out)
    }

    /// `gather(input, dim, index)`: CPU's `gather_storage` walk for walk.
    pub(crate) fn gather(
        input: &MetalStorage,
        dim: usize,
        index: &MetalStorage,
    ) -> Result<MetalStorage> {
        require_f32(input, "gather_input")?;
        let in_dims = input.shape();
        if dim >= in_dims.len() {
            return Err(Error::ShapeMismatch {
                op: "gather",
                expected: in_dims.to_vec(),
                got: vec![dim],
                msg: "gather axis out of bounds".to_string(),
            });
        }
        let data_bytes = input.as_bytes()?;
        let data: &[f32] = bytemuck::cast_slice(data_bytes);
        let idx_vals = index_values(index)?;
        let out_shape = index.shape().to_vec();
        let out_strides = host_strides(&out_shape);
        let in_strides = host_strides(in_dims);

        let mut out_vals = Vec::with_capacity(numel(&out_shape)?);
        let mut out_idx = vec![0usize; out_shape.len()];
        let total = numel(&out_shape)?;
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
                in_dims[dim],
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
        let out = storage_from_f32(&out_vals, &out_shape, input)?;

        let t_cap_shape = in_dims.to_vec();
        let index_shape = index.shape().to_vec();
        let captured_idx = idx_vals.clone();
        let input_like = input.clone();
        let (t_id, out_id) = (input.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let t_total = numel(&t_cap_shape)?;
                let mut grad_t = vec![0.0f32; t_total];
                let t_strides = host_strides(&t_cap_shape);
                let grad_bytes = grad_out.as_bytes()?;
                let grad_data: &[f32] = bytemuck::cast_slice(grad_bytes);
                let grad_strides = host_strides(grad_out.shape());
                let mut coords = vec![0usize; index_shape.len()];
                for (flat_out, &raw_target) in captured_idx.iter().enumerate() {
                    let mut rem = flat_out;
                    for axis in (0..index_shape.len()).rev() {
                        coords[axis] = rem % index_shape[axis];
                        rem /= index_shape[axis];
                    }
                    let target = raw_target as usize;
                    let mut src_idx = coords.clone();
                    src_idx[dim] = target;
                    let flat_dst: usize = src_idx
                        .iter()
                        .zip(t_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    let flat_g: usize = coords
                        .iter()
                        .zip(grad_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    if flat_dst < grad_t.len() {
                        grad_t[flat_dst] += grad_data[flat_g];
                    }
                }
                Ok(vec![storage_from_f32(&grad_t, &t_cap_shape, &input_like)?])
            }),
        });
        Ok(out)
    }

    /// `index_select(input, dim, index)`: CPU's `index_select_storage`
    /// walk for walk — output shape is `input` with `dim` replaced by
    /// `index.len()`, backward is the same scatter-add as `gather`.
    pub(crate) fn index_select(
        input: &MetalStorage,
        dim: usize,
        index: &MetalStorage,
    ) -> Result<MetalStorage> {
        require_f32(input, "index_select_input")?;
        let in_dims = input.shape();
        if dim >= in_dims.len() {
            return Err(Error::ShapeMismatch {
                op: "index_select",
                expected: in_dims.to_vec(),
                got: vec![dim],
                msg: "index_select axis out of bounds".to_string(),
            });
        }
        let data_bytes = input.as_bytes()?;
        let data: &[f32] = bytemuck::cast_slice(data_bytes);
        let idx_vals = index_values(index)?;
        let count = idx_vals.len();

        let mut out_shape = in_dims.to_vec();
        out_shape[dim] = count;
        let in_strides = host_strides(in_dims);
        let total = numel(&out_shape)?;
        let mut out_vals = Vec::with_capacity(total);
        let mut out_idx = vec![0usize; out_shape.len()];
        for _ in 0..total {
            let selected = checked_address(
                idx_vals[out_idx[dim]],
                in_dims[dim],
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
        let out = storage_from_f32(&out_vals, &out_shape, input)?;

        let t_cap_shape = in_dims.to_vec();
        let captured_idx = idx_vals.clone();
        let input_like = input.clone();
        let (t_id, out_id) = (input.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let t_total = numel(&t_cap_shape)?;
                let mut grad_t = vec![0.0f32; t_total];
                let t_strides = host_strides(&t_cap_shape);
                let grad_bytes = grad_out.as_bytes()?;
                let grad_data: &[f32] = bytemuck::cast_slice(grad_bytes);
                for (flat_g, &g) in grad_data.iter().enumerate() {
                    let mut rem = flat_g;
                    let mut gidx = vec![0usize; grad_out.shape().len()];
                    for axis in (0..grad_out.shape().len()).rev() {
                        gidx[axis] = rem % grad_out.shape()[axis];
                        rem /= grad_out.shape()[axis];
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
                Ok(vec![storage_from_f32(&grad_t, &t_cap_shape, &input_like)?])
            }),
        });
        Ok(out)
    }
}

#[cfg(test)]
/// Host-side forward/backward parity tests for the gather family.
/// Pure `Vec<f32>` math, so they run without a Metal device.
mod tests {
    use super::*;
    use incin_core::exec::GradMode;
    use incin_core::tensor::device::{DeviceId, Metal};
    use incin_core::tensor::dtype::DTypeId;

    use crate::metal::storage::MetalStorageMode;
    use crate::metal::tape::MetalGrads;

    type B = MetalBackendImpl<Metal>;

    fn storage(values: &[f32], shape: &[usize]) -> MetalStorage {
        let bytes: Vec<u8> = bytemuck::cast_slice(values).to_vec();
        let meta = incin_core::exec::TensorMeta::contiguous(
            ShapeBuf::from_slice(shape),
            DTypeId::F32.into(),
            DeviceId::metal(0),
            MetalStorage::alignment(),
            values.len(),
        )
        .expect("contiguous metadata for test storage");
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, 0)
            .expect("bytes cover the metadata span")
    }

    fn indices(values: &[i64], shape: &[usize]) -> MetalStorage {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let meta = incin_core::exec::TensorMeta::contiguous(
            ShapeBuf::from_slice(shape),
            DTypeId::I64.into(),
            DeviceId::metal(0),
            MetalStorage::alignment(),
            values.len(),
        )
        .expect("contiguous metadata for index storage");
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, 0)
            .expect("bytes cover the metadata span")
    }

    fn read(s: &MetalStorage) -> Vec<f32> {
        bytemuck::cast_slice(s.as_bytes().expect("shared-mode storage is host-readable")).to_vec()
    }

    fn assert_close(got: &[f32], want: &[f32], eps: f32) {
        assert_eq!(
            got.len(),
            want.len(),
            "length mismatch: {got:?} vs {want:?}"
        );
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if *g == *w {
                continue;
            }
            let tol = eps * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "index {i}: got {g}, want {w} (eps {eps})"
            );
        }
    }

    fn recorded<F>(f: F) -> (MetalStorage, MetalGrads)
    where
        F: FnOnce() -> MetalStorage,
    {
        let out = GradMode::Enabled.scope(f);
        let grads = crate::metal::tape::backward(&out).expect("backward walk succeeds");
        (out, grads)
    }

    fn forward_recording<F>(f: F) -> (MetalStorage, usize)
    where
        F: FnOnce() -> MetalStorage,
    {
        let before = crate::metal::tape::depth();
        let out = GradMode::Enabled.scope(f);
        let delta = crate::metal::tape::depth() - before;
        (out, delta)
    }

    /// 3x2 weight table: row r is `[r * 10, r * 10 + 1]`.
    fn weight() -> MetalStorage {
        storage(&[0.0, 1.0, 10.0, 11.0, 20.0, 21.0], &[3, 2])
    }

    // ── embedding ──────────────────────────────────────────────────────────

    #[test]
    fn embedding_gathers_the_named_rows() {
        let w = weight();
        let idx = indices(&[2, 0, 2], &[3]);
        let out = B::embedding(&idx, &w).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_close(&read(&out), &[20.0, 21.0, 0.0, 1.0, 20.0, 21.0], 1e-6);
    }

    #[test]
    fn embedding_backward_accumulates_repeated_indices() {
        let w = weight();
        let idx = indices(&[1, 1, 0], &[3]);
        let (out, grads) = recorded(|| B::embedding(&idx, &w).unwrap());
        assert_eq!(read(&out).len(), 6, "embedding keeps the hidden size");
        // Ones seed: row 1 selected twice, row 0 once, row 2 never.
        assert_close(
            read(grads.get(w.id()).expect("embedding records a weight grad")).as_slice(),
            &[1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
            1e-6,
        );
    }

    #[test]
    fn embedding_refuses_an_out_of_range_index_by_name() {
        let w = weight();
        let idx = indices(&[3], &[1]);
        let err = B::embedding(&idx, &w).unwrap_err();
        assert!(
            matches!(err, Error::Shape(_)),
            "an out-of-range index must fail closed, got {err:?}"
        );
    }

    #[test]
    fn embedding_records_exactly_one_entry_and_leaves_indices_off_tape() {
        let w = weight();
        let idx = indices(&[0, 1], &[2]);
        let (_, recorded) = forward_recording(|| B::embedding(&idx, &w).unwrap());
        assert_eq!(
            recorded, 1,
            "one TapeEntry per forward, indices off the tape"
        );
    }

    // ── gather ─────────────────────────────────────────────────────────────

    #[test]
    fn gather_selects_along_the_named_axis() {
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // GatherElements: index matches source rank; out[r, c] = in[r, idx[r, c]].
        let idx = indices(&[2, 0, 0, 1], &[2, 2]);
        let out = B::gather(&input, 1, &idx).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_close(&read(&out), &[3.0, 1.0, 4.0, 5.0], 1e-6);

        let idx0 = indices(&[1, 1, 1], &[1, 3]);
        let out0 = B::gather(&input, 0, &idx0).unwrap();
        assert_eq!(out0.shape(), &[1, 3]);
        assert_close(&read(&out0), &[4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn gather_backward_scatter_adds_across_repeats() {
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // Each row picks columns [0, 0, 1]: col 0 receives two cotangents,
        // col 1 one, col 2 none — accumulated, never overwritten.
        let idx = indices(&[0, 0, 1, 0, 0, 1], &[2, 3]);
        let (_, grads) = recorded(|| B::gather(&input, 1, &idx).unwrap());
        assert_close(
            read(grads.get(input.id()).expect("gather records an input grad")).as_slice(),
            &[2.0, 1.0, 0.0, 2.0, 1.0, 0.0],
            1e-6,
        );
    }

    // ── index_select ───────────────────────────────────────────────────────

    #[test]
    fn index_select_replaces_the_axis_extent() {
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let idx = indices(&[2, 0, 2, 1], &[4]);
        let out = B::index_select(&input, 0, &idx).unwrap();
        assert_eq!(out.shape(), &[4, 2]);
        assert_close(&read(&out), &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn index_select_backward_accumulates_repeated_selections() {
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let idx = indices(&[0, 0], &[2]);
        let (_, grads) = recorded(|| B::index_select(&input, 0, &idx).unwrap());
        assert_close(
            read(grads.get(input.id()).expect("index_select records a grad")).as_slice(),
            &[2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            1e-6,
        );
    }

    // ── recording contract ─────────────────────────────────────────────────

    #[test]
    fn nograd_records_nothing() {
        let w = weight();
        let idx = indices(&[0, 1], &[2]);
        let input = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let sel = indices(&[1, 0], &[2]);
        let before = crate::metal::tape::depth();
        let _ = GradMode::Disabled.scope(|| B::embedding(&idx, &w).unwrap());
        let _ = GradMode::Disabled.scope(|| B::gather(&input, 0, &sel).unwrap());
        let _ = GradMode::Disabled.scope(|| B::index_select(&input, 0, &sel).unwrap());
        assert_eq!(
            crate::metal::tape::depth(),
            before,
            "NoGrad must record nothing"
        );
    }

    #[test]
    fn a_float_index_is_refused_by_name() {
        let w = weight();
        let bad = storage(&[0.0, 1.0], &[2]);
        let err = B::embedding(&bad, &w).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedDType { .. }),
            "a float index has no honest conversion to a row address, got {err:?}"
        );
    }
}
