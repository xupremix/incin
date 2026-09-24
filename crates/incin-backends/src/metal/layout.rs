//! Structural shape ops on Metal: `transpose`, `narrow`, `slice`, `concat`,
//! `stack`, `squeeze` and `unsqueeze` (#92's attention-block layout gaps).
//!
//! Host-side `Vec<f32>` walks over [`MetalStorage`]'s shared bytes, with the
//! same forward semantics and tape recipes WGPU's `wgpu/backend/shape_ops.rs`
//! uses (which in turn mirror CPU's `ops/shape_ops/*`): a window op scatters
//! its cotangent back into a zeroed operand, `concat` splits the cotangent
//! with `narrow`, and the two axis views rewrite into `reshape` so they
//! inherit its tape entry rather than pushing one of their own. The module
//! compiles and its tests run on any host under `--features metal`.

use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DType;

use super::backend::{MetalBackendImpl, storage_from_f32, transpose_metal};
use super::storage::MetalStorage;

/// Row-major element count of a dims slice, as `backend.rs` spells it.
fn numel(dims: &[usize]) -> Result<usize> {
    Ok(ShapeBuf::from_slice(dims).checked_numel(OperationKind::Storage)?)
}

/// Row-major contiguous strides for a host walk (same as `reduction.rs`).
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
    /// `transpose(t, dim1, dim2)`: materialize the operand with the two axes
    /// swapped, via [`transpose_metal`]'s host walk.
    ///
    /// Its own tape entry, not a view: a transpose is its own inverse, so
    /// backward reapplies the same swap to the cotangent — the identity CPU
    /// proves (`transpose_backward_reapplies_same_transpose`).
    pub(crate) fn transpose<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = transpose_metal(t, dim1, dim2)?;
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                Ok(vec![transpose_metal(grad_out, dim1, dim2)?])
            }),
        });
        Ok(out)
    }

    /// `narrow(t, axis, start, length)`: extract `length` entries from
    /// `axis` beginning at `start` — one outer/axis/inner window copy.
    ///
    /// Backward is a zeroed operand plus the same walk in reverse, pasting
    /// the cotangent into the window: the exact inverse of the forward
    /// slice, matching CPU's `narrow_storage` and WGPU's mode-1 paste.
    pub(crate) fn narrow<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
        start: usize,
        length: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        if axis >= dims.len() {
            return Err(Error::ShapeMismatch {
                op: "narrow",
                expected: dims.to_vec(),
                got: vec![axis],
                msg: "narrow axis out of bounds".to_string(),
            });
        }
        let dim = dims[axis];
        if start.saturating_add(length) > dim || length == 0 {
            return Err(Error::ShapeMismatch {
                op: "narrow",
                expected: vec![dim],
                got: vec![start, length],
                msg: "narrow window is empty or out of bounds".to_string(),
            });
        }

        let outer = numel(&dims[..axis])?;
        let inner = numel(&dims[axis + 1..])?;
        let in_numel = outer * dim * inner;
        let mut out_dims = dims.to_vec();
        out_dims[axis] = length;

        let bytes = t.as_bytes()?;
        let input: &[f32] = bytemuck::cast_slice(bytes);
        let mut out = vec![0.0f32; outer * length * inner];
        for o in 0..outer {
            for a in 0..length {
                for i in 0..inner {
                    out[o * length * inner + a * inner + i] =
                        input[(o * dim + start + a) * inner + i];
                }
            }
        }
        let out = storage_from_f32(&out, &out_dims, t)?;

        let in_dims = dims.to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let grad_bytes = grad_out.as_bytes()?;
                let grad: &[f32] = bytemuck::cast_slice(grad_bytes);
                let mut grad_input = vec![0.0f32; in_numel];
                for o in 0..outer {
                    for a in 0..length {
                        for i in 0..inner {
                            grad_input[(o * dim + start + a) * inner + i] =
                                grad[o * length * inner + a * inner + i];
                        }
                    }
                }
                Ok(vec![storage_from_f32(&grad_input, &in_dims, grad_out)?])
            }),
        });
        Ok(out)
    }

    /// `slice(t, ranges)`: the per-axis generalization of
    /// [`narrow`](Self::narrow), composed as one `narrow` per axis so each
    /// window pushes its own tape entry and backward is their chain of
    /// scatters — CPU's `slice_storage` composed the same way.
    ///
    /// Every range is validated before the first `narrow` runs, so a bad
    /// later axis cannot leave a half-sliced chain recorded on the tape.
    pub(crate) fn slice_exact<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        if ranges.len() != dims.len() {
            return Err(Error::ShapeMismatch {
                op: "slice_exact",
                expected: dims.to_vec(),
                got: ranges.iter().map(|r| r.1.saturating_sub(r.0)).collect(),
                msg: "slice ranks must match".to_string(),
            });
        }
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start > end || end > dims[i] || start == end {
                return Err(Error::ShapeMismatch {
                    op: "slice_exact",
                    expected: vec![dims[i]],
                    got: vec![start, end],
                    msg: "slice range invalid or empty".to_string(),
                });
            }
        }

        let mut current: <Self as StorageBackend>::Storage<K> = t.clone();
        for (i, &(start, end)) in ranges.iter().enumerate() {
            current = Self::narrow::<K>(&current, i, start, end - start)?;
        }
        Ok(current)
    }

    /// `concat(inputs, axis)`: one zero-filled output plus one window copy
    /// per operand at the running offset along `axis`. Matches CPU's
    /// `concat_storage` exactly (no `broadcast_as` promotion — operands must
    /// agree on every other axis).
    ///
    /// Backward splits the cotangent with one `narrow` per operand, in input
    /// order, mirroring WGPU's mode-1 split.
    pub(crate) fn concat_exact<K: DType>(
        inputs: &[&<Self as StorageBackend>::Storage<K>],
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let Some(&first) = inputs.first() else {
            return Err(Error::ShapeMismatch {
                op: "concat_exact",
                expected: vec![],
                got: vec![],
                msg: "concat expects at least one operand".to_string(),
            });
        };
        let dims = first.metadata().shape().dims();
        let rank = dims.len();
        if axis >= rank {
            return Err(Error::ShapeMismatch {
                op: "concat_exact",
                expected: dims.to_vec(),
                got: vec![axis],
                msg: "concat axis out of bounds".to_string(),
            });
        }
        let mut out_dims = dims.to_vec();
        out_dims[axis] = 0;
        for t in inputs {
            let t_dims = t.metadata().shape().dims();
            if t_dims.len() != rank {
                return Err(Error::ShapeMismatch {
                    op: "concat_exact",
                    expected: dims.to_vec(),
                    got: t_dims.to_vec(),
                    msg: "concat operands must share rank".to_string(),
                });
            }
            for d in 0..rank {
                if d != axis && t_dims[d] != dims[d] {
                    return Err(Error::ShapeMismatch {
                        op: "concat_exact",
                        expected: dims.to_vec(),
                        got: t_dims.to_vec(),
                        msg: "concat operands must agree off the concat axis".to_string(),
                    });
                }
            }
            out_dims[axis] += t_dims[axis];
        }

        let outer = numel(&dims[..axis])?;
        let inner = numel(&dims[axis + 1..])?;
        let total_axis = out_dims[axis];
        let mut out = vec![0.0f32; outer * total_axis * inner];
        let mut offsets = Vec::with_capacity(inputs.len());
        let mut lengths = Vec::with_capacity(inputs.len());
        let mut offset = 0usize;
        for t in inputs {
            let t_dims = t.metadata().shape().dims();
            let t_axis = t_dims[axis];
            let bytes = t.as_bytes()?;
            let operand: &[f32] = bytemuck::cast_slice(bytes);
            for o in 0..outer {
                for a in 0..t_axis {
                    for i in 0..inner {
                        out[(o * total_axis + offset + a) * inner + i] =
                            operand[(o * t_axis + a) * inner + i];
                    }
                }
            }
            offsets.push(offset);
            lengths.push(t_axis);
            offset += t_axis;
        }
        let out = storage_from_f32(&out, &out_dims, first)?;

        let input_ids = inputs.iter().map(|t| t.id()).collect::<Vec<_>>();
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out.id(),
            input_ids,
            backward: Box::new(move |grad_out: &MetalStorage| {
                let mut grads = Vec::with_capacity(lengths.len());
                for (offset, length) in offsets.iter().zip(lengths.iter()) {
                    grads.push(Self::narrow::<K>(grad_out, axis, *offset, *length)?);
                }
                Ok(grads)
            }),
        });
        Ok(out)
    }

    /// `stack(inputs, axis)`: insert a new axis at `axis`, each operand
    /// unsqueezed to that rank, then [`concat_exact`](Self::concat_exact).
    /// Mirrors CPU's and WGPU's `stack` = unsqueeze + concat, so each
    /// intermediate carries `reshape`'s tape entry and backward unwinds
    /// through them into the original operands.
    pub(crate) fn stack_exact<K: DType>(
        inputs: &[&<Self as StorageBackend>::Storage<K>],
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let Some(&first) = inputs.first() else {
            return Err(Error::ShapeMismatch {
                op: "stack_exact",
                expected: vec![],
                got: vec![],
                msg: "stack expects at least one operand".to_string(),
            });
        };
        let first_dims = first.metadata().shape().dims();
        if axis > first_dims.len() {
            return Err(Error::ShapeMismatch {
                op: "stack_exact",
                expected: first_dims.to_vec(),
                got: vec![axis],
                msg: "stack axis out of bounds".to_string(),
            });
        }
        let mut projected: Vec<<Self as StorageBackend>::Storage<K>> =
            Vec::with_capacity(inputs.len());
        for t in inputs {
            projected.push(Self::unsqueeze::<K>(t, axis)?);
        }
        let refs: Vec<&_> = projected.iter().collect();
        Self::concat_exact::<K>(&refs, axis)
    }

    /// Drop an axis of extent 1 — `reshape` with the axis removed, so it
    /// inherits `reshape`'s tape entry instead of pushing one of its own.
    ///
    /// Refusing a non-unit axis is the point of the check: silently keeping
    /// an axis the caller asked to remove would hand back a tensor of a
    /// different rank than the one they wrote down (WGPU's message, verbatim
    /// in intent).
    pub(crate) fn squeeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        if dim >= dims.len() || dims[dim] != 1 {
            return Err(Error::ShapeMismatch {
                op: "squeeze",
                expected: vec![1],
                got: dims.to_vec(),
                msg: format!(
                    "squeeze requires axis {dim} to have size 1, got size {} in shape {:?}",
                    dims.get(dim).copied().unwrap_or(0),
                    dims
                ),
            });
        }
        let mut target = dims.to_vec();
        target.remove(dim);
        Self::reshape::<K>(t, &target)
    }

    /// Insert an axis of extent 1 at `dim` — the inverse of
    /// [`squeeze`](Self::squeeze), and a `reshape` view for the same reason.
    /// `dim == rank` appends rather than failing, matching CPU, CUDA and
    /// WGPU; beyond that the axis still lands at the end (a caller naming a
    /// farther axis gets the nearest legal placement, as on the other
    /// backends, rather than an error the descriptor would already have
    /// raised).
    pub(crate) fn unsqueeze<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mut target = t.metadata().shape().dims().to_vec();
        if dim <= target.len() {
            target.insert(dim, 1);
        } else {
            target.push(1);
        }
        Self::reshape::<K>(t, &target)
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
    /// diagonal of a rank 1 or 2 operand — CPU's `triangular_storage` loop,
    /// host-side over the shared bytes. Backward reapplies the same keep rule
    /// to the cotangent, which is the identity CPU proves ("zeroing is its
    /// own transpose").
    fn triangular<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        offset: i64,
        upper: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let dims = t.metadata().shape().dims();
        let rank = dims.len();
        if rank == 0 || rank > 2 {
            return Err(Error::ShapeMismatch {
                op: if upper { "triu" } else { "tril" },
                expected: vec![1, 2],
                got: vec![rank],
                msg: "tril/triu accept rank 1 or 2".to_string(),
            });
        }
        let bytes = t.as_bytes()?;
        let input: &[f32] = bytemuck::cast_slice(bytes);
        let mut masked = vec![0.0f32; input.len()];
        if rank == 1 {
            for (col, &value) in input.iter().enumerate() {
                if Self::triangular_keep(0, col as i64, offset, upper) {
                    masked[col] = value;
                }
            }
        } else {
            let cols = dims[1];
            for (idx, &value) in input.iter().enumerate() {
                let row = (idx / cols) as i64;
                let col = (idx % cols) as i64;
                if Self::triangular_keep(row, col, offset, upper) {
                    masked[idx] = value;
                }
            }
        }
        let out = storage_from_f32(&masked, dims, t)?;

        let in_dims = dims.to_vec();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let grad_bytes = grad_out.as_bytes()?;
                let grad: &[f32] = bytemuck::cast_slice(grad_bytes);
                let mut g = grad.to_vec();
                if rank == 1 {
                    for (col, item) in g.iter_mut().enumerate() {
                        if !Self::triangular_keep(0, col as i64, offset, upper) {
                            *item = 0.0;
                        }
                    }
                } else {
                    let cols = in_dims[1];
                    for (idx, item) in g.iter_mut().enumerate() {
                        let row = (idx / cols) as i64;
                        let col = (idx % cols) as i64;
                        if !Self::triangular_keep(row, col, offset, upper) {
                            *item = 0.0;
                        }
                    }
                }
                Ok(vec![storage_from_f32(&g, &in_dims, grad_out)?])
            }),
        });
        Ok(out)
    }

    /// `tril(t, offset)`: keep the lower triangle at or below the `offset`-th
    /// diagonal.
    pub(crate) fn tril<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        offset: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::triangular::<K>(t, offset, false)
    }

    /// `triu(t, offset)`: keep the upper triangle at or above the `offset`-th
    /// diagonal.
    pub(crate) fn triu<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        offset: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Self::triangular::<K>(t, offset, true)
    }

    /// `pad(t, padding, value)`: grow each axis by `(before, after)`, filling
    /// the exterior with the constant.
    ///
    /// Forward walks every output coordinate: in-window positions read the
    /// operand at `coordinate - before`, exterior positions take `value` —
    /// CPU's `pad_storage` walk for walk. Backward is the inverse window
    /// extract: each input coordinate's cotangent sits at itself shifted by
    /// the per-axis `before` padding, the same recipe CPU's reverse walk uses.
    pub(crate) fn pad<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        padding: &[(usize, usize)],
        value: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_dims = t.shape().to_vec();
        if padding.len() != in_dims.len() {
            return Err(Error::ShapeMismatch {
                op: "pad",
                expected: in_dims.clone(),
                got: vec![padding.len()],
                msg: "pad needs one (before, after) pair per axis".to_string(),
            });
        }
        let out_dims: Vec<usize> = in_dims
            .iter()
            .zip(padding.iter())
            .map(|(size, &(before, after))| size + before + after)
            .collect();
        let total = numel(&out_dims)?;
        let in_strides = host_strides(&in_dims);
        let bytes = t.as_bytes()?;
        let data: &[f32] = bytemuck::cast_slice(bytes);
        let fill = value as f32;

        let mut out_vals = vec![fill; total];
        let mut out_idx = vec![0usize; out_dims.len()];
        for _ in 0..total {
            let mut inside = true;
            let mut src_idx = Vec::with_capacity(out_idx.len());
            for (axis, &position) in out_idx.iter().enumerate() {
                let (before, _) = padding[axis];
                if position < before || position >= before + in_dims[axis] {
                    inside = false;
                    break;
                }
                src_idx.push(position - before);
            }
            if inside {
                let flat: usize = src_idx
                    .iter()
                    .zip(in_strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum();
                let out_flat: usize = out_idx
                    .iter()
                    .zip(host_strides(&out_dims).iter())
                    .map(|(&i, &s)| i * s)
                    .sum();
                out_vals[out_flat] = data[flat];
            }
            if !out_dims.is_empty() {
                host_increment(&mut out_idx, &out_dims);
            }
        }
        let out = storage_from_f32(&out_vals, &out_dims, t)?;
        let offsets: Vec<usize> = padding.iter().map(|&(before, _)| before).collect();
        let out_strides = host_strides(&out_dims);
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let grad_bytes = grad_out.as_bytes()?;
                let grad: &[f32] = bytemuck::cast_slice(grad_bytes);
                let total = numel(&in_dims)?;
                let mut grad_input = vec![0.0f32; total];
                let mut idx = vec![0usize; in_dims.len()];
                for _ in 0..total {
                    let mut out_idx = Vec::with_capacity(in_dims.len());
                    for (axis, &coordinate) in idx.iter().enumerate() {
                        out_idx.push(coordinate + offsets[axis]);
                    }
                    let out_flat: usize = out_idx
                        .iter()
                        .zip(out_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    let flat: usize = idx
                        .iter()
                        .zip(in_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    grad_input[flat] = grad[out_flat];
                    if !in_dims.is_empty() {
                        host_increment(&mut idx, &in_dims);
                    }
                }
                Ok(vec![storage_from_f32(&grad_input, &in_dims, grad_out)?])
            }),
        });
        Ok(out)
    }

    /// `repeat(t, repeats)`: tile each axis by its factor —
    /// `out[coords] = t[coords % shape]`, CPU's `repeat_storage` walk.
    ///
    /// Backward is the modulo-block sum: every tile's cotangent adds onto its
    /// source element, the exact inverse of the forward tiling.
    pub(crate) fn repeat<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        repeats: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_dims = t.shape().to_vec();
        if repeats.len() != in_dims.len() {
            return Err(Error::ShapeMismatch {
                op: "repeat",
                expected: in_dims.clone(),
                got: vec![repeats.len()],
                msg: "repeat factors must match tensor rank".to_string(),
            });
        }
        let out_dims: Vec<usize> = in_dims
            .iter()
            .zip(repeats.iter())
            .map(|(size, &rep)| size * rep)
            .collect();
        let total = numel(&out_dims)?;
        let in_strides = host_strides(&in_dims);
        let out_strides = host_strides(&out_dims);
        let bytes = t.as_bytes()?;
        let data: &[f32] = bytemuck::cast_slice(bytes);

        let mut out_vals = Vec::with_capacity(total);
        let mut out_idx = vec![0usize; out_dims.len()];
        for _ in 0..total {
            let src_idx: Vec<usize> = out_idx
                .iter()
                .enumerate()
                .map(|(axis, &value)| value % in_dims[axis])
                .collect();
            let flat: usize = src_idx
                .iter()
                .zip(in_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            out_vals.push(data[flat]);
            if !out_dims.is_empty() {
                host_increment(&mut out_idx, &out_dims);
            }
        }
        let out = storage_from_f32(&out_vals, &out_dims, t)?;

        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let grad_bytes = grad_out.as_bytes()?;
                let grad: &[f32] = bytemuck::cast_slice(grad_bytes);
                let total = numel(&in_dims)?;
                let mut grads = vec![0.0f32; total];
                let mut grad_idx = vec![0usize; out_dims.len()];
                for _ in 0..numel(&out_dims)? {
                    let flat_src: usize = grad_idx
                        .iter()
                        .enumerate()
                        .map(|(axis, &value)| (value % in_dims[axis]) * in_strides[axis])
                        .sum();
                    let flat_g: usize = grad_idx
                        .iter()
                        .zip(out_strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    grads[flat_src] += grad[flat_g];
                    if !out_dims.is_empty() {
                        host_increment(&mut grad_idx, &out_dims);
                    }
                }
                Ok(vec![storage_from_f32(&grads, &in_dims, grad_out)?])
            }),
        });
        Ok(out)
    }
}

#[cfg(test)]
/// Host-side forward/backward parity tests for the structural shape ops.
/// Pure `Vec<f32>` walks, so they run without a Metal device.
mod tests {
    use super::*;
    use incin_core::exec::GradMode;
    use incin_core::shapes::ShapeBuf;
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

    /// Run `f` under recording mode, walk back with a ones seed, return
    /// `(output, grads)`.
    fn recorded<F>(f: F) -> (MetalStorage, MetalGrads)
    where
        F: FnOnce() -> MetalStorage,
    {
        let out = GradMode::Enabled.scope(f);
        let grads = crate::metal::tape::backward(&out).expect("backward walk succeeds");
        (out, grads)
    }

    /// A 3x4 row-major field so narrow/slice windows are unmistakable.
    const FIELD: [f32; 12] = [
        0.0, 1.0, 2.0, 3.0, //
        10.0, 11.0, 12.0, 13.0, //
        20.0, 21.0, 22.0, 23.0,
    ];

    // ── transpose ──────────────────────────────────────────────────────────

    #[test]
    fn transpose_swaps_axes_and_backward_is_self_inverse() {
        let t = storage(&FIELD, &[3, 4]);
        let (out, grads) = recorded(|| B::transpose::<f32>(&t, 0, 1).unwrap());
        assert_eq!(out.shape(), &[4, 3], "transpose swaps the two axes");
        // Column-major read of the field: 0,10,20, 1,11,21, ...
        assert_close(
            &read(&out),
            &[
                0.0, 10.0, 20.0, 1.0, 11.0, 21.0, 2.0, 12.0, 22.0, 3.0, 13.0, 23.0,
            ],
            0.0,
        );
        // A ones seed on the transposed output transposes back to ones on
        // the input: the swap is an involution on gradients too.
        let grad = read(grads.get(t.id()).expect("transpose records an input grad"));
        assert_eq!(grad, vec![1.0; 12], "backward reapplies the same transpose");
    }

    #[test]
    fn transpose_twice_returns_the_original() {
        let t = storage(&FIELD, &[3, 4]);
        let once = B::transpose::<f32>(&t, 0, 1).unwrap();
        let twice = B::transpose::<f32>(&once, 0, 1).unwrap();
        assert_eq!(twice.shape(), t.shape());
        assert_eq!(read(&twice), read(&t));
    }

    #[test]
    fn transpose_refuses_out_of_bounds_axes() {
        let t = storage(&FIELD, &[3, 4]);
        let err = B::transpose::<f32>(&t, 0, 2).expect_err("axis 2 is out of bounds");
        let message = format!("{err}");
        assert!(
            message.contains("transpose"),
            "the error must name the operation: {message}"
        );
    }

    // ── narrow / slice ─────────────────────────────────────────────────────

    #[test]
    fn narrow_takes_the_requested_window_and_scatters_backward() {
        let t = storage(&FIELD, &[3, 4]);
        let (out, grads) = recorded(|| B::narrow::<f32>(&t, 1, 1, 2).unwrap());
        assert_eq!(out.shape(), &[3, 2]);
        assert_close(&read(&out), &[1.0, 2.0, 11.0, 12.0, 21.0, 22.0], 0.0);

        // Ones seed: the gradient is 1 inside the window, 0 everywhere else.
        let grad = read(grads.get(t.id()).expect("narrow records an input grad"));
        assert_eq!(
            grad,
            vec![
                0.0, 1.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 0.0,
            ],
            "backward must scatter the cotangent back to the window only"
        );
    }

    #[test]
    fn narrow_with_general_seed_lands_in_the_window() {
        let t = storage(&FIELD, &[3, 4]);
        let seed = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let out = GradMode::Enabled.scope(|| B::narrow::<f32>(&t, 1, 1, 2).unwrap());
        let grads =
            crate::metal::tape::backward_with(&out, &seed).expect("seeded backward walk succeeds");
        let grad = read(grads.get(t.id()).expect("narrow records an input grad"));
        assert_eq!(
            grad,
            vec![
                0.0, 1.0, 2.0, 0.0, //
                0.0, 3.0, 4.0, 0.0, //
                0.0, 5.0, 6.0, 0.0,
            ],
        );
    }

    #[test]
    fn narrow_refuses_empty_and_out_of_bounds_windows() {
        let t = storage(&FIELD, &[3, 4]);
        assert!(B::narrow::<f32>(&t, 4, 0, 1).is_err(), "axis out of bounds");
        assert!(
            B::narrow::<f32>(&t, 1, 3, 2).is_err(),
            "window past the end"
        );
        assert!(B::narrow::<f32>(&t, 1, 1, 0).is_err(), "empty window");
        assert!(
            B::narrow::<f32>(&t, 1, 2, 2).is_ok(),
            "exact boundary is legal"
        );
    }

    #[test]
    fn slice_is_per_axis_narrows_forward_and_backward() {
        let t = storage(&FIELD, &[3, 4]);
        let (out, grads) = recorded(|| B::slice_exact::<f32>(&t, &[(1, 3), (0, 2)]).unwrap());
        assert_eq!(out.shape(), &[2, 2]);
        assert_close(&read(&out), &[10.0, 11.0, 20.0, 21.0], 0.0);

        let grad = read(grads.get(t.id()).expect("slice records an input grad"));
        assert_eq!(
            grad,
            vec![
                0.0, 0.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, 0.0,
            ],
            "backward is the chain of window scatters"
        );
    }

    #[test]
    fn slice_refuses_rank_mismatch_and_invalid_ranges() {
        let t = storage(&FIELD, &[3, 4]);
        assert!(
            B::slice_exact::<f32>(&t, &[(0, 2)]).is_err(),
            "ranges must match rank"
        );
        assert!(
            B::slice_exact::<f32>(&t, &[(2, 1), (0, 2)]).is_err(),
            "start > end"
        );
        assert!(
            B::slice_exact::<f32>(&t, &[(1, 1), (0, 2)]).is_err(),
            "empty range"
        );
        assert!(
            B::slice_exact::<f32>(&t, &[(0, 5), (0, 2)]).is_err(),
            "range past the axis"
        );
        assert!(
            B::slice_exact::<f32>(&t, &[(1, 3), (1, 4)]).is_ok(),
            "a legal window must pass"
        );
    }

    // ── concat / stack ─────────────────────────────────────────────────────

    #[test]
    fn concat_joins_along_axis_zero_and_splits_backward() {
        let a = storage(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = storage(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let (out, grads) = recorded(|| B::concat_exact::<f32>(&[&a, &b], 0).unwrap());
        assert_eq!(out.shape(), &[4, 2]);
        assert_close(&read(&out), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0.0);
        // Ones seed on the join splits back into ones on each operand.
        assert_eq!(
            read(grads.get(a.id()).expect("concat records grad for input 0")),
            vec![1.0; 4],
        );
        assert_eq!(
            read(grads.get(b.id()).expect("concat records grad for input 1")),
            vec![1.0; 4],
        );
    }

    #[test]
    fn concat_joins_along_axis_one_without_reordering_rows() {
        let a = storage(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = storage(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let out = B::concat_exact::<f32>(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        assert_close(&read(&out), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0], 0.0);
    }

    #[test]
    fn concat_with_general_seed_splits_the_cotangent_per_operand() {
        let a = storage(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = storage(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let seed = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
        let out = GradMode::Enabled.scope(|| B::concat_exact::<f32>(&[&a, &b], 0).unwrap());
        let grads =
            crate::metal::tape::backward_with(&out, &seed).expect("seeded backward walk succeeds");
        assert_eq!(
            read(grads.get(a.id()).expect("concat records grad for input 0")),
            &[1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(
            read(grads.get(b.id()).expect("concat records grad for input 1")),
            &[5.0, 6.0, 7.0, 8.0],
        );
    }

    #[test]
    fn concat_validates_rank_axis_and_off_axis_agreement() {
        let a = storage(&[1.0, 2.0], &[2]);
        let b = storage(&[3.0, 4.0], &[1, 2]);
        assert!(
            B::concat_exact::<f32>(&[&a, &a], 0).is_ok(),
            "same-shape operands on axis 0 must join"
        );
        assert!(
            B::concat_exact::<f32>(&[&a, &b], 0).is_err(),
            "operands must share rank"
        );
        let c = storage(&[3.0, 4.0, 5.0], &[1, 3]);
        let d = storage(&[6.0, 7.0, 8.0], &[1, 3]);
        assert!(
            B::concat_exact::<f32>(&[&c, &d], 0).is_ok(),
            "matching off-axis extents must join"
        );
        let e = storage(&[3.0, 4.0], &[1, 2]);
        assert!(
            B::concat_exact::<f32>(&[&c, &e], 0).is_err(),
            "off-axis extents must agree"
        );
        assert!(
            B::concat_exact::<f32>(&[&c, &d], 2).is_err(),
            "axis out of bounds"
        );
        let none: [&MetalStorage; 0] = [];
        assert!(
            B::concat_exact::<f32>(&none, 0).is_err(),
            "at least one operand is required"
        );
    }

    #[test]
    fn stack_inserts_a_new_leading_axis() {
        let a = storage(&[1.0, 2.0], &[2]);
        let b = storage(&[3.0, 4.0], &[2]);
        let (out, grads) = recorded(|| B::stack_exact::<f32>(&[&a, &b], 0).unwrap());
        assert_eq!(out.shape(), &[2, 2], "stack axis 0 of two rank-1 operands");
        assert_close(&read(&out), &[1.0, 2.0, 3.0, 4.0], 0.0);
        // Unwind: concat splits along the new axis 0, each unsqueeze reshape
        // then hands the 2x2 slice back to its original rank-1 operand.
        assert_eq!(
            read(grads.get(a.id()).expect("stack records grad for input 0")),
            &[1.0, 1.0],
        );
        assert_eq!(
            read(grads.get(b.id()).expect("stack records grad for input 1")),
            &[1.0, 1.0],
        );
    }

    #[test]
    fn stack_validates_its_axis_and_arity() {
        let a = storage(&[1.0, 2.0], &[2]);
        assert!(B::stack_exact::<f32>(&[&a, &a], 2).is_err(), "axis > rank");
        assert!(
            B::stack_exact::<f32>(&[&a, &a], 1).is_ok(),
            "axis == rank inserts at the end"
        );
        let none: [&MetalStorage; 0] = [];
        assert!(B::stack_exact::<f32>(&none, 0).is_err(), "needs an operand");
    }

    // ── squeeze / unsqueeze ────────────────────────────────────────────────

    #[test]
    fn squeeze_drops_a_unit_axis_without_moving_data() {
        let t = storage(&[1.0, 2.0, 3.0], &[1, 3]);
        let (out, grads) = recorded(|| B::squeeze::<f32>(&t, 0).unwrap());
        assert_eq!(out.shape(), &[3]);
        assert_close(&read(&out), &[1.0, 2.0, 3.0], 0.0);
        // reshape's backward: the ones seed reshapes back to the input.
        assert_eq!(
            read(grads.get(t.id()).expect("squeeze records via reshape")),
            vec![1.0; 3],
            "the gradient restores the squeezed axis"
        );
    }

    #[test]
    fn squeeze_refuses_a_non_unit_axis() {
        let t = storage(&FIELD, &[3, 4]);
        assert!(B::squeeze::<f32>(&t, 0).is_err(), "extent 3 is not 1");
        assert!(B::squeeze::<f32>(&t, 4).is_err(), "axis out of bounds");
        let unit = storage(&[7.0], &[1]);
        assert!(B::squeeze::<f32>(&unit, 0).is_ok());
    }

    #[test]
    fn unsqueeze_then_squeeze_round_trips() {
        let t = storage(&[1.0, 2.0], &[2]);
        let widened = B::unsqueeze::<f32>(&t, 0).unwrap();
        assert_eq!(widened.shape(), &[1, 2]);
        let back = B::squeeze::<f32>(&widened, 0).unwrap();
        assert_eq!(back.shape(), t.shape());
        assert_eq!(read(&back), read(&t));

        // dim == rank appends (WGPU/CUDA semantics).
        let appended = B::unsqueeze::<f32>(&t, 2).unwrap();
        assert_eq!(appended.shape(), &[2, 1]);
    }

    // ── tril / triu ─────────────────────────────────────────────────────────

    #[test]
    fn tril_and_triu_mask_the_correct_half_and_backward_is_the_same_mask() {
        let t = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let (lower, grads) = recorded(|| B::tril::<f32>(&t, 0).unwrap());
        assert_close(
            &read(&lower),
            &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
            0.0,
        );
        // A ones seed masked by the same keep rule: the upper half is zeroed.
        assert_eq!(
            read(grads.get(t.id()).expect("tril records an input grad")),
            vec![
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, //
                1.0, 1.0, 1.0,
            ],
            "backward reapplies the same lower-triangle mask"
        );

        let (upper, _) = recorded(|| B::triu::<f32>(&t, 0).unwrap());
        assert_close(
            &read(&upper),
            &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0],
            0.0,
        );
    }

    #[test]
    fn triangular_offsets_shift_the_kept_diagonal_and_rank_one_is_a_row() {
        let t = storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        // offset 1 keeps `diag >= 1` above, `diag <= 1` below.
        let above = B::triu::<f32>(&t, 1).unwrap();
        assert_close(
            &read(&above),
            &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0],
            0.0,
        );
        let below = B::tril::<f32>(&t, 1).unwrap();
        assert_close(
            &read(&below),
            &[1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            0.0,
        );

        // Rank one is the first row of an implicit matrix (row 0, col = index).
        let row = storage(&[1.0, 2.0, 3.0], &[3]);
        assert_close(
            &read(&B::tril::<f32>(&row, 0).unwrap()),
            &[1.0, 0.0, 0.0],
            0.0,
        );
        assert_close(
            &read(&B::triu::<f32>(&row, 0).unwrap()),
            &[1.0, 2.0, 3.0],
            0.0,
        );
    }

    #[test]
    fn triangular_refuses_rank_outside_the_descriptor_contract() {
        let scalar = storage(&[7.0], &[]);
        assert!(B::tril::<f32>(&scalar, 0).is_err(), "rank 0 is refused");
        let cube = storage(&[1.0; 8], &[2, 2, 2]);
        assert!(B::triu::<f32>(&cube, 0).is_err(), "rank 3 is refused");
    }

    // ── Recording contract ─────────────────────────────────────────────────

    #[test]
    fn nograd_records_nothing() {
        let t = storage(&FIELD, &[3, 4]);
        let a = storage(&[1.0, 2.0], &[2]);
        let unit = storage(&[7.0], &[1]);
        let before = crate::metal::tape::depth();
        let _ = GradMode::Disabled.scope(|| B::transpose::<f32>(&t, 0, 1).unwrap());
        let _ = GradMode::Disabled.scope(|| B::narrow::<f32>(&t, 1, 1, 2).unwrap());
        let _ = GradMode::Disabled.scope(|| B::slice_exact::<f32>(&t, &[(0, 2), (0, 2)]).unwrap());
        let _ = GradMode::Disabled.scope(|| B::concat_exact::<f32>(&[&a, &a], 0).unwrap());
        let _ = GradMode::Disabled.scope(|| B::stack_exact::<f32>(&[&a, &a], 0).unwrap());
        let _ = GradMode::Disabled.scope(|| B::squeeze::<f32>(&unit, 0).unwrap());
        let _ = GradMode::Disabled.scope(|| B::unsqueeze::<f32>(&a, 0).unwrap());
        let _ = GradMode::Disabled.scope(|| B::tril::<f32>(&t, 0).unwrap());
        let _ = GradMode::Disabled.scope(|| B::triu::<f32>(&t, 0).unwrap());
        assert_eq!(
            crate::metal::tape::depth(),
            before,
            "NoGrad must record nothing"
        );
    }
}
