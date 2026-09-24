//! Variance, std, norm and cumsum reductions on Metal (#92's reduction gaps).
//!
//! Every method here is a composition of already tape-tracked Metal
//! primitives — the same recipes CPU's `variance_executors!` /
//! `cpu/ops/reduce` and WGPU's `backend/reduce.rs` use — so the backward is
//! the replay of those entries rather than hand-derived math. The one
//! hand-written recipe is `cumsum`'s suffix-sum backward (the Jacobian is
//! lower-triangular ones), copied walk-for-walk from CPU's
//! `reverse_cumsum`. The module compiles and its tests run on any host
//! under `--features metal`.

use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::{DType, DTypeId};

use super::backend::{MetalBackendImpl, storage_from_f32};
use super::storage::MetalStorage;

/// Row-major element count of a dims slice, as `backend.rs` spells it.
fn numel(dims: &[usize]) -> Result<usize> {
    Ok(ShapeBuf::from_slice(dims).checked_numel(OperationKind::Storage)?)
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
    /// Bessel (or population) scale for a variance over `count` samples:
    /// `1 / (count - 1)` when `unbiased` and `count > 1`, else `1 / count`,
    /// and `0` when the divisor would be non-positive. Same table CPU's and
    /// WGPU's `variance_scale` uses.
    fn variance_scale(count: usize, unbiased: bool) -> f64 {
        let count = count as f64;
        let divisor = if unbiased {
            if count <= 1.0 { 0.0 } else { count - 1.0 }
        } else {
            count
        };
        if divisor > 0.0 { 1.0 / divisor } else { 0.0 }
    }

    /// Scale a pre-reduced squared-deviation sum by
    /// [`variance_scale`](Self::variance_scale), optionally square-rooting.
    /// The mean/center/square/reduce steps live in each wrapper so all six
    /// stay readable against CPU's `variance_executors!` line by line.
    fn variance_scale_sum<K: DType>(
        summed: &<Self as StorageBackend>::Storage<K>,
        count: usize,
        unbiased: bool,
        square_root: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let scaled = Self::mul_scalar_float::<K>(summed, Self::variance_scale(count, unbiased))?;
        if square_root {
            Self::sqrt::<K>(&scaled)
        } else {
            Ok(scaled)
        }
    }

    /// `var_all`: population/Bessel variance over every element.
    pub(crate) fn variance_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_all::<K>(t)?;
        let count = numel(t.shape())?;
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_all::<K>(&squared)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, false)
    }

    /// `std_all`: [`variance_all`](Self::variance_all) then a square root.
    pub(crate) fn std_all<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_all::<K>(t)?;
        let count = numel(t.shape())?;
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_all::<K>(&squared)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, true)
    }

    /// `var_dim`: variance along `dim`, dropping the axis.
    pub(crate) fn variance_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_keepdim::<K>(t, dim)?;
        let count = t.shape().get(dim).copied().unwrap_or(0);
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_dim::<K>(&squared, dim)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, false)
    }

    /// `std_dim`: [`variance_dim`](Self::variance_dim) then a square root.
    pub(crate) fn std_dim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_keepdim::<K>(t, dim)?;
        let count = t.shape().get(dim).copied().unwrap_or(0);
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_dim::<K>(&squared, dim)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, true)
    }

    /// `var_keepdim`: variance along `dim`, keeping a unit axis.
    pub(crate) fn variance_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_keepdim::<K>(t, dim)?;
        let count = t.shape().get(dim).copied().unwrap_or(0);
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_keepdim::<K>(&squared, dim)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, false)
    }

    /// `std_keepdim`: [`variance_keepdim`](Self::variance_keepdim) then a
    /// square root.
    pub(crate) fn std_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        unbiased: bool,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let mean = Self::mean_keepdim::<K>(t, dim)?;
        let count = t.shape().get(dim).copied().unwrap_or(0);
        let centered = Self::sub::<K>(t, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let summed = Self::sum_keepdim::<K>(&squared, dim)?;
        Self::variance_scale_sum::<K>(&summed, count, unbiased, true)
    }

    /// `norm(order)`: the p-norm over every element. Order 1 and 2 take the
    /// dedicated abs-sum / Euclidean paths CPU takes (same tolerance-free
    /// special cases); any other positive order is `sum(|x|^p)^(1/p)`.
    pub(crate) fn norm<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        order: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        const NORM_ORDER_TOLERANCE: f64 = 1e-6;
        if (order - 1.0).abs() < NORM_ORDER_TOLERANCE {
            let magnitude = Self::abs::<K>(t)?;
            return Self::sum_all::<K>(&magnitude);
        }
        if (order - 2.0).abs() < NORM_ORDER_TOLERANCE {
            let squared = Self::mul::<K>(t, t)?;
            let summed = Self::sum_all::<K>(&squared)?;
            return Self::sqrt::<K>(&summed);
        }
        let magnitude = Self::abs::<K>(t)?;
        let raised = Self::powf::<K>(&magnitude, order)?;
        let summed = Self::sum_all::<K>(&raised)?;
        Self::powf::<K>(&summed, 1.0 / order)
    }

    /// `cumsum(x, dim)`: host-side prefix scan along `dim`, mirrored from
    /// `cpu::ops::reduce::cumsum`. The Jacobian is lower-triangular ones, so
    /// the backward is the suffix sum (`reverse_cumsum`) of the cotangent —
    /// every input position receives `sum_{k >= d} grad_out[k]`.
    pub(crate) fn cumsum<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if dim >= t.shape().len() {
            return Err(Error::ShapeMismatch {
                op: "cumsum",
                expected: t.shape().to_vec(),
                got: vec![dim],
                msg: format!("cumsum: axis {dim} out of range for shape {:?}", t.shape()),
            });
        }
        let shape = t.shape().to_vec();
        let total = numel(&shape)?;
        let dim_len = shape[dim];
        let strides = host_strides(&shape);
        let bytes = t.as_bytes()?;
        let data: &[f32] = bytemuck::cast_slice(bytes);
        let mut out_data = vec![0.0f32; total];
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..total {
            if idx[dim] == 0 {
                let mut current = 0.0f32;
                for step in 0..dim_len {
                    let mut step_idx = idx.clone();
                    step_idx[dim] = step;
                    let flat: usize = step_idx
                        .iter()
                        .zip(strides.iter())
                        .map(|(&i, &s)| i * s)
                        .sum();
                    current += data[flat];
                    out_data[flat] = current;
                }
            }
            host_increment(&mut idx, &shape);
        }
        let out = storage_from_f32(&out_data, &shape, t)?;

        let t_like = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let grad_bytes = grad_out.as_bytes()?;
                let grad_data: &[f32] = bytemuck::cast_slice(grad_bytes);
                let mut rev = vec![0.0f32; total];
                let mut idx = vec![0usize; shape.len()];
                for _ in 0..total {
                    if idx[dim] == 0 {
                        let mut current = 0.0f32;
                        for step in (0..dim_len).rev() {
                            let mut step_idx = idx.clone();
                            step_idx[dim] = step;
                            let flat: usize = step_idx
                                .iter()
                                .zip(strides.iter())
                                .map(|(&i, &s)| i * s)
                                .sum();
                            current += grad_data[flat];
                            rev[flat] = current;
                        }
                    }
                    host_increment(&mut idx, &shape);
                }
                Ok(vec![storage_from_f32(&rev, &shape, &t_like)?])
            }),
        });
        Ok(out)
    }

    /// Pack `i64` values into `MetalStorage` with an `I64` dtype tag,
    /// inheriting `like`'s mode/device/alignment — the typed-output
    /// counterpart of `storage_from_f32` for index results.
    fn index_storage(values: &[i64], shape: &[usize], like: &MetalStorage) -> Result<MetalStorage> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        super::backend::storage_from_raw(bytes, shape, DTypeId::I64.descriptor(), like)
    }

    /// Shared axis-slice sort for `Sort` and `TopK`.
    ///
    /// Enumerates the `n_slices` independent 1-D slices along `dim` (the
    /// `base_shape` with `dim` set to 1, unflattened — CPU's `topk` walk),
    /// stable-sorts each slice's `(value, index)` pairs (stable sort preserves
    /// tie order, matching CPU's `sort_by` on equal keys), optionally truncates
    /// to `k`, and returns the flat value/index vectors plus the output shape.
    fn sort_pairs(
        t: &MetalStorage,
        dim: usize,
        descending: bool,
        k: Option<usize>,
    ) -> Result<(Vec<f32>, Vec<i64>, Vec<usize>)> {
        let shape = t.shape().to_vec();
        if dim >= shape.len() {
            return Err(Error::ShapeMismatch {
                op: "sort_pairs",
                expected: shape.clone(),
                got: vec![dim],
                msg: format!("axis {dim} out of range for shape {shape:?}"),
            });
        }
        let mut base_shape = shape.clone();
        base_shape[dim] = 1;
        let n_slices = numel(&base_shape)?;
        let k = k.map(|k| k.min(shape[dim]));
        let mut out_shape = shape.clone();
        if let Some(k) = k {
            out_shape[dim] = k;
        }
        let out_len = numel(&out_shape)?;
        let mut out_vals = vec![0.0f32; out_len];
        let mut out_indices = vec![0i64; out_len];

        let bytes = t.as_bytes()?;
        let data: &[f32] = bytemuck::cast_slice(bytes);
        let strides = host_strides(&shape);
        let out_strides = host_strides(&out_shape);

        for i in 0..n_slices {
            let mut rem = i;
            let mut coords = vec![0usize; shape.len()];
            for dd in (0..shape.len()).rev() {
                coords[dd] = rem % base_shape[dd];
                rem /= base_shape[dd];
            }
            let mut slice: Vec<(f32, i64)> = Vec::with_capacity(shape[dim]);
            for j in 0..shape[dim] {
                coords[dim] = j;
                let flat: usize = coords
                    .iter()
                    .zip(strides.iter())
                    .map(|(&c, &s)| c * s)
                    .sum();
                slice.push((data[flat], j as i64));
            }
            if descending {
                slice.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
            } else {
                slice.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            }
            let take = k.unwrap_or(shape[dim]);
            let mut out_coords = coords.clone();
            for (j, &(val, idx)) in slice.iter().enumerate().take(take) {
                out_coords[dim] = j;
                let flat: usize = out_coords
                    .iter()
                    .zip(out_strides.iter())
                    .map(|(&c, &s)| c * s)
                    .sum();
                out_vals[flat] = val;
                out_indices[flat] = idx;
            }
        }
        Ok((out_vals, out_indices, out_shape))
    }

    /// `argmax(t, dim)`: index of the maximum along `dim` (axis removed, as
    /// `max_dim` squeezes) or, for `None`, the flat index of the global max
    /// as a scalar — CPU's `argmax`, forward-only (no tape, matching CPU's
    /// deliberate non-push and `descriptor_training(ArgMax) = false`).
    /// Indices are physically `i64`.
    pub(crate) fn argmax(input: &MetalStorage, dim: Option<usize>) -> Result<MetalStorage> {
        let shape = input.shape().to_vec();
        let total = numel(&shape)?;
        let bytes = input.as_bytes()?;
        let data: &[f32] = bytemuck::cast_slice(bytes);
        match dim {
            Some(d) => {
                if d >= shape.len() {
                    return Err(Error::ShapeMismatch {
                        op: "argmax",
                        expected: shape.clone(),
                        got: vec![d],
                        msg: format!("axis {d} out of range for shape {shape:?}"),
                    });
                }
                let mut base_shape = shape.clone();
                base_shape[d] = 1;
                let n_slices = numel(&base_shape)?;
                let strides = host_strides(&shape);
                let mut keepdim_shape = shape.clone();
                keepdim_shape[d] = 1;
                let mut winners = vec![0i64; n_slices];
                for (i, slot) in winners.iter_mut().enumerate() {
                    let mut rem = i;
                    let mut coords = vec![0usize; shape.len()];
                    for dd in (0..shape.len()).rev() {
                        coords[dd] = rem % base_shape[dd];
                        rem /= base_shape[dd];
                    }
                    let mut best_val = f32::NEG_INFINITY;
                    let mut best_axis = 0usize;
                    for j in 0..shape[d] {
                        coords[d] = j;
                        let flat: usize = coords
                            .iter()
                            .zip(strides.iter())
                            .map(|(&c, &s)| c * s)
                            .sum();
                        if data[flat] > best_val {
                            best_val = data[flat];
                            best_axis = j;
                        }
                    }
                    *slot = best_axis as i64;
                }
                let keepdim = Self::index_storage(&winners, &keepdim_shape, input)?;
                // Squeeze the unit axis the same way CPU does: reshape to the
                // axis-removed shape (a pure rewrap of the same bytes).
                let mut squeeze_shape = keepdim_shape.clone();
                squeeze_shape.remove(d);
                super::backend::reshape_metal(&keepdim, &squeeze_shape)
            }
            None => {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_flat = 0i64;
                for (flat, &v) in data.iter().enumerate().take(total) {
                    if v > best_val {
                        best_val = v;
                        best_flat = flat as i64;
                    }
                }
                Self::index_storage(&[best_flat], &[], input)
            }
        }
    }

    /// `sort(t, axis, descending)`: sorted values beside the permutation that
    /// produced them, both with the operand's own geometry — `topk` with
    /// `k = axis length`, exactly the pair CUDA's `Sort` returns. Forward-only
    /// (`descriptor_training(Sort) = false`); indices are physically `i64`.
    pub(crate) fn sort(
        input: &MetalStorage,
        axis: usize,
        descending: bool,
    ) -> Result<(MetalStorage, MetalStorage)> {
        let dim_len = *input
            .shape()
            .get(axis)
            .ok_or_else(|| Error::Msg(format!("sort: axis {axis} outside the operand's rank")))?;
        let (vals, idxs, out_shape) = Self::sort_pairs(input, axis, descending, Some(dim_len))?;
        let values = storage_from_f32(&vals, &out_shape, input)?;
        let indices = Self::index_storage(&idxs, &out_shape, input)?;
        Ok((values, indices))
    }

    /// `topk(t, k, axis, largest)`: the `k` extreme values along `axis`
    /// beside their indices; the axis shrinks to `k`. Forward-only; indices
    /// are physically `i64` (the requested `index_dtype` is what admission
    /// checked on inputs, and no Metal row post-checks its output tag —
    /// same convention as CUDA's `TopK`/`Sort`).
    pub(crate) fn topk(
        input: &MetalStorage,
        k: usize,
        axis: usize,
        largest: bool,
    ) -> Result<(MetalStorage, MetalStorage)> {
        if k == 0 {
            return Err(Error::Msg("topk: k must be at least one".into()));
        }
        let (vals, idxs, out_shape) = Self::sort_pairs(input, axis, largest, Some(k))?;
        let values = storage_from_f32(&vals, &out_shape, input)?;
        let indices = Self::index_storage(&idxs, &out_shape, input)?;
        Ok((values, indices))
    }
}

#[cfg(test)]
/// Host-side forward/backward parity tests for the reduction family.
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

    /// 2×3 input with a non-zero mean so centering is exercised.
    const VALUES: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    fn sample() -> MetalStorage {
        storage(&VALUES, &[2, 3])
    }

    /// Host reference population / Bessel variance over every element.
    fn var_all_reference(values: &[f32], unbiased: bool) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let ss: f64 = values
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum();
        if unbiased {
            if values.len() > 1 {
                ss / (n - 1.0)
            } else {
                0.0
            }
        } else {
            ss / n
        }
    }

    // ── variance / std ─────────────────────────────────────────────────────

    #[test]
    fn variance_all_matches_population_and_bessel() {
        let t = sample();
        let pop = read(&B::variance_all::<f32>(&t, false).unwrap())[0];
        let bessel = read(&B::variance_all::<f32>(&t, true).unwrap())[0];
        assert!(
            (f64::from(pop) - var_all_reference(&VALUES, false)).abs() < 1e-5,
            "population var: got {pop}"
        );
        assert!(
            (f64::from(bessel) - var_all_reference(&VALUES, true)).abs() < 1e-5,
            "bessel var: got {bessel}"
        );
        assert!(bessel > pop, "bessel scale must exceed population");
    }

    #[test]
    fn std_all_is_the_square_root_of_variance() {
        let t = sample();
        let var = read(&B::variance_all::<f32>(&t, false).unwrap())[0];
        let std = read(&B::std_all::<f32>(&t, false).unwrap())[0];
        assert!(
            (std - var.sqrt()).abs() < 1e-5,
            "std={std}, sqrt(var)={}",
            var.sqrt()
        );
    }

    #[test]
    fn variance_unbiased_on_a_singleton_axis_is_zero() {
        // count = 1, unbiased → divisor 0 → scale 0 → variance 0 (CPU's table).
        let t = storage(&[3.0], &[1]);
        let v = read(&B::variance_all::<f32>(&t, true).unwrap())[0];
        assert_eq!(v, 0.0, "unbiased variance of one sample must be 0");
    }

    #[test]
    fn variance_dim_drops_the_axis_and_matches_a_hand_reference() {
        let t = sample(); // [[1,2,3],[4,5,6]]
        // Axis 1 population var per row: var([1,2,3]) = var([4,5,6]) = 2/3.
        let got = read(&B::variance_dim::<f32>(&t, 1, false).unwrap());
        assert_eq!(got.len(), 2, "dim reduce drops the axis");
        assert_close(&got, &[2.0 / 3.0, 2.0 / 3.0], 1e-5);
    }

    #[test]
    fn variance_keepdim_keeps_a_unit_axis() {
        let t = sample();
        let out = B::variance_keepdim::<f32>(&t, 1, false).unwrap();
        assert_eq!(out.shape(), &[2, 1], "keepdim leaves a unit axis");
        assert_close(&read(&out), &[2.0 / 3.0, 2.0 / 3.0], 1e-5);
    }

    #[test]
    fn variance_dim_bessel_matches_hand_reference() {
        let t = sample();
        // Axis 0 Bessel var per column: mean([1,4])=2.5, ss=2*(1.5^2)=4.5,
        // divisor = count-1 = 1 → 4.5.
        let got = read(&B::variance_dim::<f32>(&t, 0, true).unwrap());
        assert_close(&got, &[4.5, 4.5, 4.5], 1e-5);
    }

    #[test]
    fn std_dim_is_the_square_root_of_variance_dim() {
        let t = sample();
        let var = read(&B::variance_dim::<f32>(&t, 1, false).unwrap());
        let std = read(&B::std_dim::<f32>(&t, 1, false).unwrap());
        assert_close(
            &std,
            &var.into_iter().map(f32::sqrt).collect::<Vec<_>>(),
            1e-5,
        );
    }

    #[test]
    fn std_keepdim_is_the_square_root_of_variance_keepdim() {
        let t = sample();
        let var = read(&B::variance_keepdim::<f32>(&t, 1, true).unwrap());
        let std = read(&B::std_keepdim::<f32>(&t, 1, true).unwrap());
        assert_close(
            &std,
            &var.into_iter().map(f32::sqrt).collect::<Vec<_>>(),
            1e-5,
        );
    }

    #[test]
    fn variance_backward_reaches_the_input() {
        let t = sample();
        let (_, grads) = recorded(|| B::variance_all::<f32>(&t, false).unwrap());
        let g = read(grads.get(t.id()).expect("variance records an input grad"));
        assert_eq!(g.len(), 6);
        // Ones seed on pop-var: d/dx = 2(x - mean)/n.
        let mean = 3.5f64;
        let want: Vec<f32> = VALUES
            .iter()
            .map(|&x| (2.0 * (f64::from(x) - mean) / 6.0) as f32)
            .collect();
        assert_close(&g, &want, 1e-5);
    }

    #[test]
    fn variance_records_a_tape_entry_chain() {
        let t = sample();
        let (_, recorded) = forward_recording(|| B::variance_all::<f32>(&t, false).unwrap());
        assert!(
            recorded >= 3,
            "variance is mean+center+square+sum+scale: recorded {recorded}"
        );
    }

    // ── norm ───────────────────────────────────────────────────────────────

    #[test]
    fn norm_order_one_is_the_sum_of_absolute_values() {
        let t = storage(&[-1.0, 2.0, -3.0], &[3]);
        let got = read(&B::norm::<f32>(&t, 1.0).unwrap())[0];
        assert!((got - 6.0).abs() < 1e-5, "L1 norm: got {got}");
    }

    #[test]
    fn norm_order_two_is_the_euclidean_norm() {
        let t = storage(&[3.0, 4.0], &[2]);
        let got = read(&B::norm::<f32>(&t, 2.0).unwrap())[0];
        assert!((got - 5.0).abs() < 1e-5, "L2 norm: got {got}");
    }

    #[test]
    fn norm_general_order_matches_sum_abs_pow_root() {
        let t = storage(&[1.0, 2.0], &[2]);
        let order = 3.0f64;
        let expected = (1.0f64.powf(order) + 2.0f64.powf(order)).powf(1.0 / order) as f32;
        let got = read(&B::norm::<f32>(&t, order).unwrap())[0];
        assert!((got - expected).abs() < 1e-5, "L{order} norm: got {got}");
    }

    #[test]
    fn norm_backward_reaches_the_input() {
        let t = storage(&[3.0, 4.0], &[2]);
        let (_, grads) = recorded(|| B::norm::<f32>(&t, 2.0).unwrap());
        // Ones seed on L2: d|x|_2 / dx_i = x_i / |x|.
        assert_close(
            read(grads.get(t.id()).expect("norm records an input grad")).as_slice(),
            &[3.0 / 5.0, 4.0 / 5.0],
            1e-5,
        );
    }

    // ── cumsum ─────────────────────────────────────────────────────────────

    #[test]
    fn cumsum_scans_along_the_named_axis() {
        let t = sample(); // [[1,2,3],[4,5,6]]
        let across = read(&B::cumsum::<f32>(&t, 1).unwrap());
        assert_close(&across, &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0], 1e-6);
        let down = read(&B::cumsum::<f32>(&t, 0).unwrap());
        assert_close(&down, &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0], 1e-6);
    }

    #[test]
    fn cumsum_preserves_shape() {
        let t = sample();
        let out = B::cumsum::<f32>(&t, 0).unwrap();
        assert_eq!(out.shape(), &[2, 3], "cumsum is shape-preserving");
    }

    #[test]
    fn cumsum_backward_is_the_suffix_sum() {
        // Input [a,b,c] → [a, a+b, a+b+c]; ones seed → suffix [3,2,1].
        let t = storage(&[1.0, 2.0, 3.0], &[3]);
        let (_, grads) = recorded(|| B::cumsum::<f32>(&t, 0).unwrap());
        assert_close(
            read(grads.get(t.id()).expect("cumsum records an input grad")).as_slice(),
            &[3.0, 2.0, 1.0],
            1e-6,
        );
    }

    #[test]
    fn cumsum_backward_along_axis_of_a_matrix() {
        // Ones seed over a row-prefix: each position receives how many
        // outputs sit at or after it along the scan axis.
        let t = sample();
        let (_, grads) = recorded(|| B::cumsum::<f32>(&t, 1).unwrap());
        assert_close(
            read(grads.get(t.id()).expect("cumsum records an input grad")).as_slice(),
            &[3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            1e-6,
        );
    }

    #[test]
    fn cumsum_refuses_an_out_of_range_axis() {
        let t = sample();
        let err = B::cumsum::<f32>(&t, 2).unwrap_err();
        assert!(
            matches!(err, Error::ShapeMismatch { .. }),
            "cumsum axis must fail closed, got {err:?}"
        );
    }

    // ── recording contract ─────────────────────────────────────────────────

    #[test]
    fn nograd_records_nothing() {
        let t = sample();
        let before = crate::metal::tape::depth();
        let _ = GradMode::Disabled.scope(|| B::variance_all::<f32>(&t, false).unwrap());
        let _ = GradMode::Disabled.scope(|| B::std_all::<f32>(&t, true).unwrap());
        let _ = GradMode::Disabled.scope(|| B::variance_dim::<f32>(&t, 1, false).unwrap());
        let _ = GradMode::Disabled.scope(|| B::std_dim::<f32>(&t, 1, false).unwrap());
        let _ = GradMode::Disabled.scope(|| B::variance_keepdim::<f32>(&t, 1, false).unwrap());
        let _ = GradMode::Disabled.scope(|| B::std_keepdim::<f32>(&t, 1, false).unwrap());
        let _ = GradMode::Disabled.scope(|| B::norm::<f32>(&t, 2.0).unwrap());
        let _ = GradMode::Disabled.scope(|| B::cumsum::<f32>(&t, 1).unwrap());
        assert_eq!(
            crate::metal::tape::depth(),
            before,
            "NoGrad must record nothing"
        );
    }
}
