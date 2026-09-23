//! Normalization family on Metal: `softmax`, `log_softmax`, `layer_norm` and
//! `rms_norm` (#92's attention-block normalization gaps), plus the
//! `max_keepdim` reduction the stable softmax recipe needs.
//!
//! Every method here is pure host-side `Vec<f32>` math over
//! [`MetalStorage`]'s shared bytes — the same shape as `backend.rs`'s
//! `sum_dim_impl` walk — so the module compiles and its tests run on any
//! host under `--features metal`. The composites mirror WGPU's
//! (`wgpu/backend/{elementwise,nn}.rs`) step for step: each primitive they
//! call is already tape-tracked, so a composite's backward is the replay of
//! those entries and no new derivative math is written for it. The one
//! hand-written recipe is `max_keepdim`'s, which routes the cotangent to the
//! argmax winner exactly the way WGPU's `push_extremum_dim_tape_entry` does.

use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DType;

use super::backend::{MetalBackendImpl, storage_from_f32};
use super::storage::MetalStorage;

/// Row-major element count of a dims slice, as `backend.rs` spells it.
fn numel(dims: &[usize]) -> Result<usize> {
    Ok(ShapeBuf::from_slice(dims).checked_numel(OperationKind::Storage)?)
}

/// `max_keepdim` forward: per reduction span `(outer, axis, inner)`, the
/// largest value, with the reduced axis kept at extent 1.
///
/// First-wins on ties: the scan seeds `best_val` at `-inf` with `best_flat`
/// at the span's first position and advances only on a strict `>`, so a row
/// of all `-inf` (or a repeated maximum) keeps the lowest index — the same
/// tie rule CPU's and WGPU's extremum kernels use, and the one a gradient
/// routed to "the" winner has to agree with.
///
/// A zero-extent input (any axis of length 0) has no winner to find: the
/// output is filled with the `-inf` seed without touching the input bytes,
/// which keeps the walk out of bounds when `axis_len == 0` but `outer` and
/// `inner` are not.
fn max_keepdim_impl(t: &MetalStorage, axis: usize) -> Result<MetalStorage> {
    let dims = t.metadata().shape().dims();
    if axis >= dims.len() {
        return Err(Error::ShapeMismatch {
            op: "max_keepdim",
            expected: vec![dims.len()],
            got: vec![axis],
            msg: "axis out of bounds".to_string(),
        });
    }
    let mut out_dims = dims.to_vec();
    out_dims[axis] = 1;

    let outer = numel(&dims[..axis])?;
    let axis_len = dims[axis];
    let inner = numel(&dims[axis + 1..])?;
    let mut out = vec![f32::NEG_INFINITY; outer * inner];

    if outer * axis_len * inner > 0 {
        let bytes = t.as_bytes()?;
        let input: &[f32] = bytemuck::cast_slice(bytes);
        for o in 0..outer {
            let base = o * axis_len * inner;
            for i in 0..inner {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_flat = base + i;
                for a in 0..axis_len {
                    let flat = base + a * inner + i;
                    if input[flat] > best_val {
                        best_val = input[flat];
                        best_flat = flat;
                    }
                }
                out[o * inner + i] = input[best_flat];
            }
        }
    }

    storage_from_f32(&out, &out_dims, t)
}

impl<D: Device> MetalBackendImpl<D> {
    /// `max_keepdim(t, axis)`: forward [`max_keepdim_impl`] plus the tape
    /// entry that routes the cotangent to the argmax winner.
    ///
    /// This is not advertised as its own capability row — it exists because
    /// `softmax`/`log_softmax` subtract a per-span max for numerical
    /// stability. That subtraction sits *inside* the taped composite, so the
    /// max needs a real gradient: `M` is a function of `x`, and WGPU's
    /// `max_keepdim` doc walks through why wiring it (rather than detaching
    /// it) is correct — the stable and unstable spellings are algebraically
    /// identical, so their gradients must be too. The backward recomputes
    /// the winners from the captured input, exactly as WGPU's
    /// `push_extremum_dim_tape_entry` does.
    pub(crate) fn max_keepdim<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let out = max_keepdim_impl(t, axis)?;
        let dims = t.metadata().shape().dims().to_vec();
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id(), out.id());
        crate::metal::tape::push(crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let outer = numel(&dims[..axis])?;
                let axis_len = dims[axis];
                let inner = numel(&dims[axis + 1..])?;
                let in_numel = outer * axis_len * inner;
                // A zero-extent input has no winner to route to (and indexing
                // `input[best_flat]` would be out of bounds); the gradient of
                // an empty tensor is the empty tensor.
                if in_numel == 0 {
                    return Ok(vec![storage_from_f32(&[], &dims, &t_capture)?]);
                }
                let input_bytes = t_capture.as_bytes()?;
                let input: &[f32] = bytemuck::cast_slice(input_bytes);
                let grad_bytes = grad_out.as_bytes()?;
                let grad: &[f32] = bytemuck::cast_slice(grad_bytes);
                let mut grad_input = vec![0.0f32; in_numel];
                for o in 0..outer {
                    let base = o * axis_len * inner;
                    for i in 0..inner {
                        let mut best_val = f32::NEG_INFINITY;
                        let mut best_flat = base + i;
                        for a in 0..axis_len {
                            let flat = base + a * inner + i;
                            if input[flat] > best_val {
                                best_val = input[flat];
                                best_flat = flat;
                            }
                        }
                        grad_input[best_flat] = grad[o * inner + i];
                    }
                }
                Ok(vec![storage_from_f32(&grad_input, &dims, &t_capture)?])
            }),
        });
        Ok(out)
    }

    /// `log_softmax(t, axis) = (t - max) - log(sum_keepdim(exp(t - max), axis))`.
    ///
    /// CPU's and WGPU's numerically-stable kernel, composed entirely from
    /// taped primitives so the backward is their replay. The final `exp` of
    /// `softmax` is deliberately not applied — that round trip loses the tail
    /// entries far below the span maximum.
    pub(crate) fn log_softmax<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if axis >= t.shape().len() {
            return Err(Error::ShapeMismatch {
                op: "log_softmax",
                expected: t.shape().to_vec(),
                got: vec![axis],
                msg: format!(
                    "log_softmax: axis {axis} out of range for shape {:?}",
                    t.shape()
                ),
            });
        }
        let max = Self::max_keepdim::<K>(t, axis)?;
        let diff = Self::sub::<K>(t, &max)?;
        let exp_diff = Self::exp::<K>(&diff)?;
        let sum_exp = Self::sum_keepdim::<K>(&exp_diff, axis)?;
        let log_sum_exp = Self::log::<K>(&sum_exp)?;
        Self::sub::<K>(&diff, &log_sum_exp)
    }

    /// `softmax(t, axis) = exp(log_softmax(t, axis))`.
    ///
    /// Going through `log_softmax` rather than the shorter `exp(diff) /
    /// sum_exp` is deliberate: it is the numerically stable form, it is the
    /// form CPU and WGPU use, and the `sub`/`exp` broadcasts (the reduced
    /// axis stays at extent 1 and `binary_op_metal` aligns it) are the same
    /// ones every other composite here relies on.
    pub(crate) fn softmax<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        axis: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if axis >= t.shape().len() {
            return Err(Error::ShapeMismatch {
                op: "softmax",
                expected: t.shape().to_vec(),
                got: vec![axis],
                msg: format!(
                    "softmax: axis {axis} out of range for shape {:?}",
                    t.shape()
                ),
            });
        }
        let max = Self::max_keepdim::<K>(t, axis)?;
        let diff = Self::sub::<K>(t, &max)?;
        let exp_diff = Self::exp::<K>(&diff)?;
        let sum_exp = Self::sum_keepdim::<K>(&exp_diff, axis)?;
        let log_sum_exp = Self::log::<K>(&sum_exp)?;
        let log_softmax = Self::sub::<K>(&diff, &log_sum_exp)?;
        Self::exp::<K>(&log_softmax)
    }

    /// `layer_norm(input, weight, bias?, epsilon)` over the trailing axis,
    /// step for step the way CPU's `layer_norm_impl` and WGPU's both compose
    /// it: mean, center, mean-of-squares, `+eps`, sqrt, divide, affine.
    /// Every step is already a taped primitive, so this writes no backward of
    /// its own — `training = true` on the capability row is the replay of the
    /// composite. Absent bias returns the weight-scaled term alone, which is
    /// the value CPU reaches by substituting a zero buffer (adding nothing is
    /// the same value).
    ///
    /// The trailing axis only, and the operation carries no axis attribute to
    /// say otherwise — both references agree on that, as does `normalized_shape`
    /// in the attributes, which the descriptor has already validated against
    /// the operand by the time this runs.
    pub(crate) fn layer_norm<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        epsilon: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let last = input.shape().len().saturating_sub(1);
        let mean = Self::mean_keepdim::<K>(input, last)?;
        let centered = Self::sub::<K>(input, &mean)?;
        let squared = Self::mul::<K>(&centered, &centered)?;
        let variance = Self::mean_keepdim::<K>(&squared, last)?;
        let guarded = Self::add_scalar_float::<K>(&variance, epsilon)?;
        let std = Self::sqrt::<K>(&guarded)?;
        let normalized = Self::div::<K>(&centered, &std)?;
        let scaled = Self::mul::<K>(&normalized, weight)?;
        match bias {
            Some(bias) => Self::add::<K>(&scaled, bias),
            None => Ok(scaled),
        }
    }

    /// `rms_norm(input, weight, epsilon)`: root-mean-square scaling over the
    /// last axis, no mean-centering — `x / sqrt(mean(x^2) + eps) * weight`.
    ///
    /// `epsilon` is added to the mean *before* the square root, not after:
    /// that ordering keeps the root finite when every element of a span is
    /// zero, which adding afterwards would not. The axis is the last one and
    /// the operation carries no axis attribute — matching CPU, CUDA and
    /// WGPU. Every step is a taped primitive, so — like `layer_norm` — this
    /// pushes nothing of its own.
    pub(crate) fn rms_norm<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        epsilon: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let axis = input.shape().len().saturating_sub(1);
        let squared = Self::mul::<K>(input, input)?;
        let mean = Self::mean_keepdim::<K>(&squared, axis)?;
        let guarded = Self::add_scalar_float::<K>(&mean, epsilon)?;
        let scale = Self::sqrt::<K>(&guarded)?;
        let normalized = Self::div::<K>(input, &scale)?;
        Self::mul::<K>(&normalized, weight)
    }
}

#[cfg(test)]
/// Host-side forward/backward parity tests for the normalization family.
/// Pure `Vec<f32>` math, so they run without a Metal device.
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

    fn vector(v: &[f32]) -> MetalStorage {
        storage(v, &[v.len()])
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
            if g.is_nan() && w.is_nan() {
                continue;
            }
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

    /// Run `f` under recording mode, walk back with an explicit cotangent,
    /// return `(output, grads)`.
    fn recorded_with_seed<F>(f: F, seed: &MetalStorage) -> (MetalStorage, MetalGrads)
    where
        F: FnOnce() -> MetalStorage,
    {
        let out = GradMode::Enabled.scope(f);
        let grads =
            crate::metal::tape::backward_with(&out, seed).expect("seeded backward walk succeeds");
        (out, grads)
    }

    /// Run `f` under recording mode and report the tape entries it left.
    fn forward_recording<F>(f: F) -> (MetalStorage, usize)
    where
        F: FnOnce() -> MetalStorage,
    {
        let before = crate::metal::tape::depth();
        let out = GradMode::Enabled.scope(f);
        let delta = crate::metal::tape::depth() - before;
        (out, delta)
    }

    /// Two rows of four; the second row is deliberately wide (`-8` next to
    /// `8`), the input where the unstable `exp(x)/sum(exp(x))` spelling
    /// overflows and the stable one does not — the same VALUES the WGPU
    /// softmax suite uses.
    const VALUES: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -8.0, 0.0, 8.0, 0.0];
    const ROWS: usize = 2;
    const COLS: usize = 4;

    /// The stable host reference, in `f64`, row-major over `axis`.
    fn softmax_reference(values: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; values.len()];
        for row in 0..rows {
            let slice = &values[row * cols..(row + 1) * cols];
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials: Vec<f64> = slice
                .iter()
                .map(|value| f64::from(*value - max).exp())
                .collect();
            let total: f64 = exponentials.iter().sum();
            for (column, value) in exponentials.iter().enumerate() {
                out[row * cols + column] = value / total;
            }
        }
        out
    }

    // ── Softmax / log_softmax ──────────────────────────────────────────────

    #[test]
    fn softmax_rows_sum_to_one_and_match_the_stable_reference() {
        let input = storage(&VALUES, &[ROWS, COLS]);
        let out = B::softmax::<f32>(&input, 1).unwrap();
        let got = read(&out);
        let want = softmax_reference(&VALUES, ROWS, COLS);
        for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 1e-6,
                "element {index}: got {g}, reference {w}"
            );
        }
        for row in 0..ROWS {
            let total: f64 = got[row * COLS..(row + 1) * COLS]
                .iter()
                .map(|&v| f64::from(v))
                .sum();
            assert!((total - 1.0).abs() < 1e-6, "row {row} sums to {total}");
        }
    }

    #[test]
    fn the_axis_attribute_is_honoured() {
        let input = storage(&VALUES, &[ROWS, COLS]);
        let down_columns = read(&B::softmax::<f32>(&input, 0).unwrap());
        let across_rows = read(&B::softmax::<f32>(&input, 1).unwrap());
        assert_ne!(
            down_columns, across_rows,
            "axis 0 and axis 1 must not produce the same tensor"
        );
        for column in 0..COLS {
            let total: f64 = (0..ROWS)
                .map(|row| f64::from(down_columns[row * COLS + column]))
                .sum();
            assert!(
                (total - 1.0).abs() < 1e-6,
                "column {column} sums to {total}, not 1.0"
            );
        }
    }

    #[test]
    fn a_wide_row_stays_finite_and_normalised() {
        const WIDE: [f32; 4] = [-100.0, 0.0, 100.0, 50.0];
        let input = storage(&WIDE, &[1, 4]);
        let got = read(&B::softmax::<f32>(&input, 1).unwrap());
        assert!(
            got.iter().all(|value| value.is_finite()),
            "a wide row must not overflow to inf or NaN: {got:?}"
        );
        let total: f64 = got.iter().map(|&v| f64::from(v)).sum();
        assert!((total - 1.0).abs() < 1e-6, "wide row sums to {total}");
        assert!(
            got[2] > 0.99,
            "the largest element should carry almost all the mass: {got:?}"
        );
    }

    #[test]
    fn log_softmax_matches_the_stable_formula() {
        let input = storage(&VALUES, &[ROWS, COLS]);
        let got = read(&B::log_softmax::<f32>(&input, 1).unwrap());
        let softmax = softmax_reference(&VALUES, ROWS, COLS);
        // log_softmax = log(softmax) mathematically; compare that way so the
        // reference is written once.
        let want: Vec<f64> = softmax.iter().map(|p| p.ln()).collect();
        for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 1e-5,
                "element {index}: got {g}, reference {w}"
            );
        }
        // The stable spelling caps every entry at 0 (the span maximum maps to
        // log(1) = 0); the unstable one can exceed it on a wide row.
        assert!(
            got.iter().all(|&v| v <= 1e-6),
            "log_softmax entries must not exceed 0: {got:?}"
        );
    }

    #[test]
    fn softmax_records_a_tape_entry_chain() {
        let input = storage(&VALUES, &[ROWS, COLS]);
        let (out, recorded) = forward_recording(|| B::softmax::<f32>(&input, 1).unwrap());
        assert_eq!(read(&out).len(), VALUES.len(), "softmax keeps the shape");
        assert!(
            recorded >= 2,
            "softmax advertises `training = true` and is composed from \
             tape-tracked primitives, so it must leave entries behind: \
             recorded {recorded}"
        );
    }

    #[test]
    fn max_keepdim_routes_backward_to_first_winner_on_a_tie() {
        // First-wins: both maxima are 1.0, so the seed lands on index 0.
        let t = vector(&[1.0, 1.0, 0.0]);
        let (_, grads) = recorded_with_seed(
            || B::max_keepdim::<f32>(&t, 0).unwrap(),
            &storage(&[1.0], &[1]),
        );
        assert_eq!(
            read(
                grads
                    .get(t.id())
                    .expect("max_keepdim records an input grad")
            ),
            vec![1.0, 0.0, 0.0],
            "a tied maximum must route the cotangent to the first winner"
        );

        // Two spans, general seeds: each span's seed lands on its winner.
        let t = storage(&[1.0, 1.0, 0.0, 2.0], &[2, 2]);
        let (_, grads) = recorded_with_seed(
            || B::max_keepdim::<f32>(&t, 1).unwrap(),
            &storage(&[0.5, 0.75], &[2, 1]),
        );
        assert_close(
            read(
                grads
                    .get(t.id())
                    .expect("max_keepdim records an input grad"),
            )
            .as_slice(),
            &[0.5, 0.0, 0.0, 0.75],
            1e-6,
        );
    }

    #[test]
    fn softmax_backward_with_general_seed_matches_closed_form() {
        // grad_j = y_j * (s_j - sum_i s_i y_i), y = softmax(x) — the closed
        // form of replaying the taped stable composite with cotangent s.
        let x = [2.0f64, 1.0, 0.5];
        let t = vector(&[2.0, 1.0, 0.5]);
        let seed = storage(&[1.0, 0.5, -0.25], &[3]);
        let (_, grads) = recorded_with_seed(|| B::softmax::<f32>(&t, 0).unwrap(), &seed);

        let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
        let total: f64 = exp.iter().sum();
        let y: Vec<f64> = exp.iter().map(|e| e / total).collect();
        let s = [1.0f64, 0.5, -0.25];
        let dot: f64 = s.iter().zip(&y).map(|(si, yi)| si * yi).sum();
        let want: Vec<f32> = y
            .iter()
            .zip(&s)
            .map(|(yi, si)| (yi * (si - dot)) as f32)
            .collect();
        assert_close(
            read(grads.get(t.id()).expect("softmax records an input grad")).as_slice(),
            &want,
            1e-5,
        );
    }

    #[test]
    fn log_softmax_backward_with_general_seed_matches_closed_form() {
        // grad_j = s_j - y_j * sum_i s_i, y = softmax(x).
        let x = [2.0f64, 1.0, 0.5];
        let t = vector(&[2.0, 1.0, 0.5]);
        let seed = storage(&[1.0, 0.5, -0.25], &[3]);
        let (_, grads) = recorded_with_seed(|| B::log_softmax::<f32>(&t, 0).unwrap(), &seed);

        let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
        let total: f64 = exp.iter().sum();
        let y: Vec<f64> = exp.iter().map(|e| e / total).collect();
        let s = [1.0f64, 0.5, -0.25];
        let sum_s: f64 = s.iter().sum();
        let want: Vec<f32> = s
            .iter()
            .zip(&y)
            .map(|(si, yi)| (si - yi * sum_s) as f32)
            .collect();
        assert_close(
            read(
                grads
                    .get(t.id())
                    .expect("log_softmax records an input grad"),
            )
            .as_slice(),
            &want,
            1e-5,
        );
    }

    // ── layer_norm ─────────────────────────────────────────────────────────

    /// Two rows of three, deliberately non-zero-mean so the centering step
    /// is exercised rather than a no-op.
    const NORM_IN: [f32; 6] = [1.0, 2.0, 3.0, 6.0, -4.0, 0.0];

    /// The host reference: per trailing span, `(x - mean) / sqrt(var + eps)
    /// * w + b`, computed in `f64`.
    fn layer_norm_reference(
        values: &[f32],
        rows: usize,
        cols: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; values.len()];
        for r in 0..rows {
            let row = &values[r * cols..(r + 1) * cols];
            let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / cols as f64;
            let var = row
                .iter()
                .map(|&x| {
                    let d = f64::from(x) - mean;
                    d * d
                })
                .sum::<f64>()
                / cols as f64;
            let inv = 1.0 / (var + eps).sqrt();
            for (c, &x) in row.iter().enumerate() {
                let mut value = (f64::from(x) - mean) * inv * f64::from(weight[c]);
                if let Some(bias) = bias {
                    value += f64::from(bias[c]);
                }
                out[r * cols + c] = value;
            }
        }
        out
    }

    #[test]
    fn layer_norm_without_bias_matches_the_reference() {
        let input = storage(&NORM_IN, &[2, 3]);
        let weight = storage(&[2.0, 0.5, -1.0], &[3]);
        let out = B::layer_norm::<f32>(&input, &weight, None, 1e-5).unwrap();
        let want = layer_norm_reference(&NORM_IN, 2, 3, &[2.0, 0.5, -1.0], None, 1e-5);
        let got: Vec<f64> = read(&out).into_iter().map(f64::from).collect();
        for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "element {index}: got {g}, reference {w}"
            );
        }
    }

    #[test]
    fn layer_norm_with_bias_applies_the_affine_shift() {
        let input = storage(&NORM_IN, &[2, 3]);
        let weight = storage(&[1.0f32; 3], &[3]);
        let bias = storage(&[0.5, -1.0, 2.0], &[3]);
        let out = B::layer_norm::<f32>(&input, &weight, Some(&bias), 1e-5).unwrap();
        let want = layer_norm_reference(
            &NORM_IN,
            2,
            3,
            &[1.0, 1.0, 1.0],
            Some(&[0.5, -1.0, 2.0]),
            1e-5,
        );
        let got: Vec<f64> = read(&out).into_iter().map(f64::from).collect();
        for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "element {index}: got {g}, reference {w}"
            );
        }
        // Centered rows plus a non-zero bias must actually shift: the bias
        // is not silently dropped.
        assert_ne!(
            read(&out),
            read(&B::layer_norm::<f32>(&input, &weight, None, 1e-5).unwrap()),
            "a present bias must change the output"
        );
    }

    #[test]
    fn layer_norm_backward_ones_seed_is_analytically_zero_for_input() {
        // With weight = ones, L = sum(out) is constant in x (each normalized
        // row sums to 0 identically) and = sum(bias) overall: grad_x = 0,
        // grad_bias = ones, grad_weight = column sums of (out - bias).
        let input = storage(&NORM_IN, &[2, 3]);
        let weight = storage(&[1.0f32; 3], &[3]);
        let bias = storage(&[0.5, -1.0, 2.0], &[3]);
        let (out, grads) =
            recorded(|| B::layer_norm::<f32>(&input, &weight, Some(&bias), 1e-5).unwrap());

        let grad_input = read(
            grads
                .get(input.id())
                .expect("layer_norm records an input grad"),
        );
        for (index, g) in grad_input.iter().enumerate() {
            assert!(
                g.abs() < 1e-4,
                "grad_input[{index}] = {g}, expected ~0 for a ones seed"
            );
        }
        // dL/db_j = one per row: two rows share each bias slot.
        assert_close(
            read(
                grads
                    .get(bias.id())
                    .expect("layer_norm records a bias grad"),
            )
            .as_slice(),
            &[2.0, 2.0, 2.0],
            1e-6,
        );

        // grad_w_j = sum_i norm_ij, and norm = out - bias under w = ones —
        // column sums of the un-biased forward, not zero in general (each
        // row divides by its own std, so the columns do not cancel).
        let bias_values = read(&bias);
        let normalized: Vec<f32> = read(&out)
            .chunks(3)
            .flat_map(|row| {
                row.iter()
                    .zip(bias_values.iter())
                    .map(|(o, b)| o - b)
                    .collect::<Vec<f32>>()
            })
            .collect();
        let want_w: Vec<f32> = (0..3)
            .map(|c| (0..2).map(|r| normalized[r * 3 + c]).sum::<f32>())
            .collect();
        assert_close(
            read(
                grads
                    .get(weight.id())
                    .expect("layer_norm records a weight grad"),
            )
            .as_slice(),
            &want_w,
            1e-4,
        );
    }

    // ── rms_norm ───────────────────────────────────────────────────────────

    #[test]
    fn rms_norm_forward_matches_formula_and_weight_grad() {
        let input = storage(&NORM_IN, &[2, 3]);
        let weight = storage(&[1.0f32; 3], &[3]);
        let epsilon = 1e-5f64;
        let (out, grads) = recorded(|| B::rms_norm::<f32>(&input, &weight, epsilon).unwrap());

        let got = read(&out);
        let want: Vec<f64> = NORM_IN
            .chunks(3)
            .flat_map(|row| {
                let mean_sq = row
                    .iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum::<f64>()
                    / row.len() as f64;
                let scale = 1.0 / (mean_sq + epsilon).sqrt();
                row.iter().map(move |&x| f64::from(x) * scale)
            })
            .collect();
        for (index, (g, w)) in got
            .iter()
            .copied()
            .map(f64::from)
            .zip(want.iter())
            .enumerate()
        {
            assert!(
                (g - w).abs() < 1e-5,
                "element {index}: got {g}, reference {w}"
            );
        }

        // Ones seed: grad_w_j = sum_i out_ij / w_j = column sums of out when
        // w = 1; grad_input is generally non-zero (rms does not center).
        let want_w: Vec<f32> = (0..3)
            .map(|c| (0..2).map(|r| got[r * 3 + c]).sum::<f32>())
            .collect();
        assert_close(
            read(
                grads
                    .get(weight.id())
                    .expect("rms_norm records a weight grad"),
            )
            .as_slice(),
            &want_w,
            1e-4,
        );
        let grad_input = read(
            grads
                .get(input.id())
                .expect("rms_norm records an input grad"),
        );
        assert!(
            grad_input.iter().any(|g| g.abs() > 1e-6),
            "rms_norm must record a non-trivial input gradient"
        );
    }

    // ── Recording contract ─────────────────────────────────────────────────

    #[test]
    fn nograd_records_nothing() {
        let t = storage(&NORM_IN, &[2, 3]);
        let w = storage(&[1.0f32; 3], &[3]);
        let before = crate::metal::tape::depth();
        let _ = GradMode::Disabled.scope(|| B::softmax::<f32>(&t, 1).unwrap());
        let _ = GradMode::Disabled.scope(|| B::log_softmax::<f32>(&t, 1).unwrap());
        let _ = GradMode::Disabled.scope(|| B::layer_norm::<f32>(&t, &w, None, 1e-5).unwrap());
        let _ = GradMode::Disabled.scope(|| B::rms_norm::<f32>(&t, &w, 1e-5).unwrap());
        let _ = GradMode::Disabled.scope(|| B::max_keepdim::<f32>(&t, 1).unwrap());
        assert_eq!(
            crate::metal::tape::depth(),
            before,
            "NoGrad must record nothing"
        );
    }
}
