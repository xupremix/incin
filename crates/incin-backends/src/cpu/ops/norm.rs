//! Free-function normalization helpers for `CpuBackendImpl<D>`.
//!
//! `layer_norm_impl` and `batch_norm_impl` are called by the trait dispatch
//! methods in `ops/module.rs`. They are `pub(crate)` rather than `pub` so
//! they stay internal to this crate and are not part of the public API surface.
//!
//! Both are composed entirely from already-tape-tracked primitives
//! (sub / mul / div / add / sqrt / add_scalar_float / mean_keepdim) — zero
//! new `tape::push`/backward closures are written here. The existing backward
//! closures already handle the unbroadcast math correctly for any shape.

use incin_core::prelude::{DType, Result};
use incin_core::tensor::backend::{FloatOps, NumericOps, ReductionOps};

use crate::cpu::CpuBackendImpl;
use crate::cpu::storage::{CpuBuffer, CpuStorage};

// ---------------------------------------------------------------------------
// layer_norm
// ---------------------------------------------------------------------------

/// Normalizes over the TRAILING dimension only, independently per every
/// leading (batch/sequence/spatial) position.
///
/// `layer_norm(x, weight, bias, eps)`:
/// 1. `mean = mean_keepdim(x, last_dim)`
/// 2. `centered = x - mean`
/// 3. `variance = mean_keepdim(centered * centered, last_dim)`
/// 4. `std = sqrt(variance + eps)`
/// 5. `normalized = centered / std`
/// 6. `result = normalized * weight + bias` (weight/bias broadcast over
///    all leading dims — they have shape `[hidden_size]`)
///
/// Matches `candle-nn-0.9.1`'s `LayerNorm::forward` semantics exactly
/// (trailing-dim only, not batch-level normalization).
///
/// When `bias` is `None`, a zero-filled buffer of the same shape as `weight`
/// is substituted (matching `CandleBackend::layer_norm`'s `zeros_like`
/// default-fallback convention, confirmed by direct code read).
pub(crate) fn layer_norm_impl<D: incin_core::prelude::Device, K: DType>(
    t: &CpuStorage,
    weight: &CpuStorage,
    bias: Option<&CpuStorage>,
    eps: f32,
) -> Result<CpuStorage> {
    /// `B`.
    type B<D> = CpuBackendImpl<D>;

    let rank = t.shape.len();
    let last_dim = rank - 1;

    // ── NATIVE CUDA FAST PATH ──

    // 1. mean_keepdim over the trailing dim → shape matches t with last dim = 1
    let mean = <B<D> as ReductionOps<B<D>>>::mean_keepdim::<K>(t, last_dim)?;
    // 2. centered = t - mean  (broadcast sub)
    let centered = <B<D> as NumericOps<B<D>>>::sub::<K>(t, &mean)?;
    // 3. variance = mean_keepdim(centered², trailing dim)
    let sq = <B<D> as NumericOps<B<D>>>::mul::<K>(&centered, &centered)?;
    let variance = <B<D> as ReductionOps<B<D>>>::mean_keepdim::<K>(&sq, last_dim)?;
    // 4. std = sqrt(variance + eps)
    let var_plus_eps = <B<D> as FloatOps<B<D>>>::add_scalar_float::<K>(&variance, eps as f64)?;
    let std = <B<D> as FloatOps<B<D>>>::sqrt::<K>(&var_plus_eps)?;
    // 5. normalized = centered / std
    let normalized = <B<D> as NumericOps<B<D>>>::div::<K>(&centered, &std)?;
    // 6. affine: normalized * weight + bias
    let scaled = <B<D> as NumericOps<B<D>>>::mul::<K>(&normalized, weight)?;
    // Default-fallback: absent bias → zero-filled buffer shaped like weight.
    let bias_storage: CpuStorage;
    let bias_ref = match bias {
        Some(b) => b,
        None => {
            let n = crate::cpu::stride::checked_numel(&weight.shape)?;
            bias_storage =
                CpuStorage::from_contiguous(CpuBuffer::F32(vec![0.0f32; n]), weight.shape.to_vec());
            &bias_storage
        }
    };
    <B<D> as NumericOps<B<D>>>::add::<K>(&scaled, bias_ref)
}

// ---------------------------------------------------------------------------
// batch_norm
// ---------------------------------------------------------------------------

/// Per-channel normalization using the GIVEN `running_mean`/`running_var`
/// (inference-mode only — `momentum` is intentionally ignored, matching
/// `CandleBackend`'s confirmed inference-mode-only semantic per CONTEXT.md).
///
/// `channel_dim = if rank > 1 { 1 } else { 0 }` (matching Candle's exact rule).
/// All optional args default per Candle's convention:
///   - absent `running_mean` → zeros shaped `[1, C, 1, ...]`
///   - absent `running_var`  → ones  shaped `[1, C, 1, ...]`
///   - absent `weight`       → ones  shaped `[1, C, 1, ...]`
///   - absent `bias`         → zeros shaped `[1, C, 1, ...]`
///
/// Formula: `((t - rm) / sqrt(rv + eps)) * weight + bias`, all broadcast.
pub(crate) fn batch_norm_impl<D: incin_core::prelude::Device, K: DType>(
    t: &CpuStorage,
    w: Option<&CpuStorage>,
    b: Option<&CpuStorage>,
    rm: Option<&CpuStorage>,
    rv: Option<&CpuStorage>,
    eps: f32,
    _momentum: f64, // deliberately unused — inference-mode-only (CONTEXT.md carried-forward decision)
) -> Result<CpuStorage> {
    /// `B`.
    type B<D> = CpuBackendImpl<D>;

    let rank = t.shape.len();
    let channel_dim = if rank > 1 { 1 } else { 0 };
    let num_channels = t.shape[channel_dim];

    // ── NATIVE CUDA FAST PATH ──

    // Build the broadcast shape [1, C, 1, 1, ...] for each optional arg.
    let mut bcast_shape = vec![1usize; rank];
    bcast_shape[channel_dim] = num_channels;

    // Helper: zeros or ones constant buffer in bcast_shape.
    let make_buf = |fill: f32| -> CpuStorage {
        let n = num_channels;
        CpuStorage::from_contiguous(CpuBuffer::F32(vec![fill; n]), bcast_shape.clone())
    };

    // Reshape a provided storage to bcast_shape (it arrives as a flat [C] vector).
    // CpuStorage::reshape is the inherent method (not tape-tracked here —
    // these are treated as fixed parameters, not differentiated inputs).
    let reshape_to_bcast = |s: &CpuStorage| -> Result<CpuStorage> { s.reshape(&bcast_shape) };

    let rm_s;
    let rm_ref: CpuStorage = match rm {
        Some(s) => {
            rm_s = reshape_to_bcast(s)?;
            rm_s
        }
        None => make_buf(0.0),
    };

    let rv_s;
    let rv_ref: CpuStorage = match rv {
        Some(s) => {
            rv_s = reshape_to_bcast(s)?;
            rv_s
        }
        None => make_buf(1.0),
    };

    let w_s;
    let w_ref: CpuStorage = match w {
        Some(s) => {
            w_s = reshape_to_bcast(s)?;
            w_s
        }
        None => make_buf(1.0),
    };

    let b_s;
    let b_ref: CpuStorage = match b {
        Some(s) => {
            b_s = reshape_to_bcast(s)?;
            b_s
        }
        None => make_buf(0.0),
    };

    // (t - rm) / sqrt(rv + eps) * w + b — all broadcast via existing tape-tracked ops.
    let centered = <B<D> as NumericOps<B<D>>>::sub::<K>(t, &rm_ref)?;
    let rv_eps = <B<D> as FloatOps<B<D>>>::add_scalar_float::<K>(&rv_ref, eps as f64)?;
    let std = <B<D> as FloatOps<B<D>>>::sqrt::<K>(&rv_eps)?;
    let normalized = <B<D> as NumericOps<B<D>>>::div::<K>(&centered, &std)?;
    let scaled = <B<D> as NumericOps<B<D>>>::mul::<K>(&normalized, &w_ref)?;
    <B<D> as NumericOps<B<D>>>::add::<K>(&scaled, &b_ref)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Training-mode batch normalization: normalize by the statistics of the
/// batch in front of it rather than by accumulated running ones.
///
/// This is the mode that makes a convolutional network train. Inference mode
/// divides by statistics gathered earlier, which is a constant with respect to
/// the input; training mode divides by statistics that are themselves
/// functions of every element in the batch, and that dependency is most of
/// what batch norm contributes to the gradient. `batch_norm_impl` above cannot
/// express it: its signature takes running statistics and its `momentum` is
/// bound to `_momentum`, so a training request routed there returned the
/// inference answer with nothing to distinguish it from a correct one. The
/// canonical executor refused it for exactly that reason.
///
/// Statistics are per channel, over the batch and every spatial position, so
/// for `[N, C, H, W]` each channel reduces `N * H * W` elements to one mean
/// and one variance. The variance is the population one, matching what the
/// running statistics of the inference path hold.
///
/// The running statistics are *not* updated here, and cannot be: they arrive
/// as shared references. Updating them is a mutation through an operand,
/// which the execution contract does not currently carry. A caller that
/// trains with this and then evaluates with inference mode is therefore
/// reading whatever running statistics it supplied, unchanged.
pub(crate) fn batch_norm_training_impl<D: incin_core::prelude::Device, K: DType>(
    t: &CpuStorage,
    w: Option<&CpuStorage>,
    b: Option<&CpuStorage>,
    eps: f32,
) -> Result<CpuStorage> {
    /// `B`.
    type B<D> = CpuBackendImpl<D>;

    let rank = t.shape.len();
    let channel_dim = if rank > 1 { 1 } else { 0 };
    let num_channels = t.shape[channel_dim];

    let mut bcast_shape = vec![1usize; rank];
    bcast_shape[channel_dim] = num_channels;

    // Every axis but the channel one is reduced away, keeping its position so
    // the result broadcasts back against the input without a reshape.
    let reduced_axes: Vec<usize> = (0..rank).filter(|axis| *axis != channel_dim).collect();
    let count: usize = reduced_axes.iter().map(|&axis| t.shape[axis]).product();
    if count == 0 {
        return Err(incin_core::prelude::Error::Msg(
            "batch_norm: training mode needs at least one element per channel".into(),
        ));
    }

    let sum_over_reduced = |x: &CpuStorage| -> Result<CpuStorage> {
        let mut acc = x.clone();
        for &axis in &reduced_axes {
            acc = crate::cpu::ops::reduce::sum_axis_keepdim(&acc, axis)?;
        }
        Ok(acc)
    };

    let inv_count = 1.0 / count as f64;
    let total = sum_over_reduced(t)?;
    let mean = <B<D> as FloatOps<B<D>>>::mul_scalar_float::<K>(&total, inv_count)?;
    let centered = <B<D> as NumericOps<B<D>>>::sub::<K>(t, &mean)?;
    let squared = <B<D> as NumericOps<B<D>>>::mul::<K>(&centered, &centered)?;
    let squared_total = sum_over_reduced(&squared)?;
    let variance = <B<D> as FloatOps<B<D>>>::mul_scalar_float::<K>(&squared_total, inv_count)?;

    let variance_eps = <B<D> as FloatOps<B<D>>>::add_scalar_float::<K>(&variance, eps as f64)?;
    let std = <B<D> as FloatOps<B<D>>>::sqrt::<K>(&variance_eps)?;
    let normalized = <B<D> as NumericOps<B<D>>>::div::<K>(&centered, &std)?;

    let scaled = match w {
        Some(weight) => {
            let weight = weight.reshape(&bcast_shape)?;
            <B<D> as NumericOps<B<D>>>::mul::<K>(&normalized, &weight)?
        }
        None => normalized,
    };
    match b {
        Some(bias) => {
            let bias = bias.reshape(&bcast_shape)?;
            <B<D> as NumericOps<B<D>>>::add::<K>(&scaled, &bias)
        }
        None => Ok(scaled),
    }
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::gradcheck::gradcheck;
    use crate::cpu::storage::{CpuBuffer, CpuStorage};

    /// `TestB`.
    type TestB = CpuBackendImpl<incin_core::prelude::Cpu>;

    /// `matrix`.
    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![rows, cols])
    }

    /// `tensor3`.
    fn tensor3(v: Vec<f32>, d0: usize, d1: usize, d2: usize) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![d0, d1, d2])
    }

    /// `tensor4`.
    fn tensor4(v: Vec<f32>, d0: usize, d1: usize, d2: usize, d3: usize) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![d0, d1, d2, d3])
    }

    /// `vec1`.
    fn vec1(v: Vec<f32>) -> CpuStorage {
        let n = v.len();
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![n])
    }

    /// `f32_vec`.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    // --- layer_norm tests (Plan 04-03 Task 1) ---

    /// Hand-compute layer_norm for a [2,3] input with weight=[1,1,1], bias=[0,0,0].
    /// Row 0: [1,2,3]. mean=2, centered=[-1,0,1], var=2/3,
    /// std=sqrt(2/3+eps)≈0.8165 (eps≈0). normalized=[-1.225,0,1.225].
    #[test]
    fn layer_norm_identity_weight_zero_bias_normalizes_each_row() {
        let t = matrix(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let weight = vec1(vec![1.0f32, 1.0, 1.0]);
        let bias = vec1(vec![0.0f32, 0.0, 0.0]);
        let eps = 1e-5f32;

        let out = layer_norm_impl::<incin_core::prelude::Cpu, f32>(&t, &weight, Some(&bias), eps)
            .unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        let vals = f32_vec(&out);

        // Row 0: mean=2, var=2/3, std=sqrt(2/3+eps)
        let mean0 = 2.0f32;
        let std0 = ((2.0f32 / 3.0) + eps).sqrt();
        let expected_row0 = [(1.0 - mean0) / std0, 0.0, (3.0 - mean0) / std0];
        for (i, (&got, &exp)) in vals[..3].iter().zip(expected_row0.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "row0[{i}]: got {got:.6}, expected {exp:.6}"
            );
        }

        // Each row's output should have mean≈0, var≈1 (identity weight/zero bias).
        for row in 0..2usize {
            let row_vals = &vals[row * 3..(row + 1) * 3];
            let row_mean: f32 = row_vals.iter().sum::<f32>() / 3.0;
            let row_var: f32 = row_vals.iter().map(|v| (v - row_mean).powi(2)).sum::<f32>() / 3.0;
            assert!(
                row_mean.abs() < 1e-4,
                "row{row} mean should be ~0: {row_mean:.6}"
            );
            assert!(
                (row_var - 1.0).abs() < 1e-3,
                "row{row} var should be ~1: {row_var:.6}"
            );
        }
    }

    #[test]
    /// `layer_norm_custom_weight_and_bias_applies_affine`.
    fn layer_norm_custom_weight_and_bias_applies_affine() {
        // Row 0: [1,2,3] → normalized ≈ [-1.225, 0, 1.225]
        // weight=[2, 0.5, 1], bias=[1, 1, 1]
        // result = normalized * weight + bias
        let t = matrix(vec![1.0f32, 2.0, 3.0], 1, 3);
        let weight = vec1(vec![2.0f32, 0.5, 1.0]);
        let bias = vec1(vec![1.0f32, 1.0, 1.0]);
        let eps = 1e-5f32;

        let out = layer_norm_impl::<incin_core::prelude::Cpu, f32>(&t, &weight, Some(&bias), eps)
            .unwrap();
        let vals = f32_vec(&out);

        let mean = 2.0f32;
        let std = ((2.0f32 / 3.0) + eps).sqrt();
        let norm = [(1.0 - mean) / std, 0.0, (3.0 - mean) / std];
        let w = [2.0f32, 0.5, 1.0];
        let b = [1.0f32, 1.0, 1.0];
        for i in 0..3 {
            let exp = norm[i] * w[i] + b[i];
            assert!(
                (vals[i] - exp).abs() < 1e-4,
                "affine[{i}]: got {:.6}, expected {exp:.6}",
                vals[i]
            );
        }
    }

    #[test]
    /// `layer_norm_none_bias_matches_explicit_zero_bias`.
    fn layer_norm_none_bias_matches_explicit_zero_bias() {
        // Default-fallback: passing None for bias should produce the same result
        // as passing an explicit all-zeros bias of the trailing-dim size.
        let t = matrix(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let weight = vec1(vec![1.0f32, 2.0, 0.5]);
        let bias_zeros = vec1(vec![0.0f32, 0.0, 0.0]);
        let eps = 1e-5f32;

        let with_explicit_zero =
            layer_norm_impl::<incin_core::prelude::Cpu, f32>(&t, &weight, Some(&bias_zeros), eps)
                .unwrap();
        let with_none_bias =
            layer_norm_impl::<incin_core::prelude::Cpu, f32>(&t, &weight, None, eps).unwrap();

        let a = f32_vec(&with_explicit_zero);
        let b = f32_vec(&with_none_bias);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "none bias[{i}]: explicit={x:.7}, none={y:.7}"
            );
        }
    }

    #[test]
    /// `layer_norm_gradcheck`.
    fn layer_norm_gradcheck() {
        // NOTE: with identity weight, sum(layer_norm(x)) = 0 for all x (normalized
        // values always sum to 0 by definition), so we must use a non-identity weight
        // to get a non-trivial scalar sensitivity to x for finite-difference checking.
        let t = matrix(vec![0.5f32, -1.0, 2.0, 1.0, 0.0, -0.5], 2, 3);
        let weight = vec1(vec![2.0f32, 1.0, 0.5]);
        let bias = vec1(vec![0.1f32, -0.1, 0.2]);
        let eps = 1e-5f32;
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out = layer_norm_impl::<incin_core::prelude::Cpu, f32>(
                &inputs[0],
                &weight,
                Some(&bias),
                eps,
            )
            .unwrap();
            TestB::sum_all::<f32>(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[t], 1e-3);
        assert!(
            max_rel_err < 1e-2,
            "layer_norm gradcheck too high: {max_rel_err:.6}"
        );
    }

    #[test]
    /// `layer_norm_rank3_normalizes_per_batch_seq_position`.
    fn layer_norm_rank3_normalizes_per_batch_seq_position() {
        // t shape [2, 2, 4]: batch=2, seq=2, hidden=4
        // For each (b, s) position, the 4-dim hidden vector should be normalized.
        let t = tensor3(
            vec![
                1.0, 2.0, 3.0, 4.0, // b=0,s=0
                5.0, 6.0, 7.0, 8.0, // b=0,s=1
                9.0, 10.0, 11.0, 12.0, // b=1,s=0
                13.0, 14.0, 15.0, 16.0, // b=1,s=1
            ],
            2,
            2,
            4,
        );
        let weight = vec1(vec![1.0f32; 4]);
        let bias = vec1(vec![0.0f32; 4]);
        let eps = 1e-5f32;

        let out = layer_norm_impl::<incin_core::prelude::Cpu, f32>(&t, &weight, Some(&bias), eps)
            .unwrap();
        assert_eq!(out.shape, vec![2, 2, 4]);
        let vals = f32_vec(&out);

        // Check that each 4-element row has mean≈0, var≈1 (identity affine).
        for row in 0..4usize {
            let rv: &[f32] = &vals[row * 4..(row + 1) * 4];
            let mean: f32 = rv.iter().sum::<f32>() / 4.0;
            let var: f32 = rv.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
            assert!(
                mean.abs() < 1e-4,
                "rank3 row{row} mean should be ~0: {mean:.6}"
            );
            assert!(
                (var - 1.0).abs() < 1e-3,
                "rank3 row{row} var should be ~1: {var:.6}"
            );
        }
    }

    // --- batch_norm tests (Plan 04-03 Task 2) ---

    /// `bn_expected`.
    fn bn_expected(x: f32, rm: f32, rv: f32, w: f32, b: f32, eps: f32) -> f32 {
        (x - rm) / (rv + eps).sqrt() * w + b
    }

    #[test]
    /// `batch_norm_all_args_provided_matches_hand_computed_formula`.
    fn batch_norm_all_args_provided_matches_hand_computed_formula() {
        // Input [2, 3, 2, 2] — batch=2, channels=3, 2×2 spatial.
        // running_mean = [1, 2, 3], running_var = [1, 1, 1], weight = [1, 1, 1], bias = [0, 0, 0].
        // Channel 0: all x-values are 1.0 → (1-1)/sqrt(1+eps)*1+0 ≈ 0
        let t = tensor4(vec![1.0f32; 2 * 3 * 2 * 2], 2, 3, 2, 2);
        let rm = vec1(vec![1.0f32, 2.0, 3.0]);
        let rv = vec1(vec![1.0f32, 1.0, 1.0]);
        let w = vec1(vec![1.0f32, 1.0, 1.0]);
        let b = vec1(vec![0.0f32, 0.0, 0.0]);
        let eps = 1e-5f32;

        let out = batch_norm_impl::<incin_core::prelude::Cpu, f32>(
            &t,
            Some(&w),
            Some(&b),
            Some(&rm),
            Some(&rv),
            eps,
            0.0,
        )
        .unwrap();
        assert_eq!(out.shape, vec![2, 3, 2, 2]);
        let vals = f32_vec(&out);

        // Channel 0: input=1.0, rm=1.0, rv=1.0, w=1, b=0 → (1-1)/1 = 0
        // Channel 1: input=1.0, rm=2.0, rv=1.0, w=1, b=0 → -1
        // Channel 2: input=1.0, rm=3.0, rv=1.0, w=1, b=0 → -2

        // The output layout is [batch, channel, h, w] — element at [0,0,0,0] is channel 0.
        let expected_ch = [
            bn_expected(1.0, 1.0, 1.0, 1.0, 0.0, eps),
            bn_expected(1.0, 2.0, 1.0, 1.0, 0.0, eps),
            bn_expected(1.0, 3.0, 1.0, 1.0, 0.0, eps),
        ];
        // Spot-check one element per channel (output is contiguous [B,C,H,W]).
        for (ch, expected) in expected_ch.iter().enumerate() {
            // First spatial position of first batch for this channel
            let idx = ch * 4; // batch=0, ch, h=0, w=0
            assert!(
                (vals[idx] - expected).abs() < 1e-4,
                "bn ch{ch}: got {:.6}, expected {:.6}",
                vals[idx],
                expected
            );
        }
    }

    #[test]
    /// `batch_norm_momentum_has_no_effect_on_output`.
    fn batch_norm_momentum_has_no_effect_on_output() {
        // Proves momentum is genuinely ignored (inference-mode-only, T-04-07).
        let t = tensor4(vec![1.0f32; 2 * 3 * 2 * 2], 2, 3, 2, 2);
        let rm = vec1(vec![0.0f32, 0.0, 0.0]);
        let rv = vec1(vec![1.0f32, 1.0, 1.0]);
        let w = vec1(vec![1.0f32, 1.0, 1.0]);
        let b = vec1(vec![0.0f32, 0.0, 0.0]);
        let eps = 1e-5f32;

        let out0 = batch_norm_impl::<incin_core::prelude::Cpu, f32>(
            &t,
            Some(&w),
            Some(&b),
            Some(&rm),
            Some(&rv),
            eps,
            0.0,
        )
        .unwrap();
        let out1 = batch_norm_impl::<incin_core::prelude::Cpu, f32>(
            &t,
            Some(&w),
            Some(&b),
            Some(&rm),
            Some(&rv),
            eps,
            0.9,
        )
        .unwrap();

        let v0 = f32_vec(&out0);
        let v1 = f32_vec(&out1);
        for (i, (a, b)) in v0.iter().zip(v1.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "momentum effect at [{i}]: momentum=0 gave {a:.7}, momentum=0.9 gave {b:.7}"
            );
        }
    }

    #[test]
    /// `batch_norm_none_args_match_explicit_default_fallback`.
    fn batch_norm_none_args_match_explicit_default_fallback() {
        // Passing None for all four optional args should equal explicit
        // rm=0, rv=1, w=1, b=0 (Candle's convention — T-04-08 mitigation).
        let t = tensor4(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            ],
            2,
            3,
            2,
            2,
        );
        let rm_zeros = vec1(vec![0.0f32; 3]);
        let rv_ones = vec1(vec![1.0f32; 3]);
        let w_ones = vec1(vec![1.0f32; 3]);
        let b_zeros = vec1(vec![0.0f32; 3]);
        let eps = 1e-5f32;

        let with_explicit = batch_norm_impl::<incin_core::prelude::Cpu, f32>(
            &t,
            Some(&w_ones),
            Some(&b_zeros),
            Some(&rm_zeros),
            Some(&rv_ones),
            eps,
            0.0,
        )
        .unwrap();
        let with_none =
            batch_norm_impl::<incin_core::prelude::Cpu, f32>(&t, None, None, None, None, eps, 0.0)
                .unwrap();

        let a = f32_vec(&with_explicit);
        let b = f32_vec(&with_none);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "default-fallback[{i}]: explicit={x:.7}, none={y:.7}"
            );
        }
    }

    #[test]
    /// `batch_norm_rank2_uses_channel_dim_1`.
    fn batch_norm_rank2_uses_channel_dim_1() {
        // rank=2 → channel_dim = if rank > 1 { 1 } else { 0 } = 1.
        // Input [4, 3]: 4 samples, 3 channels.
        let t = matrix(
            vec![
                1.0, 2.0, 3.0, // sample 0
                4.0, 5.0, 6.0, // sample 1
                7.0, 8.0, 9.0, // sample 2
                10.0, 11.0, 12.0, // sample 3
            ],
            4,
            3,
        );
        let rm = vec1(vec![0.0f32; 3]);
        let rv = vec1(vec![1.0f32; 3]);
        let w = vec1(vec![2.0f32, 1.0, 0.5]);
        let b = vec1(vec![0.0f32; 3]);
        let eps = 1e-5f32;

        let out = batch_norm_impl::<incin_core::prelude::Cpu, f32>(
            &t,
            Some(&w),
            Some(&b),
            Some(&rm),
            Some(&rv),
            eps,
            0.0,
        )
        .unwrap();
        assert_eq!(
            out.shape,
            vec![4, 3],
            "rank-2 batch_norm output shape should be [4,3]"
        );
        let vals = f32_vec(&out);

        // Sample 0, channel 0: (1 - 0) / sqrt(1+eps) * 2 + 0 ≈ 2.0
        let exp_s0c0 = bn_expected(1.0, 0.0, 1.0, 2.0, 0.0, eps);
        assert!(
            (vals[0] - exp_s0c0).abs() < 1e-4,
            "rank2 [0,0]: got {:.6}, expected {exp_s0c0:.6}",
            vals[0]
        );
        // Sample 1, channel 2: (6 - 0) / sqrt(1+eps) * 0.5 + 0 ≈ 3.0
        let exp_s1c2 = bn_expected(6.0, 0.0, 1.0, 0.5, 0.0, eps);
        assert!(
            (vals[5] - exp_s1c2).abs() < 1e-4,
            "rank2 [1,2]: got {:.6}, expected {exp_s1c2:.6}",
            vals[5]
        );
    }

    #[test]
    /// `batch_norm_gradcheck`.
    fn batch_norm_gradcheck() {
        // Gradcheck on [2,3,2,2] with fixed rm/rv/w/b.
        let t = tensor4(
            vec![
                0.5, 1.0, -0.5, 0.2, 1.5, -1.0, 0.3, -0.3, 0.8, -0.8, 1.2, -1.2, 0.1, 0.9, -0.1,
                0.4, 1.1, -0.9, 0.7, -0.7, 0.6, -0.6, 1.3, -1.3,
            ],
            2,
            3,
            2,
            2,
        );
        let rm = vec1(vec![0.0f32; 3]);
        let rv = vec1(vec![1.0f32; 3]);
        let w = vec1(vec![1.0f32; 3]);
        let b = vec1(vec![0.0f32; 3]);
        let eps = 1e-5f32;

        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out = batch_norm_impl::<incin_core::prelude::Cpu, f32>(
                &inputs[0],
                Some(&w),
                Some(&b),
                Some(&rm),
                Some(&rv),
                eps,
                0.0,
            )
            .unwrap();
            TestB::sum_all::<f32>(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[t], 1e-3);
        assert!(
            max_rel_err < 1e-2,
            "batch_norm gradcheck too high: {max_rel_err:.6}"
        );
    }
}
