//! Loss family on Metal: `mse_loss`, `l1_loss`, `bce_with_logits_loss` and
//! `cross_entropy_loss` (#92's training-loss gaps).
//!
//! Every method here is a composition of already tape-tracked Metal
//! primitives — the same step-for-step recipes CPU's `cpu/ops/loss.rs` and
//! WGPU's `wgpu/backend/nn.rs` use — so the backward is the replay of those
//! entries rather than hand-derived math. The one hand-written recipe is
//! BCE's custom slope at the `relu` kink (`0.5` at `x == 0`), recorded
//! through `GradMode::Disabled.restrict` plus `tape::record_with` exactly
//! as CPU and WGPU spell it. The module compiles and its tests run on any
//! host under `--features metal`.

use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::shapes::{Axis, DimensionConstraint, RankExpectation, ShapeError};
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::{DType, DTypeId};
use incin_core::tensor::reduction::Reduction;

use super::backend::{MetalBackendImpl, reshape_metal, storage_from_f32};
use super::storage::MetalStorage;

/// Refuse any dtype that is not `f32` for a value operand (same helper
/// `metal/indexing.rs` uses; private there, so restated for this module).
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

impl<D: Device> MetalBackendImpl<D> {
    /// Apply a loss reduction mode to an intermediate loss tensor.
    fn loss_reduce<K: DType>(
        t: <Self as StorageBackend>::Storage<K>,
        reduction: Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        match reduction {
            Reduction::Mean => Self::mean_all::<K>(&t),
            Reduction::Sum => Self::sum_all::<K>(&t),
            Reduction::None => Ok(t),
        }
    }

    /// `mse_loss(pred, target, reduction)`: `sub`, square, then the
    /// reduction mode — CPU's and WGPU's composition, so the gradient
    /// arrives by replay of the taped primitives.
    pub(crate) fn mse_loss<K: DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        reduction: Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let diff = Self::sub::<K>(pred, target)?;
        let squared = Self::mul::<K>(&diff, &diff)?;
        Self::loss_reduce::<K>(squared, reduction)
    }

    /// `l1_loss(pred, target, reduction)`: `sub`, `abs`, then the reduction
    /// mode — CPU's and WGPU's composition.
    pub(crate) fn l1_loss<K: DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        reduction: Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let diff = Self::sub::<K>(pred, target)?;
        let absolute = Self::abs::<K>(&diff)?;
        Self::loss_reduce::<K>(absolute, reduction)
    }

    /// `bce_with_logits_loss(pred, target, reduction)`: CPU's and WGPU's
    /// numerically-stable recipe — `relu(pred)` under `GradMode::Disabled`
    /// with a custom slope-`0.5` tape entry at the `x == 0` kink, then
    /// `max(x,0) - x*z + log(1 + exp(-|x|))`.
    ///
    /// The forward `relu` must not record its own step-style entry: the
    /// built-in recipe routes zero gradient through the kink, while BCE's
    /// derivative is `0.5` there. Running under `Disabled` silences that
    /// push; `record_with` re-records the real recipe only when the ambient
    /// mode records, so a `NoGrad` forward still allocates nothing.
    pub(crate) fn bce_with_logits_loss<K: DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        reduction: Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let max_x_0 = incin_core::exec::GradMode::Disabled.restrict(|| Self::relu::<K>(pred))?;
        let pred_like = pred.clone();
        let shape = pred.shape().to_vec();
        let pred_bytes = pred.as_bytes()?.to_vec();
        let (pred_id, out_id) = (pred.id(), max_x_0.id());
        crate::metal::tape::record_with(|| crate::metal::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![pred_id],
            backward: Box::new(move |grad_out: &MetalStorage| {
                let x: &[f32] = bytemuck::cast_slice(&pred_bytes);
                let grad_bytes = grad_out.as_bytes()?;
                let g: &[f32] = bytemuck::cast_slice(grad_bytes);
                let mut grad = Vec::with_capacity(x.len());
                for (&xi, &gi) in x.iter().zip(g.iter()) {
                    let slope = if xi == 0.0 {
                        0.5
                    } else if xi > 0.0 {
                        1.0
                    } else {
                        0.0
                    };
                    grad.push(gi * slope);
                }
                Ok(vec![storage_from_f32(&grad, &shape, &pred_like)?])
            }),
        });
        let x_times_z = Self::mul::<K>(pred, target)?;
        let term1 = Self::sub::<K>(&max_x_0, &x_times_z)?;
        let abs_x = Self::abs::<K>(pred)?;
        let neg_abs = Self::neg::<K>(&abs_x)?;
        let exp_neg = Self::exp::<K>(&neg_abs)?;
        let one_plus = Self::add_scalar_float::<K>(&exp_neg, 1.0)?;
        let term2 = Self::log::<K>(&one_plus)?;
        let loss = Self::add::<K>(&term1, &term2)?;
        Self::loss_reduce::<K>(loss, reduction)
    }

    /// `cross_entropy_loss(logits, target, reduction)`: CPU's shape checks
    /// and WGPU's composition — `log_softmax` on the class axis, a
    /// tape-tracked `gather` of the target class from each row, negate,
    /// flatten to `[batch]`, then the reduction mode. The integer target
    /// operand is off the tape (no gradient flows into it); the gather's
    /// scatter-based backward carries the gradient into the logits.
    ///
    /// Metal only creates `I64` index storage (`validate_metal_storage_dtype`
    /// refuses `u8`/`u32`), so a non-`I64` target fails closed by name —
    /// the vacuous remainder of `INDEX_AND_F32_DTYPES` is documented on the
    /// `composed_reduction_indexed` capability row.
    pub(crate) fn cross_entropy_loss(
        logits: &MetalStorage,
        target: &MetalStorage,
        reduction: Reduction,
    ) -> Result<MetalStorage> {
        if logits.shape().len() != 2 {
            return Err(ShapeError::RankMismatch {
                operation: OperationKind::Reduction,
                expected: RankExpectation::Exactly(2),
                actual: logits.shape().len(),
            }
            .into());
        }
        let batch = logits.shape()[0];
        if target.shape() != [batch] {
            return Err(ShapeError::DimensionMismatch {
                operation: OperationKind::Reduction,
                axis: Axis::Index(0),
                lhs: batch,
                rhs: target.shape().first().copied().unwrap_or(0),
                constraint: DimensionConstraint::Equal,
            }
            .into());
        }
        require_f32(logits, "cross_entropy_logits")?;
        if target.metadata().dtype() != DTypeId::I64.descriptor() {
            return Err(Error::UnsupportedDType {
                dtype: target.metadata().dtype(),
                backend: "Metal",
                op: "cross_entropy_target",
            });
        }
        // Class-index range is enforced by `gather`'s `checked_address`
        // (out-of-range → `Error::Shape`), the same fail-closed walk CPU
        // reaches after its explicit pre-pass.
        let log_probs = Self::log_softmax::<f32>(logits, 1)?;
        let target_2d = reshape_metal(target, &[batch, 1])?;
        let picked = Self::gather(&log_probs, 1, &target_2d)?;
        let negated = Self::neg::<f32>(&picked)?;
        let per_sample = Self::reshape::<f32>(&negated, &[batch])?;
        Self::loss_reduce::<f32>(per_sample, reduction)
    }
}

#[cfg(test)]
/// Host-side forward/backward parity tests for the loss family.
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

    /// pred = [[1,2,3],[4,5,6]], target = [[1,1,1],[2,2,2]] — the same
    /// VALUES the CPU loss suite uses.
    fn pred() -> MetalStorage {
        storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
    }

    fn target() -> MetalStorage {
        storage(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3])
    }

    // ── mse_loss ───────────────────────────────────────────────────────────

    #[test]
    fn mse_loss_mean_produces_correct_scalar() {
        let out = B::mse_loss::<f32>(&pred(), &target(), Reduction::Mean).unwrap();
        assert_eq!(out.shape().len(), 0, "Mean output should be scalar");
        let v = read(&out)[0];
        assert!((v - 34.0 / 6.0).abs() < 1e-4, "mse mean: got {v}");
    }

    #[test]
    fn mse_loss_sum_produces_correct_scalar() {
        let out = B::mse_loss::<f32>(&pred(), &target(), Reduction::Sum).unwrap();
        assert_eq!(out.shape().len(), 0);
        let v = read(&out)[0];
        assert!((v - 34.0).abs() < 1e-4, "mse sum: got {v}");
    }

    #[test]
    fn mse_loss_none_produces_elementwise_squared_diff() {
        let out = B::mse_loss::<f32>(&pred(), &target(), Reduction::None).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_close(&read(&out), &[0.0, 1.0, 4.0, 4.0, 9.0, 16.0], 1e-5);
    }

    #[test]
    fn mse_loss_mean_backward_matches_analytic_formula() {
        let p = pred();
        let t = target();
        let (_, grads) = recorded(|| B::mse_loss::<f32>(&p, &t, Reduction::Mean).unwrap());
        let g = read(grads.get(p.id()).expect("pred should have a gradient"));
        assert_eq!(g.len(), 6);
        let expected = [0.0f32, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0];
        assert_close(&g, &expected, 1e-4);
    }

    #[test]
    fn mse_loss_records_a_tape_entry_chain() {
        let p = pred();
        let t = target();
        let (_, recorded) =
            forward_recording(|| B::mse_loss::<f32>(&p, &t, Reduction::Mean).unwrap());
        assert!(
            recorded >= 2,
            "mse is composed from taped primitives: recorded {recorded}"
        );
    }

    // ── l1_loss ────────────────────────────────────────────────────────────

    #[test]
    fn l1_loss_mean_produces_correct_scalar() {
        let out = B::l1_loss::<f32>(&pred(), &target(), Reduction::Mean).unwrap();
        assert_eq!(out.shape().len(), 0);
        let v = read(&out)[0];
        assert!((v - 2.0).abs() < 1e-4, "l1 mean: got {v:.6}");
    }

    #[test]
    fn l1_loss_sum_produces_correct_scalar() {
        let out = B::l1_loss::<f32>(&pred(), &target(), Reduction::Sum).unwrap();
        assert_eq!(out.shape().len(), 0);
        let v = read(&out)[0];
        assert!((v - 12.0).abs() < 1e-4, "l1 sum: got {v:.6}");
    }

    #[test]
    fn l1_loss_none_produces_elementwise_absolute_diff() {
        let out = B::l1_loss::<f32>(&pred(), &target(), Reduction::None).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_close(&read(&out), &[0.0, 1.0, 2.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn l1_loss_backward_uses_sign_of_diff() {
        let p = storage(&[1.0f32, 2.0, 3.0], &[1, 3]);
        let t = storage(&[0.5f32, 0.5, 0.5], &[1, 3]);
        let (_, grads) = recorded(|| B::l1_loss::<f32>(&p, &t, Reduction::Mean).unwrap());
        // Mean over 3 elements; sign(diff) = [+1, +1, +1].
        assert_close(
            read(grads.get(p.id()).expect("pred grad")).as_slice(),
            &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            1e-5,
        );
    }

    // ── bce_with_logits_loss ───────────────────────────────────────────────

    /// Stable host reference: `max(x,0) - x*z + log(1 + exp(-|x|))`.
    fn bce_scalar(x: f32, z: f32) -> f32 {
        x.max(0.0) - x * z + (1.0 + (-x.abs()).exp()).ln()
    }

    #[test]
    fn bce_with_logits_mean_at_zero_logit_equals_log2() {
        let p = storage(&[0.0f32], &[1, 1]);
        let z = storage(&[1.0f32], &[1, 1]);
        let out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Mean).unwrap();
        let v = read(&out)[0];
        let expected = bce_scalar(0.0, 1.0);
        assert!(
            (v - expected).abs() < 1e-4,
            "bce at x=0,z=1: got {v:.6}, expected {expected:.6}"
        );
    }

    #[test]
    fn bce_with_logits_sum_and_none_dispatch_correctly() {
        let p = storage(&[0.0f32, 1.0], &[1, 2]);
        let z = storage(&[1.0f32, 0.0], &[1, 2]);
        let mean_out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Mean).unwrap();
        let sum_out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Sum).unwrap();
        let none_out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::None).unwrap();
        assert_eq!(none_out.shape(), &[1, 2], "None preserves shape");
        let mean_v = read(&mean_out)[0];
        let sum_v = read(&sum_out)[0];
        assert!(
            (sum_v - 2.0 * mean_v).abs() < 1e-4,
            "bce sum should be 2*mean: sum={sum_v:.6}"
        );
    }

    #[test]
    fn bce_with_logits_finite_on_extreme_logits() {
        let p = storage(&[50.0f32, -50.0], &[1, 2]);
        let z = storage(&[0.0f32, 1.0], &[1, 2]);
        let out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Mean).unwrap();
        let v = read(&out)[0];
        assert!(v.is_finite(), "bce extreme logits should stay finite: {v}");
        assert!(
            (v - (bce_scalar(50.0, 0.0) + bce_scalar(-50.0, 1.0)) / 2.0).abs() < 1e-2,
            "bce extreme: got {v:.4}"
        );
    }

    #[test]
    fn bce_with_logits_zero_logit_backward_matches_sigmoid_minus_target() {
        for (reduction, scale) in [(Reduction::Sum, 1.0), (Reduction::Mean, 1.0 / 3.0)] {
            let p = storage(&[0.0f32, 0.0, 0.0], &[1, 3]);
            let z = storage(&[0.0f32, 1.0, 0.25], &[1, 3]);
            let (_, grads) =
                recorded(|| B::bce_with_logits_loss::<f32>(&p, &z, reduction).unwrap());
            let got = read(grads.get(p.id()).expect("pred should have a gradient"));
            for (actual, expected) in got.into_iter().zip([0.5f32, -0.5, 0.25]) {
                assert!(
                    (actual - expected * scale).abs() < 1e-6,
                    "{reduction:?}: got {actual}, expected {}",
                    expected * scale
                );
            }
        }
    }

    #[test]
    fn bce_with_logits_negative_tail_keeps_small_gradient() {
        let p = storage(&[-20.0f32], &[1, 1]);
        let z = storage(&[0.0f32], &[1, 1]);
        let (_, grads) =
            recorded(|| B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Sum).unwrap());
        let actual = read(grads.get(p.id()).unwrap())[0];
        let expected = (-20.0_f32).exp() / (1.0 + (-20.0_f32).exp());
        assert!((actual / expected - 1.0).abs() < 1e-5, "got {actual}");
    }

    #[test]
    fn bce_with_logits_backward_finite_on_extreme_logits() {
        let p = storage(&[50.0f32, -50.0], &[1, 2]);
        let z = storage(&[0.0f32, 1.0], &[1, 2]);
        let (_, grads) =
            recorded(|| B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Mean).unwrap());
        for (i, v) in read(grads.get(p.id()).expect("pred grad"))
            .iter()
            .enumerate()
        {
            assert!(
                v.is_finite(),
                "bce backward grad[{i}] should be finite: {v}"
            );
        }
    }

    // ── cross_entropy_loss ─────────────────────────────────────────────────

    /// Hand-compute expected log_softmax for a [2,3] logits row.
    fn log_softmax_row(row: &[f32]) -> Vec<f32> {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = row.iter().map(|x| x - max).collect();
        let sum_exp: f32 = shifted.iter().map(|x| x.exp()).sum();
        let log_sum_exp = sum_exp.ln();
        shifted.iter().map(|x| x - log_sum_exp).collect()
    }

    fn cross_pred() -> MetalStorage {
        storage(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
    }

    fn cross_target_0_2() -> MetalStorage {
        indices(&[0, 2], &[2])
    }

    #[test]
    fn cross_entropy_loss_mean_matches_hand_computed_nll() {
        let expected = {
            let rows = [[1.0f32, 2.0, 3.0], [4.0f32, 5.0, 6.0]];
            let targets = [0usize, 2];
            let n = rows.len() as f32;
            rows.iter()
                .zip(targets.iter())
                .map(|(row, &t)| -log_softmax_row(row)[t])
                .sum::<f32>()
                / n
        };
        let out =
            B::cross_entropy_loss(&cross_pred(), &cross_target_0_2(), Reduction::Mean).unwrap();
        assert_eq!(out.shape().len(), 0, "Mean output should be scalar");
        let got = read(&out)[0];
        assert!(
            (got - expected).abs() < 1e-4,
            "CE mean: got {got:.6}, expected {expected:.6}"
        );
    }

    #[test]
    fn cross_entropy_loss_sum_equals_batch_times_mean() {
        let mean_out =
            B::cross_entropy_loss(&cross_pred(), &cross_target_0_2(), Reduction::Mean).unwrap();
        let sum_out =
            B::cross_entropy_loss(&cross_pred(), &cross_target_0_2(), Reduction::Sum).unwrap();
        let mean_v = read(&mean_out)[0];
        let sum_v = read(&sum_out)[0];
        assert!(
            (sum_v - 2.0 * mean_v).abs() < 1e-4,
            "CE sum should be 2*mean: sum={sum_v:.6}"
        );
    }

    #[test]
    fn cross_entropy_loss_none_produces_per_sample_nll_vector() {
        let out =
            B::cross_entropy_loss(&cross_pred(), &cross_target_0_2(), Reduction::None).unwrap();
        assert_eq!(out.shape(), &[2], "None output should be [Batch]");
        let vals = read(&out);
        let exp0 = -log_softmax_row(&[1.0f32, 2.0, 3.0])[0];
        let exp1 = -log_softmax_row(&[4.0f32, 5.0, 6.0])[2];
        assert!(
            (vals[0] - exp0).abs() < 1e-4,
            "per-sample[0]: got {:.6}",
            vals[0]
        );
        assert!(
            (vals[1] - exp1).abs() < 1e-4,
            "per-sample[1]: got {:.6}",
            vals[1]
        );
    }

    #[test]
    fn cross_entropy_loss_backward_matches_softmax_minus_one_hot() {
        // probabilities p, logits = ln(p); grad = softmax - one_hot, scaled
        // by the reduction (CPU's analytic check).
        let probabilities = [0.2f32, 0.3, 0.5, 0.6, 0.3, 0.1];
        let logits = storage(
            &probabilities.iter().map(|p| p.ln()).collect::<Vec<_>>(),
            &[2, 3],
        );
        let target = indices(&[2, 0], &[2]);
        for (reduction, scale) in [
            (Reduction::None, 1.0f32),
            (Reduction::Sum, 1.0),
            (Reduction::Mean, 0.5),
        ] {
            let (_, grads) = recorded(|| {
                let out = B::cross_entropy_loss(&logits, &target, reduction).unwrap();
                if reduction == Reduction::None {
                    B::sum_all::<f32>(&out)
                } else {
                    Ok(out)
                }
                .unwrap()
            });
            let g = read(grads.get(logits.id()).expect("logits grad"));
            let expected = [
                0.2, 0.3, -0.5, // row 0: softmax([.2,.3,.5]) - one_hot[2]
                -0.4, 0.3, 0.1, // row 1: softmax([.6,.3,.1]) - one_hot[0]
            ];
            assert_close(&g, &expected.map(|e| e * scale), 1e-5);
            assert!(
                grads.get(target.id()).is_none(),
                "the integer target must stay off the tape"
            );
        }
    }

    #[test]
    fn cross_entropy_loss_refuses_a_non_rank2_logits_tensor() {
        let flat = storage(&[1.0, 2.0, 3.0], &[3]);
        let t = indices(&[0], &[1]);
        let err = B::cross_entropy_loss(&flat, &t, Reduction::Mean).unwrap_err();
        assert!(
            matches!(err, Error::Shape(ShapeError::RankMismatch { .. })),
            "logits must be rank-2, got {err:?}"
        );
    }

    #[test]
    fn cross_entropy_loss_refuses_a_mismatched_batch() {
        let logits = cross_pred(); // batch 2
        let t = indices(&[0], &[1]); // batch 1
        let err = B::cross_entropy_loss(&logits, &t, Reduction::Mean).unwrap_err();
        assert!(
            matches!(err, Error::Shape(ShapeError::DimensionMismatch { .. })),
            "target batch must match logits, got {err:?}"
        );
    }

    #[test]
    fn cross_entropy_loss_refuses_a_float_target() {
        let logits = cross_pred();
        let t = storage(&[0.0, 2.0], &[2]);
        let err = B::cross_entropy_loss(&logits, &t, Reduction::Mean).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedDType { .. }),
            "a float target has no honest class address, got {err:?}"
        );
    }

    #[test]
    fn cross_entropy_loss_refuses_an_out_of_range_class_by_name() {
        let logits = cross_pred();
        let t = indices(&[3], &[1]);
        let err = B::cross_entropy_loss(&logits, &t, Reduction::Mean).unwrap_err();
        assert!(
            matches!(err, Error::Shape(_)),
            "an out-of-range class must fail closed, got {err:?}"
        );
    }

    #[test]
    fn cross_entropy_loss_uniform_logits_equal_log_num_classes() {
        let logits = storage(&[5.0f32; 6], &[2, 3]);
        let t = indices(&[0, 1], &[2]);
        let out = B::cross_entropy_loss(&logits, &t, Reduction::Mean).unwrap();
        let got = read(&out)[0];
        let expected = 3.0f32.ln();
        assert!(
            (got - expected).abs() < 1e-4,
            "CE on uniform logits should be ln(3)={expected:.6}: got {got:.6}"
        );
    }

    // ── recording contract ─────────────────────────────────────────────────

    #[test]
    fn nograd_records_nothing() {
        let p = pred();
        let t = target();
        let logits = cross_pred();
        let ct = cross_target_0_2();
        let before = crate::metal::tape::depth();
        let _ = GradMode::Disabled.scope(|| B::mse_loss::<f32>(&p, &t, Reduction::Mean).unwrap());
        let _ = GradMode::Disabled.scope(|| B::l1_loss::<f32>(&p, &t, Reduction::Mean).unwrap());
        let _ = GradMode::Disabled
            .scope(|| B::bce_with_logits_loss::<f32>(&p, &t, Reduction::Mean).unwrap());
        let _ = GradMode::Disabled
            .scope(|| B::cross_entropy_loss(&logits, &ct, Reduction::Mean).unwrap());
        assert_eq!(
            crate::metal::tape::depth(),
            before,
            "NoGrad must record nothing"
        );
    }
}
