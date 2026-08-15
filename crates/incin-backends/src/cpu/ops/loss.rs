//! Canonical CPU loss helpers for `CpuBackendImpl<D>`.
//!
//! `mse_loss` is composed strictly from already-tape-tracked primitives
//! (elementwise subtraction/multiplication and reduction helpers) — it does
//! NOT implement a hand-derived fused
//! backward kernel. Because each primitive already pushes its own
//! `TapeEntry`, the backward gradient through MSE is automatically correct
//! by composition without any additional code here (T-01-17 mitigation).
//!
//! `cross_entropy_loss` (Plan 04-01 Task 2): numerically-stable implementation
//! using the shared `log_softmax` kernel (D-02) plus a one-hot-multiply
//! target gather.  The one-hot buffer is a constant w.r.t. the loss (it holds
//! integer-derived zeros/ones, not a differentiable value), so it does NOT
//! need a tape entry.  The gradient flows through `log_probs` → `mul` →
//! `sum_dim` → `neg` → Reduction dispatch, all already-tape-tracked, so no
//! hand-derived backward is written here.
//!
//! `l1_loss` (mean absolute error) and `bce_with_logits_loss` (numerically-stable
//! binary cross-entropy) are both real, composed implementations landed in
//! Phase 4 Plan 04-02. `l1_loss` composes from `sub`/`abs`/`mean_all`/`sum_all`;
//! `bce_with_logits_loss` composes from the numerically-stable
//! `max(x,0) - x*z + log(1+exp(-|x|))` formula (`relu`/`mul`/`sub`/`abs`/`neg`/
//! `exp`/`add_scalar_float`/`log`/`add`/`mean_all`/`sum_all`). Both inherit
//! correct backward by composition with zero new tape entries of their own,
//! exactly like `mse_loss`/`cross_entropy_loss` above.

#[cfg(test)]
use crate::cpu::CpuBackendImpl;
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use incin_core::error::{ConversionFailure, Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::shapes::{Axis, DimensionConstraint, RankExpectation, ShapeBuf, ShapeError};
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::{DType, DTypeId};
use incin_core::tensor::reduction::Reduction;

fn reduce_loss(t: CpuStorage, reduction: Reduction) -> Result<CpuStorage> {
    match reduction {
        Reduction::Mean => crate::cpu::ops::reduce::mean_all(&t),
        Reduction::Sum => crate::cpu::ops::reduce::sum_all(&t),
        Reduction::None => Ok(t),
    }
}

pub(crate) fn mse_loss_storage(
    pred: &CpuStorage,
    target: &CpuStorage,
    reduction: Reduction,
) -> Result<CpuStorage> {
    let diff = crate::cpu::ops::elementwise::sub_storage(pred, target)?;
    let squared = crate::cpu::ops::elementwise::mul_storage(&diff, &diff)?;
    reduce_loss(squared, reduction)
}

pub(crate) fn l1_loss_storage(
    pred: &CpuStorage,
    target: &CpuStorage,
    reduction: Reduction,
) -> Result<CpuStorage> {
    let diff = crate::cpu::ops::elementwise::sub_storage(pred, target)?;
    let absolute = crate::cpu::ops::elementwise::canonical_abs(&diff)?;
    reduce_loss(absolute, reduction)
}

pub(crate) fn bce_with_logits_loss_storage(
    pred: &CpuStorage,
    target: &CpuStorage,
    reduction: Reduction,
) -> Result<CpuStorage> {
    let max_x_0 = crate::cpu::ops::elementwise::canonical_relu(pred)?;
    let x_times_z = crate::cpu::ops::elementwise::mul_storage(pred, target)?;
    let term1 = crate::cpu::ops::elementwise::sub_storage(&max_x_0, &x_times_z)?;
    let abs_x = crate::cpu::ops::elementwise::canonical_abs(pred)?;
    let neg_abs_x = crate::cpu::ops::elementwise::canonical_neg(&abs_x)?;
    let exp_neg_abs_x = crate::cpu::ops::elementwise::canonical_exp(&neg_abs_x)?;
    let one_plus = crate::cpu::ops::elementwise::canonical_add_scalar(&exp_neg_abs_x, 1.0)?;
    let term2 = crate::cpu::ops::elementwise::canonical_log(&one_plus)?;
    let loss = crate::cpu::ops::elementwise::add_storage(&term1, &term2)?;
    reduce_loss(loss, reduction)
}

pub(crate) fn cross_entropy_loss_storage<D: Device>(
    pred: &CpuStorage,
    target: &CpuStorage,
    reduction: Reduction,
) -> Result<CpuStorage> {
    if pred.shape.len() != 2 {
        return Err(ShapeError::RankMismatch {
            operation: OperationKind::Reduction,
            expected: RankExpectation::Exactly(2),
            actual: pred.shape.len(),
        }
        .into());
    }
    let batch = pred.shape[0];
    let classes = pred.shape[1];
    if target.shape.as_ref() != [batch] {
        return Err(ShapeError::DimensionMismatch {
            operation: OperationKind::Reduction,
            axis: Axis::Index(0),
            lhs: batch,
            rhs: target.shape.first().copied().unwrap_or(0),
            constraint: DimensionConstraint::Equal,
        }
        .into());
    }
    let log_probs = crate::cpu::ops::elementwise::log_softmax::<D, f32>(pred, 1)?;
    let one_hot_total =
        ShapeBuf::from_slice(&[batch, classes]).checked_numel(OperationKind::Storage)?;
    let mut one_hot_buf = vec![0.0f32; one_hot_total];
    for batch_index in 0..batch {
        let class_index = target.get_i64_checked(&[batch_index], "cross_entropy_target")?;
        let class_index = usize::try_from(class_index).map_err(|_| Error::InvalidConversion {
            operation: "cross_entropy_target",
            from: DTypeId::I64.descriptor(),
            to: DTypeId::U32.descriptor(),
            reason: ConversionFailure::OutOfRange,
        })?;
        if class_index >= classes {
            return Err(ShapeError::InvalidParameter {
                operation: OperationKind::Reduction,
                parameter: "target class index",
                value: class_index,
            }
            .into());
        }
        one_hot_buf[batch_index * classes + class_index] = 1.0;
    }
    let one_hot = CpuStorage::from_contiguous(CpuBuffer::F32(one_hot_buf), vec![batch, classes]);
    let picked = crate::cpu::ops::elementwise::mul_storage(&log_probs, &one_hot)?;
    let summed = crate::cpu::ops::reduce::sum_dim(&picked, 1)?;
    let per_nll = crate::cpu::ops::elementwise::canonical_neg(&summed)?;
    reduce_loss(per_nll, reduction)
}

impl<D: Device> crate::cpu::CpuBackendImpl<D> {
    pub fn mse_loss<K: DType>(
        pred: &CpuStorage,
        target: &CpuStorage,
        reduction: Reduction,
    ) -> Result<CpuStorage> {
        let _ = core::marker::PhantomData::<K>;
        mse_loss_storage(pred, target, reduction)
    }

    pub fn l1_loss<K: DType>(
        pred: &CpuStorage,
        target: &CpuStorage,
        reduction: Reduction,
    ) -> Result<CpuStorage> {
        let _ = core::marker::PhantomData::<K>;
        l1_loss_storage(pred, target, reduction)
    }

    pub fn bce_with_logits_loss<K: DType>(
        pred: &CpuStorage,
        target: &CpuStorage,
        reduction: Reduction,
    ) -> Result<CpuStorage> {
        let _ = core::marker::PhantomData::<K>;
        bce_with_logits_loss_storage(pred, target, reduction)
    }

    pub fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &CpuStorage,
        target: &CpuStorage,
        reduction: Reduction,
    ) -> Result<CpuStorage> {
        let _ = core::marker::PhantomData::<(K, KInt)>;
        cross_entropy_loss_storage::<D>(pred, target, reduction)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::gradcheck::gradcheck;
    use crate::cpu::storage::{CpuBuffer, CpuStorage};
    use crate::cpu::tape;

    /// `B`.
    #[allow(dead_code)]
    type B = CpuBackendImpl<incin_core::tensor::device::Cpu>;

    /// `matrix`.
    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![rows, cols])
    }

    /// `vector_i64`.
    fn vector_i64(v: Vec<i64>) -> CpuStorage {
        let n = v.len();
        let floats: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        CpuStorage::from_contiguous(CpuBuffer::F32(floats), vec![n])
    }

    #[test]
    fn cross_entropy_rejects_fractional_and_out_of_range_targets() {
        let pred = matrix(vec![1.0, 2.0, 3.0], 1, 3);
        let fractional = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.5]), vec![1]);
        assert!(matches!(
            cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
                &pred,
                &fractional,
                Reduction::Mean
            ),
            Err(Error::InvalidConversion {
                operation: "cross_entropy_target",
                ..
            })
        ));

        let out_of_range = vector_i64(vec![3]);
        assert!(matches!(
            cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
                &pred,
                &out_of_range,
                Reduction::Mean
            ),
            Err(Error::Shape(ShapeError::InvalidParameter { .. }))
        ));
    }

    /// `f32_vec`.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    /// `pred`.
    fn pred() -> CpuStorage {
        matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    }

    /// `target`.
    fn target() -> CpuStorage {
        matrix(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 2, 3)
    }

    #[test]
    /// `mse_loss_mean_produces_correct_scalar`.
    fn mse_loss_mean_produces_correct_scalar() {
        let out = mse_loss_storage(&pred(), &target(), Reduction::Mean).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new()); // scalar
        let v = out.get(&[]);
        assert!((v - 34.0 / 6.0).abs() < 1e-4, "mse mean: got {v}");
    }

    #[test]
    /// `mse_loss_sum_produces_correct_scalar`.
    fn mse_loss_sum_produces_correct_scalar() {
        let out = mse_loss_storage(&pred(), &target(), Reduction::Sum).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
        let v = out.get(&[]);
        assert!((v - 34.0).abs() < 1e-4, "mse sum: got {v}");
    }

    #[test]
    /// `mse_loss_none_produces_elementwise_squared_diff`.
    fn mse_loss_none_produces_elementwise_squared_diff() {
        let out = mse_loss_storage(&pred(), &target(), Reduction::None).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        let got = f32_vec(&out);
        let expected = [0.0f32, 1.0, 4.0, 4.0, 9.0, 16.0];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "mse none: got {g}, expected {e}");
        }
    }

    #[test]
    /// `mse_loss_mean_backward_matches_analytic_formula_2_times_pred_minus_target_over_n`.
    fn mse_loss_mean_backward_matches_analytic_formula_2_times_pred_minus_target_over_n() {
        let p = pred();
        let t = target();
        let pred_id = p.id;
        let out = mse_loss_storage(&p, &t, Reduction::Mean).unwrap();
        let grads = tape::backward(&out).unwrap();
        let g = grads.get(pred_id).expect("pred should have a gradient");
        assert_eq!(g.shape, vec![2, 3]);

        let expected = [0.0f32, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0];
        let got = f32_vec(g);
        for (i, (&gv, &ev)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (gv - ev).abs() < 1e-4,
                "mse backward grad[{i}]: got {gv}, expected {ev}"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // l1_loss tests (Plan 04-02 Task 1)
    // ---------------------------------------------------------------------------

    // pred = [[1,2,3],[4,5,6]], target = [[1,1,1],[2,2,2]]
    // |diff| = [[0,1,2],[2,3,4]], sum = 12, mean = 2.0

    #[test]
    /// `l1_loss_mean_produces_correct_scalar`.
    fn l1_loss_mean_produces_correct_scalar() {
        let out = l1_loss_storage(&pred(), &target(), Reduction::Mean).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
        let v = out.get(&[]) as f32;
        assert!((v - 2.0).abs() < 1e-4, "l1 mean: got {v:.6}");
    }

    #[test]
    /// `l1_loss_sum_produces_correct_scalar`.
    fn l1_loss_sum_produces_correct_scalar() {
        let out = l1_loss_storage(&pred(), &target(), Reduction::Sum).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
        let v = out.get(&[]) as f32;
        assert!((v - 12.0).abs() < 1e-4, "l1 sum: got {v:.6}");
    }

    #[test]
    /// `l1_loss_none_produces_elementwise_absolute_diff`.
    fn l1_loss_none_produces_elementwise_absolute_diff() {
        let out = l1_loss_storage(&pred(), &target(), Reduction::None).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        let got = f32_vec(&out);
        let expected = [0.0f32, 1.0, 2.0, 2.0, 3.0, 4.0];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "l1 none[{i}]: got {g}, expected {e}");
        }
    }

    #[test]
    /// `l1_loss_gradcheck`.
    fn l1_loss_gradcheck() {
        let p = matrix(vec![1.0f32, 2.0, 3.0], 1, 3);
        let t = matrix(vec![0.5f32, 0.5, 0.5], 1, 3);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            l1_loss_storage(&inputs[0], &t, Reduction::Mean).unwrap()
        };
        let max_rel_err = gradcheck(op, &[p], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "l1 gradcheck too high: {max_rel_err:.6}"
        );
    }

    // ---------------------------------------------------------------------------
    // bce_with_logits_loss tests (Plan 04-02 Task 2)
    // ---------------------------------------------------------------------------

    /// `bce_scalar`.
    fn bce_scalar(x: f32, z: f32) -> f32 {
        // Stable formula: max(x,0) - x*z + log(1 + exp(-|x|))
        x.max(0.0) - x * z + (1.0 + (-x.abs()).exp()).ln()
    }

    #[test]
    /// `bce_with_logits_mean_at_zero_logit_equals_log2`.
    fn bce_with_logits_mean_at_zero_logit_equals_log2() {
        // x=0, z=1 → max(0,0) - 0*1 + log(1+exp(0)) = 0 + log(2) ≈ 0.6931
        let pred_zero = matrix(vec![0.0f32], 1, 1);
        let tgt_one = matrix(vec![1.0f32], 1, 1);
        let out = bce_with_logits_loss_storage(&pred_zero, &tgt_one, Reduction::Mean).unwrap();
        let v = out.get(&[]) as f32;
        let expected = bce_scalar(0.0, 1.0);
        assert!(
            (v - expected).abs() < 1e-4,
            "bce at x=0,z=1: got {v:.6}, expected {expected:.6} (≈ ln(2))"
        );
    }

    #[test]
    /// `bce_with_logits_sum_and_none_dispatch_correctly`.
    fn bce_with_logits_sum_and_none_dispatch_correctly() {
        // Two-element test to verify Sum == 2*element and None preserves shape.
        let p = matrix(vec![0.0f32, 1.0], 1, 2);
        let z = matrix(vec![1.0f32, 0.0], 1, 2);
        let mean_out = bce_with_logits_loss_storage(&p, &z, Reduction::Mean).unwrap();
        let sum_out = bce_with_logits_loss_storage(&p, &z, Reduction::Sum).unwrap();
        let none_out = bce_with_logits_loss_storage(&p, &z, Reduction::None).unwrap();

        assert_eq!(
            none_out.shape,
            vec![1, 2],
            "None should preserve shape [1,2]"
        );
        let mean_v = mean_out.get(&[]) as f32;
        let sum_v = sum_out.get(&[]) as f32;
        assert!(
            (sum_v - 2.0 * mean_v).abs() < 1e-4,
            "bce sum should be 2*mean: sum={sum_v:.6}, 2*mean={:.6}",
            2.0 * mean_v
        );
    }

    #[test]
    /// `bce_with_logits_finite_on_large_positive_logit`.
    fn bce_with_logits_finite_on_large_positive_logit() {
        // Naive sigmoid+log form: log(1 - sigmoid(50)) = log(~0) = -inf.
        // Stable formula: max(50,0) - 50*0 + log(1+exp(-50)) ≈ 50 + ~0 (finite).
        let p = matrix(vec![50.0f32], 1, 1);
        let z = matrix(vec![0.0f32], 1, 1);
        let out = bce_with_logits_loss_storage(&p, &z, Reduction::Mean).unwrap();
        let v = out.get(&[]) as f32;
        assert!(v.is_finite(), "bce on x=50,z=0 should be finite: {v}");
        let expected = bce_scalar(50.0, 0.0);
        assert!(
            (v - expected).abs() < 1e-2,
            "bce x=50,z=0: got {v:.4}, expected {expected:.4}"
        );
    }

    #[test]
    /// `bce_with_logits_finite_on_large_negative_logit`.
    fn bce_with_logits_finite_on_large_negative_logit() {
        // Naive sigmoid+log form: log(sigmoid(-50)) = log(~0) = -inf.
        // Stable formula: max(-50,0) - (-50)*1 + log(1+exp(-50)) ≈ 0 + 50 + ~0.
        let p = matrix(vec![-50.0f32], 1, 1);
        let z = matrix(vec![1.0f32], 1, 1);
        let out = bce_with_logits_loss_storage(&p, &z, Reduction::Mean).unwrap();
        let v = out.get(&[]) as f32;
        assert!(v.is_finite(), "bce on x=-50,z=1 should be finite: {v}");
        let expected = bce_scalar(-50.0, 1.0);
        assert!(
            (v - expected).abs() < 1e-2,
            "bce x=-50,z=1: got {v:.4}, expected {expected:.4}"
        );
    }

    #[test]
    /// `bce_with_logits_gradcheck`.
    fn bce_with_logits_gradcheck() {
        let p = matrix(vec![0.5f32, -0.3, 1.2], 1, 3);
        let z = matrix(vec![1.0f32, 0.0, 1.0], 1, 3);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            bce_with_logits_loss_storage(&inputs[0], &z, Reduction::Mean).unwrap()
        };
        let max_rel_err = gradcheck(op, &[p], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "bce gradcheck too high: {max_rel_err:.6}"
        );
    }

    #[test]
    /// `bce_with_logits_backward_finite_on_extreme_logits`.
    fn bce_with_logits_backward_finite_on_extreme_logits() {
        // Both forward and backward should be finite on large-magnitude logits.
        let p = matrix(vec![50.0f32, -50.0], 1, 2);
        let z = matrix(vec![0.0f32, 1.0], 1, 2);
        let out = bce_with_logits_loss_storage(&p, &z, Reduction::Mean).unwrap();
        assert!(
            out.get(&[]).is_finite(),
            "bce extreme logits: forward should be finite"
        );

        let grads = tape::backward(&out).unwrap();
        let g = grads.get(p.id).expect("pred should have gradient");
        for (i, v) in f32_vec(g).iter().enumerate() {
            assert!(
                v.is_finite(),
                "bce backward grad[{i}] should be finite on extreme logits: {v}"
            );
        }
    }
    // ---------------------------------------------------------------------------
    // cross_entropy_loss tests (Plan 04-01 Task 2)
    // ---------------------------------------------------------------------------

    /// Hand-compute expected log_softmax for a [2,3] pred.
    fn cross_pred() -> CpuStorage {
        matrix(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    }

    /// target = [0, 2] as I64-typed (class 0 for sample 0, class 2 for sample 1)
    fn cross_target_0_2() -> CpuStorage {
        vector_i64(vec![0, 2])
    }

    /// `log_softmax_row`.
    fn log_softmax_row(row: &[f32]) -> Vec<f32> {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = row.iter().map(|x| x - max).collect();
        let sum_exp: f32 = shifted.iter().map(|x| x.exp()).sum();
        let log_sum_exp = sum_exp.ln();
        shifted.iter().map(|x| x - log_sum_exp).collect()
    }

    /// `expected_ce_mean`.
    fn expected_ce_mean(pred_rows: &[&[f32]], targets: &[usize]) -> f32 {
        let n = pred_rows.len() as f32;
        pred_rows
            .iter()
            .zip(targets.iter())
            .map(|(row, &t)| {
                let ls = log_softmax_row(row);
                -ls[t]
            })
            .sum::<f32>()
            / n
    }

    #[test]
    /// `cross_entropy_loss_mean_matches_hand_computed_nll`.
    fn cross_entropy_loss_mean_matches_hand_computed_nll() {
        let pred_row0 = [1.0f32, 2.0, 3.0];
        let pred_row1 = [4.0f32, 5.0, 6.0];
        let expected = expected_ce_mean(&[&pred_row0, &pred_row1], &[0, 2]);

        let out = cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
            &cross_pred(),
            &cross_target_0_2(),
            Reduction::Mean,
        )
        .unwrap();
        assert_eq!(
            out.shape,
            Vec::<usize>::new(),
            "Mean output should be scalar"
        );
        let got = out.get(&[]) as f32;
        assert!(
            (got - expected).abs() < 1e-4,
            "CE mean: got {got:.6}, expected {expected:.6}"
        );
    }

    #[test]
    /// `cross_entropy_loss_sum_equals_batch_times_mean`.
    fn cross_entropy_loss_sum_equals_batch_times_mean() {
        let mean_out = cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
            &cross_pred(),
            &cross_target_0_2(),
            Reduction::Mean,
        )
        .unwrap();
        let sum_out = cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
            &cross_pred(),
            &cross_target_0_2(),
            Reduction::Sum,
        )
        .unwrap();

        let mean_val = mean_out.get(&[]) as f32;
        let sum_val = sum_out.get(&[]) as f32;
        assert!(
            (sum_val - 2.0 * mean_val).abs() < 1e-4,
            "CE sum should be 2 * mean: sum={sum_val:.6}, 2*mean={:.6}",
            2.0 * mean_val
        );
    }

    #[test]
    /// `cross_entropy_loss_none_produces_per_sample_nll_vector`.
    fn cross_entropy_loss_none_produces_per_sample_nll_vector() {
        let out = cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
            &cross_pred(),
            &cross_target_0_2(),
            Reduction::None,
        )
        .unwrap();
        assert_eq!(out.shape, vec![2], "None output should be [Batch]");
        let vals = f32_vec(&out);

        let pred_row0 = [1.0f32, 2.0, 3.0];
        let pred_row1 = [4.0f32, 5.0, 6.0];
        let exp0 = -log_softmax_row(&pred_row0)[0];
        let exp1 = -log_softmax_row(&pred_row1)[2];

        assert!(
            (vals[0] - exp0).abs() < 1e-4,
            "per-sample[0]: got {:.6}, expected {exp0:.6}",
            vals[0]
        );
        assert!(
            (vals[1] - exp1).abs() < 1e-4,
            "per-sample[1]: got {:.6}, expected {exp1:.6}",
            vals[1]
        );
    }

    #[test]
    /// `cross_entropy_loss_gradcheck`.
    fn cross_entropy_loss_gradcheck() {
        let tgt = cross_target_0_2();
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
                &inputs[0],
                &tgt,
                Reduction::Mean,
            )
            .unwrap()
        };
        let max_rel_err = gradcheck(op, &[cross_pred()], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "cross_entropy_loss gradcheck error too high: {max_rel_err:.6}"
        );
    }

    #[test]
    /// `cross_entropy_loss_finite_on_extreme_logits`.
    fn cross_entropy_loss_finite_on_extreme_logits() {
        let pred_extreme = matrix(vec![1000.0f32, -1000.0, 0.0, -1000.0, 1000.0, 0.0], 2, 3);
        let target = vector_i64(vec![0, 1]);
        let out = cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
            &pred_extreme,
            &target,
            Reduction::Mean,
        )
        .unwrap();
        let loss_val = out.get(&[]) as f32;
        assert!(
            loss_val.is_finite(),
            "CE loss should be finite on extreme logits: {loss_val}"
        );

        let grads = tape::backward(&out).unwrap();
        let g = grads
            .get(pred_extreme.id)
            .expect("pred should have gradient");
        for (i, v) in f32_vec(g).iter().enumerate() {
            assert!(
                v.is_finite(),
                "CE backward grad[{i}] should be finite on extreme logits: {v}"
            );
        }
    }

    #[test]
    /// `cross_entropy_loss_uniform_logits_equal_log_num_classes`.
    fn cross_entropy_loss_uniform_logits_equal_log_num_classes() {
        let pred_uniform = matrix(vec![5.0f32, 5.0, 5.0, 5.0, 5.0, 5.0], 2, 3);
        let target = vector_i64(vec![0, 1]);
        let out = cross_entropy_loss_storage::<incin_core::tensor::device::Cpu>(
            &pred_uniform,
            &target,
            Reduction::Mean,
        )
        .unwrap();
        let loss_val = out.get(&[]) as f32;
        let expected = 3.0f32.ln();
        assert!(
            (loss_val - expected).abs() < 1e-4,
            "CE on uniform logits should be ln(3)={expected:.6}: got {loss_val:.6}"
        );
    }
}
