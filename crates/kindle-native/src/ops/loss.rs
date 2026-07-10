//! `LossOps` for `NativeBackend<T, D>`.
//!
//! `mse_loss` is composed strictly from already-tape-tracked primitives
//! (`NumericOps::sub`, `NumericOps::mul`, `ReductionOps::mean_all` /
//! `ReductionOps::sum_all`) — it does NOT implement a hand-derived fused
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

use kindle_core::nn::Reduction;
use kindle_core::prelude::{Backend, DType, FloatOps, LossOps, NumericOps, ReductionOps, Result};

use crate::NativeBackend;
use crate::storage::{NativeBuffer, NativeStorage};

impl<T: DType, D: kindle_core::prelude::Device> LossOps<Self> for NativeBackend<T, D> {
    /// Mean (or sum, or elementwise) squared error, composed from
    /// already-tape-tracked `sub` / `mul` / `mean_all` / `sum_all`
    /// primitives. The backward chain is correct by composition — no
    /// separate fused backward formula is needed or used.
    fn mse_loss<K: DType>(
        pred: &<Self as Backend>::Storage<K>,
        target: &<Self as Backend>::Storage<K>,
        reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // diff = pred - target  (tape-tracked by NumericOps::sub)
        let diff = <Self as NumericOps<Self>>::sub::<K>(pred, target)?;
        // sq = diff * diff  (tape-tracked by NumericOps::mul; captures diff's values)
        let sq = <Self as NumericOps<Self>>::mul::<K>(&diff, &diff)?;
        match reduction {
            Reduction::Mean => <Self as ReductionOps<Self>>::mean_all::<K>(&sq),
            Reduction::Sum => <Self as ReductionOps<Self>>::sum_all::<K>(&sq),
            Reduction::None => Ok(sq),
        }
    }

    /// L1 loss (mean absolute error), composed from already-tape-tracked
    /// `sub` / `abs` + `mean_all` / `sum_all` primitives, mirroring `mse_loss`'s
    /// structure with `abs` substituted for `mul(diff, diff)`. Zero new
    /// backward code needed — both `sub` and `abs` already push correct
    /// `TapeEntry` closures (Phase 1/2).
    fn l1_loss<K: DType>(
        pred: &<Self as Backend>::Storage<K>,
        target: &<Self as Backend>::Storage<K>,
        reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let diff = <Self as NumericOps<Self>>::sub::<K>(pred, target)?;
        let abs_diff = <Self as FloatOps<Self>>::abs::<K>(&diff)?;
        match reduction {
            Reduction::Mean => <Self as ReductionOps<Self>>::mean_all::<K>(&abs_diff),
            Reduction::Sum => <Self as ReductionOps<Self>>::sum_all::<K>(&abs_diff),
            Reduction::None => Ok(abs_diff),
        }
    }

    /// Binary cross-entropy with logits, using the numerically-stable formula:
    /// `loss = max(x, 0) - x * z + log(1 + exp(-|x|))`
    /// (where `x` = logit prediction, `z` = target in [0,1]).
    ///
    /// This is NOT Candle 0.9.1's own `binary_cross_entropy_with_logit`, which
    /// uses the naive `sigmoid(x)` + `log` form that overflows to `-inf` on
    /// large positive logits (RESEARCH.md Pitfall 1). `NativeBackend` exceeds
    /// `CandleBackend`'s coverage here by implementing the stable formula
    /// directly, composed entirely from already-tape-tracked Phase 2 primitives
    /// (`relu`/`mul`/`sub`/`abs`/`neg`/`exp`/`add_scalar_float`/`log`/`add`)
    /// — zero hand-derived backward (Plan 04-02, ROADMAP.md Phase 4 criterion 2).
    fn bce_with_logits_loss<K: DType>(
        pred: &<Self as Backend>::Storage<K>,
        target: &<Self as Backend>::Storage<K>,
        reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // term1 = relu(x) - x * z   [= max(x,0) - x*z]
        let max_x_0 = <Self as FloatOps<Self>>::relu::<K>(pred)?;
        let x_times_z = <Self as NumericOps<Self>>::mul::<K>(pred, target)?;
        let term1 = <Self as NumericOps<Self>>::sub::<K>(&max_x_0, &x_times_z)?;

        // term2 = log(1 + exp(-|x|))  — numerically stable: |x| bounded
        // so exp(-|x|) is in (0, 1], never overflowing.
        let abs_x = <Self as FloatOps<Self>>::abs::<K>(pred)?;
        let neg_abs_x = <Self as FloatOps<Self>>::neg::<K>(&abs_x)?;
        let exp_neg_abs_x = <Self as FloatOps<Self>>::exp::<K>(&neg_abs_x)?;
        let one_plus = <Self as FloatOps<Self>>::add_scalar_float::<K>(&exp_neg_abs_x, 1.0)?;
        let term2 = <Self as FloatOps<Self>>::log::<K>(&one_plus)?;

        // elementwise BCE loss = term1 + term2
        let loss_elem = <Self as NumericOps<Self>>::add::<K>(&term1, &term2)?;

        match reduction {
            Reduction::Mean => <Self as ReductionOps<Self>>::mean_all::<K>(&loss_elem),
            Reduction::Sum => <Self as ReductionOps<Self>>::sum_all::<K>(&loss_elem),
            Reduction::None => Ok(loss_elem),
        }
    }

    /// Numerically-stable cross-entropy loss via the shared `log_softmax`
    /// kernel (D-02, Plan 04-01).
    ///
    /// `pred`   — logit matrix, shape `[Batch, Classes]`
    /// `target` — integer class indices held as `f64` in an
    ///            `I64`-typed `NativeStorage`, shape `[Batch]`.
    ///            Each value `target[b]` is read as `… as i64 as usize`
    ///            (matching the existing codebase convention from
    ///            `argmax`/`argmin`/`embedding` — Pitfall 8).
    ///
    /// Algorithm:
    /// 1. `log_probs = log_softmax(pred, class_dim=1)` — tape-tracked via the
    ///    shared kernel (max_keepdim/sub/exp/sum_keepdim/log/sub composition).
    /// 2. Build a `one_hot` constant buffer (same shape as `pred`) with `1.0`
    ///    at `[b, target[b]]`, `0.0` elsewhere.  Not tape-tracked (it is a
    ///    constant w.r.t. the gradient, derived from integer labels).
    /// 3. `picked  = log_probs * one_hot` — tape-tracked `mul`.
    /// 4. `per_nll = -sum_dim(picked, 1)` — shape `[Batch]`.
    /// 5. Dispatch on `reduction`: mean / sum / none.
    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &<Self as Backend>::Storage<K>,
        target: &<Self as Backend>::Storage<KInt>,
        reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        debug_assert_eq!(
            pred.shape.len(),
            2,
            "cross_entropy_loss: pred must be 2-D [Batch, Classes], got {:?}",
            pred.shape
        );
        let batch = pred.shape[0];
        let classes = pred.shape[1];

        // Step 1: log_softmax over the class dimension (axis 1).
        let log_probs = crate::ops::elementwise::log_softmax::<T, D, K>(pred, 1)?;

        // Step 2: Build one-hot constant. Read target indices as i64→usize
        // (Pitfall 8 convention). No tape entry — it's a constant.
        let mut one_hot_buf = vec![0.0f32; batch * classes];
        for b in 0..batch {
            let class_idx = target.get(&[b]) as i64 as usize;
            debug_assert!(
                class_idx < classes,
                "cross_entropy_loss: target[{b}]={class_idx} out of range [0,{classes})"
            );
            one_hot_buf[b * classes + class_idx] = 1.0;
        }
        // Transmit one_hot as the same dtype K by reading through f32 bytes.
        // Since K is the float dtype of pred (typically f32), and NativeBuffer::F32
        // is what get() always reads as f64, we build it as F32 and let the
        // type system treat it as K-typed (NativeStorage is untyped at the
        // buffer level — `get()` always returns f64 regardless of K).
        let one_hot =
            NativeStorage::from_contiguous(NativeBuffer::F32(one_hot_buf), vec![batch, classes]);

        // Step 3: tape-tracked mul — gradient flows through log_probs here.
        let picked = <Self as NumericOps<Self>>::mul::<K>(&log_probs, &one_hot)?;

        // Step 4: sum over class axis → shape [Batch], then negate.
        let sum_picked = <Self as ReductionOps<Self>>::sum_dim::<K>(&picked, 1)?;
        let per_nll = <Self as FloatOps<Self>>::neg::<K>(&sum_picked)?;

        // Step 5: reduce per Reduction.
        match reduction {
            Reduction::Mean => <Self as ReductionOps<Self>>::mean_all::<K>(&per_nll),
            Reduction::Sum => <Self as ReductionOps<Self>>::sum_all::<K>(&per_nll),
            Reduction::None => Ok(per_nll),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{NativeBuffer, NativeStorage};
    use crate::tape;
    use crate::testutil::gradcheck;

    type B = NativeBackend<f32, kindle_core::prelude::Cpu>;

    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![rows, cols])
    }

    fn vector_i64(v: Vec<i64>) -> NativeStorage {
        let n = v.len();
        let floats: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        NativeStorage::from_contiguous(NativeBuffer::F32(floats), vec![n])
    }

    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    fn pred() -> NativeStorage {
        matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    }

    fn target() -> NativeStorage {
        matrix(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 2, 3)
    }

    #[test]
    fn mse_loss_mean_produces_correct_scalar() {
        let out = B::mse_loss::<f32>(&pred(), &target(), Reduction::Mean).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new()); // scalar
        let v = out.get(&[]);
        assert!((v - 34.0 / 6.0).abs() < 1e-4, "mse mean: got {v}");
    }

    #[test]
    fn mse_loss_sum_produces_correct_scalar() {
        let out = B::mse_loss::<f32>(&pred(), &target(), Reduction::Sum).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
        let v = out.get(&[]);
        assert!((v - 34.0).abs() < 1e-4, "mse sum: got {v}");
    }

    #[test]
    fn mse_loss_none_produces_elementwise_squared_diff() {
        let out = B::mse_loss::<f32>(&pred(), &target(), Reduction::None).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        let got = f32_vec(&out);
        let expected = vec![0.0f32, 1.0, 4.0, 4.0, 9.0, 16.0];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "mse none: got {g}, expected {e}");
        }
    }

    #[test]
    fn mse_loss_mean_backward_matches_analytic_formula_2_times_pred_minus_target_over_n() {
        let p = pred();
        let t = target();
        let pred_id = p.id;
        let out = B::mse_loss::<f32>(&p, &t, Reduction::Mean).unwrap();
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
    fn l1_loss_mean_produces_correct_scalar() {
        let out = B::l1_loss::<f32>(&pred(), &target(), Reduction::Mean).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
        let v = out.get(&[]) as f32;
        assert!((v - 2.0).abs() < 1e-4, "l1 mean: got {v:.6}");
    }

    #[test]
    fn l1_loss_sum_produces_correct_scalar() {
        let out = B::l1_loss::<f32>(&pred(), &target(), Reduction::Sum).unwrap();
        assert_eq!(out.shape, Vec::<usize>::new());
        let v = out.get(&[]) as f32;
        assert!((v - 12.0).abs() < 1e-4, "l1 sum: got {v:.6}");
    }

    #[test]
    fn l1_loss_none_produces_elementwise_absolute_diff() {
        let out = B::l1_loss::<f32>(&pred(), &target(), Reduction::None).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        let got = f32_vec(&out);
        let expected = vec![0.0f32, 1.0, 2.0, 2.0, 3.0, 4.0];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "l1 none[{i}]: got {g}, expected {e}");
        }
    }

    #[test]
    fn l1_loss_gradcheck() {
        let p = matrix(vec![1.0f32, 2.0, 3.0], 1, 3);
        let t = matrix(vec![0.5f32, 0.5, 0.5], 1, 3);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            B::l1_loss::<f32>(&inputs[0], &t, Reduction::Mean).unwrap()
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

    fn bce_scalar(x: f32, z: f32) -> f32 {
        // Stable formula: max(x,0) - x*z + log(1 + exp(-|x|))
        x.max(0.0) - x * z + (1.0 + (-x.abs()).exp()).ln()
    }

    #[test]
    fn bce_with_logits_mean_at_zero_logit_equals_log2() {
        // x=0, z=1 → max(0,0) - 0*1 + log(1+exp(0)) = 0 + log(2) ≈ 0.6931
        let pred_zero = matrix(vec![0.0f32], 1, 1);
        let tgt_one = matrix(vec![1.0f32], 1, 1);
        let out = B::bce_with_logits_loss::<f32>(&pred_zero, &tgt_one, Reduction::Mean).unwrap();
        let v = out.get(&[]) as f32;
        let expected = bce_scalar(0.0, 1.0);
        assert!(
            (v - expected).abs() < 1e-4,
            "bce at x=0,z=1: got {v:.6}, expected {expected:.6} (≈ ln(2))"
        );
    }

    #[test]
    fn bce_with_logits_sum_and_none_dispatch_correctly() {
        // Two-element test to verify Sum == 2*element and None preserves shape.
        let p = matrix(vec![0.0f32, 1.0], 1, 2);
        let z = matrix(vec![1.0f32, 0.0], 1, 2);
        let mean_out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Mean).unwrap();
        let sum_out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Sum).unwrap();
        let none_out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::None).unwrap();

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
    fn bce_with_logits_finite_on_large_positive_logit() {
        // Naive sigmoid+log form: log(1 - sigmoid(50)) = log(~0) = -inf.
        // Stable formula: max(50,0) - 50*0 + log(1+exp(-50)) ≈ 50 + ~0 (finite).
        let p = matrix(vec![50.0f32], 1, 1);
        let z = matrix(vec![0.0f32], 1, 1);
        let out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Mean).unwrap();
        let v = out.get(&[]) as f32;
        assert!(v.is_finite(), "bce on x=50,z=0 should be finite: {v}");
        let expected = bce_scalar(50.0, 0.0);
        assert!(
            (v - expected).abs() < 1e-2,
            "bce x=50,z=0: got {v:.4}, expected {expected:.4}"
        );
    }

    #[test]
    fn bce_with_logits_finite_on_large_negative_logit() {
        // Naive sigmoid+log form: log(sigmoid(-50)) = log(~0) = -inf.
        // Stable formula: max(-50,0) - (-50)*1 + log(1+exp(-50)) ≈ 0 + 50 + ~0.
        let p = matrix(vec![-50.0f32], 1, 1);
        let z = matrix(vec![1.0f32], 1, 1);
        let out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Mean).unwrap();
        let v = out.get(&[]) as f32;
        assert!(v.is_finite(), "bce on x=-50,z=1 should be finite: {v}");
        let expected = bce_scalar(-50.0, 1.0);
        assert!(
            (v - expected).abs() < 1e-2,
            "bce x=-50,z=1: got {v:.4}, expected {expected:.4}"
        );
    }

    #[test]
    fn bce_with_logits_gradcheck() {
        let p = matrix(vec![0.5f32, -0.3, 1.2], 1, 3);
        let z = matrix(vec![1.0f32, 0.0, 1.0], 1, 3);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            B::bce_with_logits_loss::<f32>(&inputs[0], &z, Reduction::Mean).unwrap()
        };
        let max_rel_err = gradcheck(op, &[p], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "bce gradcheck too high: {max_rel_err:.6}"
        );
    }

    #[test]
    fn bce_with_logits_backward_finite_on_extreme_logits() {
        // Both forward and backward should be finite on large-magnitude logits.
        let p = matrix(vec![50.0f32, -50.0], 1, 2);
        let z = matrix(vec![0.0f32, 1.0], 1, 2);
        let out = B::bce_with_logits_loss::<f32>(&p, &z, Reduction::Mean).unwrap();
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
    fn cross_pred() -> NativeStorage {
        matrix(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    }

    /// target = [0, 2] as I64-typed (class 0 for sample 0, class 2 for sample 1)
    fn cross_target_0_2() -> NativeStorage {
        vector_i64(vec![0, 2])
    }

    fn log_softmax_row(row: &[f32]) -> Vec<f32> {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let shifted: Vec<f32> = row.iter().map(|x| x - max).collect();
        let sum_exp: f32 = shifted.iter().map(|x| x.exp()).sum();
        let log_sum_exp = sum_exp.ln();
        shifted.iter().map(|x| x - log_sum_exp).collect()
    }

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
    fn cross_entropy_loss_mean_matches_hand_computed_nll() {
        let pred_row0 = [1.0f32, 2.0, 3.0];
        let pred_row1 = [4.0f32, 5.0, 6.0];
        let expected = expected_ce_mean(&[&pred_row0, &pred_row1], &[0, 2]);

        let out =
            B::cross_entropy_loss::<f32, i64>(&cross_pred(), &cross_target_0_2(), Reduction::Mean)
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
    fn cross_entropy_loss_sum_equals_batch_times_mean() {
        let mean_out =
            B::cross_entropy_loss::<f32, i64>(&cross_pred(), &cross_target_0_2(), Reduction::Mean)
                .unwrap();
        let sum_out =
            B::cross_entropy_loss::<f32, i64>(&cross_pred(), &cross_target_0_2(), Reduction::Sum)
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
    fn cross_entropy_loss_none_produces_per_sample_nll_vector() {
        let out =
            B::cross_entropy_loss::<f32, i64>(&cross_pred(), &cross_target_0_2(), Reduction::None)
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
    fn cross_entropy_loss_gradcheck() {
        let tgt = cross_target_0_2();
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            B::cross_entropy_loss::<f32, i64>(&inputs[0], &tgt, Reduction::Mean).unwrap()
        };
        let max_rel_err = gradcheck(op, &[cross_pred()], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "cross_entropy_loss gradcheck error too high: {max_rel_err:.6}"
        );
    }

    #[test]
    fn cross_entropy_loss_finite_on_extreme_logits() {
        let pred_extreme = matrix(vec![1000.0f32, -1000.0, 0.0, -1000.0, 1000.0, 0.0], 2, 3);
        let target = vector_i64(vec![0, 1]);
        let out =
            B::cross_entropy_loss::<f32, i64>(&pred_extreme, &target, Reduction::Mean).unwrap();
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
    fn cross_entropy_loss_uniform_logits_equal_log_num_classes() {
        let pred_uniform = matrix(vec![5.0f32, 5.0, 5.0, 5.0, 5.0, 5.0], 2, 3);
        let target = vector_i64(vec![0, 1]);
        let out =
            B::cross_entropy_loss::<f32, i64>(&pred_uniform, &target, Reduction::Mean).unwrap();
        let loss_val = out.get(&[]) as f32;
        let expected = 3.0f32.ln();
        assert!(
            (loss_val - expected).abs() < 1e-4,
            "CE on uniform logits should be ln(3)={expected:.6}: got {loss_val:.6}"
        );
    }
}
