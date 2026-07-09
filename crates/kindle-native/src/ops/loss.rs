//! `LossOps` for `NativeBackend<T, D>`.
//!
//! `mse_loss` is composed strictly from already-tape-tracked primitives
//! (`NumericOps::sub`, `NumericOps::mul`, `ReductionOps::mean_all` /
//! `ReductionOps::sum_all`) — it does NOT implement a hand-derived fused
//! backward kernel. Because each primitive already pushes its own
//! `TapeEntry`, the backward gradient through MSE is automatically correct
//! by composition without any additional code here (T-01-17 mitigation).
//!
//! `l1_loss`, `bce_with_logits_loss`, and `cross_entropy_loss` are
//! `unimplemented!()` stubs matching `CandleBackend`'s exact convention for
//! the same three methods — they are out of Phase 1 scope per the Minimal
//! Phase 1 Op Set table and are not reachable by `Linear::forward` +
//! `mse_loss`'s actual call graph.

use kindle_core::nn::Reduction;
use kindle_core::prelude::{Backend, DType, LossOps, NumericOps, ReductionOps, Result};

use crate::NativeBackend;

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

    fn l1_loss<K: DType>(
        _pred: &<Self as Backend>::Storage<K>,
        _target: &<Self as Backend>::Storage<K>,
        _reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("l1_loss not implemented for NativeBackend")
    }

    fn bce_with_logits_loss<K: DType>(
        _pred: &<Self as Backend>::Storage<K>,
        _target: &<Self as Backend>::Storage<K>,
        _reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("bce_with_logits_loss not implemented for NativeBackend")
    }

    fn cross_entropy_loss<K: DType, KInt: DType>(
        _pred: &<Self as Backend>::Storage<K>,
        _target: &<Self as Backend>::Storage<K>,
        _reduction: Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("cross_entropy_loss not implemented for NativeBackend")
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

    type B = NativeBackend<f32, kindle_core::prelude::Cpu>;

    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![rows, cols])
    }

    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    // pred = [[1, 2, 3], [4, 5, 6]], target = [[1, 1, 1], [2, 2, 2]]
    // diff = [[0, 1, 2], [2, 3, 4]]
    // sq   = [[0, 1, 4], [4, 9, 16]]
    // sum  = 34,  mean = 34/6 ≈ 5.666…

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
        // 34 / 6 ≈ 5.6667
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
        // diff^2 = [[0, 1, 4], [4, 9, 16]]
        let got = f32_vec(&out);
        let expected = vec![0.0f32, 1.0, 4.0, 4.0, 9.0, 16.0];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "mse none: got {g}, expected {e}");
        }
    }

    #[test]
    fn mse_loss_mean_backward_matches_analytic_formula_2_times_pred_minus_target_over_n() {
        // Analytic MSE gradient w.r.t. pred: d(MSE)/d(pred_i) = 2*(pred_i - target_i)/n
        // For our example with n=6:
        // diff = [[0, 1, 2], [2, 3, 4]]
        // grad = 2 * diff / 6 = [[0, 1/3, 2/3], [2/3, 1, 4/3]]
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

    // --- #[should_panic] tests for the three stub methods ---

    #[test]
    #[should_panic(expected = "l1_loss not implemented for NativeBackend")]
    fn l1_loss_panics_with_expected_message() {
        let _ = B::l1_loss::<f32>(&pred(), &target(), Reduction::Mean);
    }

    #[test]
    #[should_panic(expected = "bce_with_logits_loss not implemented for NativeBackend")]
    fn bce_with_logits_loss_panics_with_expected_message() {
        let _ = B::bce_with_logits_loss::<f32>(&pred(), &target(), Reduction::Mean);
    }

    #[test]
    #[should_panic(expected = "cross_entropy_loss not implemented for NativeBackend")]
    fn cross_entropy_loss_panics_with_expected_message() {
        let _ = B::cross_entropy_loss::<f32, i64>(&pred(), &target(), Reduction::Mean);
    }
}
