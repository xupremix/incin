//! Loss operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::executor::CandleStorage;
use crate::external::*;

impl<D: incin_core::tensor::device::Device> CandleBackend<D> {
    /// Computes L1 (Mean Absolute Error) loss: `|pred - target|` with
    /// the given `reduction` (Mean, Sum, or None).
    pub fn l1_loss<K: incin_core::tensor::dtype::DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let diff = pred
            .tensor()
            .broadcast_sub(target.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let abs_diff = diff
            .abs()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let raw = match reduction {
            incin_core::tensor::reduction::Reduction::Mean => abs_diff
                .mean_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            incin_core::tensor::reduction::Reduction::Sum => abs_diff
                .sum_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            incin_core::tensor::reduction::Reduction::None => abs_diff,
        };
        CandleStorage::try_new(raw)
    }

    /// Computes Binary Cross-Entropy from logits:
    /// `max(x, 0) - x*y + log(1 + exp(-|x|))`
    /// with the given `reduction` (Mean, Sum, or None).
    pub fn bce_with_logits_loss<K: incin_core::tensor::dtype::DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        reduction: incin_core::prelude::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        // Numerically stable: max(x, 0) - x*y + log(1 + exp(-|x|))
        let zero = pred
            .tensor()
            .zeros_like()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let relu_x = pred
            .tensor()
            .maximum(&zero)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let x_y = pred
            .tensor()
            .broadcast_mul(target.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let abs_x = pred
            .tensor()
            .abs()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let neg_abs_x = abs_x
            .neg()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let exp_neg_abs = neg_abs_x
            .exp()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let one = (exp_neg_abs + 1.0f64).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let log_term = one
            .log()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let elementwise = relu_x
            .broadcast_sub(&x_y)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
            .broadcast_add(&log_term)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let raw = match reduction {
            incin_core::tensor::reduction::Reduction::Mean => elementwise
                .mean_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            incin_core::tensor::reduction::Reduction::Sum => elementwise
                .sum_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            incin_core::tensor::reduction::Reduction::None => elementwise,
        };
        CandleStorage::try_new(raw)
    }

    /// Computes mean squared error between `pred` and `target`; the
    /// reduction argument is ignored since candle's `mse` always averages.
    pub fn mse_loss<K: incin_core::tensor::dtype::DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<K>,
        _reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let loss = candle_nn::loss::mse(pred.tensor(), target.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(loss)
    }

    /// Computes cross-entropy loss between `pred` logits and `target`
    /// class indices, casting `target` to `U32` as candle requires; the
    /// reduction argument is ignored.
    pub fn cross_entropy_loss<K: incin_core::tensor::dtype::DType, KInt: incin_core::tensor::dtype::DType>(
        pred: &<Self as StorageBackend>::Storage<K>,
        target: &<Self as StorageBackend>::Storage<KInt>,
        _reduction: incin_core::prelude::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let target_u32 = target
            .tensor()
            .to_dtype(candle_core::DType::U32)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let loss = candle_nn::loss::cross_entropy(pred.tensor(), &target_u32)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(loss)
    }
}
