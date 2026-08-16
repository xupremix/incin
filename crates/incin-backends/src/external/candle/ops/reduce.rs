//! Reduction operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::executor::CandleStorage;
use crate::external::*;

impl<D: incin_core::tensor::device::Device> CandleBackend<D> {
    /// Sums all elements into a scalar tensor.
    pub fn sum_all<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .sum_all()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Averages all elements into a scalar tensor.
    pub fn mean_all<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .mean_all()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Reduces to the maximum element as a scalar tensor.
    pub fn max_all<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .max_all()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Reduces to the minimum element as a scalar tensor.
    pub fn min_all<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .min_all()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// Sums along `dim`, removing it from the shape.
    pub fn sum_dim<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .sum(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Sums along `dim`, keeping it as a size-1 dimension.
    pub fn sum_keepdim<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .sum_keepdim(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// Averages along `dim`, removing it from the shape.
    pub fn mean_dim<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .mean(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Averages along `dim`, keeping it as a size-1 dimension.
    pub fn mean_keepdim<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .mean_keepdim(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Reduces to the maximum along `dim`, removing it from the shape.
    pub fn max_dim<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .max(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Reduces to the maximum along `dim`, keeping it as a size-1
    /// dimension.
    pub fn max_keepdim<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .max_keepdim(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Reduces to the minimum along `dim`, removing it from the shape.
    pub fn min_dim<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .min(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }
    /// Reduces to the minimum along `dim`, keeping it as a size-1
    /// dimension.
    pub fn min_keepdim<K: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let res = t
            .tensor()
            .min_keepdim(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(res)
    }

    /// Returns the index of the maximum element, along `dim` if given,
    /// otherwise over the flattened tensor.
    pub fn argmax<K: incin_core::tensor::dtype::DType, KInt: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let raw = match dim {
            Some(d) => t
                .tensor()
                .argmax(d)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => t
                .tensor()
                .flatten_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .argmax(0)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
        };
        CandleStorage::try_new(raw)
    }

    /// Returns the index of the minimum element, along `dim` if given,
    /// otherwise over the flattened tensor.
    pub fn argmin<K: incin_core::tensor::dtype::DType, KInt: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let raw = match dim {
            Some(d) => t
                .tensor()
                .argmin(d)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            None => t
                .tensor()
                .flatten_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .argmin(0)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
        };
        CandleStorage::try_new(raw)
    }

    /// `topk` is not natively available in candle; returns an error
    /// instead of panicking so callers can handle the unsupported case.
    pub fn topk<K: incin_core::tensor::dtype::DType, KInt: incin_core::tensor::dtype::DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _k: usize,
        _dim: usize,
        _largest: bool,
    ) -> Result<(
        <Self as StorageBackend>::Storage<K>,
        <Self as StorageBackend>::Storage<KInt>,
    )> {
        Err(Error::UnsupportedBackendOperation {
            op: "topk",
            backend: "CandleBackend",
        })
    }

    /// Sorts indices along the last dimension using candle's native
    /// `argsort_last_dim`. For non-last dimensions, transposes to last
    /// and back.
    pub fn argsort<K: incin_core::tensor::dtype::DType, KInt: incin_core::tensor::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let rank = t.tensor().rank();
        let last = rank.saturating_sub(1);
        // Candle's arg_sort_last_dim takes `asc: bool`; our API takes `descending: bool`.
        let asc = !descending;
        let raw = if dim == last {
            t.tensor()
                .arg_sort_last_dim(asc)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
        } else {
            // Transpose target dim to last, sort, transpose back.
            // arg_sort_last_dim requires a contiguous tensor, so make it so.
            let t_swap = t
                .tensor()
                .transpose(dim, last)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .contiguous()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let sorted = t_swap
                .arg_sort_last_dim(asc)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            sorted
                .transpose(dim, last)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
        };
        CandleStorage::try_new(raw)
    }
}
