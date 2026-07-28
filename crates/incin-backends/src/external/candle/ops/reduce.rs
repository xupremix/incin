//! Reduction operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::*;

impl<T: incin_core::prelude::DType, D: incin_core::prelude::Device>
    incin_core::prelude::ReductionOps<Self> for CandleBackend<T, D>
{
    // This adapter does not route candle's product or cumulative sum yet.
    crate::unsupported::unsupported_reduction_ops! {
        all: prod_all;
        dim: prod_dim, cumsum;
    }

    /// Sums all elements into a scalar tensor.
    fn sum_all<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.sum_all()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Averages all elements into a scalar tensor.
    fn mean_all<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.mean_all()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Reduces to the maximum element as a scalar tensor.
    fn max_all<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.max_all()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Reduces to the minimum element as a scalar tensor.
    fn min_all<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.min_all()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Sums along `dim`, removing it from the shape.
    fn sum_dim<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.sum(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Sums along `dim`, keeping it as a size-1 dimension.
    fn sum_keepdim<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.sum_keepdim(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Averages along `dim`, removing it from the shape.
    fn mean_dim<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.mean(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Averages along `dim`, keeping it as a size-1 dimension.
    fn mean_keepdim<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.mean_keepdim(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Reduces to the maximum along `dim`, removing it from the shape.
    fn max_dim<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.max(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Reduces to the maximum along `dim`, keeping it as a size-1
    /// dimension.
    fn max_keepdim<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.max_keepdim(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Reduces to the minimum along `dim`, removing it from the shape.
    fn min_dim<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.min(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Reduces to the minimum along `dim`, keeping it as a size-1
    /// dimension.
    fn min_keepdim<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(t.min_keepdim(dim)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Returns the index of the maximum element, along `dim` if given,
    /// otherwise over the flattened tensor.
    fn argmax<K: incin_core::prelude::DType, KInt: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<KInt>> {
        match dim {
            Some(d) => Ok(t
                .argmax(d)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            None => Ok(t
                .flatten_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .argmax(0)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
        }
    }

    /// Returns the index of the minimum element, along `dim` if given,
    /// otherwise over the flattened tensor.
    fn argmin<K: incin_core::prelude::DType, KInt: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<KInt>> {
        match dim {
            Some(d) => Ok(t
                .argmin(d)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            None => Ok(t
                .flatten_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .argmin(0)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
        }
    }

    /// `topk` is not natively available in candle; returns an error
    /// instead of panicking so callers can handle the unsupported case.
    fn topk<K: incin_core::prelude::DType, KInt: incin_core::prelude::DType>(
        _t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        _k: usize,
        _dim: usize,
        _largest: bool,
    ) -> Result<(
        <Self as incin_core::prelude::Backend>::Storage<K>,
        <Self as incin_core::prelude::Backend>::Storage<KInt>,
    )> {
        Err(Error::UnsupportedBackendOperation {
            op: "topk",
            backend: "CandleBackend",
        })
    }

    /// Sorts indices along the last dimension using candle's native
    /// `argsort_last_dim`. For non-last dimensions, transposes to last
    /// and back.
    fn argsort<K: incin_core::prelude::DType, KInt: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<KInt>> {
        let rank = t.rank();
        let last = rank.saturating_sub(1);
        // Candle's arg_sort_last_dim takes `asc: bool`; our API takes `descending: bool`.
        let asc = !descending;
        if dim == last {
            Ok(t.arg_sort_last_dim(asc)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        } else {
            // Transpose target dim to last, sort, transpose back.
            // arg_sort_last_dim requires a contiguous tensor, so make it so.
            let t_swap = t
                .transpose(dim, last)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .contiguous()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let sorted = t_swap
                .arg_sort_last_dim(asc)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(sorted
                .transpose(dim, last)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }
}
