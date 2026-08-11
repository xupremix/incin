//! Elementwise arithmetic operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::executor::CandleStorage;
use crate::external::*;

impl<D: incin_core::prelude::Device> incin_core::backend_authoring::NumericOps<Self>
    for CandleBackend<D>
{
    /// Element-wise addition with broadcasting.
    fn add<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t = lhs
            .tensor()
            .broadcast_add(rhs.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(t)
    }
    /// Element-wise subtraction with broadcasting.
    fn sub<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t = lhs
            .tensor()
            .broadcast_sub(rhs.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(t)
    }
    /// Element-wise multiplication with broadcasting.
    fn mul<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t = lhs
            .tensor()
            .broadcast_mul(rhs.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(t)
    }
    /// Element-wise division with broadcasting.
    fn div<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let t = lhs
            .tensor()
            .broadcast_div(rhs.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        CandleStorage::try_new(t)
    }
}
