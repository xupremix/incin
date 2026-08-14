//! Elementwise arithmetic operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::executor::CandleStorage;
use crate::external::*;

pub(crate) fn add_storage(lhs: &CandleStorage, rhs: &CandleStorage) -> Result<CandleStorage> {
    let t = lhs
        .tensor()
        .broadcast_add(rhs.tensor())
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
    CandleStorage::try_new(t)
}

pub(crate) fn sub_storage(lhs: &CandleStorage, rhs: &CandleStorage) -> Result<CandleStorage> {
    let t = lhs
        .tensor()
        .broadcast_sub(rhs.tensor())
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
    CandleStorage::try_new(t)
}

pub(crate) fn mul_storage(lhs: &CandleStorage, rhs: &CandleStorage) -> Result<CandleStorage> {
    let t = lhs
        .tensor()
        .broadcast_mul(rhs.tensor())
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
    CandleStorage::try_new(t)
}

pub(crate) fn div_storage(lhs: &CandleStorage, rhs: &CandleStorage) -> Result<CandleStorage> {
    let t = lhs
        .tensor()
        .broadcast_div(rhs.tensor())
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
    CandleStorage::try_new(t)
}

impl<D: incin_core::prelude::Device> CandleBackend<D> {
    /// Element-wise addition with broadcasting.
    pub fn add<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        add_storage(lhs, rhs)
    }
    /// Element-wise subtraction with broadcasting.
    pub fn sub<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        sub_storage(lhs, rhs)
    }
    /// Element-wise multiplication with broadcasting.
    pub fn mul<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        mul_storage(lhs, rhs)
    }
    /// Element-wise division with broadcasting.
    pub fn div<K: incin_core::prelude::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        div_storage(lhs, rhs)
    }
}
