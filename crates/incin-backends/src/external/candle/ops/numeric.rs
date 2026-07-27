//! Elementwise arithmetic operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::*;

impl<T: incin_core::prelude::DType, D: incin_core::prelude::Device>
    incin_core::prelude::NumericOps<Self> for CandleBackend<T, D>
{
    /// Element-wise addition with broadcasting.
    fn add<K: incin_core::prelude::DType>(
        lhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
        rhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(lhs
            .broadcast_add(rhs)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Element-wise subtraction with broadcasting.
    fn sub<K: incin_core::prelude::DType>(
        lhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
        rhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(lhs
            .broadcast_sub(rhs)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Element-wise multiplication with broadcasting.
    fn mul<K: incin_core::prelude::DType>(
        lhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
        rhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(lhs
            .broadcast_mul(rhs)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
    /// Element-wise division with broadcasting.
    fn div<K: incin_core::prelude::DType>(
        lhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
        rhs: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(lhs
            .broadcast_div(rhs)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
}
