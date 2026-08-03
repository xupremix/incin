//! Optimizer operations for the Candle adapter.

use crate::external::candle::CandleBackend;

impl<T: incin_core::prelude::DType, D: incin_core::prelude::Device>
    incin_core::backend_authoring::OptimizerOps<Self> for CandleBackend<T, D>
{
}
