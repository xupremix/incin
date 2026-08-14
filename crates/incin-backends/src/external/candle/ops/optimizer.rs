//! Optimizer operations for the Candle adapter.

use crate::external::candle::CandleBackend;

impl<D: incin_core::prelude::Device> incin_core::__backend_compat::legacy::OptimizerOps<Self>
    for CandleBackend<D>
{
}
