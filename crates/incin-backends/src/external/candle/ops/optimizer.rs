//! Optimizer operations for the Candle adapter.

use crate::external::candle::CandleBackend;

impl<D: incin_core::prelude::Device> incin_core::tensor::backend::OptimizerOps<Self>
    for CandleBackend<D>
{
}
