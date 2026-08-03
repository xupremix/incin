//! Quantization operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::*;

impl<T: incin_core::prelude::DType, D: incin_core::prelude::Device>
    incin_core::backend_authoring::QuantizedOps<Self> for CandleBackend<T, D>
{
    /// Not supported by candle; always returns `UnsupportedBackendOperation`.
    fn quantize<K: incin_core::prelude::FloatDType, Q: incin_core::prelude::QuantDType>(
        _t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<Q>> {
        Err(Error::UnsupportedBackendOperation {
            op: "quantize",
            backend: "Candle",
        })
    }
    /// Not supported by candle; always returns `UnsupportedBackendOperation`.
    fn dequantize<Q: incin_core::prelude::QuantDType, K: incin_core::prelude::FloatDType>(
        _t: &<Self as incin_core::prelude::Backend>::Storage<Q>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "dequantize",
            backend: "Candle",
        })
    }
    /// Not supported by candle; always returns `UnsupportedBackendOperation`.
    fn quantized_matmul<Q: incin_core::prelude::QuantDType>(
        _lhs: &<Self as incin_core::prelude::Backend>::Storage<Q>,
        _rhs: &<Self as incin_core::prelude::Backend>::Storage<Q>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<f32>> {
        Err(Error::UnsupportedBackendOperation {
            op: "quantized_matmul",
            backend: "Candle",
        })
    }
}
