//! Quantization operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::*;

impl<D: incin_core::prelude::Device> incin_core::backend_authoring::legacy::QuantizedOps<Self>
    for CandleBackend<D>
{
    /// Not supported by candle; always returns `UnsupportedBackendOperation`.
    fn quantize<K: incin_core::prelude::FloatDType, Q: incin_core::prelude::QuantDType>(
        _t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<Q>> {
        Err(Error::UnsupportedBackendOperation {
            op: "quantize",
            backend: "Candle",
        })
    }
    /// Not supported by candle; always returns `UnsupportedBackendOperation`.
    fn dequantize<Q: incin_core::prelude::QuantDType, K: incin_core::prelude::FloatDType>(
        _t: &<Self as StorageBackend>::Storage<Q>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "dequantize",
            backend: "Candle",
        })
    }
    /// Not supported by candle; always returns `UnsupportedBackendOperation`.
    fn quantized_matmul<Q: incin_core::prelude::QuantDType>(
        _lhs: &<Self as StorageBackend>::Storage<Q>,
        _rhs: &<Self as StorageBackend>::Storage<Q>,
    ) -> Result<<Self as StorageBackend>::Storage<f32>> {
        Err(Error::UnsupportedBackendOperation {
            op: "quantized_matmul",
            backend: "Candle",
        })
    }
}
