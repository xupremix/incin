use crate::candle;

/// A trait that abstracts the runtime computational engine (Candle, Burn, Wgpu, etc.).
/// It provides the raw, dynamic memory buffer used by this specific backend.
pub trait Backend {
    type RawTensor;
}

/// The default Candle backend implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandleBackend;

impl Backend for CandleBackend {
    type RawTensor = candle::Tensor;
}
