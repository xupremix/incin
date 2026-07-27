//! Third-party backend integrations.
//!
//! These adapters delegate execution to external tensor ecosystems and are
//! intentionally separate from Incin native CPU, CUDA, and WGPU backends.

pub use incin_core::prelude::*;

// ----------------------------------------------------------------------------
// CandleBackend
// ----------------------------------------------------------------------------

/// Wraps the `candle_core` crate, providing `CandleBackend` as a `Backend`
/// implementation backed by Candle's own tensor type.
pub mod candle;
