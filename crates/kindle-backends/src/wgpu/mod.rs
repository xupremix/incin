//! # Kindle WGPU
//!
//! Provides a hardware-accelerated WGPU backend implementation for the Kindle framework.
//! Enables cross-platform GPU execution (Vulkan, Metal, DX12) via `wgpu`.

#[macro_use]
// The implementation type and associated types must be public because they
// appear when the public `KindleBackend` alias is normalized. They are not
// re-exported from the public prelude.
pub(crate) mod backend;
pub(crate) mod device;
pub(crate) mod dispatch;
pub(crate) mod pipeline;
pub(crate) mod storage;
pub(crate) mod tape;

pub use backend::{WgpuBackendImpl, WgpuGrads, WgpuVar};

/// `tests`.
mod tests;
