//! # Kindle WGPU
//!
//! Provides a hardware-accelerated WGPU backend implementation for the Kindle framework.
//! Enables cross-platform GPU execution (Vulkan, Metal, DX12) via `wgpu`.

#[macro_use]
// Only `WgpuBackend` and its associated types (`WgpuVar`, `WgpuGrads`) are
// intentional public API.  Everything else (dispatch helpers, pipeline cache,
// device state, raw buffer types) is an implementation detail and is
// `pub(crate)` only.
pub(crate) mod backend;
pub(crate) mod device;
pub(crate) mod dispatch;
pub(crate) mod pipeline;
pub(crate) mod storage;
pub(crate) mod tape;

// The three types a downstream crate legitimately needs:
//   - `WgpuBackend<T, D>` to parameterise `Tensor`
//   - `WgpuVar` returned by `CreationOps::var_*`
//   - `WgpuGrads` as `Backend::Grads`
pub use backend::{WgpuBackend, WgpuGrads, WgpuVar};

/// Auto-generated documentation for tests.
mod tests;
