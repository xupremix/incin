//! # Incin WGPU
//!
//! Provides a hardware-accelerated WGPU backend implementation for the Incin framework.
//! Enables cross-platform GPU execution (Vulkan, Metal, DX12) via `wgpu`.

#[macro_use]
// The implementation type and associated types must be public because they
// appear when the public `IncinBackend` alias is normalized. They are not
// re-exported from the public prelude.
pub(crate) mod backend;
pub(crate) mod capability;
pub(crate) mod device;
pub(crate) mod dispatch;
pub(crate) mod executor;
pub(crate) mod pipeline;
pub(crate) mod storage;
pub(crate) mod tape;

pub use backend::{WgpuBackendImpl, WgpuGrads, WgpuVar};
/// Number of entries currently on this backend's autograd tape.
///
/// Re-exported since `GRD-002`: the row claims a `NoGrad` chain records
/// nothing, and its evidence test lives outside this crate. A guarantee
/// nothing outside can observe is not a guarantee.
pub use tape::depth as tape_depth;

#[cfg(test)]
mod tests;
