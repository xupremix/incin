//! Native Metal backend for Apple Silicon and macOS devices.

pub mod backend;
pub mod capability;
pub mod executor;
/// MPS and MPSGraph structured candidates with explicit native fallback.
///
/// Enabled by the `metal-mps` Cargo feature. On non-Apple-Silicon hosts the
/// module is always compiled (so tests are reachable) but every candidate
/// resolves to the `Native` path because [`MPS_AVAILABLE`] is `false`.
pub mod mps;
pub mod shaders;
pub mod storage;
pub mod tape;
pub mod tuning;

pub use backend::{MetalBackendImpl, MetalVar};
pub use storage::{MetalStorage, MetalStorageMode, is_unified_memory};
pub use tape::MetalGrads;
pub use tuning::{
    MetalLaunchCandidate, default_metal_pointwise_candidate, default_metal_reduction_candidate,
    metal_environment_fingerprint, metal_matmul_candidates, metal_pointwise_candidates,
    metal_reduction_candidates, preferred_metal_storage_mode,
};
