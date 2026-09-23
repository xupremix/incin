//! Native Metal backend for Apple Silicon and macOS devices.

pub mod backend;
/// Capability registration for the Metal backend.
pub mod capability;
pub mod executor;
/// Structural shape ops: transpose, narrow/slice, concat/stack, and the
/// squeeze/unsqueeze axis views (#92), with host-side walks and tape recipes.
pub mod layout;
/// Matmul-family compositions: `linear` and `scaled_dot_product_attention`
/// (#92), rewritten into taped Metal primitives in CPU's/WGPU's order.
pub mod linalg;
/// MPS and MPSGraph structured candidates with explicit native fallback.
///
/// Enabled by the `metal-mps` Cargo feature. On non-Apple-Silicon hosts the
/// module is always compiled (so tests are reachable) but every candidate
/// resolves to the `Native` path because [`MPS_AVAILABLE`](mps::MPS_AVAILABLE) is `false`.
pub mod mps;
/// Normalization family: softmax/log_softmax, layer_norm, rms_norm, and the
/// `max_keepdim` the stable softmax recipe needs (#92).
pub mod normalization;
/// Elementwise unaries, scalars, clamp, and the three binary pointwise ops
/// (#92 Batch A), with their tape recipes and host-side parity tests.
pub mod pointwise;
pub mod shaders;
pub mod storage;
// `pub(crate)`, matching cpu, cuda and wgpu. The thread-local itself is not
// the seam: `tape_record` and `tape_record_with` below are, and they are the
// same two names on every backend. This module was the one of the four left
// public, which meant the most important boundary in the crate, whether a
// third party can add a differentiable operation, differed by backend for no
// stated reason.
pub(crate) mod tape;
pub mod tuning;

pub use backend::{MetalBackendImpl, MetalVar};
pub use storage::{MetalStorage, MetalStorageMode, is_unified_memory};
pub use tape::MetalGrads;
/// Number of entries currently on this thread's tape.
///
/// Re-exported for the same reason it is on the other three backends: the
/// claim that a `NoGrad` chain records nothing is only a guarantee if
/// something outside can count.
pub use tape::depth as tape_depth;
/// Record a custom operation's backward recipe on this thread's tape.
///
/// The Metal instantiation of the custom-training contract documented at
/// `crate::cpu::tape_record`. The shader/MPS infrastructure from
/// MTL-001/002/003 is complete; operation coverage on top of it is #92.
pub use tape::record as tape_record;
/// Record a custom operation's backward recipe, building it only if kept.
///
/// The lazy form of `tape_record`, as on the other three backends.
pub use tape::record_with as tape_record_with;
pub use tuning::{
    MetalLaunchCandidate, default_metal_pointwise_candidate, default_metal_reduction_candidate,
    metal_environment_fingerprint, metal_matmul_candidates, metal_pointwise_candidates,
    metal_reduction_candidates, preferred_metal_storage_mode,
};
