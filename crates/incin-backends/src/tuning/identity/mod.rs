//! Stable identities for tuning environments.
//!
//! Device ordinals are deliberately absent. An ordinal is a process-local
//! address which changes under visibility masks and can name different
//! physical devices on different hosts. Persistent tuning keys instead bind
//! the vendor identifier, architecture, driver, compiler, compiler target,
//! and semantic compiler options.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `backend` is the
//! backend-family tag and the sealed `StaticBackend` marker vocabulary;
//! `error` is the failure vocabulary and canonical-field validation shared
//! by every fingerprint constructor; `primitives` is `SoftwareVersion` and
//! the length-delimited `Digest` accumulator every fingerprint's `digest()`
//! builds on; `device`, `compiler`, and `environment` are the per-backend
//! identity fingerprints, in ascending order of composition (an environment
//! wraps a device and a compiler); `topology` is the multi-rank world,
//! links, transport, and process layout; `cuda` is the CUDA-only queries
//! that populate a device/compiler/environment fingerprint from a live
//! context.

mod backend;
mod compiler;
#[cfg(feature = "cuda")]
mod cuda;
mod device;
mod environment;
mod error;
mod primitives;
mod topology;

pub use backend::{BackendIdentity, StaticBackend};
pub use compiler::CompilerFingerprint;
pub use device::DeviceFingerprint;
pub use environment::TuningEnvironmentFingerprint;
pub use error::IdentityError;
pub use primitives::SoftwareVersion;
pub use topology::{
    ProcessLayoutFingerprint, StaticWorld, TopologyLink, TransportFingerprint,
    TuningTopologyFingerprint,
};
