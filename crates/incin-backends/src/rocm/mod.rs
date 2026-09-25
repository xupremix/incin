//! ROCm/HIP backend scaffolding for AMD GPUs (issue #6).
//!
//! Compile-only slice: first-class [`DeviceKind::Rocm`](incin_core::tensor::device::DeviceKind::Rocm)
//! identity, an ordinal detection stub that reports runtime-unavailable, a
//! validated storage-metadata shell, and the exact empty capability table.
//! No HIP bindings are vendored, so no device memory is allocated and no
//! kernels are advertised.
//!
//! Split by concern per `docs/CONVENTIONS.md`, mirroring the CUDA split the
//! issue names as the model: `storage` is the value types (`RocmBuffer`,
//! `RocmStorage`) and their metadata checks; `contract` is the
//! `StorageBackend`/`HostInterop` trait implementations plus the validation
//! and hardware-boundary helpers they share; `capability` is the empty
//! capability table. Detection lives in [`crate::detect`] beside the other
//! families' probes rather than here.

pub mod capability;
pub mod contract;
pub mod storage;

pub use capability::ROCM_CAPABILITIES;
pub use contract::{
    RocmBackendImpl, checked_numel, checked_storage_byte_len, require_rocm_hardware,
    validate_rocm_device, validate_rocm_storage, validate_rocm_storage_dtype,
};
pub use storage::{ROCM_ALLOCATION_ALIGNMENT_BYTES, RocmBuffer, RocmStorage, TensorId};
