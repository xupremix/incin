//! The backend-authoring trait implementations that make
//! `RocmBackendImpl` a `StorageBackend`/`HostInterop`, and the validation
//! helpers they share.
//!
//! Mirrors the `cuda/backend/contract.rs` split: `storage.rs` owns the
//! types and their metadata checks, this module owns the contract traits and
//! the device-boundary helpers. Every method body validates its inputs first
//! (typed errors, never panics) and then refuses at
//! [`require_rocm_hardware`]: without HIP bindings there is no device memory
//! to read or write.
//!
//! Deliberately *not* a full [`incin_core::backend_authoring::Backend`]:
//! that profile claims execution capability, and this slice advertises zero
//! kernels (see [`super::capability`]). The HIP lane adds execution when
//! there is something to execute.

use super::storage::RocmStorage;
use alloc::vec::Vec;
use core::marker::PhantomData;
use incin_core::backend_authoring::{
    HostInterop, HostReadback, StorageBackend, StorageOutput, TensorMeta,
};
use incin_core::error::{Error, Result};
use incin_core::shapes::{OperationKind, ShapeBuf};
use incin_core::tensor::device::{Device, DeviceId, DeviceKind, Rocm};
use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

/// ROCm compute backend scaffolding for Incin (issue #6).
///
/// Stateless marker, like `CudaBackendImpl` in `cuda/backend`: the ordinal lives
/// on the storage's [`DeviceId`], not here.
#[derive(Clone)]
pub struct RocmBackendImpl<D = Rocm>(PhantomData<D>);

impl<D> RocmBackendImpl<D> {
    /// Construct the stateless ROCm executor scaffolding.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<D> Default for RocmBackendImpl<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Device> StorageBackend for RocmBackendImpl<D> {
    type Device = D;
    const BACKEND_NAME: &'static str = "Rocm";
    type Storage<K: DType> = RocmStorage;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        &storage.meta
    }

    fn fresh_autograd_identity<K: DType>(storage: Self::Storage<K>) -> Self::Storage<K> {
        storage.with_fresh_autograd_identity()
    }
}

impl StorageOutput for RocmStorage {}

impl<D: Device> HostReadback for RocmBackendImpl<D> {
    fn float_to_vec1<K: DType>(storage: &Self::Storage<K>) -> Result<Vec<f64>> {
        let storage: &RocmStorage = storage;
        validate_rocm_storage_dtype(storage.meta.dtype, "float_to_vec1")?;
        Err(require_rocm_hardware(
            storage.meta.device.ordinal(),
            "float_to_vec1",
        ))
    }

    fn int_to_vec1<K: DType>(storage: &Self::Storage<K>) -> Result<Vec<i64>> {
        let storage: &RocmStorage = storage;
        validate_rocm_storage_dtype(storage.meta.dtype, "int_to_vec1")?;
        Err(require_rocm_hardware(
            storage.meta.device.ordinal(),
            "int_to_vec1",
        ))
    }
}

impl<D: Device> HostInterop for RocmBackendImpl<D> {
    fn to_bytes<K: DType>(storage: &Self::Storage<K>) -> Result<Vec<u8>> {
        let storage: &RocmStorage = storage;
        validate_rocm_storage(storage.meta.dtype, &storage.meta.device, "to_bytes")?;
        Err(require_rocm_hardware(
            storage.meta.device.ordinal(),
            "to_bytes",
        ))
    }

    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_rocm_storage(dtype, device, "from_bytes")?;
        let numel = checked_numel(shape)?;
        let expected = checked_storage_byte_len(numel, dtype)?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        Err(require_rocm_hardware(device.ordinal(), "from_bytes"))
    }
}

/// The hardware boundary: every device-memory access lands here.
///
/// Without vendored HIP bindings no ordinal can be selected, so every call
/// refuses with [`Error::DeviceInitializationError`] naming the missing
/// runtime rather than the ordinal. When HIP bindings land, this becomes the
/// per-ordinal context lookup (mirroring
/// `cuda_cache::try_get_cuda_device`), and the refusal narrows to genuinely
/// absent ordinals.
pub fn require_rocm_hardware(ordinal: usize, op: &'static str) -> Error {
    let _ = op;
    Error::DeviceInitializationError {
        expected: alloc::format!("rocm:{ordinal} backed by a HIP runtime"),
        got: alloc::string::String::from(
            "ROCm scaffolding build: no HIP bindings are vendored, so no device memory exists",
        ),
    }
}

/// Validates a ROCm dtype/device pair before any device access.
pub fn validate_rocm_storage(
    dtype: DTypeDescriptor,
    device: &DeviceId,
    op: &'static str,
) -> Result<()> {
    validate_rocm_device(device)?;
    validate_rocm_storage_dtype(dtype, op)
}

/// Rejects devices that are not ROCm at all (cross-device misuse).
pub fn validate_rocm_device(device: &DeviceId) -> Result<()> {
    if device.kind() != DeviceKind::Rocm {
        return Err(Error::DeviceInitializationError {
            expected: alloc::string::String::from("rocm"),
            got: alloc::string::String::from(device.kind().name()),
        });
    }
    Ok(())
}

/// The ROCm storage dtype family: the CUDA storage width, verbatim.
///
/// Storage validation is deliberately wider than any kernel set: buffers
/// legitimately hold `f64`/`f16`/`bf16`/`i64`/`q8_0`/`bool` while kernels —
/// of which ROCm has none yet — admit narrower sets one capability row at a
/// time. Accepting a dtype here claims byte-storage only, never executability.
pub fn validate_rocm_storage_dtype(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    let is_supported = matches!(
        dtype.builtin_id(),
        Some(
            DTypeId::F32
                | DTypeId::F64
                | DTypeId::F16
                | DTypeId::BF16
                | DTypeId::I64
                | DTypeId::Q8_0
                | DTypeId::Bool
        )
    );
    if is_supported {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            dtype,
            backend: "Rocm",
            op,
        })
    }
}

/// Byte length of `numel` elements of `dtype`, or a typed overflow error.
pub fn checked_storage_byte_len(numel: usize, dtype: DTypeDescriptor) -> Result<usize> {
    dtype
        .size_bytes(numel, OperationKind::Storage)
        .map_err(Error::from)
}

/// Checked element count of a user-supplied shape (overflow-safe).
pub fn checked_numel(shape: &[usize]) -> Result<usize> {
    ShapeBuf::from_slice(shape)
        .checked_numel(OperationKind::Storage)
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use incin_core::shapes::Dyn;

    fn device() -> DeviceId {
        DeviceId::rocm(0)
    }

    #[test]
    fn cross_device_upload_is_rejected_before_hardware() {
        let error = RocmBackendImpl::<Rocm>::from_bytes::<Dyn>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::cuda(0),
        )
        .expect_err("a CUDA device must not upload through the ROCm contract");
        assert!(
            matches!(error, Error::DeviceInitializationError { .. }),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn byte_length_mismatch_is_rejected_before_hardware() {
        let error = RocmBackendImpl::<Rocm>::from_bytes::<Dyn>(
            &[0u8; 3],
            &[1],
            DTypeId::F32.descriptor(),
            &device(),
        )
        .expect_err("3 bytes cannot hold one f32");
        assert!(
            matches!(
                error,
                Error::InvalidByteLength {
                    expected: 4,
                    got: 3
                }
            ),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn well_formed_upload_refuses_at_the_hardware_boundary() {
        let error = RocmBackendImpl::<Rocm>::from_bytes::<Dyn>(
            &[0u8; 16],
            &[4],
            DTypeId::F32.descriptor(),
            &device(),
        )
        .expect_err("no HIP bindings exist, so no upload can succeed");
        assert!(
            matches!(error, Error::DeviceInitializationError { .. }),
            "unexpected rejection: {error}"
        );
        assert!(
            format!("{error}").contains("HIP"),
            "the refusal must name the missing runtime: {error}"
        );
    }

    #[test]
    fn unsupported_dtype_is_refused_before_hardware() {
        let error = RocmBackendImpl::<Rocm>::from_bytes::<Dyn>(
            &[0u8; 4],
            &[1],
            DTypeId::U32.descriptor(),
            &device(),
        )
        .expect_err("u32 is outside the ROCm storage family");
        assert!(
            matches!(error, Error::UnsupportedDType { .. }),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn readback_refuses_without_hardware_but_validates_first() {
        let storage = RocmStorage::new(
            alloc::sync::Arc::new(super::super::storage::RocmBuffer {
                len: 4,
                dtype: DTypeId::F32.descriptor(),
                device_id: 0,
            }),
            alloc::vec![4],
            device(),
        );
        let error = RocmBackendImpl::<Rocm>::float_to_vec1::<Dyn>(&storage)
            .expect_err("no HIP runtime can serve a readback");
        assert!(
            matches!(error, Error::DeviceInitializationError { .. }),
            "unexpected rejection: {error}"
        );
    }
}
