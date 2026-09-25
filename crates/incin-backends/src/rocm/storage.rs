//! Storage implementation and memory management for the ROCm backend.
//!
//! Issue #6 scaffolding: validated metadata shells without device memory. No
//! HIP bindings are vendored in this tree, so there is no runtime to allocate
//! against and [`RocmBuffer`] carries no device bytes. What this module *does*
//! provide is the full metadata-validation shape of `cuda/storage.rs`:
//! element-count sizing, contiguous/strided [`TensorMeta`] construction, and
//! ordinal binding checks, all returning typed errors instead of panicking.
//! Every path that would touch device memory refuses at
//! [`super::contract::require_rocm_hardware`] instead.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;

use incin_core::error::{Error, Result};
use incin_core::exec::{Alignment, TensorMeta};
use incin_core::shapes::OperationKind;
use incin_core::tensor::device::{DeviceId, DeviceKind};
use incin_core::tensor::dtype::DTypeDescriptor;

/// Re-exported from `incin_core::exec::tape` since `GRD-003`: one identity
/// counter serves the whole workspace.
pub use incin_core::exec::TensorId;

/// Byte alignment every ROCm device allocation is expected to satisfy.
///
/// Provisional: mirrors CUDA's documented 256-byte floor until HIP bindings
/// land and the ignored `device_pointers_meet_the_allocation_floor` smoke
/// test measures real `hipMalloc` pointers. Do not widen layout assumptions
/// past [`Alignment::BYTE`] on the strength of this constant alone.
pub const ROCM_ALLOCATION_ALIGNMENT_BYTES: usize = 256;

/// Identity half of a ROCm allocation: what was asked for and where it
/// belongs, without the device bytes (no HIP bindings exist to hold them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmBuffer {
    pub(crate) len: usize,
    pub(crate) dtype: DTypeDescriptor,
    pub(crate) device_id: usize,
}

/// Storage handle backing tensors on ROCm devices.
///
/// A validated metadata shell: [`TensorMeta`] construction runs the same
/// checks as `CudaStorage` in `cuda/storage.rs`, but no device memory is
/// attached. Any transfer to or from device memory refuses with a typed
/// error (see [`super::contract`]) until HIP bindings land.
#[derive(Clone, Debug, PartialEq)]
pub struct RocmStorage {
    pub(crate) buffer: Arc<RocmBuffer>,
    pub(crate) meta: TensorMeta,
    pub(crate) id: TensorId,
}

impl Deref for RocmStorage {
    type Target = TensorMeta;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl RocmStorage {
    pub(crate) fn with_fresh_autograd_identity(mut self) -> Self {
        self.id = TensorId::next();
        self
    }

    /// The alignment guarantee ROCm allocations are expected to carry.
    ///
    /// Infallible despite [`Alignment::new`] being fallible, because
    /// [`ROCM_ALLOCATION_ALIGNMENT_BYTES`] is a power of two by construction
    /// and a `debug_assert` pins that; a bad constant is a source error, not
    /// a runtime condition a caller could handle.
    fn allocation_alignment() -> Alignment {
        debug_assert!(ROCM_ALLOCATION_ALIGNMENT_BYTES.is_power_of_two());
        Alignment::new(ROCM_ALLOCATION_ALIGNMENT_BYTES).unwrap_or(Alignment::BYTE)
    }

    /// Check that the claimed element count sizes without overflow.
    ///
    /// The CUDA split additionally proves the device allocation covers the
    /// claim (`allocated >= required`). There is no allocation here yet — no
    /// HIP bindings — so this validates the metadata half only: the `(len,
    /// dtype)` pair must size to a byte count without overflowing. The
    /// byte-coverage half lands with the HIP allocation path; until then
    /// every device transfer refuses at `require_rocm_hardware`, so an
    /// undersized claim can never reach a kernel.
    fn check_allocation_covers_len(buffer: &RocmBuffer) -> Result<()> {
        buffer
            .dtype
            .size_bytes(buffer.len, OperationKind::Storage)
            .map_err(|error| {
                Error::Msg(alloc::format!("invalid ROCm allocation length: {error}"))
            })?;
        Ok(())
    }

    /// Check that `device` names this buffer's own ROCm ordinal.
    ///
    /// The ordinal binding is validated at wrap time because there is no
    /// driver to ask later: a scaffolding build that accepted a foreign
    /// ordinal silently would bless transfers the HIP lane cannot honor.
    fn check_device_binds_buffer(buffer: &RocmBuffer, device: &DeviceId) -> Result<()> {
        if device.kind() != DeviceKind::Rocm {
            return Err(Error::DeviceInitializationError {
                expected: alloc::string::String::from("rocm"),
                got: alloc::string::String::from(device.kind().name()),
            });
        }
        if device.ordinal() != buffer.device_id {
            return Err(Error::DeviceStorageMismatch {
                expected: *device,
                got: DeviceId::rocm(buffer.device_id),
            });
        }
        Ok(())
    }

    /// Wrap a validated buffer under explicit strided metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the dtype is outside the ROCm storage family,
    /// when `device` is not this buffer's ROCm ordinal, or when the metadata
    /// itself is inconsistent. Never panics on user input.
    pub fn try_from_parts(
        buffer: Arc<RocmBuffer>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset_elements: usize,
        device: DeviceId,
    ) -> Result<Self> {
        super::contract::validate_rocm_storage(buffer.dtype, &device, "try_from_parts")?;
        Self::check_device_binds_buffer(&buffer, &device)?;
        Self::check_allocation_covers_len(&buffer)?;
        let meta = TensorMeta::try_new(
            shape.as_slice().into(),
            strides.as_slice().into(),
            offset_elements,
            buffer.dtype,
            device,
            Self::allocation_alignment(),
            buffer.len,
        )
        .map_err(|error| Error::Msg(alloc::format!("invalid ROCm storage metadata: {error}")))?;
        Ok(Self {
            buffer,
            meta,
            id: TensorId::next(),
        })
    }

    /// Wrap a validated buffer under dense contiguous metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] under the same conditions as [`Self::try_from_parts`].
    /// Never panics on user input.
    pub fn try_new(buffer: Arc<RocmBuffer>, shape: Vec<usize>, device: DeviceId) -> Result<Self> {
        super::contract::validate_rocm_storage(buffer.dtype, &device, "try_new")?;
        Self::check_device_binds_buffer(&buffer, &device)?;
        Self::check_allocation_covers_len(&buffer)?;
        let meta = TensorMeta::contiguous(
            shape.as_slice().into(),
            buffer.dtype,
            device,
            Self::allocation_alignment(),
            buffer.len,
        )
        .map_err(|error| Error::Msg(alloc::format!("invalid ROCm storage metadata: {error}")))?;
        Ok(Self {
            buffer,
            meta,
            id: TensorId::next(),
        })
    }

    /// Wrap a backend-created buffer whose metadata is known consistent.
    ///
    /// # Panics
    ///
    /// Panics when the metadata is inconsistent, exactly like
    /// `CudaStorage::new` in `cuda/storage.rs`: backend-created storage
    /// must match its allocation, so a failure here is a backend bug rather
    /// than user input. User-supplied bytes and shapes go through
    /// [`Self::try_new`], which returns [`Error`] instead.
    pub fn new(buffer: Arc<RocmBuffer>, shape: Vec<usize>, device: DeviceId) -> Self {
        Self::try_new(buffer, shape, device)
            .expect("backend-created contiguous ROCm storage must match its allocation")
    }

    /// Validated metadata describing this storage value.
    pub fn metadata(&self) -> &TensorMeta {
        &self.meta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use incin_core::tensor::device::DeviceKind;
    use incin_core::tensor::dtype::DTypeId;

    fn buffer(len: usize, dtype: DTypeDescriptor, device_id: usize) -> Arc<RocmBuffer> {
        Arc::new(RocmBuffer {
            len,
            dtype,
            device_id,
        })
    }

    #[test]
    fn storage_wraps_validated_metadata_without_hardware() {
        let storage = RocmStorage::try_new(
            buffer(6, DTypeId::F32.descriptor(), 0),
            alloc::vec![2, 3],
            DeviceId::rocm(0),
        )
        .expect("valid metadata wraps without touching hardware");
        assert_eq!(storage.metadata().device, DeviceId::rocm(0));
        assert_eq!(storage.metadata().dtype, DTypeId::F32.descriptor());
        assert_eq!(storage.metadata().shape.dims(), &[2, 3]);
        assert_eq!(storage.metadata().device.kind(), DeviceKind::Rocm);
    }

    #[test]
    fn cross_device_wrap_is_rejected_with_a_typed_error() {
        let error = RocmStorage::try_new(
            buffer(4, DTypeId::F32.descriptor(), 0),
            alloc::vec![4],
            DeviceId::cuda(0),
        )
        .expect_err("a CUDA device must not wrap a ROCm buffer");
        assert!(
            matches!(error, Error::DeviceInitializationError { .. }),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn foreign_ordinal_wrap_is_rejected_with_a_typed_error() {
        let error = RocmStorage::try_new(
            buffer(4, DTypeId::F32.descriptor(), 1),
            alloc::vec![4],
            DeviceId::rocm(0),
        )
        .expect_err("ordinal 1's buffer must not wrap as ordinal 0");
        assert!(
            matches!(
                error,
                Error::DeviceStorageMismatch { expected, got }
                if expected == DeviceId::rocm(0) && got == DeviceId::rocm(1)
            ),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn unsupported_dtype_is_refused_before_any_hardware_is_reached() {
        let error = RocmStorage::try_new(
            buffer(4, DTypeId::U32.descriptor(), 0),
            alloc::vec![4],
            DeviceId::rocm(0),
        )
        .expect_err("u32 is outside the ROCm storage family");
        assert!(
            matches!(error, Error::UnsupportedDType { .. }),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn strided_wrap_validates_span_against_the_claimed_elements() {
        let storage = RocmStorage::try_from_parts(
            buffer(6, DTypeId::F32.descriptor(), 0),
            alloc::vec![2, 2],
            alloc::vec![3, 1],
            0,
            DeviceId::rocm(0),
        )
        .expect("a spanning strided view wraps");
        assert_eq!(storage.metadata().shape.dims(), &[2, 2]);
        let error = RocmStorage::try_from_parts(
            buffer(2, DTypeId::F32.descriptor(), 0),
            alloc::vec![2, 2],
            alloc::vec![3, 1],
            0,
            DeviceId::rocm(0),
        )
        .expect_err("a span of 4 elements over 2 claimed must be rejected");
        assert!(
            format!("{error}").contains("invalid ROCm storage metadata"),
            "unexpected rejection: {error}"
        );
    }

    /// Scaffolding shell or not, the storage family from the CUDA split
    /// applies: every dtype CUDA storage admits is the intended ROCm storage
    /// width, and everything else is refused at the boundary.
    #[test]
    fn storage_dtype_validation_mirrors_the_cuda_storage_family() {
        for dtype in [
            DTypeId::F32,
            DTypeId::F64,
            DTypeId::F16,
            DTypeId::BF16,
            DTypeId::I64,
            DTypeId::Q8_0,
            DTypeId::Bool,
        ] {
            RocmStorage::try_new(
                buffer(32, dtype.descriptor(), 0),
                alloc::vec![32],
                DeviceId::rocm(0),
            )
            .unwrap_or_else(|e| panic!("{dtype:?} should wrap: {e:?}"));
        }
        assert!(
            RocmStorage::try_new(
                buffer(4, DTypeId::U32.descriptor(), 0),
                alloc::vec![4],
                DeviceId::rocm(0)
            )
            .is_err()
        );
    }

    /// Hardware-gated: allocates on ordinal 0 through the real HIP path once
    /// bindings land, then checks the allocation floor constant against live
    /// pointers instead of trusting it.
    #[test]
    #[ignore = "requires ROCm hardware"]
    fn device_pointers_meet_the_allocation_floor() {
        let storage = RocmStorage::try_new(
            buffer(257, DTypeId::F32.descriptor(), 0),
            alloc::vec![257],
            DeviceId::rocm(0),
        )
        .expect("ordinal 0 allocates 257 f32 elements");
        let _ = storage;
        panic!("HIP bindings are not vendored yet: no device pointer to measure");
    }

    /// Hardware-gated: the allocation/round-trip contract issue #6 AC1 will
    /// exercise — zeros on an ordinal, host upload, device read-back, byte
    /// equality. Compiles today so the contract shape is fixed; runs when
    /// `from_bytes`/`to_bytes` reach a HIP runtime.
    #[test]
    #[ignore = "requires ROCm hardware"]
    fn allocation_round_trip_on_ordinal_zero() {
        use super::super::contract::RocmBackendImpl;
        use incin_core::backend_authoring::HostInterop;
        use incin_core::shapes::Dyn;
        use incin_core::tensor::device::Rocm;

        let device = DeviceId::rocm(0);
        let bytes = alloc::vec![0u8; 16];
        let storage = RocmBackendImpl::<Rocm>::from_bytes::<Dyn>(
            &bytes,
            &[4],
            DTypeId::F32.descriptor(),
            &device,
        )
        .expect("ordinal 0 accepts a 4-element f32 upload");
        let back = RocmBackendImpl::<Rocm>::to_bytes::<Dyn>(&storage)
            .expect("ordinal 0 reads its allocation back");
        assert_eq!(back, bytes);
    }
}
