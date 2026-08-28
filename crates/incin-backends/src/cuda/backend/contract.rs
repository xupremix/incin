//! The backend-authoring trait implementations that make
//! `CudaBackendImpl` a `StorageBackend`/`Backend`/`HostInterop`, and the
//! validation and byte-conversion helpers they share.

use super::*;

impl<D: Device> incin_core::backend_authoring::StorageBackend for CudaBackendImpl<D> {
    type Device = D;
    const BACKEND_NAME: &'static str = "Cuda";
    type Storage<K: DType> = CudaStorage;

    fn metadata<K: DType>(t: &Self::Storage<K>) -> &incin_core::backend_authoring::TensorMeta {
        let t: &CudaStorage = t;
        &t.meta
    }

    fn fresh_autograd_identity<K: DType>(storage: Self::Storage<K>) -> Self::Storage<K> {
        storage.with_fresh_autograd_identity()
    }
}

impl incin_core::backend_authoring::StorageOutput for CudaStorage {}

// A variable is a trainable handle, not storage, so it carries `ExecuteOutput`
// directly rather than through `StorageOutput`. `CpuVar` does the same; the
// `var_*` creation executors are what need it.
impl incin_core::backend_authoring::ExecuteOutput for super::types::CudaVar {}

impl<D: Device> Backend for CudaBackendImpl<D> {
    type InnerBackend = Self;

    // `host_format_display`/`host_format_debug` use `HostInterop`'s default,
    // which reads real values back through `float_to_vec1`/`int_to_vec1`.
}

impl<D: Device> incin_core::backend_authoring::HostReadback for CudaBackendImpl<D> {
    fn float_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<f64>> {
        let t: &CudaStorage = t;
        cuda_require_f32(t.buffer.dtype, "float_to_vec1")?;
        let data = download_f32_host(t)?;
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    fn int_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<i64>> {
        let t: &CudaStorage = t;
        cuda_require_f32(t.buffer.dtype, "int_to_vec1")?;
        let data = download_f32_host(t)?;
        data.into_iter()
            .map(|value| {
                incin_core::error::convert_f64_to_i64(
                    "int_to_vec1",
                    t.buffer.dtype,
                    f64::from(value),
                    incin_core::error::FloatToIntPolicy::Exact,
                )
            })
            .collect()
    }
}

impl<D: Device> incin_core::backend_authoring::HostInterop for CudaBackendImpl<D> {
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
        let t: &CudaStorage = t;
        let bytes = t
            .buffer
            .device
            .default_stream()
            .clone_dtoh(&*t.buffer.data)
            .map_err(|error| Error::Msg(format!("CUDA download failed: {error:?}")))?;
        let expected = checked_storage_byte_len(t.buffer.len, t.buffer.dtype)?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        Ok(bytes)
    }
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_cuda_storage(dtype, device, "from_bytes")?;
        let numel = checked_numel(shape)?;
        let expected = checked_storage_byte_len(numel, dtype)?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        let context =
            crate::cuda::gpu::cuda_cache::try_get_cuda_device(device.ordinal()).map_err(|_| {
                Error::InvalidDeviceOrdinal {
                    backend: "Cuda",
                    ordinal: device.ordinal(),
                }
            })?;
        let data = context
            .default_stream()
            .clone_htod(bytes)
            .map_err(|error| Error::Msg(format!("CUDA upload failed: {error:?}")))?;
        let buffer = crate::cuda::storage::CudaBuffer {
            len: numel,
            dtype,
            data: Arc::new(data),
            device: context,
            device_id: device.ordinal(),
        };
        Ok(CudaStorage::new(Arc::new(buffer), shape.to_vec()))
    }
}

fn validate_cuda(dtype: DTypeDescriptor, device: &DeviceId, op: &'static str) -> Result<()> {
    validate_cuda_device(device)?;
    validate_cuda_storage_dtype(dtype, op)
}

pub(crate) fn validate_cuda_storage(
    dtype: DTypeDescriptor,
    device: &DeviceId,
    op: &'static str,
) -> Result<()> {
    validate_cuda_device(device)?;
    validate_cuda_storage_dtype(dtype, op)
}

fn validate_cuda_device(device: &DeviceId) -> Result<()> {
    if device.kind() != DeviceKind::Cuda {
        return Err(Error::DeviceInitializationError {
            expected: "cuda".into(),
            got: format!("{:?}", device.kind()),
        });
    }
    Ok(())
}

pub(crate) fn checked_storage_byte_len(numel: usize, dtype: DTypeDescriptor) -> Result<usize> {
    dtype
        .size_bytes(numel, incin_core::shapes::error::OperationKind::Storage)
        .map_err(Error::from)
}

pub(crate) fn cuda_from_f32(
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
    values: Vec<f32>,
    op: &'static str,
) -> Result<CudaStorage> {
    validate_cuda(dtype, device, op)?;
    cuda_from_bytes(
        shape,
        dtype,
        device.ordinal(),
        bytemuck::cast_slice(&values),
    )
}

pub(crate) fn cuda_from_bytes(
    shape: &[usize],
    dtype: DTypeDescriptor,
    ordinal: usize,
    bytes: &[u8],
) -> Result<CudaStorage> {
    validate_cuda_storage_dtype(dtype, "from_bytes")?;
    let numel = checked_numel(shape)?;
    let expected = checked_storage_byte_len(numel, dtype)?;
    if bytes.len() != expected {
        return Err(Error::InvalidByteLength {
            expected,
            got: bytes.len(),
        });
    }
    // Through the process-wide cache rather than CudaContext::new directly. Both
    // retain the same primary context, but a fresh Arc per tensor means the last
    // one dropped releases it, and the next allocation pays 131 ms to bring it
    // back. The cache holds one handle forever, which keeps this on the 1 us
    // path. See cuda::gpu::cuda_cache::try_get_cuda_device.
    let context = crate::cuda::gpu::cuda_cache::try_get_cuda_device(ordinal).map_err(|_| {
        Error::InvalidDeviceOrdinal {
            backend: "Cuda",
            ordinal,
        }
    })?;
    let data = context
        .default_stream()
        .clone_htod(bytes)
        .map_err(|error| Error::Msg(format!("CUDA upload failed: {error:?}")))?;
    let buffer = crate::cuda::storage::CudaBuffer {
        len: numel,
        dtype,
        data: Arc::new(data),
        device: context,
        device_id: ordinal,
    };
    Ok(CudaStorage::new(Arc::new(buffer), shape.to_vec()))
}

/// Guards `download_f32_host`/`upload_f32_from_host` callers against the
/// class of bug those two helpers cannot detect on their own: they assume
/// F32 storage unconditionally, so calling them on any of CUDA's other
/// storage dtypes (I64/BF16/F16/F64 - see `CUDA_STORAGE_DTYPES` in
/// `capability/constants.rs`) would silently reinterpret the wrong bytes rather than
/// error. `topk`/`argsort` (this file, `cuda_topk_host`/`cuda_argsort_host`)
/// have this exact gap already and are tracked separately; every new
/// F32-only host-round-trip op added in this pass checks first instead of
/// repeating it.
pub(crate) fn cuda_require_f32(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    if dtype != DTypeId::F32.descriptor() {
        return Err(Error::UnsupportedDType {
            dtype,
            backend: "cuda",
            op,
        });
    }
    Ok(())
}

/// Downloads an F32 `CudaStorage`'s raw contents to a host `Vec<f32>`.
pub(crate) fn download_f32_host(t: &CudaStorage) -> Result<Vec<f32>> {
    let bytes = t
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*t.buffer.data)
        .map_err(|error| BackendError::Execution {
            operation: OperationKind::Storage,
            message: format!("CUDA download failed: {error:?}").into(),
        })?;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}
