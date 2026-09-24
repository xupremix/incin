//! The backend-authoring trait implementations that make
//! `WgpuBackendImpl` a `StorageBackend`/`Backend`/`HostInterop`.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Backend core trait
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> incin_core::backend_authoring::StorageBackend for WgpuBackendImpl<D> {
    type Device = D;
    const BACKEND_NAME: &'static str = "Wgpu";
    type Storage<K: DType> = WgpuStorage;

    fn metadata<K: DType>(t: &Self::Storage<K>) -> &incin_core::backend_authoring::TensorMeta {
        let t: &WgpuStorage = t;
        &t.meta
    }

    fn fresh_autograd_identity<K: DType>(storage: Self::Storage<K>) -> Self::Storage<K> {
        storage.with_fresh_autograd_identity()
    }
}

impl incin_core::backend_authoring::StorageOutput for WgpuStorage {}

// A variable is a trainable handle, not storage, so it carries `ExecuteOutput`
// directly rather than through `StorageOutput`. `CpuVar` does the same; the
// `var_*` creation executors are what need it.
impl incin_core::backend_authoring::ExecuteOutput for super::types::WgpuVar {}

impl<D: Device> Backend for WgpuBackendImpl<D> {
    /// `Grads`.
    /// `InnerBackend`.
    type InnerBackend = Self;

    // `host_format_display`/`host_format_debug` use `HostInterop`'s default,
    // which reads real values back through `float_to_vec1`/`int_to_vec1`.
}

impl<D: Device> incin_core::backend_authoring::HostReadback for WgpuBackendImpl<D> {
    fn float_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<f64>> {
        let t: &WgpuStorage = t;
        match t.dtype.builtin_id() {
            Some(DTypeId::F32 | DTypeId::Bool) => {
                // `Bool` is physically `f32` (0.0/1.0); the values are already
                // the numerals the API reports.
                let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
                Ok(data.iter().map(|&x| x as f64).collect())
            }
            Some(DTypeId::U8) => {
                let data: Vec<u8> = t.buffer.to_vec::<u8>()?;
                Ok(data.into_iter().map(f64::from).collect())
            }
            Some(DTypeId::U32) => {
                let data: Vec<u32> = t.buffer.to_vec::<u32>()?;
                Ok(data.into_iter().map(f64::from).collect())
            }
            Some(DTypeId::I64) => {
                let data: Vec<i64> = t.buffer.to_vec::<i64>()?;
                Ok(data.into_iter().map(|x| x as f64).collect())
            }
            _ => Err(Error::UnsupportedDType {
                dtype: t.dtype,
                backend: "Wgpu",
                op: "float_to_vec1",
            }),
        }
    }

    fn int_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<i64>> {
        let t: &WgpuStorage = t;
        match t.dtype.builtin_id() {
            Some(DTypeId::I64) => t.buffer.to_vec::<i64>(),
            Some(DTypeId::U32) => Ok(t
                .buffer
                .to_vec::<u32>()?
                .into_iter()
                .map(i64::from)
                .collect()),
            Some(DTypeId::U8) => Ok(t
                .buffer
                .to_vec::<u8>()?
                .into_iter()
                .map(i64::from)
                .collect()),
            Some(DTypeId::Bool) => Ok(t
                .buffer
                .to_vec::<f32>()?
                .into_iter()
                .map(|x| i64::from(x != 0.0))
                .collect()),
            Some(DTypeId::F32) => t
                .buffer
                .to_vec::<f32>()?
                .into_iter()
                .map(|value| {
                    incin_core::error::convert_f64_to_i64(
                        "int_to_vec1",
                        t.dtype,
                        f64::from(value),
                        incin_core::error::FloatToIntPolicy::Exact,
                    )
                })
                .collect(),
            _ => Err(Error::UnsupportedDType {
                dtype: t.dtype,
                backend: "Wgpu",
                op: "int_to_vec1",
            }),
        }
    }
}

impl<D: Device> incin_core::backend_authoring::HostInterop for WgpuBackendImpl<D> {
    /// `to_bytes`.
    ///
    /// The expected length is the *logical* `dtype.size_bytes(numel)` — for
    /// `bool` that is one byte per element of 0 or 1, not the four physical
    /// `f32` bytes on the device. A non-`bool` buffer's physical size already
    /// equals its logical size, so the raw download is returned as-is after
    /// the length check. A `bool` buffer whose physical values are not
    /// exactly 0.0 or 1.0 is refused rather than truncated to a byte.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<Vec<u8>> {
        let t: &WgpuStorage = t;
        let numel = num_elements(&t.shape)?;
        let expected = t.dtype.size_bytes(numel, OperationKind::Storage)?;
        if t.dtype.builtin_id() == Some(DTypeId::Bool) {
            let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
            if data.len() != numel {
                return Err(Error::InvalidByteLength {
                    expected: numel,
                    got: data.len(),
                });
            }
            let mut out = Vec::with_capacity(numel);
            for value in data {
                if value == 0.0 {
                    out.push(0u8);
                } else if value == 1.0 {
                    out.push(1u8);
                } else {
                    return Err(Error::Msg(alloc::format!(
                        "WGPU bool storage holds {value}, which is neither 0.0 nor 1.0; \
                         refusing to pack it as a bool byte"
                    )));
                }
            }
            Ok(out)
        } else {
            let bytes = t.buffer.to_vec::<u8>()?;
            if bytes.len() != expected {
                return Err(Error::InvalidByteLength {
                    expected,
                    got: bytes.len(),
                });
            }
            Ok(bytes)
        }
    }
    /// `from_bytes`.
    ///
    /// The expected length is the logical `dtype.size_bytes(numel)`. For
    /// `bool` the input is one byte per element of 0 or 1 and is expanded to
    /// the physical `f32` buffer the device holds; any other byte value is
    /// refused. Every other dtype uploads its bytes unchanged at its own
    /// physical width.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Storage, "from_bytes")?;
        let numel = num_elements(shape)?;
        let expected = dtype.size_bytes(numel, OperationKind::Storage)?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        if dtype.builtin_id() == Some(DTypeId::Bool) {
            let mut physical = Vec::with_capacity(numel);
            for &byte in bytes {
                match byte {
                    0 => physical.push(0.0f32),
                    1 => physical.push(1.0f32),
                    other => {
                        return Err(Error::Msg(alloc::format!(
                            "WGPU from_bytes expected a bool byte (0 or 1) for {dtype:?}, \
                             got {other}"
                        )));
                    }
                }
            }
            let buffer = WgpuBuffer::try_from_slice(&physical)?;
            Ok(WgpuStorage::new_with_dtype(buffer, shape.to_vec(), dtype))
        } else {
            let buffer = WgpuBuffer::try_from_slice(bytes)?;
            Ok(WgpuStorage::new_with_dtype(buffer, shape.to_vec(), dtype))
        }
    }
}
