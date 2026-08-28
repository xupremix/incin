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
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        Ok(data.iter().map(|&x| x as f64).collect())
    }

    fn int_to_vec1<K: DType>(t: &Self::Storage<K>) -> Result<Vec<i64>> {
        let t: &WgpuStorage = t;
        let data: Vec<f32> = t.buffer.to_vec::<f32>()?;
        data.into_iter()
            .map(|value| {
                incin_core::error::convert_f64_to_i64(
                    "int_to_vec1",
                    t.dtype,
                    f64::from(value),
                    incin_core::error::FloatToIntPolicy::Exact,
                )
            })
            .collect()
    }
}

impl<D: Device> incin_core::backend_authoring::HostInterop for WgpuBackendImpl<D> {
    /// `to_bytes`.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<Vec<u8>> {
        let t: &WgpuStorage = t;
        t.buffer.to_vec::<u8>()
    }
    /// `from_bytes`.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Storage, "from_bytes")?;
        let expected = num_elements(shape)?
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(incin_core::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "WGPU element count * element byte width",
            })?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        let buffer = WgpuBuffer::try_from_slice(bytes)?;
        Ok(WgpuStorage::new(buffer, shape.to_vec()))
    }
}
