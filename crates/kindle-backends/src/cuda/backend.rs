use crate::cuda::storage::CudaStorage;
use crate::dtype_policy::{BackendFamily, OperationFamily, resolve_dtype_policy};
use alloc::sync::Arc;
use kindle_core::prelude::*;

/// Type alias for `KindleBackend<T, D>` with a CUDA device. Kept for backwards
/// compatibility — prefer `KindleBackend<T, Cuda>` in new code.
#[derive(Clone)]
pub struct CudaBackendImpl<T = f32, D = Cuda>(core::marker::PhantomData<(T, D)>);

impl<T: DType, D: Device> SupportsDType<f32> for CudaBackendImpl<T, D> {}

impl<T: DType, D: Device> SupportsDType<Dyn> for CudaBackendImpl<T, D> {
    fn resolve_dtype(field: &DTypeId, _device: &DeviceId) -> Result<DTypeId> {
        resolve_dtype_policy(BackendFamily::Cuda, OperationFamily::Fill, *field, "create")
            .map(|_| *field)
    }
}

#[derive(Clone)]
pub struct CudaVar {
    pub storage: CudaStorage,
}

pub type CudaGrads = crate::cuda::tape::CudaGrads;

impl<T: DType, D: Device> TensorOps<Self> for CudaBackendImpl<T, D> {
    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cuda::ops::shape::launch_concat(tensors, dim)
    }
}

impl<T: DType, D: Device> NumericOps<Self> for CudaBackendImpl<T, D> {
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("add", "a + b", lhs, rhs, &out_shape)?;
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                vec![
                    crate::cuda::tape::unbroadcast(grad_out, &lhs_shape)
                        .expect("unbroadcast lhs (add)"),
                    crate::cuda::tape::unbroadcast(grad_out, &rhs_shape)
                        .expect("unbroadcast rhs (add)"),
                ]
            }),
        });
        Ok(out)
    }

    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("sub", "a - b", lhs, rhs, &out_shape)?;
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let neg_grad =
                    crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", grad_out)
                        .expect("neg (sub backward)");
                vec![
                    crate::cuda::tape::unbroadcast(grad_out, &lhs_shape)
                        .expect("unbroadcast lhs (sub)"),
                    crate::cuda::tape::unbroadcast(&neg_grad, &rhs_shape)
                        .expect("unbroadcast rhs (sub)"),
                ]
            }),
        });
        Ok(out)
    }

    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("mul", "a * b", lhs, rhs, &out_shape)?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let grad_lhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &rhs_capture.shape)
                        .expect("mul backward shape (lhs)");
                let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &rhs_capture,
                    &grad_lhs_shape,
                )
                .expect("mul backward (lhs)");
                let grad_rhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &lhs_capture.shape)
                        .expect("mul backward shape (rhs)");
                let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &lhs_capture,
                    &grad_rhs_shape,
                )
                .expect("mul backward (rhs)");
                vec![
                    crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)
                        .expect("unbroadcast lhs (mul)"),
                    crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)
                        .expect("unbroadcast rhs (mul)"),
                ]
            }),
        });
        Ok(out)
    }

    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("div", "a / b", lhs, rhs, &out_shape)?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &rhs_capture.shape)
                        .expect("div backward shape (lhs)");
                let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "div",
                    "a / b",
                    grad_out,
                    &rhs_capture,
                    &grad_lhs_shape,
                )
                .expect("div backward (lhs)");
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let rhs_sq_shape =
                    crate::cpu::stride::broadcast_shape(&rhs_capture.shape, &rhs_capture.shape)
                        .expect("div backward shape (rhs^2)");
                let rhs_sq = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    &rhs_capture,
                    &rhs_capture,
                    &rhs_sq_shape,
                )
                .expect("rhs^2 (div backward)");
                let ratio_shape =
                    crate::cpu::stride::broadcast_shape(&lhs_capture.shape, &rhs_sq.shape)
                        .expect("div backward shape (ratio)");
                let lhs_over_rhs_sq = crate::cuda::ops::elementwise::launch_binary_op(
                    "div",
                    "a / b",
                    &lhs_capture,
                    &rhs_sq,
                    &ratio_shape,
                )
                .expect("lhs/rhs^2 (div backward)");
                let neg_ratio =
                    crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", &lhs_over_rhs_sq)
                        .expect("neg (div backward)");
                let grad_rhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &neg_ratio.shape)
                        .expect("div backward shape (rhs)");
                let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &neg_ratio,
                    &grad_rhs_shape,
                )
                .expect("div backward (rhs)");
                vec![
                    crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)
                        .expect("unbroadcast lhs (div)"),
                    crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)
                        .expect("unbroadcast rhs (div)"),
                ]
            }),
        });
        Ok(out)
    }
}

impl<T: DType, D: Device> FloatOps<Self> for CudaBackendImpl<T, D> {}
impl<T: DType, D: Device> CreationOps<Self> for CudaBackendImpl<T, D> {
    fn zeros<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaStorage> {
        cuda_from_f32(
            shape,
            dtype,
            device,
            vec![0.0; checked_numel(shape)?],
            "zeros",
        )
    }

    fn ones<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaStorage> {
        cuda_from_f32(
            shape,
            dtype,
            device,
            vec![1.0; checked_numel(shape)?],
            "ones",
        )
    }

    fn rand<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaStorage> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let values = (0..checked_numel(shape)?).map(|_| rng.r#gen()).collect();
        cuda_from_f32(shape, dtype, device, values, "rand")
    }

    fn randn<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaStorage> {
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand::thread_rng();
        let values = (0..checked_numel(shape)?)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        cuda_from_f32(shape, dtype, device, values, "randn")
    }

    fn var_zeros<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaVar> {
        Self::zeros::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    fn var_ones<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaVar> {
        Self::ones::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    fn var_rand<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaVar> {
        Self::rand::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    fn var_randn<K: DType>(shape: &[usize], dtype: DTypeId, device: &DeviceId) -> Result<CudaVar> {
        Self::randn::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }
}
impl<T: DType, D: Device> ReductionOps<Self> for CudaBackendImpl<T, D> {}
impl<T: DType, D: Device> QuantizedOps<Self> for CudaBackendImpl<T, D> {}
impl<T: DType, D: Device> OptimizerOps<Self> for CudaBackendImpl<T, D> {}
impl<T: DType, D: Device> ModuleOps<Self> for CudaBackendImpl<T, D> {}
impl<T: DType, D: Device> LossOps<Self> for CudaBackendImpl<T, D> {}

impl<T: DType, D: Device> Backend for CudaBackendImpl<T, D> {
    type Device = D;
    type FloatElem = T;
    type IntElem = i64;

    type Storage<K: DType> = CudaStorage;
    type RawVar = CudaVar;
    type Grads = CudaGrads;

    type InnerBackend = Self;

    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
        t.shape.clone()
    }
    fn storage_dtype<K: DType>(t: &Self::Storage<K>) -> Option<DTypeId> {
        Some(t.buffer.dtype)
    }
    fn storage_device<K: DType>(t: &Self::Storage<K>) -> Option<DeviceId> {
        Some(DeviceId::cuda(t.buffer.device_id))
    }
    fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
        "CudaTensor(...)".to_string()
    }
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        format!("CudaTensor(shape={:?})", t.shape)
    }
    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::cuda::tape::backward(loss)
    }
    fn backward_with_nan_check<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::cuda::tape::backward_with_nan_check(loss)
    }
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
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
        dtype: DTypeId,
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
        cuda_from_bytes(shape, dtype, device.ordinal(), bytes)
    }
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(CudaVar { storage: t.clone() })
    }
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }
}

fn validate_cuda(dtype: DTypeId, device: &DeviceId, op: &'static str) -> Result<()> {
    validate_cuda_device(device)?;
    resolve_dtype_policy(BackendFamily::Cuda, OperationFamily::Fill, dtype, op).map(|_| ())
}

fn validate_cuda_storage(dtype: DTypeId, device: &DeviceId, op: &'static str) -> Result<()> {
    validate_cuda_device(device)?;
    validate_cuda_storage_dtype(dtype, op)
}

fn validate_cuda_storage_dtype(dtype: DTypeId, op: &'static str) -> Result<()> {
    resolve_dtype_policy(BackendFamily::Cuda, OperationFamily::Storage, dtype, op).map(|_| ())
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

fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |numel, &dimension| {
        numel
            .checked_mul(dimension)
            .ok_or_else(|| Error::Msg(format!("CUDA tensor shape overflows usize: {shape:?}")))
    })
}

fn checked_storage_byte_len(numel: usize, dtype: DTypeId) -> Result<usize> {
    numel.checked_mul(dtype.element_size()).ok_or_else(|| {
        Error::Msg(format!(
            "CUDA storage byte length overflow: {numel} {:?} elements",
            dtype
        ))
    })
}

fn cuda_from_f32(
    shape: &[usize],
    dtype: DTypeId,
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

fn cuda_from_bytes(
    shape: &[usize],
    dtype: DTypeId,
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
    let context =
        cudarc::driver::CudaContext::new(ordinal).map_err(|_| Error::InvalidDeviceOrdinal {
            backend: "Cuda",
            ordinal,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_length_uses_authoritative_storage_dtype() {
        assert_eq!(checked_storage_byte_len(7, DTypeId::F16).unwrap(), 14);
        assert_eq!(checked_storage_byte_len(7, DTypeId::BF16).unwrap(), 14);
        assert_eq!(checked_storage_byte_len(7, DTypeId::F32).unwrap(), 28);
        assert_eq!(checked_storage_byte_len(7, DTypeId::F64).unwrap(), 56);
        assert!(checked_storage_byte_len(usize::MAX, DTypeId::F64).is_err());
    }

    #[test]
    fn storage_validation_accepts_renderable_float_family_only() {
        let device = DeviceId::cuda(0);
        for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
            validate_cuda_storage(dtype, &device, "test").unwrap();
        }
        assert!(matches!(
            validate_cuda_storage(DTypeId::I64, &device, "test"),
            Err(Error::UnsupportedDType { .. })
        ));
        assert!(validate_cuda_storage(DTypeId::F32, &DeviceId::cpu(), "test").is_err());
    }

    #[test]
    fn shape_cardinality_is_checked_before_allocation() {
        assert_eq!(checked_numel(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_numel(&[usize::MAX, 0]).unwrap(), 0);
        assert!(checked_numel(&[usize::MAX, 2]).is_err());
    }
}
