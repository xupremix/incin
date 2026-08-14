//! Core `Backend`, `SupportsDType`, and `TransferTo` implementations.

use crate::external::candle::CandleBackend;
use crate::external::candle::convert::{from_candle_dtype, to_candle_device, to_candle_dtype};
use crate::external::candle::executor::CandleStorage;
use crate::external::*;
use candle_core as candle;

impl<D: incin_core::prelude::Device> incin_core::prelude::Backend for CandleBackend<D> {

    type InnerBackend = Self;
}

impl<D: incin_core::prelude::Device> incin_core::backend_authoring::HostReadback
    for CandleBackend<D>
{
    fn float_to_vec1<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<Vec<f64>> {
        let values: Vec<f32> = t
            .tensor()
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| anyhow::anyhow!(e))?
            .to_vec1()
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(values.into_iter().map(f64::from).collect())
    }

    fn int_to_vec1<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<Vec<i64>> {
        let tensor = t.tensor();
        match tensor.dtype() {
            candle_core::DType::U8 => Ok(tensor
                .to_vec1::<u8>()
                .map_err(|e| anyhow::anyhow!(e))?
                .into_iter()
                .map(i64::from)
                .collect()),
            candle_core::DType::U32 => Ok(tensor
                .to_vec1::<u32>()
                .map_err(|e| anyhow::anyhow!(e))?
                .into_iter()
                .map(i64::from)
                .collect()),
            candle_core::DType::I64 => Ok(tensor
                .to_vec1::<i64>()
                .map_err(|e| anyhow::anyhow!(e))?),
            dtype => {
                let values: Vec<f64> = match dtype {
                    candle_core::DType::F16 => tensor
                        .to_vec1::<half::f16>()
                        .map_err(|e| anyhow::anyhow!(e))?
                        .into_iter()
                        .map(|v| v.to_f64())
                        .collect(),
                    candle_core::DType::BF16 => tensor
                        .to_vec1::<half::bf16>()
                        .map_err(|e| anyhow::anyhow!(e))?
                        .into_iter()
                        .map(|v| v.to_f64())
                        .collect(),
                    candle_core::DType::F32 => tensor
                        .to_vec1::<f32>()
                        .map_err(|e| anyhow::anyhow!(e))?
                        .into_iter()
                        .map(f64::from)
                        .collect(),
                    candle_core::DType::F64 => tensor
                        .to_vec1::<f64>()
                        .map_err(|e| anyhow::anyhow!(e))?,
                    candle_core::DType::U8
                    | candle_core::DType::U32
                    | candle_core::DType::I64 => unreachable!(),
                };
                values
                    .into_iter()
                    .map(|value| {
                        incin_core::prelude::convert_f64_to_i64(
                            "candle_int_to_vec1",
                            from_candle_dtype(dtype),
                            value,
                            incin_core::prelude::FloatToIntPolicy::Exact,
                        )
                    })
                    .collect()
            }
        }
    }
}

impl<D: incin_core::prelude::Device> incin_core::backend_authoring::HostInterop for CandleBackend<D> {
    /// Formats the tensor using candle's own `Display` implementation.
    fn host_format_display<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> alloc::string::String {
        std::format!("{}", t.tensor())
    }
    /// Formats the tensor's raw contents together with its strides, for
    /// debugging.
    fn host_format_debug<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> alloc::string::String {
        std::format!(
            "Raw Tensor: {:?}, Strides: {:?}",
            t.tensor(),
            t.tensor().stride()
        )
    }

    /// Flattens the tensor and returns its raw byte representation according to its actual dtype.
        fn to_bytes<K: incin_core::prelude::DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<u8>> {
            let flat = t
                .tensor()
                .flatten_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            match flat.dtype() {
                candle_core::DType::F32 => {
                    let v = flat
                        .to_vec1::<f32>()
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    Ok(bytemuck::cast_slice(&v).to_vec())
                }
                candle_core::DType::F64 => {
                    let v = flat
                        .to_vec1::<f64>()
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    Ok(bytemuck::cast_slice(&v).to_vec())
                }
                candle_core::DType::U8 => {
                    let v = flat
                        .to_vec1::<u8>()
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    Ok(v)
                }
                candle_core::DType::U32 => {
                    let v = flat
                        .to_vec1::<u32>()
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    Ok(bytemuck::cast_slice(&v).to_vec())
                }
                candle_core::DType::I64 => {
                    let v = flat
                        .to_vec1::<i64>()
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    Ok(bytemuck::cast_slice(&v).to_vec())
                }
                candle_core::DType::F16 => {
                    let v = flat
                        .to_vec1::<half::f16>()
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    Ok(bytemuck::cast_slice(&v).to_vec())
                }
                candle_core::DType::BF16 => {
                    let v = flat
                        .to_vec1::<half::bf16>()
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    Ok(bytemuck::cast_slice(&v).to_vec())
                }
            }
        }
    /// Reinterprets `bytes` as typed scalar elements matching `dtype`, constructs
        /// a tensor with `shape` on `device`, and returns it.
        fn from_bytes<K: incin_core::prelude::DType>(
            bytes: &[u8],
            shape: &[usize],
            dtype: DTypeDescriptor,
            device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let d = to_candle_device(device)?;
            let c_dtype = to_candle_dtype(dtype)?;
            let expected_bytes = dtype.size_bytes(
                shape.iter().copied().product(),
                incin_core::shapes::error::OperationKind::Storage,
            )?;
            if bytes.len() != expected_bytes {
                return Err(anyhow::anyhow!(
                    "Byte length mismatch in Candle from_bytes: expected {} bytes for shape {:?} and dtype {:?}, got {}",
                    expected_bytes, shape, dtype, bytes.len()
                ).into());
            }

            let raw = match c_dtype {
                candle_core::DType::F32 => {
                    let floats: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect();
                    candle_core::Tensor::from_slice(&floats, shape, &d)
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                }
                candle_core::DType::F64 => {
                    let doubles: Vec<f64> = bytes
                        .chunks_exact(8)
                        .map(|chunk| f64::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect();
                    candle_core::Tensor::from_slice(&doubles, shape, &d)
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                }
                candle_core::DType::U8 => candle_core::Tensor::from_slice(bytes, shape, &d)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
                candle_core::DType::U32 => {
                    let uints: Vec<u32> = bytes
                        .chunks_exact(4)
                        .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect();
                    candle_core::Tensor::from_slice(&uints, shape, &d)
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                }
                candle_core::DType::I64 => {
                    let ints: Vec<i64> = bytes
                        .chunks_exact(8)
                        .map(|chunk| i64::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect();
                    candle_core::Tensor::from_slice(&ints, shape, &d)
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                }
                candle_core::DType::F16 => {
                    let f16s: Vec<half::f16> = bytes
                        .chunks_exact(2)
                        .map(|chunk| half::f16::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect();
                    candle_core::Tensor::from_slice(&f16s, shape, &d)
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                }
                candle_core::DType::BF16 => {
                    let bf16s: Vec<half::bf16> = bytes
                        .chunks_exact(2)
                        .map(|chunk| half::bf16::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect();
                    candle_core::Tensor::from_slice(&bf16s, shape, &d)
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                }
            };
            CandleStorage::try_new(raw)
        }
}

impl<K: DType, D: Device> SupportsDType<K> for CandleBackend<D> {
    fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
        let descriptor = K::descriptor(field);
        to_candle_dtype(descriptor)?;
        Ok(descriptor)
    }
}

impl<D: Device> incin_core::exec::PrecisionCapabilities for CandleBackend<D> {
    fn native_precision(
        &self,
        request: &incin_core::exec::PrecisionRequest,
    ) -> Result<incin_core::exec::ResolvedPrecision> {
        to_candle_dtype(request.storage)?;
        Ok(incin_core::exec::ResolvedPrecision::new(
            request.storage,
            request.storage,
            request.storage,
            request.output,
            incin_core::exec::LossScaling::None,
        ))
    }
}

impl<D, NewD> incin_core::prelude::StorageTransfer<NewD> for CandleBackend<D>
where
    D: incin_core::prelude::Device,
    NewD: incin_core::prelude::Device,
{
    type Output = CandleBackend<NewD>;

    fn transfer_storage<K: incin_core::prelude::DType>(
        storage: &Self::Storage<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as StorageBackend>::Storage<K>>
    where
        Self::Output: SupportsDType<K>,
    {
        let destination = NewD::to_incin(device)?;
        <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &destination)?;
        let target = to_candle_device(&destination)?;
        let transferred = storage
            .tensor()
            .to_device(&target)
            .map_err(|error| anyhow::anyhow!(error))?;
        CandleStorage::try_new(transferred)
    }

}

impl<D, NewD> incin_core::prelude::TransferTo<NewD> for CandleBackend<D>
where
    D: incin_core::prelude::Device,
    NewD: incin_core::prelude::Device,
{
    fn transfer_var<K: incin_core::prelude::DType>(
        variable: &Self::Var<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as VariableBackend>::Var<K>>
    where
        Self::Output: SupportsDType<K>,
    {
        let storage = <Self as VariableBackend>::var_as_tensor::<K>(variable)?;
        let transferred = <Self as incin_core::prelude::StorageTransfer<NewD>>::transfer_storage(
            &storage, dtype, device,
        )?;
        <Self::Output as VariableBackend>::var_from_tensor::<K>(&transferred)
    }
}


impl<D: incin_core::prelude::Device> incin_core::backend_authoring::AutogradBackend for CandleBackend<D> {
    type Grads = candle_core::backprop::GradStore;

    fn backward<K: incin_core::prelude::DType>(
        loss: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<Self::Grads> {
        loss.tensor()
            .backward()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
    }

    fn get_grad<K: incin_core::prelude::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<<Self as StorageBackend>::Storage<K>>> {
        if let Some(grad) = grads.get(t.tensor()).cloned() {
            Ok(Some(CandleStorage::try_new(grad)?))
        } else {
            Ok(None)
        }
    }
}

impl<D: incin_core::prelude::Device> incin_core::backend_authoring::VariableBackend for CandleBackend<D> {
    type Var<K: DType> = candle_core::Var;

    fn var_as_tensor<K: incin_core::prelude::DType>(
        var: &Self::Var<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        CandleStorage::try_new(var.as_tensor().clone())
    }

    fn var_from_tensor<K: incin_core::prelude::DType>(
        t: &Self::Storage<K>,
    ) -> Result<Self::Var<K>> {
        Ok(candle::Var::from_tensor(t.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    fn assign_var<K: incin_core::prelude::DType>(
        var: &mut Self::Var<K>,
        tensor: &Self::Storage<K>,
    ) -> Result<()> {
        var.set(tensor.tensor())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
    }
}
