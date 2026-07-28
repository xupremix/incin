use crate::nn::StateDict;
use crate::prelude::*;
use alloc::collections::BTreeMap;
use safetensors::SafeTensors;
use std::path::Path;

/// Loads raw storage tensors from a safetensors file into a dictionary.
pub fn load_safetensors_map<B, P>(
    path: P,
    device: &DeviceId,
) -> Result<BTreeMap<String, B::Storage<B::FloatElem>>>
where
    B: Backend,
    P: AsRef<Path>,
    B: SupportsDType<B::FloatElem>,
{
    let buffer = std::fs::read(path)
        .map_err(|e| Error::Msg(format!("Failed to read safetensors file: {}", e)))?;
    let tensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| Error::Msg(format!("Safetensors deserialization failed: {:?}", e)))?;

    let mut mapped_tensors = BTreeMap::new();

    for (name, view) in tensors.tensors() {
        let shape = view.shape().to_vec();
        let bytes = view.data();
        let st_dtype = view.dtype();
        let dtype = match st_dtype {
            safetensors::Dtype::F32 => DTypeId::F32,
            safetensors::Dtype::F64 => DTypeId::F64,
            safetensors::Dtype::F16 => DTypeId::F16,
            safetensors::Dtype::BF16 => DTypeId::BF16,
            safetensors::Dtype::I64 => DTypeId::I64,
            safetensors::Dtype::U32 | safetensors::Dtype::I32 => DTypeId::U32,
            safetensors::Dtype::U8 | safetensors::Dtype::BOOL => DTypeId::U8,
            other => {
                return Err(Error::Msg(format!(
                    "Unsupported safetensors dtype {:?} for tensor {}",
                    other, name
                )));
            }
        };

        let inner = B::from_bytes::<B::FloatElem>(bytes, &shape, dtype, device)?;
        mapped_tensors.insert(name.to_string(), inner);
    }

    Ok(mapped_tensors)
}

/// Loads weights into a module from a safetensors file.
pub fn load_safetensors<B, M, P>(module: &mut M, path: P) -> Result<()>
where
    B: Backend,
    M: StateDict<B>,
    P: AsRef<Path>,
    B: SupportsDType<B::FloatElem>,
    <<B as Backend>::Device as Device>::Field: Default,
    <<B as Backend>::FloatElem as DType>::Field: Default,
{
    let map = load_safetensors_map::<B, _>(path, &DeviceId::cpu())?;
    let mut mapped_tensors = BTreeMap::new();

    for (name, storage) in map {
        let shape = B::shape(&storage);
        let tensor = Tensor::<Dyn, B>::from_parts(
            storage,
            shape,
            Default::default(),
            Default::default(),
            core::marker::PhantomData,
        )?;
        mapped_tensors.insert(name, tensor);
    }

    module.load_state_dict("", &mapped_tensors)?;
    Ok(())
}

/// Saves the module's weights to a safetensors file.
pub fn save_safetensors<B, M, P>(module: &M, path: P) -> Result<()>
where
    B: Backend,
    M: StateDict<B>,
    P: AsRef<Path>,
{
    let mut mapped_tensors = BTreeMap::new();
    module.state_dict("", &mut mapped_tensors);

    let mut raw_data: Vec<(String, Vec<usize>, safetensors::Dtype, Vec<u8>)> = Vec::new();

    for (name, tensor) in mapped_tensors {
        let bytes = B::to_bytes::<B::FloatElem>(tensor.inner())?;
        let shape = B::shape::<B::FloatElem>(tensor.inner());
        let dtype_id = B::storage_dtype::<B::FloatElem>(tensor.inner()).unwrap_or(DTypeId::F32);

        let st_dtype = match dtype_id {
            DTypeId::F32 => safetensors::Dtype::F32,
            DTypeId::F64 => safetensors::Dtype::F64,
            DTypeId::F16 => safetensors::Dtype::F16,
            DTypeId::BF16 => safetensors::Dtype::BF16,
            DTypeId::I64 => safetensors::Dtype::I64,
            DTypeId::U32 => safetensors::Dtype::U32,
            DTypeId::U8 => safetensors::Dtype::U8,
            other => {
                return Err(Error::Msg(format!(
                    "Unsupported DTypeId {:?} for safetensors export",
                    other
                )));
            }
        };

        raw_data.push((name, shape, st_dtype, bytes));
    }

    let mut data_map: BTreeMap<String, safetensors::tensor::TensorView> = BTreeMap::new();
    for (name, shape, st_dtype, bytes) in &raw_data {
        let view = safetensors::tensor::TensorView::new(*st_dtype, shape.clone(), bytes)
            .map_err(|e| Error::Msg(format!("TensorView creation failed: {:?}", e)))?;
        data_map.insert(name.clone(), view);
    }

    let serialized = safetensors::serialize(&data_map, &None)
        .map_err(|e| Error::Msg(format!("Safetensors serialization failed: {:?}", e)))?;

    std::fs::write(path, serialized)
        .map_err(|e| Error::Msg(format!("Failed to write safetensors: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    // Integration tests for load_safetensors and save_safetensors are located in `crates/incin-backends/tests/safetensors_test.rs`.
}
