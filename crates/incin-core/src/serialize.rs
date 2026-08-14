use crate::nn::{StatePath, StateRole, StateSnapshot, StateValue};
use crate::prelude::*;
use alloc::collections::BTreeMap;

#[cfg(feature = "std")]
fn safetensors_dtype(dtype: DTypeDescriptor) -> anyhow::Result<safetensors::tensor::Dtype> {
    use safetensors::tensor::Dtype;
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok(Dtype::F32),
        Some(DTypeId::F64) => Ok(Dtype::F64),
        Some(DTypeId::F16) => Ok(Dtype::F16),
        Some(DTypeId::BF16) => Ok(Dtype::BF16),
        Some(DTypeId::U32) => Ok(Dtype::U32),
        Some(DTypeId::I64) => Ok(Dtype::I64),
        Some(DTypeId::U8) => Ok(Dtype::U8),
        Some(DTypeId::Bool) => Ok(Dtype::BOOL),
        _ => Err(anyhow::anyhow!(
            "unsupported safetensors dtype {}",
            dtype.name()
        )),
    }
}

#[cfg(feature = "std")]
fn dtype_from_safetensors(dtype: safetensors::tensor::Dtype) -> anyhow::Result<DTypeDescriptor> {
    Ok(match dtype {
        safetensors::tensor::Dtype::F32 => DTypeId::F32,
        safetensors::tensor::Dtype::F64 => DTypeId::F64,
        safetensors::tensor::Dtype::F16 => DTypeId::F16,
        safetensors::tensor::Dtype::BF16 => DTypeId::BF16,
        safetensors::tensor::Dtype::U32 => DTypeId::U32,
        safetensors::tensor::Dtype::I64 => DTypeId::I64,
        safetensors::tensor::Dtype::U8 => DTypeId::U8,
        safetensors::tensor::Dtype::BOOL => DTypeId::Bool,
        _ => return Err(anyhow::anyhow!("unsupported dtype in safetensors")),
    }
    .descriptor())
}

#[cfg(feature = "std")]
pub(crate) fn serialize_snapshot_safetensors(
    snapshot: &StateSnapshot,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    use safetensors::tensor::TensorView;
    let mut storage = Vec::new();
    let mut views = BTreeMap::new();
    for (name, value) in snapshot.iter() {
        storage.push((
            name.as_str().to_owned(),
            value.bytes().to_vec(),
            value.shape().dims().to_vec(),
            safetensors_dtype(value.dtype())?,
        ));
    }
    for (name, bytes, shape, dtype) in &storage {
        views.insert(name.clone(), TensorView::new(*dtype, shape.clone(), bytes)?);
    }
    safetensors::tensor::serialize_to_file(&views, &None, path)?;
    Ok(())
}

#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_safetensors(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    let bytes = std::fs::read(path)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut snapshot = StateSnapshot::new();
    for (name, view) in tensors.tensors() {
        snapshot.insert(
            StatePath::new(name)?,
            StateValue::new(
                ShapeBuf::from_slice(view.shape()),
                dtype_from_safetensors(view.dtype())?,
                view.data().to_vec(),
                StateRole::Parameter,
            )?,
        )?;
    }
    Ok(snapshot)
}

#[cfg(feature = "std")]
pub(crate) fn serialize_snapshot_postcard(
    snapshot: &StateSnapshot,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    std::fs::write(path, postcard::to_stdvec(snapshot)?)?;
    Ok(())
}

#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_postcard(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    Ok(postcard::from_bytes(&std::fs::read(path)?)?)
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Postcard,
    ONNX,
}

#[cfg(feature = "std")]
pub trait ModelExt<B: Backend> {
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default;

    fn load(&mut self, format: Format, path: &std::path::Path, device: &DeviceId) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
        B: crate::backend_authoring::Execute<crate::backend_authoring::op::TensorFromBytes>,
        <B as crate::backend_authoring::Execute<crate::backend_authoring::op::TensorFromBytes>>::Output:
            Into<B::Storage<f32>>;
}

#[cfg(feature = "std")]
impl<B: Backend, T: crate::nn::module::StateDict<B>> ModelExt<B> for T {
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
    {
        match format {
            Format::Safetensors => serialize_snapshot_safetensors(&self.state_dict()?, path),
            Format::Postcard => serialize_snapshot_postcard(&self.state_dict()?, path),
            Format::ONNX => Err(anyhow::anyhow!("ONNX is not a state format")),
        }
        .map_err(|e| Error::Msg(e.to_string()))
    }

    fn load(&mut self, format: Format, path: &std::path::Path, _device: &DeviceId) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
        B: crate::backend_authoring::Execute<crate::backend_authoring::op::TensorFromBytes>,
        <B as crate::backend_authoring::Execute<crate::backend_authoring::op::TensorFromBytes>>::Output:
            Into<B::Storage<f32>>,
    {
        let snapshot = match format {
            Format::Safetensors => deserialize_snapshot_safetensors(path),
            Format::Postcard => deserialize_snapshot_postcard(path),
            Format::ONNX => return Err(Error::Msg("ONNX is not a state format".into())),
        }
        .map_err(|e| Error::Msg(e.to_string()))?;
        self.load_state_dict(&snapshot)
    }
}
