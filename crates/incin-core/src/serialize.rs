use crate::nn::{StatePath, StateRole, StateSnapshot, StateValue};
use crate::prelude::*;
use alloc::{collections::BTreeMap, string::String, vec::Vec};

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
    let mut metadata = std::collections::HashMap::new();
    for (name, value) in snapshot.iter() {
        metadata.insert(
            format!("incin.state.role.{}", name.as_str()),
            match value.role() {
                StateRole::Parameter => "parameter".to_string(),
                StateRole::Buffer => "buffer".to_string(),
            },
        );
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
    safetensors::tensor::serialize_to_file(&views, &Some(metadata), path)?;
    Ok(())
}

#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_safetensors(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    let bytes = std::fs::read(path)?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let mut snapshot = StateSnapshot::new();
    let metadata = header.metadata().as_ref();
    for (name, view) in tensors.tensors() {
        let role = metadata
            .and_then(|items| items.get(&format!("incin.state.role.{}", name)))
            .map(|role| match role.as_str() {
                "buffer" => StateRole::Buffer,
                _ => StateRole::Parameter,
            })
            .unwrap_or(StateRole::Parameter);
        snapshot.insert(
            StatePath::new(name)?,
            StateValue::new(
                ShapeBuf::from_slice(view.shape()),
                dtype_from_safetensors(view.dtype())?,
                view.data().to_vec(),
                role,
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
    let wire = snapshot
        .iter()
        .map(|(path, value)| StateWireEntry {
            path: path.as_str().to_string(),
            shape: value.shape().dims().to_vec(),
            dtype: value.dtype(),
            bytes: value.bytes().to_vec(),
            role: value.role(),
        })
        .collect::<Vec<_>>();
    std::fs::write(path, postcard::to_stdvec(&wire)?)?;
    Ok(())
}

#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_postcard(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    let wire: Vec<StateWireEntry> = postcard::from_bytes(&std::fs::read(path)?)?;
    let mut snapshot = StateSnapshot::new();
    for entry in wire {
        snapshot.insert(
            StatePath::new(entry.path)?,
            StateValue::new(
                ShapeBuf::from_slice(&entry.shape),
                entry.dtype,
                entry.bytes,
                entry.role,
            )?,
        )?;
    }
    Ok(snapshot)
}

#[cfg(feature = "std")]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StateWireEntry {
    path: String,
    shape: Vec<usize>,
    dtype: DTypeDescriptor,
    bytes: Vec<u8>,
    role: StateRole,
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
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default;
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

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn fixture() -> StateSnapshot {
        let mut snapshot = StateSnapshot::new();
        for (index, dtype) in [
            DTypeId::F32,
            DTypeId::F16,
            DTypeId::BF16,
            DTypeId::I64,
            DTypeId::U32,
            DTypeId::U8,
            DTypeId::Bool,
        ]
        .into_iter()
        .enumerate()
        {
            let descriptor = dtype.descriptor();
            let byte_len = descriptor
                .size_bytes(32, crate::shapes::error::OperationKind::Storage)
                .expect("fixture dtype has storage bytes");
            snapshot
                .insert(
                    StatePath::new(format!("entry_{index}")).expect("fixture path is canonical"),
                    StateValue::new(
                        ShapeBuf::from_slice(&[32]),
                        descriptor,
                        vec![index as u8; byte_len],
                        if index % 2 == 0 {
                            StateRole::Parameter
                        } else {
                            StateRole::Buffer
                        },
                    )
                    .expect("fixture value is valid"),
                )
                .expect("fixture paths are unique");
        }
        snapshot
    }

    #[test]
    fn safetensors_round_trips_exact_supported_native_dtypes() {
        let path = std::env::temp_dir().join(format!(
            "incin-state-serialize-{}.safetensors",
            std::process::id()
        ));
        let expected = fixture();
        serialize_snapshot_safetensors(&expected, &path).expect("serialize snapshot");
        let actual = deserialize_snapshot_safetensors(&path).expect("deserialize snapshot");
        assert_eq!(actual, expected);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn postcard_round_trips_exact_supported_native_dtypes() {
        let path = std::env::temp_dir().join(format!(
            "incin-state-serialize-{}.postcard",
            std::process::id()
        ));
        let expected = fixture();
        serialize_snapshot_postcard(&expected, &path).expect("serialize snapshot");
        let actual = deserialize_snapshot_postcard(&path).expect("deserialize snapshot");
        assert_eq!(actual, expected);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn postcard_supports_q8_and_safetensors_rejects_it_explicitly() {
        let descriptor = DTypeId::Q8_0.descriptor();
        let bytes = descriptor
            .size_bytes(32, crate::shapes::error::OperationKind::Storage)
            .expect("q8 fixture has storage bytes");
        let mut snapshot = StateSnapshot::new();
        snapshot
            .insert(
                StatePath::new("quantized").expect("canonical path"),
                StateValue::new(
                    ShapeBuf::from_slice(&[32]),
                    descriptor,
                    vec![0; bytes],
                    StateRole::Parameter,
                )
                .expect("q8 fixture is valid"),
            )
            .expect("unique path");
        let postcard_path =
            std::env::temp_dir().join(format!("incin-state-q8-{}.postcard", std::process::id()));
        serialize_snapshot_postcard(&snapshot, &postcard_path).expect("serialize q8");
        assert_eq!(
            deserialize_snapshot_postcard(&postcard_path).expect("deserialize q8"),
            snapshot
        );
        std::fs::remove_file(&postcard_path).ok();

        let safetensors_path =
            std::env::temp_dir().join(format!("incin-state-q8-{}.safetensors", std::process::id()));
        assert!(serialize_snapshot_safetensors(&snapshot, &safetensors_path).is_err());
        std::fs::remove_file(safetensors_path).ok();
    }
}
