use crate::backend_authoring::{Execute, op};
use crate::exec::catalog::DataAttributes;
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::prelude::*;
use alloc::collections::BTreeMap;

fn decode_bytes<B>(
    bytes: &[u8],
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<B::Storage<f32>>
where
    B: Backend + Execute<op::TensorFromBytes>,
    <B as Execute<op::TensorFromBytes>>::Output: Into<B::Storage<f32>>,
{
    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(shape)).map_err(Error::Shape)?;
    let context =
        ExecutionContext::from_scope(B::default()).with_grad_mode(crate::exec::GradMode::Disabled);
    Ok(dispatch::execute_shaped::<op::TensorFromBytes, B, Dyn>(
        &context,
        DataAttributes {
            shape: shape.to_vec(),
            dtype,
            device: *device,
            bytes: bytes.to_vec(),
        },
        &[],
        &expected,
    )
    .map(Into::into)?)
}

/// A trait for serializing a collection of dynamic tensors to a specific format.
pub trait Serializer {
    /// The error type returned if the forward pass fails.
    type Error: core::fmt::Debug + core::fmt::Display;

    /// Serializes the state dict to the given path or stream.
    fn serialize<B: Backend>(
        &mut self,
        state_dict: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default;
}

/// A trait for deserializing a collection of dynamic tensors from a specific format.
pub trait Deserializer {
    /// The error type returned if the forward pass fails.
    type Error: core::fmt::Debug + core::fmt::Display;

    /// Deserializes the state dict from the given path or stream.
    fn deserialize<B: Backend + Execute<op::TensorFromBytes>>(
        &mut self,
        device: &DeviceId,
    ) -> core::result::Result<BTreeMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
        <B as Execute<op::TensorFromBytes>>::Output: Into<B::Storage<f32>>;
}

#[cfg(feature = "std")]
/// `SafetensorsSerializer`.
pub struct SafetensorsSerializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> SafetensorsSerializer<'a> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Serializer for SafetensorsSerializer<'a> {
    /// The error type returned if the forward pass fails.
    type Error = anyhow::Error;

    /// `serialize`.
    fn serialize<B: Backend>(
        &mut self,
        state_dict: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
    {
        use safetensors::tensor::{Dtype, TensorView};
        let mut bytes_store: Vec<(String, Vec<u8>, Vec<usize>, DTypeDescriptor)> = Vec::new();
        let mut data_map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();

        // First extract all bytes because TensorView needs references to the bytes.
        for (k, v) in state_dict.iter() {
            let bytes = <B as Backend>::to_bytes(&v.inner)
                .map_err(|e| anyhow::anyhow!("Backend to_bytes failed: {}", e))?;
            bytes_store.push((k.clone(), bytes, v.dims().as_ref().to_vec(), v.dtype()));
        }

        for (k, bytes, shape, dtype) in &bytes_store {
            let safe_dtype = match dtype.builtin_id() {
                Some(DTypeId::F32) => Dtype::F32,
                Some(DTypeId::F64) => Dtype::F64,
                Some(DTypeId::F16) => Dtype::F16,
                Some(DTypeId::BF16) => Dtype::BF16,
                Some(DTypeId::U32) => Dtype::U32,
                Some(DTypeId::I64) => Dtype::I64,
                Some(DTypeId::U8) => Dtype::U8,
                Some(DTypeId::Bool) => Dtype::BOOL,
                Some(DTypeId::Q8_0) | None => {
                    return Err(anyhow::anyhow!(
                        "safetensors does not support dtype {} for tensor '{k}'",
                        dtype.name()
                    ));
                }
            };
            let view = TensorView::new(safe_dtype, shape.clone(), bytes)
                .map_err(|e| anyhow::anyhow!("Failed to create TensorView: {}", e))?;
            data_map.insert(k.clone(), view);
        }

        safetensors::tensor::serialize_to_file(&data_map, &None, self.path)
            .map_err(|e| anyhow::anyhow!("Failed to write safetensors: {:?}", e))?;

        Ok(())
    }
}

#[cfg(feature = "std")]
/// `SafetensorsDeserializer`.
pub struct SafetensorsDeserializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> SafetensorsDeserializer<'a> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Deserializer for SafetensorsDeserializer<'a> {
    /// The error type returned if the forward pass fails.
    type Error = anyhow::Error;

    /// `deserialize`.
    fn deserialize<B: Backend + Execute<op::TensorFromBytes>>(
        &mut self,
        device: &DeviceId,
    ) -> core::result::Result<BTreeMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
        <B as Execute<op::TensorFromBytes>>::Output: Into<B::Storage<f32>>,
    {
        let buffer =
            std::fs::read(self.path).map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))?;
        let st = safetensors::SafeTensors::deserialize(&buffer)
            .map_err(|e| anyhow::anyhow!("Failed to parse safetensors: {:?}", e))?;

        let mut state_dict = BTreeMap::new();

        for (name, tensor_view) in st.tensors() {
            let dtype = match tensor_view.dtype() {
                safetensors::tensor::Dtype::F32 => DTypeId::F32,
                safetensors::tensor::Dtype::F64 => DTypeId::F64,
                safetensors::tensor::Dtype::F16 => DTypeId::F16,
                safetensors::tensor::Dtype::BF16 => DTypeId::BF16,
                safetensors::tensor::Dtype::U32 => DTypeId::U32,
                safetensors::tensor::Dtype::I64 => DTypeId::I64,
                safetensors::tensor::Dtype::U8 => DTypeId::U8,
                _ => return Err(anyhow::anyhow!("Unsupported dtype in safetensors")),
            };

            let raw_tensor = decode_bytes::<B>(
                tensor_view.data(),
                tensor_view.shape(),
                dtype.descriptor(),
                device,
            )
            .map_err(|e| anyhow::anyhow!("TensorFromBytes execution failed: {}", e))?;

            let dyn_shape = tensor_view.shape().to_vec();
            let _dtype: <f32 as crate::tensor::dtype::DType>::Field = Default::default();
            let _device = Default::default();
            let _grad = Default::default();

            let tensor: Tensor<Dyn, B> = Tensor::from_parts(
                raw_tensor,
                ShapeBuf::from_slice(&dyn_shape),
                _dtype,
                _device,
                _grad,
            )
            .map_err(|e| anyhow::anyhow!("Invalid tensor storage metadata: {}", e))?;
            state_dict.insert(name, tensor);
        }

        Ok(state_dict)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
/// `SerializedTensor`.
struct SerializedTensor {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

#[cfg(feature = "std")]
/// `PostcardSerializer`.
pub struct PostcardSerializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> PostcardSerializer<'a> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Serializer for PostcardSerializer<'a> {
    /// The error type returned if the forward pass fails.
    type Error = anyhow::Error;

    /// `serialize`.
    fn serialize<B: Backend>(
        &mut self,
        state_dict: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
    {
        let mut map: BTreeMap<String, SerializedTensor> = BTreeMap::new();
        for (k, v) in state_dict.iter() {
            let bytes = <B as Backend>::to_bytes(&v.inner)
                .map_err(|e| anyhow::anyhow!("Backend to_bytes failed: {}", e))?;
            let dtype_str = match v.builtin_dtype_id() {
                Some(DTypeId::F32) => "F32",
                Some(DTypeId::F64) => "F64",
                Some(DTypeId::F16) => "F16",
                Some(DTypeId::BF16) => "BF16",
                Some(DTypeId::U32) => "U32",
                Some(DTypeId::I64) => "I64",
                Some(DTypeId::U8) => "U8",
                Some(DTypeId::Bool) => "BOOL",
                Some(DTypeId::Q8_0) => "Q8_0",
                None => v.dtype().name(),
            }
            .to_string();
            map.insert(
                k.clone(),
                SerializedTensor {
                    shape: v.dims().as_ref().to_vec(),
                    dtype: dtype_str,
                    data: bytes,
                },
            );
        }

        let bytes = postcard::to_stdvec(&map)
            .map_err(|e| anyhow::anyhow!("Postcard serialization failed: {}", e))?;
        std::fs::write(self.path, bytes)?;
        Ok(())
    }
}

#[cfg(feature = "std")]
/// `PostcardDeserializer`.
pub struct PostcardDeserializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> PostcardDeserializer<'a> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Deserializer for PostcardDeserializer<'a> {
    /// The error type returned if the forward pass fails.
    type Error = anyhow::Error;

    /// `deserialize`.
    fn deserialize<B: Backend + Execute<op::TensorFromBytes>>(
        &mut self,
        device: &DeviceId,
    ) -> core::result::Result<BTreeMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
        <B as Execute<op::TensorFromBytes>>::Output: Into<B::Storage<f32>>,
    {
        let limits = crate::io::limits::ResourceLimits::model_load_defaults();
        let metadata = std::fs::metadata(self.path)?;
        if metadata.len() > limits.max_file_bytes {
            return Err(anyhow::anyhow!(
                "Postcard model file size {} exceeds maximum limit {}",
                metadata.len(),
                limits.max_file_bytes
            ));
        }

        let raw_bytes = std::fs::read(self.path)?;
        let map: BTreeMap<String, SerializedTensor> = postcard::from_bytes(&raw_bytes)
            .map_err(|e| anyhow::anyhow!("Postcard deserialization failed: {}", e))?;

        if map.len() > limits.max_tensor_count {
            return Err(anyhow::anyhow!(
                "Postcard model tensor count {} exceeds limit {}",
                map.len(),
                limits.max_tensor_count
            ));
        }

        let mut state_dict = BTreeMap::new();
        for (k, st) in map {
            limits
                .check_shape(&st.shape)
                .map_err(|e| anyhow::anyhow!("Invalid tensor shape: {}", e))?;
            let dtype = match st.dtype.as_str() {
                "F32" => DTypeId::F32,
                "F64" => DTypeId::F64,
                "F16" => DTypeId::F16,
                "BF16" => DTypeId::BF16,
                "U32" => DTypeId::U32,
                "I64" => DTypeId::I64,
                "U8" => DTypeId::U8,
                "Q8_0" => DTypeId::Q8_0,
                _ => return Err(anyhow::anyhow!("Unsupported dtype in postcard")),
            };
            let raw_tensor = decode_bytes::<B>(&st.data, &st.shape, dtype.descriptor(), device)
                .map_err(|e| anyhow::anyhow!("TensorFromBytes execution failed: {}", e))?;

            let dyn_shape = st.shape.clone();
            let _dtype = Default::default();
            let _device = Default::default();
            let _grad = Default::default();

            let tensor: Tensor<Dyn, B> = Tensor::from_parts(
                raw_tensor,
                ShapeBuf::from_slice(&dyn_shape),
                _dtype,
                _device,
                _grad,
            )
            .map_err(|e| anyhow::anyhow!("Invalid tensor storage metadata: {}", e))?;
            state_dict.insert(k, tensor);
        }

        Ok(state_dict)
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// `Format`.
pub enum Format {
    /// `Safetensors`.
    Safetensors,
    /// `ONNX`.
    ONNX,
}

#[cfg(feature = "std")]
/// `ModelExt`.
pub trait ModelExt<B: Backend> {
    /// `save`.
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default;

    /// `load`.
    fn load(&mut self, format: Format, path: &std::path::Path, device: &DeviceId) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
        B: Execute<op::TensorFromBytes>,
        <B as Execute<op::TensorFromBytes>>::Output: Into<B::Storage<f32>>;
}

#[cfg(feature = "std")]
impl<B: Backend, T: crate::nn::module::StateDict<B>> ModelExt<B> for T {
    /// `save`.
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
    {
        match format {
            Format::Safetensors => {
                let mut serializer = SafetensorsSerializer::new(path);
                self.save_to(&mut serializer)
                    .map_err(|e| anyhow::anyhow!(e))?
            }
            Format::ONNX => {
                let mut serializer = crate::onnx_exporter::OnnxExporter::new(path);
                self.save_to(&mut serializer)
                    .map_err(|e| anyhow::anyhow!(e))?
            }
        }
        Ok(())
    }

    /// `load`.
    fn load(&mut self, format: Format, path: &std::path::Path, device: &DeviceId) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
        B: Execute<op::TensorFromBytes>,
        <B as Execute<op::TensorFromBytes>>::Output: Into<B::Storage<f32>>,
    {
        match format {
            Format::Safetensors => {
                let mut deserializer = SafetensorsDeserializer::new(path);
                self.load_from(&mut deserializer, device)?;
            }
            Format::ONNX => {
                let mut deserializer = crate::onnx_exporter::OnnxImporter::new(path);
                self.load_from(&mut deserializer, device)?;
            }
        }
        Ok(())
    }
}
