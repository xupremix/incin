use crate::prelude::*;
use alloc::collections::BTreeMap;

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
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;
}

/// A trait for deserializing a collection of dynamic tensors from a specific format.
pub trait Deserializer {
    /// The error type returned if the forward pass fails.
    type Error: core::fmt::Debug + core::fmt::Display;

    /// Deserializes the state dict from the given path or stream.
    fn deserialize<B: Backend>(
        &mut self,
        device: &DeviceId,
    ) -> core::result::Result<BTreeMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;
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
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        use safetensors::tensor::{Dtype, TensorView};
        let mut bytes_store: Vec<(String, Vec<u8>, Vec<usize>, DTypeId)> = Vec::new();
        let mut data_map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();

        // First extract all bytes because TensorView needs references to the bytes.
        for (k, v) in state_dict.iter() {
            let bytes = <B as Backend>::to_bytes(&v.inner)
                .map_err(|e| anyhow::anyhow!("Backend to_bytes failed: {}", e))?;
            bytes_store.push((k.clone(), bytes, v.dims().clone(), v.dtype()));
        }

        for (k, bytes, shape, dtype) in &bytes_store {
            let safe_dtype = match dtype {
                DTypeId::F32 => Dtype::F32,
                DTypeId::F64 => Dtype::F64,
                DTypeId::F16 => Dtype::F16,
                DTypeId::BF16 => Dtype::BF16,
                DTypeId::U32 => Dtype::U32,
                DTypeId::I64 => Dtype::I64,
                DTypeId::U8 => Dtype::U8,
                DTypeId::Q8_0 => {
                    return Err(anyhow::anyhow!(
                        "Safetensors does not support Q8_0 dtype natively"
                    ));
                }
            };
            let view = TensorView::new(safe_dtype, shape.clone(), bytes.as_slice())
                .map_err(|e| anyhow::anyhow!("Failed to create TensorView: {:?}", e))?;
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
    fn deserialize<B: Backend>(
        &mut self,
        device: &DeviceId,
    ) -> core::result::Result<BTreeMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
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

            let raw_tensor =
                <B as Backend>::from_bytes(tensor_view.data(), tensor_view.shape(), dtype, device)
                    .map_err(|e| anyhow::anyhow!("Backend from_bytes failed: {}", e))?;

            let dyn_shape = tensor_view.shape().to_vec();
            let _dtype = Default::default();
            let _device = Default::default();
            let _grad = Default::default();

            let tensor: Tensor<Dyn, B> =
                Tensor::from_parts(raw_tensor, dyn_shape, _dtype, _device, _grad)
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
/// `BincodeSerializer`.
pub struct BincodeSerializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> BincodeSerializer<'a> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Serializer for BincodeSerializer<'a> {
    /// The error type returned if the forward pass fails.
    type Error = anyhow::Error;

    /// `serialize`.
    fn serialize<B: Backend>(
        &mut self,
        state_dict: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        let mut map: BTreeMap<String, SerializedTensor> = BTreeMap::new();
        for (k, v) in state_dict.iter() {
            let bytes = <B as Backend>::to_bytes(&v.inner)
                .map_err(|e| anyhow::anyhow!("Backend to_bytes failed: {}", e))?;
            let dtype_str = match v.dtype() {
                DTypeId::F32 => "F32",
                DTypeId::F64 => "F64",
                DTypeId::F16 => "F16",
                DTypeId::BF16 => "BF16",
                DTypeId::U32 => "U32",
                DTypeId::I64 => "I64",
                DTypeId::U8 => "U8",
                DTypeId::Q8_0 => "Q8_0",
            }
            .to_string();
            map.insert(
                k.clone(),
                SerializedTensor {
                    shape: v.dims().clone(),
                    dtype: dtype_str,
                    data: bytes,
                },
            );
        }

        let file = std::fs::File::create(self.path)?;
        bincode::serialize_into(file, &map)?;
        Ok(())
    }
}

#[cfg(feature = "std")]
/// `BincodeDeserializer`.
pub struct BincodeDeserializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> BincodeDeserializer<'a> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Deserializer for BincodeDeserializer<'a> {
    /// The error type returned if the forward pass fails.
    type Error = anyhow::Error;

    /// `deserialize`.
    fn deserialize<B: Backend>(
        &mut self,
        device: &DeviceId,
    ) -> core::result::Result<BTreeMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        let file = std::fs::File::open(self.path)?;
        let map: BTreeMap<String, SerializedTensor> = bincode::deserialize_from(file)?;

        let mut state_dict = BTreeMap::new();
        for (k, st) in map {
            let dtype = match st.dtype.as_str() {
                "F32" => DTypeId::F32,
                "F64" => DTypeId::F64,
                "F16" => DTypeId::F16,
                "BF16" => DTypeId::BF16,
                "U32" => DTypeId::U32,
                "I64" => DTypeId::I64,
                "U8" => DTypeId::U8,
                "Q8_0" => DTypeId::Q8_0,
                _ => return Err(anyhow::anyhow!("Unsupported dtype in bincode")),
            };
            let raw_tensor = <B as Backend>::from_bytes(&st.data, &st.shape, dtype, device)
                .map_err(|e| anyhow::anyhow!("Backend from_bytes failed: {}", e))?;

            let dyn_shape = st.shape.clone();
            let _dtype = Default::default();
            let _device = Default::default();
            let _grad = Default::default();

            let tensor: Tensor<Dyn, B> =
                Tensor::from_parts(raw_tensor, dyn_shape, _dtype, _device, _grad)
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
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;

    /// `load`.
    fn load(&mut self, format: Format, path: &std::path::Path, device: &DeviceId) -> Result<()>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;
}

#[cfg(feature = "std")]
impl<B: Backend, T: crate::nn::module::StateDict<B>> ModelExt<B> for T {
    /// `save`.
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
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
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
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
