use crate::prelude::*;
use std::collections::HashMap;

/// A trait for serializing a collection of dynamic tensors to a specific format.
pub trait Serializer {
    type Error: std::fmt::Debug + std::fmt::Display;

    /// Serializes the state dict to the given path or stream.
    fn serialize<B: Backend>(
        &mut self,
        state_dict: &HashMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;
}

/// A trait for deserializing a collection of dynamic tensors from a specific format.
pub trait Deserializer {
    type Error: std::fmt::Debug + std::fmt::Display;

    /// Deserializes the state dict from the given path or stream.
    fn deserialize<B: Backend>(
        &mut self,
        device: &KindleDevice,
    ) -> core::result::Result<HashMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;
}

pub struct SafetensorsSerializer<'a> {
    path: &'a std::path::Path,
}

impl<'a> SafetensorsSerializer<'a> {
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

impl<'a> Serializer for SafetensorsSerializer<'a> {
    type Error = anyhow::Error;

    fn serialize<B: Backend>(
        &mut self,
        state_dict: &HashMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        use safetensors::tensor::{Dtype, TensorView};
        let mut data_map = HashMap::new();
        let mut bytes_store = Vec::new();

        // First extract all bytes because TensorView needs references to the bytes.
        for (k, v) in state_dict.iter() {
            let bytes = <B as Backend>::to_bytes(&v.inner)
                .map_err(|e| anyhow::anyhow!("Backend to_bytes failed: {}", e))?;
            bytes_store.push((k.clone(), bytes, v.dims().clone(), v.dtype()));
        }

        for (k, bytes, shape, dtype) in &bytes_store {
            let safe_dtype = match dtype {
                KindleDType::F32 => Dtype::F32,
                KindleDType::F64 => Dtype::F64,
                KindleDType::F16 => Dtype::F16,
                KindleDType::BF16 => Dtype::BF16,
                KindleDType::U32 => Dtype::U32,
                KindleDType::I64 => Dtype::I64,
                KindleDType::U8 => Dtype::U8,
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

pub struct SafetensorsDeserializer<'a> {
    path: &'a std::path::Path,
}

impl<'a> SafetensorsDeserializer<'a> {
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

impl<'a> Deserializer for SafetensorsDeserializer<'a> {
    type Error = anyhow::Error;

    fn deserialize<B: Backend>(
        &mut self,
        device: &KindleDevice,
    ) -> core::result::Result<HashMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        let buffer =
            std::fs::read(self.path).map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))?;
        let st = safetensors::SafeTensors::deserialize(&buffer)
            .map_err(|e| anyhow::anyhow!("Failed to parse safetensors: {:?}", e))?;

        let mut state_dict = HashMap::new();

        for (name, tensor_view) in st.tensors() {
            let dtype = match tensor_view.dtype() {
                safetensors::tensor::Dtype::F32 => KindleDType::F32,
                safetensors::tensor::Dtype::F64 => KindleDType::F64,
                safetensors::tensor::Dtype::F16 => KindleDType::F16,
                safetensors::tensor::Dtype::BF16 => KindleDType::BF16,
                safetensors::tensor::Dtype::U32 => KindleDType::U32,
                safetensors::tensor::Dtype::I64 => KindleDType::I64,
                safetensors::tensor::Dtype::U8 => KindleDType::U8,
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
                Tensor::from_parts_unchecked(raw_tensor, dyn_shape, _dtype, _device, _grad);
            state_dict.insert(name, tensor);
        }

        Ok(state_dict)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializedTensor {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

pub struct BincodeSerializer<'a> {
    path: &'a std::path::Path,
}

impl<'a> BincodeSerializer<'a> {
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

impl<'a> Serializer for BincodeSerializer<'a> {
    type Error = anyhow::Error;

    fn serialize<B: Backend>(
        &mut self,
        state_dict: &HashMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        let mut map: HashMap<String, SerializedTensor> = HashMap::new();
        for (k, v) in state_dict.iter() {
            let bytes = <B as Backend>::to_bytes(&v.inner)
                .map_err(|e| anyhow::anyhow!("Backend to_bytes failed: {}", e))?;
            let dtype_str = match v.dtype() {
                KindleDType::F32 => "F32",
                KindleDType::F64 => "F64",
                KindleDType::F16 => "F16",
                KindleDType::BF16 => "BF16",
                KindleDType::U32 => "U32",
                KindleDType::I64 => "I64",
                KindleDType::U8 => "U8",
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

pub struct BincodeDeserializer<'a> {
    path: &'a std::path::Path,
}

impl<'a> BincodeDeserializer<'a> {
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

impl<'a> Deserializer for BincodeDeserializer<'a> {
    type Error = anyhow::Error;

    fn deserialize<B: Backend>(
        &mut self,
        device: &KindleDevice,
    ) -> core::result::Result<HashMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        let file = std::fs::File::open(self.path)?;
        let map: HashMap<String, SerializedTensor> = bincode::deserialize_from(file)?;

        let mut state_dict = HashMap::new();
        for (k, st) in map {
            let dtype = match st.dtype.as_str() {
                "F32" => KindleDType::F32,
                "F64" => KindleDType::F64,
                "F16" => KindleDType::F16,
                "BF16" => KindleDType::BF16,
                "U32" => KindleDType::U32,
                "I64" => KindleDType::I64,
                "U8" => KindleDType::U8,
                _ => return Err(anyhow::anyhow!("Unsupported dtype in bincode")),
            };
            let raw_tensor = <B as Backend>::from_bytes(&st.data, &st.shape, dtype, device)
                .map_err(|e| anyhow::anyhow!("Backend from_bytes failed: {}", e))?;

            let dyn_shape = st.shape.clone();
            let _dtype = Default::default();
            let _device = Default::default();
            let _grad = Default::default();

            let tensor: Tensor<Dyn, B> =
                Tensor::from_parts_unchecked(raw_tensor, dyn_shape, _dtype, _device, _grad);
            state_dict.insert(k, tensor);
        }

        Ok(state_dict)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    ONNX,
}

pub trait ModelExt<B: Backend> {
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;

    fn load(&mut self, format: Format, path: &std::path::Path, device: &KindleDevice) -> Result<()>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;
}

impl<B: Backend, T: crate::nn::module::StateDict<B>> ModelExt<B> for T {
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

    fn load(&mut self, format: Format, path: &std::path::Path, device: &KindleDevice) -> Result<()>
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
