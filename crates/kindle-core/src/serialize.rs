use crate::prelude::*;
use alloc::collections::BTreeMap;

/// A trait for serializing a collection of dynamic tensors to a specific format.
pub trait Serializer {
    /// Core abstraction for `Error` within the Kindle framework..
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
    /// Core abstraction for `Error` within the Kindle framework..
    type Error: core::fmt::Debug + core::fmt::Display;

    /// Deserializes the state dict from the given path or stream.
    fn deserialize<B: Backend>(
        &mut self,
        device: &KindleDevice,
    ) -> core::result::Result<BTreeMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;
}

#[cfg(feature = "std")]
/// Core abstraction for `SafetensorsSerializer` within the Kindle framework..
pub struct SafetensorsSerializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> SafetensorsSerializer<'a> {
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Serializer for SafetensorsSerializer<'a> {
    /// Core abstraction for `Error` within the Kindle framework..
    type Error = anyhow::Error;

    /// Core abstraction for `serialize` within the Kindle framework..
    fn serialize<B: Backend>(
        &mut self,
        state_dict: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default,
    {
        use safetensors::tensor::{Dtype, TensorView};
        let mut bytes_store: Vec<(String, Vec<u8>, Vec<usize>, KindleDType)> = Vec::new();
        let mut data_map: BTreeMap<String, TensorView<'_>> = BTreeMap::new();

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
                KindleDType::Q8_0 => panic!("Safetensors does not support Q8_0 dtype natively"),
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
/// Core abstraction for `SafetensorsDeserializer` within the Kindle framework..
pub struct SafetensorsDeserializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> SafetensorsDeserializer<'a> {
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Deserializer for SafetensorsDeserializer<'a> {
    /// Core abstraction for `Error` within the Kindle framework..
    type Error = anyhow::Error;

    /// Core abstraction for `deserialize` within the Kindle framework..
    fn deserialize<B: Backend>(
        &mut self,
        device: &KindleDevice,
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
#[allow(dead_code)]
/// Core abstraction for `SerializedTensor` within the Kindle framework..
struct SerializedTensor {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

#[cfg(feature = "std")]
/// Core abstraction for `BincodeSerializer` within the Kindle framework..
pub struct BincodeSerializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> BincodeSerializer<'a> {
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Serializer for BincodeSerializer<'a> {
    /// Core abstraction for `Error` within the Kindle framework..
    type Error = anyhow::Error;

    /// Core abstraction for `serialize` within the Kindle framework..
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
                KindleDType::F32 => "F32",
                KindleDType::F64 => "F64",
                KindleDType::F16 => "F16",
                KindleDType::BF16 => "BF16",
                KindleDType::U32 => "U32",
                KindleDType::I64 => "I64",
                KindleDType::U8 => "U8",
                KindleDType::Q8_0 => "Q8_0",
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
/// Core abstraction for `BincodeDeserializer` within the Kindle framework..
pub struct BincodeDeserializer<'a> {
    path: &'a std::path::Path,
}

#[cfg(feature = "std")]
impl<'a> BincodeDeserializer<'a> {
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new(path: &'a std::path::Path) -> Self {
        Self { path }
    }
}

#[cfg(feature = "std")]
impl<'a> Deserializer for BincodeDeserializer<'a> {
    /// Core abstraction for `Error` within the Kindle framework..
    type Error = anyhow::Error;

    /// Core abstraction for `deserialize` within the Kindle framework..
    fn deserialize<B: Backend>(
        &mut self,
        device: &KindleDevice,
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
                "F32" => KindleDType::F32,
                "F64" => KindleDType::F64,
                "F16" => KindleDType::F16,
                "BF16" => KindleDType::BF16,
                "U32" => KindleDType::U32,
                "I64" => KindleDType::I64,
                "U8" => KindleDType::U8,
                "Q8_0" => KindleDType::Q8_0,
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

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Core abstraction for `Format` within the Kindle framework..
pub enum Format {
    /// Core abstraction for `Safetensors` within the Kindle framework..
    Safetensors,
    /// Core abstraction for `ONNX` within the Kindle framework..
    ONNX,
}

#[cfg(feature = "std")]
/// Core abstraction for `ModelExt` within the Kindle framework..
pub trait ModelExt<B: Backend> {
    /// Core abstraction for `save` within the Kindle framework..
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;

    /// Core abstraction for `load` within the Kindle framework..
    fn load(&mut self, format: Format, path: &std::path::Path, device: &KindleDevice) -> Result<()>
    where
        <<B as Backend>::Device as Device>::Field: Default,
        <<B as Backend>::FloatElem as crate::tensor::dtype::DType>::Field: Default;
}

#[cfg(feature = "std")]
impl<B: Backend, T: crate::nn::module::StateDict<B>> ModelExt<B> for T {
    /// Core abstraction for `save` within the Kindle framework..
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

    /// Core abstraction for `load` within the Kindle framework..
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
