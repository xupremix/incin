use crate::prelude::*;
use std::collections::HashMap;
use std::path::Path;

pub struct OnnxExporter<'a> {
    _path: &'a Path,
}

impl<'a> OnnxExporter<'a> {
    pub fn new(path: &'a Path) -> Self {
        Self { _path: path }
    }
}

impl<'a> crate::serialize::Serializer for OnnxExporter<'a> {
    type Error = anyhow::Error;

    fn serialize<B: Backend>(&mut self, _state_dict: &HashMap<String, Tensor<Dyn, B>>) -> core::result::Result<(), Self::Error> 
    where
        <<B as Backend>::DType as DType>::Field: Default,
        <<B as Backend>::Device as Device>::Field: Default 
    {
        // TODO: Implement tracing of computation graph to ONNX.
        // Currently, kindle runs eagerly. To export to ONNX, we must trace operations.
        // For now, this returns an unimplemented error.
        Err(anyhow::anyhow!("ONNX tracing is currently unsupported. Please use Format::Safetensors instead."))
    }
}

pub struct OnnxImporter<'a> {
    _path: &'a Path,
}

impl<'a> OnnxImporter<'a> {
    pub fn new(path: &'a Path) -> Self {
        Self { _path: path }
    }
}

impl<'a> crate::serialize::Deserializer for OnnxImporter<'a> {
    type Error = anyhow::Error;

    fn deserialize<B: Backend>(&mut self, _device: &KindleDevice) -> core::result::Result<HashMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::DType as DType>::Field: Default,
        <<B as Backend>::Device as Device>::Field: Default 
    {
        Err(anyhow::anyhow!("ONNX loading is currently unsupported. Please use Format::Safetensors instead."))
    }
}
