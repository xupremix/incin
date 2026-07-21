#![allow(clippy::field_reassign_with_default)]

use crate::graph::{AttributeValue, Graph};
use crate::onnx_pb::onnx;
use crate::prelude::*;
use alloc::collections::BTreeMap;
use prost::Message;
use std::path::Path;

/// `OnnxExporter`.
pub struct OnnxExporter<'a> {
    _path: &'a Path,
}

impl<'a> OnnxExporter<'a> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(path: &'a Path) -> Self {
        Self { _path: path }
    }
}

/// `export_to_onnx`.
pub fn export_to_onnx(graph: &Graph, path: &Path) -> anyhow::Result<()> {
    let mut onnx_graph = onnx::GraphProto::default();
    onnx_graph.name = Some(alloc::string::String::from("kindle_graph"));

    // Add inputs
    for &in_id in &graph.inputs {
        let val = &graph.values[&in_id];
        onnx_graph.input.push(value_to_value_info(val));
    }

    // Add outputs
    for &out_id in &graph.outputs {
        let val = &graph.values[&out_id];
        onnx_graph.output.push(value_to_value_info(val));
    }

    // Initializers
    for (id, bytes) in &graph.initializers {
        let val = &graph.values[id];
        let mut tensor = onnx::TensorProto::default();
        tensor.name = Some(val.id.to_string());
        tensor.dims = val.shape.iter().map(|&x| x as i64).collect();
        tensor.data_type = Some(dtype_to_onnx(val.dtype) as i32);
        tensor.raw_data = Some(bytes.clone());
        onnx_graph.initializer.push(tensor);
    }

    // Nodes
    for node in &graph.nodes {
        let mut n = onnx::NodeProto::default();
        n.op_type = Some(node.op.as_str().to_string());
        n.input = node.inputs.iter().map(|x| x.to_string()).collect();
        n.output = node.outputs.iter().map(|x| x.to_string()).collect();

        for (k, v) in &node.attributes {
            let mut attr = onnx::AttributeProto::default();
            attr.name = Some(k.clone());
            match v {
                AttributeValue::Int(i) => {
                    attr.r#type = Some(onnx::attribute_proto::AttributeType::Int as i32);
                    attr.i = Some(*i);
                }
                AttributeValue::Float(f) => {
                    attr.r#type = Some(onnx::attribute_proto::AttributeType::Float as i32);
                    attr.f = Some(*f);
                }
                AttributeValue::String(s) => {
                    attr.r#type = Some(onnx::attribute_proto::AttributeType::String as i32);
                    attr.s = Some(s.as_bytes().to_vec());
                }
                AttributeValue::Ints(ints) => {
                    attr.r#type = Some(onnx::attribute_proto::AttributeType::Ints as i32);
                    attr.ints = ints.clone();
                }
                AttributeValue::Floats(floats) => {
                    attr.r#type = Some(onnx::attribute_proto::AttributeType::Floats as i32);
                    attr.floats = floats.clone();
                }
            }
            n.attribute.push(attr);
        }
        onnx_graph.node.push(n);
    }

    let mut model = onnx::ModelProto::default();
    model.ir_version = Some(8);
    let mut opset = onnx::OperatorSetIdProto::default();
    opset.version = Some(14);
    model.opset_import.push(opset);
    model.producer_name = Some(alloc::string::String::from("kindle"));
    model.graph = Some(onnx_graph);

    let mut buf = Vec::new();
    model.encode(&mut buf)?;
    std::fs::write(path, buf)?;

    Ok(())
}

/// `dtype_to_onnx`.
fn dtype_to_onnx(dt: KindleDType) -> onnx::tensor_proto::DataType {
    match dt {
        KindleDType::F32 => onnx::tensor_proto::DataType::Float,
        KindleDType::F64 => onnx::tensor_proto::DataType::Double,
        KindleDType::F16 => onnx::tensor_proto::DataType::Float16,
        KindleDType::BF16 => onnx::tensor_proto::DataType::Bfloat16,
        KindleDType::U32 => onnx::tensor_proto::DataType::Uint32,
        KindleDType::I64 => onnx::tensor_proto::DataType::Int64,
        KindleDType::U8 => onnx::tensor_proto::DataType::Uint8,
        KindleDType::Q8_0 => onnx::tensor_proto::DataType::Undefined,
    }
}

/// `value_to_value_info`.
fn value_to_value_info(val: &crate::graph::Value) -> onnx::ValueInfoProto {
    let mut vi = onnx::ValueInfoProto::default();
    vi.name = Some(val.id.to_string());

    let mut tensor_type = onnx::type_proto::Tensor::default();
    tensor_type.elem_type = Some(dtype_to_onnx(val.dtype) as i32);

    let mut shape = onnx::TensorShapeProto::default();
    for &d in &val.shape {
        let mut dim = onnx::tensor_shape_proto::Dimension::default();
        dim.value = Some(onnx::tensor_shape_proto::dimension::Value::DimValue(
            d as i64,
        ));
        shape.dim.push(dim);
    }
    tensor_type.shape = Some(shape);

    let mut t = onnx::TypeProto::default();
    t.value = Some(onnx::type_proto::Value::TensorType(tensor_type));
    vi.r#type = Some(t);
    vi
}

impl<'a> crate::serialize::Serializer for OnnxExporter<'a> {
    /// The error type returned if the forward pass fails.
    type Error = anyhow::Error;

    /// `serialize`.
    fn serialize<B: Backend>(
        &mut self,
        _state_dict: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> core::result::Result<(), Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
    {
        // Try to run export_to_onnx with thread local graph
        let g = crate::tensor::tracing::TRACING_GRAPH.lock();
        export_to_onnx(&g, self._path)
    }
}

/// `OnnxImporter`.
pub struct OnnxImporter<'a> {
    _path: &'a Path,
}

impl<'a> OnnxImporter<'a> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(path: &'a Path) -> Self {
        Self { _path: path }
    }
}

impl<'a> crate::serialize::Deserializer for OnnxImporter<'a> {
    /// The error type returned if the forward pass fails.
    type Error = anyhow::Error;

    /// `deserialize`.
    fn deserialize<B: Backend>(
        &mut self,
        _device: &KindleDevice,
    ) -> core::result::Result<BTreeMap<String, Tensor<Dyn, B>>, Self::Error>
    where
        <<B as Backend>::Device as Device>::Field: Default,
    {
        Err(anyhow::anyhow!(
            "ONNX loading is currently unsupported. Please use Format::Safetensors instead."
        ))
    }
}
