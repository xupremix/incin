#![allow(clippy::field_reassign_with_default)]

use crate::err::Error;
use crate::graph::{AttributeValue, Graph};
use crate::onnx_pb::onnx;
use crate::tensor::dtype::DTypeId;
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
    onnx_graph.name = Some(alloc::string::String::from("incin_graph"));

    // Add inputs
    for &in_id in &graph.inputs {
        let val = &graph.values[&in_id];
        onnx_graph.input.push(value_to_value_info(val)?);
    }

    // Add outputs
    for &out_id in &graph.outputs {
        let val = &graph.values[&out_id];
        onnx_graph.output.push(value_to_value_info(val)?);
    }

    // Initializers
    for (id, bytes) in &graph.initializers {
        let val = &graph.values[id];
        let mut tensor = onnx::TensorProto::default();
        tensor.name = Some(val.id.to_string());
        tensor.dims = val.shape.iter().map(|&x| x as i64).collect();
        tensor.data_type = Some(dtype_to_onnx(
            val.dtype
                .builtin_id()
                .ok_or_else(|| Error::Msg("ONNX export requires a built-in dtype".into()))?,
        )? as i32);
        tensor.raw_data = Some(bytes.clone());
        onnx_graph.initializer.push(tensor);
    }

    // Nodes
    for node in &graph.nodes {
        let mut n = onnx::NodeProto::default();
        let onnx_name = node.operation.onnx_name().ok_or_else(|| {
            anyhow::anyhow!(
                "operation {} has no ONNX mapping",
                node.operation.display_name()
            )
        })?;
        n.op_type = Some(onnx_name.to_string());
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
    model.producer_name = Some(alloc::string::String::from("incin"));
    model.graph = Some(onnx_graph);

    let mut buf = Vec::new();
    model.encode(&mut buf)?;
    std::fs::write(path, buf)?;

    Ok(())
}

/// `dtype_to_onnx`.
fn dtype_to_onnx(dt: DTypeId) -> Result<onnx::tensor_proto::DataType, Error> {
    match dt {
        DTypeId::F32 => Ok(onnx::tensor_proto::DataType::Float),
        DTypeId::F64 => Ok(onnx::tensor_proto::DataType::Double),
        DTypeId::F16 => Ok(onnx::tensor_proto::DataType::Float16),
        DTypeId::BF16 => Ok(onnx::tensor_proto::DataType::Bfloat16),
        DTypeId::U32 => Ok(onnx::tensor_proto::DataType::Uint32),
        DTypeId::I64 => Ok(onnx::tensor_proto::DataType::Int64),
        DTypeId::U8 => Ok(onnx::tensor_proto::DataType::Uint8),
        DTypeId::Bool => Ok(onnx::tensor_proto::DataType::Bool),
        DTypeId::Q8_0 => Err(Error::UnsupportedDType {
            dtype: dt.descriptor(),
            backend: "ONNX",
            op: "export_onnx",
        }),
    }
}

/// `value_to_value_info`.
fn value_to_value_info(val: &crate::graph::Value) -> anyhow::Result<onnx::ValueInfoProto> {
    let mut vi = onnx::ValueInfoProto::default();
    vi.name = Some(val.id.to_string());

    let mut tensor_type = onnx::type_proto::Tensor::default();
    tensor_type.elem_type = Some(dtype_to_onnx(
        val.dtype
            .builtin_id()
            .ok_or_else(|| Error::Msg("ONNX export requires a built-in dtype".into()))?,
    )? as i32);

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
    Ok(vi)
}

/// Reconstructs a [`Graph`] from an ONNX file on disk.
pub struct OnnxImporter<'a> {
    path: &'a Path,
}

impl<'a> OnnxImporter<'a> {
    /// Creates a new instance bound to a `.onnx` file path.
    pub fn new(path: &'a Path) -> Self {
        Self { path }
    }

    /// Parses the bound file into a [`Graph`]. See `import_from_onnx`'s own
    /// documentation (in this crate's source) for what this does and does
    /// not reconstruct - it is crate-private and cannot be linked from here.
    pub fn import(&self) -> anyhow::Result<Graph> {
        import_from_onnx(self.path)
    }
}

/// Reconstructs a [`Graph`] from an ONNX file, reversing the three sections
/// [`export_to_onnx`] writes: inputs, initializers, and nodes.
///
/// This is a structural round trip, not a re-derivation of typed
/// descriptors: a node's attributes are copied through as the generic
/// key-value pairs ONNX carries, the same shape `Node.attributes` has on
/// the graph `export_to_onnx` itself walks. It does not attempt full ONNX
/// shape inference. A node whose op has no mapping back to an
/// [`OperationKind`] (see `onnx_name`'s own forward table), or whose output
/// has no shape and dtype from either an explicit `value_info` entry or an
/// initializer, is refused rather than given an invented shape or a guessed
/// dtype - an invented `Value::shape` would be a lie every downstream reader
/// of the graph inherits.
pub fn import_from_onnx(path: &Path) -> anyhow::Result<Graph> {
    let bytes = std::fs::read(path)?;
    let model = onnx::ModelProto::decode(bytes.as_slice())?;
    let onnx_graph = model
        .graph
        .ok_or_else(|| anyhow::anyhow!("ONNX model has no graph"))?;

    let mut graph = Graph::new();
    let mut ids: std::collections::BTreeMap<String, crate::graph::ValueId> =
        std::collections::BTreeMap::new();
    let mut shapes: std::collections::BTreeMap<String, (Vec<usize>, DTypeId)> =
        std::collections::BTreeMap::new();

    // `value_info` carries shape/dtype for intermediate tensors when the
    // file was shape-inferred; inputs and outputs carry their own
    // regardless, so all three feed the same lookup.
    for value_info in onnx_graph
        .value_info
        .iter()
        .chain(onnx_graph.input.iter())
        .chain(onnx_graph.output.iter())
    {
        let Some(name) = &value_info.name else {
            continue;
        };
        if let Some(shape_dtype) = value_info_shape_dtype(value_info)? {
            shapes.insert(name.clone(), shape_dtype);
        }
    }

    for tensor in &onnx_graph.initializer {
        let name = tensor
            .name
            .clone()
            .ok_or_else(|| anyhow::anyhow!("initializer has no name"))?;
        let dtype = dtype_from_onnx(
            tensor
                .data_type
                .ok_or_else(|| anyhow::anyhow!("initializer {name} has no data_type"))?,
        )?;
        let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
        let raw = tensor.raw_data.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "initializer {name} has no raw_data; only the raw-bytes tensor \
                 encoding `export_to_onnx` itself writes is supported"
            )
        })?;
        let id = graph.add_value(shape, dtype, Some(name.clone()));
        graph.initializers.insert(id, raw);
        ids.insert(name, id);
    }

    for value_info in &onnx_graph.input {
        let name = value_info
            .name
            .clone()
            .ok_or_else(|| anyhow::anyhow!("graph input has no name"))?;
        if ids.contains_key(&name) {
            // Already bound as an initializer: ONNX permits (and some
            // exporters do) listing a weight in both `input` and
            // `initializer`.
            continue;
        }
        let (shape, dtype) = value_info_shape_dtype(value_info)?
            .ok_or_else(|| anyhow::anyhow!("graph input {name} has no shape/dtype"))?;
        let id = graph.add_value(shape, dtype, Some(name.clone()));
        ids.insert(name, id);
        graph.mark_input(id);
    }

    for node in &onnx_graph.node {
        let op_type = node
            .op_type
            .clone()
            .ok_or_else(|| anyhow::anyhow!("node has no op_type"))?;
        let input_ids = node
            .input
            .iter()
            .map(|name| {
                ids.get(name).copied().ok_or_else(|| {
                    anyhow::anyhow!("node {op_type} references unknown input {name}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let first_input_rank = node
            .input
            .first()
            .and_then(|name| shapes.get(name))
            .map(|(shape, _)| shape.len());
        let operation = operation_from_onnx(&op_type, first_input_rank, &node.attribute)
            .ok_or_else(|| anyhow::anyhow!("ONNX op {op_type} has no incin mapping"))?;
        let mut output_ids = Vec::with_capacity(node.output.len());
        for name in &node.output {
            let (shape, dtype) = shapes.get(name).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "node output {name} (op {op_type}) has no shape/dtype; only ONNX \
                     files carrying `value_info` for every intermediate tensor are \
                     supported, since this pass does not run shape inference"
                )
            })?;
            let id = graph.add_value(shape, dtype, Some(name.clone()));
            ids.insert(name.clone(), id);
            output_ids.push(id);
        }
        let attributes = translate_attributes(&node.attribute)?;
        graph.add_node(operation, input_ids, output_ids, attributes);
    }

    for value_info in &onnx_graph.output {
        let name = value_info
            .name
            .clone()
            .ok_or_else(|| anyhow::anyhow!("graph output has no name"))?;
        let id = *ids
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("graph output {name} was never produced"))?;
        graph.mark_output(id);
    }

    Ok(graph)
}

/// The inverse of `dtype_to_onnx`.
fn dtype_from_onnx(code: i32) -> anyhow::Result<DTypeId> {
    use onnx::tensor_proto::DataType;
    Ok(match code {
        x if x == DataType::Float as i32 => DTypeId::F32,
        x if x == DataType::Double as i32 => DTypeId::F64,
        x if x == DataType::Float16 as i32 => DTypeId::F16,
        x if x == DataType::Bfloat16 as i32 => DTypeId::BF16,
        x if x == DataType::Uint32 as i32 => DTypeId::U32,
        x if x == DataType::Int64 as i32 => DTypeId::I64,
        x if x == DataType::Uint8 as i32 => DTypeId::U8,
        x if x == DataType::Bool as i32 => DTypeId::Bool,
        other => return Err(anyhow::anyhow!("unsupported ONNX dtype code {other}")),
    })
}

/// The inverse of `value_to_value_info`, for the parts of a `TypeProto` this
/// module ever writes: a tensor type with a concrete (non-symbolic) shape.
/// Returns `None` rather than an error for anything else - a `value_info`
/// entry with no tensor type, or a dynamic dimension, is not itself a
/// defect, it just means this name contributes nothing to the shape lookup.
fn value_info_shape_dtype(
    value_info: &onnx::ValueInfoProto,
) -> anyhow::Result<Option<(Vec<usize>, DTypeId)>> {
    let Some(type_proto) = &value_info.r#type else {
        return Ok(None);
    };
    let Some(onnx::type_proto::Value::TensorType(tensor)) = &type_proto.value else {
        return Ok(None);
    };
    let Some(elem_type) = tensor.elem_type else {
        return Ok(None);
    };
    let dtype = dtype_from_onnx(elem_type)?;
    let Some(shape_proto) = &tensor.shape else {
        return Ok(None);
    };
    let mut shape = Vec::with_capacity(shape_proto.dim.len());
    for dim in &shape_proto.dim {
        let Some(onnx::tensor_shape_proto::dimension::Value::DimValue(value)) = &dim.value else {
            // A symbolic (`dim_param`) or unset dimension: no concrete
            // shape is available, and this pass does not guess one.
            return Ok(None);
        };
        shape.push(
            usize::try_from(*value)
                .map_err(|_| anyhow::anyhow!("negative ONNX dimension {value}"))?,
        );
    }
    Ok(Some((shape, dtype)))
}

/// The inverse of `onnx_name`. `MatMul` and `Conv` are each written by two
/// different [`OperationKind`]s (`MatMulExact`/`BatchedMatMul`,
/// `Conv1dExact`/`Conv2dExact`); the rank of the first operand - read from
/// `value_info`/an initializer when one supplied it - is what tells them
/// apart on the way back in, the same distinction ONNX's own operator
/// semantics draw from tensor rank rather than the op name. `Trilu` folds
/// `Triu`/`Tril` into one ONNX op with an `upper` attribute, defaulting to
/// `1` (upper) per the ONNX spec when the attribute is absent.
///
/// ONNX's `"Unsqueeze"` is mapped to `UnsqueezeExact` here, not the legacy
/// `Unsqueeze` kind `onnx_name`'s forward table names for it: `UnsqueezeExact`
/// is the identity every backend's executor in this codebase actually
/// implements, so a node imported this way is one dispatch can run, at the
/// cost of `export_to_onnx(import_from_onnx(path))` not being a strict
/// identity for a graph built from the legacy kind.
fn operation_from_onnx(
    op_type: &str,
    first_input_rank: Option<usize>,
    attributes: &[onnx::AttributeProto],
) -> Option<crate::shapes::error::OperationKind> {
    use crate::shapes::error::OperationKind as K;
    Some(match op_type {
        "Add" => K::Add,
        "Sub" => K::Sub,
        "Mul" => K::Mul,
        "Div" => K::Div,
        "MatMul" => match first_input_rank {
            Some(2) => K::MatMulExact,
            _ => K::BatchedMatMul,
        },
        "Relu" => K::Relu,
        "Exp" => K::Exp,
        "Neg" => K::Neg,
        "Sqrt" => K::Sqrt,
        "Log" => K::Log,
        "Tanh" => K::Tanh,
        "Sigmoid" => K::Sigmoid,
        "Softmax" => K::Softmax,
        "Reshape" => K::ReshapeExact,
        "Transpose" => K::TransposeExact,
        "Concat" => K::ConcatExact,
        "Conv" => match first_input_rank {
            Some(3) => K::Conv1dExact,
            _ => K::Conv2dExact,
        },
        "ConvTranspose" => K::ConvTranspose2d,
        "MaxPool" => K::MaxPool2d,
        "AveragePool" => K::AvgPool2d,
        "GlobalAveragePool" => K::AdaptiveAvgPool2dExact,
        "Equal" => K::CmpEq,
        "Less" => K::CmpLt,
        "LessOrEqual" => K::CmpLe,
        "Greater" => K::CmpGt,
        "GreaterOrEqual" => K::CmpGe,
        "And" => K::LogicalAnd,
        "Or" => K::LogicalOr,
        "Not" => K::LogicalNot,
        "Max" => K::Maximum,
        "Min" => K::Minimum,
        "Where" => K::WhereCond,
        "GatherElements" => K::Gather,
        "Gather" => K::IndexSelect,
        "ScatterElements" => K::Scatter,
        "Unsqueeze" => K::UnsqueezeExact,
        "Tile" => K::Repeat,
        "Pad" => K::Pad,
        "Trilu" => {
            let upper = attributes
                .iter()
                .find(|attribute| attribute.name.as_deref() == Some("upper"))
                .and_then(|attribute| attribute.i)
                .unwrap_or(1);
            if upper != 0 { K::Triu } else { K::Tril }
        }
        "DepthToSpace" => K::PixelShuffle,
        "GroupNormalization" => K::GroupNorm,
        "InstanceNormalization" => K::InstanceNorm,
        _ => return None,
    })
}

/// Copies each `AttributeProto` through as the `AttributeValue` its declared
/// `r#type` names. Refuses rather than guesses when the type field is unset
/// (legacy ONNX files relied on a has-field heuristic instead; this reader
/// does not implement that fallback) or names a kind `AttributeValue` has no
/// case for (`Tensor`, `Graph`, and the rest of the subgraph/sparse variants
/// `export_to_onnx` never writes either).
fn translate_attributes(
    attributes: &[onnx::AttributeProto],
) -> anyhow::Result<std::collections::BTreeMap<String, AttributeValue>> {
    use onnx::attribute_proto::AttributeType;
    let mut out = std::collections::BTreeMap::new();
    for attribute in attributes {
        let name = attribute
            .name
            .clone()
            .ok_or_else(|| anyhow::anyhow!("attribute has no name"))?;
        let kind = attribute
            .r#type
            .ok_or_else(|| anyhow::anyhow!("attribute {name} has no declared type"))?;
        let value = if kind == AttributeType::Int as i32 {
            AttributeValue::Int(attribute.i.ok_or_else(|| {
                anyhow::anyhow!("attribute {name} declares INT but carries no value")
            })?)
        } else if kind == AttributeType::Float as i32 {
            AttributeValue::Float(attribute.f.ok_or_else(|| {
                anyhow::anyhow!("attribute {name} declares FLOAT but carries no value")
            })?)
        } else if kind == AttributeType::String as i32 {
            let raw = attribute.s.clone().ok_or_else(|| {
                anyhow::anyhow!("attribute {name} declares STRING but carries no value")
            })?;
            AttributeValue::String(
                String::from_utf8(raw)
                    .map_err(|_| anyhow::anyhow!("attribute {name} is not valid UTF-8"))?,
            )
        } else if kind == AttributeType::Ints as i32 {
            AttributeValue::Ints(attribute.ints.clone())
        } else if kind == AttributeType::Floats as i32 {
            AttributeValue::Floats(attribute.floats.clone())
        } else {
            return Err(anyhow::anyhow!(
                "attribute {name} has ONNX type {kind}, which AttributeValue has no case for"
            ));
        };
        out.insert(name, value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::OperationIdentity;
    use crate::exec::catalog::{Descriptor, LogicalTensorMeta, NoAttributes, TraceDescriptor, op};

    #[test]
    fn export_uses_canonical_identity_and_projected_attributes() {
        let descriptor = Descriptor::<op::Add>::infer_runtime(
            NoAttributes,
            vec![
                LogicalTensorMeta {
                    shape: Some(crate::shapes::ShapeBuf::from_slice(&[2, 3])),
                    dtype: Some(crate::tensor::dtype::DTypeId::F32.into()),
                    device: Some(crate::tensor::device::DeviceId::cpu()),
                },
                LogicalTensorMeta {
                    shape: Some(crate::shapes::ShapeBuf::from_slice(&[2, 3])),
                    dtype: Some(crate::tensor::dtype::DTypeId::F32.into()),
                    device: Some(crate::tensor::device::DeviceId::cpu()),
                },
            ],
        )
        .expect("descriptor should validate");
        let mut graph = Graph::new();
        let input = graph.add_value(vec![2, 3], crate::tensor::dtype::DTypeId::F32, None);
        let rhs = graph.add_value(vec![2, 3], crate::tensor::dtype::DTypeId::F32, None);
        let output = graph.add_value(vec![2, 3], crate::tensor::dtype::DTypeId::F32, None);
        graph.add_node_with_descriptor_payload(
            OperationIdentity::Builtin(crate::shapes::error::OperationKind::Add),
            vec![input, rhs],
            vec![output],
            descriptor
                .descriptor()
                .trace_attributes()
                .expect("attributes should project"),
            descriptor
                .descriptor()
                .trace_descriptor_payload()
                .expect("payload should serialize"),
        );
        graph.inputs.push(input);
        graph.outputs.push(output);

        let path =
            std::env::temp_dir().join(format!("incin-onnx-{}-{}.onnx", std::process::id(), output));
        export_to_onnx(&graph, &path).expect("canonical graph should export");
        let bytes = std::fs::read(&path).expect("export should be readable");
        let model = onnx::ModelProto::decode(bytes.as_slice()).expect("export should decode");
        let node = &model.graph.expect("graph should exist").node[0];
        assert_eq!(node.op_type.as_deref(), Some("Add"));
        assert!(node.attribute.is_empty());
        std::fs::remove_file(path).expect("test export should be removable");
    }

    #[test]
    fn q8_0_export_is_rejected() {
        assert!(dtype_to_onnx(DTypeId::Q8_0).is_err());
    }

    #[test]
    fn a_graph_survives_export_then_import() {
        let mut graph = Graph::new();
        let lhs = graph.add_value(vec![2, 3], crate::tensor::dtype::DTypeId::F32, None);
        let rhs = graph.add_value(vec![2, 3], crate::tensor::dtype::DTypeId::F32, None);
        let output = graph.add_value(vec![2, 3], crate::tensor::dtype::DTypeId::F32, None);
        graph.add_node(
            crate::shapes::error::OperationKind::Add,
            vec![lhs, rhs],
            vec![output],
            Default::default(),
        );
        graph.inputs.push(lhs);
        graph.inputs.push(rhs);
        graph.outputs.push(output);

        let path = std::env::temp_dir().join(format!(
            "incin-onnx-roundtrip-{}-{}.onnx",
            std::process::id(),
            output
        ));
        export_to_onnx(&graph, &path).expect("graph should export");
        let imported = import_from_onnx(&path).expect("exported file should import back");
        std::fs::remove_file(&path).expect("test export should be removable");

        assert_eq!(imported.inputs.len(), 2);
        assert_eq!(imported.outputs.len(), 1);
        assert_eq!(imported.nodes.len(), 1);
        let node = &imported.nodes[0];
        assert_eq!(
            node.operation,
            crate::exec::OperationIdentity::Builtin(crate::shapes::error::OperationKind::Add)
        );
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs.len(), 1);
        for &value_id in node.inputs.iter().chain(node.outputs.iter()) {
            let value = &imported.values[&value_id];
            assert_eq!(value.shape, vec![2, 3]);
            assert_eq!(value.dtype, crate::tensor::dtype::DTypeId::F32.descriptor());
        }
    }

    #[test]
    fn an_unmapped_op_is_refused_rather_than_silently_dropped() {
        assert!(operation_from_onnx("SomeUnknownOnnxOp", None, &[]).is_none());
    }

    #[test]
    fn trilu_disambiguates_by_the_upper_attribute() {
        let upper_attr = onnx::AttributeProto {
            name: Some("upper".into()),
            r#type: Some(onnx::attribute_proto::AttributeType::Int as i32),
            i: Some(0),
            ..Default::default()
        };
        assert_eq!(
            operation_from_onnx("Trilu", None, &[upper_attr]),
            Some(crate::shapes::error::OperationKind::Tril)
        );
        // Absent `upper` defaults to the ONNX spec's own default: upper.
        assert_eq!(
            operation_from_onnx("Trilu", None, &[]),
            Some(crate::shapes::error::OperationKind::Triu)
        );
    }

    #[test]
    fn matmul_and_conv_disambiguate_by_the_first_operand_rank() {
        assert_eq!(
            operation_from_onnx("MatMul", Some(2), &[]),
            Some(crate::shapes::error::OperationKind::MatMulExact)
        );
        assert_eq!(
            operation_from_onnx("MatMul", Some(3), &[]),
            Some(crate::shapes::error::OperationKind::BatchedMatMul)
        );
        assert_eq!(
            operation_from_onnx("Conv", Some(3), &[]),
            Some(crate::shapes::error::OperationKind::Conv1dExact)
        );
        assert_eq!(
            operation_from_onnx("Conv", Some(4), &[]),
            Some(crate::shapes::error::OperationKind::Conv2dExact)
        );
    }
}
