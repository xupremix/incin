use onnx_pb::{GraphProto, ModelProto, NodeProto, ValueInfoProto};
use prost::Message;
use quote::{format_ident, quote};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use syn::Ident;

#[derive(Debug, Clone, PartialEq, Eq)]
enum OnnxDim {
    Const(usize),
    Dyn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RankMetadata {
    Known(Vec<OnnxDim>),
    Unknown,
}

fn compile_error(message: impl AsRef<str>) -> proc_macro2::TokenStream {
    let message = message.as_ref();
    quote! { compile_error!(#message); }
}

fn display_domain(domain: &str) -> &str {
    if domain.is_empty() { "ai.onnx" } else { domain }
}

fn opset_for(model: &ModelProto, domain: &str) -> Option<i64> {
    model
        .opset_import
        .iter()
        .filter(|entry| entry.domain == domain || (domain == "ai.onnx" && entry.domain.is_empty()))
        .map(|entry| entry.version)
        .max()
}

fn node_identity(node: &NodeProto, index: usize) -> String {
    if !node.name.is_empty() {
        node.name.clone()
    } else if let Some(output) = node.output.first().filter(|name| !name.is_empty()) {
        output.clone()
    } else {
        format!("node#{index}")
    }
}

fn node_diagnostic(model: &ModelProto, node: &NodeProto, index: usize, reason: &str) -> String {
    let domain = display_domain(&node.domain);
    let opset = opset_for(model, domain)
        .map(|version| version.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "unsupported ONNX node: identity='{}', operation='{}', domain='{}', opset={}: {}",
        node_identity(node, index),
        node.op_type,
        domain,
        opset,
        reason
    )
}

fn extract_rank(value: &ValueInfoProto) -> Result<RankMetadata, String> {
    let Some(value_type) = &value.r#type else {
        return Ok(RankMetadata::Unknown);
    };
    let Some(onnx_pb::type_proto::Value::TensorType(tensor)) = &value_type.value else {
        return Err(format!(
            "malformed ONNX metadata for value '{}': expected a tensor type",
            value.name
        ));
    };
    if tensor.elem_type != onnx_pb::tensor_proto::DataType::Float as i32 {
        return Err(format!(
            "unsupported ONNX value dtype for '{}': element type {} (only FLOAT is currently supported)",
            value.name, tensor.elem_type
        ));
    }
    let Some(shape) = &tensor.shape else {
        return Ok(RankMetadata::Unknown);
    };

    let mut dims = Vec::with_capacity(shape.dim.len());
    for (axis, dim) in shape.dim.iter().enumerate() {
        match &dim.value {
            Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(dim_value)) => {
                let checked_dim = usize::try_from(*dim_value).map_err(|_| {
                    format!(
                        "malformed ONNX metadata for value '{}': dimension {} is negative ({})",
                        value.name, axis, dim_value
                    )
                })?;
                dims.push(OnnxDim::Const(checked_dim));
            }
            Some(onnx_pb::tensor_shape_proto::dimension::Value::DimParam(_)) | None => {
                dims.push(OnnxDim::Dyn);
            }
        }
    }
    Ok(RankMetadata::Known(dims))
}

fn require_known_rank<'a>(
    value_name: &str,
    ranks: &'a BTreeMap<String, RankMetadata>,
) -> Result<&'a [OnnxDim], String> {
    match ranks.get(value_name) {
        Some(RankMetadata::Known(rank)) => Ok(rank),
        Some(RankMetadata::Unknown) | None => Err(format!(
            "ONNX value '{}' has unknown rank; the importer preserves unknown rank and cannot generate a static tensor signature",
            value_name
        )),
    }
}

fn validate_arity(
    model: &ModelProto,
    node: &NodeProto,
    index: usize,
    inputs: usize,
    outputs: usize,
) -> Result<(), String> {
    if node.input.len() != inputs || node.output.len() != outputs {
        return Err(node_diagnostic(
            model,
            node,
            index,
            &format!(
                "expected {inputs} input(s) and {outputs} output(s), got {} input(s) and {} output(s)",
                node.input.len(),
                node.output.len()
            ),
        ));
    }
    if node.input.iter().any(String::is_empty) || node.output.iter().any(String::is_empty) {
        return Err(node_diagnostic(
            model,
            node,
            index,
            "input and output value names must be non-empty",
        ));
    }
    Ok(())
}

fn parse_graph_nodes(
    model: &ModelProto,
    graph: &GraphProto,
    values: &mut BTreeMap<String, Ident>,
) -> Result<Vec<proc_macro2::TokenStream>, String> {
    let mut statements = Vec::with_capacity(graph.node.len());

    for (index, node) in graph.node.iter().enumerate() {
        let domain = display_domain(&node.domain);
        if opset_for(model, domain).is_none() {
            return Err(node_diagnostic(
                model,
                node,
                index,
                "the model does not declare this operator-set domain",
            ));
        }
        if domain != "ai.onnx" {
            return Err(node_diagnostic(
                model,
                node,
                index,
                "custom operator domains are not implemented",
            ));
        }
        if matches!(node.op_type.as_str(), "If" | "Loop") {
            return Err(node_diagnostic(
                model,
                node,
                index,
                "control-flow expansion is not implemented",
            ));
        }
        if !node.attribute.is_empty() {
            return Err(node_diagnostic(
                model,
                node,
                index,
                "operator attributes are not implemented for this importer path",
            ));
        }

        let (input_count, output_count) = match node.op_type.as_str() {
            "Identity" | "Relu" => (1, 1),
            "Add" | "MatMul" => (2, 1),
            _ => {
                return Err(node_diagnostic(
                    model,
                    node,
                    index,
                    "operation is not implemented",
                ));
            }
        };
        validate_arity(model, node, index, input_count, output_count)?;

        let inputs = node
            .input
            .iter()
            .map(|name| {
                values.get(name).cloned().ok_or_else(|| {
                    node_diagnostic(
                        model,
                        node,
                        index,
                        &format!("input value '{name}' is undefined at this node"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let output_name = &node.output[0];
        if values.contains_key(output_name) {
            return Err(node_diagnostic(
                model,
                node,
                index,
                &format!("output value '{output_name}' is defined more than once"),
            ));
        }
        let output = format_ident!("_node_{index}_output");

        let statement = match node.op_type.as_str() {
            "Identity" => {
                let input = &inputs[0];
                quote! { let #output = #input.clone(); }
            }
            "Relu" => {
                let input = &inputs[0];
                quote! { let #output = #input.relu()?; }
            }
            "Add" => {
                let left = &inputs[0];
                let right = &inputs[1];
                quote! { let #output = #left.add(&#right)?; }
            }
            "MatMul" => {
                let left = &inputs[0];
                let right = &inputs[1];
                quote! { let #output = #left.matmul(&#right)?; }
            }
            _ => unreachable!("the operation was matched above"),
        };
        values.insert(output_name.clone(), output);
        statements.push(statement);
    }

    Ok(statements)
}

fn shape_tokens(dims: &[OnnxDim]) -> proc_macro2::TokenStream {
    let dims = dims.iter().map(|dim| match dim {
        OnnxDim::Const(value) => {
            let path = quote! { ::incin::prelude:: };
            crate::shape::lit_to_typenum(*value, &path)
        }
        OnnxDim::Dyn => quote! { usize },
    });
    quote! { (#(#dims,)*) }
}

fn expand_model(model: &ModelProto, root_name: &Ident) -> Result<proc_macro2::TokenStream, String> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| "malformed ONNX model: missing graph".to_string())?;
    if model.opset_import.is_empty() {
        return Err("malformed ONNX model: missing operator-set import".to_string());
    }
    if graph.node.is_empty() {
        return Err("malformed ONNX graph: no nodes".to_string());
    }
    if graph.output.len() != 1 {
        return Err(format!(
            "unsupported ONNX graph output count: expected exactly 1, got {}",
            graph.output.len()
        ));
    }
    if let Some(initializer) = graph.initializer.first() {
        return Err(format!(
            "ONNX initializer loading is not implemented: initializer '{}' must not be replaced with fabricated values",
            initializer.name
        ));
    }
    if !graph.sparse_initializer.is_empty() {
        return Err(
            "ONNX sparse initializer loading is not implemented; fabricated values are forbidden"
                .to_string(),
        );
    }

    let mut ranks = BTreeMap::new();
    for value in graph
        .input
        .iter()
        .chain(graph.output.iter())
        .chain(graph.value_info.iter())
    {
        if value.name.is_empty() {
            return Err("malformed ONNX metadata: value name is empty".to_string());
        }
        let rank = extract_rank(value)?;
        if ranks.insert(value.name.clone(), rank).is_some() {
            return Err(format!(
                "malformed ONNX metadata: value '{}' is described more than once",
                value.name
            ));
        }
    }

    let mut values = BTreeMap::new();
    let mut input_parameters = Vec::new();
    let mut seen_inputs = BTreeSet::new();
    for (index, input) in graph.input.iter().enumerate() {
        if !seen_inputs.insert(input.name.clone()) {
            return Err(format!(
                "malformed ONNX graph: input '{}' is declared more than once",
                input.name
            ));
        }
        let dims = require_known_rank(&input.name, &ranks)?;
        let shape = shape_tokens(dims);
        let ident = format_ident!("_input_{index}");
        values.insert(input.name.clone(), ident.clone());
        input_parameters.push(quote! { #ident: ::incin::prelude::Tensor<#shape, B, K> });
    }

    let statements = parse_graph_nodes(model, graph, &mut values)?;
    let output_info = &graph.output[0];
    let output_dims = require_known_rank(&output_info.name, &ranks)?;
    let output_shape = shape_tokens(output_dims);
    let final_output = values.get(&output_info.name).ok_or_else(|| {
        format!(
            "malformed ONNX graph: declared output '{}' is never produced",
            output_info.name
        )
    })?;

    Ok(quote! {
        #[::incin::prelude::module]
        pub struct #root_name<B: ::incin::prelude::Backend, K: ::incin::prelude::DType = f32> {
            #[module(ignore)]
            _marker: ::core::marker::PhantomData<(B, K)>,
        }

        impl<B: ::incin::prelude::Backend, K: ::incin::prelude::ConstDType> #root_name<B, K>
        where
            B: ::incin::prelude::SupportsDType<K>,
            B::Device: ::incin::prelude::ConstDevice,
        {
            /// Creates a stateless imported graph.
            pub const fn new() -> Self {
                Self { _marker: ::core::marker::PhantomData }
            }

            /// Executes the imported eager graph.
            pub fn forward(
                &self,
                #(#input_parameters),*
            ) -> ::incin::prelude::Result<::incin::prelude::Tensor<#output_shape, B, K>> {
                #(#statements)*
                #final_output.into_shape::<#output_shape>()
            }
        }
    })
}

pub(crate) fn parse_onnx(
    rel_path: &str,
    root_name: &Ident,
    _no_meta: bool,
) -> proc_macro2::TokenStream {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = PathBuf::from(manifest_dir).join(rel_path);
    let buffer = match fs::read(&full_path) {
        Ok(buffer) => buffer,
        Err(error) => {
            return compile_error(format!(
                "failed to read ONNX file '{}': {error}",
                full_path.display()
            ));
        }
    };
    let model = match ModelProto::decode(buffer.as_ref()) {
        Ok(model) => model,
        Err(error) => return compile_error(format!("failed to parse ONNX protobuf: {error}")),
    };
    match expand_model(&model, root_name) {
        Ok(tokens) => tokens,
        Err(error) => compile_error(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_pb::{
        OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto,
        tensor_shape_proto::{Dimension, dimension},
        type_proto,
    };

    fn value(name: &str, dims: Option<Vec<i64>>) -> ValueInfoProto {
        let shape = dims.map(|dims| TensorShapeProto {
            dim: dims
                .into_iter()
                .map(|dim| Dimension {
                    value: Some(dimension::Value::DimValue(dim)),
                    ..Default::default()
                })
                .collect(),
        });
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: onnx_pb::tensor_proto::DataType::Float as i32,
                    shape,
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn model_with(node: NodeProto) -> ModelProto {
        ModelProto {
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 18,
            }],
            graph: Some(GraphProto {
                node: vec![node],
                input: vec![value("x", Some(vec![1]))],
                output: vec![value("y", Some(vec![1]))],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn identity_node() -> NodeProto {
        NodeProto {
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            name: "identity_0".to_string(),
            op_type: "Identity".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn supported_expansion_has_no_weight_loading_claim_or_runtime_panic() {
        let tokens = expand_model(&model_with(identity_node()), &format_ident!("Imported"))
            .expect("identity graph should expand")
            .to_string();
        assert!(!tokens.contains("load_default_weights"));
        assert!(!tokens.contains("Param :: zeros"));
        assert!(!tokens.contains("panic !"));
    }

    #[test]
    fn unknown_rank_remains_unknown_and_fails_static_expansion() {
        let mut model = model_with(identity_node());
        model.graph.as_mut().unwrap().input[0] = value("x", None);
        let error = expand_model(&model, &format_ident!("Imported")).unwrap_err();
        assert!(error.contains("value 'x' has unknown rank"), "{error}");
    }

    #[test]
    fn negative_dimension_is_a_precise_metadata_error() {
        let mut model = model_with(identity_node());
        model.graph.as_mut().unwrap().input[0] = value("x", Some(vec![-1]));
        let error = expand_model(&model, &format_ident!("Imported")).unwrap_err();
        assert!(error.contains("dimension 0 is negative (-1)"), "{error}");
    }

    #[test]
    fn initializers_are_not_represented_by_zero_filled_parameters() {
        let mut model = model_with(identity_node());
        model.graph.as_mut().unwrap().initializer.push(TensorProto {
            name: "weight".to_string(),
            ..Default::default()
        });
        let error = expand_model(&model, &format_ident!("Imported")).unwrap_err();
        assert!(error.contains("initializer 'weight'"), "{error}");
        assert!(error.contains("fabricated values"), "{error}");
    }

    #[test]
    fn if_and_loop_fail_during_expansion_with_complete_node_identity() {
        for operation in ["If", "Loop"] {
            let mut node = identity_node();
            node.op_type = operation.to_string();
            node.name = "control_0".to_string();
            let error = expand_model(&model_with(node), &format_ident!("Imported")).unwrap_err();
            assert!(error.contains("identity='control_0'"), "{error}");
            assert!(
                error.contains(&format!("operation='{operation}'")),
                "{error}"
            );
            assert!(error.contains("domain='ai.onnx'"), "{error}");
            assert!(error.contains("opset=18"), "{error}");
            assert!(
                error.contains("control-flow expansion is not implemented"),
                "{error}"
            );
        }
    }

    #[test]
    fn unsupported_nodes_report_identity_operation_domain_and_opset() {
        let mut node = identity_node();
        node.name = "custom_0".to_string();
        node.op_type = "Mystery".to_string();
        let error = expand_model(&model_with(node), &format_ident!("Imported")).unwrap_err();
        assert_eq!(
            error,
            "unsupported ONNX node: identity='custom_0', operation='Mystery', domain='ai.onnx', opset=18: operation is not implemented"
        );
    }

    #[test]
    fn malformed_node_arity_is_rejected_before_codegen() {
        let mut node = identity_node();
        node.output.clear();
        let error = expand_model(&model_with(node), &format_ident!("Imported")).unwrap_err();
        assert!(
            error.contains("expected 1 input(s) and 1 output(s)"),
            "{error}"
        );
    }
}
