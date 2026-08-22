use crate::onnx_pb;
use crate::onnx_pb::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto};
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

/// Fully validated, embedded ONNX initializer data.  Values are kept as IEEE
/// bit patterns so expansion is host-independent, including for NaNs and infinities.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Initializer {
    name: String,
    dims: Vec<usize>,
    values: Vec<u32>,
}

fn compile_error(message: impl AsRef<str>) -> proc_macro2::TokenStream {
    let message = message.as_ref();
    quote! { compile_error!(#message); }
}

/// Read a proto2 `optional` string the way the ONNX specification defines it.
///
/// The generated message types model every `optional` field as `Option`, while
/// the specification's default for an absent name, domain, or op type is the
/// empty string. Reading through here keeps the import rules below written
/// against the value rather than against the encoding, which is what they were
/// written against when these types came from a proto3 schema.
fn text(field: &Option<String>) -> &str {
    field.as_deref().unwrap_or_default()
}

/// The same, for an absent 32-bit field whose specified default is zero.
fn int32(field: Option<i32>) -> i32 {
    field.unwrap_or_default()
}

/// The same, for an absent 64-bit field whose specified default is zero.
fn int64(field: Option<i64>) -> i64 {
    field.unwrap_or_default()
}

fn display_domain(domain: &str) -> &str {
    if domain.is_empty() { "ai.onnx" } else { domain }
}

fn opset_for(model: &ModelProto, domain: &str) -> Option<i64> {
    model
        .opset_import
        .iter()
        .filter(|entry| {
            text(&entry.domain) == domain || (domain == "ai.onnx" && text(&entry.domain).is_empty())
        })
        .map(|entry| int64(entry.version))
        .max()
}

fn node_identity(node: &NodeProto, index: usize) -> String {
    if !text(&node.name).is_empty() {
        text(&node.name).to_string()
    } else if let Some(output) = node.output.first().filter(|name| !name.is_empty()) {
        output.clone()
    } else {
        format!("node#{index}")
    }
}

fn node_diagnostic(model: &ModelProto, node: &NodeProto, index: usize, reason: &str) -> String {
    let domain = display_domain(text(&node.domain));
    let opset = opset_for(model, domain)
        .map(|version| version.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "unsupported ONNX node: identity='{}', operation='{}', domain='{}', opset={}: {}",
        node_identity(node, index),
        text(&node.op_type),
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
            text(&value.name)
        ));
    };
    if int32(tensor.elem_type) != onnx_pb::tensor_proto::DataType::Float as i32 {
        return Err(format!(
            "unsupported ONNX value dtype for '{}': element type {} (only FLOAT is currently supported)",
            text(&value.name),
            int32(tensor.elem_type)
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
                        text(&value.name),
                        axis,
                        dim_value
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

fn initializer_dims(initializer: &TensorProto) -> Result<(Vec<usize>, usize), String> {
    let mut dims = Vec::with_capacity(initializer.dims.len());
    let mut elements = 1usize;
    for (axis, dim) in initializer.dims.iter().enumerate() {
        let dim = usize::try_from(*dim).map_err(|_| {
            format!(
                "malformed ONNX initializer '{}': dimension {axis} is negative ({dim})",
                text(&initializer.name)
            )
        })?;
        elements = elements.checked_mul(dim).ok_or_else(|| {
            format!(
                "malformed ONNX initializer '{}': element count overflows usize",
                text(&initializer.name)
            )
        })?;
        dims.push(dim);
    }
    Ok((dims, elements))
}

fn parse_initializer(initializer: &TensorProto) -> Result<Initializer, String> {
    let name = text(&initializer.name);
    if name.is_empty() {
        return Err("malformed ONNX initializer: name is empty".to_string());
    }
    if int32(initializer.data_type) != onnx_pb::tensor_proto::DataType::Float as i32 {
        return Err(format!(
            "unsupported ONNX initializer '{name}': element type {} (only FLOAT is currently supported)",
            int32(initializer.data_type)
        ));
    }
    if initializer.segment.is_some() {
        return Err(format!(
            "unsupported ONNX initializer '{name}': segmented tensor data is not supported"
        ));
    }
    if !initializer.external_data.is_empty()
        || int32(initializer.data_location) == onnx_pb::tensor_proto::DataLocation::External as i32
    {
        return Err(format!(
            "unsupported ONNX initializer '{name}': external tensor data is not supported"
        ));
    }

    let (dims, elements) = initializer_dims(initializer)?;
    let has_non_float_typed_data = !initializer.int32_data.is_empty()
        || !initializer.string_data.is_empty()
        || !initializer.int64_data.is_empty()
        || !initializer.double_data.is_empty()
        || !initializer.uint64_data.is_empty();
    if has_non_float_typed_data {
        return Err(format!(
            "malformed ONNX initializer '{name}': FLOAT data must use raw_data or float_data"
        ));
    }

    let values = match (&initializer.raw_data, initializer.float_data.is_empty()) {
        (Some(_), false) => {
            return Err(format!(
                "malformed ONNX initializer '{name}': raw_data and float_data are ambiguous"
            ));
        }
        (Some(raw), true) => {
            let expected = elements
                .checked_mul(core::mem::size_of::<f32>())
                .ok_or_else(|| {
                    format!("malformed ONNX initializer '{name}': byte count overflows usize")
                })?;
            if raw.len() != expected {
                return Err(format!(
                    "malformed ONNX initializer '{name}': raw_data has {} bytes, expected {expected}",
                    raw.len()
                ));
            }
            raw.chunks_exact(4)
                .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect()
        }
        (None, false) => {
            if initializer.float_data.len() != elements {
                return Err(format!(
                    "malformed ONNX initializer '{name}': float_data has {} values, expected {elements}",
                    initializer.float_data.len()
                ));
            }
            initializer
                .float_data
                .iter()
                .map(|value| value.to_bits())
                .collect()
        }
        (None, true) if elements == 0 => Vec::new(),
        (None, true) => {
            return Err(format!(
                "malformed ONNX initializer '{name}': missing raw_data or float_data payload"
            ));
        }
    };

    Ok(Initializer {
        name: name.to_string(),
        dims,
        values,
    })
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
        let domain = display_domain(text(&node.domain));
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
        if matches!(text(&node.op_type), "If" | "Loop") {
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

        let (input_count, output_count) = match text(&node.op_type) {
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

        let statement = match text(&node.op_type) {
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
                quote! { let #output = #left.broadcast_add(&#right)?; }
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
    let mut chain = quote! { ::incin::types::Nil };
    for dim in dims.iter().rev() {
        let d = match dim {
            OnnxDim::Const(value) => {
                let path = quote! { ::incin::prelude:: };
                crate::shape::lit_to_typenum(*value, &path)
            }
            OnnxDim::Dyn => quote! { usize },
        };
        chain = quote! { ::incin::types::DimCons<#d, #chain> };
    }
    chain
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
        if text(&value.name).is_empty() {
            return Err("malformed ONNX metadata: value name is empty".to_string());
        }
        let rank = extract_rank(value)?;
        if ranks.insert(text(&value.name).to_string(), rank).is_some() {
            return Err(format!(
                "malformed ONNX metadata: value '{}' is described more than once",
                text(&value.name)
            ));
        }
    }

    let mut initializers = Vec::with_capacity(graph.initializer.len());
    let mut initializer_names = BTreeSet::new();
    let graph_input_names = graph
        .input
        .iter()
        .map(|input| text(&input.name))
        .collect::<BTreeSet<_>>();
    for initializer in &graph.initializer {
        let initializer = parse_initializer(initializer)?;
        if !initializer_names.insert(initializer.name.clone()) {
            return Err(format!(
                "malformed ONNX graph: initializer '{}' is declared more than once",
                initializer.name
            ));
        }
        if graph_input_names.contains(initializer.name.as_str()) {
            let input_rank = ranks.get(&initializer.name).ok_or_else(|| {
                format!(
                    "malformed ONNX graph: initializer '{}' has no input metadata",
                    initializer.name
                )
            })?;
            match input_rank {
                RankMetadata::Known(input_dims)
                    if input_dims
                        .iter()
                        .map(|dim| match dim {
                            OnnxDim::Const(value) => Some(*value),
                            OnnxDim::Dyn => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .as_deref()
                        == Some(initializer.dims.as_slice()) => {}
                RankMetadata::Known(_) | RankMetadata::Unknown => {
                    return Err(format!(
                        "malformed ONNX graph: initializer '{}' does not exactly match its input metadata",
                        initializer.name
                    ));
                }
            }
        }
        initializers.push(initializer);
    }

    let mut op_bounds = Vec::new();
    for node in &graph.node {
        match text(&node.op_type) {
            "MatMul" => {
                op_bounds.push(quote! {
                    B: ::incin::__macro_support::Execute<::incin::__macro_support::op::MatMulExact>,
                    <B as ::incin::__macro_support::Execute<::incin::__macro_support::op::MatMulExact>>::Output: Into<B::Storage<f32>>,
                });
            }
            "Add" => {
                op_bounds.push(quote! {
                    B: ::incin::__macro_support::Execute<::incin::__macro_support::op::Add>,
                    <B as ::incin::__macro_support::Execute<::incin::__macro_support::op::Add>>::Output: Into<B::Storage<f32>>,
                });
            }
            "Relu" => {
                op_bounds.push(quote! {
                    B: ::incin::__macro_support::Execute<::incin::__macro_support::op::Relu>,
                    <B as ::incin::__macro_support::Execute<::incin::__macro_support::op::Relu>>::Output: Into<B::Storage<f32>>,
                });
            }
            _ => {}
        }
    }

    if initializers.is_empty() {
        let mut values = BTreeMap::new();
        let mut input_parameters = Vec::new();
        let mut seen_inputs = BTreeSet::new();
        for (index, input) in graph.input.iter().enumerate() {
            if !seen_inputs.insert(text(&input.name).to_string()) {
                return Err(format!(
                    "malformed ONNX graph: input '{}' is declared more than once",
                    text(&input.name)
                ));
            }
            let dims = require_known_rank(text(&input.name), &ranks)?;
            let shape = shape_tokens(dims);
            let ident = format_ident!("_input_{index}");
            values.insert(text(&input.name).to_string(), ident.clone());
            input_parameters.push(quote! { #ident: ::incin::prelude::Tensor<#shape, B, K> });
        }

        let mut op_bounds_empty = Vec::new();
        for node in &graph.node {
            match text(&node.op_type) {
                "MatMul" => {
                    op_bounds_empty.push(quote! {
                        B: ::incin::__macro_support::Execute<::incin::__macro_support::op::MatMulExact>,
                        <B as ::incin::__macro_support::Execute<::incin::__macro_support::op::MatMulExact>>::Output: Into<B::Storage<K>>,
                    });
                }
                "Add" => {
                    op_bounds_empty.push(quote! {
                        B: ::incin::__macro_support::Execute<::incin::__macro_support::op::Add>,
                        <B as ::incin::__macro_support::Execute<::incin::__macro_support::op::Add>>::Output: Into<B::Storage<K>>,
                    });
                }
                "Relu" => {
                    op_bounds_empty.push(quote! {
                        B: ::incin::__macro_support::Execute<::incin::__macro_support::op::Relu>,
                        <B as ::incin::__macro_support::Execute<::incin::__macro_support::op::Relu>>::Output: Into<B::Storage<K>>,
                    });
                }
                _ => {}
            }
        }

        let statements = parse_graph_nodes(model, graph, &mut values)?;
        let output_info = &graph.output[0];
        let output_dims = require_known_rank(text(&output_info.name), &ranks)?;
        let output_shape = shape_tokens(output_dims);
        let final_output = values.get(text(&output_info.name)).ok_or_else(|| {
            format!(
                "malformed ONNX graph: declared output '{}' is never produced",
                text(&output_info.name)
            )
        })?;

        return Ok(quote! {
        #[::incin::prelude::module(no_to_device)]
        pub struct #root_name<B: ::incin::__macro_support::Backend, K: ::incin::prelude::DType = f32> {
            #[module(ignore)]
            _marker: ::core::marker::PhantomData<(B, K)>,
        }

        impl<B: ::incin::__macro_support::Backend, K: ::incin::prelude::ConstDType> #root_name<B, K>
        where
            B: ::incin::__macro_support::SupportsDType<K> + ::incin::__macro_support::Capabilities,
            B::Device: ::incin::prelude::ConstDevice,
            #(#op_bounds_empty)*
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
        });
    }

    let mut values = BTreeMap::new();
    let mut input_parameters = Vec::new();
    let mut seen_inputs = BTreeSet::new();
    for (index, input) in graph.input.iter().enumerate() {
        if !seen_inputs.insert(text(&input.name).to_string()) {
            return Err(format!(
                "malformed ONNX graph: input '{}' is declared more than once",
                text(&input.name)
            ));
        }
        if initializer_names.contains(text(&input.name)) {
            continue;
        }
        let dims = require_known_rank(text(&input.name), &ranks)?;
        let shape = shape_tokens(dims);
        let ident = format_ident!("_input_{index}");
        values.insert(text(&input.name).to_string(), ident.clone());
        input_parameters.push(quote! { #ident: ::incin::prelude::Tensor<#shape, B, f32> });
    }

    let mut initializer_fields = Vec::with_capacity(initializers.len());
    let mut initializer_construction = Vec::with_capacity(initializers.len());
    let mut initializer_idents = Vec::with_capacity(initializers.len());
    let mut initializer_loads = Vec::with_capacity(initializers.len());
    for (index, initializer) in initializers.iter().enumerate() {
        let field = format_ident!("initializer_{index:03}");
        let state_name = format!("initializer_{index:03}");
        let shape = shape_tokens(
            &initializer
                .dims
                .iter()
                .copied()
                .map(OnnxDim::Const)
                .collect::<Vec<_>>(),
        );
        let value_bits = initializer.values.iter().map(|bits| {
            quote! { ::core::primitive::f32::from_bits(#bits) }
        });
        let tensor = format_ident!("_{field}_tensor");
        let raw = format_ident!("_{field}_raw");
        values.insert(initializer.name.clone(), tensor.clone());
        initializer_idents.push(field.clone());
        initializer_loads.push(quote! {
            let #tensor = self.#field.as_tensor()?;
        });
        initializer_fields.push(quote! {
            #[state(name = #state_name)]
            #field: ::incin::prelude::Param<#shape, B>,
        });
        initializer_construction.push(quote! {
            let #tensor = ::incin::prelude::Tensor::<#shape, B>::from_slice(&[#(#value_bits),*], ())?;
            let #raw = <B as ::incin::__macro_support::VariableBackend>::var_from_tensor(#tensor.inner())?;
            let #field = ::incin::prelude::Param::<#shape, B>::from_parts_checked(
                #raw,
                #tensor.shape_buf().clone(),
                <f32 as ::incin::prelude::DType>::init(()),
                <B::Device as ::incin::prelude::Device>::init(()),
            )?;
        });
    }

    let statements = parse_graph_nodes(model, graph, &mut values)?;
    let output_info = &graph.output[0];
    let output_dims = require_known_rank(text(&output_info.name), &ranks)?;
    let output_shape = shape_tokens(output_dims);
    let final_output = values.get(text(&output_info.name)).ok_or_else(|| {
        format!(
            "malformed ONNX graph: declared output '{}' is never produced",
            text(&output_info.name)
        )
    })?;

    Ok(quote! {
        #[::incin::prelude::module(no_to_device)]
        pub struct #root_name<B: ::incin::__macro_support::Backend>
        where
            B: ::incin::__macro_support::VariableBackend
                + ::incin::__macro_support::SupportsDType<f32>,
            B::Device: ::incin::prelude::ConstDevice,
        {
            #(#initializer_fields)*
        }

        impl<B> #root_name<B>
        where
            B: ::incin::__macro_support::Backend
                + ::incin::__macro_support::VariableBackend
                + ::incin::__macro_support::SupportsDType<f32>
                + ::incin::__macro_support::Capabilities
                + ::incin::__macro_support::Execute<::incin::__macro_support::op::TensorFromData>,
            <B as ::incin::__macro_support::Execute<::incin::__macro_support::op::TensorFromData>>::Output:
                Into<B::Storage<f32>>,
            B::Device: ::incin::prelude::ConstDevice,
            #(#op_bounds)*
        {
            /// Creates the imported graph with its embedded ONNX initializers.
            pub fn new() -> ::incin::prelude::Result<Self> {
                #(#initializer_construction)*
                Ok(Self { #(#initializer_idents),* })
            }

            /// Executes the imported eager graph.
            pub fn forward(
                &self,
                #(#input_parameters),*
            ) -> ::incin::prelude::Result<::incin::prelude::Tensor<#output_shape, B, f32, ::incin::prelude::Grad>> {
                #(#initializer_loads)*
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
    use crate::onnx_pb::{
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
            name: Some(name.to_string()),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: Some(onnx_pb::tensor_proto::DataType::Float as i32),
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
                domain: Some(String::new()),
                version: Some(18),
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
            name: Some("identity_0".to_string()),
            op_type: Some("Identity".to_string()),
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

    fn initializer(name: &str, dims: Vec<i64>, values: Vec<f32>) -> TensorProto {
        TensorProto {
            name: Some(name.to_string()),
            dims,
            data_type: Some(onnx_pb::tensor_proto::DataType::Float as i32),
            float_data: values,
            ..Default::default()
        }
    }

    #[test]
    fn dense_float_initializer_generates_checked_parameter_construction() {
        let mut model = model_with(identity_node());
        let graph = model.graph.as_mut().unwrap();
        graph.node[0] = NodeProto {
            input: vec!["weight".to_string()],
            output: vec!["y".to_string()],
            name: Some("identity_weight".to_string()),
            op_type: Some("Identity".to_string()),
            ..Default::default()
        };
        graph
            .initializer
            .push(initializer("weight", vec![1], vec![2.5]));
        let tokens = expand_model(&model, &format_ident!("Imported"))
            .expect("initializer graph should expand")
            .to_string();
        assert!(tokens.contains("initializer_000"), "{tokens}");
        assert!(tokens.contains("Param"), "{tokens}");
        assert!(tokens.contains("from_bits"), "{tokens}");
        assert!(!tokens.contains("Param :: zeros"), "{tokens}");
    }

    #[test]
    fn initializer_rejects_malformed_payloads_and_encodings() {
        let cases = [
            (
                TensorProto {
                    name: Some("weight".to_string()),
                    dims: vec![1],
                    data_type: Some(onnx_pb::tensor_proto::DataType::Float as i32),
                    raw_data: Some(vec![0, 0, 0]),
                    ..Default::default()
                },
                "raw_data has 3 bytes, expected 4",
            ),
            (
                TensorProto {
                    name: Some("weight".to_string()),
                    dims: vec![1],
                    data_type: Some(onnx_pb::tensor_proto::DataType::Float as i32),
                    raw_data: Some(vec![0, 0, 0, 0]),
                    float_data: vec![0.0],
                    ..Default::default()
                },
                "raw_data and float_data are ambiguous",
            ),
            (
                TensorProto {
                    name: Some("weight".to_string()),
                    dims: vec![1],
                    data_type: Some(onnx_pb::tensor_proto::DataType::Float as i32),
                    ..Default::default()
                },
                "missing raw_data or float_data payload",
            ),
        ];
        for (initializer, expected) in cases {
            let error = parse_initializer(&initializer).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn initializer_raw_data_uses_little_endian_f32_bits() {
        let initializer = TensorProto {
            name: Some("weight".to_string()),
            dims: vec![2],
            data_type: Some(onnx_pb::tensor_proto::DataType::Float as i32),
            raw_data: Some([1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat()),
            ..Default::default()
        };
        assert_eq!(
            parse_initializer(&initializer).unwrap().values,
            vec![1.5f32.to_bits(), (-2.0f32).to_bits()]
        );
    }

    #[test]
    fn imported_initializer_runs_and_has_a_deterministic_state_key() {
        let test_root = std::env::temp_dir().join(format!(
            "incin-onnx-initializer-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock must be after the Unix epoch")
                .as_nanos()
        ));
        let source_dir = test_root.join("src");
        fs::create_dir_all(&source_dir).expect("create hermetic ONNX fixture crate");

        let model = ModelProto {
            opset_import: vec![OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(18),
            }],
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    input: vec!["x".to_string(), "weight".to_string()],
                    output: vec!["y".to_string()],
                    name: Some("matmul".to_string()),
                    op_type: Some("MatMul".to_string()),
                    ..Default::default()
                }],
                input: vec![value("x", Some(vec![1, 2]))],
                output: vec![value("y", Some(vec![1, 1]))],
                initializer: vec![initializer("weight", vec![2, 1], vec![2.0, 3.0])],
                ..Default::default()
            }),
            ..Default::default()
        };
        fs::write(test_root.join("model.onnx"), model.encode_to_vec())
            .expect("write hermetic ONNX fixture");

        let incin_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../incin")
            .canonicalize()
            .expect("resolve incin facade path");
        fs::write(
            test_root.join("Cargo.toml"),
            format!(
                "[package]\nname = \"onnx-initializer-fixture\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\nincin = {{ path = {:?}, features = [\"cpu\"] }}\n",
                incin_path
            ),
        )
        .expect("write hermetic fixture manifest");
        fs::write(
            source_dir.join("lib.rs"),
            r#"
use incin::prelude::*;

incin::experimental::model!("model.onnx", Imported);

#[test]
fn initializers_are_parameters_with_exact_values() {
    let model = Imported::<DefaultBackend>::new().unwrap();
    let state = incin::state::collect_state::<DefaultBackend, _>(&model).unwrap();
    let (path, value) = state.iter().next().unwrap();
    assert_eq!(path.as_str(), "initializer_000");
    assert_eq!(value.bytes(), &[0, 0, 0, 64, 0, 0, 64, 64]);
    let input = Tensor::<s![1, 2], DefaultBackend>::from_slice(&[3.0, 4.0], ()).unwrap();
    let output = model.forward(input).unwrap();
    assert!(format!("{output}").contains("18"));
}
"#,
        )
        .expect("write hermetic fixture source");

        let status = std::process::Command::new(env!("CARGO"))
            .arg("test")
            .arg("--offline")
            .arg("--quiet")
            .current_dir(&test_root)
            .status()
            .expect("run hermetic ONNX fixture crate");
        let _ = fs::remove_dir_all(&test_root);
        assert!(status.success(), "hermetic ONNX fixture crate failed");
    }

    #[test]
    fn if_and_loop_fail_during_expansion_with_complete_node_identity() {
        for operation in ["If", "Loop"] {
            let mut node = identity_node();
            node.op_type = Some(operation.to_string());
            node.name = Some("control_0".to_string());
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
        node.name = Some("custom_0".to_string());
        node.op_type = Some("Mystery".to_string());
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

    #[test]
    fn generate_dense_initializer_fixture() {
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../incin/tests/fixtures/dense_initializers.onnx");
        let model = ModelProto {
            ir_version: Some(8),
            opset_import: vec![OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(18),
            }],
            graph: Some(GraphProto {
                name: Some("test_dense_initializers".to_string()),
                node: vec![
                    NodeProto {
                        input: vec!["x".to_string(), "weight".to_string()],
                        output: vec!["hidden".to_string()],
                        name: Some("matmul_node".to_string()),
                        op_type: Some("MatMul".to_string()),
                        ..Default::default()
                    },
                    NodeProto {
                        input: vec!["hidden".to_string(), "bias".to_string()],
                        output: vec!["y".to_string()],
                        name: Some("add_node".to_string()),
                        op_type: Some("Add".to_string()),
                        ..Default::default()
                    },
                ],
                input: vec![value("x", Some(vec![1, 2]))],
                output: vec![value("y", Some(vec![1, 3]))],
                initializer: vec![
                    initializer("weight", vec![2, 3], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
                    initializer("bias", vec![1, 3], vec![0.5, 0.5, 0.5]),
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        if let Some(parent) = fixture_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&fixture_path, model.encode_to_vec()).expect("write fixture");
        assert!(fixture_path.exists());
    }
}
