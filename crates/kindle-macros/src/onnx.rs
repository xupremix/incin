use onnx_pb::ModelProto;
use prost::Message;
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use syn::Ident;

#[derive(Serialize, Deserialize, Clone)]
pub enum OnnxDim {
    Const(usize),
    Dyn,
}

#[derive(Serialize, Deserialize)]
struct OnnxMeta {
    param_names: Vec<String>,
    param_shapes: Vec<Vec<usize>>,
    user_inputs: Vec<(String, Vec<OnnxDim>)>, // name and shape
    forward_stmts: Vec<String>,
    last_output: String,
    last_output_shape: Vec<OnnxDim>,
}

fn get_ints_attr(node: &onnx_pb::NodeProto, name: &str) -> Option<Vec<i64>> {
    for attr in &node.attribute {
        if attr.name == name {
            return Some(attr.ints.clone());
        }
    }
    None
}

fn parse_graph_nodes(nodes: &[onnx_pb::NodeProto], shape_map: &std::collections::HashMap<String, Vec<OnnxDim>>) -> Vec<String> {
    let mut stmts = Vec::new();
    for node in nodes {
        let op_type = &node.op_type;
        let out = quote::format_ident!("_{}", node.output[0].replace(".", "_").replace("/", "_").replace("-", "_"));
        let ins: Vec<_> = node
            .input
            .iter()
            .map(|n| quote::format_ident!("_{}", n.replace(".", "_").replace("/", "_").replace("-", "_")))
            .collect();

        let stmt = match op_type.as_str() {
            "Gemm" => {
                let a = &ins[0];
                let b = &ins[1];
                if ins.len() > 2 {
                    let c = &ins[2];
                    quote::quote! { let mut #out = #a.matmul(&#b.transpose::<0, 1>()?)?.add(&#c)?; }
                } else {
                    quote::quote! { let mut #out = #a.matmul(&#b.transpose::<0, 1>()?)?; }
                }
            }
            "Identity" => { let x = &ins[0]; quote::quote! { let mut #out = #x.clone(); } }
            "Relu" => {
                let x = &ins[0];
                quote::quote! { let mut #out = #x.relu()?; }
            }
            "Add" => {
                let a = &ins[0];
                let b = &ins[1];
                quote::quote! { let mut #out = #a.add(&#b)?; }
            }
            "MatMul" => {
                let a = &ins[0];
                let b = &ins[1];
                quote::quote! { let mut #out = #a.matmul(&#b)?; }
            }
            "Conv" => {
                let x = &ins[0];
                let w = &ins[1];
                let b = if ins.len() > 2 {
                    let b_in = &ins[2];
                    quote::quote! { Some(&#b_in) }
                } else {
                    quote::quote! { None }
                };
                let strides = get_ints_attr(node, "strides").unwrap_or(vec![1, 1]);
                let pads = get_ints_attr(node, "pads").unwrap_or(vec![0, 0, 0, 0]);
                let s_val = strides[0] as u64;
                let p_val = pads[0] as u64;
                let s_ident = quote::format_ident!("U{}", s_val);
                let p_ident = quote::format_ident!("U{}", p_val);
                quote::quote! { let mut #out = #x.conv2d::<typenum::#s_ident, typenum::#p_ident, _>(&#w, #b)?; }
            }
            "MaxPool" => {
                let x = &ins[0];
                let strides = get_ints_attr(node, "strides").unwrap_or(vec![1, 1]);
                let pads = get_ints_attr(node, "pads").unwrap_or(vec![0, 0, 0, 0]);
                let dilations = get_ints_attr(node, "dilations").unwrap_or(vec![1, 1]);
                let kernel_shape = get_ints_attr(node, "kernel_shape").unwrap_or(vec![1, 1]);
                let k_val = kernel_shape[0] as u64;
                let s_val = strides[0] as u64;
                let p_val = pads[0] as u64;
                let d_val = dilations[0] as u64;
                let k_ident = quote::format_ident!("U{}", k_val);
                let s_ident = quote::format_ident!("U{}", s_val);
                let p_ident = quote::format_ident!("U{}", p_val);
                let d_ident = quote::format_ident!("U{}", d_val);
                quote::quote! { let mut #out = #x.max_pool2d::<typenum::#k_ident, typenum::#s_ident, typenum::#p_ident, typenum::#d_ident>()?; }
            }
            "BatchNormalization" => {
                let x = &ins[0];
                let scale = &ins[1];
                let b = &ins[2];
                let mean = &ins[3];
                let var = &ins[4];
                let epsilon = 1e-5f32; 
                quote::quote! { let mut #out = #x.batch_norm(&#scale, &#b, &#mean, &#var, #epsilon)?; }
            }
            "Flatten" | "Reshape" => {
                let x = &ins[0];
                let in_name = node.input[0].clone();
                let in_shape = shape_map.get(&in_name).cloned().unwrap_or(vec![OnnxDim::Dyn; 4]);
                let rank = in_shape.len();
                if rank == 2 {
                    quote::quote! { let mut #out = #x.flatten::<1, 1>()?; }
                } else if rank == 3 {
                    quote::quote! { let mut #out = #x.flatten::<1, 2>()?; }
                } else if rank == 4 {
                    quote::quote! { let mut #out = #x.flatten::<1, 3>()?; }
                } else {
                    quote::quote! { let mut #out = #x.clone(); }
                }
            }
            "GlobalAveragePool" => {
                let x = &ins[0];
                quote::quote! { 
                    let mut #out = #x.mean_keepdim::<2>()?; 
                    let mut #out = #out.mean_keepdim::<3>()?; 
                }
            }
            "If" => {
                let cond = &ins[0];
                let then_attr = node.attribute.iter().find(|a| a.name == "then_branch");
                let else_attr = node.attribute.iter().find(|a| a.name == "else_branch");
                
                let then_stmts = if let Some(a) = then_attr {
                    if let Some(ref g) = a.g {
                        let stmts = parse_graph_nodes(&g.node, shape_map);
                        let s: proc_macro2::TokenStream = stmts.join("\n").parse().unwrap_or(quote::quote!{});
                        let last_out = if let Some(n) = g.node.last() {
                            quote::format_ident!("_{}", n.output[0].replace(".", "_").replace("/", "_").replace("-", "_"))
                        } else {
                            quote::format_ident!("_{}", "dummy")
                        };
                        quote::quote! {
                            #s
                            #last_out
                        }
                    } else { quote::quote!{ panic!("If node missing then_branch graph") } }
                } else { quote::quote!{ panic!("If node missing then_branch") } };

                let else_stmts = if let Some(a) = else_attr {
                    if let Some(ref g) = a.g {
                        let stmts = parse_graph_nodes(&g.node, shape_map);
                        let s: proc_macro2::TokenStream = stmts.join("\n").parse().unwrap_or(quote::quote!{});
                        let last_out = if let Some(n) = g.node.last() {
                            quote::format_ident!("_{}", n.output[0].replace(".", "_").replace("/", "_").replace("-", "_"))
                        } else {
                            quote::format_ident!("_{}", "dummy")
                        };
                        quote::quote! {
                            #s
                            #last_out
                        }
                    } else { quote::quote!{ panic!("If node missing else_branch graph") } }
                } else { quote::quote!{ panic!("If node missing else_branch") } };

                quote::quote! {
                    let mut #out = if #cond.to_scalar::<bool>()? {
                        #then_stmts
                    } else {
                        #else_stmts
                    };
                }
            }
            "Loop" => {
                let max_trip_count = &ins[0];
                let cond = &ins[1];
                let v_initial = &ins[2];
                
                let body_attr = node.attribute.iter().find(|a| a.name == "body");
                let body_stmts = if let Some(a) = body_attr {
                    if let Some(ref g) = a.g {
                        let stmts = parse_graph_nodes(&g.node, shape_map);
                        let s: proc_macro2::TokenStream = stmts.join("\n").parse().unwrap_or(quote::quote!{});
                        let last_out = if let Some(n) = g.node.last() {
                            quote::format_ident!("_{}", n.output[0].replace(".", "_").replace("/", "_").replace("-", "_"))
                        } else {
                            quote::format_ident!("_{}", "dummy")
                        };
                        quote::quote! {
                            #s
                            #last_out
                        }
                    } else { quote::quote!{ panic!("Loop node missing body graph") } }
                } else { quote::quote!{ panic!("Loop node missing body attribute") } };

                quote::quote! {
                    let mut _trip = 0;
                    let mut _cond = #cond.to_scalar::<bool>()?;
                    let mut #out = #v_initial.clone();
                    while _cond && _trip < #max_trip_count.to_scalar::<i64>()? {
                        #out = {
                            #body_stmts
                        };
                        _trip += 1;
                    }
                }
            }
            _ => {
                let msg = format!("Unsupported ONNX operation: {}", op_type);
                quote::quote! { compile_error!(#msg); }
            }
        };
        stmts.push(stmt.to_string());
    }
    stmts
}

pub(crate) fn parse_onnx(rel_path: &str, root_name: &Ident, no_meta: bool) -> proc_macro2::TokenStream {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = PathBuf::from(manifest_dir).join(rel_path);
    let meta_path = full_path.with_extension(format!("{}.kindle_meta", full_path.extension().unwrap().to_str().unwrap()));

    let use_cache = if no_meta || std::env::var("KINDLE_NO_META").unwrap_or_default() == "1" || std::env::var("KINDLE_DISABLE_META_CACHE").unwrap_or_default() == "1" {
        false
    } else if let (Ok(orig_meta), Ok(cache_meta)) = (fs::metadata(&full_path), fs::metadata(&meta_path)) {
        if let (Ok(orig_time), Ok(cache_time)) = (orig_meta.modified(), cache_meta.modified()) {
            cache_time >= orig_time
        } else {
            false
        }
    } else {
        false
    };

    let meta = if use_cache {
        if let Ok(cache_buffer) = fs::read_to_string(&meta_path) {
            serde_json::from_str::<OnnxMeta>(&cache_buffer).ok()
        } else {
            None
        }
    } else {
        None
    };

    let meta = if let Some(m) = meta {
        m
    } else {
        let buffer = match fs::read(&full_path) {
            Ok(b) => b,
            Err(e) => {
                let msg = format!("Failed to read ONNX file {:?}: {}", full_path, e);
                return quote! { compile_error!(#msg); };
            }
        };

        let model = match ModelProto::decode(buffer.as_ref()) {
            Ok(m) => m,
            Err(e) => {
                let msg = format!("Failed to parse ONNX protobuf: {}", e);
                return quote! { compile_error!(#msg); };
            }
        };

        let graph = match model.graph {
            Some(g) => g,
            None => {
                return quote! { compile_error!("ONNX model has no graph."); };
            }
        };

        let mut param_names = Vec::new();
        let mut param_shapes = Vec::new();
        let mut param_set = std::collections::HashSet::new();

        for init in &graph.initializer {
            let raw_name = init.name.clone();
            param_set.insert(raw_name.clone());
            param_names.push(raw_name);
            let shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
            param_shapes.push(shape);
        }


        let mut shape_map = std::collections::HashMap::new();
        
        let extract_shape = |v: &onnx_pb::ValueInfoProto| -> Vec<OnnxDim> {
            let mut dims = Vec::new();
            if let Some(t) = &v.r#type {
                if let Some(onnx_pb::type_proto::Value::TensorType(tt)) = &t.value {
                    if let Some(shape) = &tt.shape {
                        for dim in &shape.dim {
                            if let Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(val)) = &dim.value {
                                dims.push(OnnxDim::Const(*val as usize));
                            } else {
                                dims.push(OnnxDim::Dyn);
                            }
                        }
                    }
                }
            }
            dims
        };

        for input in &graph.input {
            shape_map.insert(input.name.clone(), extract_shape(input));
        }
        for output in &graph.output {
            shape_map.insert(output.name.clone(), extract_shape(output));
        }
        for vi in &graph.value_info {
            shape_map.insert(vi.name.clone(), extract_shape(vi));
        }

        let mut user_inputs = Vec::new();
        for input in &graph.input {
            let raw_name = input.name.clone();
            if !param_set.contains(&raw_name) {
                let shape = shape_map.get(&raw_name).cloned().unwrap_or(vec![OnnxDim::Dyn; 4]);
                user_inputs.push((raw_name, shape));
            }
        }


        let mut forward_stmts = Vec::new();
                for raw_name in &param_names {
            let ident = format_ident!("_{}", raw_name.replace(".", "_").replace("/", "_").replace("-", "_"));
            forward_stmts.push(quote! {
                let #ident = self.#ident.as_tensor()?.into_shape::<kindle::prelude::Dyn>()?;
            }.to_string());
        }

        let parsed_stmts = parse_graph_nodes(&graph.node, &shape_map);
        for s in parsed_stmts {
            forward_stmts.push(s);
        }

        let (last_output, last_output_shape) = if let Some(last_node) = graph.node.last() {
            let out_name = last_node.output[0].clone();
            let shape = shape_map.get(&out_name).cloned().unwrap_or(vec![OnnxDim::Dyn; 4]);
            let out_ident = format_ident!("_{}", out_name.replace(".", "_").replace("/", "_").replace("-", "_"));
            (quote! { #out_ident }.to_string(), shape)
        } else {
            (quote! { compile_error!("ONNX graph has no nodes."); }.to_string(), vec![])
        };

        let m = OnnxMeta {
            param_names,
            param_shapes,
            user_inputs,
            forward_stmts,
            last_output,
            last_output_shape,
        };

        if let Ok(json) = serde_json::to_string(&m) {
            let _ = fs::write(&meta_path, json);
        }
        m
    };

    let mut fields = Vec::new();
    let mut inits = Vec::new();
    for (i, raw_name) in meta.param_names.iter().enumerate() {
        let ident = format_ident!("_{}", raw_name.replace(".", "_").replace("/", "_").replace("-", "_"));
        let p_dims = meta.param_shapes[i].iter().map(|&d| quote! { kindle::prelude::Const<#d> });
        fields.push(quote! { pub #ident: kindle::nn::Param<(#(#p_dims,)*), B> });

        inits.push(quote! {
            #ident: kindle::nn::Param::zeros(()).unwrap()
        });
    }

    let mut user_inputs = Vec::new();
    for (raw_name, shape) in &meta.user_inputs {
        let ident = format_ident!("_{}", raw_name.replace(".", "_").replace("/", "_").replace("-", "_"));
        let dims = shape.iter().map(|d| match d {
            OnnxDim::Const(v) => quote! { kindle::prelude::Const<#v> },
            OnnxDim::Dyn => quote! { usize },
        });
        user_inputs.push(quote! { #ident: kindle::prelude::Tensor<(#(#dims,)*), B> });
    }

    let mut forward_stmts = Vec::new();
    for (raw_name, _) in &meta.user_inputs {
        let ident = format_ident!("_{}", raw_name.replace(".", "_").replace("/", "_").replace("-", "_"));
        forward_stmts.push(quote! {
            let #ident = #ident.into_shape::<kindle::prelude::Dyn>()?;
        });
    }
    for stmt_str in &meta.forward_stmts {
        let tokens: proc_macro2::TokenStream = stmt_str.parse().unwrap();
        forward_stmts.push(tokens);
    }
    let last_output: proc_macro2::TokenStream = meta.last_output.parse().unwrap();

    let out_dims = meta.last_output_shape.iter().map(|d| match d {
        OnnxDim::Const(v) => quote! { kindle::prelude::Const<#v> },
        OnnxDim::Dyn => quote! { usize },
    });
    let out_shape_type = quote! { (#(#out_dims,)*) };


    let forward_attr = if user_inputs.len() == 1 {
        quote! { #[kindle::prelude::forward] }
    } else {
        quote! {}
    };

    let root_impl = quote! {
        #[kindle::prelude::module]
        pub struct #root_name<B: kindle::prelude::Backend> {
            #(#fields,)*
            _marker: std::marker::PhantomData<B>,
        }

        impl<B: kindle::prelude::Backend> #root_name<B>
        where
            B::DType: kindle::prelude::ConstDType,
            B::Device: kindle::prelude::ConstDevice,
        {
            pub fn new() -> Self {
                Self {
                    #(#inits,)*
                    _marker: std::marker::PhantomData,
                }
            }

            pub fn load_default_weights(&mut self) -> kindle::prelude::Result<()> {
                Ok(())
            }
        }

        #forward_attr
        impl<B: kindle::prelude::Backend> #root_name<B>
        where
            B::DType: kindle::prelude::ConstDType,
            B::Device: kindle::prelude::ConstDevice,
        {
            pub fn forward(&self, #(#user_inputs),*) -> kindle::prelude::Result<kindle::prelude::Tensor<#out_shape_type, B>> {
                #(#forward_stmts)*
                let final_out = #last_output;
                Ok(final_out.into_shape::<#out_shape_type>()?)
            }
        }
    };

    root_impl
}
