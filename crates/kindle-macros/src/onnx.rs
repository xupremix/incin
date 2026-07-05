use onnx_pb::ModelProto;
use prost::Message;
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use syn::Ident;

#[derive(Serialize, Deserialize)]
struct OnnxMeta {
    param_names: Vec<String>,
    param_shapes: Vec<Vec<usize>>,
    user_inputs: Vec<String>,
    forward_stmts: Vec<String>,
    last_output: String,
}

pub(crate) fn parse_onnx(rel_path: &str, root_name: &Ident) -> proc_macro2::TokenStream {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = PathBuf::from(manifest_dir).join(rel_path);
    let meta_path = full_path.with_extension(format!("{}.kindle_meta", full_path.extension().unwrap().to_str().unwrap()));

    let use_cache = if std::env::var("KINDLE_DISABLE_META_CACHE").unwrap_or_default() == "1" {
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

        let mut user_inputs = Vec::new();
        for input in &graph.input {
            let raw_name = input.name.clone();
            if !param_set.contains(&raw_name) {
                user_inputs.push(raw_name);
            }
        }

        let mut forward_stmts = Vec::new();
        for raw_name in &param_names {
            let ident = format_ident!("_{}", raw_name.replace(".", "_"));
            forward_stmts.push(quote! {
                let #ident = self.#ident.as_tensor()?;
            }.to_string());
        }

        for node in &graph.node {
            let op_type = &node.op_type;
            let out = format_ident!("_{}", node.output[0].replace(".", "_"));
            let ins: Vec<_> = node
                .input
                .iter()
                .map(|n| format_ident!("_{}", n.replace(".", "_")))
                .collect();

            let stmt = match op_type.as_str() {
                "Gemm" => {
                    let a = &ins[0];
                    let b = &ins[1];
                    if ins.len() > 2 {
                        let c = &ins[2];
                        quote! { let #out = #a.matmul(&#b.transpose::<0, 1>()?)?.add(&#c)?; }
                    } else {
                        quote! { let #out = #a.matmul(&#b.transpose::<0, 1>()?)?; }
                    }
                }
                "Relu" => {
                    let x = &ins[0];
                    quote! { let #out = #x.relu()?; }
                }
                "Add" => {
                    let a = &ins[0];
                    let b = &ins[1];
                    quote! { let #out = #a.add(&#b)?; }
                }
                "MatMul" => {
                    let a = &ins[0];
                    let b = &ins[1];
                    quote! { let #out = #a.matmul(&#b)?; }
                }
                _ => {
                    let msg = format!("Unsupported ONNX operation: {}", op_type);
                    quote! { compile_error!(#msg); }
                }
            };
            forward_stmts.push(stmt.to_string());
        }

        let last_output = if let Some(last_node) = graph.node.last() {
            let out = format_ident!("_{}", last_node.output[0].replace(".", "_"));
            quote! { Ok(#out) }.to_string()
        } else {
            quote! { compile_error!("ONNX graph has no nodes."); }.to_string()
        };

        let m = OnnxMeta {
            param_names,
            param_shapes,
            user_inputs,
            forward_stmts,
            last_output,
        };

        if let Ok(json) = serde_json::to_string(&m) {
            let _ = fs::write(&meta_path, json);
        }
        m
    };

    let mut fields = Vec::new();
    let mut inits = Vec::new();
    for (i, raw_name) in meta.param_names.iter().enumerate() {
        let ident = format_ident!("_{}", raw_name.replace(".", "_"));
        fields.push(quote! { pub #ident: kindle::nn::Param<kindle::prelude::Dyn, B> });

        let shape = &meta.param_shapes[i];
        let dims = shape.iter().map(|&d| quote! { #d });
        inits.push(quote! {
            #ident: kindle::nn::Param::zeros(std::vec![#(#dims),*]).unwrap()
        });
    }

    let mut user_inputs = Vec::new();
    for raw_name in &meta.user_inputs {
        let ident = format_ident!("_{}", raw_name.replace(".", "_"));
        user_inputs.push(quote! { #ident: kindle::prelude::Tensor<kindle::prelude::Dyn, B> });
    }

    let mut forward_stmts = Vec::new();
    for stmt_str in &meta.forward_stmts {
        let tokens: proc_macro2::TokenStream = stmt_str.parse().unwrap();
        forward_stmts.push(tokens);
    }
    let last_output: proc_macro2::TokenStream = meta.last_output.parse().unwrap();

    let root_impl = quote! {
        #[kindle::prelude::module]
        pub struct #root_name<B: kindle::prelude::Backend> {
            #(#fields,)*
            _marker: std::marker::PhantomData<B>,
        }

        impl<B: kindle::prelude::Backend> #root_name<B> {
            pub fn new() -> Self {
                Self {
                    #(#inits,)*
                    _marker: std::marker::PhantomData,
                }
            }

            pub fn load_default_weights(&mut self) -> kindle::prelude::Result<()> {
                // kindle::nn::load_onnx(self, #rel_path)
                Ok(())
            }
        }

        #[kindle::prelude::forward]
        impl<B: kindle::prelude::Backend> #root_name<B> {
            pub fn forward(&self, #(#user_inputs),*) -> kindle::prelude::Result<kindle::prelude::Tensor<kindle::prelude::Dyn, B>> {
                #(#forward_stmts)*
                #last_output
            }
        }
    };

    root_impl
}
