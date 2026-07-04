use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;
use std::fs;
use std::path::PathBuf;
use onnx_pb::ModelProto;
use prost::Message;

pub(crate) fn parse_onnx(rel_path: &str, root_name: &Ident) -> proc_macro2::TokenStream {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = PathBuf::from(manifest_dir).join(rel_path);

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

    let mut fields = Vec::new();
    let mut inits = Vec::new();
    let mut param_names = std::collections::HashSet::new();

    // 1. Initializers (Parameters)
    for init in &graph.initializer {
        let raw_name = &init.name;
        param_names.insert(raw_name.clone());
        let ident = format_ident!("_{}", raw_name.replace(".", "_"));
        fields.push(quote! { pub #ident: kindle::nn::Param<kindle::prelude::Dyn, B> });

        let shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
        let dims = shape.iter().map(|&d| quote! { #d });
        inits.push(quote! {
            #ident: kindle::nn::Param::zeros(std::vec![#(#dims),*]).unwrap()
        });
    }

    // 2. User Inputs (graph.input excluding initializers)
    let mut user_inputs = Vec::new();
    let mut user_input_idents = Vec::new();
    for input in &graph.input {
        let raw_name = &input.name;
        if !param_names.contains(raw_name) {
            let ident = format_ident!("_{}", raw_name.replace(".", "_"));
            user_inputs.push(quote! { #ident: kindle::prelude::Tensor<kindle::prelude::Dyn, B> });
            user_input_idents.push(ident);
        }
    }

    // 3. Forward pass statements
    let mut forward_stmts = Vec::new();

    // Preload parameters as tensors
    for raw_name in &param_names {
        let ident = format_ident!("_{}", raw_name.replace(".", "_"));
        forward_stmts.push(quote! {
            let #ident = self.#ident.as_tensor()?;
        });
    }

    // Process Nodes
    for node in &graph.node {
        let op_type = &node.op_type;
        let out = format_ident!("_{}", node.output[0].replace(".", "_"));
        let ins: Vec<_> = node.input.iter().map(|n| format_ident!("_{}", n.replace(".", "_"))).collect();

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
        forward_stmts.push(stmt);
    }

    // Return the last output
    let last_output = if let Some(last_node) = graph.node.last() {
        let out = format_ident!("_{}", last_node.output[0].replace(".", "_"));
        quote! { Ok(#out) }
    } else {
        quote! { compile_error!("ONNX graph has no nodes."); }
    };

    let root_impl = quote! {
        #[kindle::prelude::module]
        pub struct #root_name<B: kindle::prelude::Backend<kindle::prelude::Dyn>> {
            #(#fields,)*
            _marker: std::marker::PhantomData<B>,
        }

        impl<B: kindle::prelude::Backend<kindle::prelude::Dyn>> #root_name<B> {
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
        impl<B: kindle::prelude::Backend<kindle::prelude::Dyn>> #root_name<B> {
            pub fn forward(&self, #(#user_inputs),*) -> kindle::prelude::Result<kindle::prelude::Tensor<kindle::prelude::Dyn, B>> {
                #(#forward_stmts)*
                #last_output
            }
        }
    };

    root_impl
}
