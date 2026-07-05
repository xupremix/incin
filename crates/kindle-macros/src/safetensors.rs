use proc_macro::TokenStream;
use quote::{format_ident, quote};
use safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use syn::{
    Ident, LitStr, Result, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

struct ImportModelInput {
    path: LitStr,
    _comma: Token![,],
    name: Ident,
}

impl Parse for ImportModelInput {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            path: input.parse()?,
            _comma: input.parse()?,
            name: input.parse()?,
        })
    }
}

enum Node {
    Leaf { shape: Vec<usize>, is_buffer: bool },
    Dir(BTreeMap<String, Node>),
}

impl Node {
    fn insert(&mut self, path: &[&str], shape: Vec<usize>) {
        if path.len() == 1 {
            let is_buffer = path[0] == "running_mean"
                || path[0] == "running_var"
                || path[0] == "num_batches_tracked";
            if let Node::Dir(map) = self {
                map.insert(path[0].to_string(), Node::Leaf { shape, is_buffer });
            }
        } else {
            if let Node::Dir(map) = self {
                let entry = map
                    .entry(path[0].to_string())
                    .or_insert_with(|| Node::Dir(BTreeMap::new()));
                entry.insert(&path[1..], shape);
            }
        }
    }
}

fn generate_structs(
    name: &Ident,
    node: &Node,
    structs: &mut Vec<proc_macro2::TokenStream>,
    bounds: &mut Vec<proc_macro2::TokenStream>,
) {
    let Node::Dir(map) = node else {
        return;
    };

    let mut fields = Vec::new();

    for (k, v) in map {
        let field_name_ident = if k.chars().next().unwrap().is_ascii_digit() {
            format_ident!("_{}", k)
        } else {
            format_ident!("{}", k)
        };

        match v {
            Node::Leaf { shape, is_buffer } => {
                let shape_tokens = shape.iter().map(|&d| quote! { kindle::prelude::Const<#d> });
                let shape_ty = quote! { (#(#shape_tokens,)*) };
                let ty = if *is_buffer {
                    quote! { kindle::nn::Buffer<#shape_ty, B> }
                } else {
                    quote! { kindle::nn::Param<#shape_ty, B> }
                };

                let _common_bound = quote! { RawVar = <B as kindle::prelude::Backend>::RawVar, RawTensor = <B as kindle::prelude::Backend>::RawTensor };
                bounds.push(quote! { B: kindle::prelude::Backend });
                fields.push(quote! { pub #field_name_ident: #ty });
            }
            Node::Dir(_) => {
                let sub_struct_name = format_ident!("{}_{}", name, k);
                let ty = quote! { #sub_struct_name<B> };
                fields.push(quote! { pub #field_name_ident: #ty });
                generate_structs(&sub_struct_name, v, structs, bounds);
            }
        }
    }

    let def = quote! {
        #[kindle::prelude::module]
        #[allow(non_camel_case_types)]
        pub struct #name<B: kindle::prelude::Backend>
        where
            #(#bounds,)*
        {
            #(#fields,)*
        }
    };
    structs.push(def);
}

pub(crate) fn import_model(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ImportModelInput);

    let rel_path = input.path.value();
    let root_name = input.name.clone();

    if rel_path.ends_with(".onnx") {
        return crate::onnx::parse_onnx(&rel_path, &root_name).into();
    } else if rel_path.ends_with(".pt") || rel_path.ends_with(".pth") {
        let msg = format!(
            "TorchScript parsing is scheduled for a future update! Use .onnx or .safetensors."
        );
        return quote! { compile_error!(#msg); }.into();
    } else if !rel_path.ends_with(".safetensors") {
        let msg = format!("Unsupported model file format: {}", rel_path);
        return quote! { compile_error!(#msg); }.into();
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = PathBuf::from(manifest_dir).join(&rel_path);
    let meta_path = full_path.with_extension(format!("{}.kindle_meta", full_path.extension().unwrap().to_str().unwrap()));

    let mut root = Node::Dir(BTreeMap::new());

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

    if use_cache {
        if let Ok(cache_buffer) = fs::read_to_string(&meta_path) {
            if let Ok(map) = serde_json::from_str::<BTreeMap<String, Vec<usize>>>(&cache_buffer) {
                for (tname, shape) in map {
                    let parts: Vec<&str> = tname.split('.').collect();
                    root.insert(&parts, shape);
                }
            }
        }
    }

    if let Node::Dir(ref map) = root {
        if map.is_empty() {
            // Need to parse from original file
            let buffer = match fs::read(&full_path) {
                Ok(b) => b,
                Err(e) => {
                    let msg = format!("Failed to read safetensors file {:?}: {}", full_path, e);
                    return quote! { compile_error!(#msg); }.into();
                }
            };

            let st = match SafeTensors::deserialize(&buffer) {
                Ok(st) => st,
                Err(e) => {
                    let msg = format!("Failed to parse safetensors file {:?}: {:?}", full_path, e);
                    return quote! { compile_error!(#msg); }.into();
                }
            };

            let mut meta_map = BTreeMap::new();
            for (tname, tensor) in st.tensors() {
                let shape = tensor.shape().to_vec();
                let parts: Vec<&str> = tname.split('.').collect();
                root.insert(&parts, shape.clone());
                meta_map.insert(tname.clone(), shape);
            }

            // Save cache
            if let Ok(json) = serde_json::to_string(&meta_map) {
                let _ = fs::write(&meta_path, json);
            }
        }
    }

    let mut structs = Vec::new();
    let mut bounds = Vec::new();

    generate_structs(&root_name, &root, &mut structs, &mut bounds);

    // root implementation of load_default_weights
    let path_str = input.path.value();
    let root_impl = quote! {
        impl<B: kindle::prelude::Backend> #root_name<B>
        where
            #(#bounds,)*
        {
            pub fn load_default_weights(&mut self) -> kindle::prelude::Result<()> {
                kindle::nn::load_safetensors(self, #path_str)
            }
        }
    };

    let expanded = quote! {
        #(#structs)*
        #root_impl
    };

    TokenStream::from(expanded)
}
