use proc_macro::TokenStream;
use quote::quote;
use safetensors::SafeTensors;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use syn::{
    Ident, LitStr, Result, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

/// Import model input.
struct ImportModelInput {
    path: LitStr,
    _comma: Token![,],
    name: Ident,
    no_meta: bool,
}

impl Parse for ImportModelInput {
    /// Parse.
    fn parse(input: ParseStream) -> Result<Self> {
        let path: LitStr = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let name: Ident = input.parse()?;
        let mut no_meta = false;

        if input.peek(Token![,]) {
            let _c: Token![,] = input.parse()?;
            if input.peek(syn::token::Brace) {
                let content;
                syn::braced!(content in input);
                let ident: Ident = content.parse()?;
                if ident == "no_meta" {
                    let _c2: Token![:] = content.parse()?;
                    let lit: syn::LitBool = content.parse()?;
                    no_meta = lit.value;
                }
            }
        }

        Ok(Self {
            path,
            _comma,
            name,
            no_meta,
        })
    }
}

/// Node.
enum Node {
    /// Leaf.
    Leaf { shape: Vec<usize>, is_buffer: bool },
    /// Dir.
    Dir(BTreeMap<String, Node>),
}

impl Node {
    /// Insert.
    fn insert(&mut self, path: &[&str], shape: Vec<usize>) {
        if path.len() == 1 {
            let is_buffer = path[0] == "running_mean"
                || path[0] == "running_var"
                || path[0] == "num_batches_tracked";
            if let Node::Dir(map) = self {
                map.insert(path[0].to_string(), Node::Leaf { shape, is_buffer });
            }
        } else if let Node::Dir(map) = self {
            let entry = map
                .entry(path[0].to_string())
                .or_insert_with(|| Node::Dir(BTreeMap::new()));
            entry.insert(&path[1..], shape);
        }
    }
}

/// Generate structs.
fn generate_structs(
    name: &Ident,
    node: &Node,
    structs: &mut Vec<proc_macro2::TokenStream>,
    bounds: &mut Vec<proc_macro2::TokenStream>,
) -> std::result::Result<(), String> {
    let Node::Dir(map) = node else {
        return Ok(());
    };

    let mut fields = Vec::new();

    for (k, v) in map {
        if k.is_empty() {
            return Err(format!(
                "safetensors tensor path for generated model `{name}` contains an empty segment"
            ));
        }
        let field_name = if k.starts_with(|character: char| character.is_ascii_digit()) {
            format!("_{k}")
        } else {
            k.clone()
        };
        let field_name_ident = syn::parse_str::<Ident>(&field_name).map_err(|_| {
            format!("safetensors tensor path segment `{k}` is not a valid Rust field identifier")
        })?;

        match v {
            Node::Leaf { shape, is_buffer } => {
                let shape_tokens = shape.iter().map(|&d| {
                    let path = quote! { ::incin::prelude:: };
                    crate::shape::lit_to_typenum(d, &path)
                });
                let shape_ty = quote! { (#(#shape_tokens,)*) };
                let ty = if *is_buffer {
                    quote! { ::incin::prelude::Buffer<#shape_ty, B> }
                } else {
                    quote! { ::incin::prelude::Param<#shape_ty, B> }
                };

                let _common_bound = quote! { Var<K> = <B as ::incin::prelude::VariableBackend>::Var<K> };
                bounds.push(quote! { B: ::incin::prelude::Backend + ::incin::prelude::VariableBackend });
                fields.push(quote! { pub #field_name_ident: #ty });
            }
            Node::Dir(_) => {
                let sub_struct_name =
                    syn::parse_str::<Ident>(&format!("{name}_{k}")).map_err(|_| {
                        format!("safetensors tensor path segment `{k}` cannot name a Rust type")
                    })?;
                let ty = quote! { #sub_struct_name<B> };
                fields.push(quote! { pub #field_name_ident: #ty });
                generate_structs(&sub_struct_name, v, structs, bounds)?;
            }
        }
    }

    let def = quote! {
        #[::incin::prelude::module]
        #[allow(non_camel_case_types)]
        pub struct #name<B: ::incin::prelude::Backend>
        where
            #(#bounds,)*
        {
            #(#fields,)*
        }
    };
    structs.push(def);
    Ok(())
}

pub(crate) fn import_model(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ImportModelInput);

    let rel_path = input.path.value();
    let root_name = input.name.clone();

    if rel_path.ends_with(".onnx") {
        return crate::onnx::parse_onnx(&rel_path, &root_name, input.no_meta).into();
    } else if rel_path.ends_with(".pt") || rel_path.ends_with(".pth") {
        let msg =
            "TorchScript parsing is scheduled for a future update! Use .onnx or .safetensors."
                .to_string();
        return quote! { compile_error!(#msg); }.into();
    } else if !rel_path.ends_with(".safetensors") {
        let msg = format!("Unsupported model file format: {}", rel_path);
        return quote! { compile_error!(#msg); }.into();
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = PathBuf::from(manifest_dir).join(&rel_path);
    let Some(extension) = full_path
        .extension()
        .and_then(|extension| extension.to_str())
    else {
        let msg = format!("Model path has no valid UTF-8 extension: {rel_path}");
        return quote! { compile_error!(#msg); }.into();
    };
    let meta_path = full_path.with_extension(format!("{extension}.incin_meta"));

    let mut root = Node::Dir(BTreeMap::new());

    let use_cache = if std::env::var("INCIN_DISABLE_META_CACHE").unwrap_or_default() == "1" {
        false
    } else if let (Ok(orig_meta), Ok(cache_meta)) =
        (fs::metadata(&full_path), fs::metadata(&meta_path))
    {
        if let (Ok(orig_time), Ok(cache_time)) = (orig_meta.modified(), cache_meta.modified()) {
            cache_time >= orig_time
        } else {
            false
        }
    } else {
        false
    };

    if use_cache
        && let Ok(cache_buffer) = fs::read_to_string(&meta_path)
        && let Ok(map) = serde_json::from_str::<BTreeMap<String, Vec<usize>>>(&cache_buffer)
    {
        for (tname, shape) in map {
            let parts: Vec<&str> = tname.split('.').collect();
            root.insert(&parts, shape);
        }
    }

    if let Node::Dir(ref map) = root
        && map.is_empty()
    {
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

    let mut structs = Vec::new();
    let mut bounds = Vec::new();

    if let Err(message) = generate_structs(&root_name, &root, &mut structs, &mut bounds) {
        return quote! { compile_error!(#message); }.into();
    }

    // root implementation of load_default_weights
    let path_str = input.path.value();
    let root_impl = quote! {
        impl<B: ::incin::prelude::Backend> #root_name<B>
        where
            #(#bounds,)*
        {
            /// Load default weights.
            pub fn load_default_weights(&mut self) -> ::incin::prelude::Result<()> {
                ::incin::prelude::load_safetensors(self, #path_str)
            }
        }
    };

    let expanded = quote! {
        #(#structs)*
        #root_impl
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_tensor_path_segments_are_diagnostics_not_panics() {
        let root = syn::parse_str::<Ident>("Imported").unwrap();
        for segment in ["", "not-valid", "also invalid"] {
            let mut entries = BTreeMap::new();
            entries.insert(
                segment.to_string(),
                Node::Leaf {
                    shape: vec![1],
                    is_buffer: false,
                },
            );
            let mut structs = Vec::new();
            let mut bounds = Vec::new();
            let error = generate_structs(&root, &Node::Dir(entries), &mut structs, &mut bounds)
                .unwrap_err();
            assert!(error.contains("safetensors tensor path"));
        }
    }

    #[test]
    fn numeric_tensor_path_segment_gets_a_valid_field_name() {
        let root = syn::parse_str::<Ident>("Imported").unwrap();
        let mut entries = BTreeMap::new();
        entries.insert(
            "0".to_string(),
            Node::Leaf {
                shape: vec![1],
                is_buffer: false,
            },
        );
        let mut structs = Vec::new();
        let mut bounds = Vec::new();
        generate_structs(&root, &Node::Dir(entries), &mut structs, &mut bounds).unwrap();
        assert!(structs[0].to_string().contains("_0"));
    }
}
