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

/// Reads a sharded checkpoint's `weight_map` with duplicate-key rejection and
/// bare-file-name validation, mirroring the runtime reader's contract.
///
/// Returns the resolved absolute shard paths in sorted order plus every
/// mapped tensor's shape, keyed by tensor name.
/// Resolved shard paths plus each mapped tensor's shape.
type IndexShapes = (Vec<PathBuf>, BTreeMap<String, Vec<usize>>);

fn resolve_index_shapes(index_path: &std::path::Path) -> Result<IndexShapes> {
    fn malformed(message: impl std::fmt::Display) -> syn::Error {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("safetensors index: {message}"),
        )
    }

    struct WeightMapVisitor;
    impl<'de> serde::de::Visitor<'de> for WeightMapVisitor {
        type Value = Vec<(String, String)>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a JSON object mapping tensor names to shard file names")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut pairs = Vec::new();
            while let Some(name) = map.next_key::<String>()? {
                let shard: String = map.next_value()?;
                if pairs.iter().any(|(seen, _)| seen == &name) {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate entry for tensor `{name}`"
                    )));
                }
                pairs.push((name, shard));
            }
            Ok(pairs)
        }
    }

    #[derive(serde::Deserialize)]
    struct RawIndex {
        #[serde(default)]
        #[serde(deserialize_with = "reject_duplicates")]
        weight_map: Vec<(String, String)>,
    }
    fn reject_duplicates<'de, D>(de: D) -> std::result::Result<Vec<(String, String)>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        de.deserialize_map(WeightMapVisitor)
    }

    let raw = fs::read_to_string(index_path)
        .map_err(|e| malformed(format!("reading {:?} failed: {e}", index_path.display())))?;
    let parsed: RawIndex =
        serde_json::from_str(&raw).map_err(|e| malformed(format!("invalid index JSON: {e}")))?;

    let root = index_path.parent().ok_or_else(|| {
        malformed(format!(
            "{:?} has no parent directory",
            index_path.display()
        ))
    })?;

    let mut shard_set = std::collections::BTreeSet::new();
    for (tensor, shard) in &parsed.weight_map {
        if shard.is_empty()
            || std::path::Path::new(shard).is_absolute()
            || !std::path::Path::new(shard)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)))
        {
            return Err(malformed(format!(
                "tensor `{tensor}` maps to `{shard}`, which is not a bare file name under the checkpoint root"
            )));
        }
        shard_set.insert(shard.clone());
    }

    let mut shard_paths = Vec::new();
    for shard in &shard_set {
        let path = root.join(shard);
        if !path.is_file() {
            return Err(malformed(format!(
                "shard `{shard}` referenced by the index is missing at {}",
                path.display()
            )));
        }
        shard_paths.push(path);
    }

    let mut shapes = BTreeMap::new();
    for shard_path in &shard_paths {
        let buffer = fs::read(shard_path)
            .map_err(|e| malformed(format!("reading {shard_path:?} failed: {e}")))?;
        let st = SafeTensors::deserialize(&buffer)
            .map_err(|e| malformed(format!("parsing {shard_path:?} failed: {e:?}")))?;
        for (tname, tensor) in st.tensors() {
            if shapes
                .insert(tname.clone(), tensor.shape().to_vec())
                .is_some()
            {
                return Err(malformed(format!(
                    "tensor `{tname}` appears in more than one shard"
                )));
            }
        }
    }
    Ok((shard_paths, shapes))
}

/// Newest modification time across the index and its shards, used for
/// metadata-cache validity so a re-sharded checkpoint invalidates the cache.
fn newest_input_mtime(
    index_path: &std::path::Path,
    shard_paths: &[PathBuf],
) -> Option<std::time::SystemTime> {
    let mut newest = fs::metadata(index_path).ok()?.modified().ok()?;
    for path in shard_paths {
        if let Ok(time) = fs::metadata(path).ok()?.modified()
            && time > newest
        {
            newest = time;
        }
    }
    Some(newest)
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
                let mut shape_ty = quote! { ::incin::types::Nil };
                for &d in shape.iter().rev() {
                    let path = quote! { ::incin::prelude:: };
                    let d_ty = crate::shape::lit_to_typenum(d, &path);
                    shape_ty = quote! { ::incin::types::DimCons<#d_ty, #shape_ty> };
                }
                let ty = if *is_buffer {
                    quote! { ::incin::prelude::Buffer<#shape_ty, B> }
                } else {
                    quote! { ::incin::prelude::Param<#shape_ty, B> }
                };

                let _common_bound =
                    quote! { Var<K> = <B as ::incin::__macro_support::VariableBackend>::Var<K> };
                bounds.push(
                    quote! { B: ::incin::__macro_support::Backend + ::incin::__macro_support::VariableBackend },
                );
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
        pub struct #name<B: ::incin::__macro_support::Backend>
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
    } else if !rel_path.ends_with(".safetensors") && !rel_path.ends_with(".json") {
        let msg = format!("Unsupported model file format: {}", rel_path);
        return quote! { compile_error!(#msg); }.into();
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = PathBuf::from(manifest_dir).join(&rel_path);

    // A `.json` path names a sharded checkpoint index: resolve every shard
    // offline, right here, from explicitly named local files only.
    let shard_paths = if rel_path.ends_with(".json") {
        match resolve_index_shapes(&full_path) {
            Ok((paths, _shapes)) => paths,
            Err(error) => {
                let message = error.to_string();
                return quote! { compile_error!(#message); }.into();
            }
        }
    } else {
        Vec::new()
    };

    let Some(extension) = full_path
        .extension()
        .and_then(|extension| extension.to_str())
    else {
        let msg = format!("Model path has no valid UTF-8 extension: {rel_path}");
        return quote! { compile_error!(#msg); }.into();
    };
    let meta_path = full_path.with_extension(format!("{extension}.incin_meta"));

    let mut root = Node::Dir(BTreeMap::new());

    let newest_input = if shard_paths.is_empty() {
        fs::metadata(&full_path)
            .ok()
            .and_then(|m| m.modified().ok())
    } else {
        newest_input_mtime(&full_path, &shard_paths)
    };

    let use_cache = if std::env::var("INCIN_DISABLE_META_CACHE").unwrap_or_default() == "1" {
        false
    } else if let (Some(orig_time), Ok(cache_meta)) = (newest_input, fs::metadata(&meta_path)) {
        match cache_meta.modified() {
            Ok(cache_time) => cache_time >= orig_time,
            Err(_) => false,
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
        // Parse from the original input(s): one safetensors file, or every
        // shard of an index in sorted order.
        let mut meta_map = BTreeMap::new();
        let mut record_shard =
            |buffer: &[u8], source: &PathBuf| -> std::result::Result<(), String> {
                let st = SafeTensors::deserialize(buffer)
                    .map_err(|e| format!("Failed to parse safetensors file {source:?}: {e:?}"))?;
                for (tname, tensor) in st.tensors() {
                    let shape = tensor.shape().to_vec();
                    let parts: Vec<&str> = tname.split('.').collect();
                    root.insert(&parts, shape.clone());
                    if meta_map.insert(tname.clone(), shape).is_some() {
                        return Err(format!("tensor `{tname}` appears in more than one shard"));
                    }
                }
                Ok(())
            };

        let outcome: std::result::Result<(), String> = if shard_paths.is_empty() {
            fs::read(&full_path)
                .map_err(|e| format!("Failed to read safetensors file {:?}: {}", full_path, e))
                .and_then(|buffer| record_shard(&buffer, &full_path))
        } else {
            shard_paths.iter().try_for_each(|shard_path| {
                fs::read(shard_path)
                    .map_err(|e| format!("Failed to read shard {shard_path:?}: {e}"))
                    .and_then(|buffer| record_shard(&buffer, shard_path))
            })
        };

        if let Err(message) = outcome {
            return quote! { compile_error!(#message); }.into();
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
        impl<B: ::incin::__macro_support::Backend> #root_name<B>
        where
            #(#bounds,)*
        {
            /// Load default weights.
            pub fn load_default_weights(&mut self) -> ::incin::prelude::Result<()> {
                ::incin::__macro_support::load_safetensors(self, #path_str)
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

    /// Writes one shard file and returns its byte length.
    fn write_shard(dir: &std::path::Path, file: &str, tensors: &[(&str, Vec<f32>)]) -> u64 {
        let raw: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, values)| values.iter().flat_map(|v| v.to_le_bytes()).collect())
            .collect();
        let mut views = BTreeMap::new();
        for ((name, values), bytes) in tensors.iter().zip(&raw) {
            let view = safetensors::tensor::TensorView::new(
                safetensors::tensor::Dtype::F32,
                vec![values.len()],
                bytes,
            )
            .unwrap();
            views.insert((*name).to_string(), view);
        }
        let path = dir.join(file);
        safetensors::tensor::serialize_to_file(&views, None, &path).unwrap();
        std::fs::metadata(&path).unwrap().len()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("incin-macro-index-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_index_resolves_all_shards_into_one_sorted_shape_map() {
        let dir = scratch("resolve");
        write_shard(&dir, "b.safetensors", &[("head.bias", vec![2.0, 3.0])]);
        write_shard(&dir, "a.safetensors", &[("layer.w", vec![1.0])]);
        let index_path = write_index_fixture(
            &dir,
            r#"{"weight_map": {"head.bias": "b.safetensors", "layer.w": "a.safetensors"}}"#,
        );

        let (paths, shapes) = resolve_index_shapes(&index_path).unwrap();
        // Shards are visited in sorted order regardless of map order.
        assert!(paths[0].ends_with("a.safetensors"));
        assert!(paths[1].ends_with("b.safetensors"));
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes["head.bias"], vec![2]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn duplicate_weight_map_keys_are_rejected() {
        let dir = scratch("dup");
        write_shard(&dir, "a.safetensors", &[("x", vec![1.0])]);
        let index_path = write_index_fixture(
            &dir,
            r#"{"weight_map": {"x": "a.safetensors", "x": "a.safetensors"}}"#,
        );
        let message = resolve_index_shapes(&index_path)
            .err()
            .map(|e| e.to_string())
            .expect("duplicate keys must be rejected");
        assert!(message.contains("`x`"), "{message}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn traversal_in_a_shard_reference_is_rejected() {
        let dir = scratch("traversal");
        let index_path =
            write_index_fixture(&dir, r#"{"weight_map": {"x": "../outside.safetensors"}}"#);
        let message = resolve_index_shapes(&index_path)
            .err()
            .map(|e| e.to_string())
            .expect("traversal must be rejected");
        assert!(message.contains("bare file name"), "{message}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_missing_shard_is_rejected_by_name() {
        let dir = scratch("missing");
        let index_path = write_index_fixture(&dir, r#"{"weight_map": {"x": "gone.safetensors"}}"#);
        let message = resolve_index_shapes(&index_path)
            .err()
            .map(|e| e.to_string())
            .expect("a missing shard must be rejected");
        assert!(message.contains("gone.safetensors"), "{message}");
        std::fs::remove_dir_all(dir).ok();
    }

    fn write_index_fixture(dir: &std::path::Path, body: &str) -> PathBuf {
        let path = dir.join("model.safetensors.index.json");
        std::fs::write(&path, body).unwrap();
        path
    }

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
