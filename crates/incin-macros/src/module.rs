use proc_macro::TokenStream;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{DeriveInput, Ident, Token, parse_macro_input};

/// The complete struct-level argument vocabulary.
///
/// PROPOSALS.md's macro policy requires every public macro to "have a
/// versioned grammar and reject unknown keys", and until `CI-005` this one
/// did not: the arguments were read with `attr.to_string().contains(..)`, so
/// `#[module(no_such_argument)]` expanded as though it had been written
/// `#[module]`. Substring matching also accepted `#[module(not_internal)]` as
/// `internal`, which is the failure mode that makes a typo change behaviour
/// rather than fail.
const STRUCT_ARGUMENTS: &[&str] = &[
    "internal",
    "no_stats",
    "no_parameters",
    "no_state",
    "no_named_layers",
    "no_shape_info",
    "no_train_mode",
    "no_to_device",
];

/// Parse the struct-level argument list, rejecting anything not in the
/// vocabulary.
fn parse_arguments(attr: TokenStream) -> syn::Result<(bool, bool, bool, bool, bool, bool, bool, bool)> {
    let attr: proc_macro2::TokenStream = attr.into();
    if attr.is_empty() {
        return Ok((false, false, false, false, false, false, false, false));
    }

    let parser = Punctuated::<Ident, Token![,]>::parse_terminated;
    let arguments = syn::parse::Parser::parse2(parser, attr)?;

    let (
        mut internal,
        mut no_stats,
        mut no_parameters,
        mut no_state,
        mut no_named_layers,
        mut no_shape_info,
        mut no_train_mode,
        mut no_to_device,
    ) = (false, false, false, false, false, false, false, false);
    for argument in arguments {
        match () {
            () if argument == "internal" => internal = true,
            () if argument == "no_stats" => no_stats = true,
            () if argument == "no_parameters" => no_parameters = true,
            () if argument == "no_state" => no_state = true,
            () if argument == "no_named_layers" => no_named_layers = true,
            () if argument == "no_shape_info" => no_shape_info = true,
            () if argument == "no_train_mode" => no_train_mode = true,
            () if argument == "no_to_device" => no_to_device = true,
            () => {
                return Err(syn::Error::new_spanned(
                    &argument,
                    format!(
                        "unknown attribute argument for #[module], expected one of {}",
                        STRUCT_ARGUMENTS.join(", ")
                    ),
                ));
            }
        }
    }
    Ok((
        internal,
        no_stats,
        no_parameters,
        no_state,
        no_named_layers,
        no_shape_info,
        no_train_mode,
        no_to_device,
    ))
}

fn parse_parallel_attr(a: &syn::Attribute) -> syn::Result<()> {
    if a.meta.require_path_only().is_ok() {
        return Ok(());
    }
    let nested =
        a.parse_args_with(syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated)?;
    for meta in nested {
        let path = meta.path();
        let key = path.segments.last().map(|s| s.ident.to_string());
        match key.as_deref() {
            Some("mesh" | "stage" | "dp" | "tp" | "pp") => {}
            Some(other) => {
                return Err(syn::Error::new_spanned(
                    meta,
                    format!(
                        "unknown attribute argument for #[parallel], expected one of mesh, stage, dp, tp, pp (found `{other}`)"
                    ),
                ));
            }
            None => {
                return Err(syn::Error::new_spanned(
                    meta,
                    "invalid attribute argument for #[parallel]",
                ));
            }
        }
    }
    Ok(())
}

fn parse_shard_attr(a: &syn::Attribute) -> syn::Result<()> {
    if a.meta.require_path_only().is_ok() {
        return Ok(());
    }
    let nested =
        a.parse_args_with(syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated)?;
    for meta in nested {
        let path = meta.path();
        let key = path.segments.last().map(|s| s.ident.to_string());
        match key.as_deref() {
            Some("mesh" | "axis") => {}
            Some(other) => {
                return Err(syn::Error::new_spanned(
                    meta,
                    format!(
                        "unknown attribute argument for #[shard], expected one of mesh, axis (found `{other}`)"
                    ),
                ));
            }
            None => {
                return Err(syn::Error::new_spanned(
                    meta,
                    "invalid attribute argument for #[shard]",
                ));
            }
        }
    }
    Ok(())
}

fn process_field_attributes(
    attrs: &mut Vec<syn::Attribute>,
) -> Result<(bool, Option<String>), syn::Error> {
    let mut ignore = false;
    let mut state_name = None;
    let mut has_parallel = false;
    let mut has_shard = false;
    let mut err = None;

    attrs.retain(|a| {
        if err.is_some() {
            return true;
        }
        let seg = a.path().segments.last();
        if let Some(seg) = seg {
            let ident_str = seg.ident.to_string();
            match ident_str.as_str() {
                "module" => match a.parse_args::<syn::Ident>() {
                    Ok(i) if i == "ignore" => {
                        ignore = true;
                        false
                    }
                    Ok(i) => {
                        err = Some(syn::Error::new_spanned(
                            i,
                            "unknown attribute argument for #[module], expected `ignore`",
                        ));
                        false
                    }
                    Err(e) => {
                        err = Some(syn::Error::new_spanned(
                            a,
                            format!("invalid #[module] attribute: {}", e),
                        ));
                        false
                    }
                },
                "state" => {
                    let nested = a.parse_args_with(Punctuated::<syn::Meta, Token![,]>::parse_terminated);
                    match nested {
                        Ok(items) => {
                            for meta in items {
                                match meta {
                                    syn::Meta::NameValue(value) if value.path.is_ident("name") => {
                                        match value.value {
                                            syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(lit), .. }) => state_name = Some(lit.value()),
                                            other => err = Some(syn::Error::new_spanned(other, "#[state(name = ...)] requires a string literal")),
                                        }
                                    }
                                    other => err = Some(syn::Error::new_spanned(other, "unknown #[state] argument; expected name = \"...\"")),
                                }
                            }
                        }
                        Err(e) => err = Some(e),
                    }
                    false
                },
                "parallel" => {
                    if has_shard {
                        err = Some(syn::Error::new_spanned(
                            a,
                            "conflicting attributes `#[parallel]` and `#[shard]` on the same field",
                        ));
                        return false;
                    }
                    has_parallel = true;
                    if let Err(e) = parse_parallel_attr(a) {
                        err = Some(e);
                    }
                    false
                }
                "shard" => {
                    if has_parallel {
                        err = Some(syn::Error::new_spanned(
                            a,
                            "conflicting attributes `#[parallel]` and `#[shard]` on the same field",
                        ));
                        return false;
                    }
                    has_shard = true;
                    if let Err(e) = parse_shard_attr(a) {
                        err = Some(e);
                    }
                    false
                }
                _ => true,
            }
        } else {
            true
        }
    });

    if let Some(e) = err {
        Err(e)
    } else {
        Ok((ignore, state_name))
    }
}

pub(crate) fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let (
        is_internal,
        no_stats,
        no_parameters,
        no_state,
        no_named_layers,
        no_shape_info,
        no_train_mode,
        no_to_device,
    ) = match parse_arguments(attr) {
        Ok(parsed) => parsed,
        Err(error) => return error.to_compile_error().into(),
    };

    let mut input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;
    let k_crate = if is_internal {
        quote! { crate }
    } else {
        quote! { ::incin }
    };
    let macro_support = if is_internal {
        quote! { crate::__macro_support }
    } else {
        quote! { ::incin::__macro_support }
    };
    let state_load_plan = if is_internal {
        quote! { crate::nn::StateLoadPlan }
    } else {
        quote! { ::incin::state::StateLoadPlan }
    };

    let format_mac = quote! { #macro_support::format! };

    let backend_generic = input.generics.params.iter().find_map(|p| {
        if let syn::GenericParam::Type(t) = p {
            if t.bounds.iter().any(|b| {
                if let syn::TypeParamBound::Trait(tb) = b {
                    tb.path
                        .segments
                        .last()
                        .map(|s| s.ident == "Backend" || s.ident == "VariableBackend")
                        .unwrap_or(false)
                } else {
                    false
                }
            }) {
                Some(t.ident.clone())
            } else {
                None
            }
        } else {
            None
        }
    });

    let mut generics = input.generics.clone();
    let b_ident = if let Some(ref b) = backend_generic {
        quote! { #b }
    } else if is_internal {
        generics
            .params
            .push(syn::parse_quote!(__B: #k_crate::prelude::Backend + #k_crate::prelude::VariableBackend));
        quote! { __B }
    } else {
        quote! { #k_crate::prelude::DefaultBackend }
    };

    if let Some(ref b) = backend_generic {
        generics
            .make_where_clause()
            .predicates
            .push(syn::parse_quote!(#b: #k_crate::prelude::VariableBackend));
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let mut param_calls = Vec::new();
    let mut collect_state_calls = Vec::new();
    let mut prepare_state_calls = Vec::new();
    let mut commit_state_calls = Vec::new();
    let mut state_dict_field_types = Vec::new();
    let mut to_device_fields = Vec::new();
    let mut named_layer_calls = Vec::new();
    let mut shape_info_calls = Vec::new();
    let mut train_mode_calls = Vec::new();
    let mut stats_calls = Vec::new();

    if let syn::Data::Struct(ref mut data) = input.data {
        match &mut data.fields {
            syn::Fields::Named(fields) => {
                for field in &mut fields.named {
                    let (ignore, state_name) = match process_field_attributes(&mut field.attrs) {
                        Ok(ig) => ig,
                        Err(e) => return TokenStream::from(e.to_compile_error()),
                    };

                    let fname = &field.ident;
                    let fname_str = fname
                        .as_ref()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "".to_string());
                    let state_component = state_name.as_deref().unwrap_or(&fname_str);

                    if ignore {
                        let is_phantom = match &field.ty {
                            syn::Type::Path(p) => p
                                .path
                                .segments
                                .last()
                                .map(|s| s.ident == "PhantomData")
                                .unwrap_or(false),
                            _ => false,
                        };
                        if is_phantom {
                            to_device_fields.push(quote! { #fname: core::marker::PhantomData });
                        } else {
                            to_device_fields.push(quote! { #fname: self.#fname.clone() });
                        }
                        continue;
                    }

                    state_dict_field_types.push(field.ty.clone());

                    param_calls.push(quote! {
                        #k_crate::prelude::Parameters::named_parameters(
                            &self.#fname, &#format_mac("{}{}", prefix, #fname_str), map);
                    });
                    collect_state_calls.push(quote! {
                        let child_path = path.child(#state_component);
                        #k_crate::prelude::StateDict::collect_state(
                            &self.#fname, &child_path, snapshot)?;
                    });
                    prepare_state_calls.push(quote! {
                        let child_path = path.child(#state_component);
                        #k_crate::prelude::StateDict::prepare_state(
                            &self.#fname, &child_path, snapshot, plan)?;
                    });
                    commit_state_calls.push(quote! {
                        let child_path = path.child(#state_component);
                        #k_crate::prelude::StateDict::commit_state(
                            &mut self.#fname, &child_path, plan)?;
                    });
                    named_layer_calls.push(quote! {
                        let child_prefix = if prefix.is_empty() {
                            #macro_support::String::from(#fname_str)
                        } else {
                            #format_mac("{}.{}", prefix, #fname_str)
                        };
                        children.extend(#k_crate::prelude::NamedLayers::layer_structure(
                            &self.#fname, &child_prefix));
                    });
                    if !no_shape_info {
                        shape_info_calls.push(quote! {
                            if let Some(sh) = #k_crate::prelude::ShapeInfo::shape_info(&self.#fname) {
                                shape_parts.push(#format_mac("{}: {}", #fname_str, sh));
                            }
                        });
                    }
                    stats_calls.push(quote! {
                        total += #k_crate::prelude::ComputeStats::compute_stats(&self.#fname, batch);
                    });
                    if !no_train_mode {
                        train_mode_calls.push(quote! {
                            #k_crate::prelude::TrainMode::set_training(&mut self.#fname, training);
                        });
                    }
                    to_device_fields.push(quote! {
                        #fname: #k_crate::prelude::ToDevice::to_device(self.#fname, arg)?
                    });
                }
            }
            syn::Fields::Unnamed(fields) => {
                for (i, field) in fields.unnamed.iter_mut().enumerate() {
                    let (ignore, state_name) = match process_field_attributes(&mut field.attrs) {
                        Ok(ig) => ig,
                        Err(e) => return TokenStream::from(e.to_compile_error()),
                    };

                    let idx = syn::Index::from(i);

                    let idx_str = i.to_string();
                    let state_component = state_name.as_deref().unwrap_or(&idx_str);

                    if ignore {
                        let is_phantom = match &field.ty {
                            syn::Type::Path(p) => p
                                .path
                                .segments
                                .last()
                                .map(|s| s.ident == "PhantomData")
                                .unwrap_or(false),
                            _ => false,
                        };
                        if is_phantom {
                            to_device_fields.push(quote! { core::marker::PhantomData });
                        } else {
                            to_device_fields.push(quote! { self.#idx.clone() });
                        }
                        continue;
                    }

                    state_dict_field_types.push(field.ty.clone());

                    param_calls.push(quote! {
                        #k_crate::prelude::Parameters::named_parameters(
                            &self.#idx, &#format_mac("{}{}", prefix, #idx_str), map);
                    });
                    collect_state_calls.push(quote! {
                        let child_path = path.child(#state_component);
                        #k_crate::prelude::StateDict::collect_state(
                            &self.#idx, &child_path, snapshot)?;
                    });
                    prepare_state_calls.push(quote! {
                        let child_path = path.child(#state_component);
                        #k_crate::prelude::StateDict::prepare_state(
                            &self.#idx, &child_path, snapshot, plan)?;
                    });
                    commit_state_calls.push(quote! {
                        let child_path = path.child(#state_component);
                        #k_crate::prelude::StateDict::commit_state(
                            &mut self.#idx, &child_path, plan)?;
                    });
                    named_layer_calls.push(quote! {
                        let child_prefix = if prefix.is_empty() {
                            #macro_support::String::from(#idx_str)
                        } else {
                            #format_mac("{}.{}", prefix, #idx_str)
                        };
                        children.extend(#k_crate::prelude::NamedLayers::layer_structure(
                            &self.#idx, &child_prefix));
                    });
                    if !no_shape_info {
                        shape_info_calls.push(quote! {
                            if let Some(sh) = #k_crate::prelude::ShapeInfo::shape_info(&self.#idx) {
                                shape_parts.push(#format_mac("{}: {}", #idx_str, sh));
                            }
                        });
                    }
                    stats_calls.push(quote! {
                        total += #k_crate::prelude::ComputeStats::compute_stats(&self.#idx, batch);
                    });
                    if !no_train_mode {
                        train_mode_calls.push(quote! {
                            #k_crate::prelude::TrainMode::set_training(&mut self.#idx, training);
                        });
                    }

                    to_device_fields.push(quote! {
                        #k_crate::prelude::ToDevice::to_device(self.#idx, arg)?
                    });
                }
            }
            syn::Fields::Unit => {}
        }
    } else {
        return syn::Error::new_spanned(input, "module macro only applies to structs")
            .to_compile_error()
            .into();
    }

    let output_ty_generics = {
        let mut args = Vec::new();
        for param in &input.generics.params {
            match param {
                syn::GenericParam::Type(t) => {
                    let ident = &t.ident;
                    if backend_generic.as_ref() == Some(ident) {
                        args.push(
                            quote! { <#ident as #macro_support::TransferTo<__NewD>>::Output },
                        );
                    } else if t.bounds.iter().any(|b| {
                        if let syn::TypeParamBound::Trait(tb) = b {
                            tb.path
                                .segments
                                .last()
                                .map(|s| s.ident == "Device")
                                .unwrap_or(false)
                        } else {
                            false
                        }
                    }) {
                        // This is a Device-bounded generic, substitute it with __NewD
                        args.push(quote! { __NewD });
                    } else {
                        args.push(quote! { #ident });
                    }
                }
                syn::GenericParam::Lifetime(l) => {
                    let ident = &l.lifetime;
                    args.push(quote! { #ident });
                }
                syn::GenericParam::Const(c) => {
                    let ident = &c.ident;
                    args.push(quote! { #ident });
                }
            }
        }
        if args.is_empty() {
            quote! {}
        } else {
            quote! { <#(#args),*> }
        }
    };

    let to_device_instantiation = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(_) => quote! { Ok(Self::Output { #(#to_device_fields),* }) },
            syn::Fields::Unnamed(_) => quote! { Ok(Self::Output ( #(#to_device_fields),* )) },
            syn::Fields::Unit => quote! { Ok(Self::Output) },
        },
        _ => unreachable!(),
    };

    let stats_impl = if no_stats {
        quote! {}
    } else {
        quote! {
            impl #impl_generics #macro_support::ComputeStats for #name #ty_generics #where_clause {
                /// Sums every field's parameter/MAC contribution for one
                /// forward pass at `batch`. See `#[module(no_stats)]` for
                /// how a leaf layer with its own known formula opts out of
                /// this default instead.
                fn compute_stats(&self, batch: u64) -> #macro_support::LayerStats {
                    let mut total = #macro_support::LayerStats::default();
                    #(#stats_calls)*
                    total
                }
            }
        }
    };

    let mut state_dict_where_clause = where_clause
        .cloned()
        .unwrap_or_else(|| syn::parse_quote!(where));
    for fty in &state_dict_field_types {
        state_dict_where_clause
            .predicates
            .push(syn::parse_quote!(#fty: #k_crate::prelude::StateDict<#b_ident>));
    }

    let parameters_impl = if no_parameters {
        quote! {}
    } else {
        quote! {
            impl #impl_generics #k_crate::prelude::Parameters<#b_ident> for #name #ty_generics #where_clause {
                /// Named parameters.
                fn named_parameters(&self, prefix: &str, map: &mut #macro_support::BTreeMap<#macro_support::String, <#b_ident as #k_crate::prelude::VariableBackend>::RawVar>) {
                    let prefix = if prefix.is_empty() { #macro_support::String::new() } else { #macro_support::format!("{}.", prefix) };
                    #(#param_calls)*
                }
            }
        }
    };

    let state_impl = if no_state {
        quote! {}
    } else {
        quote! {
            impl #impl_generics #k_crate::prelude::StateDict<#b_ident> for #name #ty_generics #state_dict_where_clause {
                fn collect_state(&self, path: &#k_crate::prelude::StatePath, snapshot: &mut #k_crate::prelude::StateSnapshot) -> #k_crate::prelude::Result<()> {
                    #(#collect_state_calls)*
                    Ok(())
                }

                fn prepare_state(&self, path: &#k_crate::prelude::StatePath, snapshot: &#k_crate::prelude::StateSnapshot, plan: &mut #state_load_plan) -> #k_crate::prelude::Result<()> {
                    #(#prepare_state_calls)*
                    Ok(())
                }

                fn commit_state(&mut self, path: &#k_crate::prelude::StatePath, plan: &mut #state_load_plan) -> #k_crate::prelude::Result<()> {
                    #(#commit_state_calls)*
                    Ok(())
                }
            }
        }
    };

    let named_layers_impl = if no_named_layers {
        quote! {}
    } else {
        quote! {
            impl #impl_generics #k_crate::prelude::NamedLayers for #name #ty_generics #where_clause {
                /// Layer structure.
                fn layer_structure(&self, prefix: &str) -> #macro_support::Vec<#k_crate::prelude::LayerNode> {
                    let node_name = if prefix.is_empty() {
                        #macro_support::String::from(stringify!(#name))
                    } else {
                        #macro_support::String::from(prefix)
                    };

                    let mut children: #macro_support::Vec<#k_crate::prelude::LayerNode> = #macro_support::Vec::new();
                    #(#named_layer_calls)*

                    let mut shape_parts: #macro_support::Vec<#macro_support::String> = #macro_support::Vec::new();
                    #(#shape_info_calls)*
                    let shape_info = shape_parts.join(", ");

                    #macro_support::Vec::from([#k_crate::prelude::LayerNode {
                        name: node_name,
                        type_name: #macro_support::String::from(stringify!(#name)),
                        shape_info,
                        children,
                    }])
                }
            }
        }
    };

    let train_mode_impl = if no_train_mode {
        quote! {}
    } else {
        quote! {
            impl #impl_generics #k_crate::prelude::TrainMode for #name #ty_generics #where_clause {
                /// Set training.
                fn set_training(&mut self, training: bool) {
                    #(#train_mode_calls)*
                }
            }
        }
    };

    let to_device_impl = if !no_to_device && (backend_generic.is_some() || is_internal) {
        let mut impl_generics_with_newd = generics.clone();
        impl_generics_with_newd
            .params
            .push(syn::parse_quote!(__NewD: #k_crate::prelude::Device));
        let where_clause = impl_generics_with_newd.make_where_clause();
        where_clause
            .predicates
            .push(syn::parse_quote!(#b_ident: #macro_support::TransferTo<__NewD>));

        let mut dtype_param_found = false;
        for param in &input.generics.params {
            if let syn::GenericParam::Type(t) = param {
                let ident = &t.ident;
                if t.bounds.iter().any(|b| {
                    if let syn::TypeParamBound::Trait(tb) = b {
                        tb.path
                            .segments
                            .last()
                            .map(|s| s.ident == "DType")
                            .unwrap_or(false)
                    } else {
                        false
                    }
                }) || ident == "K"
                {
                    dtype_param_found = true;
                    where_clause
                        .predicates
                        .push(syn::parse_quote!(<#b_ident as #macro_support::TransferTo<__NewD>>::Output: #macro_support::SupportsDType<#ident>));
                }
            }
        }
        if !dtype_param_found {
            where_clause
                .predicates
                .push(syn::parse_quote!(<#b_ident as #macro_support::TransferTo<__NewD>>::Output: #macro_support::SupportsDType<f32>));
        }
        let (impl_g, _, to_device_where_clause) = impl_generics_with_newd.split_for_impl();
        quote! {
            impl #impl_g #k_crate::prelude::ToDevice<#b_ident, __NewD> for #name #ty_generics #to_device_where_clause {
                /// Output.
                type Output = #name #output_ty_generics;
                /// To device.
                fn to_device(self, arg: &__NewD::Arg) -> #k_crate::prelude::Result<Self::Output> {
                    #to_device_instantiation
                }
            }
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #input

        #parameters_impl
        #state_impl
        #named_layers_impl
        #train_mode_impl
        #stats_impl

        #to_device_impl
    };

    TokenStream::from(expanded)
}

#[allow(dead_code)]
pub(crate) fn forward(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
