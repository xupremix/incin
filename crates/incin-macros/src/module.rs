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
const STRUCT_ARGUMENTS: &[&str] = &["internal", "no_stats"];

/// Parse the struct-level argument list, rejecting anything not in the
/// vocabulary.
fn parse_arguments(attr: TokenStream) -> syn::Result<(bool, bool)> {
    let attr: proc_macro2::TokenStream = attr.into();
    if attr.is_empty() {
        return Ok((false, false));
    }

    let parser = Punctuated::<Ident, Token![,]>::parse_terminated;
    let arguments = syn::parse::Parser::parse2(parser, attr)?;

    let (mut internal, mut no_stats) = (false, false);
    for argument in arguments {
        match () {
            () if argument == "internal" => internal = true,
            () if argument == "no_stats" => no_stats = true,
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
    Ok((internal, no_stats))
}

pub(crate) fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let (is_internal, no_stats) = match parse_arguments(attr) {
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

    let format_mac = quote! { #k_crate::prelude::format! };
    let _vec_ty = quote! { #k_crate::prelude::Vec };

    let backend_generic = input.generics.params.iter().find_map(|p| {
        if let syn::GenericParam::Type(t) = p {
            if t.bounds.iter().any(|b| {
                if let syn::TypeParamBound::Trait(tb) = b {
                    tb.path
                        .segments
                        .last()
                        .map(|s| s.ident == "Backend")
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
            .push(syn::parse_quote!(__B: #k_crate::prelude::Backend));
        quote! { __B }
    } else {
        quote! { #k_crate::prelude::DefaultBackend }
    };

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();
    let (orig_impl_generics, orig_ty_generics, orig_where_clause) = input.generics.split_for_impl();

    let mut param_calls = Vec::new();
    let mut load_state_calls = Vec::new();
    let mut state_dict_calls = Vec::new();
    let mut to_device_fields = Vec::new();
    let mut named_layer_calls = Vec::new();
    let mut shape_info_calls = Vec::new();
    let mut train_mode_calls = Vec::new();
    let mut stats_calls = Vec::new();

    if let syn::Data::Struct(ref mut data) = input.data {
        match &mut data.fields {
            syn::Fields::Named(fields) => {
                for field in &mut fields.named {
                    let mut ignore = false;
                    let mut error_tokens = None;
                    field.attrs.retain(|a| {
                        if a.path().segments.last().map(|s| s.ident == "module").unwrap_or(false) {
                            match a.parse_args::<syn::Ident>() {
                                Ok(i) if i == "ignore" => {
                                    ignore = true;
                                    false
                                }
                                Ok(i) => {
                                    error_tokens = Some(syn::Error::new_spanned(i, "unknown attribute argument for #[module], expected `ignore`").to_compile_error());
                                    false
                                }
                                Err(e) => {
                                    error_tokens = Some(syn::Error::new_spanned(a, format!("invalid #[module] attribute: {}", e)).to_compile_error());
                                    false
                                }
                            }
                        } else {
                            true
                        }
                    });

                    if let Some(err) = error_tokens {
                        return TokenStream::from(err);
                    }

                    let fname = &field.ident;
                    let fname_str = fname
                        .as_ref()
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "".to_string());

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

                    param_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefParameters, AutorefParametersFallback};
                            (&&self.#fname).maybe_parameters(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}", prefix, #fname_str), map);
                        }
                    });
                    load_state_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefStateDict, AutorefStateDictFallback};
                            (&mut &mut self.#fname).maybe_load_state_dict(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}.", prefix, #fname_str), tensors)?;
                        }
                    });
                    state_dict_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefStateDict, AutorefStateDictFallback};
                            (&&self.#fname).maybe_state_dict(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}.", prefix, #fname_str), tensors);
                        }
                    });
                    named_layer_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefNamedLayers, AutorefNamedLayersFallback};
                            let child_prefix = if prefix.is_empty() {
                                #k_crate::prelude::String::from(#fname_str)
                            } else {
                                #format_mac("{}.{}", prefix, #fname_str)
                            };
                            if let Some(nodes) = (&&self.#fname).maybe_layer_structure(&child_prefix) {
                                children.extend(nodes);
                            }
                        }
                    });
                    shape_info_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefShapeInfo, AutorefShapeInfoFallback};
                            if let Some(sh) = (&&self.#fname).maybe_shape_info() {
                                shape_parts.push(#format_mac("{}: {}", #fname_str, sh));
                            }
                        }
                    });
                    stats_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefComputeStats, AutorefComputeStatsFallback};
                            if let Some(s) = (&&self.#fname).maybe_compute_stats(batch) {
                                total += s;
                            }
                        }
                    });
                    train_mode_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefTrainMode, AutorefTrainModeFallback};
                            (&mut &mut self.#fname).maybe_set_training(training);
                        }
                    });
                    to_device_fields.push(quote! {
                        #fname: #k_crate::prelude::ToDevice::to_device(self.#fname, arg)?
                    });
                }
            }
            syn::Fields::Unnamed(fields) => {
                for (i, field) in fields.unnamed.iter_mut().enumerate() {
                    let mut ignore = false;
                    let mut error_tokens = None;
                    field.attrs.retain(|a| {
                        if a.path().segments.last().map(|s| s.ident == "module").unwrap_or(false) {
                            match a.parse_args::<syn::Ident>() {
                                Ok(i) if i == "ignore" => {
                                    ignore = true;
                                    false
                                }
                                Ok(i) => {
                                    error_tokens = Some(syn::Error::new_spanned(i, "unknown attribute argument for #[module], expected `ignore`").to_compile_error());
                                    false
                                }
                                Err(e) => {
                                    error_tokens = Some(syn::Error::new_spanned(a, format!("invalid #[module] attribute: {}", e)).to_compile_error());
                                    false
                                }
                            }
                        } else {
                            true
                        }
                    });

                    if let Some(err) = error_tokens {
                        return TokenStream::from(err);
                    }

                    let idx = syn::Index::from(i);
                    let idx_str = i.to_string();

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

                    param_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefParameters, AutorefParametersFallback};
                            (&&self.#idx).maybe_parameters(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}", prefix, #idx_str), map);
                        }
                    });
                    load_state_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefStateDict, AutorefStateDictFallback};
                            (&mut &mut self.#idx).maybe_load_state_dict(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}.", prefix, #idx_str), tensors)?;
                        }
                    });
                    state_dict_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefStateDict, AutorefStateDictFallback};
                            (&&self.#idx).maybe_state_dict(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}.", prefix, #idx_str), tensors);
                        }
                    });
                    named_layer_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefNamedLayers, AutorefNamedLayersFallback};
                            let child_prefix = if prefix.is_empty() {
                                #k_crate::prelude::String::from(#idx_str)
                            } else {
                                #format_mac("{}.{}", prefix, #idx_str)
                            };
                            if let Some(nodes) = (&&self.#idx).maybe_layer_structure(&child_prefix) {
                                children.extend(nodes);
                            }
                        }
                    });
                    shape_info_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefShapeInfo, AutorefShapeInfoFallback};
                            if let Some(sh) = (&&self.#idx).maybe_shape_info() {
                                shape_parts.push(#format_mac("{}: {}", #idx_str, sh));
                            }
                        }
                    });
                    stats_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefComputeStats, AutorefComputeStatsFallback};
                            if let Some(s) = (&&self.#idx).maybe_compute_stats(batch) {
                                total += s;
                            }
                        }
                    });
                    train_mode_calls.push(quote! {
                        {
                            use #k_crate::prelude::{AutorefTrainMode, AutorefTrainModeFallback};
                            (&mut &mut self.#idx).maybe_set_training(training);
                        }
                    });
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
                            quote! { <#ident as #k_crate::prelude::TransferTo<__NewD>>::Output },
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
            impl #orig_impl_generics #k_crate::prelude::ComputeStats for #name #orig_ty_generics #orig_where_clause {
                /// Sums every field's parameter/MAC contribution for one
                /// forward pass at `batch`. See `#[module(no_stats)]` for
                /// how a leaf layer with its own known formula opts out of
                /// this default instead.
                fn compute_stats(&self, batch: u64) -> #k_crate::prelude::LayerStats {
                    let mut total = #k_crate::prelude::LayerStats::default();
                    #(#stats_calls)*
                    total
                }
            }
        }
    };

    let to_device_impl = if backend_generic.is_some() || is_internal {
        let mut impl_generics_with_newd = generics.clone();
        impl_generics_with_newd
            .params
            .push(syn::parse_quote!(__NewD: #k_crate::prelude::Device));
        impl_generics_with_newd
            .make_where_clause()
            .predicates
            .push(syn::parse_quote!(#b_ident: #k_crate::prelude::TransferTo<__NewD>));
        impl_generics_with_newd.make_where_clause().predicates.push(
            syn::parse_quote!(<#b_ident as #k_crate::prelude::TransferTo<__NewD>>::Output: #k_crate::prelude::SupportsDType<<#b_ident as #k_crate::prelude::Backend>::FloatElem>),
        );
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

        impl #impl_generics #k_crate::prelude::Parameters<#b_ident> for #name #ty_generics #where_clause {
            /// Named parameters.
            fn named_parameters(&self, prefix: &str, map: &mut #k_crate::prelude::BTreeMap<#k_crate::prelude::String, <#b_ident as #k_crate::prelude::Backend>::RawVar>) {
                let prefix = if prefix.is_empty() { #k_crate::prelude::String::new() } else { #k_crate::prelude::format!("{}.", prefix) };
                #(#param_calls)*
            }
        }

        impl #impl_generics #k_crate::prelude::StateDict<#b_ident> for #name #ty_generics #where_clause {
            /// Load state dict.
            fn load_state_dict(
                &mut self,
                prefix: &str,
                tensors: &#k_crate::prelude::BTreeMap<#k_crate::prelude::String, #k_crate::prelude::Tensor<#k_crate::prelude::Dyn, #b_ident>>,
            ) -> #k_crate::prelude::Result<()> {
                #(#load_state_calls)*
                Ok(())
            }

            /// State dict.
            fn state_dict(&self, prefix: &str, tensors: &mut #k_crate::prelude::BTreeMap<#k_crate::prelude::String, #k_crate::prelude::Tensor<#k_crate::prelude::Dyn, #b_ident>>) {
                #(#state_dict_calls)*
            }
        }

        impl #orig_impl_generics #k_crate::prelude::NamedLayers for #name #orig_ty_generics #orig_where_clause {
            /// Layer structure.
            fn layer_structure(&self, prefix: &str) -> #k_crate::prelude::Vec<#k_crate::prelude::LayerNode> {
                let node_name = if prefix.is_empty() {
                    #k_crate::prelude::String::from(stringify!(#name))
                } else {
                    #k_crate::prelude::String::from(prefix)
                };

                let mut children: #k_crate::prelude::Vec<#k_crate::prelude::LayerNode> = #k_crate::prelude::Vec::new();
                #(#named_layer_calls)*

                let mut shape_parts: #k_crate::prelude::Vec<#k_crate::prelude::String> = #k_crate::prelude::Vec::new();
                #(#shape_info_calls)*
                let shape_info = shape_parts.join(", ");

                #k_crate::prelude::Vec::from([#k_crate::prelude::LayerNode {
                    name: node_name,
                    type_name: #k_crate::prelude::String::from(stringify!(#name)),
                    shape_info,
                    children,
                }])
            }
        }

        impl #orig_impl_generics #k_crate::prelude::TrainMode for #name #orig_ty_generics #orig_where_clause {
            /// Set training.
            fn set_training(&mut self, training: bool) {
                #(#train_mode_calls)*
            }
        }

        #stats_impl

        #to_device_impl
    };

    TokenStream::from(expanded)
}

#[allow(dead_code)]
pub(crate) fn forward(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
