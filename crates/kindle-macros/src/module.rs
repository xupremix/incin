use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

pub(crate) fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;

    let is_internal = attr.to_string().contains("internal");
    let k_crate = if is_internal {
        quote! { crate }
    } else {
        quote! { kindle }
    };

    let format_mac = if is_internal {
        quote! { alloc::format! }
    } else {
        quote! { std::format! }
    };
    let vec_ty = if is_internal {
        quote! { alloc::vec::Vec }
    } else {
        quote! { std::vec::Vec }
    };

    let backend_generic = input.generics.params.iter().find_map(|p| {
        if let syn::GenericParam::Type(t) = p {
            if t.bounds.iter().any(|b| {
                if let syn::TypeParamBound::Trait(tb) = b {
                    tb.path.segments.last().unwrap().ident == "Backend"
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
    } else {
        if is_internal {
            generics.params.push(syn::parse_quote!(__B: #k_crate::prelude::Backend));
            quote! { __B }
        } else {
            quote! { #k_crate::prelude::DefaultBackend }
        }
    };

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let mut param_calls = Vec::new();
    let mut load_state_calls = Vec::new();
    let mut state_dict_calls = Vec::new();
    let mut to_device_fields = Vec::new();

    if let syn::Data::Struct(ref mut data) = input.data {
        match &mut data.fields {
            syn::Fields::Named(fields) => {
                for field in &mut fields.named {
                    let mut ignore = false;
                    let mut error_tokens = None;
                    field.attrs.retain(|a| {
                        if a.path().segments.last().unwrap().ident == "module" {
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
                    let fname_str = fname.as_ref().unwrap().to_string();

                    if ignore {
                        let is_phantom = match &field.ty {
                            syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident == "PhantomData").unwrap_or(false),
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
                            use #k_crate::nn::module::{AutorefParameters, AutorefParametersFallback};
                            params.extend((&&self.#fname).maybe_parameters(core::marker::PhantomData::<#b_ident>));
                        }
                    });
                    load_state_calls.push(quote! {
                        {
                            use #k_crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                            (&mut &mut self.#fname).maybe_load_state_dict(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}.", prefix, #fname_str), tensors)?;
                        }
                    });
                    state_dict_calls.push(quote! {
                        {
                            use #k_crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                            (&&self.#fname).maybe_state_dict(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}.", prefix, #fname_str), tensors);
                        }
                    });
                    to_device_fields.push(quote! {
                        #fname: #k_crate::nn::module::ToDevice::to_device(self.#fname, arg)?
                    });
                }
            }
            syn::Fields::Unnamed(fields) => {
                for (i, field) in fields.unnamed.iter_mut().enumerate() {
                    let mut ignore = false;
                    let mut error_tokens = None;
                    field.attrs.retain(|a| {
                        if a.path().segments.last().unwrap().ident == "module" {
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
                            syn::Type::Path(p) => p.path.segments.last().map(|s| s.ident == "PhantomData").unwrap_or(false),
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
                            use #k_crate::nn::module::{AutorefParameters, AutorefParametersFallback};
                            params.extend((&&self.#idx).maybe_parameters(core::marker::PhantomData::<#b_ident>));
                        }
                    });
                    load_state_calls.push(quote! {
                        {
                            use #k_crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                            (&mut &mut self.#idx).maybe_load_state_dict(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}.", prefix, #idx_str), tensors)?;
                        }
                    });
                    state_dict_calls.push(quote! {
                        {
                            use #k_crate::nn::module::{AutorefStateDict, AutorefStateDictFallback};
                            (&&self.#idx).maybe_state_dict(core::marker::PhantomData::<#b_ident>, &#format_mac("{}{}.", prefix, #idx_str), tensors);
                        }
                    });
                    to_device_fields.push(quote! {
                        #k_crate::nn::module::ToDevice::to_device(self.#idx, arg)?
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
                        args.push(quote! { <#ident as #k_crate::prelude::Backend>::BackendWithDevice<__NewD> });
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

    let to_device_impl = if backend_generic.is_some() || is_internal {
        let mut impl_generics_with_newd = generics.clone();
        impl_generics_with_newd.params.push(syn::parse_quote!(__NewD: #k_crate::prelude::Device));
        let (impl_g, _, _) = impl_generics_with_newd.split_for_impl();
        quote! {
            impl #impl_g #k_crate::nn::module::ToDevice<#b_ident, __NewD> for #name #ty_generics #where_clause {
                type Output = #name #output_ty_generics;
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

        impl #impl_generics #k_crate::nn::Parameters<#b_ident> for #name #ty_generics #where_clause {
            fn parameters(&self) -> #vec_ty<<#b_ident as #k_crate::prelude::Backend>::RawVar> {
                let mut params = #vec_ty::new();
                #(#param_calls)*
                params
            }
        }

        impl #impl_generics #k_crate::nn::StateDict<#b_ident> for #name #ty_generics #where_clause {
            fn load_state_dict(
                &mut self,
                prefix: &str,
                tensors: &std::collections::HashMap<String, #k_crate::prelude::Tensor<#k_crate::prelude::Dyn, #b_ident>>,
            ) -> #k_crate::prelude::Result<()> {
                #(#load_state_calls)*
                Ok(())
            }

            fn state_dict(&self, prefix: &str, tensors: &mut std::collections::HashMap<String, #k_crate::prelude::Tensor<#k_crate::prelude::Dyn, #b_ident>>) {
                #(#state_dict_calls)*
            }
        }

        #to_device_impl
    };

    TokenStream::from(expanded)
}

pub(crate) fn forward(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
