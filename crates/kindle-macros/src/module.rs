use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, ReturnType, parse_macro_input};

/// The `#[kindle::module]` macro.
/// Automatically implements the `Module` trait for the annotated struct.
pub(crate) fn module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemStruct);
    let name = &input.ident;

    let is_internal = _attr.to_string().contains("internal");
    let _crate_path = if is_internal {
        quote! { crate }
    } else {
    };

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

    let mut generics = input.generics.clone();
    generics
        .params
        .push(syn::parse_quote!(__B: #k_crate::prelude::Backend));
    let (impl_generics, _, _) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let mut param_calls = Vec::new();
    let mut where_clause = input
        .generics
        .where_clause
        .clone()
        .unwrap_or_else(|| syn::parse_quote!(where));

    let mut load_state_calls = Vec::new();
    let mut state_dict_calls = Vec::new();

    match &input.fields {
        syn::Fields::Named(fields) => {
            for field in &fields.named {
                let fname = &field.ident;
                let fty = &field.ty;
                let fname_str = fname.as_ref().unwrap().to_string();
                param_calls.push(quote! {
                    params.extend(#k_crate::nn::Parameters::<__B>::parameters(&self.#fname));
                });
                load_state_calls.push(quote! {
                    #k_crate::nn::StateDict::<__B>::load_state_dict(&mut self.#fname, &#format_mac("{}{}.", prefix, #fname_str), tensors)?;
                });
                state_dict_calls.push(quote! {
                    #k_crate::nn::StateDict::<__B>::state_dict(&self.#fname, &#format_mac("{}{}.", prefix, #fname_str), tensors);
                });
                where_clause
                    .predicates
                    .push(syn::parse_quote!(#fty : #k_crate::nn::Parameters<__B>));
                where_clause
                    .predicates
                    .push(syn::parse_quote!(#fty : #k_crate::nn::StateDict<__B>));
            }
        }
        syn::Fields::Unnamed(fields) => {
            for (i, field) in fields.unnamed.iter().enumerate() {
                let idx = syn::Index::from(i);
                let fty = &field.ty;
                let idx_str = i.to_string();
                param_calls.push(quote! {
                    params.extend(#k_crate::nn::Parameters::<__B>::parameters(&self.#idx));
                });
                load_state_calls.push(quote! {
                    #k_crate::nn::StateDict::<__B>::load_state_dict(&mut self.#idx, &#format_mac("{}{}.", prefix, #idx_str), tensors)?;
                });
                state_dict_calls.push(quote! {
                    #k_crate::nn::StateDict::<__B>::state_dict(&self.#idx, &#format_mac("{}{}.", prefix, #idx_str), tensors);
                });
                where_clause
                    .predicates
                    .push(syn::parse_quote!(#fty : #k_crate::nn::Parameters<__B>));
                where_clause
                    .predicates
                    .push(syn::parse_quote!(#fty : #k_crate::nn::StateDict<__B>));
            }
        }
        syn::Fields::Unit => {}
    }

    let expanded = quote! {
        #input

        impl #impl_generics #k_crate::nn::Parameters<__B> for #name #ty_generics #where_clause {
            fn parameters(&self) -> #vec_ty<<__B as #k_crate::prelude::Backend>::RawVar> {
                let mut params = #vec_ty::new();
                #(#param_calls)*
                params
            }
        }

        impl #impl_generics #k_crate::nn::StateDict<__B> for #name #ty_generics #where_clause {
            fn load_state_dict(
                &mut self,
                prefix: &str,
                tensors: &std::collections::HashMap<String, #k_crate::prelude::Tensor<#k_crate::prelude::Dyn, __B>>,
            ) -> #k_crate::prelude::Result<()> {
                #(#load_state_calls)*
                Ok(())
            }

            fn state_dict(&self, prefix: &str, tensors: &mut std::collections::HashMap<String, #k_crate::prelude::Tensor<#k_crate::prelude::Dyn, __B>>) {
                #(#state_dict_calls)*
            }
        }
    };

    TokenStream::from(expanded)
}

/// The `#[kindle::forward]` macro.
/// Can be applied to an `impl` block to automatically generate `Module<Input>` for it,
/// and applies AST rewriting to enforce shape boundaries.
pub(crate) fn forward(_attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Ok(mut item_impl) = syn::parse::<syn::ItemImpl>(item.clone()) {
        let self_ty = &item_impl.self_ty;
        let generics = &item_impl.generics;
        let (impl_generics, _ty_generics, where_clause) = generics.split_for_impl();

        let mut forward_method = None;
        for item in &mut item_impl.items {
            if let syn::ImplItem::Fn(method) = item {
                if method.sig.ident == "forward" {
                    forward_method = Some(method);
                    break;
                }
            }
        }

        if let Some(method) = forward_method {
            let _fn_name = &method.sig.ident;
            let ret_type = match &method.sig.output {
                ReturnType::Type(_, ty) => ty.clone(),
                _ => {
                    return syn::Error::new_spanned(method, "forward must return a Result")
                        .to_compile_error()
                        .into();
                }
            };

            // Extract inner Ok type from Result<T, E>
            let mut output_type = quote!(());
            let error_type = quote!(kindle_core::prelude::Error);
            if let syn::Type::Path(p) = &*ret_type {
                if let Some(segment) = p.path.segments.last() {
                    if segment.ident == "Result" {
                        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                            if let Some(syn::GenericArgument::Type(t)) = args.args.first() {
                                output_type = quote!(#t);
                            }
                        }
                    }
                }
            }

            // Extract Input type (second argument after &self)
            let mut input_type = quote!(());
            if method.sig.inputs.len() >= 2 {
                if let syn::FnArg::Typed(pat_type) = &method.sig.inputs[1] {
                    let ty = &pat_type.ty;
                    input_type = quote!(#ty);
                }
            }

            // Rewrite AST
            if let Some(last_stmt) = method.block.stmts.last_mut()
                && let syn::Stmt::Expr(expr, _semi) = last_stmt
                && let syn::Expr::Call(call) = expr
                && let syn::Expr::Path(path) = &*call.func
                && path.path.is_ident("Ok")
                && let Some(arg) = call.args.first_mut()
            {
                let rewritten_arg: syn::Expr = syn::parse_quote! {
                    (#arg).into_shape()?
                };
                *arg = rewritten_arg;
            }

            let expanded = quote! {
                #item_impl

                impl #impl_generics kindle::nn::Module<#input_type> for #self_ty #where_clause {
                    type Output = #output_type;
                    type Error = #error_type;

                    #[inline]
                    fn forward(&self, input: #input_type) -> std::result::Result<Self::Output, Self::Error> {
                        self.forward(input)
                    }
                }
            };
            return TokenStream::from(expanded);
        }
    }

    // Fallback for older code if they put it directly on the function
    let mut func = parse_macro_input!(item as ItemFn);
    if let Some(last_stmt) = func.block.stmts.last_mut()
        && let syn::Stmt::Expr(expr, _semi) = last_stmt
        && let syn::Expr::Call(call) = expr
        && let syn::Expr::Path(path) = &*call.func
        && path.path.is_ident("Ok")
        && let Some(arg) = call.args.first_mut()
    {
        let rewritten_arg: syn::Expr = syn::parse_quote! {
            (#arg).into_shape()?
        };
        *arg = rewritten_arg;
    }
    TokenStream::from(quote!(#func))
}
