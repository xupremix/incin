use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, ReturnType, parse_macro_input};

/// The `#[kindle::module]` macro.
/// Automatically implements the `Module` trait for the annotated struct.
pub(crate) fn module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemStruct);
    let name = &input.ident;

    let mut generics = input.generics.clone();
    generics
        .params
        .push(syn::parse_quote!(__B: kindle::prelude::Backend<kindle::prelude::Dyn>));
    let (impl_generics, _, _) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let mut param_calls = Vec::new();
    let mut where_clause = input
        .generics
        .where_clause
        .clone()
        .unwrap_or_else(|| syn::parse_quote!(where));

    match &input.fields {
        syn::Fields::Named(fields) => {
            for field in &fields.named {
                let fname = &field.ident;
                let fty = &field.ty;
                param_calls.push(quote! {
                    params.extend(kindle::nn::Module::<__B>::parameters(&self.#fname));
                });
                where_clause
                    .predicates
                    .push(syn::parse_quote!(#fty : kindle::nn::Module<__B>));
            }
        }
        syn::Fields::Unnamed(fields) => {
            for (i, field) in fields.unnamed.iter().enumerate() {
                let idx = syn::Index::from(i);
                let fty = &field.ty;
                param_calls.push(quote! {
                    params.extend(kindle::nn::Module::<__B>::parameters(&self.#idx));
                });
                where_clause
                    .predicates
                    .push(syn::parse_quote!(#fty : kindle::nn::Module<__B>));
            }
        }
        syn::Fields::Unit => {}
    }

    let expanded = quote! {
        #input

        impl #impl_generics kindle::nn::Module<__B> for #name #ty_generics #where_clause {
            fn parameters(&self) -> std::vec::Vec<<__B as kindle::prelude::Backend<kindle::prelude::Dyn>>::RawVar> {
                let mut params = std::vec::Vec::new();
                #(#param_calls)*
                params
            }
        }
    };

    TokenStream::from(expanded)
}

/// The `#[kindle::forward]` macro.
/// AST shape tracer implementation for demonstration.
pub(crate) fn forward(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut func = parse_macro_input!(item as ItemFn);

    let fn_name = &func.sig.ident;
    let ret_type_str = match &func.sig.output {
        ReturnType::Type(_, ty) => quote!(#ty).to_string(),
        _ => "()".to_string(),
    };

    println!("\n--------------------------------------------------");
    println!(
        "⚙️  [AST Tracer] Analyzing #[forward] on function `{}`",
        fn_name
    );
    println!("⚙️  [AST Tracer] Target return type: {}", ret_type_str);

    if let Some(last_stmt) = func.block.stmts.last_mut()
        && let syn::Stmt::Expr(expr, _semi) = last_stmt
        && let syn::Expr::Call(call) = expr
        && let syn::Expr::Path(path) = &*call.func
        && path.path.is_ident("Ok")
        && let Some(arg) = call.args.first_mut()
    {
        println!(
            "⚙️  [AST Tracer] Found return expression: `Ok({})`",
            quote!(#arg)
        );

        let rewritten_arg: syn::Expr = syn::parse_quote! {
            (#arg).into_shape()?
        };

        println!(
            "⚙️  [AST Tracer] 🚀 REWRITING TO: `Ok({})` to enforce boundary!",
            quote!(#rewritten_arg)
        );
        *arg = rewritten_arg;
    }

    println!("--------------------------------------------------\n");

    TokenStream::from(quote!(#func))
}
