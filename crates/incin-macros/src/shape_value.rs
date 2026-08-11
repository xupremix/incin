use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

enum Axis {
    StaticLit(syn::LitInt),
    ConstPath(syn::Path),
    Runtime(syn::Expr),
}

struct ShapeInput {
    axes: Vec<Axis>,
}

fn parse_axis(input: ParseStream) -> syn::Result<Axis> {
    if input.peek(Token![-]) {
        let minus = input.parse::<Token![-]>()?;
        if input.peek(syn::LitInt) {
            let int = input.parse::<syn::LitInt>()?;
            return Err(syn::Error::new_spanned(
                quote!(#minus #int),
                "a shape! dimension cannot be negative",
            ));
        } else {
            return Err(syn::Error::new_spanned(
                minus,
                "a shape! dimension cannot be negative",
            ));
        }
    }
    if input.peek(Token![const]) {
        let const_token = input.parse::<Token![const]>()?;
        if input.peek(syn::token::Brace) || input.peek(syn::token::Paren) {
            return Err(syn::Error::new_spanned(
                const_token,
                "dimension expressions like `const { ... }` or `const (...)` are not supported in shape!",
            ));
        }
        let path: syn::Path = input.parse()?;
        if input.peek(Token![*])
            || input.peek(Token![+])
            || input.peek(Token![-])
            || input.peek(Token![/])
        {
            return Err(syn::Error::new(
                input.span(),
                "arithmetic expressions after `const` are not supported in shape!",
            ));
        }
        Ok(Axis::ConstPath(path))
    } else if input.peek(syn::LitInt) {
        let int: syn::LitInt = input.parse()?;
        if !int.suffix().is_empty() && int.suffix() != "usize" {
            return Err(syn::Error::new_spanned(
                &int,
                format!(
                    "a shape! dimension may only be suffixed `usize`, not `{}`",
                    int.suffix()
                ),
            ));
        }
        Ok(Axis::StaticLit(int))
    } else if input.peek(syn::LitFloat) {
        Err(syn::Error::new(
            input.span(),
            "a shape! dimension must be a whole number, not a float literal",
        ))
    } else {
        let expr: syn::Expr = input.parse()?;
        if let syn::Expr::Unary(unary) = &expr
            && matches!(unary.op, syn::UnOp::Neg(_))
        {
            return Err(syn::Error::new_spanned(
                &expr,
                "a shape! dimension cannot be negative",
            ));
        }
        Ok(Axis::Runtime(expr))
    }
}

impl Parse for ShapeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut axes = Vec::new();
        while !input.is_empty() {
            let axis = parse_axis(input)?;
            axes.push(axis);
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            } else if !input.is_empty() {
                return Err(input.error("unexpected tokens in shape!"));
            }
        }
        Ok(ShapeInput { axes })
    }
}

pub(crate) fn shape_value(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as ShapeInput);
    let is_fully_static = parsed
        .axes
        .iter()
        .all(|axis| !matches!(axis, Axis::Runtime(_)));

    let path = quote! { ::incin::prelude:: };

    let type_axes: Vec<_> = parsed
        .axes
        .iter()
        .map(|axis| match axis {
            Axis::StaticLit(int) => {
                let val: usize = int.base10_parse().unwrap_or(0);
                // Keep literal extents on the same logarithmic recursive
                // typenum path as `s!`; never route them through ConstDim or
                // a finite convenience-alias catalogue.
                crate::shape::lit_to_typenum(val, &path)
            }
            Axis::ConstPath(p) => quote! { #path ConstDim<{ #p }> },
            Axis::Runtime(_) => quote! { usize },
        })
        .collect();

    let init_args: Vec<_> = parsed
        .axes
        .iter()
        .map(|axis| match axis {
            Axis::StaticLit(_) => quote! { () },
            Axis::ConstPath(_) => quote! { () },
            Axis::Runtime(expr) => quote! { #expr },
        })
        .collect();

    let mut shape_ty = quote! { #path Nil };
    for d in type_axes.iter().rev() {
        shape_ty = quote! { #path DimCons<#d, #shape_ty> };
    }

    let mut tuple_arg = quote! { () };
    for arg in init_args.iter().rev() {
        tuple_arg = quote! { (#arg, #tuple_arg) };
    }

    if is_fully_static {
        quote! {
            #path ShapeValue::<#shape_ty>::try_new(
                <#shape_ty as #path Shape>::resolve(#tuple_arg)
                    .expect("shape! generated an invalid static shape")
            ).expect("shape! generated an invalid static shape")
        }
        .into()
    } else {
        quote! {
            #path ShapeArgs::<#shape_ty>::new(#tuple_arg)
        }
        .into()
    }
}
