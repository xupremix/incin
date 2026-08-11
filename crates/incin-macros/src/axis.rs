use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Expr, ExprLit, ExprUnary, Lit, LitInt, Token, UnOp, parse::Parse, parse_macro_input,
    punctuated::Punctuated,
};

struct AxisList {
    items: Punctuated<AxisItem, Token![,]>,
}

enum AxisItem {
    Expr(Expr),
    Named(syn::Path),
}

impl Parse for AxisItem {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        if input.peek(syn::Ident) && input.peek2(syn::Ident) {
            let keyword: syn::Ident = input.parse()?;
            if keyword != "named" {
                return Err(syn::Error::new(
                    keyword.span(),
                    "expected `named <AxisTag>`",
                ));
            }
            return Ok(Self::Named(input.parse()?));
        }
        input.parse().map(Self::Expr)
    }
}

impl Parse for AxisList {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        Ok(Self {
            items: Punctuated::parse_terminated(input)?,
        })
    }
}

fn cursor(value: isize) -> proc_macro2::TokenStream {
    let (name, magnitude) = if value < 0 {
        ("FromEnd", (-value) as usize)
    } else {
        ("Next", value as usize)
    };
    let path = quote! { ::incin::prelude:: };
    if name == "FromEnd" {
        let mut ty = quote! { #path Here };
        for _ in 0..magnitude.saturating_sub(1) {
            ty = quote! { #path Next::<#ty> };
        }
        quote! { #path StaticAxis::<#path FromEnd::<#ty>>::DEFAULT }
    } else if magnitude == 0 {
        quote! { #path Here }
    } else {
        let mut ty = quote! { #path Here };
        for _ in 0..magnitude {
            ty = quote! { #path Next::<#ty> };
        }
        quote! { #path StaticAxis::<#ty>::DEFAULT }
    }
}

fn literal_axis(expr: &Expr) -> Option<proc_macro2::TokenStream> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(lit), ..
        }) => lit.base10_parse::<isize>().ok().map(cursor),
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            if let Expr::Lit(ExprLit {
                lit: Lit::Int(LitInt { .. }),
                ..
            }) = &**expr
            {
                expr_to_isize(expr).map(|v| cursor(-v))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn expr_to_isize(expr: &Expr) -> Option<isize> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Int(lit), ..
    }) = expr
    {
        lit.base10_parse().ok()
    } else {
        None
    }
}

pub(crate) fn axis(input: TokenStream) -> TokenStream {
    let list = parse_macro_input!(input as AxisList);
    let values: Vec<_> = list
        .items
        .iter()
        .map(|item| match item {
            AxisItem::Expr(expr) => literal_axis(expr),
            AxisItem::Named(_) => None,
        })
        .collect();
    if values.iter().all(Option::is_some) {
        let values: Vec<_> = values.into_iter().map(Option::unwrap).collect();
        if values.len() == 1 {
            return values[0].clone().into();
        }
        return quote! {
            ::incin::prelude::AxisSelector::new(&[
                #(::incin::prelude::ToAxisIndex::to_axis_index(&#values)),*
            ])
        }
        .into();
    }
    if list.items.len() == 1
        && let Some(AxisItem::Named(path)) = list.items.first()
    {
        return quote! { ::incin::prelude::NamedAxisSelector::<#path>::default() }.into();
    }
    if list
        .items
        .iter()
        .any(|item| matches!(item, AxisItem::Named(_)))
    {
        return quote! {
            compile_error!("axis! named selectors must be resolved individually; mixed named/runtime selector lists are not supported")
        }
        .into();
    }
    let items: Vec<_> = list
        .items
        .into_iter()
        .map(|item| match item {
            AxisItem::Expr(expr) => quote! { #expr },
            AxisItem::Named(path) => quote! {
                ::incin::prelude::NamedAxisSelector::<#path>::default()
            },
        })
        .collect();
    quote! {
        ::incin::prelude::AxisSelector::new(&[
            #( ::incin::prelude::ToAxisIndex::to_axis_index(&#items) ),*
        ])
    }
    .into()
}
