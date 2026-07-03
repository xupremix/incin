use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Token,
};

enum Dim {
    Dyn,
    Lit(syn::LitInt),
}

impl Parse for Dim {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![dyn]) {
            input.parse::<Token![dyn]>()?;
            return Ok(Dim::Dyn);
        }
        Ok(Dim::Lit(input.parse::<syn::LitInt>()?))
    }
}

struct NumberList {
    internal: bool,
    items: Punctuated<Dim, Token![,]>,
}

impl Parse for NumberList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut internal = false;
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            internal = true;
        }
        Ok(NumberList {
            internal,
            items: Punctuated::parse_terminated(input)?,
        })
    }
}

pub(crate) fn shape(input: TokenStream) -> TokenStream {
    let items = parse_macro_input!(input as NumberList);
    let list = items.items;
    let internal = items.internal;
    let mut output = Vec::new();
    let path = if internal {
        quote! {  crate::prelude:: }
    } else {
        quote! {  kindle::prelude:: }
    };
    for elem in &list {
        match elem {
            Dim::Dyn => output.push(quote! { usize }),
            Dim::Lit(lit_int) => output.push(quote! { #path Const<#lit_int> }),
        }
    }
    
    if list.len() <= 7 {
        quote! {
            ( #(#output,)* )
        }
        .into()
    } else {
        let mut expanded = quote! { #path Nil };
        for out in output.into_iter().rev() {
            expanded = quote! { #path Cons<#out, #expanded> };
        }
        expanded.into()
    }
}
