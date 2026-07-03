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

fn lit_to_typenum(n: usize, path: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    if n == 0 {
        return quote! { #path typenum::UTerm };
    }
    let bit = if n % 2 == 0 {
        quote! { #path typenum::B0 }
    } else {
        quote! { #path typenum::B1 }
    };
    let rest = lit_to_typenum(n / 2, path);
    quote! { #path typenum::UInt<#rest, #bit> }
}

pub(crate) fn shape(input: TokenStream) -> TokenStream {
    let items = parse_macro_input!(input as NumberList);
    let list = items.items;
    let internal = items.internal;
    let mut output = Vec::new();
    let path = if internal {
        quote! { crate::prelude:: }
    } else {
        quote! { kindle::prelude:: }
    };
    for elem in &list {
        match elem {
            Dim::Dyn => output.push(quote! { usize }),
            Dim::Lit(lit_int) => {
                let val: usize = lit_int.base10_parse().unwrap();
                output.push(lit_to_typenum(val, &path));
            }
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
