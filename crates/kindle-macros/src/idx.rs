use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::Parse, parse::ParseStream, parse_macro_input, punctuated::Punctuated, Token, Expr};

struct IdxList {
    items: Punctuated<Expr, Token![,]>,
}

impl Parse for IdxList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(IdxList {
            items: Punctuated::parse_terminated(input)?,
        })
    }
}

pub(crate) fn idx(input: TokenStream) -> TokenStream {
    let items = parse_macro_input!(input as IdxList);
    let mut output = Vec::new();
    
    for item in items.items {
        output.push(quote! { kindle::prelude::IndexSpec::from(#item) });
    }
    
    quote! {
        &[ #(#output),* ]
    }.into()
}
