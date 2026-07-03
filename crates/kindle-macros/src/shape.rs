use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

enum Dim {
    Dyn,
    Lit(syn::LitInt),
    Path(syn::Path),
    Sym(syn::Ident),
}

impl Parse for Dim {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![dyn]) {
            input.parse::<Token![dyn]>()?;
            return Ok(Dim::Dyn);
        }
        if input.peek(syn::LitInt) {
            return Ok(Dim::Lit(input.parse::<syn::LitInt>()?));
        }
        
        let fork = input.fork();
        if fork.peek(syn::Ident) && fork.peek2(syn::Ident) {
            let first = fork.parse::<syn::Ident>()?;
            if first == "sym" {
                input.parse::<syn::Ident>()?; // consume 'sym'
                return Ok(Dim::Sym(input.parse::<syn::Ident>()?));
            }
        }
        
        Ok(Dim::Path(input.parse::<syn::Path>()?))
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
    let bit = if n.is_multiple_of(2) {
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
            Dim::Path(p) => output.push(quote! { #p }),
            Dim::Sym(ident) => output.push(quote! { #ident }),
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
