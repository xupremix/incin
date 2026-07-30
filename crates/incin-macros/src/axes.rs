use std::collections::HashSet;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Ident, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

struct AxesInput {
    internal: bool,
    axes: Vec<Ident>,
}

impl Parse for AxesInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut internal = false;
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            internal = true;
        }

        if input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "axes! macro requires at least one axis name",
            ));
        }

        let parsed_axes = Punctuated::<Ident, Token![,]>::parse_terminated(input)?;
        if parsed_axes.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "axes! macro requires at least one axis name",
            ));
        }

        let mut seen = HashSet::new();
        let mut axes = Vec::new();

        for axis in parsed_axes {
            let name = axis.to_string();
            if !seen.insert(name.clone()) {
                return Err(syn::Error::new_spanned(
                    &axis,
                    format!("duplicate axis `{name}` in axes!"),
                ));
            }
            axes.push(axis);
        }

        Ok(AxesInput { internal, axes })
    }
}

pub(crate) fn axes(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as AxesInput);
    let path = if parsed.internal {
        quote! { crate:: }
    } else {
        quote! { ::incin_core:: }
    };

    let elems = parsed.axes.iter().map(|axis| {
        quote! { #path shapes::NamedDyn<#axis> }
    });

    let expanded = quote! {
        (#(#elems,)*)
    };

    expanded.into()
}
