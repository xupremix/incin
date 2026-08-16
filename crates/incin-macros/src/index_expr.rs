use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{Expr, Token, punctuated::Punctuated};

/// Converts an arbitrary comma-separated index list into a vector of
/// `IndexSpec` values. Negative integer literals are intentionally left as
/// Rust expressions so `IndexSpec` performs the signed conversion.
pub(crate) fn index_expr(input: TokenStream) -> TokenStream {
    let parser = Punctuated::<Expr, Token![,]>::parse_terminated;
    let args = match parser.parse(input) {
        Ok(args) => args,
        Err(error) => return error.into_compile_error().into(),
    };
    let specs = args
        .iter()
        .map(|expr| quote! { ::incin::__macro_support::IndexSpec::from(#expr) });
    quote! { ::incin::__macro_support::Vec::from([#(#specs),*]) }.into()
}
