use proc_macro::TokenStream;
use quote::quote;

/// Wraps ordinary Rust indexing expressions in the tuple form accepted by
/// `Tensor::get`. Negative integer literals are intentionally left as Rust
/// expressions so `IndexSpec` performs the signed conversion.
pub(crate) fn index_expr(input: TokenStream) -> TokenStream {
    let tokens: proc_macro2::TokenStream = input.into();
    quote! { (#tokens) }.into()
}
