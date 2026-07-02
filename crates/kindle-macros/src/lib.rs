use proc_macro::TokenStream;

mod arg_into;
mod shape;

#[proc_macro]
pub fn s(input: TokenStream) -> TokenStream {
    shape::shape(input)
}

#[proc_macro]
pub fn impl_arg_into(input: TokenStream) -> TokenStream {
    arg_into::impl_arg_into(input)
}
