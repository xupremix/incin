use proc_macro::TokenStream;

mod arg_into;
mod idx;
mod module;
mod onnx;
mod safetensors;
mod shape;
mod shape_ops;

#[proc_macro]
pub fn s(input: TokenStream) -> TokenStream {
    shape::shape(input)
}

#[proc_macro]
pub fn impl_arg_into(input: TokenStream) -> TokenStream {
    arg_into::impl_arg_into(input)
}

#[proc_macro]
pub fn idx(input: TokenStream) -> TokenStream {
    idx::idx(input)
}

#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    module::module(attr, item)
}

#[proc_macro_attribute]
pub fn forward(attr: TokenStream, item: TokenStream) -> TokenStream {
    module::forward(attr, item)
}

#[proc_macro]
pub fn generate_shape_ops(input: TokenStream) -> TokenStream {
    shape_ops::generate_shape_ops(input)
}

#[proc_macro]
pub fn import_model(item: TokenStream) -> TokenStream {
    safetensors::import_model(TokenStream::new(), item)
}
