use proc_macro::TokenStream;

mod arg_into;
mod idx;
mod module;
mod onnx;
mod safetensors;
mod shape;
mod shape_ops;

/// A macro to construct static Tensor shapes ergonomically (e.g. `s!(128, 256)`).
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

/// The `#[kindle::module]` macro automatically derives `StateDict` and `Parameters`
/// for neural network modules, extracting states from child fields.
#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    module::module(attr, item)
}

/// The `#[kindle::forward]` macro automatically maps inputs/outputs to enforce
/// strict static shapes during tensor propagation.
#[proc_macro_attribute]
pub fn forward(attr: TokenStream, item: TokenStream) -> TokenStream {
    module::forward(attr, item)
}

#[proc_macro]
pub fn generate_shape_ops(input: TokenStream) -> TokenStream {
    shape_ops::generate_shape_ops(input)
}

/// The `import_model!` macro parses `.safetensors` and `.onnx` files at compile-time and
/// dynamically generates strongly-typed module structs that exactly match the
/// internal architecture of the external model.
///
/// For `.onnx`, it also automatically generates the `forward` pass by parsing the computational graph!
#[proc_macro]
pub fn import_model(item: TokenStream) -> TokenStream {
    safetensors::import_model(TokenStream::new(), item)
}
