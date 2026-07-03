use proc_macro::TokenStream;

/// The `#[kindle::module]` macro.
/// Currently acts as a pass-through. In future iterations, this will parse the struct fields
/// to build an AST definition for the `forward` macro to utilize for fast boundary enforcement.
pub(crate) fn module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// The `#[kindle::forward]` macro.
/// Currently acts as a pass-through, allowing the Rust compiler's native trait solver
/// to verify the shape math (which is 100% sound and safe).
/// In future iterations for massive networks, this will perform AST-level shape tracing
/// to bypass the Rust trait solver and inject `into_shape()` boundaries, speeding up compilation.
pub(crate) fn forward(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
