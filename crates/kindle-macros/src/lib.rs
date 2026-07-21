//! # Kindle Macros
//!
//! `kindle-macros` provides procedural macros that form the ergonomic frontend for the Kindle framework.
//! Because Kindle relies heavily on compile-time type verification using `typenum`, manipulating raw types can be extremely verbose (e.g., `Prod<UInt<UTerm, B1>, ...>`).
//! These macros hide that complexity, allowing developers to write clean, intuitive syntax while the macros generate the underlying type boundaries.
//!
//! ## Provided Macros
//!
//! * **`s![...]`**: Defines a shape directly from integer literals (e.g., `s![1, 3, 224, 224]`).
//! * **`idx![...]`**: Defines an indexing/slicing operation for tensors (e.g., `idx![1..3, .., 0..2]`).
//! * **`#[module]`**: Derives neural network module traits automatically for structs, similar to `#[derive(Module)]` but specifically tailored for Kindle.
//! * **`import_model!("model.onnx", Name)`**: Compiles an ONNX model file into a fully-typed Rust module structurally matching the ONNX graph.
#[macro_use]
extern crate alloc;

use proc_macro::TokenStream;

/// Internal helper module for generating ArgInto implementations.
mod arg_into;
/// Internal helper module for tensor index and slicing macro.
mod idx;
/// Helper module for deriving neural network module traits.
mod module;
/// Helper module for parsing ONNX model graphs into Rust structs.
mod onnx;
/// Helper module for importing model weights from safetensors.
mod safetensors;
/// Helper module for static shape macro expansion.
mod shape;
/// Internal helper module for shape operation macros.
mod shape_ops;

/// A macro to construct static Tensor shapes ergonomically in the type system.
///
/// Instead of writing raw type bounds (e.g., `(typenum::U128, typenum::U256)`), this macro generates
/// the underlying binary type trees directly from integer literals. This dramatically improves
/// readability and reduces boilerplate when defining static tensor bounds.
///
/// Under the hood, `s![1, 3, 224, 224]` is expanded by the compiler into:
/// `(typenum::U1, typenum::U3, typenum::U224, typenum::U224)`
///
/// ## Examples
/// ```rust
/// use kindle::prelude::*;
///
/// // Defines a 4D tensor shape [Batch=2, Channels=3, Height=224, Width=224]
/// type ImageBatch = s![2, 3, 224, 224];
/// let t = Tensor::<ImageBatch>::zeros(()).unwrap();
/// assert_eq!(t.dims().as_ref(), &[2, 3, 224, 224]);
/// ```
///
/// You can also mix named symbolic dimensions (if they implement `Dim`):
/// ```rust
/// use kindle::prelude::*;
/// kindle::symbolic_dim!(BatchSize);
///
/// type BatchedFeatures = s![BatchSize, 128];
/// ```
#[proc_macro]
pub fn s(input: TokenStream) -> TokenStream {
    shape::shape(input)
}

#[proc_macro]
/// Internal helper macro for generating tuple `ArgInto` trait implementations.
pub fn impl_arg_into(input: TokenStream) -> TokenStream {
    arg_into::impl_arg_into(input)
}

/// A macro to construct index and slicing arguments for tensor reshaping and subsetting.
///
/// This macro generates the highly complex underlying type trees needed for operations like
/// partial slicing, full slicing (`..`), inferred dimensions (`-1`), and exact indices.
///
/// Unlike standard array slicing, tensors often require slicing across multiple dimensions
/// simultaneously, or broadcasting axes. `idx!` abstracts the creation of the heterogeneous tuple
/// (e.g., `(Slice<U0, U5>, Ellipsis, Slice<U15, U30>)`) that the `.slice()` trait expects.
///
/// ## Syntax Supported
/// * `0..5` -> Translates to a statically bounded slice `Slice<U0, U5>`.
/// * `..` -> Translates to `SliceIdx::Full` (take the whole dimension).
/// * `...` or `..` (when alone) -> Translates to `Ellipsis` (fills missing dimensions).
/// * `-1` -> Translates to `InferDim` (used mainly in reshaping to infer the dimension size).
///
/// ## Examples
/// ```rust
/// use kindle::prelude::*;
///
/// // Given a tensor `t` of shape [10, 20, 30]
/// // Slice the first dimension 0..5, take all of the second, and 15..30 of the third.
/// let t = Tensor::<s![10, 20, 30]>::zeros(()).unwrap();
/// let view = t.slice_idx::<idx![0..5, .., 15..30]>().unwrap();
/// assert_eq!(view.dims().as_ref(), &[5, 20, 15]);
/// ```
#[proc_macro]
pub fn idx(input: TokenStream) -> TokenStream {
    idx::idx(input)
}

/// The `#[module]` attribute automatically derives `StateDict` and `Parameters`
/// traits for custom neural network structs.
///
/// When creating custom network architectures (like a Residual Block or an entire ResNet),
/// you need a way to collect all trainable weights (for the optimizer) and state buffers
/// (for saving/loading checkpoints).
///
/// This macro iterates through every field in your `struct`. If a field implements `Parameters`
/// or `StateDict` (like `Linear`, `Conv2d`, or nested `Sequential` blocks), it recursively
/// aggregates them. It ignores fields that don't store tensor state.
///
/// ## Examples
/// ```rust
/// use kindle::prelude::*;
///
/// #[module]
/// pub struct MyModel<B: Backend> {
///     fc1: Linear<s![128, 64], B>,
///     fc2: Linear<s![64, 10], B>,
/// }
///
/// // Now `MyModel` automatically implements `.parameters()` and `.state_dict()`!
/// ```
#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    module::module(attr, item)
}

#[proc_macro]
/// Internal helper macro for generating backend shape operation glue.
pub fn generate_shape_ops(input: TokenStream) -> TokenStream {
    shape_ops::generate_shape_ops(input)
}

/// Generates a fully-typed Rust struct directly from an `.onnx` or `.safetensors` model file.
///
/// This is one of Kindle's most powerful features. At compile time, it parses the `.onnx` file,
/// determines the static shapes of all parameters (weights, biases, layer norms) and connections,
/// and emits a complete Rust `struct` containing all the layers.
///
/// ## Automatic `Module` Generation
/// For `.onnx` files, this macro goes a step further: it parses the computational graph and
/// automatically generates the `forward` method. The generated `forward` pass contains verified
/// type bounds, meaning that passing an incorrectly shaped tensor into the ONNX model will
/// result in a compile-time error, rather than a runtime panic.
///
/// Supported ONNX operators are mapped natively to Kindle operations (e.g. `Gemm` -> `Linear`,
/// `Conv` -> `Conv2d`, `Relu` -> `ReLU`, etc).
///
/// ## Examples
/// ```rust,ignore
/// use kindle::prelude::*;
///
/// // Generates a struct named `MyResNet` from "resnet18.onnx".
/// // The file must exist relative to the crate root at compile time.
/// import_model!("resnet18.onnx", MyResNet);
///
/// fn main() {
///     // The generated struct requires all parameters to be populated.
///     // You typically initialize this by deserializing a safetensors file into it.
///     // let model = MyResNet { ... };
/// }
/// ```
#[proc_macro]
pub fn import_model(input: TokenStream) -> TokenStream {
    safetensors::import_model(TokenStream::new(), input)
}
