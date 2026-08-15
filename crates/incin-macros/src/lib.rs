//! # Incin Macros
//!
//! `incin-macros` provides procedural macros that form the ergonomic frontend for the Incin framework.
//! Because Incin relies heavily on compile-time type verification using `typenum`, manipulating raw types can be extremely verbose (e.g., `Prod<UInt<UTerm, B1>, ...>`).
//! These macros hide that complexity, allowing developers to write clean, intuitive syntax while the macros generate the underlying type boundaries.
//!
//! ## Provided Macros
//!
//! * **`s![...]`**: Defines a shape directly from integer literals (e.g., `s![1, 3, 224, 224]`).
//! * **`idx![...]`**: Defines an indexing/slicing operation for tensors (e.g., `idx![1..3, .., 0..2]`).
//! * **`#[module]`**: Derives neural network module traits automatically for structs, similar to `#[derive(Module)]` but specifically tailored for Incin.
//! * **`model!("model.onnx", Name)`**: Compiles an ONNX model file into a fully-typed Rust module structurally matching the ONNX graph.
//! * **`import_model!("model.onnx", Name)`**: Compiles an ONNX model file into a fully-typed Rust module structurally matching the ONNX graph.
#[macro_use]
extern crate alloc;

use proc_macro::TokenStream;

/// Internal helper module for generating ArgInto implementations.
mod arg_into;
mod axis;

/// Internal helper module for tensor index and slicing macro.
mod idx;
/// Helper module for logical device mesh topology macro.
mod mesh;
/// Helper module for deriving neural network module traits.
mod module;
/// Internal helper module for tensor placement macro.
mod placement;

/// Internal helper module for distributed main macro.
mod distributed_main;
/// Internal helper module for parallel block macro.
mod parallel_block;

/// Helper module for parsing ONNX model graphs into Rust structs.
mod onnx;
/// Helper module for importing model weights from safetensors.
mod safetensors;
/// Helper module for static shape macro expansion.
mod shape;
mod shape_value;
/// Helper module for the tensor literal construction macro.
mod tensor;

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
/// use incin::prelude::*;
///
/// // Defines a 4D tensor shape [Batch=2, Channels=3, Height=224, Width=224]
/// type ImageBatch = s![2, 3, 224, 224];
/// let t = Tensor::<ImageBatch>::zeros(()).unwrap();
/// assert_eq!(t.dims().as_ref(), &[2, 3, 224, 224]);
/// ```
///
/// You can also mix named symbolic dimensions (if they implement `Dim`):
/// ```rust
/// use incin::prelude::*;
/// dim!(BatchSize);
///
/// type BatchedFeatures = s![BatchSize, 128];
/// ```
///
/// A semantic tag and extent can also be written together. The extent stays
/// independent from the tag, so named static axes use the same raw typenum
/// representation as anonymous static axes:
/// ```rust
/// # use incin::prelude::*;
/// # dim!(Batch, Features);
/// type FeaturesBatch = s![Batch: 25, Features: 128];
/// ```
///
/// ## Path resolution
///
/// The expansion names `::incin::prelude::…` absolutely, so it resolves
/// against the crate rather than against whatever the caller happens to have
/// in scope — including a module of their own called `incin` (`CI-005`).
///
/// The one form it cannot survive is a *package* rename in the caller's
/// `Cargo.toml` (`incin_x = { package = "incin" }`), because `::incin` then
/// names a crate that is not there. Resolving the real name requires reading
/// the caller's manifest at expansion time, which the macro policy in
/// `PROPOSALS.md` forbids.
#[proc_macro]
pub fn s(input: TokenStream) -> TokenStream {
    shape::shape(input)
}

/// Builds a `Tensor` directly from a literal, inferring shape and (unless
/// told otherwise) dtype from how it's written — the macro analogue of
/// `vec![...]`.
///
/// `tensor![data...; dtype: T, grad: G]` — the `; clause, ...` tail is
/// optional and either clause may appear in either order.
///
/// **Scope.** This macro is the convenience form for literal data on the
/// default CPU backend, and nothing else. It has no `backend:` or `device:`
/// clause: allocating somewhere specific is the allocation target's job, and
/// ordinary Rust arrays carry shape and dtype just as well as a literal
/// does, so the target form is usually the better one anyway:
///
/// ```text
/// let x = Cpu.tensor([[1.0_f32, 2.0], [3.0, 4.0]])?;   // same tensor
/// let y = Wgpu::new(0).tensor([[1.0_f32, 2.0]])?;      // and this one places it
/// ```
///
/// See `incin_backends::target` (feature `target-api`). An earlier revision
/// of this macro did take a `device:` clause and inferred the backend from
/// the *token spelling* of the expression, which could not see through
/// `let d = Wgpu::new(0);`. That heuristic is gone rather than patched.
///
/// The shape comes from nesting depth, exactly like a Rust array literal:
///
/// ```rust
/// use incin::prelude::*;
///
/// let v = tensor![1.0, 2.0, 3.0].unwrap();              // shape [3], f32
/// let m = tensor![[1.0, 2.0], [3.0, 4.0]].unwrap();      // shape [2, 2], f32
/// let t = tensor![[[1, 2], [3, 4]], [[5, 6], [7, 8]]]    // shape [2, 2, 2], i64
///     .unwrap();
///
/// let empty = tensor![].unwrap();                        // shape [0], f32 — like vec![]
/// ```
///
/// A ragged literal (a row whose length disagrees with its siblings, or a
/// row mixing nested arrays with plain values) is a macro-expansion error
/// naming the offending dimension, not a best-effort reshape.
///
/// ## Dtype inference
///
/// `dtype` is picked in this order:
/// 1. An explicit `; dtype: T` clause.
/// 2. A numeric-literal suffix (`2.0f64`, `2u8`), if every leaf that has one
///    agrees.
/// 3. `i64` if every leaf is a bare integer literal (`tensor![1, 2, 3]`,
///    matching PyTorch's own `torch.tensor([1, 2, 3])`); otherwise `f32`.
///
/// Rule 3 is a default, not a proof: a leaf that is not a literal at all
/// (a variable, a call, ...) has no dtype the macro can see at expansion
/// time, so it is passed through unchanged rather than cast. If it turns out
/// not to already be the resolved dtype, that surfaces as `rustc`'s own type
/// mismatch at the `from_slice` call this macro expands to — not a silently
/// narrowed value.
///
/// ## Clauses
///
/// An optional `; clause, clause, ...` tail after the data overrides a
/// default:
///
/// ```rust
/// use incin::prelude::*;
///
/// let precise = tensor![1.0, 2.0; dtype: f64].unwrap();
/// ```
///
/// Clauses are matched by name, not position, so they can be written in any
/// order — `; grad: NoGrad, dtype: f64` and `; dtype: f64, grad: NoGrad` are
/// the same.
///
/// - `dtype: T` — the element type, overriding inference above.
/// - `grad: G` — `Grad` or `NoGrad`, overriding the default `Grad` (matching
///   `Tensor`'s own default). Only those two markers are accepted, since
///   both take no runtime argument; `Dyn` (gradient tracking toggled at
///   runtime) needs a value alongside the type, and `tensor!` has no
///   value-carrying clause — construct
///   directly with `Tensor::<S, B, K, Dyn>::from_slice(&data, (.., flag))`
///   instead:
///
///   ```rust
///   use incin::prelude::*;
///
///   let t = tensor![1.0, 2.0; grad: NoGrad].unwrap();
///   assert!(!t.requires_grad());
///   ```
///
/// ## Return type
///
/// Like every other fallible constructor on `Tensor` (`zeros`, `ones`,
/// `from_slice`, ...), `tensor!` expands to a `Result<Tensor<...>>`
/// expression rather than panicking — use `?` or `.unwrap()` at the call
/// site.
///
/// ## Path resolution
///
/// The expansion names `::incin::prelude::…` absolutely, so it resolves
/// against the crate rather than against whatever the caller happens to have
/// in scope — including a module of their own called `incin` (`CI-005`).
///
/// The one form it cannot survive is a *package* rename in the caller's
/// `Cargo.toml` (`incin_x = { package = "incin" }`), because `::incin` then
/// names a crate that is not there. Resolving the real name requires reading
/// the caller's manifest at expansion time, which the macro policy in
/// `PROPOSALS.md` forbids.
/// Builds a shape *value* for an allocation target, inferring which axes are
/// static from how they are written.
///
/// The value-level counterpart of [`s!`](crate::s). `s!` names a shape
/// *type*; `shape!` produces the argument a target's constructors take, and
/// the shape type comes out of it:
///
/// ```text
/// let w  = gpu.zeros(shape![128, 784])?;    // Tensor<s![128, 784], ..>
/// let x  = gpu.zeros(shape![batch, 784])?;  // Tensor<s![usize, 784], ..>
/// let y  = gpu.zeros(shape![rows, cols])?;  // Tensor<s![usize, usize], ..>
/// ```
///
/// # Which axes are static
///
/// An integer literal is a static axis; anything else is a runtime axis. This
/// is the same split `s!` already makes — `s![usize, 784]` — with the `usize`
/// inferred rather than spelled.
///
/// The inference is *syntactic*, so a named constant reads as an expression
/// and produces a runtime axis:
///
/// ```text
/// const N: usize = 32;
/// shape![N, 784]        // s![usize, 784], not s![32, 784]
/// ```
///
/// That is a weaker shape than was available, never a wrong one: the extent is
/// still 32, it is simply carried at runtime instead of in the type. Where the
/// stronger form matters, name it — `s![32, 784]` — which is
/// what the explicit constructors remain for.
///
/// # What it does not carry
///
/// Only geometry. **Dtype comes from elsewhere**: generated tensors take the
/// target's bound float (`gpu.zeros(..)` is `f32` unless the target was rebound
/// with `with_float`), and data tensors take the element type of the data
/// (`gpu.tensor([0_i64, 1])` is `i64`). Nothing in this macro can change
/// either, which is deliberate — a shape argument that silently decided dtype
/// would be the same mistake as a device argument that silently decided a
/// backend.
///
/// Gradient tracking is likewise not here: it follows the object being built,
/// so target constructors give `NoGrad` and parameters give `Grad`.
///
/// # Grammar
///
/// A comma-separated axis list, and nothing else. `s!`'s repeat, `Head` and
/// `Tail` forms are not supported; write those with `s!` and the explicit
/// constructors.
///
/// ```text
/// shape![]                  // rank 0
/// shape![8]                 // rank 1, static
/// shape![batch]             // rank 1, runtime
/// shape![2, 3, 4]           // rank 3, all static
/// ```
///
/// A negative or fractional dimension is a compile error rather than a
/// confusing `usize` type mismatch further along.
///
/// # Availability
///
/// Expands to `Static`/`Bound`, which live behind the `target-api` feature. It
/// will not resolve without it.
///
/// ## Path resolution
///
/// The expansion names `::incin::prelude::…` absolutely, so it resolves
/// against the crate rather than against whatever the caller happens to have
/// in scope — including a module of their own called `incin` (`CI-005`).
#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream {
    shape_value::shape_value(input)
}

#[proc_macro]
pub fn tensor(input: TokenStream) -> TokenStream {
    tensor::tensor(input)
}

#[proc_macro]
#[doc(hidden)]
pub fn impl_layer_args(input: TokenStream) -> TokenStream {
    arg_into::impl_layer_args(input)
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
/// use incin::prelude::*;
///
/// // Given a tensor `t` of shape [10, 20, 30]
/// // Slice the first dimension 0..5, take all of the second, and 15..30 of the third.
/// let t = Tensor::<s![10, 20, 30]>::zeros(()).unwrap();
/// let view = t.slice_idx::<idx![0..5, .., 15..30]>().unwrap();
/// assert_eq!(view.dims().as_ref(), &[5, 20, 15]);
/// ```
///
/// ## Path resolution
///
/// The expansion names `::incin::prelude::…` absolutely, so it resolves
/// against the crate rather than against whatever the caller happens to have
/// in scope — including a module of their own called `incin` (`CI-005`).
///
/// The one form it cannot survive is a *package* rename in the caller's
/// `Cargo.toml` (`incin_x = { package = "incin" }`), because `::incin` then
/// names a crate that is not there. Resolving the real name requires reading
/// the caller's manifest at expansion time, which the macro policy in
/// `PROPOSALS.md` forbids.
#[proc_macro]
pub fn idx(input: TokenStream) -> TokenStream {
    idx::idx(input)
}

/// Build an arbitrary static or runtime axis selector.
#[proc_macro]
pub fn axis(input: TokenStream) -> TokenStream {
    axis::axis(input)
}

/// A macro to construct typed logical device mesh specifications ergonomically.
///
/// ## Examples
/// ```text
/// type MyMesh = mesh![dp = 2, tp = 4, pp = 1];
/// ```
#[proc_macro]
pub fn mesh(input: TokenStream) -> TokenStream {
    mesh::mesh(input)
}

/// The `#[module]` attribute automatically derives composable module capabilities
/// for custom neural network structs.
///
/// When creating custom network architectures (like a Residual Block or an entire ResNet),
/// you need a way to collect all trainable weights (for the optimizer) and state buffers
/// (for saving/loading checkpoints).
///
/// This macro iterates through every field in your `struct`. If a field implements typed
/// parameter or state visitors (like `Linear`, `Conv2d`, or nested `Sequential` blocks), it recursively
/// aggregates them. It ignores fields that don't store tensor state.
///
/// Generated capabilities can be disabled explicitly for forward-only or specialized modules:
/// `no_stats`, `no_parameters`, `no_state`, `no_named_layers`, `no_shape_info`,
/// `no_train_mode`, and `no_to_device`. Unknown arguments are rejected.
///
/// ## Examples
/// ```rust
/// use incin::prelude::*;
/// use incin::VariableBackend;
///
/// #[module]
/// pub struct MyModel<B: VariableBackend> {
///     fc1: Linear<s![128, 64], B>,
///     fc2: Linear<s![64, 10], B>,
/// }
///
/// // Now `MyModel` automatically supports typed parameter and state visitors!
/// ```
///
/// ## Path resolution
///
/// The expansion names `::incin::prelude::…` absolutely, so it resolves
/// against the crate rather than against whatever the caller happens to have
/// in scope — including a module of their own called `incin` (`CI-005`).
///
/// The one form it cannot survive is a *package* rename in the caller's
/// `Cargo.toml` (`incin_x = { package = "incin" }`), because `::incin` then
/// names a crate that is not there. Resolving the real name requires reading
/// the caller's manifest at expansion time, which the macro policy in
/// `PROPOSALS.md` forbids.
#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    module::module(attr, item)
}

/// Expands a supported `.onnx` graph or imports `.safetensors` metadata.
///
/// ONNX support is intentionally partial and fail-closed. Initializers, unknown
/// rank, control flow, custom domains, attributes, and unsupported nodes produce
/// macro-expansion diagnostics instead of fabricated code or values.
///
/// ## Examples
///
/// The path is read at compile time, so this is shown rather than compiled —
/// there is no feature set in which a doctest has `resnet18.onnx` to open.
///
/// ```text
/// use incin::prelude::*;
///
/// model!("stateless_supported_graph.onnx", MyModel);
/// ```
#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    safetensors::import_model(TokenStream::new(), input)
}

#[proc_macro]
pub fn import_model(input: TokenStream) -> TokenStream {
    safetensors::import_model(TokenStream::new(), input)
}

/// Constructs a compile-time tensor placement.
///
/// Supports `Local`, `Replicated on Mesh`, `Sharded(axis) on Mesh`, `Partial(Op) on Mesh`,
/// and `PipelineStage(index) on Mesh`.
///
/// ## Examples
/// ```text
/// type P1 = placement![Local];
/// type P2 = placement![Replicated on MyMesh];
/// type P3 = placement![Sharded(0) on MyMesh];
/// type P4 = placement![Partial(Sum) on MyMesh];
/// type P5 = placement![PipelineStage(0) on MyMesh];
/// ```
#[proc_macro]
pub fn placement(input: TokenStream) -> TokenStream {
    placement::placement(input)
}

/// Evaluates a computation block in a parallel mesh context.
///
/// Accepts `parallel!(mesh => { ... })` or `parallel!({ ... })`.
///
/// ## Examples
/// ```text
/// let val = parallel!(mesh => { 42 });
/// assert_eq!(val, 42);
/// ```
#[proc_macro]
pub fn parallel(input: TokenStream) -> TokenStream {
    parallel_block::parallel_block(input)
}

/// Attributes a main function to initialize and run distributed launcher boilerplate.
#[proc_macro_attribute]
pub fn distributed_main(attr: TokenStream, item: TokenStream) -> TokenStream {
    distributed_main::distributed_main(attr, item)
}
