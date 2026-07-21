//! # Kindle
//!
//! Kindle is a lightweight, strictly-typed neural network framework for Rust, built with an emphasis on **shape safety** at compile time, while remaining fully compatible with dynamic runtime shapes. It is designed to catch tensor dimension mismatches during compilation using Rust's powerful type system, rather than failing at runtime.
//!
//! ## Key Features
//!
//! * **Compile-time Shape Verification**: Write tensor operations with `Tensor<s![Batch, Channels, Height, Width], Backend>` and let the compiler guarantee that shapes align for operations like `matmul`, `conv2d`, `concat`, etc.
//! * **Backend Agnostic**: Kindle is built on a trait-based backend system, with native CPU, CUDA, and WGPU execution backends shipped out of the box. `candle`/`ndarray`/`burn` wrappers exist behind the `legacy` feature for interop.
//! * **Macro-driven Ergonomics**: Powerful macros like `s![]` for shape definitions, `idx![]` for expressive slicing and reshaping, and `import_model![]` for generating fully typed Rust structs directly from ONNX files.
//! * **Zero-Cost Abstractions**: The static shape information (`typenum`) exists entirely in the type system and evaporates at runtime, introducing zero overhead to the underlying backend operations.
//!
//! ## Quick Start
//!
//! Basic tensor creation and operations:
//!
//! ```rust
//! use kindle::prelude::*;
//!
//! // Create a backend alias for convenience
//! type Backend = kindle_backends::cpu::CpuBackend<f32, Cpu>;
//!
//! // Create a statically shaped tensor: (Batch=2, Channels=3, Height=224, Width=224)
//! let x = Tensor::<s![2, 3, 224, 224], Backend>::zeros(()).unwrap();
//!
//! // Dynamic shapes are also supported using the `Dyn` type:
//! let y = Tensor::<Dyn, Backend>::ones(vec![2, 3, 224, 224]).unwrap();
//! ```
//!
//! ## Neural Network Modules
//!
//! Building and running a model is straightforward using the `#[module]` attribute and the `seq!` macro:
//!
//! ```rust,no_run
//! use kindle::prelude::*;
//!
//! type Backend = kindle_backends::cpu::CpuBackend<f32, Cpu>;
//!
//! #[module]
//! pub struct MLP {
//!     net: Sequential<
//!         Linear<s![768, 256], Backend>,
//!         Sequential<ReLU, Linear<s![256, 10], Backend>>
//!     >,
//! }
//!
//! impl MLP {
//!     pub fn new() -> Result<Self> {
//!         Ok(Self {
//!             net: seq!(
//!                 Linear::<s![768, 256], Backend>::new()?,
//!                 ReLU,
//!                 Linear::<s![256, 10], Backend>::new()?
//!             )
//!         })
//!     }
//!
//!     pub fn forward(&self, x: Tensor<s![2, 768], Backend>) -> Result<Tensor<s![2, 10], Backend>>
//!     {
//!         self.net.forward(x)
//!     }
//! }
//! ```
//!
//! ## ONNX Import
//!
//! Kindle can automatically generate a strongly-typed Rust struct representing an ONNX graph at compile time:
//!
//! ```rust,no_run
//! use kindle::prelude::*;
//!
//! // Reads the ONNX file at compile time, parses the graph, and generates
//! // a struct `ResNet18` with all weights, biases, and a fully typed `forward` method.
//! import_model!("resnet18.onnx", ResNet18);
//!
//! fn main() {
//!     // The generated struct requires you to provide the parameters,
//!     // typically loaded via safetensors or other deserializers.
//!     // let model = ResNet18 { ... };
//! }
//! ```
extern crate alloc;

pub use kindle_backends::*;

pub use kindle_macros::{import_model, module};

/// Core abstraction for `hub` within the Kindle framework.
pub mod hub {
    pub use kindle_data::hub::*;
}

// We define a type alias to restore the default Backend behavior without cyclical dependencies
#[cfg(feature = "cuda")]
/// Core abstraction for `DefaultDevice` within the Kindle framework.
pub type DefaultDevice = crate::prelude::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
/// Core abstraction for `DefaultDevice` within the Kindle framework.
pub type DefaultDevice = crate::prelude::Wgpu;
#[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
/// Core abstraction for `DefaultDevice` within the Kindle framework.
pub type DefaultDevice = kindle_core::prelude::Cpu;

#[cfg(feature = "cpu")]
/// Core abstraction for `DefaultBackend` within the Kindle framework.
pub type DefaultBackend = kindle_backends::cpu::CpuBackend<f32, DefaultDevice>;

// No `DefaultBackend` fallback when `cpu` is disabled: a `()` placeholder
// (the previous approach) doesn't implement `Backend`, so every type alias
// below that defaults its `B` param to it would compile-error deep inside
// trait-bound resolution the moment anyone actually used it, far from the
// real problem ("you disabled `cpu` and didn't pick another backend").
// Instead, every alias below drops the `B = DefaultBackend` default entirely
// in this configuration, same as `Tensor` already did — forcing an explicit,
// immediate "expected 2 generic arguments" error at the actual call site.

#[cfg(feature = "cpu")]
/// Core abstraction for `Tensor` within the Kindle framework.
pub type Tensor<
    S,
    B = DefaultBackend,
    K = <B as kindle_core::prelude::Backend>::FloatElem,
    D = <B as kindle_core::prelude::Backend>::Device,
    G = kindle_core::prelude::Grad,
> = kindle_core::prelude::Tensor<S, B, K, D, G>;

#[cfg(not(feature = "cpu"))]
/// Core abstraction for `Tensor` within the Kindle framework.
pub type Tensor<
    S,
    B, // User must specify backend if Cpu is disabled
    K = <B as kindle_core::prelude::Backend>::FloatElem,
    D = <B as kindle_core::prelude::Backend>::Device,
    G = kindle_core::prelude::Grad,
> = kindle_core::prelude::Tensor<S, B, K, D, G>;

// Neural Network Layer Aliases
//
// Each of these that takes a `B` (backend) param is declared twice, gated on
// the `cpu` feature: with a `= DefaultBackend` default when `cpu` is on
// (the common case), and with NO default when it's off — same reasoning as
// `Tensor` above and `DefaultBackend` itself.
#[cfg(feature = "cpu")]
/// Core abstraction for `Linear` within the Kindle framework.
pub type Linear<S, B = DefaultBackend> = kindle_core::prelude::Linear<S, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `Linear` within the Kindle framework.
pub type Linear<S, B> = kindle_core::prelude::Linear<S, B>;

#[cfg(feature = "cpu")]
/// Core abstraction for `Conv1d` within the Kindle framework.
pub type Conv1d<S, B = DefaultBackend> = kindle_core::prelude::Conv1d<S, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `Conv1d` within the Kindle framework.
pub type Conv1d<S, B> = kindle_core::prelude::Conv1d<S, B>;

#[cfg(feature = "cpu")]
/// Core abstraction for `Conv2d` within the Kindle framework.
pub type Conv2d<S, B = DefaultBackend> = kindle_core::prelude::Conv2d<S, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `Conv2d` within the Kindle framework.
pub type Conv2d<S, B> = kindle_core::prelude::Conv2d<S, B>;

#[cfg(feature = "cpu")]
/// Core abstraction for `BatchNorm2d` within the Kindle framework.
pub type BatchNorm2d<C, B = DefaultBackend> = kindle_core::prelude::BatchNorm2d<C, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `BatchNorm2d` within the Kindle framework.
pub type BatchNorm2d<C, B> = kindle_core::prelude::BatchNorm2d<C, B>;

#[cfg(feature = "cpu")]
/// Core abstraction for `LayerNorm` within the Kindle framework.
pub type LayerNorm<C, B = DefaultBackend> = kindle_core::prelude::LayerNorm<C, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `LayerNorm` within the Kindle framework.
pub type LayerNorm<C, B> = kindle_core::prelude::LayerNorm<C, B>;

/// Core abstraction for `AvgPool2d` within the Kindle framework.
pub type AvgPool2d<K, S, P = typenum::U0, D = typenum::U1> =
    kindle_core::prelude::AvgPool2d<K, S, P, D>;
/// Core abstraction for `MaxPool2d` within the Kindle framework.
pub type MaxPool2d<K, S, P = typenum::U0, D = typenum::U1> =
    kindle_core::prelude::MaxPool2d<K, S, P, D>;
/// Core abstraction for `Sequential` within the Kindle framework.
pub type Sequential<L1, L2> = kindle_core::prelude::Sequential<L1, L2>;

#[cfg(feature = "cpu")]
/// Core abstraction for `Param` within the Kindle framework.
pub type Param<T, B = DefaultBackend> = kindle_core::prelude::Param<T, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `Param` within the Kindle framework.
pub type Param<T, B> = kindle_core::prelude::Param<T, B>;

#[cfg(feature = "cpu")]
/// Core abstraction for `RNNCell` within the Kindle framework.
pub type RNNCell<S, B = DefaultBackend> = kindle_core::prelude::RNNCell<S, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `RNNCell` within the Kindle framework.
pub type RNNCell<S, B> = kindle_core::prelude::RNNCell<S, B>;

#[cfg(feature = "cpu")]
/// Core abstraction for `RNN` within the Kindle framework.
pub type RNN<S, B = DefaultBackend> = kindle_core::prelude::RNN<S, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `RNN` within the Kindle framework.
pub type RNN<S, B> = kindle_core::prelude::RNN<S, B>;

#[cfg(feature = "cpu")]
/// Core abstraction for `Embedding` within the Kindle framework.
pub type Embedding<S, B = DefaultBackend> = kindle_core::prelude::Embedding<S, B>;
#[cfg(not(feature = "cpu"))]
/// Core abstraction for `Embedding` within the Kindle framework.
pub type Embedding<S, B> = kindle_core::prelude::Embedding<S, B>;

/// Core abstraction for `macros` within the Kindle framework.
pub mod macros {
    // `impl_arg_into` deliberately excluded: it's an internal codegen helper
    // invoked once, internally, by `kindle-core` itself
    // (`kindle_macros::impl_arg_into!(7)` in `tensor/arg_into.rs`) — no
    // end-user code calls it, and it has no documented public contract.
    pub use kindle_macros::{idx, s};
}

#[allow(unused_imports)]
/// Core abstraction for `prelude` within the Kindle framework.
pub mod prelude {
    pub use kindle_backends::prelude::*;
    pub use kindle_core::prelude::*;

    // Explicit list instead of `kindle_macros::*`: the wildcard also pulled
    // in `generate_shape_ops` and `impl_arg_into`, internal codegen helpers
    // invoked only by kindle-core's own macro expansions
    // (`kindle_macros::generate_shape_ops!()` in `shapes/shape_ops.rs`,
    // `kindle_macros::impl_arg_into!(7)` in `tensor/arg_into.rs`) — neither
    // has a documented public contract or any end-user call site.
    pub use kindle_macros::{idx, import_model, module, s};

    // We intentionally overshadow kindle_core::Tensor and NN modules with our aliased versions
    #[cfg(feature = "cpu")]
    pub use super::DefaultBackend;
    pub use super::Tensor;
    pub use super::{AvgPool2d, BatchNorm2d, Conv1d, Conv2d, LayerNorm, Linear, MaxPool2d, Param};
    pub use super::{Embedding, RNN, RNNCell};
}

#[cfg(test)]
/// Core abstraction for `tests` within the Kindle framework.
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cpu")]
    /// Core abstraction for `test_tensor_export` within the Kindle framework.
    fn test_tensor_export() {
        // Verify the `Tensor` alias and its `DefaultBackend` are properly
        // exported and usable end-to-end. This used to gate on
        // `#[cfg(feature = "candle")]`, a feature that doesn't exist in this
        // crate's Cargo.toml (confirmed via a "unexpected cfg" warning) — the
        // real assertion below never ran.
        let _t: Tensor<Dyn> = Tensor::zeros(std::vec![2, 2]).unwrap();
    }
}
