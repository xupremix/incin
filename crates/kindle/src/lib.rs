//! # Kindle
//! 
//! Kindle is a lightweight, strictly-typed neural network framework for Rust, built with an emphasis on **shape safety** at compile time, while remaining fully compatible with dynamic runtime shapes. It is designed to catch tensor dimension mismatches during compilation using Rust's powerful type system, rather than failing at runtime.
//!
//! ## Key Features
//!
//! * **Compile-time Shape Verification**: Write tensor operations with `Tensor<s![Batch, Channels, Height, Width], Backend>` and let the compiler guarantee that shapes align for operations like `matmul`, `conv2d`, `concat`, etc.
//! * **Backend Agnostic**: Kindle is built on a trait-based backend system. Out of the box, it supports wrapping frameworks like [Candle](https://github.com/huggingface/candle) and `ndarray`, while making it trivial to plug in your own custom backend.
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
//! // Create a backend alias for convenience (requires `candle` feature)
//! type Backend = kindle_backends::candle::CandleBackend<f32, Cpu>;
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
//! type Backend = kindle_backends::candle::CandleBackend<f32, Cpu>;
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

pub use kindle_backends::*;
pub use kindle_core::*;

pub use kindle_macros::{module, import_model};

pub mod hub {
    pub use kindle_data::hub::*;
}

// We define a type alias to restore the default Backend behavior without cyclical dependencies
#[cfg(feature = "cuda")]
pub type DefaultDevice = crate::prelude::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "metal"))]
pub type DefaultDevice = crate::prelude::Metal;
#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub type DefaultDevice = crate::prelude::Cpu;

#[cfg(feature = "candle")]
pub type DefaultBackend = kindle_backends::candle::CandleBackend<f32, DefaultDevice>;

#[cfg(not(feature = "candle"))]
pub type DefaultBackend = (); // Fallback

#[cfg(feature = "candle")]
pub type Tensor<
    S,
    B = DefaultBackend,
    G = kindle_core::prelude::Grad,
> = kindle_core::prelude::Tensor<S, B, G>;

#[cfg(not(feature = "candle"))]
pub type Tensor<
    S,
    B, // User must specify backend if Candle is disabled
    G = kindle_core::prelude::Grad,
> = kindle_core::prelude::Tensor<S, B, G>;

// Neural Network Layer Aliases
pub type Linear<S, B = DefaultBackend> = kindle_core::nn::Linear<S, B>;
pub type Conv1d<S, B = DefaultBackend> = kindle_core::nn::Conv1d<S, B>;
pub type Conv2d<S, B = DefaultBackend> = kindle_core::nn::Conv2d<S, B>;
pub type BatchNorm2d<C, B = DefaultBackend> = kindle_core::nn::BatchNorm2d<C, B>;
pub type LayerNorm<C, B = DefaultBackend> = kindle_core::nn::LayerNorm<C, B>;
pub type AvgPool2d<K, S, P = typenum::U0, D = typenum::U1> = kindle_core::nn::AvgPool2d<K, S, P, D>;
pub type MaxPool2d<K, S, P = typenum::U0, D = typenum::U1> = kindle_core::nn::MaxPool2d<K, S, P, D>;
pub type Sequential<L1, L2> = kindle_core::nn::Sequential<L1, L2>;
pub type Param<T, B = DefaultBackend> = kindle_core::nn::Param<T, B>;
pub type RNNCell<S, B = DefaultBackend> = kindle_core::nn::rnn::RNNCell<S, B>;
pub type RNN<S, B = DefaultBackend> = kindle_core::nn::rnn::RNN<S, B>;
pub type Embedding<S, B = DefaultBackend> = kindle_core::nn::Embedding<S, B>;

pub mod macros {
    pub use kindle_macros::{idx, impl_arg_into, s};
}

pub mod prelude {
    pub use kindle_backends::prelude::*;
    pub use kindle_core::prelude::*;
    pub use kindle_macros::*;

    // We intentionally overshadow kindle_core::Tensor and NN modules with our aliased versions
    pub use super::Tensor;
    pub use super::{Linear, Conv1d, Conv2d, BatchNorm2d, LayerNorm, AvgPool2d, MaxPool2d, Param, DefaultBackend};
    pub use super::{RNNCell, RNN, Embedding};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_export() {
        // Just verify types are properly exported and accessible
        #[cfg(feature = "candle")]
        {
            // Verify our alias correctly injects CandleBackend
            let _t: Tensor<Dyn> = Tensor::zeros(std::vec![2, 2]).unwrap();
        }
    }
}
