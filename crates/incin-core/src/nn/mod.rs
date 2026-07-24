//! Neural network layers, modules, and utilities.
//!
//! This module provides the building blocks for constructing neural networks in Incin.
//! All layers implement the [`Module`] trait, which defines a strongly-typed `forward` method.
//!
//! ## Layers
//!
//! | Layer | Description |
//! |-------|-------------|
//! | [`Linear`] | Fully-connected layer: `y = xWᵀ + b` |
//! | [`Conv2d`] | 2D Convolutional layer |
//! | [`Conv1d`] | 1D Convolutional layer |
//! | [`BatchNorm2d`] | 2D Batch Normalization |
//! | [`LayerNorm`] | Layer Normalization |
//! | [`MaxPool2d`] | 2D Max Pooling |
//! | [`AvgPool2d`] | 2D Average Pooling |
//! | [`Embedding`] | Embedding lookup table |
//!
//! ## Activations
//!
//! | Activation | Description |
//! |------------|-------------|
//! | [`ReLU`] | `max(0, x)` |
//! | [`GELU`] | Gaussian Error Linear Unit |
//! | [`Swish`] | `x * sigmoid(x)` |
//! | [`Sigmoid`] | `1 / (1 + e^{-x})` |
//! | [`Tanh`] | Hyperbolic Tangent |
//! | [`Softmax`] | Normalized exponentials along an axis |
//!
//! ## Loss Functions
//!
//! | Loss | Description |
//! |------|-------------|
//! | [`MSELoss`] | Mean Squared Error |
//! | [`CrossEntropyLoss`] | Softmax + NLL Loss |
//! | [`L1Loss`] | Mean Absolute Error |
//! | [`BCEWithLogitsLoss`] | Binary Cross Entropy with Logits |
//!
//! ## Recurrent Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`RNNCell`] | Single Elman RNN step |
//! | [`RNN`] | Multi-step sequence RNN |
//!
//! ## Parameters & Buffers
//!
//! * [`Param`] — A trainable parameter (gradients are computed and updated by an optimizer).
//! * [`Buffer`] — A non-trainable state buffer (e.g., running statistics in BatchNorm).
/// `activation`.
pub mod activation;
/// `adaptive_avg_pool2d`.
pub mod adaptive_avg_pool2d;
/// `avg_pool2d`.
pub mod avg_pool2d;
/// `batch_norm`.
pub mod batch_norm;
/// `conv1d`.
pub mod conv1d;
/// `conv2d`.
pub mod conv2d;
pub mod dropout;
/// `embedding`.
pub mod embedding;
/// `flatten`.
pub mod flatten;
/// `init`.
pub mod init;
/// `layer_norm`.
pub mod layer_norm;
/// `linear`.
pub mod linear;
/// `loss`.
pub mod loss;
/// `lstm`.
pub mod lstm;
/// `max_pool2d`.
pub mod max_pool2d;
/// `module`.
pub mod module;
/// `optional`.
pub mod optional;
/// `param`.
pub mod param;
/// `rnn`.
pub mod rms_norm;
pub mod rnn;
#[cfg(feature = "std")]
/// `save`.
pub mod save;
/// `stats`.
pub mod stats;

pub use activation::*;
pub use adaptive_avg_pool2d::*;
pub use avg_pool2d::*;
pub use batch_norm::*;
pub use conv1d::*;
pub use conv2d::*;
pub use dropout::*;
pub use embedding::*;
pub use flatten::*;
pub use init::*;
pub use layer_norm::*;
pub use linear::*;
pub use loss::*;
pub use lstm::*;
pub use max_pool2d::*;
pub use module::*;
pub use optional::*;
pub use param::*;
pub use rms_norm::*;
pub use rnn::*;
#[cfg(feature = "std")]
pub use save::*;
pub use stats::*;
