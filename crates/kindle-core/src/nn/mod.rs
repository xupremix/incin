//! Neural network layers, modules, and utilities.
//!
//! This module provides the building blocks for constructing neural networks in Kindle.
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
/// Core abstraction for `activation` within the Kindle framework..
pub mod activation;
/// Core abstraction for `adaptive_avg_pool2d` within the Kindle framework..
pub mod adaptive_avg_pool2d;
/// Core abstraction for `avg_pool2d` within the Kindle framework..
pub mod avg_pool2d;
/// Core abstraction for `batch_norm` within the Kindle framework..
pub mod batch_norm;
/// Core abstraction for `conv1d` within the Kindle framework..
pub mod conv1d;
/// Core abstraction for `conv2d` within the Kindle framework..
pub mod conv2d;
pub mod dropout;
/// Core abstraction for `embedding` within the Kindle framework..
pub mod embedding;
/// Core abstraction for `flatten` within the Kindle framework..
pub mod flatten;
/// Core abstraction for `init` within the Kindle framework..
pub mod init;
/// Core abstraction for `layer_norm` within the Kindle framework..
pub mod layer_norm;
/// Core abstraction for `linear` within the Kindle framework..
pub mod linear;
/// Core abstraction for `loss` within the Kindle framework..
pub mod loss;
/// Core abstraction for `lstm` within the Kindle framework..
pub mod lstm;
/// Core abstraction for `max_pool2d` within the Kindle framework..
pub mod max_pool2d;
/// Core abstraction for `module` within the Kindle framework..
pub mod module;
/// Core abstraction for `optional` within the Kindle framework..
pub mod optional;
/// Core abstraction for `param` within the Kindle framework..
pub mod param;
/// Core abstraction for `rnn` within the Kindle framework..
pub mod rms_norm;
pub mod rnn;
#[cfg(feature = "std")]
/// Core abstraction for `save` within the Kindle framework..
pub mod save;

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
