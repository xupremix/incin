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
//! * [`Param`] - A trainable parameter (gradients are computed and updated by an optimizer).
//! * [`Buffer`] - A non-trainable state buffer (e.g., running statistics in BatchNorm).
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
/// Randomly zeroes activations during training.
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
/// Elman recurrent layer family.
pub mod rnn;
/// Sharded Safetensors checkpoint indexes (`model.safetensors.index.json`).
#[cfg(feature = "std")]
pub mod safetensors_index;
#[cfg(feature = "std")]
/// `save`.
pub mod save;
/// Backend-neutral model state artifacts.
pub mod state;
/// `stats`.
pub mod stats;

pub use activation::{ELU, GELU, Mish, ReLU, Sigmoid, Softmax, Swish, Tanh};
pub use adaptive_avg_pool2d::AdaptiveAvgPool2d;
pub use avg_pool2d::AvgPool2d;
pub use batch_norm::{BatchNorm2d, BatchNorm2dBuilder, BatchNormShape, batch_norm2d};
pub use conv1d::{Conv1d, Conv1dBuilder, Conv1dShape, conv1d};
pub use conv2d::{Conv2d, Conv2dBuilder, Conv2dShape, conv2d};
pub use dropout::Dropout;
pub use embedding::{Embedding, EmbeddingBuilder, EmbeddingShape, embedding};
pub use flatten::{Flatten, FlattenAxes, StructuralFlatten};
pub use init::{
    Fan, Init, InitContext, InitPlan, ParameterRole, constant, kaiming_normal,
    kaiming_normal_with_a, kaiming_uniform, kaiming_uniform_with_a, normal, ones, rand, randn,
    uniform, xavier_normal, xavier_uniform, zeros,
};
pub use layer_norm::{LayerNorm, LayerNormBuilder, LayerNormShape, layer_norm};
pub use linear::{Linear, LinearBuilder, LinearShape, linear};
#[cfg(feature = "distributed")]
pub use linear::{TwoWayColumnLinearShape, TwoWayRowLinearShape};
pub use loss::{BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss};
pub use lstm::{LSTM, LSTMBuilder, LSTMCell, LSTMCellBuilder, LstmShape, lstm, lstm_cell};
pub use max_pool2d::MaxPool2d;
pub use module::{
    LayerNode, Module, NamedLayers, ParameterVisitor, Sequential, ShapeInfo, TrainMode,
    VisitParameters, assign_sequential_names, clean_type_name, format_layer_summary,
    format_layer_summary_with_stats, update_node_name_prefix,
};
pub use optional::{False, OptionalField, True};
pub use param::{Buffer, Frozen, Param, TrainState, Trainable};
pub use rms_norm::{RMSNorm, RMSNormBuilder, RMSNormShape, rms_norm};
pub use rnn::{RNN, RNNBuilder, RNNCell, RNNCellBuilder, RnnShape, rnn, rnn_cell};
#[cfg(feature = "std")]
pub use safetensors_index::SafetensorsIndex;
#[cfg(feature = "std")]
pub use save::{
    CheckpointDType, GlobalCheckpointManifest, TensorCheckpointMeta, load_resharded_checkpoint,
    load_safetensors, load_safetensors_snapshot, save_checkpoint, save_checkpoint_manifest,
    save_safetensors, slice_bytes_for_rank,
};
pub use state::{
    StateMutVisitor, StatePath, StateRole, StateSnapshot, StateSnapshotVisitor, StateValue,
    StateVisitor, VisitState, VisitStateMut, collect_state, load_state,
};
pub use stats::{ComputeStats, LayerStats, ModelStats, sum_stats};
