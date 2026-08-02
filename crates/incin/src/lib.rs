//! # Incin
//!
//! Incin is a lightweight, strictly-typed neural network framework for Rust, built with an emphasis on **shape safety** at compile time, while remaining fully compatible with dynamic runtime shapes. It is designed to catch tensor dimension mismatches during compilation using Rust's powerful type system, rather than failing at runtime.
//!
//! ## Key Features
//!
//! * **Compile-time Shape Verification**: Write tensor operations with `Tensor<s![Batch, Channels, Height, Width], Backend>` and let the compiler guarantee that shapes align for operations like `matmul`, `conv2d`, `concat`, etc.
//! * **Backend Agnostic**: CPU is enabled by default; native CUDA and WGPU are explicit opt-ins. The third-party Candle adapter is available through the `external-candle` feature under `external::candle`.
//! * **Macro-driven Ergonomics**: Powerful macros like `s![]` for shape definitions, `idx![]` for expressive slicing and reshaping, and `model![]` for generating fully typed Rust structs directly from ONNX files.
//! * **Zero-Cost Abstractions**: The static shape information (`typenum`) exists entirely in the type system and evaporates at runtime, introducing zero overhead to the underlying backend operations.
//!
//! ## Quick Start
//!
//! Basic tensor creation and operations:
//!
//! ```rust
//! use incin::prelude::*;
//!
//! // Create a backend alias for convenience
//! type Backend = IncinBackend<f32, Cpu>;
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
//! Building and running a model is straightforward using the `#[module]` attribute and the
//! `seq!`/`SeqTy!` macros — `SeqTy!` names the same nested `Sequential<...>`
//! type that `seq!` builds a value of, so a layer list only needs to be written once per meaning
//! instead of the field type being hand-nested separately:
//!
//! ```rust,no_run
//! use incin::prelude::*;
//!
//! type Backend = IncinBackend<f32, Cpu>;
//!
//! #[module]
//! pub struct MLP {
//!     net: SeqTy!(
//!         Linear<s![768, 256], Backend>,
//!         ReLU,
//!         Linear<s![256, 10], Backend>
//!     ),
//! }
//!
//! impl MLP {
//!     pub fn new() -> Result<Self> {
//!         Ok(Self {
//!             net: seq!(
//!                 Linear::<s![768, 256], Backend>::build(())?,
//!                 ReLU,
//!                 Linear::<s![256, 10], Backend>::build(())?
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
//! Incin can automatically generate a strongly-typed Rust struct representing an ONNX graph at compile time:
//!
//! ```rust,no_run
//! use incin::prelude::*;
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

pub use incin_core::prelude::{
    Backend, ConstShape, Cpu, DType, DTypeId, DeviceId, Dyn, DynShape, Error, Grad, Gradients,
    Module, NoGrad, PartialDynShape, Result, Shape, StateDict,
};
pub use incin_core::optim::{
    Adam, AdamW, ConstantLR, LRScheduler, LinearLR, Optimizer, SGD,
};
#[cfg(feature = "std")]
pub use incin_core::optim::{CosineAnnealingLR, StepLR};
pub use incin_backends::IncinBackend;

#[cfg(feature = "cuda")]
pub use incin_core::prelude::{Cuda, CudaN};
#[cfg(feature = "wgpu")]
pub use incin_core::prelude::{Wgpu, WgpuN};
#[cfg(feature = "metal")]
pub use incin_core::prelude::{Metal, MetalN};

pub use incin_core::dim;
pub use incin_macros::{import_model, mesh, model, module};

#[cfg(feature = "compiled")]
/// Curated preview types for compiled execution.
pub mod compile {
    pub use incin_core::compile::*;
}

#[cfg(feature = "backend-authoring")]
/// Contracts and extension traits for backend authors.
pub mod backend_authoring {
    pub use incin_core::backend_authoring::*;
}

#[cfg(feature = "autotune")]
/// Preview tuning configuration, inspection, context, and fingerprint types.
pub mod tuning {
    pub use incin_backends::tuning::{
        AlignmentClass, AutotunePolicy, CacheLimits, CacheRecovery, CompilerFingerprint,
        DeviceFingerprint, DTypePolicyId, KernelSignature, PersistentTuningCache, RankClass,
        SelectionSource, TuningContext, TuningEnvironmentFingerprint, TuningExplain,
        TuningProvenance, TuningScope, TuningSelection,
    };
}

#[cfg(feature = "test-utils")]
/// Test utilities and test backend implementations.
pub mod test_utils {
    pub use incin_core::test_utils::*;
}

/// Neural network modules, activation functions, layers, and building blocks.
pub mod nn {
    pub use incin_core::nn::*;
}

/// Optimization algorithms, loss functions, and learning rate schedulers.
pub mod optim {
    pub use incin_core::optim::{
        Adam, AdamW, ConstantLR, Gradients, LRScheduler, LinearLR, Optimizer, SGD,
    };
    #[cfg(feature = "std")]
    pub use incin_core::optim::{CosineAnnealingLR, StepLR};
}

/// Evaluation metrics (Accuracy, Precision, Recall, F1Score, MSE, ConfusionMatrix).
pub mod metrics {
    pub use incin_core::metrics::*;
}

/// Typed meshes, placement rules, and distributed tensor metadata.
#[cfg(feature = "distributed")]
pub mod dist {
    #[cfg(feature = "distributed-nccl")]
    pub use incin_backends::dist::{
        BootstrapRole, NcclBuffer, NcclEvent, NcclTopology, NcclTransport, NcclTransportError,
        TwoRankBootstrapConfig,
    };
    pub use incin_core::dist::*;
}

/// Dataset abstractions and data loading utilities.
pub mod data {
    pub use incin_data::*;
}

/// Data transformations and augmentation pipeline.
pub mod transforms {
    pub use incin_data::transforms::*;
}

/// HuggingFace Hub downloading & pretrained model loading utilities.
pub mod hub {
    pub use incin_data::hub::*;
}

/// Typenum compile-time type-level integers.
pub use incin_core::typenum;

/// The `cargo incin doctor` report: devices, features, caches, and probes.
///
/// In the library rather than in `src/bin/cargo-incin.rs` because an
/// integration test links the library and not the binary, and `UX-014`'s
/// evidence command is `cargo test -p incin --test doctor`.
#[cfg(feature = "std")]
pub mod doctor;

// Enabling an accelerator must never silently change application behavior.
// CPU remains the default whenever it is available; accelerator-only builds
// get the one enabled device family.
#[cfg(feature = "cpu")]
/// Default device used by the standard installation.
pub type DefaultDevice = incin_core::prelude::Cpu;
#[cfg(all(not(feature = "cpu"), feature = "wgpu"))]
/// Default device for a WGPU-only build.
pub type DefaultDevice = crate::prelude::Wgpu;
#[cfg(all(not(feature = "cpu"), not(feature = "wgpu"), feature = "cuda"))]
/// Default device for a CUDA-only build.
pub type DefaultDevice = crate::prelude::Cuda;

#[cfg(feature = "train")]
pub mod plan_report;
#[cfg(feature = "cpu")]
/// Default backend (CPU with f32). Equivalent to `IncinBackend<f32, Cpu>`.
/// The automatic `Trainer` (`UX-001`). Preview tier, so it ships behind the
/// non-default `train` feature.
#[cfg(feature = "train")]
pub mod train;
#[cfg(feature = "std")]
pub mod tune_report;

pub type DefaultBackend = incin_backends::IncinBackend<f32, incin_core::prelude::Cpu>;

// No `DefaultBackend` fallback when `cpu` is disabled: a `()` placeholder
// (the previous approach) doesn't implement `Backend`, so every type alias
// below that defaults its `B` param to it would compile-error deep inside
// trait-bound resolution the moment anyone actually used it, far from the
// real problem ("you disabled `cpu` and didn't pick another backend").
// Instead, every alias below drops the `B = DefaultBackend` default entirely
// in this configuration, same as `Tensor` already did — forcing an explicit,
// immediate "expected 2 generic arguments" error at the actual call site.

#[cfg(feature = "cpu")]
/// Tensor.
pub type Tensor<
    S,
    B = DefaultBackend,
    K = <B as incin_core::prelude::Backend>::FloatElem,
    G = incin_core::prelude::Grad,
    P = incin_core::dist::Local,
> = incin_core::prelude::Tensor<S, B, K, G, P>;

#[cfg(not(feature = "cpu"))]
/// Tensor.
pub type Tensor<
    S,
    B, // User must specify backend if Cpu is disabled
    K = <B as incin_core::prelude::Backend>::FloatElem,
    G = incin_core::prelude::Grad,
    P = incin_core::dist::Local,
> = incin_core::prelude::Tensor<S, B, K, G, P>;

// Neural Network Layer Aliases
//
// Each of these that takes a `B` (backend) param is declared twice, gated on
// the `cpu` feature: with a `= DefaultBackend` default when `cpu` is on
// (the common case), and with NO default when it's off — same reasoning as
// `Tensor` above and `DefaultBackend` itself.
#[cfg(feature = "cpu")]
/// Linear.
pub type Linear<S, B = DefaultBackend> = incin_core::prelude::Linear<S, B>;
#[cfg(not(feature = "cpu"))]
/// Linear.
pub type Linear<S, B> = incin_core::prelude::Linear<S, B>;

#[cfg(feature = "cpu")]
/// Conv1d.
pub type Conv1d<S, B = DefaultBackend> = incin_core::prelude::Conv1d<S, B>;
#[cfg(not(feature = "cpu"))]
/// Conv1d.
pub type Conv1d<S, B> = incin_core::prelude::Conv1d<S, B>;

#[cfg(feature = "cpu")]
/// Conv2d.
pub type Conv2d<S, B = DefaultBackend> = incin_core::prelude::Conv2d<S, B>;
#[cfg(not(feature = "cpu"))]
/// Conv2d.
pub type Conv2d<S, B> = incin_core::prelude::Conv2d<S, B>;

#[cfg(feature = "cpu")]
/// Batch norm2d.
pub type BatchNorm2d<C, B = DefaultBackend> = incin_core::prelude::BatchNorm2d<C, B>;
#[cfg(not(feature = "cpu"))]
/// Batch norm2d.
pub type BatchNorm2d<C, B> = incin_core::prelude::BatchNorm2d<C, B>;

#[cfg(feature = "cpu")]
/// Layer norm.
pub type LayerNorm<C, B = DefaultBackend> = incin_core::prelude::LayerNorm<C, B>;
#[cfg(not(feature = "cpu"))]
/// Layer norm.
pub type LayerNorm<C, B> = incin_core::prelude::LayerNorm<C, B>;

/// Avg pool2d.
pub type AvgPool2d<K, S, P = typenum::U0, D = typenum::U1> =
    incin_core::prelude::AvgPool2d<K, S, P, D>;
/// Max pool2d.
pub type MaxPool2d<K, S, P = typenum::U0, D = typenum::U1> =
    incin_core::prelude::MaxPool2d<K, S, P, D>;
/// Sequential.
pub type Sequential<L1, L2> = incin_core::prelude::Sequential<L1, L2>;

#[cfg(feature = "cpu")]
/// Param.
pub type Param<T, B = DefaultBackend> = incin_core::prelude::Param<T, B>;
#[cfg(not(feature = "cpu"))]
/// Param.
pub type Param<T, B> = incin_core::prelude::Param<T, B>;

#[cfg(feature = "cpu")]
/// Rnncell.
pub type RNNCell<S, B = DefaultBackend> = incin_core::prelude::RNNCell<S, B>;
#[cfg(not(feature = "cpu"))]
/// Rnncell.
pub type RNNCell<S, B> = incin_core::prelude::RNNCell<S, B>;

#[cfg(feature = "cpu")]
/// Rnn.
pub type RNN<S, B = DefaultBackend> = incin_core::prelude::RNN<S, B>;
#[cfg(not(feature = "cpu"))]
/// Rnn.
pub type RNN<S, B> = incin_core::prelude::RNN<S, B>;

#[cfg(feature = "cpu")]
/// Embedding.
pub type Embedding<S, B = DefaultBackend> = incin_core::prelude::Embedding<S, B>;
#[cfg(not(feature = "cpu"))]
/// Embedding.
pub type Embedding<S, B> = incin_core::prelude::Embedding<S, B>;

/// Macros.
pub mod macros {
    // `impl_arg_into` deliberately excluded: it's an internal codegen helper
    // invoked once, internally, by `incin-core` itself
    // (`incin_macros::impl_arg_into!()` in `tensor/arg_into.rs`) — no
    // end-user code calls it, and it has no documented public contract.
    pub use incin_macros::{idx, mesh, s};
}

/// Prelude re-exporting high-frequency user types, macros, NN modules, and optimizers.
pub mod prelude {
    pub use incin_core::prelude::{
        Backend, BTreeMap, ComputeStats, ConstDevice, ConstDType, ConstShape, Cpu, DType, DTypeId,
        Device, DeviceId, DeviceKind, Dim, Dyn, DynShape, Ellipsis, Error, Grad, HeadShape,
        InferDim, LayerStats, ModelStats, Module, NamedDyn, NoGrad, PartialDynShape, Result, SeqTy,
        Shape, Slice, SpanShape, StateDict, String, SupportsDType, TailShape, TransferTo, Vec,
        format,
    };
    pub use incin_core::nn::stats::{AutorefComputeStats, AutorefComputeStatsFallback, sum_stats};
    pub use super::Tensor;

    #[cfg(feature = "cuda")]
    pub use incin_core::prelude::{Cuda, CudaN};
    #[cfg(feature = "wgpu")]
    pub use incin_core::prelude::{Wgpu, WgpuN};
    #[cfg(feature = "metal")]
    pub use incin_core::prelude::{Metal, MetalN};

    pub use incin_backends::IncinBackend;
    #[cfg(feature = "cpu")]
    pub use super::DefaultBackend;
    #[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda"))]
    pub use super::DefaultDevice;

    pub use incin_core::dim;
    pub use incin_core::seq;
    pub use incin_core::typenum;

    pub use incin_macros::{
        axes, einsum, idx, import_model, mesh, model, module, parallel, placement, s,
    };

    pub use super::{
        BatchNorm2d, Conv1d, Conv2d, Embedding, LayerNorm, Linear, Param, RNN, RNNCell,
    };

    pub use incin_core::nn::{
        activation::{GELU, ReLU, Sigmoid, Softmax, Swish, Tanh},
        avg_pool2d::AvgPool2d,
        dropout::Dropout,
        flatten::Flatten,
        init::Init,
        loss::{
            BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, Mean, NoneReduction, Reduction,
        },
        lstm::{LSTM, LSTMCell},
        max_pool2d::MaxPool2d,
        module::{
            LayerNode, NamedLayers, Parameters, Sequential, ToDevice, TrainMode,
        },
        rms_norm::RMSNorm,
    };

    #[cfg(feature = "std")]
    pub use incin_core::prelude::{Format, ModelExt};

    pub use incin_core::optim::{
        Adam, AdamW, ConstantLR, Gradients, LRScheduler, LinearLR, Optimizer, SGD,
    };
    #[cfg(feature = "std")]
    pub use incin_core::optim::{CosineAnnealingLR, StepLR};

    pub use incin_core::metrics::{
        Accuracy, ConfusionMatrix, F1Score, MSE, Metric, Precision, Recall,
    };

    #[cfg(feature = "distributed")]
    pub use incin_core::dist::{Local, Placement, PlacementKind};
}


#[cfg(test)]
/// Tests.
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cpu")]
    /// Runtime dtype creation keeps the dtype tag in a `Dyn` tensor.
    fn test_runtime_dtype_tensor_creation() {
        let t = Tensor::<Dyn, DefaultBackend, Dyn>::ones((std::vec![2, 2], DTypeId::F64)).unwrap();
        assert_eq!(t.dtype(), DTypeId::F64);
    }

    #[test]
    #[cfg(feature = "cpu")]
    /// Test tensor export.
    fn test_tensor_export() {
        let _t = Tensor::<Dyn, DefaultBackend>::zeros(std::vec![2, 2]).unwrap();
    }
}
