//! Core tensor operations, static shape checking, and autograd for Incin.
#![allow(dead_code)]
#![allow(unused_imports)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    feature = "nightly",
    feature(generic_const_exprs),
    allow(incomplete_features)
)]

#[macro_use]
pub(crate) extern crate alloc;

pub(crate) mod err;

pub mod compiled;
pub mod dist;
pub mod distributions;
pub mod exec;
pub mod graph;

#[cfg(feature = "std")]
pub mod io;
pub mod metrics;
pub mod nn;
#[cfg(feature = "std")]
pub(crate) mod onnx_exporter;
#[cfg(feature = "std")]
pub(crate) mod onnx_pb;
pub mod optim;
pub(crate) mod serialize;
pub mod shapes;

pub(crate) mod tensor;
pub use typenum;

/// Loss functions and reduction definitions.
pub mod loss {
    pub use crate::nn::loss::*;
}

/// Core prelude re-exporting common types, neural network modules, shapes, and backend traits.
pub mod prelude {
    pub use super::err::*;
    pub use crate::SeqTy;
    pub use crate::compiled::{
        AllocationPlanner, ArtifactHeader, ArtifactVersion, BufferSlot, CapturedGraph,
        CapturedNode, CompileOptions, CompiledArtifact, CompiledPlan, ConstantFolder,
        DynamicShapePolicy, FusedKernel, FusionBlocker, FusionCandidate, FusionPass, FusionPolicy,
        LivenessInterval, LivenessMap, MemoryPlan, SavedTensorSet, ShapeBucket, ShapeGuard,
        WeightPrepacker,
    };

    pub use crate::dim;

    pub use crate::dist::{Local, Placement, PlacementKind};
    pub use crate::distributions::{Bernoulli, Distribution, Exponential, Gumbel, Normal, Uniform};
    pub use crate::exec::{LossScaling, PrecisionPolicy};

    pub use crate::graph::{Graph, OpType};
    pub use crate::metrics::{Accuracy, ConfusionMatrix, F1Score, MSE, Metric, Precision, Recall};
    pub use crate::nn::{
        activation::{GELU, ReLU, Sigmoid, Softmax, Swish, Tanh},
        avg_pool2d::AvgPool2d,
        batch_norm::BatchNorm2d,
        conv1d::Conv1d,
        conv2d::Conv2d,
        dropout::Dropout,
        embedding::Embedding,
        flatten::Flatten,
        init::Init,
        layer_norm::LayerNorm,
        linear::{Linear, LinearShape},
        loss::{
            BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, Mean, NoneReduction, Reduction,
        },
        lstm::{LSTM, LSTMCell},
        max_pool2d::MaxPool2d,
        module::{
            AutorefNamedLayers, AutorefNamedLayersFallback, AutorefParameters,
            AutorefParametersFallback, AutorefShapeInfo, AutorefShapeInfoFallback,
            AutorefStateDict, AutorefStateDictFallback, AutorefTrainMode, AutorefTrainModeFallback,
            LayerNode, Module, NamedLayers, Parameters, Sequential, StateDict, ToDevice, TrainMode,
        },
        optional::{False, OptionalField, True},
        param::Param,
        rms_norm::RMSNorm,
        rnn::{RNN, RNNCell},
        stats::{
            AutorefComputeStats, AutorefComputeStatsFallback, ComputeStats, LayerStats, ModelStats,
        },
    };
    pub use crate::seq;
    pub use crate::tensor::ops::index::IndexSpec;
    pub use incin_macros::{idx, module, s};

    pub use super::shapes::prelude::*;
    pub use super::tensor::prelude::*;
    #[cfg(feature = "std")]
    pub use crate::io::{GgufExporter, GgufMetadata, MlxExporter, QuantScheme, inspect_file};
    #[cfg(feature = "std")]
    pub use crate::nn::save::{load_safetensors, load_safetensors_map, save_safetensors};
    #[cfg(feature = "std")]
    pub use crate::onnx_exporter::{OnnxExporter, OnnxImporter};
    pub use crate::optim::{
        Adam, AdamW, ConstantLR, Gradients, LRScheduler, LinearLR, Optimizer, SGD,
    };
    #[cfg(feature = "std")]
    pub use crate::optim::{CosineAnnealingLR, StepLR};
    pub use crate::serialize::{Deserializer, Serializer};
    #[cfg(feature = "std")]
    pub use crate::serialize::{Format, ModelExt};
    pub use crate::shapes::dim::Dim;
    pub use crate::shapes::shape::{ConstShape, DynShape, PartialDynShape, Shape};
    pub use alloc::boxed::Box;
    pub use alloc::collections::BTreeMap;
    pub use alloc::format;
    pub use alloc::string::{String, ToString};
    pub use alloc::vec::Vec;
    pub use typenum;
}
