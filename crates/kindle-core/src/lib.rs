#![doc = include_str!("../../../README.md")]
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

pub(crate) mod graph;
pub(crate) mod nn;
#[cfg(feature = "std")]
pub(crate) mod onnx_exporter;
#[cfg(feature = "std")]
pub(crate) mod onnx_pb;
pub(crate) mod optim;
pub(crate) mod serialize;
pub(crate) mod shapes;
pub(crate) mod tensor;

/// Loss functions and reduction definitions.
pub mod loss {
    pub use crate::nn::loss::*;
}

/// Core prelude re-exporting common types, neural network modules, shapes, and backend traits.
pub mod prelude {
    pub use super::err::*;
    pub use crate::graph::{Graph, OpType};
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
        max_pool2d::MaxPool2d,
        module::{
            AutorefNamedLayers, AutorefNamedLayersFallback, AutorefParameters,
            AutorefParametersFallback, AutorefShapeInfo, AutorefShapeInfoFallback,
            AutorefStateDict, AutorefStateDictFallback, LayerNode, Module, NamedLayers, Parameters,
            Sequential, StateDict, ToDevice,
        },
        optional::{False, OptionalField, True},
        param::Param,
        rms_norm::RMSNorm,
        rnn::{RNN, RNNCell},
    };
    pub use crate::seq;
    pub use kindle_macros::{idx, module, s};

    pub use super::shapes::prelude::*;
    pub use super::tensor::prelude::*;
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
    pub use crate::symbolic_dim;
    pub use alloc::boxed::Box;
    pub use alloc::collections::BTreeMap;
    pub use alloc::format;
    pub use alloc::string::{String, ToString};
    pub use alloc::vec::{self, Vec};
    pub use typenum::{self, B0, B1, Bit, Diff, Prod, Quot, Sum, UInt, UTerm, Unsigned};
}
