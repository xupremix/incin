//! Core tensor operations, static shape checking, and autograd for Incin.
// Incin errors intentionally carry rich operation, dtype, device, and shape
// context. Keep that error contract by allowing clippy's size heuristic at
// this crate boundary instead of boxing every error variant.
#![allow(clippy::result_large_err)]
#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
pub(crate) extern crate alloc;

pub(crate) mod err;

#[macro_use]
mod operation_catalog;

#[cfg(feature = "compiled")]
pub mod compiled;
pub mod dist;
pub mod distributions;
/// Structured errors returned by core and its extension crates.
pub mod error {
    pub use crate::err::{
        BackendError, BackwardError, ConversionFailure, Error, ErrorMessage, FloatToIntPolicy,
        NonFiniteSite, Result, convert_f64_to_i64,
    };
}
pub mod exec;
pub mod graph;
mod graph_recording;

pub mod io;
pub mod metrics;
pub mod nn;
pub mod resource;
#[cfg(feature = "std")]
pub(crate) mod onnx_exporter;
#[cfg(feature = "std")]
pub(crate) mod onnx_pb;
pub mod optim;
pub(crate) mod serialize;
pub mod shapes;

pub mod tensor;
pub mod autograd;
pub use typenum;

/// Implementation details used by procedural macros expanded inside this crate.
#[doc(hidden)]
pub mod __macro_support {
    pub use crate::nn::{ComputeStats, LayerStats, StateMutVisitor, StatePath, StateSnapshot, StateVisitor, VisitState, VisitStateMut};
    pub use crate::tensor::backend::{StorageTransfer, SupportsDType, TransferTo, VariableBackend};
    pub use alloc::{collections::BTreeMap, format, string::String, vec::Vec};
}

/// Loss functions and reduction definitions.
pub mod loss {
    pub use crate::nn::loss::{
        BCEWithLogitsLoss, BCEWithLogitsShape, BceReductionShape, CrossEntropyLoss,
        CrossEntropyReductionShape, CrossEntropyShape, L1Loss, L1ReductionShape, L1Shape, MSELoss,
        MSEShape, Mean, MseReductionShape, NoneReduction, Reduction, ReductionMode, Sum,
    };
}

/// Unstable APIs that carry no compatibility guarantee.
pub mod experimental {
    #[cfg(feature = "compiled")]
    /// Compiled graph plans and symbolic guards. CPU execution is exposed by
    /// the matching `incin-backends/compiled` feature.
    pub mod compiled {
        pub use crate::compiled::{
            AllocationPlanner, ArtifactHeader, ArtifactVersion, BoundedPlanTuner, BufferSlot,
            CapturedGraph, CapturedNode, CompileOptions, CompiledArtifact, CompiledPlan,
            ConstantFolder, DynamicShapePolicy, FusedKernel, FusionBlocker, FusionCandidate,
            FusionPass, FusionPolicy, LivenessInterval, LivenessMap, MemoryPlan, PlanTuningReport,
            ReproducibilityManifest, SavedTensorSet, ShapeBucket, ShapeGuard, TuningUnavailable,
            WeightPrepacker,
        };
    }
}

/// Extension traits and operation descriptor contracts for backend authors.
pub mod backend_authoring {
    pub use crate::exec::dispatch::{
        execute, execute_shaped, execute_shaped_with_payload, execute_with_payload,
    };
    pub use crate::exec::{
        Alignment, AttributeContract, CanonicalOperation, Capabilities, CapabilityQuery,
        CapabilityRegistry, Descriptor, DescriptorError, ExecutionContext, ExecutionDescriptor,
        LogicalTensorMeta, LossScaling, OPERATION_CATALOG, Operation, OperationCatalogEntry,
        OperationIdentity, OperationKey, PrecisionSpec, RuntimePrecisionPolicy, SupportLevel,
        TensorMeta, UnsupportedReason, Validated, ValidatedInvocation, op,
    };
    pub use crate::shapes::ShapeBuf;
    pub use crate::tensor::backend::{
        AutogradBackend, Backend, Execute, ExecuteOutput, ExecutionRequest, HostInterop, HostReadback,
        StorageBackend, StorageOutput, StorageTransfer, SupportsDType, TensorBackend, TransferBackend, TransferTo, VariableTransfer,
        VariableBackend,
    };
    /// Read the tracing graph mid-flight, without draining it.
    ///
    /// Here rather than in the prelude because the caller is a backend's tape
    /// emitting telemetry, not a user collecting a graph they built: the
    /// prelude's [`extract_graph`](crate::prelude::extract_graph) drains and is
    /// the one a user wants. `crate::tensor` is `pub(crate)`, so a re-export is
    /// the only way out of the crate at all, and putting a snapshot hook in the
    /// user prelude implies an audience that does not call it.
    pub use crate::tensor::tracing::tracing_graph_snapshot;

    /// Exact operation markers, typed attributes, and storage-free metadata.
    pub mod operations {
        pub use crate::exec::catalog::{
            AdamAttributes, AdamWAttributes, AdaptivePool2dAttributes, AddmmAttributes,
            ArangeAttributes, ArgsortAttributes, AttentionAttributes, AvgPool2dAttributes,
            AxisAttributes, AxisVarianceAttributes, BatchNormAttributes, CanonicalOperation,
            ChunkAttributes, ClampAttributes, Conv1dAttributes, Conv2dAttributes,
            ConvTranspose2dAttributes, CreationAttributes, DTypeAttributes, Descriptor,
            DescriptorError, DeviceAttributes, DiagonalAttributes, DistributionAttributes,
            DropoutAttributes, DuplicateIndexRule, EpsilonAttributes, FlattenAttributes,
            FullAttributes, GroupNormAttributes, IndexReductionAttributes, LayerNormAttributes,
            LerpAttributes, LinearAttributes, LinspaceAttributes, LogicalTensorMeta,
            LossAttributes, LossReduction, NarrowAttributes, NoAttributes, NormAttributes,
            OPERATION_CATALOG, Operation, OperationCatalogEntry, OperationKey, PadAttributes,
            PixelShuffleAttributes, Pool2dAttributes, QuantizationAttributes, RecurrentAttributes,
            RepeatAttributes, ScalarAttributes, ScatterAttributes, SgdAttributes, ShapeAttributes,
            SliceAttributes, SplitAttributes, TopKAttributes, TransposeAttributes,
            UnfoldAttributes, ValidatedInvocation, VarianceAttributes, catalog_entry, op,
        };
        // The enums that every classification field of `OperationCatalogEntry`
        // is typed as. The entry is re-exported above, so without these a
        // backend author can read `row.site` or `row.profile` but cannot write
        // down the type of what they read, or match on it exhaustively.
        pub use crate::exec::catalog::{
            BroadcastingRule, DTypeRule, EmptyRule, ExecutionSite, GradientRule, LayoutRule,
            NumericRule, OutputRule, SemanticProfile,
        };
        #[cfg(feature = "std")]
        pub use crate::exec::catalog::{CapturedDescriptor, DescriptorCaptureError};
    }
}

/// Expert type-level shape vocabulary kept outside the normal prelude.
pub mod types {
    pub use crate::prelude::{ConcreteStaticExtent, DimCons, Nil, ReplaceAt, StructuralConcatShape};
}

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils {
    pub use crate::tensor::backend::dummy::DummyBackend;
}

/// Core prelude re-exporting common types, neural network modules, shapes, and backend traits.
pub mod prelude {
    pub use super::err::{
        BackendError, BackwardError, ConversionFailure, Error, ErrorMessage, FloatToIntPolicy,
        NonFiniteSite, Result, convert_f64_to_i64,
    };
    pub use crate::SeqTy;
    pub use crate::graph::Graph;
    pub use half::{bf16, f16};

    pub use crate::dim;

    pub use crate::dist::{Local, Placement, PlacementKind};
    pub use crate::distributions::{Bernoulli, Distribution, Exponential, Gumbel, Normal, Uniform};

    pub use crate::metrics::{Accuracy, ConfusionMatrix, F1Score, MSE, Metric, Precision, Recall};
    pub use crate::nn::{
        activation::{GELU, ReLU, Sigmoid, Softmax, Swish, Tanh},
        avg_pool2d::AvgPool2d,
        batch_norm::{BatchNorm2d, BatchNorm2dBuilder, BatchNormShape, batch_norm2d},
        conv1d::{Conv1d, Conv1dBuilder, Conv1dShape, conv1d},
        conv2d::{Conv2d, Conv2dBuilder, Conv2dShape, conv2d},
        dropout::Dropout,
        embedding::{Embedding, EmbeddingBuilder, EmbeddingShape, embedding},
        flatten::Flatten,
        init::{self, Fan, Init, InitContext, InitPlan, ParameterRole},
        layer_norm::{LayerNorm, LayerNormBuilder, LayerNormShape, layer_norm},
        linear::{Linear, LinearBuilder, LinearShape, linear},
        loss::{
            BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, Mean, NoneReduction, Reduction,
        },
        lstm::{LSTM, LSTMBuilder, LSTMCell, LSTMCellBuilder, LstmShape, lstm, lstm_cell},
        max_pool2d::MaxPool2d,
        module::{
            LayerNode, Module, NamedLayers, ParameterVisitor, Sequential, ShapeInfo, TrainMode, VisitParameters,
        },
        optional::{False, OptionalField, True},
        param::{Buffer, Frozen, Param, TrainState, Trainable},
        rms_norm::{RMSNorm, RMSNormBuilder, RMSNormShape, rms_norm},
        rnn::{RNN, RNNBuilder, RNNCell, RNNCellBuilder, RnnShape, rnn, rnn_cell},
        state::{collect_state, load_state, StateMutVisitor, StatePath, StateRole, StateSnapshot, StateSnapshotVisitor, StateValue, StateVisitor, VisitState, VisitStateMut},
        stats::{ComputeStats, LayerStats, ModelStats},
    };
    pub use crate::seq;
    pub use crate::tensor::ops::index::IndexSpec;
    pub use incin_macros::{axis, idx, mesh, module, s, shape};
    pub use super::exec::{AxisSet, RankSupport};
    pub use super::tensor::matmul::{StaticDim, StaticOrNamedDim};

    pub use super::shapes::prelude::{
        AdaptiveAvgPool2dShape, AppendDim, At, Axis, AxisIdentity, AxisKey, AxisSchema,
        AxisSelector, AxisTag, BroadcastDim, BroadcastExtent, BroadcastShape,
        CheckedNumel, ConcatShape, ConcreteStaticExtent, ConstDim, ConvOutDim, Dim,
        DimCons, DimIdx, DimensionConstraint, DynShape, ElementCount, Ellipsis, EndsWith, FlatDim,
        FromEnd, HasChannels1D, HasChannels2D, Here, INLINE_RANK, InferDim, InlineOrHeap,
        NamedAxisLookup, NamedAxisSelector, NamedDim, Next, Nil, OperationKind, Pool2dShape,
        ProductDims, RankExpectation, Ranked, ReduceAt, ReduceKeepAt, RemoveAt,
        ReplaceAt, ReplaceLastDim, ReshapeShape, ReshapeTarget, SameCount, Scalar, Shape,
        ShapeArgs, ShapeBuf, ShapeError, ShapeSpec, ShapeValue, Slice, SliceIdx, SliceTarget,
        SpatialConv1d, SpatialConv2d, SpatialOut, StackShape, StaticAxis, StrideBuf,
        StructuralConcatShape, SwapAt, ToAxisIndex, TryConcatShape,
        TryReshape, broadcast_dim_slices, checked_numel_from_dims,
        shape_buf_from_dims, spatial_out_size,
    };
    #[cfg(feature = "distributed")]
    pub use super::tensor::prelude::PlacedTensorError;
    pub use super::tensor::prelude::{
        ArgInto, AutogradBackend, Backend, BestDevice, BestDeviceAt, BoolDType, BuiltinDType, ConstDType,
        ConstDevice, Cpu, DType, DTypeDescriptor, DTypeId, DTypeKey, DTypeKind, Device, DeviceId,
        DeviceKind, DevicePreference, DeviceSet, DeviceSetError, Dyn, FloatDType, Grad, GradJoin,
        IntDType, JoinedGrad, MatMulShape, NoGrad, PlainDType, Q8_0, QuantDType, RequiresGrad,
        HostInterop, StorageBackend, StorageEncoding, SupportsDType, Tensor, TensorArgs,
        TensorArgsData, TensorElement, ToDevice, TracingBackend, TransferBackend, StorageTransfer, TransferTo,
        VariableBackend,
        CheckedByteLen, checked_byte_len_from_dims,
        extract_graph, tracing_mark_input,
        tracing_mark_input_typed, tracing_mark_output,
    };
    pub use crate::distributions::TensorDistributionExt;
    #[cfg(feature = "cuda")]
    pub use super::tensor::prelude::{Cuda, CudaN};
    #[cfg(feature = "metal")]
    pub use super::tensor::prelude::{Metal, MetalN};
    #[cfg(feature = "wgpu")]
    pub use super::tensor::prelude::{Wgpu, WgpuN};
    #[cfg(feature = "std")]
    pub use crate::io::{
        GgufExporter, GgufMetadata, MlxExporter, QuantScheme, ResourceLimits, inspect_file,
    };
    #[cfg(feature = "std")]
    pub use crate::nn::save::{load_safetensors, load_safetensors_snapshot, save_safetensors};
    #[cfg(feature = "std")]
    pub use crate::onnx_exporter::{OnnxExporter, OnnxImporter, export_to_onnx};
    pub use crate::optim::{
        Adam, AdamW, ConstantLR, Gradients, LRScheduler, LinearLR, Optimizer, ParameterGroup,
        SGD,
    };
    #[cfg(feature = "std")]
    pub use crate::optim::{CosineAnnealingLR, StepLR};
    #[cfg(feature = "std")]
    pub use crate::serialize::{Format, ModelExt};
    pub use alloc::boxed::Box;
    pub use alloc::collections::BTreeMap;
    pub use alloc::format;
    pub use alloc::string::{String, ToString};
    pub use alloc::vec::Vec;
    pub use typenum;
}

pub use incin_macros::axis;
