//! # Incin
//!
//! Incin is a lightweight, strictly-typed neural network framework for Rust, built with an emphasis on **shape safety** at compile time, while remaining fully compatible with dynamic runtime shapes. It is designed to catch tensor dimension mismatches during compilation using Rust's powerful type system, rather than failing at runtime.
//!
//! ## Key Features
//!
//! * **Compile-time Shape Verification**: Write tensor operations with `Tensor<s![Batch, Channels, Height, Width], Backend>` and let the compiler guarantee that shapes align for operations like `matmul`, `conv2d`, `concat`, etc.
//! * **Backends**: CPU is enabled by default and is the complete backend - every backend-executable operation in the canonical catalog has a CPU executor. The native CUDA, WGPU, and Metal backends are explicit opt-ins and are previews: each covers a documented subset of the catalog, so a model built from the layers in this crate trains on CPU. `docs/capabilities.md` is generated from the backend registrations and answers this per operation. The third-party Candle adapter is available through the `external-candle` feature under `external::candle`.
//! * **Macro-driven Ergonomics**: Stable macros such as `s![]`, `shape![]`, `axis![]`, and `i![]` define shapes, selectors, and indexing. Partial ONNX expansion is available separately as `experimental::model!`.
//! * **Zero-Cost Abstractions**: The static shape information (`typenum`) exists entirely in the type system and evaporates at runtime, introducing zero overhead to the underlying backend operations.
//!
//! ## Quick Start
//!
//! Tensors are allocated by starting from a target device such as `Cpu`:
//!
//! ```rust
//! use incin::prelude::*;
//!
//! // Statically shaped, zero-filled: (Batch=2, Channels=3, Height=224, Width=224)
//! let x = Cpu.zeros(shape![2, 3, 224, 224]).unwrap();
//! ```
//!
//! [`TargetExt`](prelude::TargetExt) documents the rest of the creation API
//! (`randn`, `rand`, `ones`, `full`, `arange`, `linspace`, `tensor`, dynamic
//! runtime shapes) with a runnable example on every method.

//!
//! ## Neural Network Modules
//!
//! Building and running a model is straightforward using the `#[module]` attribute and the
//! `seq!`/`SeqTy!` macros - `SeqTy!` names the same nested `Sequential<...>`
//! type that `seq!` builds a value of, so a layer list only needs to be written once per meaning
//! instead of the field type being hand-nested separately:
//!
//! ```rust,no_run
//! use incin::prelude::*;
//!
//! type Backend = IncinBackend<Cpu>;
//!
//! #[module(no_shape_info)]
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
//!     pub fn forward(&self, x: Tensor<s![2, 768], Backend>) -> Result<Tensor<s![2, 10], Backend, f32, Grad>>
//!     {
//!         self.net.forward(x)
//!     }
//! }
//! ```
//!
//! ## ONNX Import
//!
//! ONNX expansion is currently partial and fail-closed. Stateless graphs using
//! the documented supported subset can produce typed eager code. Models with
//! initializers, unknown rank, control flow, custom domains, or unsupported
//! nodes are rejected during macro expansion; no weights or metadata are
//! invented.
// mdBook's standalone tester does not receive Cargo's `--extern` arguments.
// Keep a Cargo-backed doctest mirror so the user-facing chapters are checked
// against the real facade and feature set.
#![warn(missing_docs)]

#[cfg(all(doctest, feature = "backend-authoring"))]
#[doc = concat!(
    include_str!("../../../docs/book/src/introduction.md"),
    include_str!("../../../docs/book/src/installation.md"),
    include_str!("../../../docs/book/src/editor_integrations.md"),
    include_str!("../../../docs/book/src/quickstart.md"),
    include_str!("../../../docs/book/src/tensors.md"),
    include_str!("../../../docs/book/src/shapes.md"),
    include_str!("../../../docs/book/src/advanced_shapes.md"),
    include_str!("../../../docs/book/src/autograd.md"),
    include_str!("../../../docs/book/src/building_models.md"),
    include_str!("../../../docs/book/src/sequential.md"),
    include_str!("../../../docs/book/src/transformer.md"),
    include_str!("../../../docs/book/src/training.md"),
    include_str!("../../../docs/book/src/data_loading.md"),
    include_str!("../../../docs/book/src/quantization.md"),
    include_str!("../../../docs/book/src/distributed.md"),
    include_str!("../../../docs/book/src/metrics.md"),
    include_str!("../../../docs/book/src/saving_loading.md"),
    include_str!("../../../docs/book/src/backends.md"),
    include_str!("../../../docs/book/src/target_api.md"),
    include_str!("../../../docs/book/src/backend_authoring.md"),
    include_str!("../../../docs/book/src/proofs_to_execution.md"),
    include_str!("../../../docs/book/src/custom_operations.md"),
    include_str!("../../../docs/book/src/macros.md"),
    include_str!("../../../docs/book/src/feature_flags.md"),
    include_str!("../../../docs/book/src/invariants.md"),
    include_str!("../../../docs/book/src/errors.md"),
    include_str!("../../../docs/book/src/experimental.md"),
    include_str!("../../../docs/book/src/pytorch_cheatsheet.md"),
    include_str!("../../../docs/book/src/release_notes.md"),
    include_str!("../../../docs/book/src/whats_not_finished.md"),
)]
mod book_docs {}

#[cfg(all(doctest, feature = "backend-authoring"))]
mod target_api_book_docs {}

extern crate alloc;

pub use incin_backends::IncinBackend;
pub use incin_core::autograd::Gradients;
pub use incin_core::error::{
    BackendError, BackwardError, ConversionFailure, Error, ErrorMessage, FloatToIntPolicy,
    NonFiniteSite, Result, convert_f64_to_i64,
};
pub use incin_core::nn::module::Module;
pub use incin_core::nn::param::{Buffer, TrainState};
pub use incin_core::nn::state::{
    StateMutVisitor, StatePath, StateRole, StateSnapshot, StateValue, StateVisitor, VisitState,
    VisitStateMut,
};
pub use incin_core::optim::{
    Adam, AdamW, ConstantLR, LRScheduler, LinearLR, Optimizer, OptimizerBackend, SGD,
};
#[cfg(feature = "std")]
pub use incin_core::optim::{CosineAnnealingLR, StepLR};
pub use incin_core::shapes::{Dyn, DynShape, Shape};
pub use incin_core::tensor::device::{
    Cpu, Device, DeviceId, DeviceKind, DevicePreference, DeviceSet, DeviceSetError,
};
pub use incin_core::tensor::dtype::{
    BoolDType, BuiltinDType, ConstDType, DType, DTypeDescriptor, DTypeId, DTypeKey, DTypeKind,
    FloatDType, IntDType, PlainDType, Q8_0, QuantDType, TensorElement, bf16, f16,
};
pub use incin_core::tensor::grad::{Grad, NoGrad, RequiresGrad};

/// Incin's target-backed adapter for model-ready MNIST batches.
#[derive(Clone)]
pub struct TargetBatch<T>(pub T);

impl<T> incin_data::vision::mnist::MnistBatchTarget for TargetBatch<T>
where
    T: incin_backends::target::TensorTarget + Clone + Send + Sync + 'static,
    T::Backend: incin_core::backend_authoring::HostInterop,
    <T::Backend as incin_core::backend_authoring::StorageBackend>::Storage<f32>: Send + 'static,
    <T::Backend as incin_core::backend_authoring::StorageBackend>::Storage<u8>: Send + 'static,
    <T::Device as incin_core::tensor::device::Device>::Field: Send + 'static,
{
    type Images = incin_backends::target::TargetTensor<T, incin_core::shapes::Dyn, f32>;
    type Labels = incin_backends::target::TargetTensor<T, incin_core::shapes::Dyn, u8>;

    fn batch(
        &self,
        images: Vec<f32>,
        labels: Vec<u8>,
        batch_size: usize,
    ) -> incin_data::BatchResult<(Self::Images, Self::Labels)> {
        use incin_backends::target::TargetExt;

        let images = self
            .0
            .tensor_from_vec(images, vec![batch_size, 1, 28, 28])
            .map_err(|error| incin_data::DataError::InvalidBatch(error.to_string()))?;
        let labels = self
            .0
            .tensor_from_vec(labels, vec![batch_size])
            .map_err(|error| incin_data::DataError::InvalidBatch(error.to_string()))?;
        Ok((images, labels))
    }
}

/// Target-value MNIST loader extension for the facade crate.
pub trait MnistTargetExt {
    /// Builds a model-ready loader on the supplied target value.
    fn loader_on<T>(
        self,
        target: T,
    ) -> incin_data::loader::DataLoaderBuilder<
        incin_data::vision::mnist::MnistDataset,
        incin_data::vision::mnist::TensorCollate<TargetBatch<T>>,
    >
    where
        T: incin_backends::target::TensorTarget + Clone + Send + Sync + 'static,
        T::Backend: incin_core::backend_authoring::HostInterop,
        <T::Backend as incin_core::backend_authoring::StorageBackend>::Storage<f32>: Send + 'static,
        <T::Backend as incin_core::backend_authoring::StorageBackend>::Storage<u8>: Send + 'static,
        <T::Device as incin_core::tensor::device::Device>::Field: Send + 'static;
}

impl MnistTargetExt for incin_data::vision::mnist::MnistDataset {
    fn loader_on<T>(
        self,
        target: T,
    ) -> incin_data::loader::DataLoaderBuilder<
        incin_data::vision::mnist::MnistDataset,
        incin_data::vision::mnist::TensorCollate<TargetBatch<T>>,
    >
    where
        T: incin_backends::target::TensorTarget + Clone + Send + Sync + 'static,
        T::Backend: incin_core::backend_authoring::HostInterop,
        <T::Backend as incin_core::backend_authoring::StorageBackend>::Storage<f32>: Send + 'static,
        <T::Backend as incin_core::backend_authoring::StorageBackend>::Storage<u8>: Send + 'static,
        <T::Device as incin_core::tensor::device::Device>::Field: Send + 'static,
    {
        incin_data::vision::mnist::MnistDataset::loader(self, TargetBatch(target))
    }
}

#[cfg(feature = "cuda")]
pub use incin_core::tensor::device::{Cuda, CudaN};
#[cfg(feature = "metal")]
pub use incin_core::tensor::device::{Metal, MetalN};
#[cfg(feature = "wgpu")]
pub use incin_core::tensor::device::{Wgpu, WgpuN};

pub use incin_core::dim;
pub use incin_macros::module;

/// Expert type-level and physical-layout contracts that are not part of the
/// ordinary user prelude.
pub mod types {
    pub use incin_core::shapes::{
        ConcreteStaticExtent, DimCons, Nil, ReplaceAt, StructuralConcatShape,
    };

    /// Physical storage encoding used by dtype descriptors and serializers.
    pub mod dtype {
        pub use incin_core::tensor::dtype::StorageEncoding;
    }
}

/// Implementation details used by exported procedural macros.
///
/// This module is public solely because macro expansion occurs in the
/// consumer crate. Its contents are not part of the stable end-user facade.
#[doc(hidden)]
pub mod __macro_support {
    pub use alloc::{collections::BTreeMap, format, string::String, vec::Vec};
    pub use incin_core::backend_authoring::Backend;
    pub use incin_core::backend_authoring::{
        Capabilities, Execute, StorageTransfer, SupportsDType, TransferTo, VariableBackend, op,
    };
    pub use incin_core::error::Result;
    pub use incin_core::nn::{ComputeStats, LayerStats};
    pub use incin_core::nn::{
        LayerNode, NamedLayers, ParameterVisitor, ShapeInfo, StateMutVisitor, StatePath,
        StateVisitor, TrainMode, VisitParameters, VisitState, VisitStateMut,
    };
    pub use incin_core::tensor::device::Device;
    pub use incin_core::tensor::ops::index::IndexSpec;
    pub use incin_core::tensor::transfer::ToDevice;
    #[cfg(feature = "std")]
    pub use incin_core::nn::save::load_safetensors;
}

/// Unstable APIs that carry no compatibility guarantee.
pub mod experimental {
    /// Partial, fail-closed model import macros.
    pub use incin_macros::{import_model, model};
    #[cfg(feature = "distributed")]
    /// Experimental distributed declaration macros.
    pub use incin_macros::{mesh, parallel, placement};

    #[cfg(feature = "compiled")]
    /// Preview compiled-plan inspection and CPU reference evaluation.
    ///
    /// This namespace is deliberately excluded from Incin's stable API. It is
    /// a CPU reference evaluator and inspection surface, not an optimized
    /// compiler, deployment target, or portable artifact ABI. Optimizations
    /// that lack an executable lowering fail closed.
    pub mod compiled {
        pub use incin_backends::cpu::{
            CpuBuffer, CpuCompiledFunction, CpuCompiledInvocation, CpuCompiledPlan,
            CpuCompiledSupport, CpuStorage, compiled_support,
        };
        pub use incin_core::experimental::compiled::{
            AllocationPlanner, ArtifactHeader, ArtifactVersion, BoundedPlanTuner, BufferSlot,
            CapturedGraph, CapturedNode, CompileOptions, CompiledArtifact, CompiledPlan,
            ConstantFolder, DynamicShapePolicy, FusedKernel, FusionBlocker, FusionCandidate,
            FusionPass, FusionPolicy, LivenessInterval, LivenessMap, MemoryPlan, PlanTuningReport,
            ReproducibilityManifest, SavedTensorSet, ShapeBucket, ShapeGuard, TuningUnavailable,
            WeightPrepacker,
        };
        /// Minimal graph and descriptor-building types for the preview CPU
        /// evaluator. These remain experimental with the rest of this module.
        pub use incin_core::{
            exec::{
                OperationIdentity,
                catalog::{CapturedDescriptor, Descriptor, LogicalTensorMeta, NoAttributes, op},
            },
            graph::{AttributeValue, DescriptorPayload, Graph},
            shapes::OperationKind,
            tensor::{device::DeviceId, dtype::DTypeId},
        };
    }

    #[cfg(feature = "autotune")]
    /// Preview tuning configuration and inspection types.
    pub mod tuning {
        pub use incin_backends::tuning::{
            AlignmentClass, AutotunePolicy, CacheLimits, CacheRecovery, CompilerFingerprint,
            DTypePolicyId, DeviceFingerprint, KernelSignature, PersistentTuningCache, RankClass,
            SelectionSource, TuningContext, TuningEnvironmentFingerprint, TuningExplain,
            TuningProvenance, TuningScope, TuningSelection,
        };
    }

    #[cfg(feature = "distributed")]
    /// Experimental distributed planning and placement contracts.
    pub mod distributed {
        /// Typed logical mesh declarations and runtime binding contracts.
        pub mod mesh {
            pub use incin_core::dist::mesh::{
                AXIS_COUNT, BindError, CollectiveGroups, Data, DeviceIdentity, DeviceMesh,
                LinkClass, MeshAxis, MeshId, MeshSpec, Pipeline, ProcessLayout, TensorParallel,
                TopologyFingerprint, TopologyProbe, TransportVersion, ValidMesh,
            };
        }

        #[cfg(feature = "distributed-nccl")]
        pub use incin_backends::dist::{
            BootstrapRole, NcclBuffer, NcclEvent, NcclTopology, NcclTransport, NcclTransportError,
            TwoRankBootstrapConfig,
        };
        pub use incin_core::dist::{
            ActivationCheckpoint, AgreedPlan, CollectiveDType, CollectiveDescriptor,
            CollectiveError, CollectiveKind, CollectivePlan, CollectivePlanBuilder,
            CollectiveReductionDType, CollectiveTag, CommunicationEvidence, CompletePlacement,
            ConstPlacement, ContextError, ContextFailure, DataParallelDType, DataParallelError,
            DataParallelPlan, DataParallelPlanBuilder, DistributedContext,
            DistributedContextHandle, DistributedContextState, DistributedError,
            DistributedIdentity, DistributedInputs, DistributedRule, ElementwisePlacement,
            FsdpError, FsdpMemoryReport, FsdpParameterDescriptor, FsdpParameterId, FsdpPlan,
            FsdpPlanBuilder, GPipe, GradientDescriptor, GradientId, GroupId, HybridPlanDType,
            HybridPlanError, HybridPlanReport, HybridPlanner, HybridWorkload,
            LOCAL_CUDA_DEVICE_ENV, LegalTransition, Local, Max, Mean, MemoryLimit, Min,
            OneForwardOneBackward, ParallelOptions, ParallelStrategy, ParallelStrategyKind,
            Partial, PartialReduction, PipelineAction, PipelineBoundaryId, PipelineClock,
            PipelineDType, PipelineError, PipelinePhase, PipelinePlan, PipelinePlanBuilder,
            PipelineSchedule, PipelineScheduleDescriptor, PipelineStage, PipelineTransfer,
            PipelineTransferDescriptor, Placement, PlacementAxis, PlacementBuf, PlacementKind,
            PlacementOn, PlacementTransition, PlacementTransitionRule, PlanError, PlanObjective,
            PlanSummary, PlannedCollectiveTransition, PlanningCollectiveKind, Prod, RANK_ENV,
            RENDEZVOUS_ADDR_ENV, RENDEZVOUS_TIMEOUT_MS_ENV, RUN_ID_ENV, ReduceShardedAxis,
            RejectedStrategy, Replicated, RunId, SequenceToken, ShardDivisible, ShardEvidence,
            ShardRemainderPolicy, Sharded, StaticParallelOptions, StaticPipelineSchedule,
            StaticTwoRank, StrategyCandidate, StrategyRejection, StrategySet, StreamId, Sum,
            TWO_RANK_WORLD, TensorParallelCollective, TensorParallelDType,
            TensorParallelDescriptor, TensorParallelDimension, TensorParallelError,
            TensorParallelId, TensorParallelPlan, TensorParallelPlanBuilder, TwoRankDataParallel,
            TwoRankPipeline, TwoRankPlanningTopology, TwoRankTensorParallel, TwoWayShard,
            ValidatedDistributed, WORLD_SIZE_ENV, WorkloadField, ZeROStage, preflight,
            validate_collective_dtype, validate_collective_reduction, validate_data_parallel_dtype,
            validate_hybrid_plan_dtype, validate_microbatches, validate_pipeline_dtype,
            validate_pipeline_stage, validate_shard, validate_tensor_parallel_dtype,
            validate_transition, validate_two_way_extent,
        };
        #[cfg(feature = "std")]
        pub use incin_core::dist::{
            DynRendezvousConfig, RankLaunch, RendezvousEndpoint, StaticRendezvousConfig,
            TwoRankLaunchPlan,
        };
    }

    #[cfg(feature = "train")]
    /// Preview training planner and trainer.
    pub mod training {
        pub use crate::train::{
            Decision, FitOutcome, HostMachine, Machine, Plan, TrainError, Trainer, TrainerBuilder,
        };

        /// Preview training-plan report renderer.
        pub mod plan_report {
            pub use crate::plan_report::{EXIT_OK, EXIT_USAGE, run, run_with_machine};
        }
    }

    #[cfg(feature = "std")]
    /// Preview tuning-report renderer used by the CLI.
    pub mod tuning_report {
        pub use crate::tune_report::{EXIT_OK, EXIT_USAGE, run};
    }
}

#[cfg(feature = "backend-authoring")]
/// Contracts and extension traits for backend authors.
pub mod backend_authoring {
    pub use incin_core::backend_authoring::{
        Alignment, AttributeContract, AutogradBackend, Backend, Capabilities, CapabilityQuery,
        CapabilityRegistry, DescriptorError, Execute, ExecuteOutput, ExecutionContext,
        ExecutionDescriptor, ExecutionRequest, HostInterop, HostReadback, LogicalTensorMeta,
        LossScaling, Operation, OperationIdentity, OperationKey, PrecisionSpec,
        RuntimePrecisionPolicy, ShapeBuf, StorageBackend, StorageOutput, StorageTransfer,
        SupportLevel, SupportsDType, TensorBackend, TensorMeta, TransferBackend, TransferTo,
        UnsupportedReason, Validated, VariableBackend, VariableTransfer,
    };
    pub use incin_core::backend_authoring::{
        execute, execute_shaped, execute_shaped_with_payload, execute_with_payload,
    };

    /// Canonical exact operation descriptors and typed attributes.
    pub mod operations {
        pub use incin_core::backend_authoring::operations::{
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
        // Mirrors the core tier exactly: the classification enums every field
        // of the re-exported `OperationCatalogEntry` is typed as.
        pub use incin_core::backend_authoring::operations::{
            BroadcastingRule, DTypeRule, EmptyRule, ExecutionSite, GradientRule, LayoutRule,
            NumericRule, OutputRule, SemanticProfile,
        };
        #[cfg(feature = "std")]
        pub use incin_core::backend_authoring::operations::{
            CapturedDescriptor, DescriptorCaptureError,
        };
    }
}

#[cfg(feature = "test-utils")]
/// Deterministic fault injection for tests that need a backend operation to
/// fail on demand. There is no stand-in backend here: a test that needs a
/// backend uses a real one.
pub mod test_utils {
    #[cfg(feature = "cpu")]
    pub use incin_backends::test_utils::{AssignFailureGuard, fail_assign_on, fail_assignments_at};
}

/// Neural network modules, activation functions, layers, and building blocks.
pub mod nn {
    pub use incin_core::nn::loss::{
        BCEWithLogitsShape, CrossEntropyReductionShape, CrossEntropyShape, L1ReductionShape,
        L1Shape, MSEShape, Mean, NoneReduction, Reduction, ReductionMode, Sum,
    };
    pub use incin_core::nn::param;
    pub use incin_core::nn::{
        AdaptiveAvgPool2d, AvgPool2d, BCEWithLogitsLoss, BatchNorm2d, BatchNormShape, Buffer,
        ComputeStats, Conv1d, Conv1dShape, Conv2d, Conv2dShape, CrossEntropyLoss, Dropout, ELU,
        Embedding, EmbeddingShape, False, Flatten, GELU, Init, L1Loss, LSTM, LSTMCell, LayerNode,
        LayerNorm, LayerNormShape, LayerStats, Linear, LinearShape, LstmShape, MSELoss, MaxPool2d,
        Mish, ModelStats, Module, NamedLayers, OptionalField, Param, ParameterVisitor, RMSNorm,
        RMSNormShape, RNN, RNNCell, ReLU, RnnShape, Sequential, Sigmoid, Softmax, Swish, Tanh,
        TrainMode, TrainState, True, VisitParameters, batch_norm2d, conv1d, conv2d, embedding,
        format_layer_summary, format_layer_summary_with_stats, layer_norm, linear, lstm, rms_norm,
        rnn, sum_stats,
    };
    #[cfg(feature = "distributed")]
    pub use incin_core::nn::{TwoWayColumnLinearShape, TwoWayRowLinearShape};
    pub use incin_core::tensor::transfer::ToDevice;
}

/// Optimization algorithms, loss functions, and learning rate schedulers.
pub mod optim {
    pub use incin_core::optim::{
        Adam, AdamW, ConstantLR, Gradients, LRScheduler, LinearLR, Optimizer, OptimizerBackend,
        ParameterGroup, SGD, clip_grad_norm, clip_grad_value,
    };
    #[cfg(feature = "std")]
    pub use incin_core::optim::{CosineAnnealingLR, StepLR};
}

/// Evaluation metrics (Accuracy, Precision, Recall, F1Score, MSE, ConfusionMatrix).
pub mod metrics {
    pub use incin_core::metrics::{
        Accuracy, ConfusionMatrix, F1Score, MSE, Metric, Precision, Recall,
    };
}

/// Dataset abstractions and data loading utilities.
pub mod data {
    pub use super::MnistTargetExt;
    pub use incin_data::vision;
    pub use incin_data::vision::mnist::MnistBatchTarget;
    pub use incin_data::{
        BatchResult, Collate, DataError, DataLoader, DataLoaderBuilder, Dataset, Downloader,
    };
}

/// Data transformations and augmentation pipeline.
pub mod transforms {
    pub use incin_data::transforms::{
        CenterCrop, Compose, Normalize, RandomHorizontalFlip, Scale, Transform,
    };
}

/// HuggingFace Hub downloading & pretrained model loading utilities.
#[cfg(feature = "data-hub")]
pub mod hub {
    pub use incin_data::hub::{HubApi, HubRepo, download, download_dataset, from_pretrained};
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

/// Model checkpoint artifacts and transactional state loading.
pub mod state {
    pub use incin_core::nn::state::{
        StateMutVisitor, StatePath, StateRole, StateSnapshot, StateSnapshotVisitor, StateValue,
        StateVisitor, VisitState, VisitStateMut, collect_state, load_state,
    };
}

// Enabling an accelerator must never silently change application behavior.
// CPU remains the default whenever it is available; accelerator-only builds
// get the one enabled device family.
#[cfg(feature = "cpu")]
/// Default device used by the standard installation.
pub type DefaultDevice = incin_core::tensor::device::Cpu;
#[cfg(all(not(feature = "cpu"), feature = "wgpu"))]
/// Default device for a WGPU-only build.
pub type DefaultDevice = incin_core::tensor::device::Wgpu;
#[cfg(all(not(feature = "cpu"), not(feature = "wgpu"), feature = "cuda"))]
/// Default device for a CUDA-only build.
pub type DefaultDevice = incin_core::tensor::device::Cuda;

#[cfg(feature = "train")]
mod plan_report;
#[cfg(feature = "train")]
mod train;
#[cfg(feature = "std")]
mod tune_report;

#[cfg(feature = "cpu")]
/// Default backend on the CPU, equivalent to `IncinBackend<Cpu>`.
pub type DefaultBackend = incin_backends::IncinBackend<incin_core::tensor::device::Cpu>;

// No `DefaultBackend` fallback when `cpu` is disabled: a `()` placeholder
// (the previous approach) doesn't implement `Backend`, so every type alias
// below that defaults its `B` param to it would compile-error deep inside
// trait-bound resolution the moment anyone actually used it, far from the
// real problem ("you disabled `cpu` and didn't pick another backend").
// Instead, every alias below drops the `B = DefaultBackend` default entirely
// in this configuration, same as `Tensor` already did - forcing an explicit,
// immediate "expected 2 generic arguments" error at the actual call site.

#[cfg(feature = "cpu")]
/// Tensor.
pub type Tensor<
    S,
    B = DefaultBackend,
    K = f32,
    G = incin_core::tensor::grad::NoGrad,
    P = incin_core::dist::Local,
> = incin_core::tensor::base::Tensor<S, B, K, G, P>;

#[cfg(not(feature = "cpu"))]
/// Tensor.
pub type Tensor<
    S,
    B, // User must specify backend if Cpu is disabled
    K = f32,
    G = incin_core::tensor::grad::NoGrad,
    P = incin_core::dist::Local,
> = incin_core::tensor::base::Tensor<S, B, K, G, P>;

// Neural Network Layer Aliases
//
// Each of these that takes a `B` (backend) param is declared twice, gated on
// the `cpu` feature: with a `= DefaultBackend` default when `cpu` is on
// (the common case), and with NO default when it's off - same reasoning as
// `Tensor` above and `DefaultBackend` itself.
#[cfg(feature = "cpu")]
/// Linear.
pub type Linear<S, B = DefaultBackend> = incin_core::nn::Linear<S, B>;
#[cfg(not(feature = "cpu"))]
/// Linear.
pub type Linear<S, B> = incin_core::nn::Linear<S, B>;

#[cfg(feature = "cpu")]
/// Conv1d.
pub type Conv1d<S, B = DefaultBackend> = incin_core::nn::Conv1d<S, B>;
#[cfg(not(feature = "cpu"))]
/// Conv1d.
pub type Conv1d<S, B> = incin_core::nn::Conv1d<S, B>;

#[cfg(feature = "cpu")]
/// Conv2d.
pub type Conv2d<S, B = DefaultBackend> = incin_core::nn::Conv2d<S, B>;
#[cfg(not(feature = "cpu"))]
/// Conv2d.
pub type Conv2d<S, B> = incin_core::nn::Conv2d<S, B>;

#[cfg(feature = "cpu")]
/// Batch norm2d.
pub type BatchNorm2d<C, B = DefaultBackend> = incin_core::nn::BatchNorm2d<C, B>;
#[cfg(not(feature = "cpu"))]
/// Batch norm2d.
pub type BatchNorm2d<C, B> = incin_core::nn::BatchNorm2d<C, B>;

#[cfg(feature = "cpu")]
/// Layer norm.
pub type LayerNorm<C, B = DefaultBackend> = incin_core::nn::LayerNorm<C, B>;
#[cfg(not(feature = "cpu"))]
/// Layer norm.
pub type LayerNorm<C, B> = incin_core::nn::LayerNorm<C, B>;

/// Avg pool2d.
pub type AvgPool2d<K, S, P = typenum::U0, D = typenum::U1> = incin_core::nn::AvgPool2d<K, S, P, D>;
/// Max pool2d.
pub type MaxPool2d<K, S, P = typenum::U0, D = typenum::U1> = incin_core::nn::MaxPool2d<K, S, P, D>;
/// Sequential.
pub type Sequential<L1, L2> = incin_core::nn::Sequential<L1, L2>;

#[cfg(feature = "cpu")]
/// Param.
pub type Param<T, B = DefaultBackend> = incin_core::nn::Param<T, B>;
#[cfg(not(feature = "cpu"))]
/// Param.
pub type Param<T, B> = incin_core::nn::Param<T, B>;

#[cfg(feature = "cpu")]
/// Rnncell.
pub type RNNCell<S, B = DefaultBackend> = incin_core::nn::RNNCell<S, B>;
#[cfg(not(feature = "cpu"))]
/// Rnncell.
pub type RNNCell<S, B> = incin_core::nn::RNNCell<S, B>;

#[cfg(feature = "cpu")]
/// Rnn.
pub type RNN<S, B = DefaultBackend> = incin_core::nn::RNN<S, B>;
#[cfg(not(feature = "cpu"))]
/// Rnn.
pub type RNN<S, B> = incin_core::nn::RNN<S, B>;

#[cfg(feature = "cpu")]
/// Embedding.
pub type Embedding<S, B = DefaultBackend> = incin_core::nn::Embedding<S, B>;
#[cfg(not(feature = "cpu"))]
/// Embedding.
pub type Embedding<S, B> = incin_core::nn::Embedding<S, B>;

/// Macros.
pub mod macros {
    // `impl_arg_into` deliberately excluded: it's an internal codegen helper
    // invoked once, internally, by `incin-core` itself
    // (`incin_macros::impl_arg_into!()` in `tensor/arg_into.rs`) - no
    // end-user code calls it, and it has no documented public contract.
    pub use incin_macros::{i, s, shape, tensor};

    /// Advanced type-level slicing and reshape syntax.
    pub mod advanced {
        pub use incin_macros::idx;
    }
}

/// Prelude re-exporting high-frequency user types, macros, NN modules, and optimizers.
pub mod prelude {
    pub use super::Tensor;
    pub use incin_core::SeqTy;
    pub use incin_core::error::{
        BackendError, BackwardError, ConversionFailure, Error, ErrorMessage, FloatToIntPolicy,
        NonFiniteSite, Result, convert_f64_to_i64,
    };
    pub use incin_core::nn::module::Module;
    pub use incin_core::nn::state::{StatePath, StateRole, StateSnapshot, StateValue};
    pub use incin_core::shapes::{
        AxisIdentity, AxisSchema, ConstDim, Dim, Dyn, DynShape, InferShape, NamedDim, Ranked,
        Shape, ShapeArgs, ShapeSpec, ShapeValue,
    };
    pub use incin_core::tensor::device::{
        ConstDevice, Cpu, Device, DeviceId, DeviceKind, DevicePreference, DeviceSet, DeviceSetError,
    };
    pub use incin_core::tensor::dtype::{
        BoolDType, BuiltinDType, ConstDType, DType, DTypeDescriptor, DTypeId, DTypeKey, DTypeKind,
        FloatDType, IntDType, PlainDType, Q8_0, QuantDType, TensorElement, bf16, f16,
    };
    pub use incin_core::tensor::grad::{Grad, NoGrad, RequiresGrad};
    pub use incin_core::tensor::matmul::MatMulShape;
    pub use incin_core::tensor::transfer::ToDevice;

    pub use incin_core::exec::AxisSet;
    pub use incin_core::shapes::broadcast::BroadcastDim;
    pub use incin_core::shapes::concat::ConcatShape;
    pub use incin_core::shapes::stack::StackShape;
    pub use incin_core::shapes::{
        AppendDim, Axis, BroadcastExtent, BroadcastShape, ReshapeShape, SameCount, Scalar,
        TryReshape,
    };

    #[cfg(feature = "cuda")]
    pub use incin_core::tensor::device::{Cuda, CudaN};
    #[cfg(feature = "metal")]
    pub use incin_core::tensor::device::{Metal, MetalN};
    #[cfg(feature = "wgpu")]
    pub use incin_core::tensor::device::{Wgpu, WgpuN};

    #[cfg(feature = "cpu")]
    pub use super::DefaultBackend;
    #[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda"))]
    pub use super::DefaultDevice;
    pub use incin_backends::IncinBackend;

    // Device values are the preferred allocation targets.
    // These are extension traits, so they only resolve when in scope - which
    // is the whole reason they are in the prelude rather than left to a
    // module path. See `docs/plan/UX-ARCHITECTURE-HANDOFF.md`.
    pub use incin_backends::nn_target::InitOnTarget;
    #[cfg(feature = "external-candle")]
    pub use incin_backends::target::Candle;
    pub use incin_backends::target::{
        DtypeTarget, EngineSpec, GeneratedFill, Native, PrecisionSpec, Target, TargetExt,
        TensorData, TensorTarget, precision,
    };

    pub use incin_core::dim;
    pub use incin_core::seq;
    pub use incin_core::typenum;

    pub use incin_macros::{axis, i, idx, module, s, shape, tensor};

    pub use super::{
        BatchNorm2d, Buffer, Conv1d, Conv2d, Embedding, LayerNorm, Linear, Param, RNN, RNNCell,
    };

    pub use incin_core::nn::{
        activation::{GELU, ReLU, Sigmoid, Softmax, Swish, Tanh},
        avg_pool2d::AvgPool2d,
        dropout::Dropout,
        flatten::{Flatten, FlattenAxes, StructuralFlatten},
        init::Init,
        loss::{
            BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, Mean, NoneReduction, Reduction,
        },
        lstm::{LSTM, LSTMCell},
        max_pool2d::MaxPool2d,
        module::{Sequential, TrainMode},
        rms_norm::RMSNorm,
    };

    #[cfg(feature = "std")]
    pub use incin_core::serialization::{Format, ModelExt, STATE_FORMAT_VERSION};

    pub use incin_core::exec::{LossScaleState, LossScaling};
    pub use incin_core::optim::{
        Adam, AdamW, ConstantLR, Gradients, LRScheduler, LinearLR, Optimizer, ParameterGroup, SGD,
        ScaledOptimizer,
    };
    #[cfg(feature = "std")]
    pub use incin_core::optim::{CosineAnnealingLR, StepLR};

    pub use incin_core::metrics::{
        Accuracy, ConfusionMatrix, F1Score, MSE, Metric, Precision, Recall,
    };

    #[cfg(feature = "distributed")]
    pub use incin_core::dist::{Local, Placement, PlacementKind};
}

/// Structural shape proofs for advanced generic code.
pub mod advanced {
    pub use incin_core::shapes::idx::{
        AxisCursor, AxisSelector, DimIdx, Ellipsis, ForwardAxis, FromEnd, Here, InferDim,
        NamedAxisLookup, NamedAxisSelector, Next, ReshapeTarget, ReverseAxis, Slice, SliceIdx,
        SliceTarget, StaticAxis, StaticCursor, ToAxisIndex,
    };
    pub use incin_core::shapes::{At, ReduceAt, ReduceKeepAt, RemoveAt, SwapAt};
    pub use incin_core::typenum;
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
        assert_eq!(t.dtype(), DTypeId::F64.descriptor());
    }

    #[test]
    #[cfg(feature = "cpu")]
    /// Test tensor export.
    fn test_tensor_export() {
        let _t = Tensor::<Dyn, DefaultBackend>::zeros(std::vec![2, 2]).unwrap();
    }
}
