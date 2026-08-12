//! Compiled graph representation, IR capture, and execution planning.

pub mod alloc;
pub mod artifact;
pub mod capture;
pub mod fold;
pub mod fusion;
pub mod manifest;
pub mod plan;
pub mod tuning;

pub use alloc::{
    AllocationPlanner, BufferSlot, LivenessInterval, LivenessMap, MemoryPlan, SavedTensorSet,
};
pub use artifact::{
    ARTIFACT_FORMAT_VERSION, ARTIFACT_MAGIC, ArtifactHeader, ArtifactVersion, CompiledArtifact,
};
pub use capture::{CapturedGraph, CapturedNode};
pub use fold::{ConstantFolder, ShapeBucket, WeightPrepacker};
pub use fusion::{FusedKernel, FusionBlocker, FusionCandidate, FusionPass};
pub use manifest::ReproducibilityManifest;
pub use plan::{CompileOptions, CompiledPlan, DynamicShapePolicy, FusionPolicy, ShapeGuard};
pub use tuning::{BoundedPlanTuner, PlanTuningReport, TuningUnavailable};
