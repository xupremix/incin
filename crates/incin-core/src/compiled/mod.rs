//! Compiled graph representation, IR capture, and execution planning.

pub mod alloc;
pub mod artifact;
pub mod capture;
pub mod fold;
pub mod fusion;
pub mod plan;

pub use alloc::{AllocationPlanner, BufferSlot, LivenessInterval, LivenessMap, MemoryPlan, SavedTensorSet};
pub use artifact::{
    ArtifactHeader, ArtifactVersion, CompiledArtifact, ARTIFACT_FORMAT_VERSION, ARTIFACT_MAGIC,
};
pub use capture::{CapturedGraph, CapturedNode};
pub use fold::{ConstantFolder, ShapeBucket, WeightPrepacker};
pub use fusion::{FusedKernel, FusionBlocker, FusionCandidate, FusionPass};
pub use plan::{CompileOptions, CompiledPlan, DynamicShapePolicy, FusionPolicy, ShapeGuard};
