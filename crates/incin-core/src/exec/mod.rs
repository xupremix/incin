//! Proof-carrying operation lowering.
//!
//! `PROPOSALS.md` §1.1.1 names the gap this module closes: the typed frontend
//! proves an operation legal, wraps the result in `Tensor<Output, ..>`, and
//! then hands the backend nothing but storage and loose runtime integers. The
//! backend must either repeat the frontend's work or assume it happened.
//!
//! The layer here is the missing middle. An operation is resolved *once* into a
//! descriptor — a plain, inspectable struct holding the output shape, the
//! iteration geometry, and the launch parameters that follow from the shape
//! proof. Backends read the descriptor and trust it.
//!
//! ```text
//!   typed frontend            this module              native execution
//!   ──────────────            ───────────              ────────────────
//!   BroadcastShape      →     BroadcastSpec      →     kernel launch
//!   MatMulShape         →     MatMulSpec         →     GEMM call
//!   ReduceDim           →     ReductionSpec      →     reduction launch
//!   Conv2dShape         →     Conv2dSpec         →     conv call
//! ```
//!
//! The module fills in over several tasks, in dependency order:
//!
//! | Submodule | Task | Contents |
//! |---|---|---|
//! | [`spec`] | `EXE-001` | the descriptors themselves, and the schema they are frozen at |
//! | [`proof`] | `EXE-002` | [`ProofLevel`] and the sealed [`Validated<O>`](Validated) wrapper |
//! | [`rule`] | `EXE-003` | [`ShapeRule`], binding each descriptor to the frontend trait that names its `Output` |
//! | `meta` | `EXE-004` | `TensorMeta`, `LayoutClass`, `Alignment` |
//! | `capability` | `EXE-005` | the capability registry |
//!
//! The first three exist today, and together they close the loop. A bare
//! descriptor from [`spec`] is internally consistent — its constructors derive
//! every field rather than accepting it — but anyone can build one, so it
//! carries no evidence that a shape proof stood behind it.
//! [`Validated<O>`](Validated) is that evidence, and [`rule`] is what mints it:
//! a descriptor wrapped in `Validated` came from a typed operand whose frontend
//! trait had already proved the operation legal.
//!
//! [`Execute<O>`](crate::backend_authoring::Execute) is now the descriptor consumer.
//! Concrete backend migrations are staged through `EXE-007` and `EXE-008`, so
//! the legacy operation families remain callable until `EXE-009` removes them.

/// Backend-neutral capability queries and registry resolution.
pub mod capability;
/// Canonical exact-operation inventory and typed descriptor vocabulary.
pub mod catalog;
/// Backend-reusable storage-free semantic vectors.
pub mod conformance;
/// Backend-owning execution context foundation.
pub mod context;
/// Checked physical storage metadata shared by all backends.
pub mod meta;
/// Backend-neutral execution policy vocabulary.
pub mod policy;
/// Floating-point precision policies and loss scaling for mixed-precision training.
pub mod precision;
/// The sealed wrapper and the provenance it carries.
pub mod proof;
/// Checked, type-erased inputs for descriptor execution.
pub mod request;
/// Lowering rules binding each descriptor to its frontend shape trait.
pub mod rule;
/// Frozen operation descriptors and the schema version they are pinned to.
pub mod spec;
/// The backend-neutral autograd tape.
pub mod tape;

pub use capability::{
    Capabilities, CapabilityQuery, CapabilityRegistry, CapabilityRule, ImplementationKind,
    SupportLevel, UnsupportedReason,
};
pub use catalog::{
    CanonicalOperation, Descriptor, DescriptorError, LogicalTensorMeta, OPERATION_CATALOG,
    OperationCatalogEntry, ValidatedInvocation, catalog_entry, op, operation_semantics_document,
};
#[cfg(feature = "std")]
pub use catalog::{CapturedDescriptor, DescriptorCaptureError};
pub use conformance::{
    ConformanceClass, ConformanceVector, ExpectedDisposition, SEMANTIC_CONFORMANCE_VECTORS,
};
pub use context::ExecutionContext;
pub use meta::{Alignment, LayoutClass, MetaError, TensorMeta};
pub use policy::{
    AllocatorPolicy, Determinism, ExecutionPolicy, FallbackPolicy, GradMode, MathMode, NanPolicy,
};
#[cfg(feature = "std")]
pub use policy::{check_gradients, no_grad};
pub use precision::{LossScaling, PrecisionPolicy};
pub use proof::{ProofLevel, Validated};

pub use request::TensorHandle;
pub use rule::{
    BroadcastRule, Conv2dArgs, Conv2dRule, MatMulRule, Pool2dRule, ReduceAllRule, ReduceKeepRule,
    ReduceRule, ReshapeRule, ShapeRule,
};
pub use spec::{
    AxisMask, BinaryOp, BroadcastSpec, Conv2dSpec, DescriptorSchemaVersion, ExecutionDescriptor,
    MatMulSpec, OperationSpec, Pool2dSpec, PoolOp, ReduceOp, ReductionSpec, ReshapeSpec,
};
pub use tape::{BackwardFn, GradientMap, Tape, TapeNode, TapeStorage, TensorId};

/// Supertrait used to seal public traits in this module against outside
/// implementations.
///
/// Sealing [`OperationSpec`] is deliberate. A descriptor is the *contract*
/// between the shape proof and native execution, so adding one is a change to
/// that contract and belongs in `incin-core` alongside the shape rule that
/// produces it. External backend authors implement backends (`EXE-006`), which
/// consume descriptors; they never need to define a new one, and a descriptor
/// defined outside the crate would carry no proof that anything validated it.
pub(crate) mod sealed {
    /// Implemented only inside `incin-core`.
    pub trait Sealed {}
}
