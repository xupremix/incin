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
//!   typed shape rule     →     exact catalog descriptor → kernel launch
//! ```
//!
//! The module fills in over several tasks, in dependency order:
//!
//! | Submodule | Task | Contents |
//! |---|---|---|
//! | [`spec`] | `EXE-001` | shared descriptor identity and axis collections |
//! | [`proof`] | `EXE-002` | [`ProofLevel`] and the sealed [`Validated<O>`](Validated) wrapper |
//! | `meta` | `EXE-004` | `TensorMeta`, `LayoutClass`, `Alignment` |
//! | `capability` | `EXE-005` | the capability registry |
//!
//! The canonical catalog descriptors are internally consistent because their
//! constructors derive every field rather than accepting it. A bare descriptor
//! still carries no evidence that a shape proof stood behind it.
//! [`Validated<O>`](Validated) is that evidence, and [`rule`] is what mints it:
//! a descriptor wrapped in `Validated` came from a typed operand whose frontend
//! trait had already proved the operation legal.
//!
//! [`Execute<O>`](crate::backend_authoring::Execute) is now the descriptor consumer.
//! Concrete backend execution consumes the canonical catalog descriptors.

/// Backend-neutral capability queries and registry resolution.
pub mod capability;
/// Canonical exact-operation inventory and typed descriptor vocabulary.
pub mod catalog;
/// Backend-reusable storage-free semantic vectors.
pub mod conformance;
/// Backend-owning execution context foundation.
pub mod context;
/// The production path from a canonical operation to native execution.
pub mod dispatch;
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
/// Frozen operation descriptors and the schema version they are pinned to.
pub mod spec;
/// The backend-neutral autograd tape.
pub mod tape;

pub use capability::{
    Capabilities, CapabilityQuery, CapabilityRegistry, CapabilityRule, ImplementationKind,
    OperationIdentity, RankSupport, SupportLevel, UnsupportedReason,
};
pub use catalog::{
    AttributeContract, CanonicalOperation, Descriptor, DescriptorError, LogicalTensorMeta,
    OPERATION_CATALOG, Operation, OperationCatalogEntry, OperationKey, ValidatedInvocation,
    catalog_entry, op, operation_semantics_document,
};
// The classification fields of a catalog entry, re-exported beside the entry
// itself. Reading one of these fields off a public struct should not require
// knowing which submodule its type was declared in.
pub use catalog::{
    BroadcastingRule, DTypeRule, EmptyRule, ExecutionSite, GradientRule, LayoutRule, NumericRule,
    OutputRule, SemanticProfile,
};
#[cfg(feature = "std")]
pub use catalog::{CapturedDescriptor, DescriptorCaptureError};
pub use conformance::{
    ConformanceClass, ConformanceVector, ExpectedDisposition, SEMANTIC_CONFORMANCE_VECTORS,
};
pub use context::ExecutionContext;
pub use dispatch::CanonicalError;
pub use meta::{Alignment, LayoutClass, MetaError, TensorMeta};
pub use policy::{
    AllocatorPolicy, Determinism, ExecutionPolicy, FallbackPolicy, GradMode, MathMode, NanPolicy,
};
#[cfg(feature = "std")]
pub use policy::{check_gradients, no_grad};
pub use precision::{
    LossScaleState, LossScaling, PrecisionCapabilities, PrecisionChoice, PrecisionRequest,
    PrecisionRole, PrecisionSpec, ResolvedPrecision, RuntimePrecisionPolicy, resolve_precision,
};
pub use proof::{ProofLevel, Validated};

pub use request::TensorHandle;
pub use spec::{AxisSet, DescriptorSchemaVersion, ExecutionDescriptor, ReduceOp};
pub use tape::{BackwardFn, GradientMap, Tape, TapeNode, TapeStorage, TensorId};
