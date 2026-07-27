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
//! | `rule` | `EXE-003` | `ShapeRule`, binding each descriptor to the frontend trait that names its `Output` |
//! | `meta` | `EXE-004` | `TensorMeta`, `LayoutClass`, `Alignment` |
//! | `capability` | `EXE-005` | the capability registry |
//!
//! The first two exist today. A bare descriptor from [`spec`] is internally
//! consistent — its constructors derive every field rather than accepting it —
//! but anyone can build one, so it carries no evidence that a shape proof
//! stood behind it. [`Validated<O>`](Validated) is that evidence, and
//! `EXE-003` supplies the rules that mint it. Until then nothing inside the
//! crate calls the constructor, which is why the descriptors keep validating
//! their own arguments.

/// The sealed wrapper and the provenance it carries.
pub mod proof;
/// Frozen operation descriptors and the schema version they are pinned to.
pub mod spec;

pub use proof::{ProofLevel, Validated};
pub use spec::{
    AxisMask, BroadcastSpec, Conv2dSpec, DescriptorSchemaVersion, MatMulSpec, OperationSpec,
    ReductionSpec,
};

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
