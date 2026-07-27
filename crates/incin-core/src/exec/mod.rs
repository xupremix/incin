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
//! | `proof` | `EXE-002` | `ProofLevel` and the sealed `Validated<O>` wrapper |
//! | `rule` | `EXE-003` | `ShapeRule`, binding each descriptor to the frontend trait that names its `Output` |
//! | `meta` | `EXE-004` | `TensorMeta`, `LayoutClass`, `Alignment` |
//! | `capability` | `EXE-005` | the capability registry |
//!
//! Only the first exists today. A descriptor built through [`spec`] is
//! internally consistent but is not yet *proof-carrying*: until `EXE-002` seals
//! `Validated<O>`, any caller can build one. That is why the constructors here
//! validate everything they derive rather than trusting their arguments.

/// Frozen operation descriptors and the schema version they are pinned to.
pub mod spec;

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
