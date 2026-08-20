//! The central `Tensor` type: its definition, invariant-preserving
//! constructors, accessors, and conversions.
//!
//! Split by concern rather than kept as one flat file: `types` holds the
//! struct itself plus its `Clone`/`Display`/`Debug` impls; `error` holds
//! the one piece of validation logic shared across constructors;
//! `accessors` holds plain read-only getters; `placed` holds the
//! placement-generic invariant-preserving constructors; `distributed`
//! holds the `distributed`-gated proof-joining machinery; `local` holds
//! the `Local`-placement constructor family (the crown-jewel invariant
//! boundary, kept together in one file, matching `docs/HANDOFF.md`'s
//! guidance not to scatter a single construction path); `creation` holds
//! the public value-creation front door (`zeros`/`ones`/`from_slice`/etc.);
//! `convert` holds the grad-state/device/shape conversion family.
//!
//! All six `Tensor` fields are already `pub(crate)`, so this split changes
//! no privacy boundary except one: `ConstructionWitness` stays private to
//! `local.rs`, the only file that constructs or consumes it.

mod accessors;
mod convert;
mod creation;
#[cfg(feature = "distributed")]
mod distributed;
mod error;
mod local;
mod placed;
mod types;

pub use types::Tensor;

#[cfg(feature = "distributed")]
pub use distributed::PlacedTensorError;

// `Tensor`'s generic-over-`Backend` machinery is exercised in
// `tests/constructor_ranks.rs` against the real CPU backend. A backend crate
// cannot be linked into this crate's own unit tests without duplicating
// `incin-core`, so the proofs live in the integration tests instead.
