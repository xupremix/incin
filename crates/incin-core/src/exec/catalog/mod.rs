//! Canonical stable-operation catalog and storage-free typed descriptors.
//!
//! Every executable semantic identity is declared exactly once in
//! `operation_catalog.rs`.  This module turns those rows into the inventory,
//! one marker (and therefore one concrete `Descriptor<Marker>` type) per
//! operation, and an owned representation suitable for tracing/capture.

// Validation is intentionally staged as readable guard clauses. Collapsing
// nested metadata-presence and predicate checks makes the fail-closed branches
// harder to audit and does not change generated code.
#![allow(clippy::collapsible_if, clippy::collapsible_match)]

// Shared across every submodule below via `use super::*;` — these mirror
// the original single-file catalog.rs's top-level imports so a split
// submodule needs no import work of its own beyond what it adds locally.
pub(crate) use crate::exec::OperationIdentity;
pub(crate) use crate::shapes::ShapeBuf;
pub(crate) use crate::shapes::error::OperationKind;
pub(crate) use crate::tensor::device::DeviceId;
pub(crate) use crate::tensor::dtype::{DTypeDescriptor, DTypeId};
pub(crate) use alloc::borrow::Cow;
pub(crate) use alloc::string::ToString;
pub(crate) use alloc::vec::Vec;
pub(crate) use core::fmt;
pub(crate) use core::marker::PhantomData;

mod attributes;
mod classification;
mod coverage;
mod descriptor;
mod error;
mod inference;
mod lookup;
mod meta;
mod shape_transform;
mod table;
mod validated;

#[cfg(test)]
mod tests;

pub use classification::*;
pub use coverage::*;
pub use descriptor::*;
pub use error::*;
pub use lookup::*;
// `inference`'s items are `pub(super)` (cross-visible within this module
// tree only, matching the original single-file catalog.rs's privacy), so
// this re-export uses `pub(crate)` rather than `pub` — just enough for
// sibling modules (e.g. `validated`) to reach them through `use super::*;`.
// The lint below misreports this as unused because it under-counts
// `pub(super)` items imported from a genuinely private context; the
// re-export is load-bearing — removing it breaks `validated.rs`.
#[allow(unused_imports)]
pub(crate) use inference::*;
pub use meta::*;
pub use shape_transform::*;
pub use table::*;
pub use validated::*;
