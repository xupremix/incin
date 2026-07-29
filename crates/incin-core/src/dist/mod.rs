//! Placement and distributed-execution contracts.
//!
//! EXE-006 introduces only the local placement foundation required by
//! `StorageBackend`. DST-001 adds the typed logical mesh; DST-003 onward extend
//! this module with distributed typestates and collective planning.

#[cfg(feature = "distributed")]
pub mod mesh;
pub mod placement;

pub use placement::{Local, Placement, PlacementKind};
