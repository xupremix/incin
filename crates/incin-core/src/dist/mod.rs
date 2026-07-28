//! Placement and distributed-execution contracts.
//!
//! EXE-006 introduces only the local placement foundation required by
//! `StorageBackend`. DST-001 onward extend this module with meshes, distributed
//! typestates, and collective planning.

pub mod placement;

pub use placement::{Local, Placement, PlacementKind};
