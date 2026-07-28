//! Compile-time tensor placement typestates.

/// Runtime projection of a compile-time placement.
///
/// Only local placement exists before the distributed track. The enum is
/// non-exhaustive so DST-003 can add replicated, sharded, partial, and pipeline
/// variants without creating a second runtime vocabulary.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlacementKind {
    #[default]
    Local,
}

/// Compile-time placement carried by backend storage.
pub trait Placement: 'static + Clone + core::fmt::Debug + Send + Sync {
    fn kind() -> PlacementKind;
}

/// A tensor held by one backend device.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Local;

impl Placement for Local {
    fn kind() -> PlacementKind {
        PlacementKind::Local
    }
}
