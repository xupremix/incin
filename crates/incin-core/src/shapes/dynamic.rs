//! Runtime-selected marker types used by shape-bearing public APIs.

use core::fmt;

/// A marker for a runtime-selected shape, dtype, device, or placement.
///
/// The marker is defined in the shapes layer because it is a neutral type-level
/// value shared by tensor metadata and execution-facing APIs. It does not own
/// any tensor storage or runtime state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Dyn(());

impl Dyn {
    #[inline]
    pub(crate) const fn marker() -> Self {
        Self(())
    }
}

impl fmt::Display for Dyn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Dyn")
    }
}
