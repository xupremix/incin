//! Shape-level proof strength used by execution validation.

use core::fmt;

/// How much of an operation's legality the compiler settled.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ProofLevel {
    /// Rank and every semantic dimension constraint came from type-level data.
    Static,
    /// Structure is typed, but named or dynamic dimensions were checked once.
    Mixed,
    /// Rank and all semantic dimensions were checked at runtime.
    Dynamic,
}

impl ProofLevel {
    /// The proof carried by shape `S`.
    #[must_use]
    pub const fn of<S: crate::shapes::Shape>() -> Self {
        S::PROOF
    }

    /// The weaker of two proofs.
    #[must_use]
    pub const fn meet(self, other: Self) -> Self {
        if (self as u8) >= (other as u8) {
            self
        } else {
            other
        }
    }

    /// The level for a known-rank shape, given whether every axis is static.
    #[doc(hidden)]
    #[must_use]
    pub const fn of_ranked(all_axes_static: bool) -> Self {
        if all_axes_static {
            Self::Static
        } else {
            Self::Mixed
        }
    }

    /// Whether the geometry is entirely a compile-time constant.
    #[must_use]
    pub const fn is_static(self) -> bool {
        matches!(self, Self::Static)
    }

    /// Whether the rank was known before runtime.
    #[must_use]
    pub const fn has_static_rank(self) -> bool {
        !matches!(self, Self::Dynamic)
    }
}

impl fmt::Display for ProofLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Static => "static",
            Self::Mixed => "mixed",
            Self::Dynamic => "dynamic",
        })
    }
}
