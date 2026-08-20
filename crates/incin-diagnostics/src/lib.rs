//! Typenum-to-decimal diagnostic humanization for Incin tooling.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `humanize` is the public
//! entry-point API (`humanize_diagnostic`, inlay/hover label handling,
//! path-qualifier stripping); `mismatch` is the family of structured
//! shape-mismatch diagnostic parsers (`MatMulMismatch`, `ConcatMismatch`,
//! and friends); `typenum` is the lower-level typenum expression parsing
//! and substitution machinery `humanize` builds on.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod humanize;
mod mismatch;
#[cfg(test)]
mod tests;
mod typenum;

pub use humanize::*;
pub use mismatch::*;
pub use typenum::*;
