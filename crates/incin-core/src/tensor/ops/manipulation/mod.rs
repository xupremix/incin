//! Shape manipulation and restructuring operations.
//!
//! This module provides methods to change the logical or physical shape of a tensor
//! without necessarily changing the underlying data. It includes reshaping, transposition,
//! squeezing, flattening, and broadcasting. These operations heavily leverage the
//! compile-time type system to ensure the resulting shapes are strictly valid.

pub mod concat;
pub mod indexing;
pub mod interop;
pub mod reshape;
pub mod selectors;
pub mod transfer;
pub mod transpose;
pub mod vision;

pub use concat::try_stack_tensors;
pub(crate) use reshape::reshape_storage_exact;
pub use selectors::{
    AxisPairSelector, AxisSelectorArg, ConcatSelector, FlattenSelector, ReplaceAxisSelector,
    StackSelector, UnsqueezeSelector,
};
