//! Type-level shape definitions, dimension arithmetic, and shape verification traits.
//!
//! The `shapes` module is the type-theoretic core of Incin. It contains:
//!
//! * [`dim`] - The [`Dim`] trait, raw typenum static extents, derived extent
//!   specifications, and semantic named-axis tags.
//! * [`shape`] - The structural [`Shape`] algebra, runtime [`ShapeBuf`], and
//!   dynamic-shape adapters.
//! * [`reshape`] - The [`ReshapeShape`] trait for compile-time element-count preservation.
//! * [`idx`] - The [`DimIdx`], [`SliceIdx`], [`Slice`], [`InferDim`], and [`Ellipsis`] types.
//! * [`broadcast`] - The [`BroadcastShape`] trait for verifying broadcasting compatibility.
//! * [`spatial`] - Shape traits for convolution (`Conv2dShape`, `Conv1dShape`) and pooling.
//! * [`mod@concat`] - Shape traits for verified concatenation along an axis.
//! * [`stack`] - Shape traits for verified tensor stacking.
mod arithmetic;
/// Broadcast compatibility proofs for tensor operations.
pub mod broadcast;
mod buf;
/// Compile-time and runtime proofs for verified shape concatenation.
pub mod concat;
/// The dimension along which this operation is applied.
pub mod dim;
/// Runtime-selected marker types shared by shape-bearing APIs.
pub mod dynamic;
/// Errors produced while validating or transforming shapes.
pub mod error;
/// Type-level axis, slice, and reshape selectors.
pub mod idx;
/// Named dimensions and their compile-time compatibility rules.
pub mod named;
mod proof;
/// Rank-only shape proofs for runtime and partially-known shapes.
pub mod rank;
/// Compile-time element-count-preserving reshape proofs.
pub mod reshape;
/// Structural shapes, runtime shape values, and shape metadata.
pub mod shape;
/// Type-level operations that transform structural shapes.
pub mod shape_ops;
mod shape_utils;
/// Shape rules for convolution, pooling, and related spatial operations.
pub mod spatial;
/// Compile-time proofs for stacking tensors along a new axis.
pub mod stack;

pub use arithmetic::{ConvOutDim, FlatDim};
pub use broadcast::BroadcastShape;
pub use buf::{INLINE_RANK, ShapeBuf, StrideBuf};
pub use dim::*;
pub use dynamic::*;
pub use error::*;
pub use idx::*;
pub use named::*;
pub use proof::ProofLevel;
pub use rank::*;
pub use reshape::*;
pub use shape::*;
pub use shape_ops::*;
pub use shape_utils::Scalar;
pub use spatial::*;

/// `prelude`.
pub mod prelude {
    pub use super::arithmetic::{ConvOutDim, FlatDim};
    pub use super::broadcast::*;
    pub use super::buf::{INLINE_RANK, ShapeBuf, StrideBuf};
    pub use super::concat::*;

    pub use super::dim::*;
    pub use super::dynamic::*;
    pub use super::error::*;
    pub use super::named::*;
    pub use super::rank::*;
    pub use super::reshape::*;
    pub use super::shape::*;
    pub use super::shape_ops::*;
    pub use super::shape_utils::Scalar;
    pub use super::spatial::*;
    pub use super::stack::*;
}
