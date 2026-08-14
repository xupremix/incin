//! Type-level shape definitions, dimension arithmetic, and shape verification traits.
//!
//! The `shapes` module is the type-theoretic core of Incin. It contains:
//!
//! * [`dim`] — The [`Dim`] trait, raw typenum static extents, derived extent
//!   specifications, and semantic named-axis tags.
//! * [`shape`] — The structural [`Shape`] algebra, runtime [`ShapeBuf`], and
//!   dynamic-shape adapters.
//! * [`reshape`] — The [`ReshapeShape`] trait for compile-time element-count preservation.
//! * [`idx`] — The [`DimIdx`], [`SliceIdx`], [`Slice`], [`InferDim`], and [`Ellipsis`] types.
//! * [`broadcast`] — The [`BroadcastShape`] trait for verifying broadcasting compatibility.
//! * [`spatial`] — Shape traits for convolution (`Conv2dShape`, `Conv1dShape`) and pooling.
//! * `concat` — Shape traits for verified concatenation along an axis.
//! * `stack` — Shape traits for verified tensor stacking.
/// `arithmetic`.
pub mod arithmetic;
/// `broadcast`.
pub mod broadcast;

/// `buf`.
pub mod buf;
/// Runtime-selected marker types shared by shape-bearing APIs.
pub mod dynamic;
/// `concat`.
pub mod concat;
/// The dimension along which this operation is applied.
pub mod dim;
/// `error`.
pub mod error;
/// `idx`.
pub mod idx;
/// `named`.
pub mod named;
/// Shape proof strength.
pub mod proof;
/// `reshape`.
pub mod reshape;
/// `shape`.
pub mod shape;
/// `shape_ops`.
pub mod shape_ops;
/// `spatial`.
pub mod spatial;
/// `stack`.
pub mod stack;

pub use arithmetic::*;
pub use broadcast::BroadcastShape;
pub use buf::*;
pub use dim::*;
pub use dynamic::*;
pub use error::*;
pub use idx::*;
pub use named::*;
pub use proof::*;
pub use reshape::*;
pub use shape::*;
pub use shape_ops::*;
pub use spatial::*;

/// `prelude`.
pub mod prelude {
    pub use super::arithmetic::*;
    pub use super::broadcast::*;
    pub use super::buf::*;
    pub use super::concat::*;

    pub use super::dim::*;
    pub use super::dynamic::*;
    pub use super::error::*;
    pub use super::idx::*;
    pub use super::named::*;
    pub use super::reshape::*;
    pub use super::shape::*;
    pub use super::shape_ops::*;
    pub use super::spatial::*;
    pub use super::stack::*;
}
