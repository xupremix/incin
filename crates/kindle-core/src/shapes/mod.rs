//! Type-level shape definitions, dimension arithmetic, and shape verification traits.
//!
//! The `shapes` module is the type-theoretic core of Kindle. It contains:
//!
//! * [`dim`] — The [`Dim`] trait, `typenum` implementations, [`ProdDim`], and the `symbolic_dim!` macro.
//! * [`shape`] — The [`Shape`], [`DynShape`], [`ConstShape`], and [`PartialDynShape`] traits.
//! * [`reshape`] — The [`ReshapeShape`] trait for compile-time element-count preservation.
//! * [`idx`] — The [`DimIdx`], [`SliceIdx`], [`Slice`], [`InferDim`], and [`Ellipsis`] types.
//! * [`broadcast`] — The [`BroadcastShape`] trait for verifying broadcasting compatibility.
//! * [`spatial`] — Shape traits for convolution (`Conv2dShape`, `Conv1dShape`) and pooling.
//! * `concat` — Shape traits for verified concatenation along an axis.
//! * `stack` — Shape traits for verified tensor stacking.
/// Auto-generated documentation for arithmetic.
pub mod arithmetic;
/// Auto-generated documentation for broadcast.
pub mod broadcast;

/// Auto-generated documentation for concat.
pub mod concat;
/// Auto-generated documentation for dim.
pub mod dim;
/// Auto-generated documentation for idx.
pub mod idx;
/// Auto-generated documentation for named.
pub mod named;
/// Auto-generated documentation for reshape.
pub mod reshape;
/// Auto-generated documentation for shape.
pub mod shape;
/// Auto-generated documentation for shape_ops.
pub mod shape_ops;
/// Auto-generated documentation for spatial.
pub mod spatial;
/// Auto-generated documentation for stack.
pub mod stack;

pub use arithmetic::*;
pub use broadcast::BroadcastShape;
pub use dim::*;
pub use idx::*;
pub use reshape::*;
pub use shape::*;
pub use shape_ops::*;
pub use spatial::*;

/// Auto-generated documentation for prelude.
pub mod prelude {
    pub use super::arithmetic::*;
    pub use super::broadcast::*;

    pub use super::dim::*;
    pub use super::idx::*;
    pub use super::named::*;
    pub use super::shape::*;
}
