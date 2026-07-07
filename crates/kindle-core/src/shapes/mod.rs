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
pub mod arithmetic;
pub mod broadcast;

pub mod concat;
pub mod dim;
pub mod idx;
pub mod named;
pub mod reshape;
pub mod shape;
pub mod shape_ops;
pub mod spatial;
pub mod stack;

pub use arithmetic::*;
pub use broadcast::BroadcastShape;
pub use dim::*;
pub use idx::*;
pub use reshape::*;
pub use shape::*;
pub use shape_ops::*;
pub use spatial::*;

pub mod prelude {
    pub use super::arithmetic::*;
    pub use super::broadcast::*;

    pub use super::dim::*;
    pub use super::named::*;
    pub use super::shape::*;
}
