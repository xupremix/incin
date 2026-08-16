//! Rank-only shape proofs for runtime and partially-known shapes.

use super::{Dim, DimCons, Dyn, DynShape, Nil, Shape, ShapeBuf};
use core::ops::{Add, Sub};
use typenum::{U1, Unsigned};

/// Generic known-rank runtime shape. The rank is a typenum fact, so rank
/// arithmetic remains in the type system without a generated const-generic
/// rank ladder.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Ranked<R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static>(
    core::marker::PhantomData<R>,
);

/// Rank-preserving transformation for generic known-rank shapes.
pub trait PreserveRank {
    type Output: Shape;
}

impl<R> PreserveRank for Ranked<R>
where
    R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = Ranked<R>;
}

/// Generic known-rank reduction/axis removal.
pub trait RemoveOneRank {
    type Output: Shape;
}

impl<R, ROut> RemoveOneRank for Ranked<R>
where
    R: Unsigned + Sub<U1, Output = ROut> + core::fmt::Debug + Eq + Send + Sync + 'static,
    ROut: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = Ranked<ROut>;
}

/// Generic known-rank insertion/stacking.
pub trait AddOneRank {
    type Output: Shape;
}

#[doc(hidden)]
pub trait ShapeRank {
    type Output: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static;
}

impl ShapeRank for Nil {
    type Output = typenum::U0;
}

impl<H: Dim, T> ShapeRank for DimCons<H, T>
where
    T: ShapeRank,
    <T as ShapeRank>::Output: Add<U1>,
    <<T as ShapeRank>::Output as Add<U1>>::Output:
        Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = <<T as ShapeRank>::Output as Add<U1>>::Output;
}

impl AddOneRank for Nil {
    type Output = Ranked<typenum::U1>;
}

impl<H: Dim, T> AddOneRank for DimCons<H, T>
where
    T: ShapeRank,
    <T as ShapeRank>::Output: Add<U1>,
    <<T as ShapeRank>::Output as Add<U1>>::Output: Unsigned + Add<U1>,
    <<<T as ShapeRank>::Output as Add<U1>>::Output as Add<U1>>::Output:
        Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = Ranked<<<<T as ShapeRank>::Output as Add<U1>>::Output as Add<U1>>::Output>;
}

impl<R, ROut> AddOneRank for Ranked<R>
where
    R: Unsigned + Add<U1, Output = ROut> + core::fmt::Debug + Eq + Send + Sync + 'static,
    ROut: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = Ranked<ROut>;
}

/// Runtime-axis projection for shapes whose rank is known structurally.
/// Extents are erased, but the rank change remains in the public tensor type.
pub trait RuntimeRankProjection: Shape {
    type Keep: Shape;
    type Drop: Shape;
}

impl RuntimeRankProjection for Nil {
    type Keep = Ranked<typenum::U0>;
    type Drop = Ranked<typenum::U0>;
}

impl<H: Dim, T> RuntimeRankProjection for DimCons<H, T>
where
    T: Shape + RuntimeRankProjection,
    T::Keep: AddOneRank,
    <T::Keep as AddOneRank>::Output: Shape,
{
    type Keep = <T::Keep as AddOneRank>::Output;
    type Drop = T::Keep;
}

impl<R, ROut> RuntimeRankProjection for Ranked<R>
where
    R: Unsigned
        + core::ops::Sub<typenum::U1, Output = ROut>
        + core::fmt::Debug
        + Eq
        + Send
        + Sync
        + 'static,
    ROut: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Keep = Ranked<R>;
    type Drop = Ranked<ROut>;
}

impl RuntimeRankProjection for Dyn {
    type Keep = Dyn;
    type Drop = Dyn;
}

impl AddOneRank for Dyn {
    type Output = Dyn;
}

impl<R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static> Shape for Ranked<R> {
    const RANK: Option<usize> = Some(R::USIZE);
    const PROOF: crate::shapes::ProofLevel = crate::shapes::ProofLevel::Mixed;
    const STATIC_NUMEL: Option<usize> = if R::USIZE == 0 { Some(1) } else { None };
    type Arg = ShapeBuf;

    fn resolve(arg: Self::Arg) -> core::result::Result<ShapeBuf, crate::shapes::error::ShapeError> {
        Self::try_from_dims(arg.as_ref()).map(|_| arg)
    }

    fn validate_dims(dims: &[usize]) -> core::result::Result<(), crate::shapes::error::ShapeError> {
        if dims.len() == R::USIZE {
            Ok(())
        } else {
            Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: dims.len(),
            })
        }
    }
}

impl<R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static> DynShape for Ranked<R> {
    fn rank(_: &ShapeBuf) -> usize {
        R::USIZE
    }
}
