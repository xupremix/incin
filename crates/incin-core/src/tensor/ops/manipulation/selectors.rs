//! Axis and shape selector traits and implementations.

use crate::err::Result;
use crate::shapes::RuntimeRankProjection;
use crate::shapes::idx::ReverseAxis;
use crate::shapes::idx::StaticCursor;
use crate::shapes::{Dyn, DynShape, Shape};
use crate::shapes::{FlattenAt, FlattenFromEndAt, SwapAxes, SwapFromEndAt};
use alloc::string::ToString;

/// Public axis-pair selector used by transpose. Static pairs retain the exact
/// `SwapAxes` output, while runtime and named pairs retain the input rank.
pub trait AxisPairSelector<S: Shape, L, R> {
    /// The resulting shape of [`AxisPairSelector`].
    type Output: Shape;

    /// Resolves both axes of the pair against rank.
    fn resolve(pair: &(L, R), rank: usize) -> Result<(usize, usize)>;
}

/// Public static selector pair for flattening an inclusive axis range.
pub trait FlattenSelector<S: Shape, L, R> {
    /// The resulting shape of [`FlattenSelector`].
    type Output: Shape;

    /// Resolves both axes of the pair against rank.
    fn resolve(pair: &(L, R), rank: usize) -> Result<(usize, usize)>;
}

/// Selects the axis used by a two-tensor concatenation.
pub trait ConcatSelector<S: Shape, S2: Shape> {
    /// The resulting shape of [`ConcatSelector`].
    type Output: Shape;

    /// Resolves this selector against a concrete rank.
    fn resolve(&self, rank: usize) -> Result<usize>;
}

/// Selects the insertion axis for stacking two tensors.
pub trait StackSelector<S: Shape> {
    /// The resulting shape of [`StackSelector`].
    type Output: Shape;

    /// Resolves this selector against a concrete rank.
    fn resolve(&self, rank: usize) -> Result<usize>;
}

/// Selects one axis for operations whose output geometry does not depend on
/// the selector's proof type.
pub trait AxisSelectorArg<S: Shape> {
    /// Resolves this selector against a concrete rank.
    fn resolve(&self, rank: usize) -> Result<usize>;
}

/// Selects an axis for an operation that replaces its extent at runtime.
/// Static selectors preserve all unaffected dimensions; runtime selectors
/// preserve the input rank.
pub trait ReplaceAxisSelector<S: Shape> {
    /// The resulting shape of [`ReplaceAxisSelector`].
    type Output: Shape;

    /// Resolves this selector against a concrete rank.
    fn resolve(&self, rank: usize) -> Result<usize>;
}

impl<S, C> ReplaceAxisSelector<S> for crate::shapes::idx::ForwardAxis<C>
where
    S: Shape + DynShape + crate::shapes::ReplaceAt<C, usize>,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as crate::shapes::ReplaceAt<C, usize>>::Output: Shape,
{
    type Output = <S as crate::shapes::ReplaceAt<C, usize>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".into())
        })
    }
}

impl<S, C> ReplaceAxisSelector<S> for ReverseAxis<C>
where
    S: Shape + DynShape + crate::shapes::ReplaceAt<crate::shapes::idx::FromEnd<C>, usize>,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as crate::shapes::ReplaceAt<crate::shapes::idx::FromEnd<C>, usize>>::Output: Shape,
{
    type Output = <S as crate::shapes::ReplaceAt<crate::shapes::idx::FromEnd<C>, usize>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".into())
        })
    }
}

impl<S> ReplaceAxisSelector<S> for isize
where
    S: Shape + RuntimeRankProjection,
{
    type Output = S::Keep;

    fn resolve(&self, rank: usize) -> Result<usize> {
        crate::shapes::idx::AxisSelector::new(&[*self])
            .normalize(rank)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::err::Error::Msg("runtime axis selector resolved to no axis".into())
            })
    }
}

impl<S, Tag> ReplaceAxisSelector<S> for crate::shapes::idx::NamedAxisSelector<Tag>
where
    S: Shape + RuntimeRankProjection + crate::shapes::idx::NamedAxisLookup<Tag>,
    Tag: crate::shapes::AxisTag,
{
    type Output = S::Keep;

    fn resolve(&self, _rank: usize) -> Result<usize> {
        self.resolve::<S>()
    }
}

impl<S, C> AxisSelectorArg<S> for crate::shapes::idx::ForwardAxis<C>
where
    S: Shape,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
{
    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".into())
        })
    }
}

impl<S, C> AxisSelectorArg<S> for ReverseAxis<C>
where
    S: Shape,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
{
    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".into())
        })
    }
}

impl<S> AxisSelectorArg<S> for isize
where
    S: Shape,
{
    fn resolve(&self, rank: usize) -> Result<usize> {
        crate::shapes::idx::AxisSelector::new(&[*self])
            .normalize(rank)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::err::Error::Msg("runtime axis selector resolved to no axis".into())
            })
    }
}

impl<S, Tag> AxisSelectorArg<S> for crate::shapes::idx::NamedAxisSelector<Tag>
where
    S: Shape + crate::shapes::idx::NamedAxisLookup<Tag>,
    Tag: crate::shapes::AxisTag,
{
    fn resolve(&self, _rank: usize) -> Result<usize> {
        self.resolve::<S>()
    }
}

/// Selects the insertion axis for unsqueeze while retaining the strongest
/// shape proof available from the input and selector.
pub trait UnsqueezeSelector<S: Shape> {
    /// The resulting shape of [`UnsqueezeSelector`].
    type Output: Shape;

    /// Resolves this selector against a concrete rank.
    fn resolve(&self, rank: usize) -> Result<usize>;
}

impl<S, C> UnsqueezeSelector<S> for crate::shapes::idx::ForwardAxis<C>
where
    S: Shape + DynShape + crate::shapes::InsertAxis<C, typenum::U1>,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as crate::shapes::InsertAxis<C, typenum::U1>>::Output: Shape,
{
    type Output = <S as crate::shapes::InsertAxis<C, typenum::U1>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank + 1)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static unsqueeze axis resolved to no axis".into())
        })
    }
}

impl<S, C> UnsqueezeSelector<S> for ReverseAxis<C>
where
    S: Shape + DynShape + crate::shapes::InsertAxis<crate::shapes::idx::FromEnd<C>, typenum::U1>,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as crate::shapes::InsertAxis<crate::shapes::idx::FromEnd<C>, typenum::U1>>::Output: Shape,
{
    type Output =
        <S as crate::shapes::InsertAxis<crate::shapes::idx::FromEnd<C>, typenum::U1>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank + 1)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static unsqueeze axis resolved to no axis".into())
        })
    }
}

impl<S> UnsqueezeSelector<S> for isize
where
    S: Shape + crate::shapes::shape::AddOneRank,
{
    type Output = <S as crate::shapes::shape::AddOneRank>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        crate::shapes::idx::AxisSelector::new(&[*self])
            .normalize(rank + 1)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::err::Error::Msg("runtime unsqueeze axis resolved to no axis".into())
            })
    }
}

impl<S, Tag> UnsqueezeSelector<S> for crate::shapes::idx::NamedAxisSelector<Tag>
where
    S: Shape + crate::shapes::shape::AddOneRank + crate::shapes::idx::NamedAxisLookup<Tag>,
    Tag: crate::shapes::AxisTag,
{
    type Output = <S as crate::shapes::shape::AddOneRank>::Output;

    fn resolve(&self, _rank: usize) -> Result<usize> {
        self.resolve::<S>()
    }
}

impl<S, C> StackSelector<S> for crate::shapes::idx::ForwardAxis<C>
where
    S: Shape + DynShape + crate::shapes::stack::StackShape<C>,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as crate::shapes::stack::StackShape<C>>::Output: Shape,
{
    type Output = <S as crate::shapes::stack::StackShape<C>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank + 1)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static stack axis selector resolved to no axis".into())
        })
    }
}

impl<S, C> StackSelector<S> for ReverseAxis<C>
where
    S: Shape + DynShape + crate::shapes::stack::StackShape<crate::shapes::idx::FromEnd<C>>,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as crate::shapes::stack::StackShape<crate::shapes::idx::FromEnd<C>>>::Output: Shape,
{
    type Output = <S as crate::shapes::stack::StackShape<crate::shapes::idx::FromEnd<C>>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank + 1)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static stack axis selector resolved to no axis".into())
        })
    }
}

impl<S> StackSelector<S> for isize
where
    S: Shape + crate::shapes::shape::AddOneRank,
{
    type Output = <S as crate::shapes::shape::AddOneRank>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        crate::shapes::idx::AxisSelector::new(&[*self])
            .normalize(rank + 1)?
            .into_iter()
            .next()
            .ok_or_else(|| crate::err::Error::Msg("runtime stack axis resolved to no axis".into()))
    }
}

impl<S, Tag> StackSelector<S> for crate::shapes::idx::NamedAxisSelector<Tag>
where
    S: Shape + crate::shapes::shape::AddOneRank + crate::shapes::idx::NamedAxisLookup<Tag>,
    Tag: crate::shapes::AxisTag,
{
    type Output = <S as crate::shapes::shape::AddOneRank>::Output;

    fn resolve(&self, _rank: usize) -> Result<usize> {
        self.resolve::<S>()
    }
}

impl<S, S2, C> ConcatSelector<S, S2> for crate::shapes::idx::ForwardAxis<C>
where
    S: Shape + DynShape + crate::shapes::concat::ConcatShape<S2, C>,
    S2: Shape,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as crate::shapes::concat::ConcatShape<S2, C>>::Output: Shape,
{
    type Output = <S as crate::shapes::concat::ConcatShape<S2, C>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".to_string())
        })
    }
}

impl<S, S2, C> ConcatSelector<S, S2> for ReverseAxis<C>
where
    S: Shape + DynShape + crate::shapes::concat::ConcatShape<S2, crate::shapes::idx::FromEnd<C>>,
    S2: Shape,
    C: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as crate::shapes::concat::ConcatShape<S2, crate::shapes::idx::FromEnd<C>>>::Output: Shape,
{
    type Output =
        <S as crate::shapes::concat::ConcatShape<S2, crate::shapes::idx::FromEnd<C>>>::Output;

    fn resolve(&self, rank: usize) -> Result<usize> {
        self.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".to_string())
        })
    }
}

impl<S, S2> ConcatSelector<S, S2> for isize
where
    S: Shape + RuntimeRankProjection,
    S2: Shape,
{
    type Output = S::Keep;

    fn resolve(&self, rank: usize) -> Result<usize> {
        crate::shapes::idx::AxisSelector::new(&[*self])
            .normalize(rank)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::err::Error::Msg("runtime axis selector resolved to no axis".into())
            })
    }
}

impl<S, S2, Tag> ConcatSelector<S, S2> for crate::shapes::idx::NamedAxisSelector<Tag>
where
    S: Shape + RuntimeRankProjection + crate::shapes::idx::NamedAxisLookup<Tag>,
    S2: Shape,
    Tag: crate::shapes::AxisTag,
{
    type Output = S::Keep;

    fn resolve(&self, _rank: usize) -> Result<usize> {
        self.resolve::<S>()
    }
}

impl<S, L, R>
    FlattenSelector<S, crate::shapes::idx::ForwardAxis<L>, crate::shapes::idx::ForwardAxis<R>>
    for ()
where
    S: Shape + DynShape + FlattenAt<L, R>,
    L: StaticCursor + crate::shapes::shape::ForwardCursor,
    R: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as FlattenAt<L, R>>::Output: Shape + DynShape,
{
    type Output = <S as FlattenAt<L, R>>::Output;

    fn resolve(
        pair: &(
            crate::shapes::idx::ForwardAxis<L>,
            crate::shapes::idx::ForwardAxis<R>,
        ),
        rank: usize,
    ) -> Result<(usize, usize)> {
        let start = pair.0.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".to_string())
        })?;
        let end = pair.1.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".to_string())
        })?;
        Ok((start, end))
    }
}

impl<S, L, R>
    FlattenSelector<S, crate::shapes::idx::ForwardAxis<L>, crate::shapes::idx::ReverseAxis<R>>
    for ()
where
    S: Shape + DynShape + FlattenFromEndAt<L, crate::shapes::idx::FromEnd<R>>,
    L: StaticCursor + crate::shapes::shape::ForwardCursor,
    R: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as FlattenFromEndAt<L, crate::shapes::idx::FromEnd<R>>>::Output: Shape + DynShape,
{
    type Output = <S as FlattenFromEndAt<L, crate::shapes::idx::FromEnd<R>>>::Output;

    fn resolve(
        pair: &(
            crate::shapes::idx::ForwardAxis<L>,
            crate::shapes::idx::ReverseAxis<R>,
        ),
        rank: usize,
    ) -> Result<(usize, usize)> {
        let start = pair.0.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".to_string())
        })?;
        let end = pair.1.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".to_string())
        })?;
        Ok((start, end))
    }
}

impl<S, L, R>
    FlattenSelector<S, crate::shapes::idx::ReverseAxis<L>, crate::shapes::idx::ReverseAxis<R>>
    for ()
where
    S: Shape
        + DynShape
        + FlattenFromEndAt<crate::shapes::idx::FromEnd<L>, crate::shapes::idx::FromEnd<R>>,
    L: StaticCursor + crate::shapes::shape::ForwardCursor,
    R: StaticCursor + crate::shapes::shape::ForwardCursor,
    <S as FlattenFromEndAt<crate::shapes::idx::FromEnd<L>, crate::shapes::idx::FromEnd<R>>>::Output:
        Shape + DynShape,
{
    type Output = <S as FlattenFromEndAt<
        crate::shapes::idx::FromEnd<L>,
        crate::shapes::idx::FromEnd<R>,
    >>::Output;

    fn resolve(
        pair: &(
            crate::shapes::idx::ReverseAxis<L>,
            crate::shapes::idx::ReverseAxis<R>,
        ),
        rank: usize,
    ) -> Result<(usize, usize)> {
        let start = pair.0.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".to_string())
        })?;
        let end = pair.1.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".to_string())
        })?;
        Ok((start, end))
    }
}

impl<S> FlattenSelector<S, isize, isize> for ()
where
    S: Shape + DynShape,
{
    type Output = Dyn;

    fn resolve(pair: &(isize, isize), rank: usize) -> Result<(usize, usize)> {
        let axes = crate::shapes::idx::AxisSelector::new(&[pair.0, pair.1]).normalize(rank)?;
        Ok((axes[0], axes[1]))
    }
}

pub(crate) trait AxisValue {
    fn resolve_axis(&self, rank: usize) -> Result<usize>;
}

impl<C: StaticCursor + crate::shapes::shape::ForwardCursor> AxisValue
    for crate::shapes::idx::ForwardAxis<C>
{
    fn resolve_axis(&self, rank: usize) -> Result<usize> {
        self.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".into())
        })
    }
}

impl<C: StaticCursor> AxisValue for crate::shapes::idx::ReverseAxis<C> {
    fn resolve_axis(&self, rank: usize) -> Result<usize> {
        self.normalize(rank)?.into_iter().next().ok_or_else(|| {
            crate::err::Error::Msg("static axis selector resolved to no axis".into())
        })
    }
}

fn resolve_axis_pair<L: AxisValue, R: AxisValue>(
    pair: &(L, R),
    rank: usize,
) -> Result<(usize, usize)> {
    Ok((pair.0.resolve_axis(rank)?, pair.1.resolve_axis(rank)?))
}

impl<S, L, R>
    AxisPairSelector<S, crate::shapes::idx::ForwardAxis<L>, crate::shapes::idx::ForwardAxis<R>>
    for ()
where
    L: StaticCursor + crate::shapes::shape::ForwardCursor,
    R: StaticCursor + crate::shapes::shape::ForwardCursor,
    S: Shape + DynShape + SwapAxes<L, R>,
    <S as SwapAxes<L, R>>::Output: Shape + DynShape,
{
    type Output = <S as SwapAxes<L, R>>::Output;

    fn resolve(
        pair: &(
            crate::shapes::idx::ForwardAxis<L>,
            crate::shapes::idx::ForwardAxis<R>,
        ),
        rank: usize,
    ) -> Result<(usize, usize)> {
        resolve_axis_pair(pair, rank)
    }
}

impl<S, L, R>
    AxisPairSelector<S, crate::shapes::idx::ForwardAxis<L>, crate::shapes::idx::ReverseAxis<R>>
    for ()
where
    L: StaticCursor + crate::shapes::shape::ForwardCursor,
    R: StaticCursor + crate::shapes::shape::ForwardCursor,
    S: Shape + DynShape + SwapFromEndAt<L, crate::shapes::idx::FromEnd<R>>,
    <S as SwapFromEndAt<L, crate::shapes::idx::FromEnd<R>>>::Output: Shape + DynShape,
{
    type Output = <S as SwapFromEndAt<L, crate::shapes::idx::FromEnd<R>>>::Output;

    fn resolve(
        pair: &(
            crate::shapes::idx::ForwardAxis<L>,
            crate::shapes::idx::ReverseAxis<R>,
        ),
        rank: usize,
    ) -> Result<(usize, usize)> {
        resolve_axis_pair(pair, rank)
    }
}

impl<S, L, R>
    AxisPairSelector<S, crate::shapes::idx::ReverseAxis<L>, crate::shapes::idx::ForwardAxis<R>>
    for ()
where
    L: StaticCursor + crate::shapes::shape::ForwardCursor,
    R: StaticCursor + crate::shapes::shape::ForwardCursor,
    S: Shape + DynShape + SwapFromEndAt<crate::shapes::idx::FromEnd<L>, R>,
    <S as SwapFromEndAt<crate::shapes::idx::FromEnd<L>, R>>::Output: Shape + DynShape,
{
    type Output = <S as SwapFromEndAt<crate::shapes::idx::FromEnd<L>, R>>::Output;

    fn resolve(
        pair: &(
            crate::shapes::idx::ReverseAxis<L>,
            crate::shapes::idx::ForwardAxis<R>,
        ),
        rank: usize,
    ) -> Result<(usize, usize)> {
        resolve_axis_pair(pair, rank)
    }
}

impl<S, L, R>
    AxisPairSelector<S, crate::shapes::idx::ReverseAxis<L>, crate::shapes::idx::ReverseAxis<R>>
    for ()
where
    L: StaticCursor + crate::shapes::shape::ForwardCursor,
    R: StaticCursor + crate::shapes::shape::ForwardCursor,
    S: Shape
        + DynShape
        + SwapFromEndAt<crate::shapes::idx::FromEnd<L>, crate::shapes::idx::FromEnd<R>>,
    <S as SwapFromEndAt<crate::shapes::idx::FromEnd<L>, crate::shapes::idx::FromEnd<R>>>::Output:
        Shape + DynShape,
{
    type Output = <S as SwapFromEndAt<
        crate::shapes::idx::FromEnd<L>,
        crate::shapes::idx::FromEnd<R>,
    >>::Output;

    fn resolve(
        pair: &(
            crate::shapes::idx::ReverseAxis<L>,
            crate::shapes::idx::ReverseAxis<R>,
        ),
        rank: usize,
    ) -> Result<(usize, usize)> {
        resolve_axis_pair(pair, rank)
    }
}

impl<S> AxisPairSelector<S, isize, isize> for ()
where
    S: Shape + RuntimeRankProjection,
    S::Keep: Shape + DynShape,
{
    type Output = S::Keep;

    fn resolve(pair: &(isize, isize), rank: usize) -> Result<(usize, usize)> {
        let axes = crate::shapes::idx::AxisSelector::new(&[pair.0, pair.1]).normalize(rank)?;
        Ok((axes[0], axes[1]))
    }
}

impl<S, L, R>
    AxisPairSelector<
        S,
        crate::shapes::idx::NamedAxisSelector<L>,
        crate::shapes::idx::NamedAxisSelector<R>,
    > for ()
where
    S: Shape
        + DynShape
        + RuntimeRankProjection
        + crate::shapes::idx::NamedAxisLookup<L>
        + crate::shapes::idx::NamedAxisLookup<R>,
    L: crate::shapes::AxisTag,
    R: crate::shapes::AxisTag,
    S::Keep: Shape + DynShape,
{
    type Output = S::Keep;

    fn resolve(
        pair: &(
            crate::shapes::idx::NamedAxisSelector<L>,
            crate::shapes::idx::NamedAxisSelector<R>,
        ),
        _rank: usize,
    ) -> Result<(usize, usize)> {
        Ok((pair.0.resolve::<S>()?, pair.1.resolve::<S>()?))
    }
}
