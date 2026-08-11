use crate::prelude::Shape;
use crate::shapes::idx::{FromEnd, Here, Next, StaticCursor};
use crate::shapes::{At, Dim, DimCons, Nil, RemoveAt, RemoveFromEnd, ReplaceAt, SwapAt};

/// Unified selector-facing swap operation. Both positive and from-end
/// selectors use the one structural `SwapAt` algebra.
pub trait SwapAxes<Left, Right>: Shape {
    type Output: Shape;

    fn swap_shape(dims: &crate::shapes::ShapeBuf) -> crate::err::Result<crate::shapes::ShapeBuf>
    where
        Left: crate::shapes::idx::StaticCursor,
        Right: crate::shapes::idx::StaticCursor,
    {
        let left = crate::shapes::idx::StaticAxis::<Left>::DEFAULT.normalize(dims.len())?;
        let right = crate::shapes::idx::StaticAxis::<Right>::DEFAULT.normalize(dims.len())?;
        let mut output = dims.clone();
        output.dims_mut().swap(left[0], right[0]);
        Ok(output)
    }
}

impl<S, L, R> SwapAxes<L, R> for S
where
    L: StaticCursor,
    R: StaticCursor,
    S: SwapAt<L, R>,
{
    type Output = <S as SwapAt<L, R>>::Output;
}

/// Structural reduction which removes one axis.
pub trait ReduceAt<Cursor>: Shape {
    type Output: Shape;

    fn reduce_shape(dims: &crate::shapes::ShapeBuf) -> crate::err::Result<crate::shapes::ShapeBuf>
    where
        Cursor: crate::shapes::idx::StaticCursor,
    {
        let axis = crate::shapes::idx::StaticAxis::<Cursor>::DEFAULT.normalize(dims.len())?[0];
        let mut output = dims.as_ref().to_vec();
        output.remove(axis);
        Ok(crate::shapes::ShapeBuf::from_slice(&output))
    }
}

impl<S, Cursor> ReduceAt<Cursor> for S
where
    Cursor: crate::shapes::shape::ForwardCursor,
    S: RemoveAt<Cursor>,
{
    type Output = <S as RemoveAt<Cursor>>::Output;
}

impl<S, Cursor> ReduceAt<FromEnd<Cursor>> for S
where
    Cursor: crate::shapes::shape::ForwardCursor,
    S: RemoveFromEnd<Cursor>,
{
    type Output = <S as RemoveFromEnd<Cursor>>::Output;
}

/// Structural keepdim reduction. Rebinding is owned by the dimension, so a
/// semantic axis name is preserved while its extent becomes one.
pub trait ReduceKeepAt<Cursor>: Shape {
    type Output: Shape;
}

impl<H: Dim, T: Shape> ReduceKeepAt<Here> for DimCons<H, T> {
    type Output = DimCons<H::KeepDim, T>;
}

impl<H: Dim, T: Shape, Cursor> ReduceKeepAt<Next<Cursor>> for DimCons<H, T>
where
    T: ReduceKeepAt<Cursor>,
{
    type Output = DimCons<H, <T as ReduceKeepAt<Cursor>>::Output>;
}

impl<Cursor> ReduceKeepAt<Cursor> for crate::prelude::Dyn {
    type Output = crate::prelude::Dyn;
}
