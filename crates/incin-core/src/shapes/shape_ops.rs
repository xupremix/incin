use crate::shapes::Shape;
use crate::shapes::broadcast::ReverseShape;
use crate::shapes::idx::{FromEnd, Here, Next};
use crate::shapes::{Dim, DimCons, Dyn, RemoveAt, RemoveFromEnd};

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
        let left = left
            .first()
            .copied()
            .ok_or(crate::shapes::ShapeError::InvalidAxis {
                axis: 0,
                rank: dims.len(),
            })?;
        let right = right
            .first()
            .copied()
            .ok_or(crate::shapes::ShapeError::InvalidAxis {
                axis: 0,
                rank: dims.len(),
            })?;
        if left >= dims.len() || right >= dims.len() {
            return Err(crate::shapes::ShapeError::InvalidAxis {
                axis: left.max(right),
                rank: dims.len(),
            }
            .into());
        }
        let mut output = dims.clone();
        output.dims_mut().swap(left, right);
        Ok(output)
    }
}

impl<S, L, R> SwapAxes<L, R> for S
where
    L: crate::shapes::idx::StaticCursor + crate::shapes::shape::ForwardCursor,
    R: crate::shapes::idx::StaticCursor + crate::shapes::shape::ForwardCursor,
    S: crate::shapes::SwapAt<L, R>,
{
    type Output = <S as crate::shapes::SwapAt<L, R>>::Output;
}

/// Structural reduction which removes one axis.
pub trait ReduceAt<Cursor>: Shape {
    type Output: Shape;

    fn reduce_shape(dims: &crate::shapes::ShapeBuf) -> crate::err::Result<crate::shapes::ShapeBuf>
    where
        Cursor: crate::shapes::idx::AxisCursor,
    {
        let axis = crate::shapes::idx::AxisSelector::new(&[Cursor::INDEX])
            .normalize(dims.len())?
            .first()
            .copied()
            .ok_or(crate::shapes::ShapeError::InvalidAxis {
                axis: 0,
                rank: dims.len(),
            })?;
        if axis >= dims.len() {
            return Err(crate::shapes::ShapeError::InvalidAxis {
                axis,
                rank: dims.len(),
            }
            .into());
        }
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

impl ReduceKeepAt<Here> for crate::shapes::Dyn {
    type Output = crate::shapes::Dyn;
}

impl<Cursor> ReduceKeepAt<Next<Cursor>> for Dyn {
    type Output = Dyn;
}

impl<Cursor> ReduceKeepAt<FromEnd<Cursor>> for Dyn {
    type Output = Dyn;
}

impl<H: Dim, T: Shape> ReduceKeepAt<FromEnd<Here>> for DimCons<H, T>
where
    DimCons<H, T>: ReverseShape,
    <DimCons<H, T> as ReverseShape>::Output: ReduceKeepAt<Here>,
    <<DimCons<H, T> as ReverseShape>::Output as ReduceKeepAt<Here>>::Output: ReverseShape,
{
    type Output = <<<DimCons<H, T> as ReverseShape>::Output as ReduceKeepAt<Here>>::Output
        as ReverseShape>::Output;
}

impl<H: Dim, T: Shape, Cursor> ReduceKeepAt<FromEnd<Next<Cursor>>> for DimCons<H, T>
where
    DimCons<H, T>: ReverseShape,
    <DimCons<H, T> as ReverseShape>::Output: ReduceKeepAt<Next<Cursor>>,
    <<DimCons<H, T> as ReverseShape>::Output as ReduceKeepAt<Next<Cursor>>>::Output: ReverseShape,
{
    type Output = <<<DimCons<H, T> as ReverseShape>::Output as ReduceKeepAt<Next<Cursor>>>::Output
        as ReverseShape>::Output;
}
