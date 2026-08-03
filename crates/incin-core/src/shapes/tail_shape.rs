use crate::prelude::{Dim, DynShape, EndsWith, HasChannels1D, HasChannels2D, Shape};
use alloc::vec::Vec;
use core::fmt::Debug;

/// Represents a tensor shape with a variable leading rank (`..`) and fixed trailing axes (`Tail`).
///
/// For example, `s![.., 128]` expands to `TailShape<(U128,)>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailShape<Tail: DynShape> {
    _marker: core::marker::PhantomData<Tail>,
}

/// The runtime field storage for a `TailShape<Tail>`.
/// Stores the combined full dimensions as a `Vec<usize>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailShapeField<Tail: DynShape> {
    /// Dynamic leading dimensions (e.g. batch/sequence sizes).
    pub leading_dims: Vec<usize>,
    /// Stored field for the fixed trailing dimensions.
    pub tail_field: Tail::Field,
}

impl<Tail: DynShape<Field: Default>> Shape for TailShape<Tail>
where
    Tail::Arg: Default,
{
    type Arg = Vec<usize>;
    type Field = TailShapeField<Tail>;
    type Dims = Vec<usize>;

    fn init(arg: Self::Arg) -> Self::Field {
        let default_tail = Tail::init(<Tail::Arg as Default>::default());
        let tail_dims = Tail::dims(&default_tail);
        let tail_rank = tail_dims.as_ref().len();

        if arg.len() >= tail_rank {
            let split_idx = arg.len() - tail_rank;
            let leading = arg[..split_idx].to_vec();
            let tail_slice = &arg[split_idx..];
            if let Some(tail_field) = Tail::from_dyn(tail_slice) {
                return TailShapeField {
                    leading_dims: leading,
                    tail_field,
                };
            }
        }

        TailShapeField {
            leading_dims: arg,
            tail_field: default_tail,
        }
    }

    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        let default_tail = Tail::init(<Tail::Arg as Default>::default());
        let tail_dims = Tail::dims(&default_tail);
        let tail_rank = tail_dims.as_ref().len();
        if dims.len() < tail_rank {
            return None;
        }
        let tail_slice = &dims[dims.len() - tail_rank..];
        let tail_field = Tail::from_dyn(tail_slice)?;
        let leading_len = dims.len() - tail_rank;
        Some(TailShapeField {
            leading_dims: dims[..leading_len].to_vec(),
            tail_field,
        })
    }
    fn dims(shape: &Self::Field) -> Self::Dims {
        let mut full = shape.leading_dims.clone();
        let tail_dims = Tail::dims(&shape.tail_field);
        full.extend_from_slice(tail_dims.as_ref());
        full
    }
}

impl<Tail: DynShape<Field: Default>> DynShape for TailShape<Tail>
where
    Tail::Arg: Default,
{
    fn rank(shape: &Self::Field) -> usize {
        let tail_dims = Tail::dims(&shape.tail_field);
        shape.leading_dims.len() + tail_dims.as_ref().len()
    }
}

impl<Tail: DynShape<Field: Default>, D: Dim> EndsWith<D> for TailShape<Tail>
where
    Tail: EndsWith<D>,
    Tail::Arg: Default,
{
}

impl<Tail: DynShape<Field: Default>, D: Dim> HasChannels1D<D> for TailShape<Tail>
where
    Tail: HasChannels1D<D>,
    Tail::Arg: Default,
{
}

impl<Tail: DynShape<Field: Default>, D: Dim> HasChannels2D<D> for TailShape<Tail>
where
    Tail: HasChannels2D<D>,
    Tail::Arg: Default,
{
}

/// Represents a tensor shape with fixed leading axes (`Head`) and variable trailing rank (`..`).
///
/// For example, `s![128, ..]` expands to `HeadShape<(U128,)>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadShape<Head: DynShape> {
    _marker: core::marker::PhantomData<Head>,
}

/// The runtime field storage for a `HeadShape<Head>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadShapeField<Head: DynShape> {
    /// Stored field for the fixed leading dimensions.
    pub head_field: Head::Field,
    /// Dynamic trailing dimensions.
    pub trailing_dims: Vec<usize>,
}

impl<Head: DynShape<Field: Default>> Shape for HeadShape<Head>
where
    Head::Arg: Default,
{
    type Arg = Vec<usize>;
    type Field = HeadShapeField<Head>;
    type Dims = Vec<usize>;

    fn init(arg: Self::Arg) -> Self::Field {
        let default_head = Head::init(<Head::Arg as Default>::default());
        let head_dims = Head::dims(&default_head);
        let head_rank = head_dims.as_ref().len();

        if arg.len() >= head_rank {
            let head_slice = &arg[..head_rank];
            let trailing = arg[head_rank..].to_vec();
            if let Some(head_field) = Head::from_dyn(head_slice) {
                return HeadShapeField {
                    head_field,
                    trailing_dims: trailing,
                };
            }
        }

        HeadShapeField {
            head_field: default_head,
            trailing_dims: arg,
        }
    }

    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        let default_head = Head::init(<Head::Arg as Default>::default());
        let head_dims = Head::dims(&default_head);
        let head_rank = head_dims.as_ref().len();
        if dims.len() < head_rank {
            return None;
        }
        let head_slice = &dims[..head_rank];
        let head_field = Head::from_dyn(head_slice)?;
        Some(HeadShapeField {
            head_field,
            trailing_dims: dims[head_rank..].to_vec(),
        })
    }
    fn dims(shape: &Self::Field) -> Self::Dims {
        let head_dims = Head::dims(&shape.head_field);
        let mut full = head_dims.as_ref().to_vec();
        full.extend_from_slice(&shape.trailing_dims);
        full
    }
}

impl<Head: DynShape<Field: Default>> DynShape for HeadShape<Head>
where
    Head::Arg: Default,
{
    fn rank(shape: &Self::Field) -> usize {
        let head_dims = Head::dims(&shape.head_field);
        head_dims.as_ref().len() + shape.trailing_dims.len()
    }
}

impl<Head: DynShape<Field: Default>, D: Dim> EndsWith<D> for HeadShape<Head> where Head::Arg: Default
{}
impl<Head: DynShape<Field: Default>, D: Dim> HasChannels1D<D> for HeadShape<Head> where
    Head::Arg: Default
{
}
impl<Head: DynShape<Field: Default>, D: Dim> HasChannels2D<D> for HeadShape<Head> where
    Head::Arg: Default
{
}

/// Represents a tensor shape with fixed leading axes (`Head`), variable middle rank (`..`), and fixed trailing axes (`Tail`).
///
/// For example, `s![32, .., 128]` expands to `SpanShape<(U32,), (U128,)>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanShape<Head: DynShape, Tail: DynShape> {
    _marker: core::marker::PhantomData<(Head, Tail)>,
}

/// The runtime field storage for a `SpanShape<Head, Tail>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanShapeField<Head: DynShape, Tail: DynShape> {
    /// Stored field for the fixed leading dimensions.
    pub head_field: Head::Field,
    /// Dynamic middle dimensions.
    pub middle_dims: Vec<usize>,
    /// Stored field for the fixed trailing dimensions.
    pub tail_field: Tail::Field,
}

impl<Head: DynShape<Field: Default>, Tail: DynShape<Field: Default>> Shape for SpanShape<Head, Tail>
where
    Head::Arg: Default,
    Tail::Arg: Default,
{
    type Arg = Vec<usize>;
    type Field = SpanShapeField<Head, Tail>;
    type Dims = Vec<usize>;

    fn init(arg: Self::Arg) -> Self::Field {
        let default_head = Head::init(<Head::Arg as Default>::default());
        let head_dims = Head::dims(&default_head);
        let head_rank = head_dims.as_ref().len();

        let default_tail = Tail::init(<Tail::Arg as Default>::default());
        let tail_dims = Tail::dims(&default_tail);
        let tail_rank = tail_dims.as_ref().len();

        if arg.len() >= head_rank + tail_rank {
            let head_slice = &arg[..head_rank];
            let middle = arg[head_rank..arg.len() - tail_rank].to_vec();
            let tail_slice = &arg[arg.len() - tail_rank..];

            let head_field = Head::from_dyn(head_slice).unwrap_or(default_head);
            let tail_field = Tail::from_dyn(tail_slice).unwrap_or(default_tail);

            return SpanShapeField {
                head_field,
                middle_dims: middle,
                tail_field,
            };
        }

        SpanShapeField {
            head_field: default_head,
            middle_dims: Vec::new(),
            tail_field: default_tail,
        }
    }

    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        let default_head = Head::init(<Head::Arg as Default>::default());
        let head_dims = Head::dims(&default_head);
        let head_rank = head_dims.as_ref().len();

        let default_tail = Tail::init(<Tail::Arg as Default>::default());
        let tail_dims = Tail::dims(&default_tail);
        let tail_rank = tail_dims.as_ref().len();

        if dims.len() < head_rank + tail_rank {
            return None;
        }

        let head_field = Head::from_dyn(&dims[..head_rank])?;
        let tail_field = Tail::from_dyn(&dims[dims.len() - tail_rank..])?;
        let middle_dims = dims[head_rank..dims.len() - tail_rank].to_vec();

        Some(SpanShapeField {
            head_field,
            middle_dims,
            tail_field,
        })
    }
    fn dims(shape: &Self::Field) -> Self::Dims {
        let head_dims = Head::dims(&shape.head_field);
        let tail_dims = Tail::dims(&shape.tail_field);
        let mut full = head_dims.as_ref().to_vec();
        full.extend_from_slice(&shape.middle_dims);
        full.extend_from_slice(tail_dims.as_ref());
        full
    }
}

impl<Head: DynShape<Field: Default>, Tail: DynShape<Field: Default>> DynShape
    for SpanShape<Head, Tail>
where
    Head::Arg: Default,
    Tail::Arg: Default,
{
    fn rank(shape: &Self::Field) -> usize {
        let head_dims = Head::dims(&shape.head_field);
        let tail_dims = Tail::dims(&shape.tail_field);
        head_dims.as_ref().len() + shape.middle_dims.len() + tail_dims.as_ref().len()
    }
}

impl<Head: DynShape<Field: Default>, Tail: DynShape<Field: Default>, D: Dim> EndsWith<D>
    for SpanShape<Head, Tail>
where
    Tail: EndsWith<D>,
    Head::Arg: Default,
    Tail::Arg: Default,
{
}

impl<Head: DynShape<Field: Default>, Tail: DynShape<Field: Default>, D: Dim> HasChannels1D<D>
    for SpanShape<Head, Tail>
where
    Tail: HasChannels1D<D>,
    Head::Arg: Default,
    Tail::Arg: Default,
{
}

impl<Head: DynShape<Field: Default>, Tail: DynShape<Field: Default>, D: Dim> HasChannels2D<D>
    for SpanShape<Head, Tail>
where
    Tail: HasChannels2D<D>,
    Head::Arg: Default,
    Tail::Arg: Default,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use typenum::{U8, U16, U32, U128};

    #[test]
    fn test_tail_shape_from_dyn() {
        type S = TailShape<(U128,)>;
        assert!(S::from_dyn(&[32, 16]).is_none());

        let field = S::from_dyn(&[32, 16, 128]).unwrap();
        assert_eq!(S::rank(&field), 3);
        assert_eq!(S::dims(&field), vec![32, 16, 128]);
        assert_eq!(S::numel(&field), 32 * 16 * 128);

        // Wrong fixed dimension size should fail from_dyn
        assert!(S::from_dyn(&[32, 16, 64]).is_none());
    }

    #[test]
    fn test_head_shape_from_dyn() {
        type S = HeadShape<(U128,)>;
        assert!(S::from_dyn(&[]).is_none());

        let field = S::from_dyn(&[128, 64, 32]).unwrap();
        assert_eq!(S::rank(&field), 3);
        assert_eq!(S::dims(&field), vec![128, 64, 32]);
        assert_eq!(S::numel(&field), 128 * 64 * 32);

        // Wrong fixed head size should fail from_dyn
        assert!(S::from_dyn(&[64, 64, 32]).is_none());
    }

    #[test]
    fn test_span_shape_from_dyn() {
        type S = SpanShape<(U32,), (U128,)>;
        assert!(S::from_dyn(&[32]).is_none()); // Insufficient rank

        let field = S::from_dyn(&[32, 16, 8, 128]).unwrap();
        assert_eq!(S::rank(&field), 4);
        assert_eq!(S::dims(&field), vec![32, 16, 8, 128]);
        assert_eq!(S::numel(&field), 32 * 16 * 8 * 128);

        // Mismatched head or tail size should fail from_dyn
        assert!(S::from_dyn(&[64, 16, 8, 128]).is_none());
        assert!(S::from_dyn(&[32, 16, 8, 64]).is_none());
    }
}
