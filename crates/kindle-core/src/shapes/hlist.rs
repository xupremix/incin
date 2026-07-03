use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use crate::prelude::{Dim, DynShape, PartialDynShape, Shape};

/// The end of an HList shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Nil;

/// A node in an HList shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cons<Head, Tail>(PhantomData<(Head, Tail)>);

impl Shape for Nil {
    type Arg = ();
    type Field = ();
    type Dims = Vec<usize>;

    fn init(_arg: Self::Arg) -> Self::Field {}

    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        if dims.is_empty() { Some(()) } else { None }
    }
}

impl DynShape for Nil {
    #[inline(always)]
    fn rank(_shape: &Self::Field) -> usize {
        0
    }

    #[inline(always)]
    fn numel(_shape: &Self::Field) -> usize {
        1
    }

    #[inline(always)]
    fn dims(_shape: &Self::Field) -> Self::Dims {
        Vec::new()
    }
}

impl PartialDynShape for Nil {
    const RANK: usize = 0;
}

impl<Head: Dim, Tail: Shape<Dims = Vec<usize>>> Shape for Cons<Head, Tail> {
    type Arg = (Head::Arg, Tail::Arg);
    type Field = (Head, Tail::Field);
    type Dims = Vec<usize>;

    fn init(arg: Self::Arg) -> Self::Field {
        (Head::from_arg(arg.0), Tail::init(arg.1))
    }

    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        if dims.is_empty() {
            return None;
        }
        Some((Head::from_size(dims[0])?, Tail::from_dyn(&dims[1..])?))
    }
}

impl<Head: Dim, Tail: DynShape<Dims = Vec<usize>> + PartialDynShape> DynShape for Cons<Head, Tail> {
    #[inline(always)]
    fn rank(_shape: &Self::Field) -> usize {
        Self::RANK
    }

    #[inline(always)]
    fn numel(shape: &Self::Field) -> usize {
        shape.0.size() * Tail::numel(&shape.1)
    }

    #[inline(always)]
    fn dims(shape: &Self::Field) -> Self::Dims {
        let mut v = alloc::vec![shape.0.size()];
        v.extend(Tail::dims(&shape.1));
        v
    }
}

impl<Head: Dim, Tail: PartialDynShape<Dims = Vec<usize>>> PartialDynShape for Cons<Head, Tail> {
    const RANK: usize = 1 + Tail::RANK;
}

// TODO: ConstShape for Cons
