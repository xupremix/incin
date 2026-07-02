use crate::prelude::{ConstDim, Dim, Dyn, NotUnit};
use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::{Index, IndexMut};

pub trait Shape: 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    type Arg;
    type Field: Debug + Clone + Send + Sync;
    type Dims: Debug
        + Clone
        + Default
        + Eq
        + PartialEq
        + Send
        + Sync
        + IntoIterator<Item = usize>
        + Into<Vec<usize>>
        + Index<usize, Output = usize>
        + IndexMut<usize>
        + AsRef<[usize]>;
    fn init(arg: Self::Arg) -> Self::Field;
}

pub trait DynShape: Shape {
    fn rank(shape: &Self::Field) -> usize;
    fn numel(shape: &Self::Field) -> usize;
    fn dims(shape: &Self::Field) -> Self::Dims;
}

pub trait PartialDynShape: DynShape {
    const RANK: usize;
}

pub trait ConstShape: Shape<Field: Default> {
    // const RANK: usize; // impl PartialDynShape for it and DynShape
    const NUMEL: usize;
    const DIMS: <Self as Shape>::Dims;
}

///
/// --- Dyn ---
///
impl Shape for Dyn {
    type Arg = Vec<usize>;
    type Field = Vec<usize>;
    type Dims = Vec<usize>;
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}

impl DynShape for Dyn {
    #[inline(always)]
    fn rank(shape: &Self::Field) -> usize {
        shape.len()
    }

    #[inline(always)]
    fn numel(shape: &Self::Field) -> usize {
        shape.iter().product()
    }

    #[inline(always)]
    fn dims(shape: &Self::Field) -> Self::Dims {
        shape.clone()
    }
}

macro_rules! impl_shape_for_tuple {
    ($n:expr $(, $name:ident $idx:tt)* $(,)?) => {
        impl< $($name: Dim,)* > NotUnit for ( $($name,)* ) {}
        impl<D: Dim> NotUnit for [D; ($n)] {}
        impl<'a, D: Dim> NotUnit for &'a [D; ($n)] {}
        impl< $($name: Dim,)* > Shape for ( $($name,)*) {
            type Arg = ($(<$name as Dim>::Arg,)*);
            type Field = Self;
            type Dims = [usize; ($n)];
            fn init(arg: Self::Arg) -> Self::Field {
                ($(Dim::from_arg(arg.$idx),)*)
            }
        }
        impl< $($name: Dim,)* > PartialDynShape for ( $($name,)*) {
            const RANK: usize = $n;
        }
        impl< $($name: Dim,)* > DynShape for ( $($name,)*) {
            #[inline(always)]
            fn dims(shape: &Self::Field) -> Self::Dims {
                [$(shape.$idx.size()),*]
            }

            #[inline(always)]
            fn rank(_: &Self::Field) -> usize {
                ($n)
            }

            #[inline(always)]
            fn numel(shape: &Self::Field) -> usize {
                1 $( * shape.$idx.size())*
            }
        }

        impl<$($name: ConstDim, )*> ConstShape for ($($name, )*) {
            const NUMEL: usize = $($name::SIZE * )* 1;
            const DIMS: Self::Dims = [$($name::SIZE),*];
        }

        impl Shape for [usize; ($n)] {
            type Arg = Self;
            type Field = Self;
            type Dims = Self;
            fn init(arg: Self::Arg) -> Self::Field {
                arg
            }
        }
        impl DynShape for [usize; ($n)] {
            #[inline(always)]
            fn dims(shape: &Self::Field) -> Self::Dims {
                *shape
            }

            #[inline(always)]
            fn rank(_: &Self::Field) -> usize {
                ($n)
            }

            #[inline(always)]
            fn numel(shape: &Self::Field) -> usize {
                1 $( * shape[$idx])*
            }
        }
        impl PartialDynShape for [usize; ($n)] {
            const RANK: usize = ($n);
        }
    };
}

impl Shape for () {
    type Arg = ();
    type Field = ();
    type Dims = [usize; 0];
    fn init(_: Self::Arg) {}
}

impl PartialDynShape for () {
    const RANK: usize = 0;
}

impl ConstShape for () {
    const NUMEL: usize = 1;
    const DIMS: <Self as Shape>::Dims = [];
}

impl DynShape for () {
    #[inline(always)]
    fn rank(_: &Self::Field) -> usize {
        0
    }

    #[inline(always)]
    fn numel(_: &Self::Field) -> usize {
        1
    }

    #[inline(always)]
    fn dims(_: &Self::Field) -> Self::Dims {
        []
    }
}

impl_shape_for_tuple!(1, D0 0);
impl_shape_for_tuple!(2, D0 0, D1 1);
impl_shape_for_tuple!(3, D0 0, D1 1, D2 2);
impl_shape_for_tuple!(4, D0 0, D1 1, D2 2, D3 3);
impl_shape_for_tuple!(5, D0 0, D1 1, D2 2, D3 3, D4 4);
impl_shape_for_tuple!(6, D0 0, D1 1, D2 2, D3 3, D4 4, D5 5);
impl_shape_for_tuple!(7, D0 0, D1 1, D2 2, D3 3, D4 4, D5 5, D6 6);

impl<D: Dim> Shape for Vec<D> {
    type Arg = Self;
    type Field = Self;
    type Dims = Vec<usize>;
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}

impl<D: Dim> DynShape for Vec<D> {
    #[inline(always)]
    fn rank(shape: &Self::Field) -> usize {
        shape.len()
    }

    #[inline(always)]
    fn numel(shape: &Self::Field) -> usize {
        shape.iter().map(|d| d.size()).product()
    }

    #[inline(always)]
    fn dims(shape: &Self::Field) -> Self::Dims {
        shape.iter().map(|d| d.size()).collect()
    }
}

pub type Scalar = ();
