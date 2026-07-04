use crate::prelude::{ConstDim, Dim, Dyn};
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
    fn from_dyn(dims: &[usize]) -> Option<Self::Field>;
}

pub trait DynShape: Shape {
    fn rank(shape: &Self::Field) -> usize;
    fn numel(shape: &Self::Field) -> usize;
    fn dims(shape: &Self::Field) -> Self::Dims;
}

pub trait AppendDim<D: Dim> {
    type Output: Shape;
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
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        Some(dims.to_vec())
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

impl<D: Dim> AppendDim<D> for Dyn {
    type Output = Dyn;
}

macro_rules! impl_shape_for_tuple {
    ($n:expr $(, $name:ident $idx:tt)* $(,)?) => {
        impl< $($name: Dim,)* > Shape for ( $($name,)*) {
            type Arg = ($(<$name as Dim>::Arg,)*);
            type Field = Self;
            type Dims = [usize; ($n)];
            fn init(arg: Self::Arg) -> Self::Field {
                ($(Dim::from_arg(arg.$idx),)*)
            }
            fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
                if dims.len() != $n {
                    return None;
                }
                Some(($(
                    $name::from_size(dims[$idx])?,
                )*))
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
            fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
                dims.try_into().ok()
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
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        if dims.is_empty() { Some(()) } else { None }
    }
}

impl PartialDynShape for () {
    const RANK: usize = 0;
}

impl<D: Dim> AppendDim<D> for () {
    type Output = (D,);
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
impl_shape_for_tuple!(8, D0 0, D1 1, D2 2, D3 3, D4 4, D5 5, D6 6, D7 7);
impl_shape_for_tuple!(9, D0 0, D1 1, D2 2, D3 3, D4 4, D5 5, D6 6, D7 7, D8 8);
impl_shape_for_tuple!(10, D0 0, D1 1, D2 2, D3 3, D4 4, D5 5, D6 6, D7 7, D8 8, D9 9);
impl_shape_for_tuple!(11, D0 0, D1 1, D2 2, D3 3, D4 4, D5 5, D6 6, D7 7, D8 8, D9 9, D10 10);
impl_shape_for_tuple!(12, D0 0, D1 1, D2 2, D3 3, D4 4, D5 5, D6 6, D7 7, D8 8, D9 9, D10 10, D11 11);

macro_rules! impl_append_dim_for_tuple {
    ($($name:ident),*) => {
        impl< $($name: Dim,)* Append: Dim > AppendDim<Append> for ( $($name,)*) {
            type Output = ( $($name,)* Append);
        }
    };
}

impl_append_dim_for_tuple!(D0);
impl_append_dim_for_tuple!(D0, D1);
impl_append_dim_for_tuple!(D0, D1, D2);
impl_append_dim_for_tuple!(D0, D1, D2, D3);
impl_append_dim_for_tuple!(D0, D1, D2, D3, D4);
impl_append_dim_for_tuple!(D0, D1, D2, D3, D4, D5);
impl_append_dim_for_tuple!(D0, D1, D2, D3, D4, D5, D6);
impl_append_dim_for_tuple!(D0, D1, D2, D3, D4, D5, D6, D7);
impl_append_dim_for_tuple!(D0, D1, D2, D3, D4, D5, D6, D7, D8);
impl_append_dim_for_tuple!(D0, D1, D2, D3, D4, D5, D6, D7, D8, D9);
impl_append_dim_for_tuple!(D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10);
// Note: Rust standard library only implements traits (Debug, Eq, etc.) for tuples up to size 12.
// For dimensions > 12, use `[usize; N]` which is fully supported via const generics.

impl<D: Dim> Shape for Vec<D> {
    type Arg = Self;
    type Field = Self;
    type Dims = Vec<usize>;
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        dims.iter().map(|&d| D::from_size(d)).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_shape() {
        assert_eq!(<() as DynShape>::rank(&()), 0);
        assert_eq!(<() as DynShape>::numel(&()), 1);
        let empty_dims: [usize; 0] = [];
        assert_eq!(<() as DynShape>::dims(&()), empty_dims);
        assert_eq!(<() as DynShape>::rank(&()), 0);
        assert_eq!(<() as ConstShape>::DIMS, empty_dims);
    }

    #[test]
    fn test_dyn_shape() {
        let d = vec![2, 3, 4];
        assert_eq!(<Dyn as DynShape>::rank(&d), 3);
        assert_eq!(<Dyn as DynShape>::numel(&d), 24);
        assert_eq!(<Dyn as DynShape>::dims(&d), vec![2, 3, 4]);
    }

    #[test]
    fn test_array_shape() {
        let shape: [usize; 3] = [2, 3, 4];
        assert_eq!(<[usize; 3] as DynShape>::rank(&shape), 3);
        assert_eq!(<[usize; 3] as DynShape>::numel(&shape), 24);
        assert_eq!(<[usize; 3] as DynShape>::dims(&shape), [2, 3, 4]);
        assert_eq!(<[usize; 3] as PartialDynShape>::RANK, 3);
    }
}
