use crate::prelude::{Dim, Dyn};
use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::{Index, IndexMut};
use typenum::Unsigned;

/// The fundamental trait for all tensor shape types.
///
/// A `Shape` encodes the rank (number of dimensions) and, optionally, the static size of each
/// dimension into the type system. The three primary implementors are:
///
/// * **Tuple of `Dim` types** (e.g., `(U2, U3)`) — Fully static. All dimension sizes are known at compile time.
/// * **`Dyn`** — Fully dynamic. Shape is determined at runtime.
/// * **Tuples mixing `usize` and `typenum`** — Partially static (e.g., `(U3, usize)`).
///
/// In practice, shapes are most often constructed via the `s![]` macro.
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

/// A shape with runtime-accessible dimension information (rank, total elements, per-axis sizes).
///
/// All implementors of `Shape` that support dynamic rank queries also implement `DynShape`.
/// This includes both `Dyn` and fully static shapes (tuples). Operations that need to introspect
/// the shape at runtime (e.g., computing strides) require a `DynShape` bound.
pub trait DynShape: Shape {
    fn rank(shape: &Self::Field) -> usize;
    fn numel(shape: &Self::Field) -> usize;
    fn dims(shape: &Self::Field) -> Self::Dims;
}

pub trait AppendDim<D: Dim> {
    type Output: Shape;
}

pub trait ReplaceLastDim<NewDim: Dim> {
    type Output: Shape;
}

pub trait EndsWith<D: Dim>: Shape {}
pub trait HasChannels1D<D: Dim>: Shape {}
pub trait HasChannels2D<D: Dim>: Shape {}

impl<D: Dim> EndsWith<D> for Dyn {}
impl<D: Dim> HasChannels1D<D> for Dyn {}
impl<D: Dim> HasChannels2D<D> for Dyn {}

pub trait PartialDynShape: DynShape {
    const RANK: usize;
}

/// A fully static shape whose total number of elements and dimension sizes are available as compile-time constants.
///
/// This is implemented for all shapes built exclusively from `typenum` types (e.g., `(U2, U3, U4)`).
/// The key property is that `NUMEL` and `DIMS` are `const`, enabling the compiler to verify
/// that operations (like reshape) are element-count-preserving without any runtime checks.
///
/// ## Example
/// ```rust,ignore
/// use kindle_core::shapes::shape::ConstShape;
/// type MyShape = s![2, 3, 4];
/// assert_eq!(<MyShape as ConstShape>::NUMEL, 24);
/// ```
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

        impl<$($name: Unsigned + Dim, )*> ConstShape for ($($name, )*) {
            const NUMEL: usize = $($name::USIZE * )* 1;
            const DIMS: Self::Dims = [$($name::USIZE),*];
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
        impl EndsWith<usize> for [usize; ($n)] {}
        impl HasChannels1D<usize> for [usize; ($n)] {}
        impl HasChannels2D<usize> for [usize; ($n)] {}
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
// Note: Rust standard library only implements traits (Debug, Eq, etc.) for tuples up to size 12.
// We cap at rank 8 — appending to a 7-dim tuple yields rank 8, the maximum.

macro_rules! impl_replace_last_dim_for_tuple {
    ($last:ident) => {
        impl<$last: Dim, NewDim: Dim> ReplaceLastDim<NewDim> for ($last,) {
            type Output = (NewDim,);
        }
    };
    ($n1:ident, $last:ident) => {
        impl<$n1: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim> for ($n1, $last) {
            type Output = ($n1, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $last)
        {
            type Output = ($n1, $n2, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $n3, $last)
        {
            type Output = ($n1, $n2, $n3, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $n3, $n4, $last)
        {
            type Output = ($n1, $n2, $n3, $n4, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $n5: Dim, $last: Dim, NewDim: Dim>
            ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $last)
        {
            type Output = ($n1, $n2, $n3, $n4, $n5, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $n6:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $n5: Dim, $n6: Dim, $last: Dim, NewDim: Dim>
            ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $n6, $last)
        {
            type Output = ($n1, $n2, $n3, $n4, $n5, $n6, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $n6:ident, $n7:ident, $last:ident) => {
        impl<
            $n1: Dim,
            $n2: Dim,
            $n3: Dim,
            $n4: Dim,
            $n5: Dim,
            $n6: Dim,
            $n7: Dim,
            $last: Dim,
            NewDim: Dim,
        > ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $n6, $n7, $last)
        {
            type Output = ($n1, $n2, $n3, $n4, $n5, $n6, $n7, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $n6:ident, $n7:ident, $n8:ident, $last:ident) => {
        impl<
            $n1: Dim,
            $n2: Dim,
            $n3: Dim,
            $n4: Dim,
            $n5: Dim,
            $n6: Dim,
            $n7: Dim,
            $n8: Dim,
            $last: Dim,
            NewDim: Dim,
        > ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $n6, $n7, $n8, $last)
        {
            type Output = ($n1, $n2, $n3, $n4, $n5, $n6, $n7, $n8, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $n6:ident, $n7:ident, $n8:ident, $n9:ident, $last:ident) => {
        impl<
            $n1: Dim,
            $n2: Dim,
            $n3: Dim,
            $n4: Dim,
            $n5: Dim,
            $n6: Dim,
            $n7: Dim,
            $n8: Dim,
            $n9: Dim,
            $last: Dim,
            NewDim: Dim,
        > ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $n6, $n7, $n8, $n9, $last)
        {
            type Output = ($n1, $n2, $n3, $n4, $n5, $n6, $n7, $n8, $n9, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $n6:ident, $n7:ident, $n8:ident, $n9:ident, $n10:ident, $last:ident) => {
        impl<
            $n1: Dim,
            $n2: Dim,
            $n3: Dim,
            $n4: Dim,
            $n5: Dim,
            $n6: Dim,
            $n7: Dim,
            $n8: Dim,
            $n9: Dim,
            $n10: Dim,
            $last: Dim,
            NewDim: Dim,
        > ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $n6, $n7, $n8, $n9, $n10, $last)
        {
            type Output = ($n1, $n2, $n3, $n4, $n5, $n6, $n7, $n8, $n9, $n10, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $n6:ident, $n7:ident, $n8:ident, $n9:ident, $n10:ident, $n11:ident, $last:ident) => {
        impl<
            $n1: Dim,
            $n2: Dim,
            $n3: Dim,
            $n4: Dim,
            $n5: Dim,
            $n6: Dim,
            $n7: Dim,
            $n8: Dim,
            $n9: Dim,
            $n10: Dim,
            $n11: Dim,
            $last: Dim,
            NewDim: Dim,
        > ReplaceLastDim<NewDim>
            for (
                $n1,
                $n2,
                $n3,
                $n4,
                $n5,
                $n6,
                $n7,
                $n8,
                $n9,
                $n10,
                $n11,
                $last,
            )
        {
            type Output = (
                $n1,
                $n2,
                $n3,
                $n4,
                $n5,
                $n6,
                $n7,
                $n8,
                $n9,
                $n10,
                $n11,
                NewDim,
            );
        }
    };
}

impl_replace_last_dim_for_tuple!(D0);
impl_replace_last_dim_for_tuple!(D0, D1);
impl_replace_last_dim_for_tuple!(D0, D1, D2);
impl_replace_last_dim_for_tuple!(D0, D1, D2, D3);
impl_replace_last_dim_for_tuple!(D0, D1, D2, D3, D4);
impl_replace_last_dim_for_tuple!(D0, D1, D2, D3, D4, D5);
impl_replace_last_dim_for_tuple!(D0, D1, D2, D3, D4, D5, D6);
impl_replace_last_dim_for_tuple!(D0, D1, D2, D3, D4, D5, D6, D7);

impl<NewDim: Dim> ReplaceLastDim<NewDim> for Dyn {
    type Output = Dyn;
}

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

macro_rules! impl_ends_with_for_tuple {
    ($last:ident) => {
        impl<$last: Dim> EndsWith<$last> for ($last,) {}
    };
    ($n1:ident, $last:ident) => {
        impl<$n1: Dim, $last: Dim> EndsWith<$last> for ($n1, $last) {}
    };
    ($n1:ident, $n2:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $last: Dim> EndsWith<$last> for ($n1, $n2, $last) {}
    };
    ($n1:ident, $n2:ident, $n3:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $last: Dim> EndsWith<$last> for ($n1, $n2, $n3, $last) {}
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $last: Dim> EndsWith<$last>
            for ($n1, $n2, $n3, $n4, $last)
        {
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $n5: Dim, $last: Dim> EndsWith<$last>
            for ($n1, $n2, $n3, $n4, $n5, $last)
        {
        }
    };
}

impl_ends_with_for_tuple!(D0);
impl_ends_with_for_tuple!(D0, D1);
impl_ends_with_for_tuple!(D0, D1, D2);
impl_ends_with_for_tuple!(D0, D1, D2, D3);
impl_ends_with_for_tuple!(D0, D1, D2, D3, D4);
impl_ends_with_for_tuple!(D0, D1, D2, D3, D4, D5);

macro_rules! impl_has_channels_1d_for_tuple {
    ($n1:ident, $c:ident, $n3:ident) => {
        impl<$n1: Dim, $c: Dim, $n3: Dim> HasChannels1D<$c> for ($n1, $c, $n3) {}
    };
}

// Conv1d typically accepts 3D tensors: (Batch, Channels, Length)
impl_has_channels_1d_for_tuple!(D0, D1, D2);

macro_rules! impl_has_channels_2d_for_tuple {
    ($n1:ident, $c:ident, $n3:ident, $n4:ident) => {
        impl<$n1: Dim, $c: Dim, $n3: Dim, $n4: Dim> HasChannels2D<$c> for ($n1, $c, $n3, $n4) {}
    };
}

// Conv2d typically accepts 4D tensors: (Batch, Channels, Height, Width)
impl_has_channels_2d_for_tuple!(D0, D1, D2, D3);
