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
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field: Debug + Clone + Send + Sync;
    /// Core abstraction for `Dims` within the Kindle framework..
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
    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field;
    /// Core abstraction for `from_dyn` within the Kindle framework..
    fn from_dyn(dims: &[usize]) -> Option<Self::Field>;
}

/// A shape with runtime-accessible dimension information (rank, total elements, per-axis sizes).
///
/// All implementors of `Shape` that support dynamic rank queries also implement `DynShape`.
/// This includes both `Dyn` and fully static shapes (tuples). Operations that need to introspect
/// the shape at runtime (e.g., computing strides) require a `DynShape` bound.
pub trait DynShape: Shape {
    /// Core abstraction for `rank` within the Kindle framework..
    fn rank(shape: &Self::Field) -> usize;
    /// Core abstraction for `numel` within the Kindle framework..
    fn numel(shape: &Self::Field) -> usize;
    /// Core abstraction for `dims` within the Kindle framework..
    fn dims(shape: &Self::Field) -> Self::Dims;
}

/// Core abstraction for `AppendDim` within the Kindle framework..
pub trait AppendDim<D: Dim> {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output: Shape;
}

/// Core abstraction for `ReplaceLastDim` within the Kindle framework..
pub trait ReplaceLastDim<NewDim: Dim> {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output: Shape;
}

/// Core abstraction for `EndsWith` within the Kindle framework..
pub trait EndsWith<D: Dim>: Shape {}
/// Core abstraction for `HasChannels1D` within the Kindle framework..
pub trait HasChannels1D<D: Dim>: Shape {}
/// Core abstraction for `HasChannels2D` within the Kindle framework..
pub trait HasChannels2D<D: Dim>: Shape {}

impl<D: Dim> EndsWith<D> for Dyn {}
impl<D: Dim> HasChannels1D<D> for Dyn {}
impl<D: Dim> HasChannels2D<D> for Dyn {}

/// Core abstraction for `PartialDynShape` within the Kindle framework..
pub trait PartialDynShape: DynShape {
    /// Core abstraction for `RANK` within the Kindle framework..
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
    /// Core abstraction for `NUMEL` within the Kindle framework..
    const NUMEL: usize;
    /// Core abstraction for `DIMS` within the Kindle framework..
    const DIMS: <Self as Shape>::Dims;
}

///
/// --- Dyn ---
///
impl Shape for Dyn {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = Vec<usize>;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = Vec<usize>;
    /// Core abstraction for `Dims` within the Kindle framework..
    type Dims = Vec<usize>;
    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
    /// Core abstraction for `from_dyn` within the Kindle framework..
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        Some(dims.to_vec())
    }
}

impl DynShape for Dyn {
    #[inline(always)]
    /// Core abstraction for `rank` within the Kindle framework..
    fn rank(shape: &Self::Field) -> usize {
        shape.len()
    }

    #[inline(always)]
    /// Core abstraction for `numel` within the Kindle framework..
    fn numel(shape: &Self::Field) -> usize {
        shape.iter().product()
    }

    #[inline(always)]
    /// Core abstraction for `dims` within the Kindle framework..
    fn dims(shape: &Self::Field) -> Self::Dims {
        shape.clone()
    }
}

impl<D: Dim> AppendDim<D> for Dyn {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
}

macro_rules! impl_shape_for_tuple {
    ($n:expr $(, $name:ident $idx:tt)* $(,)?) => {
        impl< $($name: Dim,)* > Shape for ( $($name,)*) {
            /// Core abstraction for `Arg` within the Kindle framework..
            type Arg = ($(<$name as Dim>::Arg,)*);
            /// Core abstraction for `Field` within the Kindle framework..
            type Field = Self;
            /// Core abstraction for `Dims` within the Kindle framework..
            type Dims = [usize; ($n)];
            /// Core abstraction for `init` within the Kindle framework..
            fn init(arg: Self::Arg) -> Self::Field {
                ($(Dim::from_arg(arg.$idx),)*)
            }
            /// Core abstraction for `from_dyn` within the Kindle framework..
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
            /// Core abstraction for `RANK` within the Kindle framework..
            const RANK: usize = $n;
        }
        impl< $($name: Dim,)* > DynShape for ( $($name,)*) {
            #[inline(always)]
            /// Core abstraction for `dims` within the Kindle framework..
            fn dims(shape: &Self::Field) -> Self::Dims {
                [$(shape.$idx.size()),*]
            }

            #[inline(always)]
            /// Core abstraction for `rank` within the Kindle framework..
            fn rank(_: &Self::Field) -> usize {
                ($n)
            }

            #[inline(always)]
            /// Core abstraction for `numel` within the Kindle framework..
            fn numel(shape: &Self::Field) -> usize {
                1 $( * shape.$idx.size())*
            }
        }

        impl<$($name: Unsigned + Dim, )*> ConstShape for ($($name, )*) {
            /// Core abstraction for `NUMEL` within the Kindle framework..
            const NUMEL: usize = $($name::USIZE * )* 1;
            /// Core abstraction for `DIMS` within the Kindle framework..
            const DIMS: Self::Dims = [$($name::USIZE),*];
        }

        impl Shape for [usize; ($n)] {
            /// Core abstraction for `Arg` within the Kindle framework..
            type Arg = Self;
            /// Core abstraction for `Field` within the Kindle framework..
            type Field = Self;
            /// Core abstraction for `Dims` within the Kindle framework..
            type Dims = Self;
            /// Core abstraction for `init` within the Kindle framework..
            fn init(arg: Self::Arg) -> Self::Field {
                arg
            }
            /// Core abstraction for `from_dyn` within the Kindle framework..
            fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
                dims.try_into().ok()
            }
        }
        impl DynShape for [usize; ($n)] {
            #[inline(always)]
            /// Core abstraction for `dims` within the Kindle framework..
            fn dims(shape: &Self::Field) -> Self::Dims {
                *shape
            }

            #[inline(always)]
            /// Core abstraction for `rank` within the Kindle framework..
            fn rank(_: &Self::Field) -> usize {
                ($n)
            }

            #[inline(always)]
            /// Core abstraction for `numel` within the Kindle framework..
            fn numel(shape: &Self::Field) -> usize {
                1 $( * shape[$idx])*
            }
        }
        impl PartialDynShape for [usize; ($n)] {
            /// Core abstraction for `RANK` within the Kindle framework..
            const RANK: usize = ($n);
        }
        impl EndsWith<usize> for [usize; ($n)] {}
        impl HasChannels1D<usize> for [usize; ($n)] {}
        impl HasChannels2D<usize> for [usize; ($n)] {}
    };
}

impl Shape for () {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = ();
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = ();
    /// Core abstraction for `Dims` within the Kindle framework..
    type Dims = [usize; 0];
    /// Core abstraction for `init` within the Kindle framework..
    fn init(_: Self::Arg) {}
    /// Core abstraction for `from_dyn` within the Kindle framework..
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        if dims.is_empty() { Some(()) } else { None }
    }
}

impl PartialDynShape for () {
    /// Core abstraction for `RANK` within the Kindle framework..
    const RANK: usize = 0;
}

impl<D: Dim> AppendDim<D> for () {
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = (D,);
}

impl ConstShape for () {
    /// Core abstraction for `NUMEL` within the Kindle framework..
    const NUMEL: usize = 1;
    /// Core abstraction for `DIMS` within the Kindle framework..
    const DIMS: <Self as Shape>::Dims = [];
}

impl DynShape for () {
    #[inline(always)]
    /// Core abstraction for `rank` within the Kindle framework..
    fn rank(_: &Self::Field) -> usize {
        0
    }

    #[inline(always)]
    /// Core abstraction for `numel` within the Kindle framework..
    fn numel(_: &Self::Field) -> usize {
        1
    }

    #[inline(always)]
    /// Core abstraction for `dims` within the Kindle framework..
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
            /// Core abstraction for `Output` within the Kindle framework..
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
            /// Core abstraction for `Output` within the Kindle framework..
            type Output = (NewDim,);
        }
    };
    ($n1:ident, $last:ident) => {
        impl<$n1: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim> for ($n1, $last) {
            /// Core abstraction for `Output` within the Kindle framework..
            type Output = ($n1, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $last)
        {
            /// Core abstraction for `Output` within the Kindle framework..
            type Output = ($n1, $n2, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $n3, $last)
        {
            /// Core abstraction for `Output` within the Kindle framework..
            type Output = ($n1, $n2, $n3, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $n3, $n4, $last)
        {
            /// Core abstraction for `Output` within the Kindle framework..
            type Output = ($n1, $n2, $n3, $n4, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $n5: Dim, $last: Dim, NewDim: Dim>
            ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $last)
        {
            /// Core abstraction for `Output` within the Kindle framework..
            type Output = ($n1, $n2, $n3, $n4, $n5, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $n6:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $n5: Dim, $n6: Dim, $last: Dim, NewDim: Dim>
            ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $n6, $last)
        {
            /// Core abstraction for `Output` within the Kindle framework..
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
            /// Core abstraction for `Output` within the Kindle framework..
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
            /// Core abstraction for `Output` within the Kindle framework..
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
            /// Core abstraction for `Output` within the Kindle framework..
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
            /// Core abstraction for `Output` within the Kindle framework..
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
            /// Core abstraction for `Output` within the Kindle framework..
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
    /// Core abstraction for `Output` within the Kindle framework..
    type Output = Dyn;
}

impl<D: Dim> Shape for Vec<D> {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = Self;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = Self;
    /// Core abstraction for `Dims` within the Kindle framework..
    type Dims = Vec<usize>;
    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
    /// Core abstraction for `from_dyn` within the Kindle framework..
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        dims.iter().map(|&d| D::from_size(d)).collect()
    }
}

impl<D: Dim> DynShape for Vec<D> {
    #[inline(always)]
    /// Core abstraction for `rank` within the Kindle framework..
    fn rank(shape: &Self::Field) -> usize {
        shape.len()
    }

    #[inline(always)]
    /// Core abstraction for `numel` within the Kindle framework..
    fn numel(shape: &Self::Field) -> usize {
        shape.iter().map(|d| d.size()).product()
    }

    #[inline(always)]
    /// Core abstraction for `dims` within the Kindle framework..
    fn dims(shape: &Self::Field) -> Self::Dims {
        shape.iter().map(|d| d.size()).collect()
    }
}

/// Core abstraction for `Scalar` within the Kindle framework..
pub type Scalar = ();

#[cfg(test)]
/// Core abstraction for `tests` within the Kindle framework..
mod tests {
    use super::*;

    #[test]
    /// Core abstraction for `test_scalar_shape` within the Kindle framework..
    fn test_scalar_shape() {
        assert_eq!(<() as DynShape>::rank(&()), 0);
        assert_eq!(<() as DynShape>::numel(&()), 1);
        let empty_dims: [usize; 0] = [];
        assert_eq!(<() as DynShape>::dims(&()), empty_dims);
        assert_eq!(<() as DynShape>::rank(&()), 0);
        assert_eq!(<() as ConstShape>::DIMS, empty_dims);
    }

    #[test]
    /// Core abstraction for `test_dyn_shape` within the Kindle framework..
    fn test_dyn_shape() {
        let d = vec![2, 3, 4];
        assert_eq!(<Dyn as DynShape>::rank(&d), 3);
        assert_eq!(<Dyn as DynShape>::numel(&d), 24);
        assert_eq!(<Dyn as DynShape>::dims(&d), vec![2, 3, 4]);
    }

    #[test]
    /// Core abstraction for `test_array_shape` within the Kindle framework..
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
