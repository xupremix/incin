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
    /// The user-facing constructor argument type (e.g. a tuple of
    /// `usize`/`typenum` values, or `Vec<usize>` for `Dyn`).
    type Arg;
    /// The runtime-stored representation of this shape inside a
    /// `Tensor` (produced from `Arg` via `init`).
    type Field: Debug + Clone + Send + Sync;
    /// A fixed-size or `Vec`-backed collection of this shape's
    /// per-dimension sizes, as returned by `DynShape::dims`.
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
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field;
    /// Attempts to construct this shape's `Field` from raw runtime
    /// dimensions, returning `None` if `dims` doesn't match `Self`
    /// (e.g. wrong rank, or a statically-fixed dimension that disagrees).
    fn from_dyn(dims: &[usize]) -> Option<Self::Field>;
}

/// A shape with runtime-accessible dimension information (rank, total elements, per-axis sizes).
///
/// All implementors of `Shape` that support dynamic rank queries also implement `DynShape`.
/// This includes both `Dyn` and fully static shapes (tuples). Operations that need to introspect
/// the shape at runtime (e.g., computing strides) require a `DynShape` bound.
pub trait DynShape: Shape {
    /// Returns the number of dimensions.
    fn rank(shape: &Self::Field) -> usize;
    /// Returns the total element count (product of all dimension sizes).
    fn numel(shape: &Self::Field) -> usize;
    /// Returns each dimension's size.
    fn dims(shape: &Self::Field) -> Self::Dims;
}

/// Appends dimension `D` to the end of `Self`'s shape.
pub trait AppendDim<D: Dim> {
    /// `Self`'s dimensions with `D` appended at the end.
    type Output: Shape;
}

/// Replaces `Self`'s last dimension with `NewDim`.
pub trait ReplaceLastDim<NewDim: Dim> {
    /// `Self`'s dimensions with the last one replaced by `NewDim`.
    type Output: Shape;
}

/// Marker: `Self`'s last dimension is `D` — used to bound layer
/// `forward` impls (e.g. `Linear`) to inputs whose trailing feature
/// dimension matches the layer's expected input size.
pub trait EndsWith<D: Dim>: Shape {}
/// Marker: `Self` has `D` channels at the `Conv1d`-expected channel
/// position (second-to-last dimension, `[.., C, L]`).
pub trait HasChannels1D<D: Dim>: Shape {}
/// Marker: `Self` has `D` channels at the `Conv2d`/`BatchNorm2d`-expected
/// channel position (third-to-last dimension, `[.., C, H, W]`).
pub trait HasChannels2D<D: Dim>: Shape {}

impl<D: Dim> EndsWith<D> for Dyn {}
impl<D: Dim> HasChannels1D<D> for Dyn {}
impl<D: Dim> HasChannels2D<D> for Dyn {}

/// A `DynShape` whose rank is additionally known at compile time (as
/// opposed to `Dyn`, whose rank is runtime-only).
pub trait PartialDynShape: DynShape {
    /// The compile-time-known number of dimensions.
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
    /// The compile-time-known total element count.
    const NUMEL: usize;
    /// The compile-time-known per-dimension sizes.
    const DIMS: <Self as Shape>::Dims;
}

///
/// --- Dyn ---
///
impl Shape for Dyn {
    /// The user-facing constructor argument type for this concrete shape.
    type Arg = Vec<usize>;
    /// The runtime-stored representation for this concrete shape.
    type Field = Vec<usize>;
    /// The per-dimension-sizes collection type for this concrete shape.
    type Dims = Vec<usize>;
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
    /// Attempts to construct this shape's `Field` from raw runtime dimensions.
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        Some(dims.to_vec())
    }
}

impl DynShape for Dyn {
    #[inline(always)]
    /// Returns the number of dimensions.
    fn rank(shape: &Self::Field) -> usize {
        shape.len()
    }

    #[inline(always)]
    /// Returns the total element count.
    fn numel(shape: &Self::Field) -> usize {
        shape.iter().product()
    }

    #[inline(always)]
    /// Returns each dimension's size.
    fn dims(shape: &Self::Field) -> Self::Dims {
        shape.clone()
    }
}

impl<D: Dim> AppendDim<D> for Dyn {
    /// `Self`'s dimensions with `D` appended at the end.
    type Output = Dyn;
}

macro_rules! impl_shape_for_tuple {
    ($n:expr $(, $name:ident $idx:tt)* $(,)?) => {
        impl< $($name: Dim,)* > Shape for ( $($name,)*) {
            /// The user-facing constructor argument type for this concrete shape.
            type Arg = ($(<$name as Dim>::Arg,)*);
            /// The runtime-stored representation for this concrete shape.
            type Field = Self;
            /// The per-dimension-sizes collection type for this concrete shape.
            type Dims = [usize; ($n)];
            /// Converts a user-facing `Arg` into the stored `Field` representation.
            fn init(arg: Self::Arg) -> Self::Field {
                ($(Dim::from_arg(arg.$idx),)*)
            }
            /// Attempts to construct this shape's `Field` from raw runtime dimensions.
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
            /// The compile-time-known number of dimensions.
            const RANK: usize = $n;
        }
        impl< $($name: Dim,)* > DynShape for ( $($name,)*) {
            #[inline(always)]
            /// Returns each dimension's size.
            fn dims(shape: &Self::Field) -> Self::Dims {
                [$(shape.$idx.size()),*]
            }

            #[inline(always)]
            /// Returns the number of dimensions.
            fn rank(_: &Self::Field) -> usize {
                ($n)
            }

            #[inline(always)]
            /// Returns the total element count.
            fn numel(shape: &Self::Field) -> usize {
                1 $( * shape.$idx.size())*
            }
        }

        impl<$($name: Unsigned + Dim, )*> ConstShape for ($($name, )*) {
            /// The compile-time total element count.
            const NUMEL: usize = $($name::USIZE * )* 1;
            /// The compile-time per-dimension sizes.
            const DIMS: Self::Dims = [$($name::USIZE),*];
        }

        impl Shape for [usize; ($n)] {
            /// The user-facing constructor argument type for this concrete shape.
            type Arg = Self;
            /// The runtime-stored representation for this concrete shape.
            type Field = Self;
            /// The per-dimension-sizes collection type for this concrete shape.
            type Dims = Self;
            /// Converts a user-facing `Arg` into the stored `Field` representation.
            fn init(arg: Self::Arg) -> Self::Field {
                arg
            }
            /// Attempts to construct this shape's `Field` from raw runtime dimensions.
            fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
                dims.try_into().ok()
            }
        }
        impl DynShape for [usize; ($n)] {
            #[inline(always)]
            /// Returns each dimension's size.
            fn dims(shape: &Self::Field) -> Self::Dims {
                *shape
            }

            #[inline(always)]
            /// Returns the number of dimensions.
            fn rank(_: &Self::Field) -> usize {
                ($n)
            }

            #[inline(always)]
            /// Returns the total element count.
            fn numel(shape: &Self::Field) -> usize {
                1 $( * shape[$idx])*
            }
        }
        impl PartialDynShape for [usize; ($n)] {
            /// The compile-time-known number of dimensions.
            const RANK: usize = ($n);
        }
        impl EndsWith<usize> for [usize; ($n)] {}
        impl HasChannels1D<usize> for [usize; ($n)] {}
        impl HasChannels2D<usize> for [usize; ($n)] {}
    };
}

impl Shape for () {
    /// The user-facing constructor argument type for this concrete shape.
    type Arg = ();
    /// The runtime-stored representation for this concrete shape.
    type Field = ();
    /// The per-dimension-sizes collection type for this concrete shape.
    type Dims = [usize; 0];
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(_: Self::Arg) {}
    /// Attempts to construct this shape's `Field` from raw runtime dimensions.
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        if dims.is_empty() { Some(()) } else { None }
    }
}

impl PartialDynShape for () {
    /// The compile-time-known number of dimensions.
    const RANK: usize = 0;
}

impl<D: Dim> AppendDim<D> for () {
    /// `Self`'s dimensions with `D` appended at the end.
    type Output = (D,);
}

impl ConstShape for () {
    /// The compile-time total element count.
    const NUMEL: usize = 1;
    /// The compile-time per-dimension sizes.
    const DIMS: <Self as Shape>::Dims = [];
}

impl DynShape for () {
    #[inline(always)]
    /// Returns the number of dimensions.
    fn rank(_: &Self::Field) -> usize {
        0
    }

    #[inline(always)]
    /// Returns the total element count.
    fn numel(_: &Self::Field) -> usize {
        1
    }

    #[inline(always)]
    /// Returns each dimension's size.
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
            /// `Self`'s dimensions with `Append` appended at the end.
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
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
            type Output = (NewDim,);
        }
    };
    ($n1:ident, $last:ident) => {
        impl<$n1: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim> for ($n1, $last) {
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
            type Output = ($n1, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $last)
        {
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
            type Output = ($n1, $n2, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $n3, $last)
        {
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
            type Output = ($n1, $n2, $n3, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($n1, $n2, $n3, $n4, $last)
        {
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
            type Output = ($n1, $n2, $n3, $n4, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $n5: Dim, $last: Dim, NewDim: Dim>
            ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $last)
        {
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
            type Output = ($n1, $n2, $n3, $n4, $n5, NewDim);
        }
    };
    ($n1:ident, $n2:ident, $n3:ident, $n4:ident, $n5:ident, $n6:ident, $last:ident) => {
        impl<$n1: Dim, $n2: Dim, $n3: Dim, $n4: Dim, $n5: Dim, $n6: Dim, $last: Dim, NewDim: Dim>
            ReplaceLastDim<NewDim> for ($n1, $n2, $n3, $n4, $n5, $n6, $last)
        {
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
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
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
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
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
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
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
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
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
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
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
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
    /// `Self`'s dimensions with the last one replaced by `NewDim`.
    type Output = Dyn;
}

impl<D: Dim> Shape for Vec<D> {
    /// The user-facing constructor argument type for this concrete shape.
    type Arg = Self;
    /// The runtime-stored representation for this concrete shape.
    type Field = Self;
    /// The per-dimension-sizes collection type for this concrete shape.
    type Dims = Vec<usize>;
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
    /// Attempts to construct this shape's `Field` from raw runtime dimensions.
    fn from_dyn(dims: &[usize]) -> Option<Self::Field> {
        dims.iter().map(|&d| D::from_size(d)).collect()
    }
}

impl<D: Dim> DynShape for Vec<D> {
    #[inline(always)]
    /// Returns the number of dimensions.
    fn rank(shape: &Self::Field) -> usize {
        shape.len()
    }

    #[inline(always)]
    /// Returns the total element count.
    fn numel(shape: &Self::Field) -> usize {
        shape.iter().map(|d| d.size()).product()
    }

    #[inline(always)]
    /// Returns each dimension's size.
    fn dims(shape: &Self::Field) -> Self::Dims {
        shape.iter().map(|d| d.size()).collect()
    }
}

/// The 0-dimensional (scalar) shape — an alias for `()`.
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
