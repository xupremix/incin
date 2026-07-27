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
    /// How much of this shape the compiler settled, as opposed to the runtime.
    ///
    /// This is the shape-level lift of `Dim::STATIC_SIZE`: rank and every
    /// axis size known from the type gives
    /// [`ProofLevel::Static`](crate::exec::ProofLevel::Static); a known rank
    /// with at least one runtime or named axis gives
    /// [`Mixed`](crate::exec::ProofLevel::Mixed); a runtime rank gives
    /// [`Dynamic`](crate::exec::ProofLevel::Dynamic).
    ///
    /// A lowering rule reads this to stamp the `Validated<O>` it produces
    /// without knowing which concrete shape it was handed. It defaults to
    /// `Dynamic` so a `Shape` implemented outside this crate is credited with
    /// no proof it has not shown.
    const PROOF: crate::exec::ProofLevel = crate::exec::ProofLevel::Dynamic;

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

/// The highest tuple rank every shape rule is implemented for.
///
/// Single-sourced from `incin_macros::max_rank!()`. A proc-macro crate cannot
/// export a `const`, so the number lives there and is re-exported here;
/// duplicating it would reintroduce exactly the per-rule drift `SHP-006`
/// removes.
///
/// A rule that *adds* an axis (`AppendDim`, `StackShape`) correctly stops one
/// rank below this, because its `Output` is bounded by `Shape` and no tuple
/// above `MAX_RANK` implements `Shape`.
pub const MAX_RANK: usize = incin_macros::max_rank!();

/// Rebuild a typed shape field from computed dimensions, reporting instead of
/// panicking.
///
/// This is the checked replacement for the `from_dyn(&dims).unwrap()` chain
/// that `SHP-001` inventoried across 39 sites. The unwrap was a proof
/// obligation that no type stated and no test covered: the caller had already
/// erased a known-rank shape to a `Vec<usize>`, and then asserted the
/// round-trip back would succeed.
///
/// Prefer building the field axis by axis where the arity is known — that
/// avoids the erasure entirely and yields a
/// [`DimensionMismatch`](crate::shapes::error::ShapeError::DimensionMismatch)
/// naming the offending axis. Use this where the shape is only available
/// generically.
pub fn field_from_dims<S: Shape>(
    operation: crate::shapes::error::OperationKind,
    dims: &[usize],
) -> Result<S::Field, crate::shapes::error::ShapeError> {
    S::from_dyn(dims).ok_or(crate::shapes::error::ShapeError::TargetShapeRejected {
        operation,
        rank: dims.len(),
    })
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
#[diagnostic::on_unimplemented(
    message = "Cannot use shape `{Self}` here: its last dimension must be `{D}`",
    label = "wrong trailing dimension",
    note = "the input's last dimension must match this layer's expected input size"
)]
pub trait EndsWith<D: Dim>: Shape {}
/// Marker: `Self` has `D` channels at the `Conv1d`-expected channel
/// position (second-to-last dimension, `[.., C, L]`).
#[diagnostic::on_unimplemented(
    message = "Cannot use shape `{Self}` here: it must have `{D}` channels",
    label = "wrong channel count",
    note = "Conv1d/BatchNorm1d expect channels at the second-to-last dimension: [.., C, L]"
)]
pub trait HasChannels1D<D: Dim>: Shape {}
/// Marker: `Self` has `D` channels at the `Conv2d`/`BatchNorm2d`-expected
/// channel position (third-to-last dimension, `[.., C, H, W]`).
#[diagnostic::on_unimplemented(
    message = "Cannot use shape `{Self}` here: it must have `{D}` channels",
    label = "wrong channel count",
    note = "Conv2d/BatchNorm2d expect channels at the third-to-last dimension: [.., C, H, W]"
)]
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
/// use incin_core::shapes::shape::ConstShape;
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
    /// Not even the rank is known until the shape exists, which is the whole
    /// point of `Dyn`. Stated rather than inherited from the default so that
    /// changing the default cannot silently upgrade it.
    const PROOF: crate::exec::ProofLevel = crate::exec::ProofLevel::Dynamic;

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
            /// A tuple fixes its rank, so the only question is the axes: all
            /// statically sized means `Static`, otherwise `Mixed`. One `Shape`
            /// impl covers `(U2, U3)` and `(U2, usize)` alike, which is why
            /// this is folded from the axes rather than written per rank.
            const PROOF: $crate::exec::ProofLevel =
                $crate::exec::ProofLevel::of_ranked(true $(&& $name::STATIC_SIZE)*);

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
            /// Rank comes from the array length; every size is a runtime
            /// `usize`. That is `Mixed` by definition, including at rank 0,
            /// where the claim is vacuous but costs nothing to keep uniform.
            const PROOF: $crate::exec::ProofLevel = $crate::exec::ProofLevel::Mixed;

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
    /// A scalar has no axis that could be dynamic, so everything about it is
    /// known at compile time.
    const PROOF: crate::exec::ProofLevel = crate::exec::ProofLevel::Static;

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

// Rank ladder: rank-preserving, so it reaches `MAX_RANK` itself.
incin_macros::rank_sweep!(ranked_pairs => impl_shape_for_tuple);

macro_rules! impl_append_dim_for_tuple {
    ($($name:ident),*) => {
        impl< $($name: Dim,)* Append: Dim > AppendDim<Append> for ( $($name,)*) {
            /// `Self`'s dimensions with `Append` appended at the end.
            type Output = ( $($name,)* Append);
        }
    };
}

// `AppendDim`'s `Output` is rank N+1 and is bounded by `Shape`, so its input
// ceiling is one below `MAX_RANK` — at `MAX_RANK` the output tuple would have
// no `Shape` impl. This is a real ceiling, not a gap.
incin_macros::rank_sweep!(names => impl_append_dim_for_tuple, max = 7);
// Note: Rust standard library only implements traits (Debug, Eq, etc.) for tuples up to size 12.
// We cap at rank 8 — appending to a 7-dim tuple yields rank 8, the maximum.

macro_rules! impl_replace_last_dim_for_tuple {
    // Variadic, replacing twelve hand-written arms. The last four described
    // tuples of rank 9 through 12 — ranks at which no tuple implements `Shape`
    // at all, so those impls could never be selected. `ReplaceLastDim` is
    // rank-preserving, so its ceiling is `MAX_RANK` exactly.
    ($($n:ident),+) => { impl_replace_last_dim_for_tuple!(@split [] $($n),+); };
    (@split [$($acc:ident)*] $last:ident) => {
        impl<$($acc: Dim,)* $last: Dim, NewDim: Dim> ReplaceLastDim<NewDim>
            for ($($acc,)* $last,)
        {
            /// `Self`'s dimensions with the last one replaced by `NewDim`.
            type Output = ($($acc,)* NewDim,);
        }
    };
    (@split [$($acc:ident)*] $head:ident, $($rest:ident),+) => {
        impl_replace_last_dim_for_tuple!(@split [$($acc)* $head] $($rest),+);
    };
}

// Rank-preserving. This family used to run to rank 12, four above `Shape`'s
// ceiling: those impls could never be selected, because no tuple above rank 8
// implements `Shape` in the first place.
incin_macros::rank_sweep!(names => impl_replace_last_dim_for_tuple);

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
    // Variadic, so one arm covers every rank the sweep asks for. This used to
    // be six hand-written arms, which is why `EndsWith` capped at rank 6 while
    // `Shape` reached 8: the ceiling was the arm count, not a property of the
    // rule. `EndsWith` is rank-preserving and has no reason to cap below
    // `MAX_RANK`.
    ($($n:ident),+) => { impl_ends_with_for_tuple!(@split [] $($n),+); };
    // Peel one name at a time into the accumulator until only the last
    // remains; `macro_rules!` cannot match "all but the final token" directly.
    (@split [$($acc:ident)*] $last:ident) => {
        // The trailing comma is required: at rank 1 the accumulator is empty and
        // `($last)` is a parenthesized type, not a 1-tuple.
        impl<$($acc: Dim,)* $last: Dim> EndsWith<$last> for ($($acc,)* $last,) {}
    };
    (@split [$($acc:ident)*] $head:ident, $($rest:ident),+) => {
        impl_ends_with_for_tuple!(@split [$($acc)* $head] $($rest),+);
    };
}

// Rank-preserving marker.
incin_macros::rank_sweep!(names => impl_ends_with_for_tuple);

macro_rules! impl_has_channels_1d_for_tuple {
    // Channels sit at the second-to-last axis: `[.., C, L]`. The rule is
    // rank-preserving and cares only about the last two axes, so it holds for
    // every rank from 2 up. It used to be a single arm covering rank 3 alone —
    // so `(C, L)` itself, which the trait's own documentation names as valid,
    // did not implement it.
    ($($n:ident),+) => { impl_has_channels_1d_for_tuple!(@split [] $($n),+); };
    // The two-element arm must precede the recursive one, or the recursion
    // consumes the pair it is meant to terminate on.
    (@split [$($acc:ident)*] $c:ident, $l:ident) => {
        impl<$($acc: Dim,)* $c: Dim, $l: Dim> HasChannels1D<$c> for ($($acc,)* $c, $l,) {}
    };
    (@split [$($acc:ident)*] $head:ident, $($rest:ident),+) => {
        impl_has_channels_1d_for_tuple!(@split [$($acc)* $head] $($rest),+);
    };
}

// Conv1d/BatchNorm1d: (.., Channels, Length), from rank 2 to the ceiling.
incin_macros::rank_sweep!(names => impl_has_channels_1d_for_tuple, min = 2);

macro_rules! impl_has_channels_2d_for_tuple {
    // Channels sit at the third-to-last axis: `[.., C, H, W]`. As with the 1D
    // form this held for exactly one rank, so `(C, H, W)` did not implement it.
    ($($n:ident),+) => { impl_has_channels_2d_for_tuple!(@split [] $($n),+); };
    (@split [$($acc:ident)*] $c:ident, $h:ident, $w:ident) => {
        impl<$($acc: Dim,)* $c: Dim, $h: Dim, $w: Dim> HasChannels2D<$c>
            for ($($acc,)* $c, $h, $w,)
        {
        }
    };
    (@split [$($acc:ident)*] $head:ident, $($rest:ident),+) => {
        impl_has_channels_2d_for_tuple!(@split [$($acc)* $head] $($rest),+);
    };
}

// Conv2d/BatchNorm2d: (.., Channels, Height, Width), from rank 3 to the ceiling.
incin_macros::rank_sweep!(names => impl_has_channels_2d_for_tuple, min = 3);
