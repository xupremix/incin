use crate::shapes::ShapeBuf;
use crate::shapes::broadcast::ReverseShape;
/// The deepest rank whose per-axis extents can be carried statically.
///
/// This bounds only the compile-time construction buffer behind
/// [`Shape::STATIC_EXTENTS`], never the rank of a shape: a deeper shape is
/// still a perfectly good `Shape` and still works everywhere, it simply reports
/// no per-axis extents and forgoes the specialisations that need them.
///
/// Eight covers the ranks the operation catalog admits -- the deepest is
/// `Conv3d`'s five -- with headroom. The test suite does construct deeper
/// shapes than this (`tensor_ops` builds a rank-18 one), which is precisely why
/// exceeding the bound has to degrade rather than fail.
pub const MAX_STATIC_RANK: usize = 8;

/// Prepends one axis to an extent buffer, dropping what falls off the end.
///
/// The dropped tail never escapes: `STATIC_EXTENTS` reports an empty slice for
/// any rank past the buffer rather than the prefix that survived here.
#[doc(hidden)]
#[must_use]
pub const fn prepend_extent(
    head: Option<usize>,
    tail: [Option<usize>; MAX_STATIC_RANK],
) -> [Option<usize>; MAX_STATIC_RANK] {
    let mut out = [None; MAX_STATIC_RANK];
    out[0] = head;
    let mut index = 0;
    while index < MAX_STATIC_RANK - 1 {
        out[index + 1] = tail[index];
        index += 1;
    }
    out
}

use crate::shapes::idx::{FromEnd, Here, Next};
pub use crate::shapes::rank::{AddOneRank, PreserveRank, Ranked, RemoveOneRank};
use crate::shapes::{Dim, Dyn};
use alloc::vec::Vec;
use core::fmt::Debug;
use typenum::Unsigned;

/// Forward structural cursors.  Keeping reverse cursors out of the recursive
/// `FromEnd` adapters prevents the trait solver from exploring an infinite
/// reverse-of-reverse candidate chain.
#[doc(hidden)]
pub trait ForwardCursor: crate::shapes::idx::StaticCursor {}
impl ForwardCursor for Here {}
impl<I: ForwardCursor> ForwardCursor for Next<I> {}

mod sealed {
    pub trait Shape {}
    pub trait ExpectedShapes {}
}

/// The fundamental trait for all tensor shape types.
///
/// A `Shape` encodes the rank (number of dimensions) and, optionally, the static size of each
/// dimension into the type system. The three primary implementors are:
///
/// * **`DimCons`/`Nil`** (e.g., `s![2, 3]`) - Fully static or mixed.
/// * **`Dyn`** - Fully dynamic. Shape and rank are determined at runtime.
/// * **`Ranked<R>`** - Runtime extents with a typenum-known rank.
///
/// In practice, shapes are most often constructed via the `s![]` macro.
pub trait Shape: sealed::Shape + 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    /// Compile-time validity gate for exact structural shape expressions.
    ///
    /// The default is intentionally permissive for dynamic and legacy input
    /// adapters.  `DimCons` recursively specializes it so canonical
    /// structural operations reject invalid static extent arithmetic.
    const STATIC_VALID: () = ();
    /// Compile-time rank when this shape representation preserves it.
    /// `None` denotes a runtime-rank shape such as [`Dyn`].
    const RANK: Option<usize> = None;
    /// How much of this shape the compiler settled, as opposed to the runtime.
    ///
    /// This is the shape-level lift of `Dim::STATIC_SIZE`: rank and every
    /// axis size known from the type gives
    /// [`ProofLevel::Static`](crate::shapes::ProofLevel::Static); a known rank
    /// with at least one runtime or named axis gives
    /// [`Mixed`](crate::shapes::ProofLevel::Mixed); a runtime rank gives
    /// [`Dynamic`](crate::shapes::ProofLevel::Dynamic).
    ///
    /// A lowering rule reads this to stamp the `Validated<O>` it produces
    /// without knowing which concrete shape it was handed. It defaults to
    /// `Dynamic` so a `Shape` implemented outside this crate is credited with
    /// no proof it has not shown.
    const PROOF: crate::shapes::ProofLevel = crate::shapes::ProofLevel::Dynamic;

    /// This shape's element count, when the type alone settles it.
    ///
    /// The same trick as [`PROOF`](Shape::PROOF), for the same reason. A
    /// backend's `execute_shaped<S>` is generic over `S: Shape`. Restating the
    /// count here as an `Option` lets any
    /// `S` be asked, and because `S` is a type parameter the answer is a
    /// constant: `if let Some(n) = S::STATIC_NUMEL` collapses to one arm at
    /// monomorphization rather than branching at run time.
    ///
    /// `None` is the honest default. A shape with any runtime axis has no
    /// element count until its field exists, and one implemented outside this
    /// crate is credited with nothing it has not shown - the same rule `PROOF`
    /// follows.
    const STATIC_NUMEL: Option<usize> = None;

    /// Per-axis extents the type settles, outermost first.
    ///
    /// [`STATIC_NUMEL`](Shape::STATIC_NUMEL) collapses the whole geometry to
    /// one number, which is enough to prove a packed kernel's ragged tail
    /// unreachable but not enough to index. This is the per-axis form, and it
    /// is deliberately `Option` per axis rather than all-or-nothing: a shape
    /// with a dynamic batch axis and static inner axes still lets a backend
    /// fold the inner divisions, which is the common transformer case and is
    /// exactly what an all-or-nothing answer would throw away.
    ///
    /// The slice's length is the rank when the rank is known, and zero
    /// otherwise. `&[]` is the honest default, following the same rule as
    /// `PROOF` and `STATIC_NUMEL`: a `Shape` implemented outside this crate is
    /// credited with nothing it has not shown.
    ///
    /// See [`EXTENT_BUF`](Shape::EXTENT_BUF) for why this is a slice rather
    /// than an array.
    const STATIC_EXTENTS: &'static [Option<usize>] = &[];

    /// Construction buffer behind [`STATIC_EXTENTS`](Shape::STATIC_EXTENTS).
    ///
    /// Building the extent list for `DimCons<H, T>` means prepending `H` to
    /// `T`'s list, and the result's length depends on `Self::RANK`. An array
    /// whose length is an associated const would need `generic_const_exprs`,
    /// which is unstable, so the recursion runs through a fixed-size buffer
    /// instead.
    ///
    /// The buffer is the construction vehicle, never the payload:
    /// `STATIC_EXTENTS` slices the used prefix out of it, and const promotion
    /// puts that slice in read-only static memory. So what travels on
    /// `ShapeEvidence` is a two-word slice of exact length rather than a
    /// rank-sized array, which matters because that type is `Copy` and rides
    /// along on every dispatch.
    ///
    /// A rank above [`MAX_STATIC_RANK`] reports no extents at all rather than
    /// the prefix that fits. The distinction matters: a proof that quietly
    /// claims less than the truth costs an optimisation, while one that quietly
    /// claims a *wrong* geometry is a miscompile. Deep shapes are rare and the
    /// loss is only a missed specialisation, so silence is the safe direction
    /// and it matches what the rest of this trait does when it cannot prove
    /// something.
    #[doc(hidden)]
    const EXTENT_BUF: [Option<usize>; MAX_STATIC_RANK] = [None; MAX_STATIC_RANK];

    /// The user-facing constructor argument type (e.g. a tuple of
    /// `usize`/`typenum` values, or `Vec<usize>` for `Dyn`).
    type Arg;
    /// Resolves a user-facing argument into canonical runtime dimensions.
    fn resolve(arg: Self::Arg) -> core::result::Result<ShapeBuf, crate::shapes::error::ShapeError>;
    /// Fallible raw-dimension boundary for callers that need a typed shape
    /// error.
    #[inline]
    fn try_from_dims(
        dims: &[usize],
    ) -> core::result::Result<ShapeBuf, crate::shapes::error::ShapeError> {
        Self::STATIC_VALID;
        Self::validate_dims(dims)?;
        Ok(crate::shapes::ShapeBuf::from_slice(dims))
    }

    /// Validates raw runtime dimensions against this shape's static contract.
    /// Implementors must define this proof obligation explicitly.
    fn validate_dims(dims: &[usize]) -> core::result::Result<(), crate::shapes::error::ShapeError>;
}

impl sealed::Shape for Nil {}
impl<H: Dim, T: Shape> sealed::Shape for DimCons<H, T> {}
impl<R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static> sealed::Shape for Ranked<R> {}
impl sealed::Shape for Dyn {}

/// Terminator node for canonical recursive fixed-rank shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Nil;

impl Shape for Nil {
    const RANK: Option<usize> = Some(0);
    const PROOF: crate::shapes::ProofLevel = crate::shapes::ProofLevel::Static;
    const STATIC_NUMEL: Option<usize> = Some(1);
    const STATIC_EXTENTS: &'static [Option<usize>] = &[];
    const EXTENT_BUF: [Option<usize>; MAX_STATIC_RANK] = [None; MAX_STATIC_RANK];
    type Arg = ();
    #[inline(always)]
    fn resolve(_: Self::Arg) -> core::result::Result<ShapeBuf, crate::shapes::error::ShapeError> {
        Self::try_from_dims(&[])
    }

    fn validate_dims(dims: &[usize]) -> core::result::Result<(), crate::shapes::error::ShapeError> {
        if dims.is_empty() {
            Ok(())
        } else {
            Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: dims.len(),
            })
        }
    }
}

/// Cons-cell node for canonical recursive fixed-rank shapes (`DimCons<Head, Tail>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct DimCons<H, T> {
    /// First dimension of the pair.
    pub head: H,
    /// Remaining dimensions.
    pub tail: T,
}

impl<H: Dim, T: Shape> Shape for DimCons<H, T> {
    const STATIC_VALID: () = {
        H::STATIC_VALID;
        T::STATIC_VALID;
    };
    const RANK: Option<usize> = match T::RANK {
        Some(rank) => Some(rank + 1),
        None => None,
    };
    const PROOF: crate::shapes::ProofLevel = match (H::STATIC_SIZE, T::PROOF) {
        (true, crate::shapes::ProofLevel::Static) => crate::shapes::ProofLevel::Static,
        _ => crate::shapes::ProofLevel::Mixed,
    };

    const STATIC_NUMEL: Option<usize> = match (H::STATIC, T::STATIC_NUMEL) {
        (crate::shapes::StaticExtent::Value(h), Some(t)) => h.checked_mul(t),
        _ => None,
    };

    const EXTENT_BUF: [Option<usize>; MAX_STATIC_RANK] = prepend_extent(
        match H::STATIC {
            crate::shapes::StaticExtent::Value(head) => Some(head),
            // A named or runtime axis contributes a hole rather than
            // collapsing the whole list: the axes around it are still known.
            _ => None,
        },
        T::EXTENT_BUF,
    );

    // A rank past the buffer reports nothing rather than reporting a prefix.
    // Truncation would be a wrong geometry, which is a miscompile; silence is
    // just an optimisation left on the table, and it is what the rest of this
    // trait already does when it cannot prove something.
    const STATIC_EXTENTS: &'static [Option<usize>] = match Self::RANK {
        Some(rank) if rank <= MAX_STATIC_RANK => Self::EXTENT_BUF.split_at(rank).0,
        _ => &[],
    };

    type Arg = (H::Arg, T::Arg);
    #[inline]
    fn resolve(arg: Self::Arg) -> core::result::Result<ShapeBuf, crate::shapes::error::ShapeError> {
        let head_size = H::resolve_arg(arg.0)?;
        let tail_dims = T::resolve(arg.1)?;
        let mut buf = crate::shapes::ShapeBuf::from_slice(&[head_size]);
        for &d in tail_dims.as_ref() {
            buf.push(d);
        }
        Ok(buf)
    }

    fn validate_dims(dims: &[usize]) -> core::result::Result<(), crate::shapes::error::ShapeError> {
        if dims.is_empty() {
            return Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: 0,
            });
        }
        H::validate_size(dims[0]).then_some(()).ok_or(
            crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: dims.len(),
            },
        )?;
        T::validate_dims(&dims[1..]).map_err(|error| match error {
            crate::shapes::error::ShapeError::TargetShapeRejected { operation, .. } => {
                crate::shapes::error::ShapeError::TargetShapeRejected {
                    operation,
                    rank: dims.len(),
                }
            }
            other => other,
        })
    }
}

impl DynShape for Nil {
    fn rank(_: &ShapeBuf) -> usize {
        0
    }
}

impl<H: Dim, T: Shape + DynShape> DynShape for DimCons<H, T> {
    fn rank(_: &ShapeBuf) -> usize {
        T::RANK.map_or(1, |rank| rank + 1)
    }
}

/// Prepends a dimension to a structural shape.
pub trait PrependDim<D: Dim>: Shape {
    /// The resulting shape of [`PrependDim`].
    type Output: Shape;
}

impl<D: Dim> PrependDim<D> for Nil {
    type Output = DimCons<D, Nil>;
}

impl<D: Dim, H: Dim, T: Shape> PrependDim<D> for DimCons<H, T> {
    type Output = DimCons<D, DimCons<H, T>>;
}

impl<D: Dim> AppendDim<D> for Nil {
    type Output = DimCons<D, Nil>;
}

impl<D: Dim, H: Dim, T: Shape> AppendDim<D> for DimCons<H, T>
where
    T: AppendDim<D>,
{
    type Output = DimCons<H, <T as AppendDim<D>>::Output>;
}

/// Concatenates two structural shapes.
pub trait StructuralConcatShape<Rhs: Shape>: Shape {
    /// The resulting shape of [`StructuralConcatShape`].
    type Output: Shape;
}

impl<Rhs: Shape> StructuralConcatShape<Rhs> for Nil {
    type Output = Rhs;
}

impl<H: Dim, T: Shape, Rhs: Shape> StructuralConcatShape<Rhs> for DimCons<H, T>
where
    T: StructuralConcatShape<Rhs>,
{
    type Output = DimCons<H, <T as StructuralConcatShape<Rhs>>::Output>;
}

/// Accesses the dimension at structural cursor `Cursor`.
pub trait At<Cursor>: Shape {
    /// The dimension at the resolved position.
    type Output: Dim;
}

impl<H: Dim, T: Shape> At<crate::shapes::idx::Here> for DimCons<H, T> {
    type Output = H;
}

impl<H: Dim, T: Shape, SubCursor> At<crate::shapes::idx::Next<SubCursor>> for DimCons<H, T>
where
    T: At<SubCursor>,
{
    type Output = <T as At<SubCursor>>::Output;
}

/// Positional access counting from the end.
pub trait AtFromEnd<Cursor>: Shape {
    /// The dimension at the from-end position.
    type Output: Dim;
}

#[doc(hidden)]
pub trait LastDim: Shape {
    type Output: Dim;
}

impl<H: Dim> LastDim for DimCons<H, Nil> {
    type Output = H;
}

impl<H: Dim, T: Shape> LastDim for DimCons<H, T>
where
    T: LastDim,
{
    type Output = <T as LastDim>::Output;
}

impl<H: Dim, T: Shape> AtFromEnd<Here> for DimCons<H, T>
where
    DimCons<H, T>: LastDim,
{
    type Output = <DimCons<H, T> as LastDim>::Output;
}

impl<H: Dim, T: Shape, Cursor: ForwardCursor> AtFromEnd<Next<Cursor>> for DimCons<H, T>
where
    DimCons<H, T>: RemoveLastDim,
    <DimCons<H, T> as RemoveLastDim>::Output: AtFromEnd<Cursor>,
{
    type Output = <<DimCons<H, T> as RemoveLastDim>::Output as AtFromEnd<Cursor>>::Output;
}

/// Removes the dimension at structural cursor `Cursor`.
pub trait RemoveAt<Cursor>: Shape {
    /// The resulting shape of [`RemoveAt`].
    type Output: Shape;
}

impl<H: Dim, T: Shape> RemoveAt<crate::shapes::idx::Here> for DimCons<H, T> {
    type Output = T;
}

impl<H: Dim, T: Shape, SubCursor> RemoveAt<crate::shapes::idx::Next<SubCursor>> for DimCons<H, T>
where
    T: RemoveAt<SubCursor>,
{
    type Output = DimCons<H, <T as RemoveAt<SubCursor>>::Output>;
}

impl RemoveAt<crate::shapes::idx::Here> for Dyn {
    type Output = Dyn;
}

impl<SubCursor> RemoveAt<crate::shapes::idx::Next<SubCursor>> for Dyn {
    type Output = Dyn;
}

/// Removal counting from the end.
pub trait RemoveFromEnd<Cursor>: Shape {
    /// The resulting shape of [`RemoveFromEnd`].
    type Output: Shape;
}

#[doc(hidden)]
pub trait RemoveLastDim: Shape {
    type Output: Shape;
}

impl<H: Dim> RemoveLastDim for DimCons<H, Nil> {
    type Output = Nil;
}

impl<H: Dim, H2: Dim, T: Shape> RemoveLastDim for DimCons<H, DimCons<H2, T>>
where
    DimCons<H2, T>: RemoveLastDim,
{
    type Output = DimCons<H, <DimCons<H2, T> as RemoveLastDim>::Output>;
}

impl<H: Dim, T: Shape> RemoveFromEnd<Here> for DimCons<H, T>
where
    DimCons<H, T>: RemoveLastDim,
{
    type Output = <DimCons<H, T> as RemoveLastDim>::Output;
}

impl<H: Dim, T: Shape, Cursor: ForwardCursor> RemoveFromEnd<Next<Cursor>> for DimCons<H, T>
where
    T: RemoveFromEnd<Cursor>,
{
    type Output = DimCons<H, <T as RemoveFromEnd<Cursor>>::Output>;
}

/// Replaces the dimension at structural cursor `Cursor` with `NewDim`.
pub trait ReplaceAt<Cursor, NewDim: Dim>: Shape {
    /// The resulting shape of [`ReplaceAt`].
    type Output: Shape;
}

impl<S, Cursor, NewDim> ReplaceAt<crate::shapes::idx::FromEnd<Cursor>, NewDim> for S
where
    Cursor: ForwardCursor,
    NewDim: Dim,
    S: ReplaceFromEnd<Cursor, NewDim>,
{
    type Output = <S as ReplaceFromEnd<Cursor, NewDim>>::Output;
}

impl<H: Dim, T: Shape, NewDim: Dim> ReplaceAt<crate::shapes::idx::Here, NewDim> for DimCons<H, T> {
    type Output = DimCons<NewDim, T>;
}

impl<H: Dim, T: Shape, SubCursor, NewDim: Dim>
    ReplaceAt<crate::shapes::idx::Next<SubCursor>, NewDim> for DimCons<H, T>
where
    T: ReplaceAt<SubCursor, NewDim>,
{
    type Output = DimCons<H, <T as ReplaceAt<SubCursor, NewDim>>::Output>;
}

/// Replacement counting from the end.
pub trait ReplaceFromEnd<Cursor, NewDim: Dim>: Shape {
    /// The resulting shape of [`ReplaceFromEnd`].
    type Output: Shape;
}

impl<H: Dim, T: Shape, NewDim: Dim> ReplaceFromEnd<Here, NewDim> for DimCons<H, T>
where
    DimCons<H, T>: ReplaceLastDim<NewDim>,
{
    type Output = <DimCons<H, T> as ReplaceLastDim<NewDim>>::Output;
}

impl<H: Dim, T: Shape, Cursor: ForwardCursor, NewDim: Dim> ReplaceFromEnd<Next<Cursor>, NewDim>
    for DimCons<H, T>
where
    T: ReplaceFromEnd<Cursor, NewDim>,
{
    type Output = DimCons<H, <T as ReplaceFromEnd<Cursor, NewDim>>::Output>;
}

/// Inserts `NewDim` before structural cursor `Cursor`.
pub trait InsertAt<Cursor, NewDim: Dim>: Shape {
    /// The resulting shape of [`InsertAt`].
    type Output: Shape;
}

impl<NewDim: Dim> InsertAt<crate::shapes::idx::Here, NewDim> for Nil {
    type Output = DimCons<NewDim, Nil>;
}

impl<H: Dim, T: Shape, NewDim: Dim> InsertAt<crate::shapes::idx::Here, NewDim> for DimCons<H, T> {
    type Output = DimCons<NewDim, DimCons<H, T>>;
}

impl<H: Dim, T: Shape, SubCursor, NewDim: Dim> InsertAt<crate::shapes::idx::Next<SubCursor>, NewDim>
    for DimCons<H, T>
where
    T: InsertAt<SubCursor, NewDim>,
{
    type Output = DimCons<H, <T as InsertAt<SubCursor, NewDim>>::Output>;
}

/// Inserts a dimension at a static axis, including from-end cursors.
pub trait InsertAxis<Cursor, NewDim: Dim>: Shape {
    /// The resulting shape of [`InsertAxis`].
    type Output: Shape;
}

impl<S, Cursor, NewDim> InsertAxis<Cursor, NewDim> for S
where
    Cursor: ForwardCursor,
    S: InsertAt<Cursor, NewDim>,
    <S as InsertAt<Cursor, NewDim>>::Output: Shape,
    NewDim: Dim,
{
    type Output = <S as InsertAt<Cursor, NewDim>>::Output;
}

impl<S, Cursor, NewDim> InsertAxis<crate::shapes::idx::FromEnd<Cursor>, NewDim> for S
where
    Cursor: ForwardCursor,
    S: ReverseShape,
    S::Output: InsertAt<Cursor, NewDim>,
    <S::Output as InsertAt<Cursor, NewDim>>::Output: ReverseShape,
    NewDim: Dim,
{
    type Output = <<S::Output as InsertAt<Cursor, NewDim>>::Output as ReverseShape>::Output;
}

/// Swaps two dimensions in a structural shape.
///
/// This is deliberately expressed in terms of the generic cursor operations
/// above.  It therefore works for arbitrary structural ranks without a
/// generated rank ladder, and moves each complete dimension type (including
/// any semantic name metadata) rather than reconstructing it from a size.
pub trait SwapAt<Left, Right>: Shape {
    /// The resulting shape of [`SwapAt`].
    type Output: Shape;
}

impl<H: Dim, T: Shape> SwapAt<Here, Here> for DimCons<H, T> {
    type Output = DimCons<H, T>;
}

impl<H: Dim, T: Shape, R, RD> SwapAt<Here, Next<R>> for DimCons<H, T>
where
    R: ForwardCursor,
    T: At<R, Output = RD> + ReplaceAt<R, H>,
    RD: Dim,
{
    type Output = DimCons<RD, <T as ReplaceAt<R, H>>::Output>;
}

impl<H: Dim, T: Shape, L, LD> SwapAt<Next<L>, Here> for DimCons<H, T>
where
    L: ForwardCursor,
    T: At<L, Output = LD> + ReplaceAt<L, H>,
    LD: Dim,
{
    type Output = DimCons<LD, <T as ReplaceAt<L, H>>::Output>;
}

impl<H: Dim, T: Shape, L, R> SwapAt<Next<L>, Next<R>> for DimCons<H, T>
where
    L: ForwardCursor,
    R: ForwardCursor,
    T: SwapAt<L, R>,
{
    type Output = DimCons<H, <T as SwapAt<L, R>>::Output>;
}

// From-end selectors use a separate algebra.  The helper traits below make
// the two edge cases explicit, so a proof of a forward cursor never explores
// an unconstrained reverse cursor candidate.
/// Swap counting both positions from the end.
pub trait SwapFromEndAt<Left, Right>: Shape {
    /// The resulting shape of [`SwapFromEndAt`].
    type Output: Shape;
}

#[doc(hidden)]
pub trait SwapLastWith<Cursor>: Shape {
    type Output: Shape;
}

impl<H: Dim, T: Shape, Last> SwapLastWith<Here> for DimCons<H, T>
where
    DimCons<H, T>: AtFromEnd<Here, Output = Last> + ReplaceAt<FromEnd<Here>, H>,
    <DimCons<H, T> as ReplaceAt<FromEnd<Here>, H>>::Output: ReplaceAt<Here, Last>,
    Last: Dim,
{
    type Output =
        <<DimCons<H, T> as ReplaceAt<FromEnd<Here>, H>>::Output as ReplaceAt<Here, Last>>::Output;
}

impl<H: Dim, T: Shape, R> SwapLastWith<Next<R>> for DimCons<H, T>
where
    T: SwapLastWith<R>,
{
    type Output = DimCons<H, <T as SwapLastWith<R>>::Output>;
}

#[doc(hidden)]
pub trait SwapFirstWith<Cursor>: Shape {
    type Output: Shape;
}

impl<S: Shape> SwapFirstWith<Here> for S {
    type Output = S;
}

impl<H: Dim, T: Shape, R, Other> SwapFirstWith<Next<R>> for DimCons<H, T>
where
    T: At<R, Output = Other> + ReplaceAt<R, H>,
    Other: Dim,
{
    type Output = DimCons<Other, <T as ReplaceAt<R, H>>::Output>;
}

impl<S, R> SwapFromEndAt<Here, FromEnd<R>> for S
where
    R: ForwardCursor,
    S: ReverseShape,
    S::Output: SwapLastWith<R>,
    <S::Output as SwapLastWith<R>>::Output: ReverseShape,
{
    type Output = <<S::Output as SwapLastWith<R>>::Output as ReverseShape>::Output;
}

impl<H: Dim, T: Shape, L, R> SwapFromEndAt<Next<L>, FromEnd<R>> for DimCons<H, T>
where
    L: ForwardCursor,
    R: ForwardCursor,
    T: SwapFromEndAt<L, FromEnd<R>>,
{
    type Output = DimCons<H, <T as SwapFromEndAt<L, FromEnd<R>>>::Output>;
}

impl<S, L, R> SwapFromEndAt<FromEnd<L>, FromEnd<R>> for S
where
    L: ForwardCursor,
    R: ForwardCursor,
    S: ReverseShape,
    S::Output: SwapAt<L, R>,
    <S::Output as SwapAt<L, R>>::Output: ReverseShape,
{
    type Output = <<S::Output as SwapAt<L, R>>::Output as ReverseShape>::Output;
}

impl<H: Dim, T: Shape, L, R> SwapFromEndAt<FromEnd<L>, Next<R>> for DimCons<H, T>
where
    L: ForwardCursor,
    R: ForwardCursor,
    T: SwapFromEndAt<FromEnd<L>, R>,
{
    type Output = DimCons<H, <T as SwapFromEndAt<FromEnd<L>, R>>::Output>;
}

/// Multiplies all dimensions in a structural shape into a single product dimension.
pub trait ProductDims: Shape {
    /// The dimension resulting from the product.
    type Output: Dim;
}

/// Collapses an inclusive structural cursor range into one extent.
///
/// The recursion consumes the selected prefix and then rebuilds the untouched
/// suffix, so this operation has no rank-specific implementations.
pub trait FlattenAt<Start, End>: Shape {
    /// The resulting shape of [`FlattenAt`].
    type Output: Shape;
}

/// From-end flattening has separate dispatch to keep reverse recursion out of
/// the positive-axis implementation.
pub trait FlattenFromEndAt<Start, End>: Shape {
    /// The resulting shape of [`FlattenFromEndAt`].
    type Output: Shape;
}

#[doc(hidden)]
pub trait FlattenPrefix<End>: Shape {
    type Product: Dim;
    type Suffix: Shape;
}

impl<H: Dim, T: Shape> FlattenPrefix<crate::shapes::idx::Here> for DimCons<H, T> {
    type Product = H;
    type Suffix = T;
}

impl<H: Dim, T: Shape, End> FlattenPrefix<crate::shapes::idx::Next<End>> for DimCons<H, T>
where
    T: FlattenPrefix<End>,
    <T as FlattenPrefix<End>>::Product: Dim,
{
    type Product = crate::shapes::dim::MulDim<H, <T as FlattenPrefix<End>>::Product>;
    type Suffix = <T as FlattenPrefix<End>>::Suffix;
}

impl<S: Shape> FlattenAt<crate::shapes::idx::Here, crate::shapes::idx::Here> for S
where
    S: FlattenPrefix<crate::shapes::idx::Here>,
{
    type Output = DimCons<
        <S as FlattenPrefix<crate::shapes::idx::Here>>::Product,
        <S as FlattenPrefix<crate::shapes::idx::Here>>::Suffix,
    >;
}

impl<End, S: Shape> FlattenAt<crate::shapes::idx::Here, crate::shapes::idx::Next<End>> for S
where
    End: ForwardCursor,
    S: FlattenPrefix<crate::shapes::idx::Next<End>>,
{
    type Output = DimCons<
        <S as FlattenPrefix<crate::shapes::idx::Next<End>>>::Product,
        <S as FlattenPrefix<crate::shapes::idx::Next<End>>>::Suffix,
    >;
}

impl<H: Dim, T: Shape, Start, End>
    FlattenAt<crate::shapes::idx::Next<Start>, crate::shapes::idx::Next<End>> for DimCons<H, T>
where
    Start: ForwardCursor,
    End: ForwardCursor,
    T: FlattenAt<Start, End>,
{
    type Output = DimCons<H, <T as FlattenAt<Start, End>>::Output>;
}

/// A forward start with a from-end end is handled by reversing the shape,
/// flattening the corresponding forward prefix, and reversing the result.
/// This keeps the range algebra rank-independent and preserves static proof.
#[doc(hidden)]
pub trait FlattenSuffix<Start>: Shape {
    type Output: Shape;
}

impl<S: Shape> FlattenSuffix<Here> for S
where
    S: ProductDims,
{
    type Output = DimCons<<S as ProductDims>::Output, Nil>;
}

impl<H: Dim, T: Shape, Start> FlattenSuffix<Next<Start>> for DimCons<H, T>
where
    Start: ForwardCursor,
    T: FlattenSuffix<Start>,
{
    type Output = DimCons<H, <T as FlattenSuffix<Start>>::Output>;
}

impl<End, S> FlattenFromEndAt<Here, FromEnd<End>> for S
where
    End: ForwardCursor,
    S: ReverseShape,
    S::Output: FlattenSuffix<End>,
    <S::Output as FlattenSuffix<End>>::Output: ReverseShape,
{
    type Output = <<S::Output as FlattenSuffix<End>>::Output as ReverseShape>::Output;
}

impl<Start, End, H: Dim, T: Shape>
    FlattenFromEndAt<crate::shapes::idx::Next<Start>, crate::shapes::idx::FromEnd<End>>
    for DimCons<H, T>
where
    T: FlattenFromEndAt<Start, crate::shapes::idx::FromEnd<End>>,
{
    type Output =
        DimCons<H, <T as FlattenFromEndAt<Start, crate::shapes::idx::FromEnd<End>>>::Output>;
}

/// A range with both endpoints counted from the end is mapped through the
/// existing reverse-shape algebra. Reversing the endpoint order is necessary
/// because the source range's start becomes the reversed range's end.
impl<Start, End, H: Dim, T: Shape>
    FlattenFromEndAt<crate::shapes::idx::FromEnd<Start>, crate::shapes::idx::FromEnd<End>>
    for DimCons<H, T>
where
    DimCons<H, T>: ReverseShape,
    <DimCons<H, T> as ReverseShape>::Output: FlattenAt<End, Start>,
    <<DimCons<H, T> as ReverseShape>::Output as FlattenAt<End, Start>>::Output: ReverseShape,
{
    type Output =
        <<<DimCons<H, T> as ReverseShape>::Output as FlattenAt<End, Start>>::Output as ReverseShape>::Output;
}

impl ProductDims for Nil {
    type Output = typenum::U1;
}

impl<H: Dim, T: Shape> ProductDims for DimCons<H, T>
where
    T: ProductDims,
{
    type Output = crate::shapes::dim::MulDim<H, <T as ProductDims>::Output>;
}

/// Structural suffix decomposition for the last 2 dimensions.
pub trait SplitLast2: Shape {
    /// All dimensions before the final two.
    type Prefix: Shape;
    /// Second-to-last dimension.
    type Penultimate: Dim;
    /// Final dimension.
    type Last: Dim;
}

impl<D1: Dim, D2: Dim> SplitLast2 for DimCons<D1, DimCons<D2, Nil>> {
    type Prefix = Nil;
    type Penultimate = D1;
    type Last = D2;
}

impl<H: Dim, T: Shape> SplitLast2 for DimCons<H, T>
where
    T: SplitLast2,
{
    type Prefix = DimCons<H, <T as SplitLast2>::Prefix>;
    type Penultimate = <T as SplitLast2>::Penultimate;
    type Last = <T as SplitLast2>::Last;
}

/// Structural suffix decomposition for the last 3 dimensions.
pub trait SplitLast3: Shape {
    /// All dimensions before the final three.
    type Prefix: Shape;
    /// Third-from-last dimension.
    type ThirdLast: Dim;
    /// Second-from-last dimension.
    type SecondLast: Dim;
    /// Final dimension.
    type Last: Dim;
}

impl<D1: Dim, D2: Dim, D3: Dim> SplitLast3 for DimCons<D1, DimCons<D2, DimCons<D3, Nil>>> {
    type Prefix = Nil;
    type ThirdLast = D1;
    type SecondLast = D2;
    type Last = D3;
}

impl<H: Dim, T: Shape> SplitLast3 for DimCons<H, T>
where
    T: SplitLast3,
{
    type Prefix = DimCons<H, <T as SplitLast3>::Prefix>;
    type ThirdLast = <T as SplitLast3>::ThirdLast;
    type SecondLast = <T as SplitLast3>::SecondLast;
    type Last = <T as SplitLast3>::Last;
}

/// Rebuild a typed shape buffer from computed dimensions, reporting instead of
/// panicking.
///
/// This is the checked replacement for the old optional raw-dimension chain
/// that `SHP-001` inventoried across 39 sites. The unwrap was a proof
/// obligation that no type stated and no test covered: the caller had already
/// erased a known-rank shape to a `Vec<usize>`, and then asserted the
/// round-trip back would succeed.
///
/// Prefer building the field axis by axis where the arity is known - that
/// avoids the erasure entirely and yields a
/// [`DimensionMismatch`](crate::shapes::error::ShapeError::DimensionMismatch)
/// naming the offending axis. Use this where the shape is only available
/// generically.
pub fn shape_buf_from_dims<S: Shape>(
    operation: crate::shapes::error::OperationKind,
    dims: &[usize],
) -> Result<ShapeBuf, crate::shapes::error::ShapeError> {
    S::STATIC_VALID;
    S::try_from_dims(dims).map_err(|error| match error {
        crate::shapes::error::ShapeError::TargetShapeRejected { .. } => {
            crate::shapes::error::ShapeError::TargetShapeRejected {
                operation,
                rank: dims.len(),
            }
        }
        other => other,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Bounded and verified element count (`SEC-011`).
pub struct CheckedNumel(usize);

impl CheckedNumel {
    /// Computes and validates an element count for `dims`.
    pub fn from_dims(
        operation: crate::shapes::error::OperationKind,
        dims: &[usize],
        limits: &crate::resource::ResourceLimits,
    ) -> Result<Self, crate::shapes::error::ShapeError> {
        if dims.len() > limits.max_rank {
            return Err(crate::shapes::error::ShapeError::RankMismatch {
                operation,
                expected: crate::shapes::error::RankExpectation::AtMost(limits.max_rank),
                actual: dims.len(),
            });
        }
        if let Some(&dimension) = dims.iter().find(|&&dimension| {
            u64::try_from(dimension).map_or(true, |value| value > limits.max_dimension)
        }) {
            return Err(crate::shapes::error::ShapeError::InvalidParameter {
                operation,
                parameter: "dimension",
                value: dimension,
            });
        }
        crate::shapes::ShapeBuf::from_slice(dims)
            .checked_numel(operation)
            .map(Self)
    }

    #[inline]
    /// Concrete extent value.
    pub fn get(self) -> usize {
        self.0
    }
}

/// Safely computes shape element count using checked multiplication and limits (`SEC-011`).
pub fn checked_numel_from_dims(
    dims: &[usize],
    limits: &crate::resource::ResourceLimits,
) -> Result<CheckedNumel, crate::shapes::error::ShapeError> {
    CheckedNumel::from_dims(crate::shapes::error::OperationKind::Reshape, dims, limits)
}

/// A shape with runtime-accessible dimension information (rank, total elements, per-axis sizes).
///
/// All implementors of `Shape` that support dynamic rank queries also implement `DynShape`.
/// This includes both `Dyn` and fully static structural shapes. Operations that need to introspect
/// the shape at runtime (e.g., computing strides) require a `DynShape` bound.
pub trait DynShape: Shape {
    /// Returns the number of dimensions.
    fn rank(shape: &ShapeBuf) -> usize;
    /// Returns the total element count (product of all dimension sizes) after
    /// checking for arithmetic overflow.
    fn checked_numel(
        shape: &ShapeBuf,
        operation: crate::shapes::error::OperationKind,
    ) -> Result<usize, crate::shapes::error::ShapeError> {
        shape.checked_numel(operation)
    }

    /// Returns the total element count for shape metadata that has already
    /// crossed a checked tensor-construction boundary.
    fn numel(shape: &ShapeBuf) -> usize {
        Self::checked_numel(shape, crate::shapes::error::OperationKind::Storage)
            .expect("validated tensor shape must have a representable element count")
    }
}

/// The single authoritative relationship between type-level shape information
/// (`S`) and runtime extents.
///
/// Runtime dimensions are stored only in [`ShapeBuf`]. The type parameter
/// carries compile-time knowledge; it is not a second runtime representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeValue<S: Shape> {
    dims: crate::shapes::ShapeBuf,
    marker: core::marker::PhantomData<fn() -> S>,
}

impl<S: Shape> ShapeValue<S> {
    /// Constructs a validated relationship between `S` and runtime dimensions.
    #[inline]
    pub fn try_new(dims: ShapeBuf) -> Result<Self, crate::shapes::error::ShapeError> {
        S::STATIC_VALID;
        S::validate_dims(dims.as_ref())?;
        Ok(Self::from_validated(dims))
    }

    /// Constructs a ShapeValue after an internal caller has already validated
    /// the dimensions through a shape rule or tensor construction boundary.
    /// This constructor is crate-private so downstream code cannot forge the
    /// relationship between a shape type and incompatible dimensions.
    #[inline]
    pub(crate) fn from_validated(dims: ShapeBuf) -> Self {
        S::STATIC_VALID;
        debug_assert!(S::validate_dims(dims.as_ref()).is_ok());
        Self {
            dims,
            marker: core::marker::PhantomData,
        }
    }

    /// Builds the canonical runtime shape value from already validated
    /// dimensions. Operation rules use this when their output shape is
    /// computed structurally and therefore starts directly from canonical
    /// runtime dimensions rather than a constructor adapter.
    #[inline]
    pub(crate) fn from_validated_buf(dims: crate::shapes::ShapeBuf) -> Self {
        S::STATIC_VALID;
        debug_assert!(S::validate_dims(dims.as_ref()).is_ok());
        Self {
            dims,
            marker: core::marker::PhantomData,
        }
    }

    #[inline]
    /// Borrows the shape buffer.
    pub fn shape_buf(&self) -> &crate::shapes::ShapeBuf {
        &self.dims
    }

    #[inline]
    /// Copies extents as a vector.
    pub fn dims(&self) -> Vec<usize> {
        self.dims.as_ref().to_vec()
    }

    #[inline]
    /// Proof level proven for this shape.
    pub fn proof_level(&self) -> crate::shapes::ProofLevel {
        S::PROOF
    }

    #[inline]
    /// Checked element count over possibly symbolic extents.
    pub fn checked_numel(
        &self,
        op: crate::shapes::error::OperationKind,
        limits: &crate::resource::ResourceLimits,
    ) -> Result<CheckedNumel, crate::shapes::error::ShapeError> {
        CheckedNumel::from_dims(op, &self.dims(), limits)
    }
}

/// The caller-held shape proofs for a typed dispatch's outputs.
///
/// `execute_shaped` used to take exactly one `&ShapeValue<S>`; an operation
/// with two outputs (TopK's values and indices, or any custom operation
/// inferring more than one) could not travel that path. This trait states
/// what the typed comparison needs of the expectation — one borrowed buffer
/// per output, in output order — so a single value and a tuple of values are
/// interchangeable callers. Sealed: every implementor lives here, and the
/// compare loop trusts `ARITY` to bound the iteration.
pub trait ExpectedShapes: sealed::ExpectedShapes {
    /// How many outputs the caller holds proofs for. Compared against the
    /// inference's output count before any geometry is touched.
    const ARITY: usize;

    /// Borrow each expected buffer in output order, without allocating.
    ///
    /// Exactly `ARITY` items. An array `IntoIter` keeps this allocation-free
    /// the same way `shape_buf` is: the buffers are borrowed, never copied.
    fn shape_bufs(&self) -> impl ExactSizeIterator<Item = &ShapeBuf>;

    /// The weakest proof across the outputs.
    ///
    /// Exactly `S::PROOF` for arity 1, so the single-output path observes the
    /// same level it always did. Over several outputs the descriptor can only
    /// promise what every output proves, hence the meet.
    fn combined_proof(&self) -> crate::shapes::ProofLevel;

    /// The evidence lowered with the descriptor.
    ///
    /// Exactly `ShapeEvidence::of::<S>()` for arity 1. Over several outputs
    /// the static geometry is per-shape knowledge — one shape's rank and
    /// extents say nothing about another's — so the combined value carries
    /// the weakest proof with no statics, the same posture as `Dynamic`. A
    /// backend reading statics off a multi-output descriptor would otherwise
    /// attribute one output's constants to all of them.
    fn combined_evidence(&self) -> crate::exec::ShapeEvidence;
}

impl<S: Shape> sealed::ExpectedShapes for ShapeValue<S> {}

impl<S: Shape> ExpectedShapes for ShapeValue<S> {
    const ARITY: usize = 1;

    fn shape_bufs(&self) -> impl ExactSizeIterator<Item = &ShapeBuf> {
        [self.shape_buf()].into_iter()
    }

    fn combined_proof(&self) -> crate::shapes::ProofLevel {
        S::PROOF
    }

    fn combined_evidence(&self) -> crate::exec::ShapeEvidence {
        crate::exec::ShapeEvidence::of::<S>()
    }
}

// Tuples mirror the arities `ExecuteOutput` admits: pairs through quadruples.
// Each member keeps its own shape type, so a values/indices pair like TopK's
// can hold two different proofs side by side.
impl<S0: Shape, S1: Shape> sealed::ExpectedShapes for (ShapeValue<S0>, ShapeValue<S1>) {}

impl<S0: Shape, S1: Shape> ExpectedShapes for (ShapeValue<S0>, ShapeValue<S1>) {
    const ARITY: usize = 2;

    fn shape_bufs(&self) -> impl ExactSizeIterator<Item = &ShapeBuf> {
        let (first, second) = self;
        [first.shape_buf(), second.shape_buf()].into_iter()
    }

    fn combined_proof(&self) -> crate::shapes::ProofLevel {
        S0::PROOF.meet(S1::PROOF)
    }

    fn combined_evidence(&self) -> crate::exec::ShapeEvidence {
        crate::exec::ShapeEvidence::weakened(self.combined_proof())
    }
}

impl<S0: Shape, S1: Shape, S2: Shape> sealed::ExpectedShapes
    for (ShapeValue<S0>, ShapeValue<S1>, ShapeValue<S2>)
{
}

impl<S0: Shape, S1: Shape, S2: Shape> ExpectedShapes
    for (ShapeValue<S0>, ShapeValue<S1>, ShapeValue<S2>)
{
    const ARITY: usize = 3;

    fn shape_bufs(&self) -> impl ExactSizeIterator<Item = &ShapeBuf> {
        let (first, second, third) = self;
        [
            first.shape_buf(),
            second.shape_buf(),
            third.shape_buf(),
        ]
        .into_iter()
    }

    fn combined_proof(&self) -> crate::shapes::ProofLevel {
        S0::PROOF.meet(S1::PROOF).meet(S2::PROOF)
    }

    fn combined_evidence(&self) -> crate::exec::ShapeEvidence {
        crate::exec::ShapeEvidence::weakened(self.combined_proof())
    }
}

impl<S0: Shape, S1: Shape, S2: Shape, S3: Shape> sealed::ExpectedShapes
    for (
        ShapeValue<S0>,
        ShapeValue<S1>,
        ShapeValue<S2>,
        ShapeValue<S3>,
    )
{
}

impl<S0: Shape, S1: Shape, S2: Shape, S3: Shape> ExpectedShapes
    for (
        ShapeValue<S0>,
        ShapeValue<S1>,
        ShapeValue<S2>,
        ShapeValue<S3>,
    )
{
    const ARITY: usize = 4;

    fn shape_bufs(&self) -> impl ExactSizeIterator<Item = &ShapeBuf> {
        let (first, second, third, fourth) = self;
        [
            first.shape_buf(),
            second.shape_buf(),
            third.shape_buf(),
            fourth.shape_buf(),
        ]
        .into_iter()
    }

    fn combined_proof(&self) -> crate::shapes::ProofLevel {
        S0::PROOF.meet(S1::PROOF).meet(S2::PROOF).meet(S3::PROOF)
    }

    fn combined_evidence(&self) -> crate::exec::ShapeEvidence {
        crate::exec::ShapeEvidence::weakened(self.combined_proof())
    }
}

/// Resolution specification for shape parameters.
pub trait ShapeSpec {
    /// The shape type the resulting tensor carries.
    type Shape: Shape + DynShape;

    /// Resolves to the single authoritative `ShapeValue<Self::Shape>`.
    fn resolve(self) -> Result<ShapeValue<Self::Shape>, crate::err::Error>;
}

/// Runtime reshape specification with one inferred extent.
///
/// This value-level type keeps reshape inference separate from indexing. It
/// is produced by `shape![..., infer]` and resolved only when a source tensor
/// supplies its element count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferShape<S: Shape + DynShape = Dyn> {
    extents: alloc::vec::Vec<Option<usize>>,
    marker: core::marker::PhantomData<fn() -> S>,
}

impl<S: Shape + DynShape> InferShape<S> {
    /// Creates an inference specification. Exactly one extent must be `None`.
    #[must_use]
    pub fn new(extents: alloc::vec::Vec<Option<usize>>) -> Self {
        Self {
            extents,
            marker: core::marker::PhantomData,
        }
    }

    /// Resolves the inferred extent against a source element count.
    pub fn resolve(self, source_numel: usize) -> crate::err::Result<ShapeBuf> {
        let missing = self
            .extents
            .iter()
            .filter(|extent| extent.is_none())
            .count();
        if missing != 1 || self.extents.is_empty() {
            return Err(crate::err::Error::Msg(
                "reshape inference requires exactly one `infer` extent".into(),
            ));
        }
        let known_numel = self
            .extents
            .iter()
            .flatten()
            .try_fold(1usize, |acc, &extent| acc.checked_mul(extent))
            .ok_or_else(|| crate::err::Error::Msg("reshape extent product overflowed".into()))?;
        if known_numel == 0 || !source_numel.is_multiple_of(known_numel) {
            return Err(crate::err::Error::Msg(
                "inferred reshape extent is not integral".into(),
            ));
        }
        let inferred = source_numel / known_numel;
        Ok(ShapeBuf::from_slice(
            &self
                .extents
                .into_iter()
                .map(|extent| extent.unwrap_or(inferred))
                .collect::<alloc::vec::Vec<_>>(),
        ))
    }
}

/// Unvalidated shape constructor input produced by the `shape!` macro.
///
/// This is deliberately a specification, not a `ShapeValue`: dimensions are
/// checked only when [`ShapeSpec::resolve`] joins them to the canonical
/// `ShapeBuf` representation.
#[derive(Debug, Clone)]
pub struct ShapeArgs<S: Shape + DynShape> {
    args: S::Arg,
    marker: core::marker::PhantomData<fn() -> S>,
}

impl<S: Shape + DynShape> ShapeArgs<S> {
    /// Creates constructor input without claiming that its runtime values
    /// satisfy the type-level shape specification.
    #[inline]
    pub fn new(args: S::Arg) -> Self {
        Self {
            args,
            marker: core::marker::PhantomData,
        }
    }
}

impl<S: Shape + DynShape> ShapeSpec for ShapeArgs<S> {
    type Shape = S;

    fn resolve(self) -> Result<ShapeValue<S>, crate::err::Error> {
        S::resolve(self.args)
            .map_err(crate::err::Error::Shape)
            .and_then(|dims| ShapeValue::try_new(dims).map_err(crate::err::Error::Shape))
    }
}

impl<S: Shape + DynShape> ShapeSpec for ShapeValue<S> {
    type Shape = S;

    fn resolve(self) -> Result<ShapeValue<S>, crate::err::Error> {
        Ok(self)
    }
}

impl<const N: usize> ShapeSpec for [usize; N] {
    type Shape = Dyn;

    fn resolve(self) -> Result<ShapeValue<Self::Shape>, crate::err::Error> {
        ShapeValue::try_new(crate::shapes::ShapeBuf::from_slice(&self))
            .map_err(crate::err::Error::Shape)
    }
}

impl ShapeSpec for Vec<usize> {
    type Shape = Dyn;

    fn resolve(self) -> Result<ShapeValue<Dyn>, crate::err::Error> {
        ShapeValue::try_new(crate::shapes::ShapeBuf::from_slice(&self))
            .map_err(crate::err::Error::Shape)
    }
}

impl ShapeSpec for &[usize] {
    type Shape = Dyn;

    fn resolve(self) -> Result<ShapeValue<Dyn>, crate::err::Error> {
        ShapeValue::try_new(crate::shapes::ShapeBuf::from_slice(self))
            .map_err(crate::err::Error::Shape)
    }
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

/// Marker: `Self`'s last dimension is `D` - used to bound layer
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

impl<NewDim: Dim> ReplaceLastDim<NewDim> for Nil {
    type Output = Nil;
}
/// Replaces only the trailing dimension.
pub trait ReplaceLastTail<NewDim: Dim>: Shape {
    /// The resulting shape of [`ReplaceLastTail`].
    type Output: Shape;
}
impl<NewDim: Dim> ReplaceLastTail<NewDim> for Nil {
    type Output = DimCons<NewDim, Nil>;
}
impl<H: Dim, NewDim: Dim> ReplaceLastTail<NewDim> for DimCons<H, Nil> {
    type Output = DimCons<NewDim, Nil>;
}
impl<H: Dim, H2: Dim, T: Shape, NewDim: Dim> ReplaceLastTail<NewDim> for DimCons<H, DimCons<H2, T>>
where
    DimCons<H2, T>: ReplaceLastTail<NewDim>,
{
    type Output = DimCons<H, <DimCons<H2, T> as ReplaceLastTail<NewDim>>::Output>;
}
impl<H: Dim, T: Shape, NewDim: Dim> ReplaceLastDim<NewDim> for DimCons<H, T>
where
    T: ReplaceLastTail<NewDim>,
{
    type Output = DimCons<H, <T as ReplaceLastTail<NewDim>>::Output>;
}

impl<D: Dim> EndsWith<D> for DimCons<D, Nil> {}
impl<H: Dim, H2: Dim, T: Shape, D: Dim> EndsWith<D> for DimCons<H, DimCons<H2, T>> where
    DimCons<H2, T>: EndsWith<D>
{
}

impl<H: Dim, T: Shape, D: Dim> HasChannels1D<D> for DimCons<H, T> where
    DimCons<H, T>: AtFromEnd<crate::shapes::idx::Next<crate::shapes::idx::Here>, Output = D>
{
}

impl<H: Dim, T: Shape, D: Dim> HasChannels2D<D> for DimCons<H, T> where
    DimCons<H, T>: AtFromEnd<
            crate::shapes::idx::Next<crate::shapes::idx::Next<crate::shapes::idx::Here>>,
            Output = D,
        >
{
}

///
/// --- Dyn ---
///
impl Shape for Dyn {
    const RANK: Option<usize> = None;
    /// Not even the rank is known until the shape exists, which is the whole
    /// point of `Dyn`. Stated rather than inherited from the default so that
    /// changing the default cannot silently upgrade it.
    const PROOF: crate::shapes::ProofLevel = crate::shapes::ProofLevel::Dynamic;

    /// The user-facing constructor argument type for this concrete shape.
    type Arg = Vec<usize>;
    /// Runtime values for this shape are held in `ShapeBuf`.
    /// Converts a user-facing argument into canonical `ShapeBuf` storage.
    fn resolve(arg: Self::Arg) -> core::result::Result<ShapeBuf, crate::shapes::error::ShapeError> {
        Self::try_from_dims(&arg).map(|_| arg.into_iter().collect())
    }
    fn validate_dims(_: &[usize]) -> core::result::Result<(), crate::shapes::error::ShapeError> {
        Ok(())
    }
}

impl DynShape for Dyn {
    #[inline(always)]
    /// Returns the number of dimensions.
    fn rank(shape: &ShapeBuf) -> usize {
        shape.len()
    }
}

impl<D: Dim> AppendDim<D> for Dyn {
    /// `Self`'s dimensions with `D` appended at the end.
    type Output = Dyn;
}

impl<NewDim: Dim> ReplaceLastDim<NewDim> for Dyn {
    /// `Self`'s dimensions with the last one replaced by `NewDim`.
    type Output = Dyn;
}

#[cfg(test)]
mod tests;
