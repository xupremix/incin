use crate::shapes::ShapeBuf;
use crate::shapes::broadcast::ReverseShape;
use crate::shapes::idx::{FromEnd, Here, Next};
use crate::shapes::{Dim, Dyn};
use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::{Add, Sub};
use typenum::{U1, Unsigned};

/// Forward structural cursors.  Keeping reverse cursors out of the recursive
/// `FromEnd` adapters prevents the trait solver from exploring an infinite
/// reverse-of-reverse candidate chain.
#[doc(hidden)]
pub trait ForwardCursor {}
impl ForwardCursor for Here {}
impl<I: ForwardCursor> ForwardCursor for Next<I> {}

mod sealed {
    pub trait Shape {}
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
    pub head: H,
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

pub trait AtFromEnd<Cursor>: Shape {
    type Output: Dim;
}

impl<H: Dim, T: Shape, Cursor: ForwardCursor> AtFromEnd<Cursor> for DimCons<H, T>
where
    DimCons<H, T>: ReverseShape,
    <DimCons<H, T> as ReverseShape>::Output: At<Cursor>,
{
    type Output = <<DimCons<H, T> as ReverseShape>::Output as At<Cursor>>::Output;
}

/// Removes the dimension at structural cursor `Cursor`.
pub trait RemoveAt<Cursor>: Shape {
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

pub trait RemoveFromEnd<Cursor>: Shape {
    type Output: Shape;
}

impl<H: Dim, T: Shape, Cursor: ForwardCursor> RemoveFromEnd<Cursor> for DimCons<H, T>
where
    DimCons<H, T>: ReverseShape,
    <DimCons<H, T> as ReverseShape>::Output: RemoveAt<Cursor>,
    <<DimCons<H, T> as ReverseShape>::Output as RemoveAt<Cursor>>::Output: ReverseShape,
{
    type Output =
        <<<DimCons<H, T> as ReverseShape>::Output as RemoveAt<Cursor>>::Output as ReverseShape>::Output;
}

/// Replaces the dimension at structural cursor `Cursor` with `NewDim`.
pub trait ReplaceAt<Cursor, NewDim: Dim>: Shape {
    type Output: Shape;
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

pub trait ReplaceFromEnd<Cursor, NewDim: Dim>: Shape {
    type Output: Shape;
}

impl<H: Dim, T: Shape, Cursor: ForwardCursor, NewDim: Dim> ReplaceFromEnd<Cursor, NewDim>
    for DimCons<H, T>
where
    DimCons<H, T>: ReverseShape,
    <DimCons<H, T> as ReverseShape>::Output: ReplaceAt<Cursor, NewDim>,
    <<DimCons<H, T> as ReverseShape>::Output as ReplaceAt<Cursor, NewDim>>::Output: ReverseShape,
{
    type Output =
        <<<DimCons<H, T> as ReverseShape>::Output as ReplaceAt<Cursor, NewDim>>::Output as ReverseShape>::Output;
}

/// Inserts `NewDim` before structural cursor `Cursor`.
pub trait InsertAt<Cursor, NewDim: Dim>: Shape {
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
    type Output: Shape;
}

impl<H: Dim, T: Shape> SwapAt<Here, Here> for DimCons<H, T> {
    type Output = DimCons<H, T>;
}

impl<H: Dim, T: Shape, R, RD> SwapAt<Here, Next<R>> for DimCons<H, T>
where
    T: At<R, Output = RD> + ReplaceAt<R, H>,
    RD: Dim,
{
    type Output = DimCons<RD, <T as ReplaceAt<R, H>>::Output>;
}

impl<H: Dim, T: Shape, L, LD> SwapAt<Next<L>, Here> for DimCons<H, T>
where
    T: At<L, Output = LD> + ReplaceAt<L, H>,
    LD: Dim,
{
    type Output = DimCons<LD, <T as ReplaceAt<L, H>>::Output>;
}

impl<H: Dim, T: Shape, L, R> SwapAt<Next<L>, Next<R>> for DimCons<H, T>
where
    T: SwapAt<L, R>,
{
    type Output = DimCons<H, <T as SwapAt<L, R>>::Output>;
}

// Reverse selectors are part of the same structural operation, rather than
// a competing public dispatch trait.  Keeping these implementations on
// `SwapAt` lets the selector-facing layer have one non-overlapping blanket
// implementation for every `StaticCursor`.
impl<S, L, R> SwapAt<crate::shapes::idx::FromEnd<L>, R> for S
where
    L: ForwardCursor,
    R: ForwardCursor,
    S: SwapFromEnd<crate::shapes::idx::FromEnd<L>, R>,
{
    type Output = <S as SwapFromEnd<crate::shapes::idx::FromEnd<L>, R>>::Output;
}

impl<S, L, R> SwapAt<crate::shapes::idx::FromEnd<L>, crate::shapes::idx::FromEnd<R>> for S
where
    L: ForwardCursor,
    R: ForwardCursor,
    S: SwapFromEnd<crate::shapes::idx::FromEnd<L>, crate::shapes::idx::FromEnd<R>>,
{
    type Output =
        <S as SwapFromEnd<crate::shapes::idx::FromEnd<L>, crate::shapes::idx::FromEnd<R>>>::Output;
}

impl<S, R> SwapAt<Here, crate::shapes::idx::FromEnd<R>> for S
where
    R: ForwardCursor,
    S: SwapFromEnd<Here, crate::shapes::idx::FromEnd<R>>,
{
    type Output = <S as SwapFromEnd<Here, crate::shapes::idx::FromEnd<R>>>::Output;
}

impl<S, L, R> SwapAt<Next<L>, crate::shapes::idx::FromEnd<R>> for S
where
    L: ForwardCursor,
    R: ForwardCursor,
    S: SwapFromEnd<Next<L>, crate::shapes::idx::FromEnd<R>>,
{
    type Output = <S as SwapFromEnd<Next<L>, crate::shapes::idx::FromEnd<R>>>::Output;
}

pub trait SwapFromEnd<Left, Right>: Shape {
    type Output: Shape;
}

impl<S, L, R> SwapFromEnd<FromEnd<L>, R> for S
where
    R: ForwardCursor,
    S: ReverseShape,
    <S as ReverseShape>::Output: SwapFromEnd<L, FromEnd<R>>,
    <<S as ReverseShape>::Output as SwapFromEnd<L, FromEnd<R>>>::Output: ReverseShape,
{
    type Output =
        <<<S as ReverseShape>::Output as SwapFromEnd<L, FromEnd<R>>>::Output as ReverseShape>::Output;
}

impl<S, L, R> SwapFromEnd<FromEnd<L>, FromEnd<R>> for S
where
    L: ForwardCursor,
    R: ForwardCursor,
    S: ReverseShape,
    <S as ReverseShape>::Output: SwapAt<L, R>,
    <<S as ReverseShape>::Output as SwapAt<L, R>>::Output: ReverseShape,
{
    type Output = <<<S as ReverseShape>::Output as SwapAt<L, R>>::Output as ReverseShape>::Output;
}

impl<S, R> SwapFromEnd<Here, FromEnd<R>> for S
where
    S: ReverseShape,
    <S as ReverseShape>::Output: SwapFromEnd<FromEnd<Here>, R>,
    <<S as ReverseShape>::Output as SwapFromEnd<FromEnd<Here>, R>>::Output: ReverseShape,
{
    type Output =
        <<<S as ReverseShape>::Output as SwapFromEnd<FromEnd<Here>, R>>::Output as ReverseShape>::Output;
}

impl<S, L, R> SwapFromEnd<Next<L>, FromEnd<R>> for S
where
    S: ReverseShape,
    <S as ReverseShape>::Output: SwapFromEnd<FromEnd<Next<L>>, R>,
    <<S as ReverseShape>::Output as SwapFromEnd<FromEnd<Next<L>>, R>>::Output: ReverseShape,
{
    type Output =
        <<<S as ReverseShape>::Output as SwapFromEnd<FromEnd<Next<L>>, R>>::Output as ReverseShape>::Output;
}

/// Multiplies all dimensions in a structural shape into a single product dimension.
pub trait ProductDims: Shape {
    type Output: Dim;
}

/// Collapses an inclusive structural cursor range into one extent.
///
/// The recursion consumes the selected prefix and then rebuilds the untouched
/// suffix, so this operation has no rank-specific implementations.
pub trait FlattenAt<Start, End>: Shape {
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

impl<End, S: Shape> FlattenAt<crate::shapes::idx::Here, End> for S
where
    S: FlattenPrefix<End>,
{
    type Output = DimCons<<S as FlattenPrefix<End>>::Product, <S as FlattenPrefix<End>>::Suffix>;
}

impl<H: Dim, T: Shape, Start, End>
    FlattenAt<crate::shapes::idx::Next<Start>, crate::shapes::idx::Next<End>> for DimCons<H, T>
where
    T: FlattenAt<Start, End>,
{
    type Output = DimCons<H, <T as FlattenAt<Start, End>>::Output>;
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
    type Prefix: Shape;
    type Penultimate: Dim;
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
    type Prefix: Shape;
    type ThirdLast: Dim;
    type SecondLast: Dim;
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

/// Generic known-rank runtime shape. The rank is a typenum fact, so rank
/// arithmetic remains in the type system without a generated const-generic
/// rank ladder.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Ranked<R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static>(
    core::marker::PhantomData<R>,
);

/// Rank-preserving transformation for generic known-rank shapes.
pub trait PreserveRank {
    type Output: Shape;
}

impl<R> PreserveRank for Ranked<R>
where
    R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = Ranked<R>;
}

/// Generic known-rank reduction/axis removal.
pub trait RemoveOneRank {
    type Output: Shape;
}

impl<R, ROut> RemoveOneRank for Ranked<R>
where
    R: Unsigned + Sub<U1, Output = ROut> + core::fmt::Debug + Eq + Send + Sync + 'static,
    ROut: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = Ranked<ROut>;
}

/// Generic known-rank insertion/stacking.
pub trait AddOneRank {
    type Output: Shape;
}

#[doc(hidden)]
pub trait ShapeRank {
    type Output: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static;
}

impl ShapeRank for Nil {
    type Output = typenum::U0;
}

impl<H: Dim, T> ShapeRank for DimCons<H, T>
where
    T: ShapeRank,
    <T as ShapeRank>::Output: Add<U1>,
    <<T as ShapeRank>::Output as Add<U1>>::Output:
        Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = <<T as ShapeRank>::Output as Add<U1>>::Output;
}

impl AddOneRank for Nil {
    type Output = Ranked<typenum::U1>;
}

impl<H: Dim, T> AddOneRank for DimCons<H, T>
where
    T: ShapeRank,
    <T as ShapeRank>::Output: Add<U1>,
    <<T as ShapeRank>::Output as Add<U1>>::Output: Unsigned + Add<U1>,
    <<<T as ShapeRank>::Output as Add<U1>>::Output as Add<U1>>::Output:
        Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = Ranked<<<<T as ShapeRank>::Output as Add<U1>>::Output as Add<U1>>::Output>;
}

impl<R, ROut> AddOneRank for Ranked<R>
where
    R: Unsigned + Add<U1, Output = ROut> + core::fmt::Debug + Eq + Send + Sync + 'static,
    ROut: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Output = Ranked<ROut>;
}

/// Runtime-axis projection for shapes whose rank is known structurally.
/// Extents are erased, but the rank change remains in the public tensor type.
pub trait RuntimeRankProjection: Shape {
    type Keep: Shape;
    type Drop: Shape;
}

impl RuntimeRankProjection for Nil {
    type Keep = Ranked<typenum::U0>;
    type Drop = Ranked<typenum::U0>;
}

impl<H: Dim, T> RuntimeRankProjection for DimCons<H, T>
where
    T: Shape + RuntimeRankProjection,
    T::Keep: AddOneRank,
    <T::Keep as AddOneRank>::Output: Shape,
{
    type Keep = <T::Keep as AddOneRank>::Output;
    type Drop = T::Keep;
}

impl<R, ROut> RuntimeRankProjection for Ranked<R>
where
    R: Unsigned
        + core::ops::Sub<typenum::U1, Output = ROut>
        + core::fmt::Debug
        + Eq
        + Send
        + Sync
        + 'static,
    ROut: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static,
{
    type Keep = Ranked<R>;
    type Drop = Ranked<ROut>;
}

impl RuntimeRankProjection for Dyn {
    type Keep = Dyn;
    type Drop = Dyn;
}

impl AddOneRank for Dyn {
    type Output = Dyn;
}

impl<R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static> Shape for Ranked<R> {
    const RANK: Option<usize> = Some(R::USIZE);
    const PROOF: crate::shapes::ProofLevel = crate::shapes::ProofLevel::Mixed;
    const STATIC_NUMEL: Option<usize> = if R::USIZE == 0 { Some(1) } else { None };
    type Arg = crate::shapes::ShapeBuf;
    fn resolve(arg: Self::Arg) -> core::result::Result<ShapeBuf, crate::shapes::error::ShapeError> {
        Self::try_from_dims(arg.as_ref()).map(|_| arg)
    }

    fn validate_dims(dims: &[usize]) -> core::result::Result<(), crate::shapes::error::ShapeError> {
        if dims.len() == R::USIZE {
            Ok(())
        } else {
            Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: dims.len(),
            })
        }
    }
}

impl<R: Unsigned + core::fmt::Debug + Eq + Send + Sync + 'static> DynShape for Ranked<R> {
    fn rank(_: &ShapeBuf) -> usize {
        R::USIZE
    }
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
    pub fn shape_buf(&self) -> &crate::shapes::ShapeBuf {
        &self.dims
    }

    #[inline]
    pub fn dims(&self) -> Vec<usize> {
        self.dims.as_ref().to_vec()
    }

    #[inline]
    pub fn proof_level(&self) -> crate::shapes::ProofLevel {
        S::PROOF
    }

    #[inline]
    pub fn checked_numel(
        &self,
        op: crate::shapes::error::OperationKind,
        limits: &crate::resource::ResourceLimits,
    ) -> Result<CheckedNumel, crate::shapes::error::ShapeError> {
        CheckedNumel::from_dims(op, &self.dims(), limits)
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
pub struct InferShape {
    extents: alloc::vec::Vec<Option<usize>>,
}

impl InferShape {
    /// Creates an inference specification. Exactly one extent must be `None`.
    #[must_use]
    pub fn new(extents: alloc::vec::Vec<Option<usize>>) -> Self {
        Self { extents }
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
pub trait ReplaceLastTail<NewDim: Dim>: Shape {
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

/// Fold per-axis static extents into a shape's static element count.
///
/// `None` from any axis, or an overflowing product, makes the whole answer
/// `None`. The multiplication is checked for the same reason
/// [`checked_numel`](crate::shapes::CheckedNumel) is: a wrapped element count
/// undersizes an allocation, and one baked into a kernel as a constant would do
/// so with no runtime check left to catch it.
#[must_use]
pub const fn fold_static_numel(extents: &[Option<usize>]) -> Option<usize> {
    let mut product: usize = 1;
    let mut index = 0;
    while index < extents.len() {
        match extents[index] {
            Some(extent) => match product.checked_mul(extent) {
                Some(next) => product = next,
                None => return None,
            },
            None => return None,
        }
        index += 1;
    }
    Some(product)
}

impl<NewDim: Dim> ReplaceLastDim<NewDim> for Dyn {
    /// `Self`'s dimensions with the last one replaced by `NewDim`.
    type Output = Dyn;
}

pub type Scalar = Nil;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceLimits;
    use crate::shapes::error::{OperationKind, ShapeError};

    #[test]
    fn test_scalar_shape() {
        let scalar = ShapeBuf::scalar();
        assert_eq!(<Nil as DynShape>::rank(&scalar), 0);
        assert_eq!(<Nil as DynShape>::numel(&scalar), 1);
        let empty_dims: [usize; 0] = [];
        assert_eq!(scalar, empty_dims);
        assert_eq!(<Nil as DynShape>::rank(&scalar), 0);
    }

    #[test]
    fn test_dyn_shape() {
        let d = ShapeBuf::from_slice(&[2, 3, 4]);
        assert_eq!(<Dyn as DynShape>::rank(&d), 3);
        assert_eq!(<Dyn as DynShape>::numel(&d), 24);
        let dims = d;
        assert_eq!(dims.as_ref(), &[2, 3, 4]);
    }

    #[test]
    fn test_array_shape() {
        let shape: [usize; 3] = [2, 3, 4];
        let field = ShapeBuf::from_slice(&shape);
        assert_eq!(<Ranked<typenum::U3> as DynShape>::rank(&field), 3);
        assert_eq!(<Ranked<typenum::U3> as DynShape>::numel(&field), 24);
        assert_eq!(field.dims(), &[2, 3, 4]);
        assert_eq!(<Ranked<typenum::U3> as Shape>::RANK, Some(3));
    }

    #[test]
    fn dyn_is_zero_sized() {
        assert_eq!(core::mem::size_of::<Dyn>(), 0);
        let marker = Dyn::marker();
        assert_eq!(core::mem::size_of_val(&marker), 0);
    }

    #[test]
    fn checked_allocation_lengths_cover_scalar_zero_limit_and_overflow_edges() {
        let mut limits = ResourceLimits::trusted_local_large_model();
        limits.max_rank = 8;
        // Keep the local reusable for the following checks under non-strict lints.
        #[allow(unused_assignments)]
        {
            limits.max_dimension = u64::MAX;
        }
        limits.max_tensor_bytes = u64::MAX;

        assert_eq!(
            CheckedNumel::from_dims(OperationKind::Storage, &[], &limits)
                .unwrap()
                .get(),
            1
        );
        assert_eq!(
            CheckedNumel::from_dims(
                OperationKind::Storage,
                &[usize::MAX, 0, usize::MAX],
                &limits
            )
            .unwrap()
            .get(),
            0
        );
        assert!(matches!(
            CheckedNumel::from_dims(OperationKind::Storage, &[usize::MAX, 2], &limits),
            Err(ShapeError::ArithmeticOverflow { .. })
        ));

        limits.max_rank = 1;
        assert!(matches!(
            CheckedNumel::from_dims(OperationKind::Storage, &[2, 3], &limits),
            Err(ShapeError::RankMismatch { .. })
        ));
        limits.max_rank = 8;
        limits.max_dimension = 2;
        assert!(matches!(
            CheckedNumel::from_dims(OperationKind::Storage, &[3], &limits),
            Err(ShapeError::InvalidParameter {
                parameter: "dimension",
                value: 3,
                ..
            })
        ));
    }
}
