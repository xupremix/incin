//! Compile-time broadcasting shape verification.
use crate::shapes::dim::Dim;
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, ShapeError};
use crate::shapes::shape::{DimCons, DynShape, Nil, Shape};
use crate::shapes::ShapeBuf;
use crate::shapes::Dyn;
use alloc::vec::Vec;

/// Resolve one runtime (`Dyn`) broadcast dimension, reporting incompatible
/// sizes instead of silently fabricating a wrong result via a bare `.max()`.
/// NumPy/PyTorch broadcast rule: two dims are compatible iff they're equal or
/// one of them is 1.
///
/// Every `BroadcastShape` call site reachable through the public `Tensor` API
/// (`broadcast_add`/`sub`/`mul`/`div` and the `+`/`-`/`*`/`/` operator
/// overloads in `tensor::ops::binary`) already calls into the backend's own
/// validated `broadcast_shape` first and propagates its `Err` via `?` before
/// this value is ever used. It exists as defense-in-depth for any future or
/// direct caller of `BroadcastShape::output_shape` that doesn't already
/// validate independently, and since a `symbolic_dim!` name, unlike
/// `typenum`) can legitimately hold a *different* runtime value on each operand
/// even when both share the exact same type, as the actual guard against two
/// same-typed named dims whose real sizes happen to disagree.
///
/// `SHP-004` converts it from an `assert!` to a `Result` per decision `D-013`,
/// which records that it must be converted rather than deleted: it is the only
/// guard against that same-typed-named-dims case.
#[inline]
fn checked_broadcast_dim(
    axis: Axis,
    lhs: usize,
    rhs: usize,
) -> core::result::Result<usize, ShapeError> {
    if lhs == rhs || lhs == 1 || rhs == 1 {
        // Not `lhs.max(rhs)`: a size-1 axis broadcast against a size-**0** one
        // must yield 0, and `max` yields 1. Picking the non-1 side is the
        // actual NumPy rule and is correct at 0.
        Ok(if lhs == 1 { rhs } else { lhs })
    } else {
        Err(ShapeError::DimensionMismatch {
            operation: OperationKind::Broadcast,
            axis,
            lhs,
            rhs,
            constraint: DimensionConstraint::Broadcastable,
        })
    }
}

/// Generic NumPy-style right-aligned broadcast: computes the output's
/// runtime per-axis sizes from `lhs`/`rhs`'s own dims, prepending implicit
/// size-1 axes on whichever operand has fewer dimensions. This one function
/// backs every `BroadcastShape` impl below instead of each hand-rolling its
/// own per-arity dimension arithmetic. That is what let a real bug slip in
/// previously (see the module-level history in `docs/growth/03-named-
/// dimensions.md`): building the output shape from `Default::default()`
/// happened to be invisible for `typenum` dims (zero-sized `PhantomData`,
/// so "the default" and "the real value" coincide) but would have silently
/// zeroed any runtime-carrying dimension (a `usize` axis, or a
/// `symbolic_dim!` name).
fn broadcast_dims(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<Vec<usize>, ShapeError> {
    let lhs_dims: Vec<usize> = lhs.clone().into();
    let rhs_dims: Vec<usize> = rhs.clone().into();
    broadcast_dim_slices(&lhs_dims, &rhs_dims)
}

/// The same right-aligned rule as the typed broadcast helper, reached from raw
/// dimensions rather than from a typed `Field`.
///
/// A backend holds `&[usize]`, not an `L::Field`, so without this it would have
/// to re-derive NumPy's alignment rule, and a second copy of a broadcast rule
/// is a second answer waiting to disagree with the first. `broadcast_dims`
/// delegates here so there is exactly one.
pub fn broadcast_dim_slices(
    lhs_dims: &[usize],
    rhs_dims: &[usize],
) -> core::result::Result<Vec<usize>, ShapeError> {
    let out_rank = lhs_dims.len().max(rhs_dims.len());
    let mut out = Vec::with_capacity(out_rank);
    for i in 0..out_rank {
        let from_end = out_rank - i;
        let l = lhs_dims
            .len()
            .checked_sub(from_end)
            .map(|idx| lhs_dims[idx]);
        let r = rhs_dims
            .len()
            .checked_sub(from_end)
            .map(|idx| rhs_dims[idx]);
        // An axis the shorter operand does not reach is an implicit 1. That is
        // exactly NumPy's right-alignment rule, and 1 is the identity for
        // broadcasting. Writing it that way makes the fourth case ("neither
        // operand reaches this axis") disappear rather than be asserted away
        // with `unreachable!`: `out_rank` is the max of the two ranks, so it
        // cannot occur, and if it did, the answer would still be 1.
        out.push(checked_broadcast_dim(
            Axis::Index(i),
            l.unwrap_or(1),
            r.unwrap_or(1),
        )?);
    }
    Ok(out)
}

/// Trait that verifies two shapes are broadcastable and determines the output shape.
#[diagnostic::on_unimplemented(
    message = "Cannot broadcast shape `{Self}` to `{Rhs}`",
    label = "Shape mismatch during broadcast",
    note = "Broadcast requires dimensions to be equal, or one of them to be 1"
)]
/// Compile-time-checked NumPy-style broadcast shape rule: `Self`
/// broadcast against `Rhs` produces `Output`.
pub trait BroadcastShape<Rhs: Shape>: Shape {
    /// The resulting shape after broadcasting `Self` against `Rhs`.
    type Output: Shape;
    /// Computes the runtime `ShapeBuf` of `Output`,
    /// resolving any `usize` (runtime) dimensions via `checked_broadcast_dim`.
    fn output_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<ShapeBuf, ShapeError>;
}

// ============================================================================
// Per-axis rule: the ways two axes may meet.
// ============================================================================

/// One axis of the left operand against the axis facing it on the right.
///
/// Broadcasting is a per-axis rule, and lifting it to whole shapes one axis at
/// a time is what lets `(N, C, H, W)` meet `(U1, C, U1, U1)`, the bias-add
/// shape, and the reason this trait exists. Before `SHP-007` the same-rank
/// family required every axis to be the *identical* type, so that pair did not
/// typecheck at all and callers reached for a rank-changing spelling or `Dyn`.
///
/// The output is a symbolic `BroadcastExtent<L, R>`. Its `Dim::STATIC`
/// classification preserves equal/one facts that can be proved from the
/// operand types, while runtime construction checks the actual values.
#[diagnostic::on_unimplemented(
    message = "Cannot broadcast axis `{Self}` against `{Rhs}`",
    label = "incompatible axis",
    note = "two axes broadcast when their types are the same, or one of them is `U1`"
)]
pub trait BroadcastDim<Rhs: Dim>: Dim {
    /// The axis the two resolve to.
    type Output: Dim;
}

/// Marker for dimensions that carry no semantic axis name.
///
/// This is deliberately implemented for Incin's concrete dimension families
/// instead of using a blanket `Dim` implementation. The disjointness lets a
/// named axis normalize to `NamedDim<Tag, ...>` without overlapping impls.
pub trait AnonymousDim: Dim {}

impl AnonymousDim for usize {}
impl<const N: usize> AnonymousDim for crate::shapes::dim::ConstDim<N> {}
impl AnonymousDim for typenum::UTerm {}
impl<U, B> AnonymousDim for typenum::UInt<U, B> where typenum::UInt<U, B>: Dim {}
impl<A: Dim, B: Dim> AnonymousDim for crate::shapes::dim::AddDim<A, B> {}
impl<A: Dim, B: Dim> AnonymousDim for crate::shapes::dim::CheckedSubDim<A, B> {}
impl<A: Dim, B: Dim> AnonymousDim for crate::shapes::dim::ExactDivDim<A, B> {}
impl<A: Dim, B: Dim> AnonymousDim for crate::shapes::dim::MulDim<A, B> {}
impl<A: AnonymousDim, B: AnonymousDim> AnonymousDim for crate::shapes::dim::BroadcastExtent<A, B> {}

type StaticBroadcast<L, R> = crate::shapes::dim::BroadcastStaticNat<L, R>;
type RuntimeStatic<R> =
    <<R as typenum::IsEqual<typenum::U1>>::Output as crate::shapes::dim::BroadcastChoice<
        usize,
        R,
    >>::Output;
type StaticRuntime<L> =
    <<L as typenum::IsEqual<typenum::U1>>::Output as crate::shapes::dim::BroadcastChoice<
        usize,
        L,
    >>::Output;

/// Closed local marker for the recursive typenum natural types emitted by
/// the shape macros. Keeping this marker local lets the runtime `usize` axis
/// remain disjoint from static broadcast implementations under Stable Rust's
/// coherence rules.
trait StaticNatDim: AnonymousDim + typenum::Unsigned + typenum::IsEqual<typenum::U1> {}

impl StaticNatDim for typenum::UTerm {}

impl<U, B> StaticNatDim for typenum::UInt<U, B> where
    typenum::UInt<U, B>: AnonymousDim + typenum::Unsigned + typenum::IsEqual<typenum::U1>
{
}

/// Static anonymous axes normalize directly to the selected typenum extent.
/// The comparison bounds are deliberately part of the implementation so an
/// incompatible pair has no `BroadcastDim` implementation at all.
impl<L, R> BroadcastDim<R> for L
where
    L: StaticNatDim,
    R: StaticNatDim,
    L: typenum::IsEqual<typenum::U1> + typenum::IsEqual<R>,
    R: typenum::IsEqual<typenum::U1>,
    <L as typenum::IsEqual<typenum::U1>>::Output:
        crate::shapes::dim::BroadcastChoice<R, crate::shapes::dim::BroadcastRightNat<L, R>>,
    <R as typenum::IsEqual<typenum::U1>>::Output:
        crate::shapes::dim::BroadcastChoice<L, crate::shapes::dim::BroadcastSameNat<L, R>>,
    <L as typenum::IsEqual<R>>::Output: crate::shapes::dim::BroadcastChoice<L, ()>,
    StaticBroadcast<L, R>: Dim,
{
    type Output = StaticBroadcast<L, R>;
}

/// Runtime anonymous axes retain a symbolic extent expression.
impl BroadcastDim<usize> for usize {
    type Output = crate::shapes::dim::BroadcastExtent<Self, usize>;
}

impl<R> BroadcastDim<R> for usize
where
    R: StaticNatDim,
    <R as typenum::IsEqual<typenum::U1>>::Output: crate::shapes::dim::BroadcastChoice<usize, R>,
    RuntimeStatic<R>: Dim,
{
    type Output = RuntimeStatic<R>;
}

impl<L> BroadcastDim<usize> for L
where
    L: StaticNatDim,
    <L as typenum::IsEqual<typenum::U1>>::Output: crate::shapes::dim::BroadcastChoice<usize, L>,
    StaticRuntime<L>: Dim,
{
    type Output = StaticRuntime<L>;
}

/// A named runtime axis remains named when it meets an anonymous axis.
impl<Tag, R> BroadcastDim<R> for crate::shapes::dim::NamedDim<Tag, usize>
where
    Tag: crate::shapes::AxisTag,
    R: StaticNatDim,
    R: typenum::IsEqual<typenum::U1>,
    <R as typenum::IsEqual<typenum::U1>>::Output: crate::shapes::dim::BroadcastChoice<usize, R>,
    RuntimeStatic<R>: Dim,
{
    type Output = crate::shapes::dim::NamedDim<Tag, RuntimeStatic<R>>;
}

impl<Tag> BroadcastDim<usize> for crate::shapes::dim::NamedDim<Tag, usize>
where
    Tag: crate::shapes::AxisTag,
{
    type Output =
        crate::shapes::dim::NamedDim<Tag, crate::shapes::dim::BroadcastExtent<usize, usize>>;
}

/// An anonymous static axis adopts the semantic name of a named static axis.
impl<Tag, L, R> BroadcastDim<crate::shapes::dim::NamedDim<Tag, R>> for L
where
    Tag: crate::shapes::AxisTag,
    L: StaticNatDim,
    R: StaticNatDim,
    L: typenum::IsEqual<typenum::U1> + typenum::IsEqual<R>,
    R: typenum::IsEqual<typenum::U1>,
    <L as typenum::IsEqual<typenum::U1>>::Output:
        crate::shapes::dim::BroadcastChoice<R, crate::shapes::dim::BroadcastRightNat<L, R>>,
    <R as typenum::IsEqual<typenum::U1>>::Output:
        crate::shapes::dim::BroadcastChoice<L, crate::shapes::dim::BroadcastSameNat<L, R>>,
    <L as typenum::IsEqual<R>>::Output: crate::shapes::dim::BroadcastChoice<L, ()>,
    StaticBroadcast<L, R>: Dim,
{
    type Output = crate::shapes::dim::NamedDim<Tag, StaticBroadcast<L, R>>;
}

impl<Tag, L> BroadcastDim<crate::shapes::dim::NamedDim<Tag, usize>> for L
where
    Tag: crate::shapes::AxisTag,
    L: StaticNatDim,
    L: typenum::IsEqual<typenum::U1>,
    <L as typenum::IsEqual<typenum::U1>>::Output: crate::shapes::dim::BroadcastChoice<usize, L>,
    StaticRuntime<L>: Dim,
{
    type Output = crate::shapes::dim::NamedDim<Tag, StaticRuntime<L>>;
}

/// A named static axis remains named when it meets an anonymous static axis.
impl<Tag, L, R> BroadcastDim<R> for crate::shapes::dim::NamedDim<Tag, L>
where
    Tag: crate::shapes::AxisTag,
    L: StaticNatDim,
    R: StaticNatDim,
    L: typenum::IsEqual<typenum::U1> + typenum::IsEqual<R>,
    R: typenum::IsEqual<typenum::U1>,
    <L as typenum::IsEqual<typenum::U1>>::Output:
        crate::shapes::dim::BroadcastChoice<R, crate::shapes::dim::BroadcastRightNat<L, R>>,
    <R as typenum::IsEqual<typenum::U1>>::Output:
        crate::shapes::dim::BroadcastChoice<L, crate::shapes::dim::BroadcastSameNat<L, R>>,
    <L as typenum::IsEqual<R>>::Output: crate::shapes::dim::BroadcastChoice<L, ()>,
    StaticBroadcast<L, R>: Dim,
{
    type Output = crate::shapes::dim::NamedDim<Tag, StaticBroadcast<L, R>>;
}

impl<Tag, L> BroadcastDim<usize> for crate::shapes::dim::NamedDim<Tag, L>
where
    Tag: crate::shapes::AxisTag,
    L: StaticNatDim,
    L: typenum::IsEqual<typenum::U1>,
    <L as typenum::IsEqual<typenum::U1>>::Output: crate::shapes::dim::BroadcastChoice<usize, L>,
    StaticRuntime<L>: Dim,
{
    type Output = crate::shapes::dim::NamedDim<Tag, StaticRuntime<L>>;
}

/// Equal semantic names remain named for static extents. Different tags
/// intentionally have no implementation, so a statically conflicting pair
/// is rejected.
impl<Tag, L, R> BroadcastDim<crate::shapes::dim::NamedDim<Tag, R>>
    for crate::shapes::dim::NamedDim<Tag, L>
where
    Tag: crate::shapes::AxisTag,
    L: StaticNatDim,
    R: StaticNatDim,
    L: typenum::IsEqual<typenum::U1> + typenum::IsEqual<R>,
    R: typenum::IsEqual<typenum::U1>,
    <L as typenum::IsEqual<typenum::U1>>::Output:
        crate::shapes::dim::BroadcastChoice<R, crate::shapes::dim::BroadcastRightNat<L, R>>,
    <R as typenum::IsEqual<typenum::U1>>::Output:
        crate::shapes::dim::BroadcastChoice<L, crate::shapes::dim::BroadcastSameNat<L, R>>,
    <L as typenum::IsEqual<R>>::Output: crate::shapes::dim::BroadcastChoice<L, ()>,
    StaticBroadcast<L, R>: Dim,
{
    type Output = crate::shapes::dim::NamedDim<Tag, StaticBroadcast<L, R>>;
}

impl<Tag, R> BroadcastDim<crate::shapes::dim::NamedDim<Tag, R>>
    for crate::shapes::dim::NamedDim<Tag, usize>
where
    Tag: crate::shapes::AxisTag,
    R: StaticNatDim,
    R: typenum::IsEqual<typenum::U1>,
    <R as typenum::IsEqual<typenum::U1>>::Output: crate::shapes::dim::BroadcastChoice<usize, R>,
    RuntimeStatic<R>: Dim,
{
    type Output = crate::shapes::dim::NamedDim<Tag, RuntimeStatic<R>>;
}

impl<Tag> BroadcastDim<crate::shapes::dim::NamedDim<Tag, usize>>
    for crate::shapes::dim::NamedDim<Tag, usize>
where
    Tag: crate::shapes::AxisTag,
{
    type Output =
        crate::shapes::dim::NamedDim<Tag, crate::shapes::dim::BroadcastExtent<usize, usize>>;
}

// ============================================================================
// Shape families: the per-axis rule lifted across whole shapes.
//
// Three families cover every pair of tuple shapes: equal ranks, one operand
// empty, and one operand a suffix of the other. They are bounded by `Dim`
// rather than `StaticOrNamedDim`, so a `usize` axis is an axis like any other
// and `BroadcastDim` decides what it may meet.
//
// `SHP-007` deleted a parallel set of five families that existed only to
// handle a `usize`. Each required the runtime axis to be axis 0 and every
// other axis to be identical on both sides, which meant `(U3, usize)` had no
// partner at all and `(usize, U4)` could not meet `(U1, U4)`. Those are not
// special cases of broadcasting; they were special cases of a rule that could
// not talk about one axis at a time.
// ============================================================================

// Exact known-rank broadcasting is handled by the structural DimCons engine
// below; there is no tuple or generated-rank family.

// Rank-0 (`()`) on one side, a full static shape on the other, in both directions.
// Rank zero and scalar broadcasting are implemented by Nil/DimCons below.

// Different ranks: the shorter shape right-aligns against the longer one's
// trailing axes, and the leading axes it does not reach pass through. The
// overlapping axes are related by `BroadcastDim`, not required to be
// identical, so `(N, C, H, W)` accepts `(C, U1, U1)`. Both directions.
// Different-rank alignment is handled by ReverseShape/BroadcastReversed below.

// ============================================================================
// Fully dynamic: `Dyn` on at least one side. The backend itself independently
// validates and computes the real result shape before any `Tensor` carrying
// this `Output` field is used (see `checked_broadcast_dim`'s doc comment).
// so unlike the families above, cloning whichever side is `Dyn` (or, when
// neither is, doing the same right-aligned computation) is the existing,
// intentionally-lightweight contract here, not a shortcut this change needs
// to correct. Only the bound (`StaticOrNamedDim` instead of `StaticDim`)
// needed relaxing to admit named dims. The bodies never used
// `Default::default()` and don't change.
// ============================================================================

/// Runtime-rank broadcast against any shape.  The runtime dimensions are
/// authoritative; the other shape contributes its actual `Field`, so this
/// path validates the same right-aligned NumPy rule as the exact structural
/// implementation without a generated tuple-rank family.
impl<R: Shape + DynShape> BroadcastShape<R> for Dyn {
    type Output = Dyn;

    fn output_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<ShapeBuf, ShapeError> {
        broadcast_dims(lhs, rhs).map(|dims| crate::shapes::ShapeBuf::from_slice(&dims))
    }
}

impl BroadcastShape<Dyn> for Nil {
    type Output = Dyn;

    fn output_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<ShapeBuf, ShapeError> {
        broadcast_dims(lhs, rhs).map(|dims| crate::shapes::ShapeBuf::from_slice(&dims))
    }
}

impl<H: Dim, T: Shape + DynShape> BroadcastShape<Dyn> for DimCons<H, T> {
    type Output = Dyn;

    fn output_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<ShapeBuf, ShapeError> {
        broadcast_dims(lhs, rhs).map(|dims| crate::shapes::ShapeBuf::from_slice(&dims))
    }
}

// ============================================================================
// DimCons Structural Broadcast
// ============================================================================

/// Reverses an exact structural shape.  Broadcasting is naturally recursive
/// from the trailing axis, so reversing lets the type-level algorithm follow
/// the same right-aligned order as the runtime validator.
pub trait ReverseShape: Shape {
    type Output: Shape;
}

impl ReverseShape for Nil {
    type Output = Nil;
}

impl<H: Dim, T: Shape + ReverseShape> ReverseShape for DimCons<H, T>
where
    <T as ReverseShape>::Output: crate::shapes::AppendDim<H>,
{
    type Output = <<T as ReverseShape>::Output as crate::shapes::AppendDim<H>>::Output;
}

trait BroadcastReversed<Rhs: Shape>: Shape {
    type Output: Shape;
}

trait StaticShapeNames {
    fn names() -> Vec<Option<&'static str>>;
}

impl StaticShapeNames for Nil {
    fn names() -> Vec<Option<&'static str>> {
        Vec::new()
    }
}

impl<H: Dim, T: Shape + StaticShapeNames> StaticShapeNames for DimCons<H, T> {
    fn names() -> Vec<Option<&'static str>> {
        let mut names = Vec::with_capacity(1 + T::RANK.unwrap_or(0));
        names.push(H::NAME);
        names.extend(T::names());
        names
    }
}

fn validate_static_names<L: StaticShapeNames, R: StaticShapeNames>()
-> core::result::Result<(), ShapeError> {
    let lhs = L::names();
    let rhs = R::names();
    let rank = lhs.len().max(rhs.len());
    for output_axis in 0..rank {
        let lhs_name = output_axis
            .checked_add(lhs.len())
            .and_then(|index| index.checked_sub(rank))
            .and_then(|index| lhs.get(index))
            .copied()
            .flatten();
        let rhs_name = output_axis
            .checked_add(rhs.len())
            .and_then(|index| index.checked_sub(rank))
            .and_then(|index| rhs.get(index))
            .copied()
            .flatten();
        if let (Some(lhs), Some(rhs)) = (lhs_name, rhs_name)
            && lhs != rhs
        {
            return Err(ShapeError::ConflictingNamedAxes {
                axis: output_axis,
                lhs,
                rhs,
            });
        }
    }
    Ok(())
}

impl BroadcastReversed<Nil> for Nil {
    type Output = Nil;
}

impl<H: Dim, T: Shape> BroadcastReversed<Nil> for DimCons<H, T> {
    type Output = DimCons<H, T>;
}

impl<H: Dim, T: Shape> BroadcastReversed<DimCons<H, T>> for Nil {
    type Output = DimCons<H, T>;
}

impl<LH: Dim, LT: Shape, RH: Dim, RT: Shape> BroadcastReversed<DimCons<RH, RT>> for DimCons<LH, LT>
where
    LH: BroadcastDim<RH>,
    LT: BroadcastReversed<RT>,
{
    type Output = DimCons<<LH as BroadcastDim<RH>>::Output, <LT as BroadcastReversed<RT>>::Output>;
}

impl BroadcastShape<Nil> for Nil {
    type Output = Nil;

    fn output_shape(
        _: &crate::shapes::ShapeBuf,
        _: &crate::shapes::ShapeBuf,
    ) -> core::result::Result<crate::shapes::ShapeBuf, ShapeError> {
        Ok(crate::shapes::ShapeBuf::scalar())
    }
}

impl<H: Dim, T: Shape> BroadcastShape<Nil> for DimCons<H, T> {
    type Output = DimCons<H, T>;

    fn output_shape(
        lhs: &ShapeBuf,
        _: &crate::shapes::ShapeBuf,
    ) -> core::result::Result<ShapeBuf, ShapeError> {
        Ok(lhs.clone())
    }
}

impl<H: Dim, T: Shape> BroadcastShape<DimCons<H, T>> for Nil {
    type Output = DimCons<H, T>;

    fn output_shape(
        _: &crate::shapes::ShapeBuf,
        rhs: &ShapeBuf,
    ) -> core::result::Result<ShapeBuf, ShapeError> {
        Ok(rhs.clone())
    }
}

impl<LH: Dim, LT, RH: Dim, RT, LRev, RRev, RevOut> BroadcastShape<DimCons<RH, RT>>
    for DimCons<LH, LT>
where
    LT: Shape + crate::shapes::DynShape + StaticShapeNames,
    RT: Shape + crate::shapes::DynShape + StaticShapeNames,
    DimCons<LH, LT>: ReverseShape<Output = LRev>,
    DimCons<RH, RT>: ReverseShape<Output = RRev>,
    LRev: Shape + BroadcastReversed<RRev>,
    RRev: Shape,
    <LRev as BroadcastReversed<RRev>>::Output: ReverseShape<Output = RevOut>,
    RevOut: Shape,
{
    type Output = RevOut;

    fn output_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<ShapeBuf, ShapeError> {
        validate_static_names::<Self, DimCons<RH, RT>>()?;
        // Force invalid fully-static axis expressions to fail at the public
        // operation boundary instead of silently degrading to a runtime
        // broadcast check.
        <Self::Output as Shape>::STATIC_VALID;
        let dims = broadcast_dims(lhs, rhs)?;
        crate::shapes::shape::shape_buf_from_dims::<Self::Output>(OperationKind::Broadcast, &dims)
    }
}
