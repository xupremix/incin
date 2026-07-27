//! Compile-time broadcasting shape verification.
use crate::prelude::*;
use crate::tensor::matmul::StaticOrNamedDim;

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
/// validate independently, and — since a `symbolic_dim!` name (unlike
/// `typenum`) can legitimately hold a *different* runtime value on each operand
/// even when both share the exact same type — as the actual guard against two
/// same-typed named dims whose real sizes happen to disagree.
///
/// `SHP-004` converts it from an `assert!` to a `Result` per decision `D-013`,
/// which records that it must be converted rather than deleted: it is the only
/// guard against that same-typed-named-dims case.
#[inline]
fn checked_broadcast_dim(axis: Axis, lhs: usize, rhs: usize) -> core::result::Result<usize, ShapeError> {
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
/// own per-arity dimension arithmetic — which is what let a real bug slip in
/// previously (see the module-level history in `docs/growth/03-named-
/// dimensions.md`): building the output shape from `Default::default()`
/// happened to be invisible for `typenum` dims (zero-sized `PhantomData`,
/// so "the default" and "the real value" coincide) but would have silently
/// zeroed any runtime-carrying dimension (a `usize` axis, or a
/// `symbolic_dim!` name).
fn broadcast_dims<L: DynShape, R: DynShape>(
    lhs: &L::Field,
    rhs: &R::Field,
) -> core::result::Result<Vec<usize>, ShapeError> {
    let lhs_dims: Vec<usize> = L::dims(lhs).into();
    let rhs_dims: Vec<usize> = R::dims(rhs).into();
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
        // An axis the shorter operand does not reach is an implicit 1 — that is
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
    /// Computes the runtime `Field` (dimension values) of `Output`,
    /// resolving any `usize` (runtime) dimensions via `checked_broadcast_dim`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Rhs as Shape>::Field,
    ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError>;
}

impl BroadcastShape<()> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = ();
    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(_: &(), _: &()) -> core::result::Result<(), ShapeError> {
        Ok(())
    }
}

// ============================================================================
// Static family: every axis is `StaticOrNamedDim` (typenum or `symbolic_dim!`).
// ============================================================================

/// Same rank on both operands, every axis the identical type.
macro_rules! impl_broadcast_same_rank {
    ($($dim:ident),+) => {
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<($($dim,)+)> for ($($dim,)+) {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = ($($dim,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<($($dim,)+) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, ($($dim,)+)>(lhs, rhs)?)
            }
        }
    };
}
// Both operands the same rank, every axis the identical type.
incin_macros::rank_sweep!(letters => impl_broadcast_same_rank);

/// Rank-0 (`()`) on one side, a full static shape on the other, in both directions.
macro_rules! impl_broadcast_empty_to_full {
    ($($dim:ident),+) => {
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<($($dim,)+)> for () {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = ($($dim,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<($($dim,)+) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, ($($dim,)+)>(lhs, rhs)?)
            }
        }
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<()> for ($($dim,)+) {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = ($($dim,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<() as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, ()>(lhs, rhs)?)
            }
        }
    };
}
// Rank 0 against a full shape, in both directions.
incin_macros::rank_sweep!(letters => impl_broadcast_empty_to_full);

/// Different ranks, the shorter shape's dims a literal *suffix* of the
/// longer one's (implicit leading 1s fill the rest) — both directions.
macro_rules! impl_broadcast_prepend {
    ( ($($prefix:ident),+) ; ($($suffix:ident),+) ) => {
        impl<$($prefix: StaticOrNamedDim,)+ $($suffix: StaticOrNamedDim),+> BroadcastShape<($($prefix,)+ $($suffix,)+)> for ($($suffix,)+)
        where
            ($($suffix,)+): DynShape,
        {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = ($($prefix,)+ $($suffix,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<($($prefix,)+ $($suffix,)+) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, ($($prefix,)+ $($suffix,)+)>(lhs, rhs)?)
            }
        }
        impl<$($prefix: StaticOrNamedDim,)+ $($suffix: StaticOrNamedDim),+> BroadcastShape<($($suffix,)+)> for ($($prefix,)+ $($suffix,)+)
        where
            ($($suffix,)+): DynShape,
        {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = ($($prefix,)+ $($suffix,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<($($suffix,)+) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, ($($suffix,)+)>(lhs, rhs)?)
            }
        }
    };
}
// Shorter against longer: one invocation per split of the output rank.
incin_macros::rank_sweep!(prepend => impl_broadcast_prepend);

// ============================================================================
// Partially dynamic: a leading `usize` batch dim, `StaticOrNamedDim` tail.
// ============================================================================

impl BroadcastShape<(usize,)> for (usize,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize,);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Self as Shape>::Field,
    ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
        Ok((checked_broadcast_dim(Axis::Index(0), lhs.0, rhs.0)?,))
    }
}
impl BroadcastShape<()> for (usize,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize,);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
        Ok((lhs.0,))
    }
}
impl BroadcastShape<(usize,)> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = (usize,);
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(usize,) as Shape>::Field,
    ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
        Ok((rhs.0,))
    }
}

/// Same rank, leading `usize` batch dim shared, `StaticOrNamedDim` tail identical.
macro_rules! impl_broadcast_usize_same_rank {
    ($($dim:ident),+) => {
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<(usize, $($dim,)+)> for (usize, $($dim,)+) {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = (usize, $($dim,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<(usize, $($dim,)+) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, (usize, $($dim,)+)>(lhs, rhs)?)
            }
        }
    };
}
// Shape is `(usize, Tail..)`, so rank = tail length + 1.
incin_macros::rank_sweep!(letters_from_b => impl_broadcast_usize_same_rank, max = 7);

/// Rank-0 (`()`) on one side, a `(usize, ...)` shape on the other.
macro_rules! impl_broadcast_usize_empty_to_full {
    ($($dim:ident),+) => {
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<(usize, $($dim,)+)> for () {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = (usize, $($dim,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<(usize, $($dim,)+) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, (usize, $($dim,)+)>(lhs, rhs)?)
            }
        }
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<()> for (usize, $($dim,)+) {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = (usize, $($dim,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<() as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, ()>(lhs, rhs)?)
            }
        }
    };
}
// Shape is `(usize, Tail..)`, so rank = tail length + 1.
incin_macros::rank_sweep!(letters_from_b => impl_broadcast_usize_empty_to_full, max = 7);

/// Different ranks, one of them `(usize, prefix..., suffix...)`, the other
/// just `(suffix...)` (a literal suffix of the first) — both directions.
/// `prefix` may be empty (the `usize` alone is the whole prefix).
macro_rules! impl_broadcast_usize_prepend {
    ( ($($prefix:ident),*) ; ($($suffix:ident),+) ) => {
        impl<$($prefix: StaticOrNamedDim,)* $($suffix: StaticOrNamedDim),+> BroadcastShape<(usize, $($prefix,)* $($suffix,)+)> for ($($suffix,)+)
        where
            ($($suffix,)+): DynShape,
        {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = (usize, $($prefix,)* $($suffix,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<(usize, $($prefix,)* $($suffix,)+) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, (usize, $($prefix,)* $($suffix,)+)>(lhs, rhs)?)
            }
        }
        impl<$($prefix: StaticOrNamedDim,)* $($suffix: StaticOrNamedDim),+> BroadcastShape<($($suffix,)+)> for (usize, $($prefix,)* $($suffix,)+)
        where
            ($($suffix,)+): DynShape,
        {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = (usize, $($prefix,)* $($suffix,)+);
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<($($suffix,)+) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                field_from_dims::<Self::Output>(OperationKind::Broadcast, &broadcast_dims::<Self, ($($suffix,)+)>(lhs, rhs)?)
            }
        }
    };
}
// Shape is `(usize, Tail..)`, so rank = tail length + 1.
incin_macros::rank_sweep!(usize_prepend => impl_broadcast_usize_prepend, max = 7);

// ============================================================================
// Fully dynamic: `Dyn` on at least one side. The backend itself independently
// validates and computes the real result shape before any `Tensor` carrying
// this `Output` field is used (see `checked_broadcast_dim`'s doc comment) —
// so unlike the families above, cloning whichever side is `Dyn` (or, when
// neither is, doing the same right-aligned computation) is the existing,
// intentionally-lightweight contract here, not a shortcut this change needs
// to correct. Only the bound (`StaticOrNamedDim` instead of `StaticDim`)
// needed relaxing to admit named dims — the bodies never used
// `Default::default()` and don't change.
// ============================================================================

impl BroadcastShape<Dyn> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
        broadcast_dims::<Dyn, Dyn>(lhs, rhs)
    }
}
impl BroadcastShape<()> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<() as Shape>::Field,
    ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
        Ok(lhs.clone()) // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<Dyn> for () {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
        Ok(rhs.clone()) // At runtime candle will compute output shape properly
    }
}

/// `Dyn` against a full static shape, in both directions — `Dyn`'s own
/// field is authoritative (see the module note above), so the static
/// operand's dims are unused; only its arity (rank) matters for which
/// impl applies.
macro_rules! impl_broadcast_dyn_static {
    ($($dim:ident),+) => {
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<($($dim,)+)> for Dyn {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = Dyn;
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Dyn as Shape>::Field,
                _: &<($($dim,)+) as Shape>::Field,
            ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
                Ok(lhs.clone()) // At runtime candle will compute output shape properly
            }
        }
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<Dyn> for ($($dim,)+) {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = Dyn;
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                _: &<Self as Shape>::Field,
                rhs: &<Dyn as Shape>::Field,
            ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
                Ok(rhs.clone()) // At runtime candle will compute output shape properly
            }
        }
    };
}
// `Dyn` against a full static shape, in both directions.
incin_macros::rank_sweep!(letters => impl_broadcast_dyn_static);

impl BroadcastShape<(usize,)> for Dyn {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        _: &<(usize,) as Shape>::Field,
    ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
        Ok(lhs.clone()) // At runtime candle will compute output shape properly
    }
}
impl BroadcastShape<Dyn> for (usize,) {
    /// The resulting shape after broadcasting `Self` against the other operand.
    type Output = Dyn;
    /// Computes the runtime `Field` (dimension values) of `Output`.
    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
        Ok(rhs.clone()) // At runtime candle will compute output shape properly
    }
}

/// `Dyn` against a `(usize, ...)` shape, in both directions.
macro_rules! impl_broadcast_dyn_usize {
    ($($dim:ident),+) => {
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<(usize, $($dim,)+)> for Dyn {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = Dyn;
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                lhs: &<Dyn as Shape>::Field,
                _: &<(usize, $($dim,)+) as Shape>::Field,
            ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
                Ok(lhs.clone()) // At runtime candle will compute output shape properly
            }
        }
        impl<$($dim: StaticOrNamedDim),+> BroadcastShape<Dyn> for (usize, $($dim,)+) {
            /// The resulting shape after broadcasting `Self` against the other operand.
            type Output = Dyn;
            /// Computes the runtime `Field` (dimension values) of `Output`.
            fn output_shape(
                _: &<Self as Shape>::Field,
                rhs: &<Dyn as Shape>::Field,
            ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
                Ok(rhs.clone()) // At runtime candle will compute output shape properly
            }
        }
    };
}
// Shape is `(usize, Tail..)`, so rank = tail length + 1.
incin_macros::rank_sweep!(letters_from_b => impl_broadcast_dyn_usize, max = 7);
