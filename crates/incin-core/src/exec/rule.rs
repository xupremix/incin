//! Lowering rules: the only way a [`Validated`] descriptor comes into being.
//!
//! `EXE-001` froze the descriptors and `EXE-002` sealed them behind
//! [`Validated<O>`](Validated), whose constructor is `pub(crate)`. This module
//! is what that constructor was reserved for. A [`ShapeRule`] takes the runtime
//! half of a shape the frontend has already proved legal, resolves it into a
//! descriptor, and stamps it with the [`ProofLevel`] the operand types earned.
//!
//! # The binding to the frontend
//!
//! Every rule here restates the `Output` its frontend trait already names —
//! decision `D-007`, and the reason this module has no shape rules of its own:
//!
//! | Rule | Frontend trait | Descriptor |
//! |---|---|---|
//! | [`BroadcastRule`] | `BroadcastShape<Rhs>` | [`BroadcastSpec`] |
//! | [`MatMulRule`] | `MatMulShape<Rhs>` | [`MatMulSpec`] |
//! | [`ReduceRule`] / [`ReduceKeepRule`] | structural axis cursors | [`ReductionSpec`] |
//! | [`ReduceAllRule`] | none; the output is always [`Scalar`] | [`ReductionSpec`] |
//!
//! [`Scalar`]: crate::shapes::shape::Scalar
//! | [`ReshapeRule`] | `ReshapeShape<Target>` | [`ReshapeSpec`] |
//! | [`Conv2dRule`] | `SpatialConv2d<COut, K, S, P, D>` | [`Conv2dSpec`] |
//! | [`Pool2dRule`] | `Pool2dShape<K, S, P, D>` | [`Pool2dSpec`] |
//!
//! Restating the `Output` makes a divergence between the two a compile error,
//! but only for the axes the compiler settled. For a `Mixed` or `Dynamic` shape
//! the type says nothing about the sizes, so each rule also checks its
//! descriptor's output against the frontend's answer at runtime, by whichever
//! route the frontend trait offers:
//!
//! * traits that compute a runtime `ShapeBuf` — `BroadcastShape`, `MatMulShape`,
//!   `SpatialConv2d`, `Pool2dShape` — are called, and their dimensions are
//!   compared against the descriptor's axis by axis;
//! * traits that only name a type — structural reduction cursors — are checked
//!   by validating the output `ShapeBuf` against that type's shape contract, which
//!   fails if the rank differs or a statically fixed axis disagrees.
//!
//! [`ReduceAllRule`] has no frontend trait to restate, because reducing every
//! axis constrains nothing about the input: any shape reduces to a scalar. The
//! runtime check is the same rebuild against [`Scalar`], which has rank 0, so a
//! descriptor claiming `[1]` is rejected rather than accepted as close enough.
//!
//! Either way the two computations are independent and must agree, which is the
//! runtime half of what `D-007` asks the type system for.
//!
//! # What a rule does not take
//!
//! Operand strides, offsets, dtype, and device are per-tensor facts owned by
//! `TensorMeta` (`EXE-004`), so no rule here accepts them and every rule lowers
//! dense row-major operands. The one descriptor field that depends on layout,
//! [`MatMulSpec::transpose_lhs`], is set by
//! [`transposed`](MatMulSpec::transposed) after lowering for exactly that
//! reason. When `EXE-004` lands, `Args` is where the operand metadata arrives.
//!
//! # Why `ShapeRule` is not sealed
//!
//! [`OperationSpec`] is sealed, so [`Descriptor`](ShapeRule::Descriptor) can
//! only ever be one of this crate's descriptors, and
//! [`Validated::new`](Validated) is `pub(crate)`. An outside implementation of
//! [`ShapeRule`] therefore has no way to mint a descriptor of its own or to
//! wrap one it did not obtain from a rule here — the most it can do is call one
//! of these rules and pass the result along. That is delegation, not forgery,
//! and there is no reason to forbid it.

use core::marker::PhantomData;

use super::catalog::{Descriptor, LogicalTensorMeta, NoAttributes, ShapeAttributes, op};
use super::proof::{ProofLevel, Validated};
use super::spec::{
    BinaryOp, BroadcastSpec, Conv2dSpec, ExecutionDescriptor, MatMulSpec, Pool2dSpec, PoolOp,
    ReduceOp, ReductionSpec, ReshapeSpec,
};
use crate::shapes::broadcast::BroadcastShape;
use crate::shapes::buf::ShapeBuf;
use crate::shapes::dim::Dim;
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation, ShapeError};
use crate::shapes::idx::StaticCursor;
use crate::shapes::reshape::ReshapeShape;
use crate::shapes::shape::{DynShape, Scalar, Shape, shape_buf_from_dims};
use crate::shapes::shape_ops::{ReduceAt, ReduceKeepAt};
use crate::shapes::spatial::{Pool2dShape, SpatialConv2d};
use crate::tensor::matmul::MatMulShape;
use typenum::Unsigned;

/// Resolving one operation into the descriptor a backend executes.
///
/// `Inputs` is the operand *shape types*, which is what binds a rule to the
/// frontend trait that governs them: `BroadcastRule` implements
/// `ShapeRule<(L, R)>` for every `L: BroadcastShape<R>`, and nothing else.
///
/// A rule is a type-level function, never a value. Every method is associated,
/// and the rule types in this module are uninhabited or empty precisely because
/// there is nothing to carry: the operand shapes are the `Inputs` parameter and
/// the runtime dimensions are an argument.
pub trait ShapeRule<Inputs>: Sized {
    /// The output shape.
    ///
    /// Not a second opinion: `D-007` requires this to be the same `Output` the
    /// frontend trait for this operation already computes, so the two cannot
    /// drift apart without failing to compile.
    type Output: Shape;

    /// The runtime half of `Inputs` — each operand's authoritative
    /// [`ShapeBuf`].
    ///
    /// Stated as an associated type rather than written out as
    /// `<Inputs as Shape>` because `Inputs` is a *tuple* of shapes for
    /// the binary rules, and a tuple of shapes is not itself a `Shape`.
    type Operands;

    /// Everything the shape types do not determine.
    ///
    /// `()` wherever the operation is fixed by its operand types alone, as
    /// matmul and reshape are. It is not empty where a family of operations
    /// shares one shape rule: a structural cursor carries the axis but not the
    /// accumulation, `Pool2dShape<K, S, P, D>` carries the window but not what
    /// runs inside it, and `BroadcastShape<Rhs>` carries the geometry of a
    /// stretch that four binary operations read the same way, so each takes its
    /// operator here. `Conv2dArgs` carries grouping for the same reason — it
    /// constrains the channel axes without being a shape parameter.
    type Args;

    /// The descriptor this rule resolves to.
    ///
    /// A descriptor that can expose the resolved output shape. Built-in rules
    /// increasingly use exact catalog descriptors while the remaining
    /// geometry rules are migrated without changing this seam.
    type Descriptor: super::spec::ExecutionDescriptor;

    /// Resolve the operands into a descriptor, or report why they do not.
    fn lower(
        operands: &Self::Operands,
        args: Self::Args,
    ) -> Result<Validated<Self::Descriptor>, ShapeError>;
}

// --- shared machinery -------------------------------------------------------

/// One operand's runtime dimensions, as the descriptors want them.
fn dims_of(dims: &ShapeBuf) -> ShapeBuf {
    dims.clone()
}

/// Cross-check the frontend's output shape against the descriptor's.
///
/// The two are computed independently — the frontend from typed operand fields,
/// the descriptor from erased dimensions — and `D-007` exists because a
/// disagreement between them would otherwise surface inside a kernel. A rank
/// difference or any differing axis is reported here, at the operation the
/// caller actually asked for.
fn agree(
    operation: OperationKind,
    frontend: &ShapeBuf,
    descriptor: &ShapeBuf,
) -> Result<(), ShapeError> {
    if frontend.rank() != descriptor.rank() {
        return Err(ShapeError::RankMismatch {
            operation,
            expected: RankExpectation::SameAs {
                operand: "shape rule output",
                rank: frontend.rank(),
            },
            actual: descriptor.rank(),
        });
    }
    for (axis, (&lhs, &rhs)) in frontend
        .dims()
        .iter()
        .zip(descriptor.dims().iter())
        .enumerate()
    {
        if lhs != rhs {
            return Err(ShapeError::DimensionMismatch {
                operation,
                axis: Axis::Index(axis),
                lhs,
                rhs,
                constraint: DimensionConstraint::Equal,
            });
        }
    }
    Ok(())
}

// --- broadcast --------------------------------------------------------------

/// Lowers elementwise operations, including binary pointwise arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct BroadcastRule;

impl<L, R> ShapeRule<(L, R)> for BroadcastRule
where
    L: BroadcastShape<R> + DynShape,
    R: Shape + DynShape,
    <L as BroadcastShape<R>>::Output: DynShape,
{
    type Output = <L as BroadcastShape<R>>::Output;
    type Operands = (ShapeBuf, ShapeBuf);
    /// The operator, which the shape types do not determine: the same
    /// broadcast geometry serves a stretch and all four binary operations.
    /// This is where `Conv2dArgs` puts grouping and the reduce rules put
    /// [`ReduceOp`], for the same reason.
    type Args = Option<BinaryOp>;
    type Descriptor = BroadcastSpec;

    fn lower(
        operands: &Self::Operands,
        op: Self::Args,
    ) -> Result<Validated<BroadcastSpec>, ShapeError> {
        // Materialize the structural static proof at the canonical rule
        // boundary.  Without this, the deliberately general BroadcastExtent
        // fallback would turn a statically impossible pair into a runtime
        // error and erase a useful compile-time rejection.
        <Self::Output as Shape>::STATIC_VALID;
        // The frontend runs first because it, not the descriptor, is bound to
        // the output *type*: `output_shape` rebuilds that type's field from the
        // dimensions it resolved and fails if one does not fit. The descriptor
        // sees erased numbers and has nothing to check them against.
        let resolved = L::output_shape(&operands.0, &operands.1)?;
        let expected = dims_of(&resolved);

        let spec = BroadcastSpec::contiguous(&dims_of(&operands.0), &dims_of(&operands.1), op)?;
        agree(OperationKind::Broadcast, &expected, &spec.output)?;

        Ok(Validated::new(
            spec,
            ProofLevel::of::<L>().meet(ProofLevel::of::<R>()),
        ))
    }
}

// --- matrix multiplication --------------------------------------------------

/// Lowers matrix multiplication, including its batched form.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct MatMulRule;

impl<L, R> ShapeRule<(L, R)> for MatMulRule
where
    L: MatMulShape<R> + DynShape,
    R: Shape + DynShape,
    <L as MatMulShape<R>>::Output: DynShape,
{
    type Output = <L as MatMulShape<R>>::Output;
    type Operands = (ShapeBuf, ShapeBuf);
    type Args = ();
    type Descriptor = Descriptor<op::MatMulExact>;

    fn lower(
        operands: &Self::Operands,
        (): Self::Args,
    ) -> Result<Validated<Descriptor<op::MatMulExact>>, ShapeError> {
        let resolved = L::output_shape(&operands.0, &operands.1)?;
        let expected = dims_of(&resolved);

        let descriptor = Descriptor::<op::MatMulExact>::infer_runtime(
            NoAttributes,
            alloc::vec![
                LogicalTensorMeta {
                    shape: Some(dims_of(&operands.0)),
                    dtype: None,
                    device: None,
                },
                LogicalTensorMeta {
                    shape: Some(dims_of(&operands.1)),
                    dtype: None,
                    device: None,
                },
            ],
        )
        .map_err(|error| match error {
            super::catalog::DescriptorError::Shape(error) => error,
            _ => ShapeError::TargetShapeRejected {
                operation: OperationKind::MatMul,
                rank: expected.rank(),
            },
        })?;
        let descriptor_shape =
            descriptor
                .descriptor()
                .output_shape()
                .ok_or(ShapeError::TargetShapeRejected {
                    operation: OperationKind::MatMul,
                    rank: expected.rank(),
                })?;
        agree(OperationKind::MatMul, &expected, descriptor_shape)?;

        Ok(Validated::new(
            descriptor.into_descriptor(),
            ProofLevel::of::<L>().meet(ProofLevel::of::<R>()),
        ))
    }
}

// --- reduction --------------------------------------------------------------

/// Lowers a reduction selected by the canonical structural cursor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct ReduceAtRule<C>(PhantomData<fn() -> C>);

/// Lowers a keep-dimension reduction selected by the canonical structural
/// cursor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct ReduceKeepAtRule<C>(PhantomData<fn() -> C>);

/// Lowers a reduction over every axis, producing a scalar.
///
/// The single-axis rules cover a structural cursor, which names one axis at compile
/// time. Total reduction has no axis to name: `ReductionOps::sum_all` and its
/// siblings already exist on every backend and are what autograd calls to turn a
/// loss into a scalar, but they had no descriptor, so the one operation every
/// training step ends with was the one that could not be expressed as one.
///
/// The output is [`Scalar`], which is the honest answer and was not always the
/// one given: `EXE-005` found WGPU reporting `[1]` for an all-reduction, a rank-1
/// tensor standing in for a rank-0 one. `()` has rank 0 and `NUMEL` 1, so
/// `shape_buf_from_dims` rejects any descriptor that disagrees.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct ReduceAllRule;

impl<S> ShapeRule<S> for ReduceAllRule
where
    S: DynShape,
{
    type Output = Scalar;
    type Operands = ShapeBuf;
    /// See [`ReduceRule::Args`](ShapeRule::Args).
    type Args = ReduceOp;
    type Descriptor = ReductionSpec;

    fn lower(
        operands: &Self::Operands,
        op: Self::Args,
    ) -> Result<Validated<ReductionSpec>, ShapeError> {
        let spec = ReductionSpec::over_all(&dims_of(operands), false, op)?;
        shape_buf_from_dims::<Self::Output>(OperationKind::Reduction, spec.output.dims())?;
        Ok(Validated::new(spec, ProofLevel::of::<S>()))
    }
}

fn lower_reduction<S, Output>(
    field: &ShapeBuf,
    axis: usize,
    keep_dims: bool,
    op: ReduceOp,
) -> Result<Validated<ReductionSpec>, ShapeError>
where
    S: DynShape,
    Output: Shape,
{
    let spec = ReductionSpec::over_axes(&dims_of(field), [axis], keep_dims, op)?;
    shape_buf_from_dims::<Output>(OperationKind::Reduction, spec.output.dims())?;
    Ok(Validated::new(spec, ProofLevel::of::<S>()))
}

impl<S, C> ShapeRule<S> for ReduceAtRule<C>
where
    C: StaticCursor,
    S: ReduceAt<C> + DynShape,
    <S as ReduceAt<C>>::Output: DynShape,
{
    type Output = <S as ReduceAt<C>>::Output;
    type Operands = ShapeBuf;
    type Args = ReduceOp;
    type Descriptor = ReductionSpec;

    fn lower(
        operands: &Self::Operands,
        op: Self::Args,
    ) -> Result<Validated<ReductionSpec>, ShapeError> {
        let rank = dims_of(operands).as_ref().len();
        let axis = crate::shapes::idx::AxisSelector::new(&[C::INDEX])
            .normalize(rank)
            .map_err(|_| ShapeError::InvalidAxis {
                axis: C::INDEX.unsigned_abs(),
                rank,
            })?[0];
        lower_reduction::<S, Self::Output>(operands, axis, false, op)
    }
}

impl<S, C> ShapeRule<S> for ReduceKeepAtRule<C>
where
    C: StaticCursor,
    S: ReduceKeepAt<C> + DynShape,
    <S as ReduceKeepAt<C>>::Output: DynShape,
{
    type Output = <S as ReduceKeepAt<C>>::Output;
    type Operands = ShapeBuf;
    type Args = ReduceOp;
    type Descriptor = ReductionSpec;

    fn lower(
        operands: &Self::Operands,
        op: Self::Args,
    ) -> Result<Validated<ReductionSpec>, ShapeError> {
        let rank = dims_of(operands).as_ref().len();
        let axis = crate::shapes::idx::AxisSelector::new(&[C::INDEX])
            .normalize(rank)
            .map_err(|_| ShapeError::InvalidAxis {
                axis: C::INDEX.unsigned_abs(),
                rank,
            })?[0];
        lower_reduction::<S, Self::Output>(operands, axis, true, op)
    }
}

// --- reshape ----------------------------------------------------------------

/// Lowers a reinterpretation of one shape as another.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct ReshapeRule;

impl<S, T> ShapeRule<(S, T)> for ReshapeRule
where
    S: ReshapeShape<T> + DynShape,
    T: Shape + DynShape,
{
    /// The target shape is the output. `ReshapeShape<Target>` is a marker with
    /// no `Output` of its own, so restating it means naming `Target` — there is
    /// no second computation to drift from.
    type Output = T;
    type Operands = (ShapeBuf, ShapeBuf);
    type Args = ();
    type Descriptor = Descriptor<op::ReshapeExact>;

    fn lower(
        operands: &Self::Operands,
        (): Self::Args,
    ) -> Result<Validated<Descriptor<op::ReshapeExact>>, ShapeError> {
        let descriptor = Descriptor::<op::ReshapeExact>::infer_runtime(
            ShapeAttributes {
                shape: operands.1.as_ref().to_vec(),
            },
            alloc::vec![LogicalTensorMeta {
                shape: Some(dims_of(&operands.0)),
                dtype: None,
                device: None,
            }],
        )
        .map_err(|error| match error {
            super::catalog::DescriptorError::Shape(error) => error,
            _ => ShapeError::TargetShapeRejected {
                operation: OperationKind::Reshape,
                rank: operands.1.rank(),
            },
        })?;
        Ok(Validated::new(
            descriptor.into_descriptor(),
            ProofLevel::of::<S>().meet(ProofLevel::of::<T>()),
        ))
    }
}

// --- convolution ------------------------------------------------------------

/// Everything a two-dimensional convolution needs that its shape types do not
/// carry.
///
/// `SpatialConv2d` fixes the window, stride, padding, and dilation as `typenum`
/// parameters, so the only shape-relevant value left is the output channel
/// count — which `COut` may leave as a runtime `usize`. Grouping is not a shape
/// parameter at all: it constrains how the channel axes divide, and the typed
/// frontend does not model it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Conv2dArgs {
    /// The runtime value of `COut`.
    pub out_channels: usize,
    /// How many independent channel groups the convolution splits into.
    pub groups: usize,
}

impl Conv2dArgs {
    /// A dense convolution, every input channel feeding every output channel.
    #[must_use]
    pub const fn dense(out_channels: usize) -> Self {
        Self {
            out_channels,
            groups: 1,
        }
    }
}

/// Lowers a two-dimensional convolution.
#[derive(Debug)]
pub struct Conv2dRule<COut, K, S, P, D>(PhantomData<fn() -> (COut, K, S, P, D)>);

impl<Sh, COut, K, S, P, D> ShapeRule<Sh> for Conv2dRule<COut, K, S, P, D>
where
    Sh: SpatialConv2d<COut, K, S, P, D> + DynShape,
    COut: Dim,
    K: Unsigned,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
{
    type Output = <Sh as SpatialConv2d<COut, K, S, P, D>>::Output;
    type Operands = ShapeBuf;
    type Args = Conv2dArgs;
    type Descriptor = Conv2dSpec;

    fn lower(
        operands: &Self::Operands,
        args: Conv2dArgs,
    ) -> Result<Validated<Conv2dSpec>, ShapeError> {
        let resolved = Sh::compute_output_shape(operands, args.out_channels)?;
        let expected = dims_of(&resolved);

        let spec = Conv2dSpec::new(
            &dims_of(operands),
            args.out_channels,
            [K::USIZE; 2],
            [S::USIZE; 2],
            [P::USIZE; 2],
            [D::USIZE; 2],
            args.groups,
        )?;
        agree(OperationKind::Conv2d, &expected, &spec.output)?;

        Ok(Validated::new(
            spec,
            // The input shape is only half the proof: an output channel count
            // the type leaves as `usize` was settled at runtime like any other
            // dynamic axis, and the descriptor is no stronger than that.
            ProofLevel::of::<Sh>().meet(ProofLevel::of_ranked(COut::STATIC_SIZE)),
        ))
    }
}

// --- pooling ----------------------------------------------------------------

/// Lowers two-dimensional pooling, maximum or average alike.
#[derive(Debug)]
pub struct Pool2dRule<K, S, P, D>(PhantomData<fn() -> (K, S, P, D)>);

impl<Sh, K, S, P, D> ShapeRule<Sh> for Pool2dRule<K, S, P, D>
where
    Sh: Pool2dShape<K, S, P, D> + DynShape,
    K: Unsigned,
    S: Unsigned,
    P: Unsigned,
    D: Unsigned,
{
    type Output = <Sh as Pool2dShape<K, S, P, D>>::Output;
    type Operands = ShapeBuf;
    /// `Pool2dShape` fixes the window as `typenum` parameters, which leaves the
    /// accumulation inside it as the one thing the shape types do not determine.
    type Args = PoolOp;
    type Descriptor = Pool2dSpec;

    fn lower(
        operands: &Self::Operands,
        op: Self::Args,
    ) -> Result<Validated<Pool2dSpec>, ShapeError> {
        let resolved = Sh::compute_output_shape(operands)?;
        let expected = dims_of(&resolved);

        let spec = Pool2dSpec::new(
            &dims_of(operands),
            [K::USIZE; 2],
            [S::USIZE; 2],
            [P::USIZE; 2],
            [D::USIZE; 2],
            op,
        )?;
        agree(OperationKind::Pool2d, &expected, &spec.output)?;

        Ok(Validated::new(spec, ProofLevel::of::<Sh>()))
    }
}
