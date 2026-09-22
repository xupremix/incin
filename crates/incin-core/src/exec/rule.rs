//! Typed frontend shape rules for canonical descriptor lowering.
//!
//! A rule binds a structural frontend proof to the exact descriptor family
//! consumed by execution. Runtime dimensions remain in `ShapeBuf`.

use super::catalog::{
    AxisAttributes, Descriptor, LogicalTensorMeta, NoAttributes, ShapeAttributes, op,
};
use super::proof::{ShapeEvidence, Validated};
use super::spec::ExecutionDescriptor;
use crate::shapes::ShapeBuf;
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation, ShapeError};
use crate::shapes::idx::AxisSelector;
use crate::shapes::reshape::ReshapeShape;
use crate::shapes::shape::{DynShape, Shape};
use crate::shapes::shape_ops::{ReduceAt, ReduceKeepAt};
use crate::tensor::matmul::MatMulShape;

/// Resolves typed shape operands into a validated canonical descriptor.
pub trait ShapeRule<Inputs>: Sized {
    /// Output shape selected by the frontend operation.
    type Output: Shape;
    /// Runtime shape values supplied by the operands.
    type Operands;
    /// Operation attributes not encoded by the input shape types.
    type Args;
    /// Canonical descriptor consumed by execution.
    type Descriptor: ExecutionDescriptor;

    /// Lower the typed operation, validating runtime dimensions as needed.
    fn lower(
        operands: &Self::Operands,
        args: Self::Args,
    ) -> Result<Validated<Self::Descriptor>, ShapeError>;
}

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
    for (axis, (&lhs, &rhs)) in frontend.dims().iter().zip(descriptor.dims()).enumerate() {
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

/// Check the descriptor's runtime output against the frontend output type
/// `Out` and seal it with that type's full [`ShapeEvidence`].
///
/// The rule-side construction path ([`Descriptor::infer_runtime`]) attaches
/// only a proof level with empty static geometry: it knows the operation was
/// checked, not which shape type the frontend proved. The rule holds that type
/// (`Self::Output`), so before the static extents on
/// [`ShapeEvidence::of`] travel to a backend this helper discharges three
/// obligations:
///
/// 1. the descriptor actually carries an output shape;
/// 2. the runtime dimensions the catalog inferred are accepted by
///    `Out::validate_dims` - the same check [`shape_buf_from_dims`] performs
///    when rebuilding a typed field from computed dimensions. Going through
///    `validate_dims` rather than zipping `STATIC_EXTENTS` is deliberate: the
///    recursive check keeps validating past [`MAX_STATIC_RANK`], where the
///    const extent buffer is truncated to silence. A manual zip would stop
///    checking at the buffer bound and accept a shape it cannot see - fail
///    open, which is the one direction this layer may not go;
/// 3. only then is `ShapeEvidence::of::<Out>()` minted, because a static
///    extent the runtime shape contradicts is a miscompile, not a missed
///    optimisation.
///
/// A rejection is remapped to `operation` with the real rank, matching
/// [`shape_buf_from_dims`]'s convention, so the diagnostic names the rule that
/// failed rather than the generic `Storage` identity the recursive check
/// carries. `fallback_rank` is used only when obligation 1 fails and there is
/// no runtime shape to measure.
///
/// [`shape_buf_from_dims`]: crate::shapes::shape::shape_buf_from_dims
/// [`MAX_STATIC_RANK`]: crate::shapes::MAX_STATIC_RANK
fn seal_output<O, Out: Shape>(
    operation: OperationKind,
    fallback_rank: usize,
    descriptor: Validated<Descriptor<O>>,
) -> Result<Validated<Descriptor<O>>, ShapeError>
where
    O: super::catalog::Operation,
{
    let actual = descriptor
        .descriptor()
        .output_shape()
        .ok_or(ShapeError::TargetShapeRejected {
            operation,
            rank: fallback_rank,
        })?;
    Out::validate_dims(actual.as_ref()).map_err(|error| match error {
        ShapeError::TargetShapeRejected { .. } => ShapeError::TargetShapeRejected {
            operation,
            rank: actual.rank(),
        },
        other => other,
    })?;
    Ok(Validated::new_with_evidence(
        descriptor.into_descriptor(),
        ShapeEvidence::of::<Out>(),
    ))
}

fn cursor_axis<C: crate::shapes::idx::AxisCursor>(rank: usize) -> Result<usize, ShapeError> {
    AxisSelector::new(&[C::INDEX])
        .normalize(rank)
        .map_err(|error| match error {
            crate::err::Error::Shape(error) => error,
            _ => ShapeError::InvalidAxis { axis: rank, rank },
        })?
        .into_iter()
        .next()
        .ok_or(ShapeError::InvalidAxis { axis: rank, rank })
}

/// Canonical batched matrix multiplication shape rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
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
    ) -> Result<Validated<Self::Descriptor>, ShapeError> {
        let expected = L::output_shape(&operands.0, &operands.1)?;
        let descriptor = Descriptor::<op::MatMulExact>::infer_runtime(
            NoAttributes,
            alloc::vec![
                LogicalTensorMeta {
                    shape: Some(operands.0.clone()),
                    dtype: None,
                    device: None,
                },
                LogicalTensorMeta {
                    shape: Some(operands.1.clone()),
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
        let actual =
            descriptor
                .descriptor()
                .output_shape()
                .ok_or(ShapeError::TargetShapeRejected {
                    operation: OperationKind::MatMul,
                    rank: expected.rank(),
                })?;
        agree(OperationKind::MatMul, &expected, actual)?;
        seal_output::<_, Self::Output>(OperationKind::MatMul, expected.rank(), descriptor)
    }
}

/// Canonical exact reshape shape rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ReshapeRule;

impl<S, T> ShapeRule<(S, T)> for ReshapeRule
where
    S: ReshapeShape<T> + DynShape,
    T: Shape + DynShape,
{
    type Output = T;
    type Operands = (ShapeBuf, ShapeBuf);
    type Args = ();
    type Descriptor = Descriptor<op::ReshapeExact>;

    fn lower(
        operands: &Self::Operands,
        (): Self::Args,
    ) -> Result<Validated<Self::Descriptor>, ShapeError> {
        let descriptor = Descriptor::<op::ReshapeExact>::infer_runtime(
            ShapeAttributes {
                shape: operands.1.as_ref().to_vec(),
            },
            alloc::vec![LogicalTensorMeta {
                shape: Some(operands.0.clone()),
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
        seal_output::<_, T>(OperationKind::Reshape, operands.1.rank(), descriptor)
    }
}

/// Canonical structural reduction shape rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ReduceRule;

impl<S, C> ShapeRule<(S, C)> for ReduceRule
where
    C: crate::shapes::idx::AxisCursor,
    S: DynShape + ReduceAt<C>,
    <S as ReduceAt<C>>::Output: DynShape,
{
    type Output = <S as ReduceAt<C>>::Output;
    type Operands = ShapeBuf;
    type Args = AxisAttributes;
    type Descriptor = Descriptor<op::SumDim>;

    fn lower(
        operands: &Self::Operands,
        args: Self::Args,
    ) -> Result<Validated<Self::Descriptor>, ShapeError> {
        let expected_axis = cursor_axis::<C>(operands.rank())?;
        if args.axis != expected_axis {
            return Err(ShapeError::InvalidAxis {
                axis: args.axis,
                rank: operands.rank(),
            });
        }
        let expected = S::reduce_shape(operands).map_err(|error| match error {
            crate::err::Error::Shape(error) => error,
            _ => ShapeError::TargetShapeRejected {
                operation: OperationKind::SumDim,
                rank: operands.rank(),
            },
        })?;
        let descriptor = Descriptor::<op::SumDim>::infer_runtime(
            args,
            alloc::vec![LogicalTensorMeta {
                shape: Some(operands.clone()),
                dtype: None,
                device: None,
            }],
        )
        .map_err(|error| match error {
            super::catalog::DescriptorError::Shape(error) => error,
            _ => ShapeError::TargetShapeRejected {
                operation: OperationKind::SumDim,
                rank: expected.rank(),
            },
        })?;
        let actual =
            descriptor
                .descriptor()
                .output_shape()
                .ok_or(ShapeError::TargetShapeRejected {
                    operation: OperationKind::SumDim,
                    rank: expected.rank(),
                })?;
        agree(OperationKind::SumDim, &expected, actual)?;
        seal_output::<_, Self::Output>(OperationKind::SumDim, expected.rank(), descriptor)
    }
}

/// Canonical structural keepdim reduction shape rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ReduceKeepRule;

impl<S, C> ShapeRule<(S, C)> for ReduceKeepRule
where
    C: crate::shapes::idx::AxisCursor,
    S: DynShape + ReduceKeepAt<C>,
    <S as ReduceKeepAt<C>>::Output: DynShape,
{
    type Output = <S as ReduceKeepAt<C>>::Output;
    type Operands = ShapeBuf;
    type Args = AxisAttributes;
    type Descriptor = Descriptor<op::SumKeepDim>;

    fn lower(
        operands: &Self::Operands,
        args: Self::Args,
    ) -> Result<Validated<Self::Descriptor>, ShapeError> {
        let expected_axis = cursor_axis::<C>(operands.rank())?;
        if args.axis != expected_axis {
            return Err(ShapeError::InvalidAxis {
                axis: args.axis,
                rank: operands.rank(),
            });
        }
        let descriptor = Descriptor::<op::SumKeepDim>::infer_runtime(
            args,
            alloc::vec![LogicalTensorMeta {
                shape: Some(operands.clone()),
                dtype: None,
                device: None,
            }],
        )
        .map_err(|error| match error {
            super::catalog::DescriptorError::Shape(error) => error,
            _ => ShapeError::TargetShapeRejected {
                operation: OperationKind::SumKeepDim,
                rank: operands.rank(),
            },
        })?;
        seal_output::<_, Self::Output>(OperationKind::SumKeepDim, operands.rank(), descriptor)
    }
}
