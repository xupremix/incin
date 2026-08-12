//! Typed frontend shape rules for canonical descriptor lowering.
//!
//! A rule binds a structural frontend proof to the exact descriptor family
//! consumed by execution. Runtime dimensions remain in `ShapeBuf`.

use super::catalog::{
    AxisAttributes, Descriptor, LogicalTensorMeta, NoAttributes, ShapeAttributes, op,
};
use super::proof::{ProofLevel, Validated};
use super::spec::ExecutionDescriptor;
use crate::shapes::buf::ShapeBuf;
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation, ShapeError};
use crate::shapes::idx::StaticCursor;
use crate::shapes::reshape::ReshapeShape;
use crate::shapes::shape::{DynShape, Shape, ShapeValue};
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
        Ok(Validated::new(
            descriptor.into_descriptor(),
            ProofLevel::of::<L>().meet(ProofLevel::of::<R>()),
        ))
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
        Ok(Validated::new(
            descriptor.into_descriptor(),
            ProofLevel::of::<S>().meet(ProofLevel::of::<T>()),
        ))
    }
}

/// Canonical structural reduction shape rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ReduceRule;

impl<S, C> ShapeRule<(S, C)> for ReduceRule
where
    C: StaticCursor,
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
        Ok(Validated::new(
            descriptor.into_descriptor(),
            ProofLevel::of::<S>(),
        ))
    }
}

/// Canonical structural keepdim reduction shape rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ReduceKeepRule;

impl<S, C> ShapeRule<(S, C)> for ReduceKeepRule
where
    C: StaticCursor,
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
        let actual =
            descriptor
                .descriptor()
                .output_shape()
                .ok_or(ShapeError::TargetShapeRejected {
                    operation: OperationKind::SumKeepDim,
                    rank: operands.rank(),
                })?;
        let expected = ShapeValue::<<S as ReduceKeepAt<C>>::Output>::try_new(actual.clone())?;
        agree(OperationKind::SumKeepDim, expected.shape_buf(), actual)?;
        Ok(Validated::new(
            descriptor.into_descriptor(),
            ProofLevel::of::<S>(),
        ))
    }
}
