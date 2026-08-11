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
//! Every rule here restates the `Output` its frontend trait already names.
//! decision `D-007`, and the reason this module has no shape rules of its own:
//!
//! | Rule | Frontend trait | Descriptor |
//! |---|---|---|
//! | [`MatMulRule`] | `MatMulShape<Rhs>` | `Descriptor<op::MatMulExact>` |
//! | [`ReshapeRule`] | `ReshapeShape<Target>` | `Descriptor<op::ReshapeExact>` |
//!
//! Restating the `Output` makes a divergence between the two a compile error,
//! but only for the axes the compiler settled. For a `Mixed` or `Dynamic` shape
//! the type says nothing about the sizes, so each rule also checks its
//! descriptor's output against the frontend's answer at runtime, by whichever
//! route the frontend trait offers:
//!
//! The exact descriptors are independently inferred from erased dimensions and
//! checked against the typed frontend output.
//!
//! Either way the two computations are independent and must agree, which is the
//! runtime half of what `D-007` asks the type system for.
//!
//! # What a rule does not take
//!
//! Operand strides, offsets, dtype, and device are per-tensor facts owned by
//! `TensorMeta` (`EXE-004`), so no rule here accepts them.
//!
//! # Why `ShapeRule` is not sealed
//!
//! [`ExecutionDescriptor`](super::spec::ExecutionDescriptor) is implemented
//! only for canonical descriptors, and [`Validated::new`](Validated) is
//! `pub(crate)`. An outside implementation of
//! [`ShapeRule`] therefore has no way to mint a descriptor of its own or to
//! wrap one it did not obtain from a rule here. The most it can do is call one
//! of these rules and pass the result along. That is delegation, not forgery,
//! and there is no reason to forbid it.

use super::catalog::{Descriptor, LogicalTensorMeta, NoAttributes, ShapeAttributes, op};
use super::proof::{ProofLevel, Validated};
use super::spec::ExecutionDescriptor;
use crate::shapes::buf::ShapeBuf;
use crate::shapes::error::{Axis, DimensionConstraint, OperationKind, RankExpectation, ShapeError};
use crate::shapes::reshape::ReshapeShape;
use crate::shapes::shape::{DynShape, Shape};
use crate::tensor::matmul::MatMulShape;

/// Resolving one operation into the descriptor a backend executes.
///
/// `Inputs` is the operand shape types, which binds a rule to its frontend
/// shape trait.
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

    /// The runtime half of `Inputs`. Each operand's authoritative
    /// [`ShapeBuf`].
    ///
    /// Stated as an associated type rather than written out as
    /// `<Inputs as Shape>` because `Inputs` is a *tuple* of shapes for
    /// the binary rules, and a tuple of shapes is not itself a `Shape`.
    type Operands;

    /// Everything the shape types do not determine.
    ///
    /// `()` for the canonical rules. Operation-specific attributes belong to
    /// the exact descriptor catalog.
    type Args;

    /// The descriptor this rule resolves to.
    ///
    /// A canonical descriptor that exposes the resolved output shape.
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
/// The two are computed independently. The frontend uses typed operand fields,
/// while the descriptor uses erased dimensions. `D-007` exists because a
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
    /// no `Output` of its own, so restating it means naming `Target`. There is
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
