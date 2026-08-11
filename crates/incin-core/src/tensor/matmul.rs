//! Matrix multiplication with compile-time shape verification.
//!
//! The `MatMulShape` trait encodes the shape compatibility rules for matmul:
//! - `(M, K) × (K, N) -> (M, N)`, inner dimensions must match
//! - Batched and fully-dynamic variants are also supported.
//!
//! **Static shapes**: The compiler rejects mismatched inner dims at compile time.
//! **Dynamic shapes**: Mismatches are caught at runtime by candle.

use crate::dist::Local;
use crate::exec::catalog::{AddmmAttributes, AttentionAttributes, Descriptor, op};
use crate::exec::context::ExecutionContext;
use crate::exec::request::TensorHandle;
use crate::prelude::*;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::shape_buf_from_dims;
use crate::tensor::backend::Execute;
use alloc::vec::Vec;

// ============================================================================
// MatMulShape trait: compile-time shape compatibility for matmul
// ============================================================================

/// Trait that verifies two shapes are compatible for matrix multiplication
/// and determines the output shape.
///
/// Implement this for shape pairs that can be multiplied together.
/// The compiler will reject any `matmul` call where this trait is not implemented.
#[diagnostic::on_unimplemented(
    message = "Cannot matrix-multiply shape `{Self}` with `{Rhs}`",
    label = "Shape mismatch for matrix multiplication",
    note = "Matrix multiplication requires the inner dimensions (last dim of lhs, second-to-last of rhs) to match"
)]
/// Compile-time-checked matrix multiplication shape rule: `Self` (lhs)
/// multiplied by `Rhs` produces `Output`.
pub trait MatMulShape<Rhs: Shape>: Shape + DynShape {
    /// The resulting shape after multiplying `Self` by `Rhs`.
    type Output: Shape + DynShape;

    /// Compute the output `ShapeBuf` from the input buffers.
    fn output_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<ShapeBuf, ShapeError>;
}

/// Marker for a compile-time-fixed (`typenum`) dimension usable in
/// static matmul shape rules, as opposed to a runtime `usize`.
pub trait StaticDim: Dim + Default + crate::shapes::ConcreteStaticExtent {}
impl<U, B> StaticDim for typenum::UInt<U, B>
where
    U: typenum::Unsigned + Dim,
    B: typenum::Bit
        + Default
        + Copy
        + Clone
        + core::fmt::Debug
        + Send
        + Sync
        + Eq
        + PartialEq
        + 'static,
    typenum::UInt<U, B>: typenum::Unsigned
        + Default
        + Copy
        + Clone
        + core::fmt::Debug
        + Send
        + Sync
        + Eq
        + PartialEq
        + 'static,
{
}
impl StaticDim for typenum::UTerm {}

/// Marker for a dimension with a type-level identity: typenum dims, checked
/// `MulDim` composites, and symbolic names, but not a runtime `usize`.
///
/// ContractsWith and the broadcast axis rule use this marker to keep their identity and runtime impls disjoint. Whole matmul shapes are otherwise bounded by Dim, so a runtime axis may appear at any supported position.
pub trait StaticOrNamedDim: Dim {}
impl<T: StaticDim> StaticOrNamedDim for T {}
impl<const N: usize> StaticOrNamedDim for crate::shapes::dim::ConstDim<N> {}
impl<Tag: crate::shapes::AxisTag, Extent: Dim> StaticOrNamedDim
    for crate::shapes::dim::NamedDim<Tag, Extent>
{
}

/// Check that the contracted dimension agrees between the two operands.
///
/// For a `typenum` `K` the type system already proves this. For a `dim!` name
/// it does not: the same named type can legitimately carry a *different*
/// runtime size on each operand, which is exactly the case decision `D-013`
/// records for broadcasting and which applies identically to the matmul
/// contraction. For a `usize` axis nothing is proven at all.
///
/// Before `SHP-004` no impl checked this --- `output_shape` took `lhs.0` and
/// `rhs.1` and never looked at `K`, so a disagreement produced a confidently
/// wrong output shape rather than an error.
#[inline]
fn checked_contraction(lhs_k: usize, rhs_k: usize) -> core::result::Result<(), ShapeError> {
    if lhs_k == rhs_k {
        Ok(())
    } else {
        Err(ShapeError::DimensionMismatch {
            operation: OperationKind::MatMul,
            axis: Axis::Named("k"),
            lhs: lhs_k,
            rhs: rhs_k,
            constraint: DimensionConstraint::Equal,
        })
    }
}

/// Check that a batch axis agrees between the two operands.
///
/// The same obligation the contraction check discharges, one axis over. A
/// batch axis typed the same on both sides is only proved equal when that type
/// is a `typenum`; a `dim!` name carries a runtime size and may differ. The
/// output takes its batch axes from the left operand, so a disagreement would
/// otherwise be resolved silently in the left's favour.
#[inline]
fn checked_batch(axis: usize, lhs: usize, rhs: usize) -> core::result::Result<(), ShapeError> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(ShapeError::DimensionMismatch {
            operation: OperationKind::MatMul,
            axis: Axis::Index(axis),
            lhs,
            rhs,
            constraint: DimensionConstraint::Equal,
        })
    }
}

/// Two contraction axes that may be multiplied against each other.
///
/// Matrix multiplication requires the left operand's trailing axis and the
/// right operand's leading one to have the same extent. Where both are sized by
/// their types that is a compile-time fact; where either is a runtime `usize`
/// it is not, and the check moves to a runtime comparison. This trait is the
/// line between the two, and it exists so the rank-2 rule can be written once
/// rather than once per combination of which axes happen to be runtime.
///
/// Before `SHP-007` the five rank-2 impls each required `K` to be the identical
/// type on both sides, so `(U2, usize)` could not be multiplied by
/// `(usize, U4)` at all --- a contraction nobody can settle statically had no
/// rule, rather than a runtime one.
///
/// The impls are disjoint for the same reason [`BroadcastDim`] is: no
/// downstream crate can implement `StaticOrNamedDim` for `usize`, since both
/// the trait and the type would be foreign to it.
///
/// [`BroadcastDim`]: crate::shapes::broadcast::BroadcastDim
#[diagnostic::on_unimplemented(
    message = "Cannot contract dimension `{Self}` with `{Rhs}`",
    label = "inner dimensions do not match",
    note = "Matrix multiplication requires the last dim of lhs and the second-to-last of rhs to be the same, or one of them to be a runtime `usize`"
)]
pub trait ContractsWith<Rhs: Dim>: Dim {}

/// Two raw static axes of the same type contract, and the compiler has already
/// agreed. Named axes use the semantic-name implementation below so that a
/// runtime extent can still contract against a concrete extent of the same
/// semantic axis.
impl<D: StaticDim> ContractsWith<D> for D {}

impl<const N: usize> ContractsWith<crate::shapes::dim::ConstDim<N>>
    for crate::shapes::dim::ConstDim<N>
{
}

/// A runtime axis contracts against a sized one if their values agree.
impl<D: StaticOrNamedDim> ContractsWith<D> for usize {}

/// The same, with the operands the other way round.
impl<D: StaticOrNamedDim> ContractsWith<usize> for D {}

/// A raw static extent and a named axis with that same concrete extent are
/// compatible. The semantic name constrains identity when both axes are
/// named, but it must not make an anonymous, statically equal contraction
/// fail merely because one operand carries metadata.
impl<Tag, N> ContractsWith<crate::shapes::dim::NamedDim<Tag, N>> for N
where
    Tag: crate::shapes::AxisTag,
    N: StaticDim,
{
}

impl<Tag, N> ContractsWith<N> for crate::shapes::dim::NamedDim<Tag, N>
where
    Tag: crate::shapes::AxisTag,
    N: StaticDim,
{
}

/// Named contraction axes with the same semantic identity are compatible when
/// their extents differ in static/runtime knowledge. The runtime shape check
/// remains authoritative for the unresolved extent.
impl<Tag, L, R> ContractsWith<crate::shapes::dim::NamedDim<Tag, R>>
    for crate::shapes::dim::NamedDim<Tag, L>
where
    Tag: crate::shapes::AxisTag,
    L: Dim,
    R: Dim,
{
}

/// Two runtime axes settle nothing in advance and are checked as values.
impl ContractsWith<usize> for usize {}

/// Structural matmul for every exact rank, with the final two axes treated as
/// the matrix portion and arbitrary prefixes broadcast through the canonical
/// shape broadcast rule.
impl<L, R, LP, RP, LM, LK, RK, RN, BOut, MatrixOut> MatMulShape<R> for L
where
    L: Shape + DynShape + crate::shapes::SplitLast2<Prefix = LP, Penultimate = LM, Last = LK>,
    R: Shape + DynShape + crate::shapes::SplitLast2<Prefix = RP, Penultimate = RK, Last = RN>,
    LP: Shape + DynShape + BroadcastShape<RP, Output = BOut>,
    RP: Shape + DynShape,
    LM: Dim,
    RK: Dim,
    RN: Dim,
    BOut: Shape
        + crate::shapes::StructuralConcatShape<DimCons<LM, DimCons<RN, Nil>>, Output = MatrixOut>,
    MatrixOut: Shape + DynShape,
    LK: ContractsWith<RK>,
{
    type Output = MatrixOut;

    fn output_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<ShapeBuf, ShapeError> {
        let lhs_dims: Vec<usize> = lhs.clone().into();
        let rhs_dims: Vec<usize> = rhs.clone().into();
        let lhs_k = *lhs_dims.last().ok_or(ShapeError::RankMismatch {
            operation: OperationKind::MatMul,
            expected: RankExpectation::AtLeast(2),
            actual: lhs_dims.len(),
        })?;
        let rhs_k = rhs_dims
            .get(
                rhs_dims
                    .len()
                    .checked_sub(2)
                    .ok_or(ShapeError::RankMismatch {
                        operation: OperationKind::MatMul,
                        expected: RankExpectation::AtLeast(2),
                        actual: rhs_dims.len(),
                    })?,
            )
            .copied()
            .ok_or(ShapeError::RankMismatch {
                operation: OperationKind::MatMul,
                expected: RankExpectation::AtLeast(2),
                actual: rhs_dims.len(),
            })?;
        checked_contraction(lhs_k, rhs_k)?;
        let lhs_batch = &lhs_dims[..lhs_dims.len() - 2];
        let rhs_batch = &rhs_dims[..rhs_dims.len() - 2];
        let mut out = crate::shapes::broadcast::broadcast_dim_slices(lhs_batch, rhs_batch)?;
        out.push(lhs_dims[lhs_dims.len() - 2]);
        out.push(rhs_dims[rhs_dims.len() - 1]);
        shape_buf_from_dims::<Self::Output>(OperationKind::MatMul, &out)
    }
}

// ============================================================================
// Fully dynamic: Dyn × Dyn - Dyn
// ============================================================================
impl MatMulShape<Dyn> for Dyn {
    /// The resulting shape after multiplying `Self` by `Rhs`.
    type Output = Dyn;

    /// Computes the runtime `ShapeBuf` of `Output` from the operand buffers.
    fn output_shape(lhs: &ShapeBuf, rhs: &ShapeBuf) -> core::result::Result<ShapeBuf, ShapeError> {
        const OP: OperationKind = OperationKind::MatMul;

        if lhs.len() < 2 {
            return Err(ShapeError::RankMismatch {
                operation: OP,
                expected: RankExpectation::AtLeast(2),
                actual: lhs.len(),
            });
        }
        if rhs.is_empty() {
            return Err(ShapeError::RankMismatch {
                operation: OP,
                expected: RankExpectation::AtLeast(1),
                actual: 0,
            });
        }

        // A rank-1 `rhs` is a vector: it contracts against `lhs`'s last axis
        // and contributes no output axis. This used to return the *empty*
        // shape, a scalar, for `[m, k] x [k]`, whose correct result is `[m]`.
        let vector_rhs = rhs.len() == 1;
        let rhs_k = if vector_rhs {
            rhs[0]
        } else {
            rhs[rhs.len() - 2]
        };
        checked_contraction(lhs[lhs.len() - 1], rhs_k)?;

        let mut out: alloc::vec::Vec<usize> = lhs[..lhs.len() - 1].to_vec();
        if !vector_rhs {
            out.push(rhs[rhs.len() - 1]);
        }
        Ok(ShapeBuf::from_slice(&out))
    }
}

// ============================================================================
// The matmul method on Tensor
// ============================================================================

use crate::tensor::backend::{FloatOps, NumericOps, TensorOps};

impl<
    S1: Shape,
    B: Backend + TensorOps<B> + NumericOps<B> + FloatOps<B>,
    K: crate::tensor::dtype::DType,
    G1: RequiresGrad,
> Tensor<S1, B, K, G1>
{
    /// Batched matrix multiplication over the trailing two dimensions,
    /// with the output shape checked at compile time via `MatMulShape`.
    pub fn matmul<S2, G2>(
        &self,
        rhs: &Tensor<S2, B, K, G2>,
    ) -> Result<Tensor<S1::Output, B, K, crate::tensor::grad::JoinedGrad<G1, G2>>>
    where
        S2: Shape + DynShape,
        G2: RequiresGrad,
        G1: crate::tensor::grad::GradJoin<G2>,
        S1: MatMulShape<S2>,
        B: Execute<Descriptor<op::MatMulExact>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::MatMulExact>>>::Output: Into<B::Storage<K>>,
    {
        let spec = crate::exec::MatMulSpec::new(&self.shape_buf_value(), &rhs.shape_buf_value())?;
        let rhs_grad = &rhs._grad;
        let output_shape = crate::shapes::ShapeValue::<S1::Output>::try_new(spec.output.clone())
            .map_err(crate::prelude::Error::Shape)?;
        let lhs = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let rhs = TensorHandle::from_storage::<B, K, Local>(&rhs.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                crate::exec::dispatch::execute_shaped::<op::MatMulExact, B, S1::Output>(
                    &context,
                    crate::exec::catalog::NoAttributes,
                    &[lhs, rhs],
                    &output_shape,
                )
            })?
            .into();
        let joined_grad =
            <G1 as crate::tensor::grad::GradJoin<G2>>::join_field(&self._grad, rhs_grad);
        Tensor::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            joined_grad,
        )
    }

    /// Computes vector dot product of 1D/matching tensors `self` and `rhs`, returning a scalar tensor.
    pub fn dot<S2: Shape>(
        &self,
        rhs: &Tensor<S2, B, K, G1>,
    ) -> Result<Tensor<(), B, K, JoinedGrad<G1, G1>>>
    where
        S1: crate::tensor::ops::ShapeEq<S2>,
        B: Execute<Descriptor<op::Mul>>
            + Execute<Descriptor<op::SumAll>>
            + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::Mul>>>::Output: Into<B::Storage<K>>,
        <B as Execute<Descriptor<op::SumAll>>>::Output: Into<B::Storage<K>>,
    {
        <S1 as crate::tensor::ops::ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        let mul = self.mul(rhs)?;
        mul.sum_all()
    }

    /// Computes outer product of vectors `self` and `rhs`.
    pub fn outer<S2: Shape + DynShape>(
        &self,
        rhs: &Tensor<S2, B, K, G1>,
    ) -> Result<Tensor<Dyn, B, K, JoinedGrad<G1, G1>>>
    where
        S1: DynShape,
        B: Execute<Descriptor<op::Mul>>,
        <B as Execute<Descriptor<op::Mul>>>::Output: Into<B::Storage<K>>,
    {
        let u1 = self.unsqueeze(1)?;
        let u2 = rhs.unsqueeze(0)?;
        u1.broadcast_mul(&u2)
    }

    /// Fused add-matmul: `beta * self + alpha * (mat1 x mat2)`.
    pub fn addmm<S2: Shape, S3: Shape>(
        &self,
        mat1: &Tensor<S2, B, K, G1>,
        mat2: &Tensor<S3, B, K, G1>,
        beta: f64,
        alpha: f64,
    ) -> Result<Self>
    where
        S1: DynShape,
        S2: Shape + DynShape,
        S3: Shape + DynShape,
        B: Execute<Descriptor<op::Addmm>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::Addmm>>>::Output: Into<B::Storage<K>>,
    {
        let output_shape = crate::shapes::ShapeValue::<S1>::try_new(self.shape_buf_value())
            .map_err(crate::prelude::Error::Shape)?;
        let bias = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let lhs = TensorHandle::from_storage::<B, K, Local>(&mat1.inner);
        let rhs = TensorHandle::from_storage::<B, K, Local>(&mat2.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                crate::exec::dispatch::execute_shaped::<op::Addmm, B, S1>(
                    &context,
                    AddmmAttributes { alpha, beta },
                    &[bias, lhs, rhs],
                    &output_shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Batched matrix multiplication for 3D tensors: `(B, M, K) x (B, K, N) -> (B, M, N)`.
    ///
    /// This compatibility spelling uses the same exact structural matmul
    /// descriptor as `matmul`; it does not call a parallel backend family
    /// implementation.
    pub fn bmm<S2: Shape>(&self, rhs: &Tensor<S2, B, K, G1>) -> Result<Tensor<Dyn, B, K, G1>>
    where
        S1: DynShape + MatMulShape<S2>,
        S2: DynShape,
        B: Execute<Descriptor<op::MatMulExact>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::MatMulExact>>>::Output: Into<B::Storage<K>>,
    {
        let spec = crate::exec::MatMulSpec::new(&self.shape_buf_value(), &rhs.shape_buf_value())?;
        let output_shape = spec.output.clone();
        let expected = crate::shapes::ShapeValue::<S1::Output>::try_new(output_shape.clone())
            .map_err(crate::prelude::Error::Shape)?;
        let lhs = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let rhs = TensorHandle::from_storage::<B, K, Local>(&rhs.inner);
        let context = ExecutionContext::from_scope(B::default());
        let inner = self
            .under_grad_mode(|| {
                crate::exec::dispatch::execute_shaped::<op::MatMulExact, B, S1::Output>(
                    &context,
                    crate::exec::catalog::NoAttributes,
                    &[lhs, rhs],
                    &expected,
                )
            })?
            .into();
        Tensor::<Dyn, B, K, G1>::from_shape_buf(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Scaled Dot-Product Attention: `softmax(q * k^T / scale) * v`.
    pub fn scaled_dot_product_attention<S2: Shape, S3: Shape, S4: Shape>(
        q: &Tensor<S1, B, K, G1>,
        k: &Tensor<S2, B, K, G1>,
        v: &Tensor<S3, B, K, G1>,
        mask: Option<&Tensor<S4, B, K, G1>>,
        scale: Option<f64>,
    ) -> Result<Tensor<Dyn, B, K, G1>>
    where
        S1: DynShape,
        S2: DynShape,
        S3: DynShape,
        S4: DynShape,
        B: Execute<Descriptor<op::ScaledDotProductAttention>> + crate::exec::Capabilities,
        <B as Execute<Descriptor<op::ScaledDotProductAttention>>>::Output: Into<B::Storage<K>>,
    {
        let output_shape = crate::shapes::ShapeValue::<S1>::try_new(q.shape_buf_value())
            .map_err(crate::prelude::Error::Shape)?;
        let q_handle = TensorHandle::from_storage::<B, K, Local>(&q.inner);
        let k_handle = TensorHandle::from_storage::<B, K, Local>(&k.inner);
        let v_handle = TensorHandle::from_storage::<B, K, Local>(&v.inner);
        let mask_handle = mask.map(|m| TensorHandle::from_storage::<B, K, Local>(&m.inner));
        let mut inputs = vec![q_handle, k_handle, v_handle];
        if let Some(mask_handle) = mask_handle {
            inputs.push(mask_handle);
        }
        let context = ExecutionContext::from_scope(B::default());
        let inner = q
            .under_grad_mode(|| {
                crate::exec::dispatch::execute_shaped::<op::ScaledDotProductAttention, B, S1>(
                    &context,
                    AttentionAttributes {
                        scale,
                        has_mask: mask.is_some(),
                    },
                    &inputs,
                    &output_shape,
                )
            })?
            .into();
        Tensor::<Dyn, B, K, G1>::from_shape_buf(
            inner,
            output_shape.shape_buf().clone(),
            q._dtype.clone(),
            q._device.clone(),
            q._grad.clone(),
        )
    }
}
