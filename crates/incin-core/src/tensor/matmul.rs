//! Matrix multiplication with compile-time shape verification.
//!
//! The `MatMulShape` trait encodes the shape compatibility rules for matmul:
//! - `(M, K) × (K, N) → (M, N)` — inner dimensions must match
//! - Batched and fully-dynamic variants are also supported.
//!
//! **Static shapes**: The compiler rejects mismatched inner dims at compile time.
//! **Dynamic shapes**: Mismatches are caught at runtime by candle.

use crate::prelude::*;
use crate::shapes::error::OperationKind;
use crate::shapes::shape::field_from_dims;

// ============================================================================
// MatMulShape trait — compile-time shape compatibility for matmul
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
pub trait MatMulShape<Rhs: Shape>: Shape {
    /// The resulting shape after multiplying `Self` by `Rhs`.
    type Output: Shape;

    /// Compute the output shape's Field from the inputs' fields.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Rhs as Shape>::Field,
    ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError>;
}

/// Marker for a compile-time-fixed (`typenum`) dimension usable in
/// static matmul shape rules, as opposed to a runtime `usize`.
pub trait StaticDim: Dim + Default {}
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

impl<A: StaticDim, B: StaticDim> StaticDim for ProdDim<A, B> {}

/// Marker for a dimension with a type-level identity: typenum dims, ProdDim composites, and symbolic names, but not a runtime usize.
///
/// ContractsWith and the broadcast axis rule use this marker to keep their identity and runtime impls disjoint. Whole matmul shapes are otherwise bounded by Dim, so a runtime axis may appear at any supported position.
pub trait StaticOrNamedDim: Dim {}
impl<T: StaticDim> StaticOrNamedDim for T {}

/// Check that the contracted dimension agrees between the two operands.
///
/// For a `typenum` `K` the type system already proves this. For a `dim!` name
/// it does not: the same named type can legitimately carry a *different*
/// runtime size on each operand, which is exactly the case decision `D-013`
/// records for broadcasting and which applies identically to the matmul
/// contraction. For a `usize` axis nothing is proven at all.
///
/// Before `SHP-004` no impl checked this — `output_shape` took `lhs.0` and
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
/// `(usize, U4)` at all — a contraction nobody can settle statically had no
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

/// Two axes of the same type contract, and the compiler has already agreed.
impl<D: StaticOrNamedDim> ContractsWith<D> for D {}

/// A runtime axis contracts against a sized one if their values agree.
impl<D: StaticOrNamedDim> ContractsWith<D> for usize {}

/// The same, with the operands the other way round.
impl<D: StaticOrNamedDim> ContractsWith<usize> for D {}

/// Two runtime axes settle nothing in advance and are checked as values.
impl ContractsWith<usize> for usize {}

// ============================================================================
// Rank 2: (M, K) × (K, N) → (M, N), whichever axes are runtime.
// ============================================================================
impl<M: Dim, KL: Dim, KR: Dim, N: Dim> MatMulShape<(KR, N)> for (M, KL)
where
    KL: ContractsWith<KR>,
{
    /// The resulting shape after multiplying `Self` by `Rhs`.
    ///
    /// `M` and `N` pass through from the operands that named them, so a runtime
    /// axis stays runtime and a sized one stays sized. The contraction axis
    /// does not appear, which is why the two sides may disagree about how it is
    /// spelled as long as [`ContractsWith`] relates them.
    type Output = (M, N);

    #[inline(always)]
    /// Computes the runtime `Field` (dimension values) of `Output` from
    /// the operands' own runtime fields.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<(KR, N) as Shape>::Field,
    ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
        checked_contraction(lhs.1.size(), rhs.0.size())?;
        Ok((lhs.0, rhs.1))
    }
}

// ============================================================================
// Fully dynamic: Dyn × Dyn → Dyn
// ============================================================================
impl MatMulShape<Dyn> for Dyn {
    /// The resulting shape after multiplying `Self` by `Rhs`.
    type Output = Dyn;

    /// Computes the runtime `Field` (dimension values) of `Output` from
    /// the operands' own runtime fields.
    fn output_shape(
        lhs: &<Dyn as Shape>::Field,
        rhs: &<Dyn as Shape>::Field,
    ) -> core::result::Result<<Dyn as Shape>::Field, ShapeError> {
        const OP: OperationKind = OperationKind::MatMul;

        if lhs.len() == 4 && rhs.len() == 2 {
            // Preserved as-is: the "flattened batch" convention, where a
            // rank-4 `[N, C, H, W]` operand meets a `[C*H*W, out]` weight and
            // the caller folds the three trailing axes into the contraction.
            // The contracted extents therefore do not match axis-for-axis and
            // deliberately are not checked here.
            return Ok(alloc::vec![lhs[0], rhs[1]]);
        }

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
        // shape — a scalar — for `[m, k] x [k]`, whose correct result is `[m]`.
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
        Ok(out)
    }
}

// Handled by impl_batched_matmul macro

// ============================================================================
// Batched MatMul: (Batch..., M, K) x (Batch..., K, N) -> (Batch..., M, N)
// ============================================================================

macro_rules! impl_batched_matmul {
    // Both have same batch
    ( $( $batch:ident ),+ ) => {
        impl< $($batch: Dim,)* M: Dim, KL: Dim, KR: Dim, N: Dim> MatMulShape<( $($batch,)* KR, N)> for ( $($batch,)* M, KL)
        where
            ( $($batch,)* M, KL): DynShape,
            ( $($batch,)* KR, N): DynShape,
            KL: ContractsWith<KR>,
        {
            /// The resulting shape after multiplying `Self` by `Rhs`.
            type Output = ( $($batch,)* M, N);
            /// Computes the runtime `Field` (dimension values) of `Output` from
            /// the operands' own runtime fields: the batch dims and `M` are
            /// carried through from `lhs` unchanged, `K` (`lhs`'s last dim)
            /// is replaced by `rhs`'s last dim (`N`) — arity-agnostic, so
            /// this one body covers every `$batch` count this macro is
            /// invoked with.
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<( $($batch,)* KR, N) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                let mut dims: Vec<usize> = <Self as Shape>::dims(lhs).into();
                let rhs_dims: Vec<usize> = <( $($batch,)* KR, N) as Shape>::dims(rhs).into();
                let last = dims.len() - 1;
                // The batch axes and the contraction share a type on both
                // sides, which proves nothing when that type is a `dim!` name:
                // the same name may carry a different size on each operand.
                // Neither the contraction nor a disagreeing batch axis survives
                // into `Output`, so nothing downstream can catch it.
                for axis in 0..last - 1 {
                    checked_batch(axis, dims[axis], rhs_dims[axis])?;
                }
                checked_contraction(dims[last], rhs_dims[rhs_dims.len() - 2])?;
                dims[last] = rhs_dims[rhs_dims.len() - 1];
                field_from_dims::<Self::Output>(OperationKind::MatMul, &dims)
            }
        }
        // Lhs has batch
        impl< $($batch: Dim,)* M: Dim, KL: Dim, KR: Dim, N: Dim> MatMulShape<(KR, N)> for ( $($batch,)* M, KL)
        where
            ( $($batch,)* M, KL): DynShape,
            KL: ContractsWith<KR>,
        {
            /// The resulting shape after multiplying `Self` by `Rhs`.
            type Output = ( $($batch,)* M, N);
            /// Batch dims and `M` carried through from `lhs`; `K` (`lhs`'s
            /// last dim) replaced by `rhs`'s own `N` (`rhs` is always a
            /// plain `(K, N)` here, so direct field access is simplest).
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<(KR, N) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                let mut dims: Vec<usize> = <Self as Shape>::dims(lhs).into();
                let last = dims.len() - 1;
                checked_contraction(dims[last], rhs.0.size())?;
                dims[last] = rhs.1.size();
                field_from_dims::<Self::Output>(OperationKind::MatMul, &dims)
            }
        }
        // Rhs has batch
        impl< $($batch: Dim,)* M: Dim, KL: Dim, KR: Dim, N: Dim> MatMulShape<( $($batch,)* KR, N)> for (M, KL)
        where
            ( $($batch,)* KR, N): DynShape,
            KL: ContractsWith<KR>,
        {
            /// The resulting shape after multiplying `Self` by `Rhs`.
            type Output = ( $($batch,)* M, N);
            /// Batch dims and `N` carried through from `rhs`; `K`
            /// (`rhs`'s second-to-last dim) replaced by `lhs`'s own `M`
            /// (`lhs` is always a plain `(M, K)` here).
            fn output_shape(
                lhs: &<Self as Shape>::Field,
                rhs: &<( $($batch,)* KR, N) as Shape>::Field,
            ) -> core::result::Result<<Self::Output as Shape>::Field, ShapeError> {
                let mut dims: Vec<usize> = <( $($batch,)* KR, N) as Shape>::dims(rhs).into();
                let second_last = dims.len() - 2;
                checked_contraction(lhs.1.size(), dims[second_last])?;
                dims[second_last] = lhs.0.size();
                field_from_dims::<Self::Output>(OperationKind::MatMul, &dims)
            }
        }
    };
}

incin_macros::rank_sweep!(matmul_batch => impl_batched_matmul, min = 3);

// ============================================================================
// The matmul method on Tensor
// ============================================================================

impl<S1: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad> Tensor<S1, B, K, G> {
    /// Batched matrix multiplication over the trailing two dimensions,
    /// with the output shape checked at compile time via `MatMulShape`.
    pub fn matmul<S2>(&self, rhs: &Tensor<S2, B, K, G>) -> Result<Tensor<S1::Output, B, K, G>>
    where
        S2: Shape,
        S1: MatMulShape<S2>,
    {
        let inner = self.under_grad_mode(|| B::matmul::<K>(&self.inner, &rhs.inner))?;
        let output_shape = S1::output_shape(&self._shape, &rhs._shape)?;
        Tensor::from_parts(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Computes vector dot product of 1D/matching tensors `self` and `rhs`, returning a scalar tensor.
    pub fn dot<S2: Shape>(&self, rhs: &Tensor<S2, B, K, G>) -> Result<Tensor<(), B, K, G>>
    where
        S1: crate::tensor::ops::ShapeEq<S2>,
    {
        <S1 as crate::tensor::ops::ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        let mul = self.mul(rhs)?;
        mul.sum_all()
    }

    /// Computes outer product of vectors `self` and `rhs`.
    pub fn outer<S2: Shape + DynShape>(
        &self,
        rhs: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<Dyn, B, K, G>>
    where
        S1: DynShape,
    {
        let u1 = self.unsqueeze(1)?;
        let u2 = rhs.unsqueeze(0)?;
        u1.broadcast_mul(&u2)
    }

    /// Fused add-matmul: `beta * self + alpha * (mat1 x mat2)`.
    pub fn addmm<S2: Shape, S3: Shape>(
        &self,
        mat1: &Tensor<S2, B, K, G>,
        mat2: &Tensor<S3, B, K, G>,
        beta: f64,
        alpha: f64,
    ) -> Result<Self>
    where
        S1: DynShape,
    {
        let inner = self.under_grad_mode(|| {
            B::addmm::<K>(&self.inner, &mat1.inner, &mat2.inner, beta, alpha)
        })?;
        Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Batched matrix multiplication for 3D tensors: `(B, M, K) x (B, K, N) -> (B, M, N)`.
    pub fn bmm<S2: Shape>(&self, rhs: &Tensor<S2, B, K, G>) -> Result<Tensor<Dyn, B, K, G>>
    where
        S1: DynShape,
        S2: DynShape,
    {
        let inner = self.under_grad_mode(|| B::bmm::<K>(&self.inner, &rhs.inner))?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Scaled Dot-Product Attention: `softmax(q * k^T / scale) * v`.
    pub fn scaled_dot_product_attention<S2: Shape, S3: Shape, S4: Shape>(
        q: &Tensor<S1, B, K, G>,
        k: &Tensor<S2, B, K, G>,
        v: &Tensor<S3, B, K, G>,
        mask: Option<&Tensor<S4, B, K, G>>,
        scale: Option<f64>,
    ) -> Result<Tensor<Dyn, B, K, G>>
    where
        S1: DynShape,
        S2: DynShape,
        S3: DynShape,
        S4: DynShape,
    {
        let mask_inner = mask.map(|m| &m.inner);
        let inner = q.under_grad_mode(|| {
            B::scaled_dot_product_attention::<K>(&q.inner, &k.inner, &v.inner, mask_inner, scale)
        })?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            q._dtype.clone(),
            q._device.clone(),
            q._grad.clone(),
        )
    }
}
