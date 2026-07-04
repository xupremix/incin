//! Matrix multiplication with compile-time shape verification.
//!
//! The `MatMulShape` trait encodes the shape compatibility rules for matmul:
//! - `(M, K) × (K, N) → (M, N)` — inner dimensions must match
//! - Batched and fully-dynamic variants are also supported.
//!
//! **Static shapes**: The compiler rejects mismatched inner dims at compile time.
//! **Dynamic shapes**: Mismatches are caught at runtime by candle.

use crate::prelude::*;

// ============================================================================
// MatMulShape trait — compile-time shape compatibility for matmul
// ============================================================================

/// Trait that verifies two shapes are compatible for matrix multiplication
/// and determines the output shape.
///
/// Implement this for shape pairs that can be multiplied together.
/// The compiler will reject any `matmul` call where this trait is not implemented.
pub trait MatMulShape<Rhs: Shape>: Shape {
    type Output: Shape;

    /// Compute the output shape's Field from the inputs' fields.
    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<Rhs as Shape>::Field,
    ) -> <Self::Output as Shape>::Field;
}

pub trait StaticDim: Dim<Arg = ()> + Default {}
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
impl<const N: usize> StaticDim for Const<N> {}
impl<A: StaticDim, B: StaticDim> StaticDim for ProdDim<A, B> {}

// ============================================================================
// Fully static: (M, K) × (K, N) → (M, N)
// ============================================================================
impl<M: StaticDim, K: StaticDim, N: StaticDim> MatMulShape<(K, N)> for (M, K) {
    type Output = (M, N);

    #[inline(always)]
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(K, N) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (Default::default(), Default::default())
    }
}

// ============================================================================
// Partially dynamic: dynamic batch or dynamic inner dims
// ============================================================================

// (usize, K) × (K, N) → (usize, N)
impl<K: StaticDim, N: StaticDim> MatMulShape<(K, N)> for (usize, K) {
    type Output = (usize, N);

    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(K, N) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default())
    }
}

// (M, K) × (K, usize) → (M, usize)
impl<M: StaticDim, K: StaticDim> MatMulShape<(K, usize)> for (M, K) {
    type Output = (M, usize);

    fn output_shape(
        _: &<Self as Shape>::Field,
        rhs: &<(K, usize) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (Default::default(), rhs.1)
    }
}

// (usize, K) × (K, usize) → (usize, usize)
impl<K: StaticDim> MatMulShape<(K, usize)> for (usize, K) {
    type Output = (usize, usize);

    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<(K, usize) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, rhs.1)
    }
}

// (usize, usize) × (usize, usize) → (usize, usize)
impl MatMulShape<(usize, usize)> for (usize, usize) {
    type Output = (usize, usize);

    fn output_shape(
        lhs: &<Self as Shape>::Field,
        rhs: &<(usize, usize) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, rhs.1)
    }
}

// ============================================================================
// Fully dynamic: Dyn × Dyn → Dyn
// ============================================================================
impl MatMulShape<Dyn> for Dyn {
    type Output = Dyn;

    fn output_shape(
        _lhs: &<Dyn as Shape>::Field,
        _rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        alloc::vec![]
    }
}

// ============================================================================
// Batched matmul: 3D shapes
// ============================================================================

// (B, M, K) × (B, K, N) → (B, M, N)
impl<B: StaticDim, M: StaticDim, K: StaticDim, N: StaticDim> MatMulShape<(B, K, N)> for (B, M, K) {
    type Output = (B, M, N);

    #[inline(always)]
    fn output_shape(
        _: &<Self as Shape>::Field,
        _: &<(B, K, N) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (Default::default(), Default::default(), Default::default())
    }
}

// (usize, M, K) × (usize, K, N) → (usize, M, N)
impl<M: StaticDim, K: StaticDim, N: StaticDim> MatMulShape<(usize, K, N)> for (usize, M, K) {
    type Output = (usize, M, N);

    fn output_shape(
        lhs: &<Self as Shape>::Field,
        _: &<(usize, K, N) as Shape>::Field,
    ) -> <Self::Output as Shape>::Field {
        (lhs.0, Default::default(), Default::default())
    }
}

// ============================================================================
// The matmul method on Tensor
// ============================================================================

impl<S1, B: Backend<S1>, T, D, G> Tensor<S1, B, T, D, G>
where
    S1: Shape,
    T: DType,
    D: Device,
    G: RequiresGrad,
{
    pub fn matmul<S2>(&self, rhs: &Tensor<S2, B, T, D, G>) -> Result<Tensor<S1::Output, B, T, D, G>>
    where
        S2: Shape,
        S1: MatMulShape<S2>,
        B: Backend<S2, RawTensor = <B as Backend<S1>>::RawTensor>
            + Backend<S1::Output, RawTensor = <B as Backend<S1>>::RawTensor>,
    {
        let inner = <B as Backend<S1>>::matmul(&self.inner, &rhs.inner)?;
        let output_shape = S1::output_shape(&self._shape, &rhs._shape);
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            output_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}
