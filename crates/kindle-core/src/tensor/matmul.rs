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

// ============================================================================
// Fully static: (Const<M>, Const<K>) × (Const<K>, Const<N>) → (Const<M>, Const<N>)
//
// The K dimensions MUST match at the type level — the compiler enforces this.
// ============================================================================

impl<const M: usize, const K: usize, const N: usize>
    MatMulShape<(Const<K>, Const<N>)> for (Const<M>, Const<K>)
{
    type Output = (Const<M>, Const<N>);

    #[inline(always)]
    fn output_shape(_: &Self::Field, _: &<(Const<K>, Const<N>) as Shape>::Field) -> (Const<M>, Const<N>) {
        (Const, Const)
    }
}

// ============================================================================
// Partially dynamic: dynamic batch or dynamic inner dims
// ============================================================================

// (usize, Const<K>) × (Const<K>, Const<N>) → (usize, Const<N>)
// Dynamic rows, static inner/cols
impl<const K: usize, const N: usize>
    MatMulShape<(Const<K>, Const<N>)> for (usize, Const<K>)
{
    type Output = (usize, Const<N>);

    fn output_shape(
        lhs: &(usize, Const<K>),
        _: &(Const<K>, Const<N>),
    ) -> (usize, Const<N>) {
        (lhs.0, Const)
    }
}

// (Const<M>, Const<K>) × (Const<K>, usize) → (Const<M>, usize)
// Static rows/inner, dynamic cols
impl<const M: usize, const K: usize>
    MatMulShape<(Const<K>, usize)> for (Const<M>, Const<K>)
{
    type Output = (Const<M>, usize);

    fn output_shape(
        _: &(Const<M>, Const<K>),
        rhs: &(Const<K>, usize),
    ) -> (Const<M>, usize) {
        (Const, rhs.1)
    }
}

// (usize, Const<K>) × (Const<K>, usize) → (usize, usize)
impl<const K: usize>
    MatMulShape<(Const<K>, usize)> for (usize, Const<K>)
{
    type Output = (usize, usize);

    fn output_shape(
        lhs: &(usize, Const<K>),
        rhs: &(Const<K>, usize),
    ) -> (usize, usize) {
        (lhs.0, rhs.1)
    }
}

// (usize, usize) × (usize, usize) → (usize, usize)
// Both shapes fully dynamic — runtime check only
impl MatMulShape<(usize, usize)> for (usize, usize) {
    type Output = (usize, usize);

    fn output_shape(
        lhs: &(usize, usize),
        rhs: &(usize, usize),
    ) -> (usize, usize) {
        // Inner dimension check happens at the candle level
        (lhs.0, rhs.1)
    }
}

// ============================================================================
// Fully dynamic: Dyn × Dyn → Dyn
// No compile-time shape checking, defers everything to candle runtime.
// ============================================================================

impl MatMulShape<Dyn> for Dyn {
    type Output = Dyn;

    fn output_shape(
        _lhs: &<Dyn as Shape>::Field,
        _rhs: &<Dyn as Shape>::Field,
    ) -> <Dyn as Shape>::Field {
        // We can't know the output shape without actually doing the matmul.
        // Return empty — we'll extract the real shape from the candle result.
        alloc::vec![]
    }
}

// ============================================================================
// Batched matmul: 3D shapes
// ============================================================================

// (Const<B>, Const<M>, Const<K>) × (Const<B>, Const<K>, Const<N>) → (Const<B>, Const<M>, Const<N>)
impl<const B: usize, const M: usize, const K: usize, const N: usize>
    MatMulShape<(Const<B>, Const<K>, Const<N>)> for (Const<B>, Const<M>, Const<K>)
{
    type Output = (Const<B>, Const<M>, Const<N>);

    #[inline(always)]
    fn output_shape(
        _: &Self::Field,
        _: &<(Const<B>, Const<K>, Const<N>) as Shape>::Field,
    ) -> (Const<B>, Const<M>, Const<N>) {
        (Const, Const, Const)
    }
}

// (usize, Const<M>, Const<K>) × (usize, Const<K>, Const<N>) → (usize, Const<M>, Const<N>)
// Dynamic batch size, static matrix dims
impl<const M: usize, const K: usize, const N: usize>
    MatMulShape<(usize, Const<K>, Const<N>)> for (usize, Const<M>, Const<K>)
{
    type Output = (usize, Const<M>, Const<N>);

    fn output_shape(
        lhs: &(usize, Const<M>, Const<K>),
        _: &(usize, Const<K>, Const<N>),
    ) -> (usize, Const<M>, Const<N>) {
        (lhs.0, Const, Const)
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
        B: Backend<S2, RawTensor = <B as Backend<S1>>::RawTensor> + Backend<S1::Output, RawTensor = <B as Backend<S1>>::RawTensor>,
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
