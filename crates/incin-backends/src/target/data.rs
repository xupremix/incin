//! Ordinary Rust data entering a tensor.

use super::*;

/// Ordinary Rust data that carries its own shape and element type.
///
/// Nested arrays already encode both, which is why this exists instead of a
/// literal macro: `[[1.0f32, 2.0], [3.0, 4.0]]` is a `s![2, 2]` tensor of
/// `f32` with no new syntax and no dtype inference rules to learn.
pub trait TensorData {
    /// The element type. Taken from the data, never from the target.
    type Elem: PlainDType<Elem = Self::Elem> + BuiltinDType + bytemuck::Pod;
    /// The static shape the nesting describes.
    type Shape: Shape + DynShape;
    /// Flattens into row-major order.
    fn into_row_major(self) -> Vec<Self::Elem>;
    /// Returns the statically known dimensions represented by the data.
    fn dims() -> Vec<usize>;
}

/// Rank-1 and rank-2 Rust arrays. `typenum`'s `Const<N>`/`ToUInt` bridge turns
/// each array length into the type-level dimension a static shape needs, so
/// `[[1.0f32, 2.0], [3.0, 4.0]]` arrives as `s![2, 2]` without the caller
/// writing a shape at all.
///
/// Higher ranks follow the same pattern and are left out only because each one
/// is another impl; nothing here is rank-specific in principle.
macro_rules! impl_tensor_data {
    ($($elem:ty),* $(,)?) => {
        $(
            impl<const A: usize> TensorData for [$elem; A]
            where
                ConstDim<A>: Dim,
            {
                type Elem = $elem;
                type Shape = DimCons<ConstDim<A>, Nil>;
                fn into_row_major(self) -> Vec<$elem> {
                    self.to_vec()
                }
                fn dims() -> Vec<usize> { alloc::vec![A] }
            }

            impl<const A: usize, const B: usize> TensorData for [[$elem; B]; A]
            where
                ConstDim<A>: Dim,
                ConstDim<B>: Dim,
            {
                type Elem = $elem;
                type Shape = DimCons<ConstDim<A>, DimCons<ConstDim<B>, Nil>>;
                fn into_row_major(self) -> Vec<$elem> {
                    // Row-major: the outer array indexes the slowest axis, so
                    // flattening in declaration order is already correct.
                    self.iter().flat_map(|row| row.iter().copied()).collect()
                }
                fn dims() -> Vec<usize> { alloc::vec![A, B] }
            }
        )*
    };
}

impl_tensor_data!(f32, f64, u8, u32, i64);
