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
/// The bridge is the point, and this comment described it before the code did:
/// the shapes used to be built from `ConstDim<A>`, which is a *different* type
/// from the `s![..]` spelling and does not implement `ConcreteStaticExtent`.
/// The consequence was that nothing built by these constructors could reach
/// `ElementCount`, so `reshape` and `reshape_view` were unavailable on the most
/// ergonomic way to make a tensor. See issue #116.
///
/// Higher ranks follow the same pattern and are left out only because each one
/// is another impl; nothing here is rank-specific in principle.
macro_rules! impl_tensor_data {
    ($($elem:ty),* $(,)?) => {
        $(
            impl<const A: usize> TensorData for [$elem; A]
            where
                typenum::Const<A>: typenum::ToUInt,
                typenum::U<A>: Dim,
            {
                type Elem = $elem;
                type Shape = DimCons<typenum::U<A>, Nil>;
                fn into_row_major(self) -> Vec<$elem> {
                    self.to_vec()
                }
                fn dims() -> Vec<usize> { alloc::vec![A] }
            }

            impl<const A: usize, const B: usize> TensorData for [[$elem; B]; A]
            where
                typenum::Const<A>: typenum::ToUInt,
                typenum::Const<B>: typenum::ToUInt,
                typenum::U<A>: Dim,
                typenum::U<B>: Dim,
            {
                type Elem = $elem;
                type Shape = DimCons<typenum::U<A>, DimCons<typenum::U<B>, Nil>>;
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
