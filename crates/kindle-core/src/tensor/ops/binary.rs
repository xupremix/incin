//! Binary tensor operations with static and dynamic shape checking.
//!
//! This module provides strict element-wise binary operations (`add`, `sub`, `mul`, `div`)
//! that require exactly matching shapes at compile time via the `ShapeEq` trait. 
//! 
//! It also provides broadcasting variants (`broadcast_add`, etc.) and implements standard 
//! `core::ops` traits (like `std::ops::Add`) which automatically leverage compile-time 
//! broadcast shape resolution (`BroadcastShape`).
use crate::tensor::ops::*;
use crate::prelude::{Backend, RequiresGrad, Result, Shape, Tensor};


macro_rules! impl_binary_op {
    ($trait_name:ident, $method:ident, $backend_method:ident) => {
        // Tensor op Tensor → Result<Tensor> (owned)
        impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
            pub fn $method<S2: Shape, G2: RequiresGrad>(
                &self,
                rhs: &Tensor<S2, B, G2>,
            ) -> Result<Self>
            where
                S: ShapeEq<S2>,
            {
                let _ = <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
                let inner = B::$backend_method(&self.inner, &rhs.inner)?;
                Ok(Tensor::from_parts_unchecked(
                    inner,
                    self._shape.clone(),
                    self._dtype.clone(),
                    self._device.clone(),
                    self._grad.clone(),
                ))
            }
        }
    };
}

impl_binary_op!(Add, add, add);
impl_binary_op!(Sub, sub, sub);
impl_binary_op!(Mul, mul, mul);
impl_binary_op!(Div, div, div);

macro_rules! impl_broadcast_binary_op {
    ($trait_name:ident, $method:ident, $backend_method:ident) => {
        impl<S1: Shape + crate::shapes::DynShape, B: Backend, G: RequiresGrad> Tensor<S1, B, G> {
            #[inline]
            pub fn $method<S2>(&self, rhs: &Tensor<S2, B, G>) -> Result<Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, G>>
            where
                S2: Shape + crate::shapes::DynShape,
                S1: crate::shapes::broadcast::BroadcastShape<S2>,
                <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
            {
                let b_shape = <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::output_shape(self.shape_field(), rhs.shape_field());

                let inner = B::$backend_method(&self.inner, &rhs.inner)?;
                Ok(Tensor::from_parts_unchecked(
                    inner,
                    b_shape,
                    self._dtype.clone(),
                    self._device.clone(),
                    self._grad.clone(),
                ))
            }
        }
    };
}

impl_broadcast_binary_op!(BroadcastAdd, broadcast_add, add);
impl_broadcast_binary_op!(BroadcastSub, broadcast_sub, sub);
impl_broadcast_binary_op!(BroadcastMul, broadcast_mul, mul);
impl_broadcast_binary_op!(BroadcastDiv, broadcast_div, div);


macro_rules! impl_std_ops {
    ($trait:ident, $method:ident, $backend_method:ident) => {
        // Tensor + Tensor
        impl<S1: Shape + crate::shapes::DynShape, S2: Shape + crate::shapes::DynShape, B: Backend, G: RequiresGrad> core::ops::$trait<Tensor<S2, B, G>> for Tensor<S1, B, G>
        where
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
        {
            type Output = Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, G>;
            #[inline]
            fn $method(self, rhs: Tensor<S2, B, G>) -> Self::Output {
                self.$backend_method(&rhs).unwrap()
            }
        }

        // &Tensor + &Tensor
        impl<'a, 'b, S1: Shape + crate::shapes::DynShape, S2: Shape + crate::shapes::DynShape, B: Backend, G: RequiresGrad> core::ops::$trait<&'b Tensor<S2, B, G>> for &'a Tensor<S1, B, G>
        where
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
        {
            type Output = Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, G>;
            #[inline]
            fn $method(self, rhs: &'b Tensor<S2, B, G>) -> Self::Output {
                self.$backend_method(rhs).unwrap()
            }
        }

        // Tensor + &Tensor
        impl<'a, S1: Shape + crate::shapes::DynShape, S2: Shape + crate::shapes::DynShape, B: Backend, G: RequiresGrad> core::ops::$trait<&'a Tensor<S2, B, G>> for Tensor<S1, B, G>
        where
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
        {
            type Output = Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, G>;
            #[inline]
            fn $method(self, rhs: &'a Tensor<S2, B, G>) -> Self::Output {
                self.$backend_method(rhs).unwrap()
            }
        }

        // &Tensor + Tensor
        impl<'a, S1: Shape + crate::shapes::DynShape, S2: Shape + crate::shapes::DynShape, B: Backend, G: RequiresGrad> core::ops::$trait<Tensor<S2, B, G>> for &'a Tensor<S1, B, G>
        where
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
        {
            type Output = Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, G>;
            #[inline]
            fn $method(self, rhs: Tensor<S2, B, G>) -> Self::Output {
                self.$backend_method(&rhs).unwrap()
            }
        }
    };
}

impl_std_ops!(Add, add, broadcast_add);
impl_std_ops!(Sub, sub, broadcast_sub);
impl_std_ops!(Mul, mul, broadcast_mul);
impl_std_ops!(Div, div, broadcast_div);
