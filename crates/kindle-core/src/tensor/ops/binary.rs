//! Binary tensor operations with static and dynamic shape checking.
//!
//! This module provides strict element-wise binary operations (`add`, `sub`, `mul`, `div`)
//! that require exactly matching shapes at compile time via the `ShapeEq` trait.
//!
//! It also provides broadcasting variants (`broadcast_add`, etc.) and implements standard
//! `core::ops` traits (like `core::ops::Add`) which automatically leverage compile-time
//! broadcast shape resolution (`BroadcastShape`).
use crate::prelude::{Backend, RequiresGrad, Result, Shape, Tensor};
use crate::tensor::ops::*;

macro_rules! impl_binary_op {
    (
        $(#[$meta:meta])*
        $trait_name:ident, $method:ident, $backend_method:ident
    ) => {
        // Tensor op Tensor → Result<Tensor> (owned)
        impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad> Tensor<S, B, K, G>
{
            $(#[$meta])*
            pub fn $method<S2: Shape, G2: RequiresGrad>(
                &self,
                rhs: &Tensor<S2, B, K, G2>,
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

impl_binary_op!(
    /// Adds another tensor element-wise.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let a = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
    /// let b = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
    /// let c = a.add(&b).unwrap(); // Elements are 2.0
    /// ```
    Add, add, add
);

impl_binary_op!(
    /// Subtracts another tensor element-wise.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let a = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
    /// let b = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
    /// let c = a.sub(&b).unwrap(); // Elements are 0.0
    /// ```
    Sub, sub, sub
);

impl_binary_op!(
    /// Multiplies by another tensor element-wise.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let a = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
    /// let b = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
    /// let c = a.mul(&b).unwrap(); // Elements are 1.0
    /// ```
    Mul, mul, mul
);

impl_binary_op!(
    /// Divides by another tensor element-wise.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let a = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
    /// let b = Tensor::<s![2, 2], DefaultBackend>::ones(()).unwrap();
    /// let c = a.div(&b).unwrap(); // Elements are 1.0
    /// ```
    Div, div, div
);

macro_rules! impl_broadcast_binary_op {
    (
        $(#[$meta:meta])*
        $trait_name:ident, $method:ident, $backend_method:ident
    ) => {
        impl<S1: Shape + crate::shapes::DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad> Tensor<S1, B, K, G>
        {
            $(#[$meta])*
            #[inline]
            pub fn $method<S2>(&self, rhs: &Tensor<S2, B, K, G>) -> Result<Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, K, G>>
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

impl_broadcast_binary_op!(
    /// Adds two tensors, broadcasting their shapes if necessary according to NumPy semantics.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let a = Tensor::<s![2, 1], DefaultBackend>::ones(()).unwrap();
    /// let b = Tensor::<s![1, 2], DefaultBackend>::ones(()).unwrap();
    /// let c = a.broadcast_add(&b).unwrap(); // shape is [2, 2]
    /// ```
    BroadcastAdd, broadcast_add, add
);

impl_broadcast_binary_op!(
    /// Subtracts the right tensor from the left tensor, broadcasting shapes if necessary.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let a = Tensor::<s![2, 1], DefaultBackend>::ones(()).unwrap();
    /// let b = Tensor::<s![1, 2], DefaultBackend>::ones(()).unwrap();
    /// let c = a.broadcast_sub(&b).unwrap(); // shape is [2, 2]
    /// ```
    BroadcastSub, broadcast_sub, sub
);

impl_broadcast_binary_op!(
    /// Multiplies two tensors element-wise, broadcasting shapes if necessary.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let a = Tensor::<s![2, 1], DefaultBackend>::ones(()).unwrap();
    /// let b = Tensor::<s![1, 2], DefaultBackend>::ones(()).unwrap();
    /// let c = a.broadcast_mul(&b).unwrap(); // shape is [2, 2]
    /// ```
    BroadcastMul, broadcast_mul, mul
);

impl_broadcast_binary_op!(
    /// Divides the left tensor by the right tensor, broadcasting shapes if necessary.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let a = Tensor::<s![2, 1], DefaultBackend>::ones(()).unwrap();
    /// let b = Tensor::<s![1, 2], DefaultBackend>::ones(()).unwrap();
    /// let c = a.broadcast_div(&b).unwrap(); // shape is [2, 2]
    /// ```
    BroadcastDiv, broadcast_div, div
);

macro_rules! impl_std_ops {
    ($trait:ident, $method:ident, $backend_method:ident) => {
        // Tensor + Tensor
        impl<
            S1: Shape + crate::shapes::DynShape,
            S2: Shape + crate::shapes::DynShape,
            B: Backend,
            K: crate::tensor::dtype::DType,
            G: RequiresGrad,
        > core::ops::$trait<Tensor<S2, B, K, G>> for Tensor<S1, B, K, G>
        where
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
        {
            /// The output tensor type produced by this module's forward pass.
            type Output =
                Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, K, G>;
            #[inline]
            fn $method(self, rhs: Tensor<S2, B, K, G>) -> Self::Output {
                self.$backend_method(&rhs).unwrap_or_else(|e| {
                    panic!(
                        "Tensor `{}` operator panicked: {:?} (operands were not \
                         broadcast-compatible at runtime; call `.{}()` directly \
                         instead of the operator to handle this as a `Result`)",
                        stringify!($method),
                        e,
                        stringify!($backend_method)
                    )
                })
            }
        }

        // &Tensor + &Tensor
        impl<
            'a,
            'b,
            S1: Shape + crate::shapes::DynShape,
            S2: Shape + crate::shapes::DynShape,
            B: Backend,
            K: crate::tensor::dtype::DType,
            G: RequiresGrad,
        > core::ops::$trait<&'b Tensor<S2, B, K, G>> for &'a Tensor<S1, B, K, G>
        where
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
        {
            /// The output tensor type produced by this module's forward pass.
            type Output =
                Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, K, G>;
            #[inline]
            fn $method(self, rhs: &'b Tensor<S2, B, K, G>) -> Self::Output {
                self.$backend_method(rhs).unwrap_or_else(|e| {
                    panic!(
                        "Tensor `{}` operator panicked: {:?} (operands were not \
                         broadcast-compatible at runtime; call `.{}()` directly \
                         instead of the operator to handle this as a `Result`)",
                        stringify!($method),
                        e,
                        stringify!($backend_method)
                    )
                })
            }
        }

        // Tensor + &Tensor
        impl<
            'a,
            S1: Shape + crate::shapes::DynShape,
            S2: Shape + crate::shapes::DynShape,
            B: Backend,
            K: crate::tensor::dtype::DType,
            G: RequiresGrad,
        > core::ops::$trait<&'a Tensor<S2, B, K, G>> for Tensor<S1, B, K, G>
        where
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
        {
            /// The output tensor type produced by this module's forward pass.
            type Output =
                Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, K, G>;
            #[inline]
            fn $method(self, rhs: &'a Tensor<S2, B, K, G>) -> Self::Output {
                self.$backend_method(rhs).unwrap_or_else(|e| {
                    panic!(
                        "Tensor `{}` operator panicked: {:?} (operands were not \
                         broadcast-compatible at runtime; call `.{}()` directly \
                         instead of the operator to handle this as a `Result`)",
                        stringify!($method),
                        e,
                        stringify!($backend_method)
                    )
                })
            }
        }

        // &Tensor + Tensor
        impl<
            'a,
            S1: Shape + crate::shapes::DynShape,
            S2: Shape + crate::shapes::DynShape,
            B: Backend,
            K: crate::tensor::dtype::DType,
            G: RequiresGrad,
        > core::ops::$trait<Tensor<S2, B, K, G>> for &'a Tensor<S1, B, K, G>
        where
            S1: crate::shapes::broadcast::BroadcastShape<S2>,
            <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
        {
            /// The output tensor type produced by this module's forward pass.
            type Output =
                Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, K, G>;
            #[inline]
            fn $method(self, rhs: Tensor<S2, B, K, G>) -> Self::Output {
                self.$backend_method(&rhs).unwrap_or_else(|e| {
                    panic!(
                        "Tensor `{}` operator panicked: {:?} (operands were not \
                         broadcast-compatible at runtime; call `.{}()` directly \
                         instead of the operator to handle this as a `Result`)",
                        stringify!($method),
                        e,
                        stringify!($backend_method)
                    )
                })
            }
        }
    };
}

impl_std_ops!(Add, add, broadcast_add);
impl_std_ops!(Sub, sub, broadcast_sub);
impl_std_ops!(Mul, mul, broadcast_mul);
impl_std_ops!(Div, div, broadcast_div);
