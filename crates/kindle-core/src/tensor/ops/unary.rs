//! Unary tensor operations.
//!
//! This module provides element-wise unary operations (e.g., `abs`, `relu`, `exp`) that act 
//! on a single tensor and return a new tensor with the exact same shape. It also includes 
//! operations that interact with a scalar (e.g., `mul_scalar`, `add_scalar`).
use crate::prelude::{Backend, RequiresGrad, Result, Shape, Tensor};


macro_rules! impl_unary_op {
    ($method:ident, $backend_method:ident) => {
        pub fn $method(&self) -> Result<Self> {
            let inner = B::$backend_method(&self.inner)?;
            Ok(Tensor::from_parts_unchecked(
                inner,
                self._shape.clone(),
                self._dtype.clone(),
                self._device.clone(),
                self._grad.clone(),
            ))
        }
    };
}

impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    impl_unary_op!(abs, abs);
    impl_unary_op!(relu, relu);
    impl_unary_op!(gelu, gelu);
    impl_unary_op!(swish, swish);

    #[inline]
    pub fn softmax(&self, dim: usize) -> Result<Tensor<S, B, G>> {
        let inner = B::softmax(&self.inner, dim)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
    impl_unary_op!(neg, neg);
    impl_unary_op!(sqrt, sqrt);
    impl_unary_op!(exp, exp);
    impl_unary_op!(log, log);
    impl_unary_op!(tanh, tanh);
    impl_unary_op!(sigmoid, sigmoid);

    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = B::mul_scalar(&self.inner, scalar)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = B::add_scalar(&self.inner, scalar)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

