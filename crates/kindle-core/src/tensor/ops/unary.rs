//! Unary tensor operations.
//!
//! This module provides element-wise unary operations (e.g., `abs`, `relu`, `exp`) that act
//! on a single tensor and return a new tensor with the exact same shape. It also includes
//! operations that interact with a scalar (e.g., `mul_scalar`, `add_scalar`).
use crate::prelude::{Backend, RequiresGrad, Result, Shape, Tensor};

macro_rules! impl_unary_op {
    (
        $(#[$meta:meta])*
        $method:ident, $backend_method:ident
    ) => {
        $(#[$meta])*
        pub fn $method(&self) -> Result<Self> {
            let inner = B::$backend_method::<K>(&self.inner)?;
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

impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad> Tensor<S, B, K, G> {
    impl_unary_op!(
        /// Computes the absolute value of each element in the tensor.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2, 2], DefaultBackend>::from_slice(&[-1.0, 2.0, -3.0, 4.0]).unwrap();
        /// let abs_t = t.abs().unwrap();
        /// ```
        abs, abs
    );

    impl_unary_op!(
        /// Applies the Rectified Linear Unit (ReLU) function element-wise.
        /// $\text{ReLU}(x) = \max(0, x)$
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[-1.0, 2.0]).unwrap();
        /// let relu_t = t.relu().unwrap(); // [0.0, 2.0]
        /// ```
        relu, relu
    );

    impl_unary_op!(
        /// Applies the Gaussian Error Linear Unit (GELU) function element-wise.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[-1.0, 2.0]).unwrap();
        /// let gelu_t = t.gelu().unwrap();
        /// ```
        gelu, gelu
    );

    impl_unary_op!(
        /// Applies the Step function element-wise (1.0 if x > 0.0 else 0.0).
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[-1.0, 2.0]).unwrap();
        /// let step_t = t.step().unwrap(); // [0.0, 1.0]
        /// ```
        step, step
    );

    impl_unary_op!(
        /// Applies the Mish function element-wise.
        /// $\text{Mish}(x) = x \cdot \text{tanh}(\text{softplus}(x))$
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[-1.0, 2.0]).unwrap();
        /// let mish_t = t.mish().unwrap();
        /// ```
        mish, mish
    );

    impl_unary_op!(
        /// Applies the Exponential Linear Unit (ELU) function element-wise with alpha=1.0.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[-1.0, 2.0]).unwrap();
        /// let elu_t = t.elu().unwrap();
        /// ```
        elu, elu
    );

    impl_unary_op!(
        /// Applies the Swish function element-wise (also known as SiLU).
        /// $\text{Swish}(x) = x \cdot \text{sigmoid}(x)$
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[-1.0, 2.0]).unwrap();
        /// let swish_t = t.swish().unwrap();
        /// ```
        swish, swish
    );

    /// Applies the Softmax function over the specified dimension.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![2, 2], DefaultBackend>::from_slice(&[1.0, 1.0, 0.0, 1.0]).unwrap();
    /// let sm = t.softmax(1).unwrap();
    /// ```
    #[inline]
    pub fn softmax(&self, dim: usize) -> Result<Tensor<S, B, K, G>> {
        let inner = B::softmax::<K>(&self.inner, dim)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
    impl_unary_op!(
        /// Negates the tensor element-wise.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[1.0, -2.0]).unwrap();
        /// let neg_t = t.neg().unwrap(); // [-1.0, 2.0]
        /// ```
        neg, neg
    );

    impl_unary_op!(
        /// Computes the square root of each element.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[4.0, 9.0]).unwrap();
        /// let sqrt_t = t.sqrt().unwrap(); // [2.0, 3.0]
        /// ```
        sqrt, sqrt
    );

    impl_unary_op!(
        /// Computes the exponential of each element.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![1], DefaultBackend>::from_slice(&[0.0]).unwrap();
        /// let exp_t = t.exp().unwrap(); // [1.0]
        /// ```
        exp, exp
    );

    impl_unary_op!(
        /// Computes the natural logarithm of each element.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![1], DefaultBackend>::from_slice(&[1.0]).unwrap();
        /// let log_t = t.log().unwrap(); // [0.0]
        /// ```
        log, log
    );

    impl_unary_op!(
        /// Computes the hyperbolic tangent of each element.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![1], DefaultBackend>::from_slice(&[0.0]).unwrap();
        /// let tanh_t = t.tanh().unwrap(); // [0.0]
        /// ```
        tanh, tanh
    );

    impl_unary_op!(
        /// Computes the sigmoid of each element.
        ///
        /// # Examples
        /// ```rust,ignore
        /// use kindle::prelude::*;
        /// let t = Tensor::<s![1], DefaultBackend>::from_slice(&[0.0]).unwrap();
        /// let sig_t = t.sigmoid().unwrap(); // [0.5]
        /// ```
        sigmoid, sigmoid
    );

    /// Multiplies the tensor by a scalar value element-wise.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[1.0, 2.0]).unwrap();
    /// let res = t.mul_scalar(3.0).unwrap(); // [3.0, 6.0]
    /// ```
    pub fn mul_scalar<Sc: Into<crate::tensor::backend::ScalarValue>>(
        &self,
        scalar: Sc,
    ) -> Result<Self> {
        let scalar_val = scalar.into();
        let inner = B::mul_scalar_float(&self.inner, scalar_val.to_f64())?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Adds a scalar value to the tensor element-wise.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[1.0, 2.0]).unwrap();
    /// let res = t.add_scalar(3.0).unwrap(); // [4.0, 5.0]
    /// ```
    pub fn add_scalar<Sc: Into<crate::tensor::backend::ScalarValue>>(
        &self,
        scalar: Sc,
    ) -> Result<Self> {
        let scalar_val = scalar.into();
        let inner = B::add_scalar_float(&self.inner, scalar_val.to_f64())?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

macro_rules! impl_std_scalar_ops {
    ($t:ty) => {
        impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
            core::ops::Mul<$t> for Tensor<S, B, K, G>
        {
            /// Scalar multiplication preserves shape/dtype/device.
            type Output = Tensor<S, B, K, G>;
            #[inline]
            /// `tensor * scalar`. Panics on error (`*` cannot return
            /// `Result`) — use `mul_scalar` directly to handle failure.
            fn mul(self, rhs: $t) -> Self::Output {
                self.mul_scalar(rhs)
                    .unwrap_or_else(|e| panic!("Tensor `*` (scalar) operator panicked: {e:?}"))
            }
        }
        impl<'a, S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
            core::ops::Mul<$t> for &'a Tensor<S, B, K, G>
        {
            /// Scalar multiplication preserves shape/dtype/device.
            type Output = Tensor<S, B, K, G>;
            #[inline]
            /// `tensor * scalar`. Panics on error (`*` cannot return
            /// `Result`) — use `mul_scalar` directly to handle failure.
            fn mul(self, rhs: $t) -> Self::Output {
                self.mul_scalar(rhs)
                    .unwrap_or_else(|e| panic!("Tensor `*` (scalar) operator panicked: {e:?}"))
            }
        }
        impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
            core::ops::Add<$t> for Tensor<S, B, K, G>
        {
            /// Scalar addition preserves shape/dtype/device.
            type Output = Tensor<S, B, K, G>;
            #[inline]
            /// `tensor + scalar`. Panics on error (`+` cannot return
            /// `Result`) — use `add_scalar` directly to handle failure.
            fn add(self, rhs: $t) -> Self::Output {
                self.add_scalar(rhs)
                    .unwrap_or_else(|e| panic!("Tensor `+` (scalar) operator panicked: {e:?}"))
            }
        }
        impl<'a, S: Shape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
            core::ops::Add<$t> for &'a Tensor<S, B, K, G>
        {
            /// Scalar addition preserves shape/dtype/device.
            type Output = Tensor<S, B, K, G>;
            #[inline]
            /// `tensor + scalar`. Panics on error (`+` cannot return
            /// `Result`) — use `add_scalar` directly to handle failure.
            fn add(self, rhs: $t) -> Self::Output {
                self.add_scalar(rhs)
                    .unwrap_or_else(|e| panic!("Tensor `+` (scalar) operator panicked: {e:?}"))
            }
        }
    };
}

impl_std_scalar_ops!(f32);
impl_std_scalar_ops!(f64);
impl_std_scalar_ops!(i32);
impl_std_scalar_ops!(i64);
