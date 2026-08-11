//! Unary tensor operations.
//!
//! This module provides element-wise unary operations (e.g., `abs`, `relu`, `exp`) that act
//! on a single tensor and return a new tensor with the exact same shape. It also includes
//! operations that interact with a scalar (e.g., `mul_scalar`, `add_scalar`).
use crate::exec::catalog::{op, Descriptor};
use crate::prelude::{Backend, RequiresGrad, Result, Shape, Tensor};
use crate::tensor::backend::{FloatOps, NumericOps, TensorOps};

macro_rules! impl_unary_op {
    (
        $(#[$meta:meta])*
        $method:ident, $operation:ident
    ) => {
        $(#[$meta])*
        pub fn $method(&self) -> Result<Self>
        where
            B: Execute<Descriptor<op::$operation>>,
            <B as Execute<Descriptor<op::$operation>>>::Output: Into<B::Storage<K>>,
        {
            execute_unary_descriptor::<op::$operation, S, B, K, G>(self)
        }
    };
}

impl<
    S: Shape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S, B, K, G>
{
    impl_unary_op!(
        /// Elementwise tangent.
        tan, Tan
    );
    impl_unary_op!(
        /// Elementwise arcsine.
        asin, Asin
    );
    impl_unary_op!(
        /// Elementwise arccosine.
        acos, Acos
    );
    impl_unary_op!(
        /// Elementwise arctangent.
        atan, Atan
    );
    impl_unary_op!(
        /// Elementwise hyperbolic sine.
        sinh, Sinh
    );
    impl_unary_op!(
        /// Elementwise hyperbolic cosine.
        cosh, Cosh
    );
    impl_unary_op!(
        /// Elementwise inverse hyperbolic sine.
        asinh, Asinh
    );
    impl_unary_op!(
        /// Elementwise inverse hyperbolic cosine.
        acosh, Acosh
    );
    impl_unary_op!(
        /// Elementwise inverse hyperbolic tangent.
        atanh, Atanh
    );
    impl_unary_op!(
        /// Elementwise error function.
        erf, Erf
    );
    impl_unary_op!(
        /// Elementwise reciprocal square root $1/\sqrt{x}$.
        rsqrt, Rsqrt
    );
    impl_unary_op!(
        /// Elementwise integer truncation toward zero.
        trunc, Trunc
    );
    impl_unary_op!(
        /// Elementwise fractional part.
        frac, Frac
    );

    /// Computes the absolute value of each element in the tensor.
    pub fn abs(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Abs>>,
        <B as Execute<Descriptor<op::Abs>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Abs, S, B, K, G>(self)
    }

    /// Applies the Rectified Linear Unit (ReLU) function element-wise.
    pub fn relu(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Relu>>,
        <B as Execute<Descriptor<op::Relu>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Relu, S, B, K, G>(self)
    }

    /// Applies the Gaussian Error Linear Unit (GELU) function element-wise.
    pub fn gelu(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Gelu>>,
        <B as Execute<Descriptor<op::Gelu>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Gelu, S, B, K, G>(self)
    }

    /// Applies the Step function element-wise.
    pub fn step(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Step>>,
        <B as Execute<Descriptor<op::Step>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Step, S, B, K, G>(self)
    }

    /// Applies the Mish function element-wise.
    pub fn mish(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Mish>>,
        <B as Execute<Descriptor<op::Mish>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Mish, S, B, K, G>(self)
    }

    /// Applies the Exponential Linear Unit (ELU) function element-wise with alpha=1.0.
    pub fn elu(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Elu>>,
        <B as Execute<Descriptor<op::Elu>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Elu, S, B, K, G>(self)
    }

    /// Applies the Swish function element-wise (also known as SiLU).
    pub fn swish(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Swish>>,
        <B as Execute<Descriptor<op::Swish>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Swish, S, B, K, G>(self)
    }

    /// Applies the Softmax function over the specified dimension.
    #[inline]
    pub fn softmax(&self, dim: usize) -> Result<Tensor<S, B, K, G>>
    where
        B: Execute<Descriptor<op::Softmax>>,
        <B as Execute<Descriptor<op::Softmax>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor_with_attributes::<op::Softmax, S, B, K, G>(
            self,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Negates the tensor element-wise.
    pub fn neg(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Neg>>,
        <B as Execute<Descriptor<op::Neg>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Neg, S, B, K, G>(self)
    }

    /// Computes the square root of each element.
    pub fn sqrt(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Sqrt>>,
        <B as Execute<Descriptor<op::Sqrt>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Sqrt, S, B, K, G>(self)
    }

    /// Computes the exponential of each element.
    pub fn exp(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Exp>>,
        <B as Execute<Descriptor<op::Exp>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Exp, S, B, K, G>(self)
    }

    /// Raises tensor elements to power `exponent`.
    #[inline]
    pub fn powf(&self, exponent: f64) -> Result<Self>
    where
        B: Execute<Descriptor<op::Powf>>,
        <B as Execute<Descriptor<op::Powf>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor_with_attributes::<op::Powf, S, B, K, G>(
            self,
            crate::exec::catalog::ScalarAttributes { value: exponent },
        )
    }

    /// Clamps tensor elements to range `[min, max]`.
    #[inline]
    pub fn clamp(&self, min: f64, max: f64) -> Result<Self>
    where
        B: Execute<Descriptor<op::Clamp>>,
        <B as Execute<Descriptor<op::Clamp>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor_with_attributes::<op::Clamp, S, B, K, G>(
            self,
            crate::exec::catalog::ClampAttributes { min, max },
        )
    }

    impl_unary_op!(
        /// Elementwise sign function (-1.0, 0.0, or +1.0).
        sign, Sign
    );

    impl_unary_op!(
        /// Computes the floor of each element.
        floor, Floor
    );

    impl_unary_op!(
        /// Computes the ceiling of each element.
        ceil, Ceil
    );

    impl_unary_op!(
        /// Rounds each element to nearest integer.
        round, Round
    );

    impl_unary_op!(
        /// Computes base-2 logarithm elementwise.
        log2, Log2
    );

    impl_unary_op!(
        /// Computes base-10 logarithm elementwise.
        log10, Log10
    );

    impl_unary_op!(
        /// Computes sine elementwise.
        sin, Sin
    );

    impl_unary_op!(
        /// Computes cosine elementwise.
        cos, Cos
    );

    /// Computes the natural logarithm of each element.
    pub fn log(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Log>>,
        <B as Execute<Descriptor<op::Log>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Log, S, B, K, G>(self)
    }

    /// Computes the hyperbolic tangent of each element.
    pub fn tanh(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Tanh>>,
        <B as Execute<Descriptor<op::Tanh>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Tanh, S, B, K, G>(self)
    }

    /// Computes the sigmoid of each element.
    pub fn sigmoid(&self) -> Result<Self>
    where
        B: Execute<Descriptor<op::Sigmoid>>,
        <B as Execute<Descriptor<op::Sigmoid>>>::Output: Into<B::Storage<K>>,
    {
        execute_unary_descriptor::<op::Sigmoid, S, B, K, G>(self)
    }

    /// Multiplies the tensor by a scalar value element-wise.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[1.0, 2.0], ()).unwrap();
    /// let res = t.mul_scalar(3.0).unwrap(); // [3.0, 6.0]
    /// ```
    pub fn mul_scalar<Sc: Into<crate::tensor::backend::ScalarValue>>(
        &self,
        scalar: Sc,
    ) -> Result<Self>
    where
        B: FloatOps<B> + NumericOps<B> + TensorOps<B>,
    {
        let scalar_val = scalar.into();
        let inner =
            self.under_grad_mode(|| B::mul_scalar_float(&self.inner, scalar_val.to_f64()))?;
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Adds a scalar value to the tensor element-wise.
    ///
    /// # Examples
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # type DefaultBackend = incin_core::test_utils::DummyBackend<incin_core::prelude::Cpu>;
    /// use incin::prelude::*;
    /// let t = Tensor::<s![2], DefaultBackend>::from_slice(&[1.0, 2.0], ()).unwrap();
    /// let res = t.add_scalar(3.0).unwrap(); // [4.0, 5.0]
    /// ```
    pub fn add_scalar<Sc: Into<crate::tensor::backend::ScalarValue>>(
        &self,
        scalar: Sc,
    ) -> Result<Self>
    where
        B: FloatOps<B> + NumericOps<B> + TensorOps<B>,
    {
        let scalar_val = scalar.into();
        let inner =
            self.under_grad_mode(|| B::add_scalar_float(&self.inner, scalar_val.to_f64()))?;
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }
}

use crate::dist::placement::Local;
use crate::exec::capability::Capabilities;
use crate::exec::catalog::{AttributeContract, CanonicalOperation, NoAttributes};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::ShapeValue;
use crate::tensor::backend::Execute;

pub(crate) fn execute_unary_descriptor<
    O,
    S: Shape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
>(
    tensor: &Tensor<S, B, K, G>,
) -> Result<Tensor<S, B, K, G>>
where
    O: CanonicalOperation + crate::exec::catalog::Operation<Attributes = NoAttributes>,
    B: Execute<Descriptor<O>>,
    <B as Execute<Descriptor<O>>>::Output: Into<B::Storage<K>>,
{
    let handle = TensorHandle::from_storage::<B, K, Local>(&tensor.inner);
    let shape_val = tensor._shape.clone();
    let context = ExecutionContext::from_scope(B::default());
    let storage =
        dispatch::execute_shaped::<O, B, S>(&context, NoAttributes, &[handle], &shape_val)
            .map_err(crate::prelude::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        tensor._shape.clone(),
        tensor._dtype.clone(),
        tensor._device.clone(),
        tensor._grad.clone(),
    )
}

pub(crate) fn execute_unary_descriptor_with_attributes<
    O,
    S: Shape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
>(
    tensor: &Tensor<S, B, K, G>,
    attributes: O::Attributes,
) -> Result<Tensor<S, B, K, G>>
where
    O: CanonicalOperation,
    O::Attributes: AttributeContract,
    B: Execute<Descriptor<O>>,
    <B as Execute<Descriptor<O>>>::Output: Into<B::Storage<K>>,
{
    let handle = TensorHandle::from_storage::<B, K, Local>(&tensor.inner);
    let shape_val = tensor._shape.clone();
    let context = ExecutionContext::from_scope(B::default());
    let storage = dispatch::execute_shaped::<O, B, S>(&context, attributes, &[handle], &shape_val)
        .map_err(crate::prelude::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        tensor._shape.clone(),
        tensor._dtype.clone(),
        tensor._device.clone(),
        tensor._grad.clone(),
    )
}

pub(crate) fn execute_logical_unary_descriptor<O, S: Shape, B: Backend, G: RequiresGrad>(
    tensor: &Tensor<S, B, bool, G>,
) -> Result<Tensor<S, B, bool, crate::tensor::grad::NoGrad>>
where
    O: CanonicalOperation + crate::exec::catalog::Operation<Attributes = NoAttributes>,
    B: Execute<Descriptor<O>>,
    <B as Execute<Descriptor<O>>>::Output: Into<B::Storage<bool>>,
{
    let handle = TensorHandle::from_storage::<B, bool, Local>(&tensor.inner);
    let shape_val = tensor._shape.clone();
    let context =
        ExecutionContext::from_scope(B::default()).with_grad_mode(crate::exec::GradMode::Disabled);
    let storage =
        dispatch::execute_shaped::<O, B, S>(&context, NoAttributes, &[handle], &shape_val)
            .map_err(crate::prelude::Error::from)?;
    Tensor::from_shape_value(
        storage.into(),
        tensor._shape.clone(),
        Default::default(),
        tensor._device.clone(),
        Default::default(),
    )
}

impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, bool, G> {
    /// Logical NOT element-wise.
    pub fn logical_not(&self) -> Result<Tensor<S, B, bool, crate::tensor::grad::NoGrad>>
    where
        B: Execute<Descriptor<op::LogicalNot>>,
        <B as Execute<Descriptor<op::LogicalNot>>>::Output: Into<B::Storage<bool>>,
    {
        execute_logical_unary_descriptor::<op::LogicalNot, S, B, G>(self)
    }
}

macro_rules! impl_std_scalar_ops {
    ($t:ty) => {
        impl<
            S: Shape,
            B: Backend + FloatOps<B> + NumericOps<B> + TensorOps<B>,
            K: crate::tensor::dtype::DType,
            G: RequiresGrad,
        > core::ops::Mul<$t> for Tensor<S, B, K, G>
        {
            /// Scalar multiplication preserves shape/dtype/device.
            type Output = crate::prelude::Result<Tensor<S, B, K, G>>;
            #[inline]
            /// `tensor * scalar`, returning the same typed failure contract as
            /// [`Tensor::mul_scalar`].
            fn mul(self, rhs: $t) -> Self::Output {
                self.mul_scalar(rhs)
            }
        }
        impl<
            'a,
            S: Shape,
            B: Backend + FloatOps<B> + NumericOps<B> + TensorOps<B>,
            K: crate::tensor::dtype::DType,
            G: RequiresGrad,
        > core::ops::Mul<$t> for &'a Tensor<S, B, K, G>
        {
            /// Scalar multiplication preserves shape/dtype/device.
            type Output = crate::prelude::Result<Tensor<S, B, K, G>>;
            #[inline]
            /// `tensor * scalar`, returning the same typed failure contract as
            /// [`Tensor::mul_scalar`].
            fn mul(self, rhs: $t) -> Self::Output {
                self.mul_scalar(rhs)
            }
        }
        impl<
            S: Shape,
            B: Backend + FloatOps<B> + NumericOps<B> + TensorOps<B>,
            K: crate::tensor::dtype::DType,
            G: RequiresGrad,
        > core::ops::Add<$t> for Tensor<S, B, K, G>
        {
            /// Scalar addition preserves shape/dtype/device.
            type Output = crate::prelude::Result<Tensor<S, B, K, G>>;
            #[inline]
            /// `tensor + scalar`, returning the same typed failure contract as
            /// [`Tensor::add_scalar`].
            fn add(self, rhs: $t) -> Self::Output {
                self.add_scalar(rhs)
            }
        }
        impl<
            'a,
            S: Shape,
            B: Backend + FloatOps<B> + NumericOps<B> + TensorOps<B>,
            K: crate::tensor::dtype::DType,
            G: RequiresGrad,
        > core::ops::Add<$t> for &'a Tensor<S, B, K, G>
        {
            /// Scalar addition preserves shape/dtype/device.
            type Output = crate::prelude::Result<Tensor<S, B, K, G>>;
            #[inline]
            /// `tensor + scalar`, returning the same typed failure contract as
            /// [`Tensor::add_scalar`].
            fn add(self, rhs: $t) -> Self::Output {
                self.add_scalar(rhs)
            }
        }
    };
}

impl_std_scalar_ops!(f32);
impl_std_scalar_ops!(f64);
impl_std_scalar_ops!(i32);
impl_std_scalar_ops!(i64);
