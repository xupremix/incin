//! Element-wise tensor operations with compile-time shape checking.
//!
//! Operations require matching Shape, DType, Device, and RequiresGrad.
//! This ensures at compile time that you can't accidentally add tensors
//! of different shapes, dtypes, or on different devices.

use crate::{
    candle,
    prelude::{DType, Device, RequiresGrad, Result, Shape, Tensor},
};

// ============================================================================
// Element-wise binary ops via std::ops
//
// These require EXACT type match on all four axes. The compiler
// will reject `Tensor<(Const<2>,), f32, Cpu, Grad> + Tensor<(Const<3>,), f32, Cpu, Grad>`
// at compile time for static shapes.
// ============================================================================

macro_rules! impl_binary_op {
    ($trait_name:ident, $method:ident, $candle_method:ident) => {
        // Tensor op Tensor → Result<Tensor> (owned)
        impl<S: Shape, T: DType, D: Device, G: RequiresGrad> Tensor<S, T, D, G> {
            pub fn $method(&self, rhs: &Self) -> Result<Self> {
                let inner = self.inner.$candle_method(&rhs.inner)?;
                Ok(Self::from_parts(
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

// ============================================================================
// Scalar operations
// ============================================================================

impl<S: Shape, T: DType, D: Device, G: RequiresGrad> Tensor<S, T, D, G> {
    /// Multiply all elements by a scalar.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = (&self.inner * scalar)?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Add a scalar to all elements.
    pub fn add_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = (&self.inner + scalar)?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Negate all elements.
    pub fn neg(&self) -> Result<Self> {
        let inner = self.inner.neg()?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Self> {
        let inner = self.inner.abs()?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Self> {
        let inner = self.inner.sqrt()?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Element-wise exponential (e^x).
    pub fn exp(&self) -> Result<Self> {
        let inner = self.inner.exp()?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Element-wise natural log.
    pub fn log(&self) -> Result<Self> {
        let inner = self.inner.log()?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Element-wise ReLU (max(0, x)).
    pub fn relu(&self) -> Result<Self> {
        let inner = self.inner.relu()?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Element-wise tanh.
    pub fn tanh(&self) -> Result<Self> {
        let inner = self.inner.tanh()?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Element-wise sigmoid (1 / (1 + e^(-x))).
    pub fn sigmoid(&self) -> Result<Self> {
        // sigmoid = 1 / (1 + exp(-x))
        let inner = candle_nn::ops::sigmoid(&self.inner)?;
        Ok(Self::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

// ============================================================================
// Reduction operations — these change the shape
// ============================================================================

impl<S: Shape, T: DType, D: Device, G: RequiresGrad> Tensor<S, T, D, G> {
    /// Sum all elements, producing a scalar tensor.
    pub fn sum_all(&self) -> Result<Tensor<(), T, D, G>> {
        let inner = self.inner.sum_all()?;
        Ok(Tensor::from_parts(
            inner,
            (),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Mean of all elements, producing a scalar tensor.
    pub fn mean_all(&self) -> Result<Tensor<(), T, D, G>> {
        let inner = self.inner.mean_all()?;
        Ok(Tensor::from_parts(
            inner,
            (),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}
