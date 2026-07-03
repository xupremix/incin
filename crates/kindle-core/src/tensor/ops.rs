//! Element-wise tensor operations with compile-time shape checking.
//!
//! Operations require matching Shape, DType, Device, and RequiresGrad.
//! This ensures at compile time that you can't accidentally add tensors
//! of different shapes, dtypes, or on different devices.

use crate::prelude::{Backend, DType, Device, RequiresGrad, Result, Shape, DynShape, Tensor, Dyn};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexSpec {
    All,
    Range(usize, usize),
    RangeFrom(usize),
    RangeTo(usize),
    Index(usize),
}

impl From<usize> for IndexSpec {
    fn from(idx: usize) -> Self { IndexSpec::Index(idx) }
}
impl From<core::ops::Range<usize>> for IndexSpec {
    fn from(r: core::ops::Range<usize>) -> Self { IndexSpec::Range(r.start, r.end) }
}
impl From<core::ops::RangeFrom<usize>> for IndexSpec {
    fn from(r: core::ops::RangeFrom<usize>) -> Self { IndexSpec::RangeFrom(r.start) }
}
impl From<core::ops::RangeTo<usize>> for IndexSpec {
    fn from(r: core::ops::RangeTo<usize>) -> Self { IndexSpec::RangeTo(r.end) }
}
impl From<core::ops::RangeFull> for IndexSpec {
    fn from(_: core::ops::RangeFull) -> Self { IndexSpec::All }
}

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
        impl<S: Shape, B: Backend, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
            pub fn $method(&self, rhs: &Self) -> Result<Self> {
                let inner = self.inner.$candle_method(&rhs.inner)?;
                Ok(Tensor::<_, B, _, _, _>::from_parts(
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

impl<S: Shape, B: Backend, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    /// Multiply all elements by a scalar.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = (&self.inner * scalar)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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
        Ok(Tensor::<_, B, _, _, _>::from_parts(
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

impl<S: Shape, B: Backend, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    /// Sum all elements, producing a scalar tensor.
    pub fn sum_all(&self) -> Result<Tensor<(), B, T, D, G>> {
        let inner = self.inner.sum_all()?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            (),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Mean of all elements, producing a scalar tensor.
    pub fn mean_all(&self) -> Result<Tensor<(), B, T, D, G>> {
        let inner = self.inner.mean_all()?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            (),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

// ============================================================================
// Slice operations (dynamic shapes)
// ============================================================================

impl<S: DynShape, B: Backend, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    pub fn slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, T, D, G>> {
        let mut inner = self.inner.clone();
        for (dim, spec) in specs.iter().enumerate() {
            match spec {
                IndexSpec::All => {}
                IndexSpec::Range(start, end) => {
                    inner = inner.narrow(dim, *start, *end - *start)?;
                }
                IndexSpec::RangeFrom(start) => {
                    let len = inner.dims()[dim] - start;
                    inner = inner.narrow(dim, *start, len)?;
                }
                IndexSpec::RangeTo(end) => {
                    inner = inner.narrow(dim, 0, *end)?;
                }
                IndexSpec::Index(idx) => {
                    inner = inner.narrow(dim, *idx, 1)?.squeeze(dim)?;
                }
            }
        }
        
        let new_shape: alloc::vec::Vec<usize> = inner.dims().to_vec();
        
        Ok(Tensor::<Dyn, B, T, D, G>::from_parts(
            inner,
            new_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}
