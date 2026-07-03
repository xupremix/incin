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

macro_rules! impl_binary_op {
    ($trait_name:ident, $method:ident, $backend_method:ident) => {
        // Tensor op Tensor → Result<Tensor> (owned)
        impl<S: Shape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
            pub fn $method(&self, rhs: &Self) -> Result<Self> {
                let inner = B::$backend_method(&self.inner, &rhs.inner)?;
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

macro_rules! impl_unary_op {
    ($method:ident, $backend_method:ident) => {
        pub fn $method(&self) -> Result<Self> {
            let inner = B::$backend_method(&self.inner)?;
            Ok(Tensor::<_, B, _, _, _>::from_parts(
                inner,
                self._shape.clone(),
                self._dtype.clone(),
                self._device.clone(),
                self._grad.clone(),
            ))
        }
    };
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    impl_unary_op!(abs, abs);
    impl_unary_op!(relu, relu);
    impl_unary_op!(gelu, gelu);
    impl_unary_op!(neg, neg);
    impl_unary_op!(sqrt, sqrt);
    impl_unary_op!(exp, exp);
    impl_unary_op!(log, log);
    impl_unary_op!(tanh, tanh);
    impl_unary_op!(sigmoid, sigmoid);

    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = B::mul_scalar(&self.inner, scalar)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = B::add_scalar(&self.inner, scalar)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

macro_rules! impl_reduction_op {
    ($method:ident, $backend_method:ident) => {
        pub fn $method(self) -> Result<Tensor<(), B, T, D, G>> 
        where 
            B: Backend<(), RawTensor = <B as Backend<S>>::RawTensor> 
        {
            let inner = <B as Backend<S>>::$backend_method(&self.inner)?;
            Ok(Tensor::<_, B, _, _, _>::from_parts(
                inner,
                (), // Scalar shape field
                self._dtype,
                self._device,
                self._grad,
            ))
        }
    };
}

impl<S: Shape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    impl_reduction_op!(sum_all, sum_all);
    impl_reduction_op!(mean_all, mean_all);
}

impl<S: DynShape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> 
where 
    B: Backend<Dyn, RawTensor = <B as Backend<S>>::RawTensor>
{
    pub fn dyn_slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, T, D, G>> {
        let mut inner = self.inner.clone();
        for (dim, spec) in specs.iter().enumerate() {
            match spec {
                IndexSpec::All => {}
                IndexSpec::Range(start, end) => {
                    inner = <B as Backend<S>>::narrow(&inner, dim, *start, *end - *start)?;
                }
                IndexSpec::RangeFrom(start) => {
                    let current_dims = S::dims(&self._shape);
                    let len = current_dims.as_ref()[dim] - start;
                    inner = <B as Backend<S>>::narrow(&inner, dim, *start, len)?;
                }
                IndexSpec::RangeTo(end) => {
                    inner = <B as Backend<S>>::narrow(&inner, dim, 0, *end)?;
                }
                IndexSpec::Index(idx) => {
                    let narrowed = <B as Backend<S>>::narrow(&inner, dim, *idx, 1)?;
                    inner = <B as Backend<S>>::squeeze(&narrowed, dim)?;
                }
            }
        }

        Ok(Tensor::<Dyn, B, T, D, G>::from_parts(
            inner,
            alloc::vec![], 
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use crate::prelude::{KindleDType, KindleDevice};

    pub struct DummyOpsBackend;
    impl<S: Shape> Backend<S> for DummyOpsBackend {
        type RawTensor = ();
        fn zeros(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn ones(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        
        fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        
        fn mul_scalar(_t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> { Ok(()) }
        fn add_scalar(_t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> { Ok(()) }

        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }

        fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
        
        fn narrow(_t: &Self::RawTensor, _dim: usize, _s: usize, _l: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn conv2d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
    }

    #[test]
    fn test_tensor_ops() {
        let t1: Tensor<Dyn, DummyOpsBackend> = Tensor::zeros(vec![2, 2]).unwrap();
        let t2: Tensor<Dyn, DummyOpsBackend> = Tensor::ones(vec![2, 2]).unwrap();
        
        // Binary ops
        let _res_add = t1.add(&t2).unwrap();
        let _res_sub = t1.sub(&t2).unwrap();
        let _res_mul = t1.mul(&t2).unwrap();
        let _res_div = t1.div(&t2).unwrap();
        
        // Unary ops
        let _res_abs = t1.abs().unwrap();
        let _res_relu = t1.relu().unwrap();
        let _res_exp = t1.exp().unwrap();
        
        // Scalar ops
        let _res_muls = t1.mul_scalar(2.0).unwrap();
        
        // Slicing
        let _res_slice = t1.dyn_slice(&[IndexSpec::All, IndexSpec::Index(0)]).unwrap();
    }
}
