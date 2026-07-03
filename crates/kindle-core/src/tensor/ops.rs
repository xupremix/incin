//! Element-wise tensor operations with compile-time shape checking.
//!
//! Operations require matching Shape, DType, Device, and RequiresGrad.
//! This ensures at compile time that you can't accidentally add tensors
//! of different shapes, dtypes, or on different devices.

use crate::prelude::{Backend, DType, Device, RequiresGrad, Result, Shape, DynShape, Tensor, Dyn};

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

impl<S: Shape, B: Backend<S>, T: DType, D: Device, G: RequiresGrad> Tensor<S, B, T, D, G> {
    pub fn abs(&self) -> Result<Self> {
        let inner = B::abs(&self.inner)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn relu(&self) -> Result<Self> {
        let inner = B::relu(&self.inner)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn gelu(&self) -> Result<Self> {
        let inner = B::gelu(&self.inner)?;
        Ok(Tensor::<_, B, _, _, _>::from_parts(
            inner,
            self._shape.clone(),
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
        fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
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
        let _res_gelu = t1.gelu().unwrap();
    }
}
