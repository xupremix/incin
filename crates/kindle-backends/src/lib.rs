pub use kindle_core::prelude::*;
use core::result::Result;

pub mod prelude {
    #[cfg(feature = "candle")]
    pub use super::candle::*;
    #[cfg(feature = "ndarray")]
    pub use super::ndarray_backend::*;
    #[cfg(feature = "burn")]
    pub use super::burn_backend::*;
}

// ----------------------------------------------------------------------------
// CandleBackend
// ----------------------------------------------------------------------------
#[cfg(feature = "candle")]
pub mod candle {
    use super::*;
    use candle_core as candle;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CandleBackend;

    pub fn to_candle_device(dev: &KindleDevice) -> Result<candle::Device, Error> {
        use kindle_core::tensor::device::DeviceVariant;
        match dev.variant() {
            DeviceVariant::Cpu => Ok(candle::Device::Cpu),
            #[cfg(feature = "cuda")]
            DeviceVariant::Cuda(ord) => Ok(candle::Device::new_cuda(ord).map_err(|e| anyhow::anyhow!(e))?),
            #[cfg(feature = "metal")]
            DeviceVariant::Metal(ord) => Ok(candle::Device::new_metal(ord).map_err(|e| anyhow::anyhow!(e))?),
        }
    }

    pub fn to_candle_dtype(dtype: KindleDType) -> candle::DType {
        match dtype {
            KindleDType::U8 => candle::DType::U8,
            KindleDType::U32 => candle::DType::U32,
            KindleDType::I64 => candle::DType::I64,
            KindleDType::BF16 => candle::DType::BF16,
            KindleDType::F16 => candle::DType::F16,
            KindleDType::F32 => candle::DType::F32,
            KindleDType::F64 => candle::DType::F64,
        }
    }

    impl<S: Shape> Backend<S> for CandleBackend {
        type RawTensor = candle::Tensor;

        fn zeros(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor, Error> {
            Ok(candle::Tensor::zeros(shape, to_candle_dtype(dtype), &to_candle_device(device)?).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn ones(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor, Error> {
            Ok(candle::Tensor::ones(shape, to_candle_dtype(dtype), &to_candle_device(device)?).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn rand(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor, Error> {
            Ok(candle::Tensor::rand(0f32, 1f32, shape, &to_candle_device(device)?).map_err(|e| anyhow::anyhow!(e))?.to_dtype(to_candle_dtype(dtype)).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn randn(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor, Error> {
            Ok(candle::Tensor::randn(0f32, 1f32, shape, &to_candle_device(device)?).map_err(|e| anyhow::anyhow!(e))?.to_dtype(to_candle_dtype(dtype)).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.relu().map_err(|e| anyhow::anyhow!(e))?) }
        fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.gelu_erf().map_err(|e| anyhow::anyhow!(e))?) } // using gelu_erf as fallback for general
        fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.abs().map_err(|e| anyhow::anyhow!(e))?) }
        fn neg(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.neg().map_err(|e| anyhow::anyhow!(e))?) }
        fn sqrt(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.sqrt().map_err(|e| anyhow::anyhow!(e))?) }
        fn exp(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.exp().map_err(|e| anyhow::anyhow!(e))?) }
        fn log(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.log().map_err(|e| anyhow::anyhow!(e))?) }
        fn tanh(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.tanh().map_err(|e| anyhow::anyhow!(e))?) }
        fn sigmoid(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(::candle_nn::ops::sigmoid(t).map_err(|e| anyhow::anyhow!(e))?) }

        fn mul_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor, Error> { Ok((t * scalar).map_err(|e| anyhow::anyhow!(e))?) }
        fn add_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor, Error> { Ok((t + scalar).map_err(|e| anyhow::anyhow!(e))?) }

        fn sum_all(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.sum_all().map_err(|e| anyhow::anyhow!(e))?) }
        fn mean_all(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Ok(t.mean_all().map_err(|e| anyhow::anyhow!(e))?) }

        fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs.broadcast_add(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs.broadcast_sub(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs.broadcast_mul(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs.broadcast_div(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs.broadcast_matmul(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor, Error> {
            Ok(t.reshape(shape).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn narrow(t: &Self::RawTensor, dim: usize, start: usize, len: usize) -> Result<Self::RawTensor, Error> {
            Ok(t.narrow(dim, start, len).map_err(|e| anyhow::anyhow!(e))?)
        }
        
        fn squeeze(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor, Error> {
            Ok(t.squeeze(dim).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn conv2d(
            t: &Self::RawTensor,
            weight: &Self::RawTensor,
            _bias: Option<&Self::RawTensor>,
            stride: usize,
            padding: usize,
            dilation: usize,
        ) -> Result<Self::RawTensor, Error> {
            // Candle's conv2d handles dilation and padding through conv2d operation arguments.
            // Using candle_nn::conv2d or directly the conv2d method if available on Tensor.
            Ok(t.conv2d(weight, padding, stride, dilation, 1).map_err(|e| anyhow::anyhow!(e))?)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_to_candle_dtype() {
            assert_eq!(to_candle_dtype(KindleDType::F32), candle::DType::F32);
            assert_eq!(to_candle_dtype(KindleDType::I64), candle::DType::I64);
        }

        #[test]
        fn test_to_candle_device() {
            let cpu = KindleDevice::cpu();
            let c_dev = to_candle_device(&cpu).unwrap();
            assert!(matches!(c_dev, candle::Device::Cpu));
        }
    }
}

// ----------------------------------------------------------------------------
// NdarrayBackend
// ----------------------------------------------------------------------------
#[cfg(feature = "ndarray")]
pub mod ndarray_backend {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NdarrayBackend;

    impl<S: Shape> Backend<S> for NdarrayBackend {
        type RawTensor = ndarray::ArrayD<f32>; // STUB: forced to f32

        fn zeros(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor, Error> {
            Ok(ndarray::ArrayD::<f32>::zeros(shape))
        }
        fn ones(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor, Error> {
            Ok(ndarray::ArrayD::<f32>::ones(shape))
        }
        fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor, Error> {
            Err(Error::UnsupportedBackendOperation { op: "rand", backend: "Ndarray" })
        }
        fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor, Error> {
            Err(Error::UnsupportedBackendOperation { op: "randn", backend: "Ndarray" })
        }
        fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(t.mapv(|x| if x > 0.0 { x } else { 0.0 }))
        }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Err(Error::UnsupportedBackendOperation { op: "gelu", backend: "Ndarray" })
        }
        fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(t.mapv(|x| x.abs()))
        }
        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "neg", backend: "Ndarray" }) }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "sqrt", backend: "Ndarray" }) }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "exp", backend: "Ndarray" }) }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "log", backend: "Ndarray" }) }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "tanh", backend: "Ndarray" }) }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "sigmoid", backend: "Ndarray" }) }
        fn mul_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "mul_scalar", backend: "Ndarray" }) }
        fn add_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "add_scalar", backend: "Ndarray" }) }
        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "sum_all", backend: "Ndarray" }) }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "mean_all", backend: "Ndarray" }) }
        fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs + rhs)
        }
        fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs - rhs)
        }
        fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs * rhs)
        }
        fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Ok(lhs / rhs)
        }
        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
            Err(Error::UnsupportedBackendOperation { op: "matmul", backend: "Ndarray" })
        }
        fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor, Error> {
            t.to_owned().into_shape_with_order(shape).map_err(|e| anyhow::anyhow!(e).into())
        }
        fn narrow(_t: &Self::RawTensor, _dim: usize, _start: usize, _len: usize) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "narrow", backend: "Ndarray" }) }
        fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "squeeze", backend: "Ndarray" }) }
        fn conv2d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "conv2d", backend: "Ndarray" }) }
    }
}

// ----------------------------------------------------------------------------
// BurnBackend
// ----------------------------------------------------------------------------
#[cfg(feature = "burn")]
pub mod burn_backend {
    use super::*;

    pub struct BurnBackend<B: burn::tensor::backend::Backend>(core::marker::PhantomData<B>);

    macro_rules! impl_burn_backend {
        ($n:expr, $($D:ident),*) => {
            impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> Backend<($($D,)*)> for BurnBackend<B> {
                type RawTensor = burn::tensor::Tensor<B, $n>;

                fn zeros(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor, Error> {
                    let d: [usize; $n] = shape.try_into().map_err(|_| Error::ShapeMismatch { expected: alloc::vec![$n], got: shape.to_vec() })?;
                    Ok(burn::tensor::Tensor::<B, $n>::zeros(d, &B::Device::default()))
                }
                fn ones(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor, Error> {
                    let d: [usize; $n] = shape.try_into().map_err(|_| Error::ShapeMismatch { expected: alloc::vec![$n], got: shape.to_vec() })?;
                    Ok(burn::tensor::Tensor::<B, $n>::ones(d, &B::Device::default()))
                }
                fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor, Error> {
                    Err(Error::UnsupportedBackendOperation { op: "rand", backend: "Burn" })
                }
                fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor, Error> {
                    Err(Error::UnsupportedBackendOperation { op: "randn", backend: "Burn" })
                }
                fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
                    Ok(burn::tensor::activation::relu(t.clone()))
                }
                fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
                    Ok(burn::tensor::activation::gelu(t.clone()))
                }
                fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
                    Ok(t.clone().abs())
                }
                fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "neg", backend: "Burn" }) }
                fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "sqrt", backend: "Burn" }) }
                fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "exp", backend: "Burn" }) }
                fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "log", backend: "Burn" }) }
                fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "tanh", backend: "Burn" }) }
                fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "sigmoid", backend: "Burn" }) }
                fn mul_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "mul_scalar", backend: "Burn" }) }
                fn add_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "add_scalar", backend: "Burn" }) }
                fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "sum_all", backend: "Burn" }) }
                fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "mean_all", backend: "Burn" }) }
                fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
                    Ok(lhs.clone() + rhs.clone())
                }
                fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
                    Ok(lhs.clone() - rhs.clone())
                }
                fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
                    Ok(lhs.clone() * rhs.clone())
                }
                fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
                    Ok(lhs.clone() / rhs.clone())
                }
                fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error> {
                    Ok(lhs.clone().matmul(rhs.clone()))
                }
                fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor, Error> {
                    Err(Error::UnsupportedBackendOperation { op: "reshape", backend: "Burn" })
                }
                fn narrow(_t: &Self::RawTensor, _dim: usize, _start: usize, _len: usize) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "narrow", backend: "Burn" }) }
                fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "squeeze", backend: "Burn" }) }
                fn conv2d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor, Error> { Err(Error::UnsupportedBackendOperation { op: "conv2d", backend: "Burn" }) }
            }
        };
    }

    impl_burn_backend!(1, D0);
    impl_burn_backend!(2, D0, D1);
    impl_burn_backend!(3, D0, D1, D2);
    impl_burn_backend!(4, D0, D1, D2, D3);
    impl_burn_backend!(5, D0, D1, D2, D3, D4);
}
