pub use kindle_core::prelude::*;

pub mod prelude {
    #[cfg(feature = "burn")]
    pub use super::burn_backend::*;
    #[cfg(feature = "candle")]
    pub use super::candle::*;
    #[cfg(feature = "ndarray")]
    pub use super::ndarray_backend::*;
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

    pub fn to_candle_device(dev: &KindleDevice) -> Result<candle::Device> {
        use kindle_core::tensor::device::DeviceVariant;
        match dev.variant() {
            DeviceVariant::Cpu => Ok(candle::Device::Cpu),
            #[cfg(feature = "cuda")]
            DeviceVariant::Cuda(ord) => {
                Ok(candle::Device::new_cuda(ord).map_err(|e| anyhow::anyhow!(e))?)
            }
            #[cfg(feature = "metal")]
            DeviceVariant::Metal(ord) => {
                Ok(candle::Device::new_metal(ord).map_err(|e| anyhow::anyhow!(e))?)
            }
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
        type RawVar = candle::Var;
        type Grads = candle_core::backprop::GradStore;

        fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor> {
            Ok(var.as_tensor().clone())
        }

        fn zeros(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(
                candle::Tensor::zeros(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e| anyhow::anyhow!(e))?,
            )
        }

        fn ones(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(
                candle::Tensor::ones(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e| anyhow::anyhow!(e))?,
            )
        }

        fn rand(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(
                candle::Tensor::rand(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e| anyhow::anyhow!(e))?
                    .to_dtype(to_candle_dtype(dtype))
                    .map_err(|e| anyhow::anyhow!(e))?,
            )
        }

        fn randn(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(
                candle::Tensor::randn(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e| anyhow::anyhow!(e))?
                    .to_dtype(to_candle_dtype(dtype))
                    .map_err(|e| anyhow::anyhow!(e))?,
            )
        }

        fn var_zeros(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(
                candle::Var::zeros(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e| anyhow::anyhow!(e))?,
            )
        }

        fn var_ones(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(
                candle::Var::ones(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e| anyhow::anyhow!(e))?,
            )
        }

        fn var_rand(
            shape: &[usize],
            _dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(
                candle::Var::rand(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e| anyhow::anyhow!(e))?,
            )
        }

        fn tensor_to_device(t: &Self::RawTensor, device: &KindleDevice) -> Result<Self::RawTensor> {
            let dev = to_candle_device(device)?;
            t.to_device(&dev).map_err(|e| anyhow::anyhow!(e).into())
        }

        fn var_to_device(var: &Self::RawVar, device: &KindleDevice) -> Result<Self::RawVar> {
            let dev = to_candle_device(device)?;
            // Candle variables are fundamentally tensors inside a refcell. To move a Var,
            // we have to get the underlying tensor, move it, and wrap it in a new Var.
            // Let's create a new Var from the moved tensor.
            let t = var.as_tensor().to_device(&dev).map_err(|e| anyhow::anyhow!(e))?;
            candle::Var::from_tensor(&t).map_err(|e| anyhow::anyhow!(e).into())
        }

        fn var_randn(
            shape: &[usize],
            _dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            let dev = to_candle_device(device)?;
            Ok(
                candle::Var::randn(0f32, 1f32, shape, &dev)
                    .map_err(|e| anyhow::anyhow!(e))?,
            )
        }

        fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.relu().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.gelu_erf().map_err(|e| anyhow::anyhow!(e))?)
        } // using gelu_erf as fallback for general
        fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.abs().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn neg(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.neg().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn sqrt(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.sqrt().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn exp(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.exp().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn log(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.log().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn tanh(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.tanh().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn sigmoid(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(::candle_nn::ops::sigmoid(t).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn mul_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor> {
            Ok((t * scalar).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn add_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor> {
            Ok((t + scalar).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn sum_all(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.sum_all().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn mean_all(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.mean_all().map_err(|e| anyhow::anyhow!(e))?)
        }

        fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs.broadcast_add(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs.broadcast_sub(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs.broadcast_mul(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs.broadcast_div(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs.broadcast_matmul(rhs).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(t.reshape(shape).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn narrow(
            t: &Self::RawTensor,
            dim: usize,
            start: usize,
            len: usize,
        ) -> Result<Self::RawTensor> {
            Ok(t.narrow(dim, start, len).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn squeeze(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.squeeze(dim).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn conv2d(
            t: &Self::RawTensor,
            weight: &Self::RawTensor,
            _bias: Option<&Self::RawTensor>,
            stride: usize,
            padding: usize,
            dilation: usize,
        ) -> Result<Self::RawTensor> {
            // Candle's conv2d handles dilation and padding through conv2d operation arguments.
            // Using candle_nn::conv2d or directly the conv2d method if available on Tensor.
            Ok(t.conv2d(weight, padding, stride, dilation, 1)
                .map_err(|e| anyhow::anyhow!(e))?)
        }
        
        fn backward(loss: &Self::RawTensor) -> Result<Self::Grads> {
            Ok(loss.backward().map_err(|e| anyhow::anyhow!(e))?)
        }

        fn step_sgd(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()> {
            use candle_nn::optim::Optimizer;
            let mut sgd = candle_nn::optim::SGD::new(params.to_vec(), lr).map_err(|e| anyhow::anyhow!(e))?;
            sgd.step(grads).map_err(|e| anyhow::anyhow!(e))?;
            Ok(())
        }

        fn step_adamw(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()> {
            use candle_nn::optim::Optimizer;
            let mut adamw = candle_nn::optim::AdamW::new_lr(params.to_vec(), lr).map_err(|e| anyhow::anyhow!(e))?;
            adamw.step(grads).map_err(|e| anyhow::anyhow!(e))?;
            Ok(())
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
        type RawVar = ndarray::ArrayD<f32>;
        type Grads = ();

        fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor> {
            Ok(var.clone())
        }

        fn zeros(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(ndarray::ArrayD::<f32>::zeros(shape))
        }
        fn ones(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(ndarray::ArrayD::<f32>::ones(shape))
        }
        fn rand(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "rand",
                backend: "Ndarray",
            })
        }
        fn randn(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "randn",
                backend: "Ndarray",
            })
        }

        fn var_zeros(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<Self::RawVar> {
            Err(Error::UnsupportedBackendOperation {
                op: "var_zeros",
                backend: "Ndarray",
            })
        }
        fn var_ones(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<Self::RawVar> {
            Err(Error::UnsupportedBackendOperation {
                op: "var_ones",
                backend: "Ndarray",
            })
        }
        fn var_rand(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<Self::RawVar> {
            Err(Error::UnsupportedBackendOperation {
                op: "var_rand",
                backend: "Ndarray",
            })
        }
        fn tensor_to_device(t: &Self::RawTensor, _device: &KindleDevice) -> Result<Self::RawTensor> {
            Ok(t.clone())
        }
        fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
            Ok(var.clone())
        }
        fn var_randn(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<Self::RawVar> {
            Err(Error::UnsupportedBackendOperation {
                op: "var_randn",
                backend: "Ndarray",
            })
        }
        fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.mapv(|x| if x > 0.0 { x } else { 0.0 }))
        }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "gelu",
                backend: "Ndarray",
            })
        }
        fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.mapv(|x| x.abs()))
        }
        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "neg",
                backend: "Ndarray",
            })
        }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "sqrt",
                backend: "Ndarray",
            })
        }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "exp",
                backend: "Ndarray",
            })
        }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "log",
                backend: "Ndarray",
            })
        }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "tanh",
                backend: "Ndarray",
            })
        }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "sigmoid",
                backend: "Ndarray",
            })
        }
        fn mul_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "mul_scalar",
                backend: "Ndarray",
            })
        }
        fn add_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "add_scalar",
                backend: "Ndarray",
            })
        }
        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_all",
                backend: "Ndarray",
            })
        }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_all",
                backend: "Ndarray",
            })
        }
        fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs + rhs)
        }
        fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs - rhs)
        }
        fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs * rhs)
        }
        fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(lhs / rhs)
        }
        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "matmul",
                backend: "Ndarray",
            })
        }
        fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor> {
            t.to_owned()
                .into_shape_with_order(shape)
                .map_err(|e| anyhow::anyhow!(e).into())
        }
        fn narrow(
            _t: &Self::RawTensor,
            _dim: usize,
            _start: usize,
            _len: usize,
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "narrow",
                backend: "Ndarray",
            })
        }
        fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "squeeze",
                backend: "Ndarray",
            })
        }
        fn conv2d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _s: usize,
            _p: usize,
            _d: usize,
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "conv2d",
                backend: "Ndarray",
            })
        }
        
        fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> {
            Err(Error::UnsupportedBackendOperation {
                op: "backward",
                backend: "Ndarray",
            })
        }

        fn step_sgd(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> {
            Err(Error::UnsupportedBackendOperation {
                op: "step_sgd",
                backend: "Ndarray",
            })
        }

        fn step_adamw(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> {
            Err(Error::UnsupportedBackendOperation {
                op: "step_adamw",
                backend: "Ndarray",
            })
        }
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
                type RawVar = burn::tensor::Tensor<B, $n>;
                type Grads = (); // TODO: update when Burn supports this

                fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor> {
                    Ok(var.clone())
                }

                fn zeros(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> {
                    let d: [usize; $n] = shape.try_into().map_err(|_| Error::ShapeMismatch { expected: alloc::vec![$n], got: shape.to_vec() })?;
                    Ok(burn::tensor::Tensor::<B, $n>::zeros(d, &B::Device::default()))
                }
                fn ones(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> {
                    let d: [usize; $n] = shape.try_into().map_err(|_| Error::ShapeMismatch { expected: alloc::vec![$n], got: shape.to_vec() })?;
                    Ok(burn::tensor::Tensor::<B, $n>::ones(d, &B::Device::default()))
                }
                fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "rand", backend: "Burn" })
                }
                fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "randn", backend: "Burn" })
                }

                fn var_zeros(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<Self::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_zeros", backend: "Burn" }) }
                fn var_ones(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<Self::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_ones", backend: "Burn" }) }
                fn var_rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_rand", backend: "Burn" }) }
                fn tensor_to_device(_t: &Self::RawTensor, _device: &KindleDevice) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "tensor_to_device", backend: "Burn" }) }
                fn var_to_device(_var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_to_device", backend: "Burn" }) }
                fn var_randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_randn", backend: "Burn" }) }
                fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(burn::tensor::activation::relu(t.clone()))
                }
                fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(burn::tensor::activation::gelu(t.clone()))
                }
                fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(t.clone().abs())
                }
                fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "neg", backend: "Burn" }) }
                fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sqrt", backend: "Burn" }) }
                fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "exp", backend: "Burn" }) }
                fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "log", backend: "Burn" }) }
                fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "tanh", backend: "Burn" }) }
                fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sigmoid", backend: "Burn" }) }
                fn mul_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mul_scalar", backend: "Burn" }) }
                fn add_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "add_scalar", backend: "Burn" }) }
                fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_all", backend: "Burn" }) }
                fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_all", backend: "Burn" }) }
                fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(lhs.clone() + rhs.clone())
                }
                fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(lhs.clone() - rhs.clone())
                }
                fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(lhs.clone() * rhs.clone())
                }
                fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(lhs.clone() / rhs.clone())
                }
                fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(lhs.clone().matmul(rhs.clone()))
                }
                fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "reshape", backend: "Burn" })
                }
                fn narrow(_t: &Self::RawTensor, _dim: usize, _start: usize, _len: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "narrow", backend: "Burn" }) }
                fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "squeeze", backend: "Burn" }) }
                fn conv2d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "conv2d", backend: "Burn" }) }
                
                fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> { Err(Error::UnsupportedBackendOperation { op: "backward", backend: "Burn" }) }
                fn step_sgd(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> { Err(Error::UnsupportedBackendOperation { op: "step_sgd", backend: "Burn" }) }
                fn step_adamw(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> { Err(Error::UnsupportedBackendOperation { op: "step_adamw", backend: "Burn" }) }
            }
        };
    }

    impl_burn_backend!(1, D0);
    impl_burn_backend!(2, D0, D1);
    impl_burn_backend!(3, D0, D1, D2);
    impl_burn_backend!(4, D0, D1, D2, D3);
    impl_burn_backend!(5, D0, D1, D2, D3, D4);
}
