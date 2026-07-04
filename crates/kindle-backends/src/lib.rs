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
        type RawVar = candle_core::Var;
        type Grads = std::collections::HashMap<candle_core::TensorId, candle_core::Tensor>;

        fn shape(t: &Self::RawTensor) -> Vec<usize> {
            t.dims().to_vec()
        }

        fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor> {
            Ok(var.as_tensor().clone())
        }
        fn var_from_tensor(t: &Self::RawTensor) -> Result<Self::RawVar> {
            Ok(candle::Var::from_tensor(t).map_err(|e| anyhow::anyhow!(e))?)
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
            let t = var
                .as_tensor()
                .to_device(&dev)
                .map_err(|e| anyhow::anyhow!(e))?;
            candle::Var::from_tensor(&t).map_err(|e| anyhow::anyhow!(e).into())
        }

        fn var_randn(
            shape: &[usize],
            _dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            let dev = to_candle_device(device)?;
            Ok(candle::Var::randn(0f32, 1f32, shape, &dev).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.relu().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.gelu_erf().map_err(|e| anyhow::anyhow!(e))?)
        } // using gelu_erf as fallback for general

        fn softmax(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(candle_nn::ops::softmax(t, dim).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn swish(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            // swish is x * sigmoid(x)
            Ok(candle_nn::ops::silu(t).map_err(|e| anyhow::anyhow!(e))?)
        }
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
        fn max_all(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.max_all().map_err(|e| anyhow::anyhow!(e))?)
        }
        fn min_all(t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(t.min_all().map_err(|e| anyhow::anyhow!(e))?)
        }

        fn sum_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.sum(dim).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn sum_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.sum_keepdim(dim).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn stack(tensors: &[&Self::RawTensor], dim: usize) -> Result<Self::RawTensor> {
            Ok(candle_core::Tensor::stack(tensors, dim).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn concat(tensors: &[&Self::RawTensor], dim: usize) -> Result<Self::RawTensor> {
            Ok(candle_core::Tensor::cat(tensors, dim).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn layer_norm(
            t: &Self::RawTensor,
            weight: &Self::RawTensor,
            bias: &Self::RawTensor,
            eps: f32,
        ) -> Result<Self::RawTensor> {
            Ok(candle_nn::ops::layer_norm(t, weight, bias, eps).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn batch_norm(
            t: &Self::RawTensor,
            weight: &Self::RawTensor,
            bias: &Self::RawTensor,
            running_mean: &Self::RawTensor,
            running_var: &Self::RawTensor,
            eps: f32,
        ) -> Result<Self::RawTensor> {
            let mut shape = vec![1; t.rank()];
            if t.rank() > 1 {
                shape[1] = running_mean.dim(0).map_err(|e| anyhow::anyhow!(e))?;
            } else {
                shape[0] = running_mean.dim(0).map_err(|e| anyhow::anyhow!(e))?;
            }
            
            let r_mean = running_mean.reshape(shape.as_slice()).map_err(|e| anyhow::anyhow!(e))?;
            let r_var = running_var.reshape(shape.as_slice()).map_err(|e| anyhow::anyhow!(e))?;
            let w = weight.reshape(shape.as_slice()).map_err(|e| anyhow::anyhow!(e))?;
            let b = bias.reshape(shape.as_slice()).map_err(|e| anyhow::anyhow!(e))?;

            let eps_t =
                candle_core::Tensor::new(&[eps], t.device()).map_err(|e| anyhow::anyhow!(e))?;
            let var_eps = r_var
                .broadcast_add(&eps_t)
                .map_err(|e| anyhow::anyhow!(e))?;
            let std = var_eps.sqrt().map_err(|e| anyhow::anyhow!(e))?;
            let normalized = t
                .broadcast_sub(&r_mean)
                .map_err(|e| anyhow::anyhow!(e))?
                .broadcast_div(&std)
                .map_err(|e| anyhow::anyhow!(e))?;

            let scaled = normalized
                .broadcast_mul(&w)
                .map_err(|e| anyhow::anyhow!(e))?;
            let out = scaled.broadcast_add(&b).map_err(|e| anyhow::anyhow!(e))?;
            Ok(out)
        }

        fn embedding(t: &Self::RawTensor, w: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(candle_nn::ops::embedding(t, w).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn mean_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.mean(dim).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn mean_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.mean_keepdim(dim).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn max_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.max(dim).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn max_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.max_keepdim(dim).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn min_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.min(dim).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn min_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
            Ok(t.min_keepdim(dim).map_err(|e| anyhow::anyhow!(e))?)
        }

        fn to_dtype(t: &Self::RawTensor, dtype: KindleDType) -> Result<Self::RawTensor> {
            Ok(t.to_dtype(to_candle_dtype(dtype))
                .map_err(|e| anyhow::anyhow!(e))?)
        }

        fn broadcast_as(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(t.broadcast_as(shape).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn broadcast_left(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(t.broadcast_left(shape).map_err(|e| anyhow::anyhow!(e))?)
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
        fn transpose(t: &Self::RawTensor, dim1: usize, dim2: usize) -> Result<Self::RawTensor> {
            Ok(t.transpose(dim1, dim2).map_err(|e| anyhow::anyhow!(e))?)
        }
        fn flatten(
            t: &Self::RawTensor,
            start_dim: usize,
            end_dim: usize,
        ) -> Result<Self::RawTensor> {
            Ok(t.flatten(start_dim, end_dim)
                .map_err(|e| anyhow::anyhow!(e))?)
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

        fn conv1d(
            t: &Self::RawTensor,
            w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            stride: usize,
            padding: usize,
            dilation: usize,
        ) -> Result<Self::RawTensor> {
            Ok(t.conv1d(w, padding, stride, dilation, 1)
                .map_err(|e| anyhow::anyhow!(e))?)
        }

        fn conv2d(
            t: &Self::RawTensor,
            weight: &Self::RawTensor,
            _bias: Option<&Self::RawTensor>,
            stride: usize,
            padding: usize,
            dilation: usize,
        ) -> Result<Self::RawTensor> {
            Ok(t.conv2d(weight, padding, stride, dilation, 1)
                .map_err(|e| anyhow::anyhow!(e))?)
        }

        fn conv_transpose2d(
            t: &Self::RawTensor,
            weight: &Self::RawTensor,
            _bias: Option<&Self::RawTensor>,
            stride: usize,
            padding: usize,
            output_padding: usize,
            dilation: usize,
        ) -> Result<Self::RawTensor> {
            Ok(t.conv_transpose2d(weight, padding, output_padding, stride, dilation)
                .map_err(|e| anyhow::anyhow!(e))?)
        }

        fn max_pool2d(
            t: &Self::RawTensor,
            kernel_size: (usize, usize),
            stride: (usize, usize),
        ) -> Result<Self::RawTensor> {
            Ok(t.max_pool2d_with_stride(
                (kernel_size.0, kernel_size.1),
                (stride.0, stride.1),
            )
            .map_err(|e| anyhow::anyhow!(e))?)
        }

        fn avg_pool2d(
            t: &Self::RawTensor,
            kernel_size: (usize, usize),
            stride: (usize, usize),
        ) -> Result<Self::RawTensor> {
            Ok(t.avg_pool2d_with_stride(
                (kernel_size.0, kernel_size.1),
                (stride.0, stride.1),
            )
            .map_err(|e| anyhow::anyhow!(e))?)
        }

        fn backward(loss: &Self::RawTensor) -> Result<Self::Grads> {
            Ok(loss.backward().map_err(|e| anyhow::anyhow!(e))?)
        }

        fn step_sgd(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()> {
            use candle_nn::optim::Optimizer;
            let mut sgd =
                candle_nn::optim::SGD::new(params.to_vec(), lr).map_err(|e| anyhow::anyhow!(e))?;
            sgd.step(grads).map_err(|e| anyhow::anyhow!(e))?;
            Ok(())
        }

        fn step_adamw(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()> {
            use candle_nn::optim::Optimizer;
            let mut adamw = candle_nn::optim::AdamW::new_lr(params.to_vec(), lr)
                .map_err(|e| anyhow::anyhow!(e))?;
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
        type RawVar = NdarrayVar;
        type Grads = NdarrayGrads;

        fn shape(_t: &Self::RawTensor) -> Vec<usize> {
            Vec::new()
        }

        fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor> {
            Ok(var.clone())
        }
        fn var_from_tensor(t: &Self::RawTensor) -> Result<Self::RawVar> {
            Ok(t.clone())
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
        fn tensor_to_device(
            t: &Self::RawTensor,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
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
        fn softmax(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "softmax",
                backend: "Ndarray",
            })
        }
        fn swish(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "swish",
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
        fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_all",
                backend: "Ndarray",
            })
        }
        fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_all",
                backend: "Ndarray",
            })
        }
        fn sum_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_dim",
                backend: "Ndarray",
            })
        }
        fn stack(_tensors: &[&Self::RawTensor], _dim: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "stack",
                backend: "Ndarray",
            })
        }
        fn concat(_tensors: &[&Self::RawTensor], _dim: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "concat",
                backend: "Ndarray",
            })
        }
        fn layer_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "layer_norm",
                backend: "Ndarray",
            })
        }
        fn batch_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _rm: &Self::RawTensor,
            _rv: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "batch_norm",
                backend: "Ndarray",
            })
        }

        fn embedding(_t: &Self::RawTensor, _w: &Self::RawTensor) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "embedding",
                backend: "Ndarray",
            })
        }
        fn sum_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_keepdim",
                backend: "Ndarray",
            })
        }
        fn mean_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_dim",
                backend: "Ndarray",
            })
        }
        fn mean_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_keepdim",
                backend: "Ndarray",
            })
        }
        fn max_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_dim",
                backend: "Ndarray",
            })
        }
        fn max_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_keepdim",
                backend: "Ndarray",
            })
        }
        fn min_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_dim",
                backend: "Ndarray",
            })
        }
        fn min_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_keepdim",
                backend: "Ndarray",
            })
        }
        fn to_dtype(_t: &Self::RawTensor, _dtype: KindleDType) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "to_dtype",
                backend: "Ndarray",
            })
        }
        fn broadcast_as(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "broadcast_as",
                backend: "Ndarray",
            })
        }
        fn broadcast_left(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "broadcast_left",
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
        fn transpose(_t: &Self::RawTensor, _d1: usize, _d2: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "transpose",
                backend: "Ndarray",
            })
        }
        fn flatten(_t: &Self::RawTensor, _s: usize, _e: usize) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "flatten",
                backend: "Ndarray",
            })
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

        fn conv1d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _s: usize,
            _p: usize,
            _d: usize,
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "conv1d",
                backend: "Ndarray",
            })
        }

        fn conv_transpose2d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _s: usize,
            _p: usize,
            _op: usize,
            _d: usize,
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "conv_transpose2d",
                backend: "Ndarray",
            })
        }

        fn max_pool2d(
            _t: &Self::RawTensor,
            _k: (usize, usize),
            _s: (usize, usize),
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_pool2d",
                backend: "Ndarray",
            })
        }

        fn avg_pool2d(
            _t: &Self::RawTensor,
            _k: (usize, usize),
            _s: (usize, usize),
        ) -> Result<Self::RawTensor> {
            Err(Error::UnsupportedBackendOperation {
                op: "avg_pool2d",
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
                fn softmax(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
                    Ok(burn::tensor::activation::softmax(t.clone(), dim))
                }
                fn swish(t: &Self::RawTensor) -> Result<Self::RawTensor> {
                    Ok(burn::tensor::activation::silu(t.clone()))
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
                fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_all", backend: "Burn" }) }
                fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_all", backend: "Burn" }) }
                fn sum_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_dim", backend: "Burn" }) }
                fn stack(_tensors: &[&Self::RawTensor], _dim: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "stack", backend: "Burn" }) }
                fn concat(_tensors: &[&Self::RawTensor], _dim: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "concat", backend: "Burn" }) }
                fn layer_norm(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: &Self::RawTensor, _e: f32) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "layer_norm", backend: "Burn" }) }
                fn batch_norm(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: &Self::RawTensor, _rm: &Self::RawTensor, _rv: &Self::RawTensor, _e: f32) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "batch_norm", backend: "Burn" }) }
                fn embedding(_t: &Self::RawTensor, _w: &Self::RawTensor) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "embedding", backend: "Burn" }) }
                fn sum_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_keepdim", backend: "Burn" }) }
                fn mean_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_dim", backend: "Burn" }) }
                fn mean_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_keepdim", backend: "Burn" }) }
                fn max_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_dim", backend: "Burn" }) }
                fn max_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_keepdim", backend: "Burn" }) }
                fn min_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_dim", backend: "Burn" }) }
                fn min_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_keepdim", backend: "Burn" }) }
                fn to_dtype(_t: &Self::RawTensor, _dtype: KindleDType) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "to_dtype", backend: "Burn" }) }
                fn broadcast_as(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "broadcast_as", backend: "Burn" }) }
                fn broadcast_left(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "broadcast_left", backend: "Burn" }) }
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
                fn transpose(_t: &Self::RawTensor, _d1: usize, _d2: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "transpose", backend: "Burn" }) }
                fn flatten(_t: &Self::RawTensor, _s: usize, _e: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "flatten", backend: "Burn" }) }
                fn narrow(_t: &Self::RawTensor, _dim: usize, _start: usize, _len: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "narrow", backend: "Burn" }) }
                fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "squeeze", backend: "Burn" }) }
                fn conv2d(_: &Self::RawTensor, _: &Self::RawTensor, _: Option<&Self::RawTensor>, _: usize, _: usize, _: usize) -> Result<Self::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv2d", backend: "Burn" })
                }
                fn conv1d(_: &Self::RawTensor, _: &Self::RawTensor, _: Option<&Self::RawTensor>, _: usize, _: usize, _: usize) -> Result<Self::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv1d", backend: "Burn" })
                }
                fn conv_transpose2d(_: &Self::RawTensor, _: &Self::RawTensor, _: Option<&Self::RawTensor>, _: usize, _: usize, _: usize, _: usize) -> Result<Self::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv_transpose2d", backend: "Burn" })
                }
                fn max_pool2d(_: &Self::RawTensor, _: (usize, usize), _: (usize, usize)) -> Result<Self::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "max_pool2d", backend: "Burn" })
                }
                fn avg_pool2d(_: &Self::RawTensor, _: (usize, usize), _: (usize, usize)) -> Result<Self::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "avg_pool2d", backend: "Burn" })
                } fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> { Err(Error::UnsupportedBackendOperation { op: "backward", backend: "Burn" }) }
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
