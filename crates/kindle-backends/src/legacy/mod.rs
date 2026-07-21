// ----------------------------------------------------------------------------
// Legacy Wrappers (Burn, Candle, Ndarray)
// ----------------------------------------------------------------------------

pub use kindle_core::prelude::*;


// ----------------------------------------------------------------------------
// CandleBackend
// ----------------------------------------------------------------------------

/// Auto-generated documentation for candle.
pub mod candle {
    use super::*;
    use candle_core as candle;

    /// # Backend Float Element Limitation (B-4)
    /// **Known Limitation:** `CandleBackend` ignores its compile-time `T` generic
    /// for inner allocation precision and relies on the dynamic `KindleDType`
    /// supplied to creation methods.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CandleBackend<T, D>(core::marker::PhantomData<(T, D)>);

    /// Auto-generated documentation for to_candle_device.
    pub fn to_candle_device(dev: &KindleDevice) -> Result<candle::Device> {
        use kindle_core::prelude::DeviceVariant;
        match dev.variant() {
            DeviceVariant::Cpu => Ok(candle::Device::Cpu),
            #[cfg(feature = "cuda")]
            DeviceVariant::Cuda(ord) => Ok(candle::Device::new_cuda(ord)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            #[cfg(feature = "wgpu")]
            DeviceVariant::Wgpu(ord) => Ok(candle::Device::new_metal(ord)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            _ => Err(Error::UnsupportedBackendOperation {
                op: "to_candle_device",
                backend: "Candle",
            }),
        }
    }

    /// Auto-generated documentation for to_candle_dtype.
    pub fn to_candle_dtype(dtype: KindleDType) -> candle::DType {
        match dtype {
            KindleDType::U8 => candle::DType::U8,
            KindleDType::U32 => candle::DType::U32,
            KindleDType::I64 => candle::DType::I64,
            KindleDType::BF16 => candle::DType::BF16,
            KindleDType::F16 => candle::DType::F16,
            KindleDType::F32 => candle::DType::F32,
            KindleDType::F64 => candle::DType::F64,
            KindleDType::Q8_0 => unimplemented!("Q8_0 is not natively supported in candle yet"),
            _ => unimplemented!("Unsupported dtype in candle"),
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::Backend for CandleBackend<T, D>
    {
        /// Auto-generated documentation for Device.
        type Device = D;
        /// Auto-generated documentation for FloatElem.
        type FloatElem = T;
        /// Auto-generated documentation for IntElem.
        type IntElem = i64;
        /// Auto-generated documentation for BackendWithDevice.
        type BackendWithDevice<NewD: kindle_core::prelude::Device> = CandleBackend<T, NewD>;

        /// Auto-generated documentation for Storage.
        type Storage<K: kindle_core::prelude::DType> = candle_core::Tensor;
        /// Auto-generated documentation for RawVar.
        type RawVar = candle_core::Var;
        /// Auto-generated documentation for Grads.
        type Grads = candle_core::backprop::GradStore;
        /// Auto-generated documentation for InnerBackend.
        type InnerBackend = Self;

        /// Auto-generated documentation for shape.
        fn shape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Vec<usize> {
            t.dims().to_vec()
        }

        /// Auto-generated documentation for format_tensor_display.
        fn format_tensor_display<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("{}", t)
        }
        /// Auto-generated documentation for format_tensor_debug.
        fn format_tensor_debug<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("Raw Tensor: {:?}, Strides: {:?}", t, t.stride())
        }

        /// Auto-generated documentation for var_as_tensor.
        fn var_as_tensor<K: kindle_core::prelude::DType>(
            var: &<Self as kindle_core::prelude::Backend>::RawVar,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(var.as_tensor().clone())
        }
        /// Auto-generated documentation for var_from_tensor.
        fn var_from_tensor<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(candle::Var::from_tensor(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for var_to_device.
        fn var_to_device(
            var: &<Self as kindle_core::prelude::Backend>::RawVar,
            device: &kindle_core::prelude::KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            let dev = to_candle_device(device)?;
            var.as_tensor()
                .to_device(&dev)
                .and_then(|t| candle::Var::from_tensor(&t))
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        /// Auto-generated documentation for assign_var.
        fn assign_var<K: kindle_core::prelude::DType>(
            var: &mut <Self as kindle_core::prelude::Backend>::RawVar,
            tensor: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<()> {
            var.set(tensor)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        /// Auto-generated documentation for backward.
        fn backward<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            loss.backward()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        /// Auto-generated documentation for backward_with_nan_check.
        fn backward_with_nan_check<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            Self::backward::<K>(loss)
        }

        /// Auto-generated documentation for get_grad.
        fn get_grad<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            grads: &<Self as kindle_core::prelude::Backend>::Grads,
        ) -> Result<Option<<Self as kindle_core::prelude::Backend>::Storage<K>>> {
            Ok(grads.get(t).cloned())
        }

        /// Auto-generated documentation for to_bytes.
        fn to_bytes<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<u8>> {
            let v = t
                .flatten_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .to_vec1::<f32>()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let bytes = unsafe {
                core::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * core::mem::size_of::<f32>(),
                )
            };
            Ok(bytes.to_vec())
        }

        /// Auto-generated documentation for from_bytes.
        fn from_bytes<K: kindle_core::prelude::DType>(
            bytes: &[u8],
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let floats = unsafe {
                core::slice::from_raw_parts(
                    bytes.as_ptr() as *const f32,
                    bytes.len() / core::mem::size_of::<f32>(),
                )
            };
            let d = to_candle_device(device)?;
            let c_dtype = to_candle_dtype(dtype);
            let t = candle_core::Tensor::from_slice(floats, shape, &d)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let t = t
                .to_dtype(c_dtype)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(t)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::QuantizedOps<Self> for CandleBackend<T, D>
    {
        /// Auto-generated documentation for quantize.
        fn quantize<K: kindle_core::prelude::FloatDType, Q: kindle_core::prelude::QuantDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<Q>> {
            Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Candle",
            })
        }
        /// Auto-generated documentation for dequantize.
        fn dequantize<Q: kindle_core::prelude::QuantDType, K: kindle_core::prelude::FloatDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "dequantize",
                backend: "Candle",
            })
        }
        /// Auto-generated documentation for quantized_matmul.
        fn quantized_matmul<Q: kindle_core::prelude::QuantDType>(
            _lhs: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
            _rhs: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<f32>> {
            Err(Error::UnsupportedBackendOperation {
                op: "quantized_matmul",
                backend: "Candle",
            })
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::CreationOps<Self> for CandleBackend<T, D>
    {
        /// Auto-generated documentation for zeros.
        fn zeros<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                candle::Tensor::zeros(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for ones.
        fn ones<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                candle::Tensor::ones(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for rand.
        fn rand<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                candle::Tensor::rand(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                    .to_dtype(to_candle_dtype(dtype))
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for randn.
        fn randn<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                candle::Tensor::randn(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                    .to_dtype(to_candle_dtype(dtype))
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for var_zeros.
        fn var_zeros<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(
                candle::Var::zeros(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for var_ones.
        fn var_ones<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(
                candle::Var::ones(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for var_rand.
        fn var_rand<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(
                candle::Var::rand(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for var_randn.
        fn var_randn<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            let dev = to_candle_device(device)?;
            Ok(candle::Var::randn(0f32, 1f32, shape, &dev)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for tensor_to_device.
        fn tensor_to_device<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let dev = to_candle_device(device)?;
            t.to_device(&dev)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::NumericOps<Self> for CandleBackend<T, D>
    {
        /// Auto-generated documentation for add.
        fn add<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_add(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for sub.
        fn sub<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_sub(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for mul.
        fn mul<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_mul(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for div.
        fn div<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_div(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::TensorOps<Self> for CandleBackend<T, D>
    {
        /// Auto-generated documentation for matmul.
        fn matmul<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let lhs_contig = lhs
                .contiguous()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let rhs_contig = rhs
                .contiguous()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;

            let l_shape = lhs_contig.dims();
            let r_shape = rhs_contig.dims();

            if l_shape.len() > 3 || r_shape.len() > 3 {
                let max_len = core::cmp::max(l_shape.len(), r_shape.len());
                let mut out_shape = vec![];

                for i in 0..max_len - 2 {
                    let l = if i < max_len - l_shape.len() {
                        1
                    } else {
                        l_shape[i - (max_len - l_shape.len())]
                    };
                    let r = if i < max_len - r_shape.len() {
                        1
                    } else {
                        r_shape[i - (max_len - r_shape.len())]
                    };
                    out_shape.push(core::cmp::max(l, r));
                }

                let m = l_shape[l_shape.len() - 2];
                let k = l_shape[l_shape.len() - 1];
                let n = r_shape[r_shape.len() - 1];

                let mut lhs_b_shape = out_shape.clone();
                lhs_b_shape.push(m);
                lhs_b_shape.push(k);

                let mut rhs_b_shape = out_shape.clone();
                rhs_b_shape.push(k);
                rhs_b_shape.push(n);

                let lhs_b = lhs_contig
                    .broadcast_as(lhs_b_shape.as_slice())
                    .map_err(|e| anyhow::anyhow!(e))?
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!(e))?;
                let rhs_b = rhs_contig
                    .broadcast_as(rhs_b_shape.as_slice())
                    .map_err(|e| anyhow::anyhow!(e))?
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!(e))?;

                let batch_size: usize = out_shape.iter().product();
                let lhs_flat = lhs_b
                    .reshape((batch_size, m, k))
                    .map_err(|e| anyhow::anyhow!(e))?;
                let rhs_flat = rhs_b
                    .reshape((batch_size, k, n))
                    .map_err(|e| anyhow::anyhow!(e))?;

                let res_flat = lhs_flat.matmul(&rhs_flat).map_err(|e| anyhow::anyhow!(e))?;

                let mut res_shape = out_shape;
                res_shape.push(m);
                res_shape.push(n);
                return Ok(res_flat
                    .reshape(res_shape.as_slice())
                    .map_err(|e| anyhow::anyhow!(e))?);
            }

            Ok(lhs_contig
                .broadcast_matmul(&rhs_contig)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for stack.
        fn stack<K: kindle_core::prelude::DType>(
            tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_core::Tensor::stack(tensors, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for concat.
        fn concat<K: kindle_core::prelude::DType>(
            tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_core::Tensor::cat(tensors, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for broadcast_as.
        fn broadcast_as<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.broadcast_as(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for broadcast_left.
        fn broadcast_left<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.broadcast_left(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for reshape.
        fn reshape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.reshape(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for transpose.
        fn transpose<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim1: usize,
            dim2: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.transpose(dim1, dim2)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for slice.
        fn slice<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            ranges: &[(usize, usize)],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let mut out = t.clone();
            for (dim, &(start, end)) in ranges.iter().enumerate() {
                out = out
                    .narrow(dim, start, end - start)
                    .map_err(|e| Error::Msg(format!("Candle narrow failed for slice: {}", e)))?;
            }
            Ok(out)
        }

        /// Auto-generated documentation for flatten.
        fn flatten<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            start_dim: usize,
            end_dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.flatten(start_dim, end_dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for narrow.
        fn narrow<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
            start: usize,
            len: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.narrow(dim, start, len)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for squeeze.
        fn squeeze<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.squeeze(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for float_to_scalar.
        fn float_to_scalar<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<f64> {
            let v = t
                .to_dtype(candle_core::DType::F32)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let s: f32 = v
                .to_scalar()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(s as f64)
        }
        /// Auto-generated documentation for float_to_vec1.
        fn float_to_vec1<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<Vec<f64>> {
            let v = t
                .to_dtype(candle_core::DType::F32)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let vec: Vec<f32> = v
                .to_vec1()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(vec.into_iter().map(|x| x as f64).collect())
        }
        /// Auto-generated documentation for int_to_scalar.
        fn int_to_scalar<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<i64> {
            let v = t
                .to_dtype(candle_core::DType::I64)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let s: i64 = v
                .to_scalar()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(s)
        }
        /// Auto-generated documentation for int_to_vec1.
        fn int_to_vec1<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<Vec<i64>> {
            let v = t
                .to_dtype(candle_core::DType::I64)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let vec: Vec<i64> = v
                .to_vec1()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(vec)
        }
        /// Auto-generated documentation for tensor_to_dtype.
        fn tensor_to_dtype<K: kindle_core::prelude::DType, K2: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dtype: KindleDType,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K2>> {
            Ok(t.to_dtype(to_candle_dtype(dtype))
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::FloatOps<Self> for CandleBackend<T, D>
    {
        /// Auto-generated documentation for add_scalar_float.
        fn add_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok((t + scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for mul_scalar_float.
        fn mul_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok((t * scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for relu.
        fn relu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.relu()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for gelu.
        fn gelu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.gelu_erf()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        } // using gelu_erf as fallback for general

        fn step<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("step not implemented for CandleBackend")
        }

        fn mish<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("mish not implemented for CandleBackend")
        }

        fn elu<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("elu not implemented for CandleBackend")
        }

        /// Auto-generated documentation for softmax.
        fn softmax<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_nn::ops::softmax(t, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for swish.
        fn swish<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            // swish is x * sigmoid(x)
            Ok(candle_nn::ops::silu(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for abs.
        fn abs<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.abs()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for neg.
        fn neg<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.neg()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for sqrt.
        fn sqrt<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sqrt()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for exp.
        fn exp<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.exp()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for log.
        fn log<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.log()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for tanh.
        fn tanh<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.tanh()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for sigmoid.
        fn sigmoid<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(::candle_nn::ops::sigmoid(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ReductionOps<Self> for CandleBackend<T, D>
    {
        /// Auto-generated documentation for sum_all.
        fn sum_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for mean_all.
        fn mean_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for max_all.
        fn max_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for min_all.
        fn min_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for sum_dim.
        fn sum_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for sum_keepdim.
        fn sum_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for mean_dim.
        fn mean_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for mean_keepdim.
        fn mean_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for max_dim.
        fn max_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for max_keepdim.
        fn max_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for min_dim.
        fn min_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Auto-generated documentation for min_keepdim.
        fn min_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for argmax.
        fn argmax<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: Option<usize>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            match dim {
                Some(d) => Ok(t
                    .argmax(d)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
                None => Ok(t
                    .flatten_all()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                    .argmax(0)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            }
        }

        /// Auto-generated documentation for argmin.
        fn argmin<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: Option<usize>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            match dim {
                Some(d) => Ok(t
                    .argmin(d)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
                None => Ok(t
                    .flatten_all()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                    .argmin(0)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            }
        }

        fn topk<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _k: usize,
            _dim: usize,
            _largest: bool,
        ) -> Result<(<Self as kindle_core::prelude::Backend>::Storage<K>, <Self as kindle_core::prelude::Backend>::Storage<KInt>)> {
            unimplemented!("topk not implemented for CandleBackend")
        }

        fn argsort<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: usize,
            _descending: bool,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            unimplemented!("argsort not implemented for CandleBackend")
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ModuleOps<Self> for CandleBackend<T, D>
    {
        /// Auto-generated documentation for adaptive_avg_pool2d.
        fn adaptive_avg_pool2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _output_size: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("adaptive_avg_pool2d not implemented for CandleBackend")
        }

        /// Auto-generated documentation for layer_norm.
        fn layer_norm<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            weight: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            bias: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            eps: f32,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let zero_bias;
            let bias = match bias {
                Some(b) => b,
                None => {
                    zero_bias = weight
                        .zeros_like()
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    &zero_bias
                }
            };
            Ok(candle_nn::ops::layer_norm(t, weight, bias, eps)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for batch_norm.
        fn batch_norm<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            weight: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            bias: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            running_mean: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            running_var: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            eps: f32,
            _momentum: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let channel_dim = if t.rank() > 1 { 1 } else { 0 };
            let num_channels = t
                .dim(channel_dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;

            let mut shape = vec![1; t.rank()];
            shape[channel_dim] = num_channels;

            let owned_rm;
            let r_mean = match running_mean {
                Some(rm) => rm
                    .reshape(shape.as_slice())
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
                None => {
                    owned_rm = candle_core::Tensor::zeros(shape.as_slice(), t.dtype(), t.device())
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    owned_rm
                }
            };
            let owned_rv;
            let r_var = match running_var {
                Some(rv) => rv
                    .reshape(shape.as_slice())
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
                None => {
                    owned_rv = candle_core::Tensor::ones(shape.as_slice(), t.dtype(), t.device())
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    owned_rv
                }
            };
            let owned_w;
            let w = match weight {
                Some(w) => w
                    .reshape(shape.as_slice())
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
                None => {
                    owned_w = candle_core::Tensor::ones(shape.as_slice(), t.dtype(), t.device())
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    owned_w
                }
            };
            let owned_b;
            let b = match bias {
                Some(b) => b
                    .reshape(shape.as_slice())
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
                None => {
                    owned_b = candle_core::Tensor::zeros(shape.as_slice(), t.dtype(), t.device())
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                    owned_b
                }
            };

            let eps_t = candle_core::Tensor::new(&[eps], t.device())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let var_eps = r_var
                .broadcast_add(&eps_t)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let std = var_eps
                .sqrt()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let normalized = t
                .broadcast_sub(&r_mean)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .broadcast_div(&std)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;

            let scaled = normalized
                .broadcast_mul(&w)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let out = scaled
                .broadcast_add(&b)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(out)
        }

        /// Auto-generated documentation for embedding.
        fn embedding<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<KInt>,
            w: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            use candle_nn::Module;
            let hidden_size = w
                .dim(1)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let emb = candle_nn::Embedding::new(w.clone(), hidden_size);

            // Candle requires U32 or I64 for embedding indices
            let t_idx =
                if t.dtype() != candle_core::DType::U32 && t.dtype() != candle_core::DType::I64 {
                    t.to_dtype(candle_core::DType::U32)
                        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                } else {
                    t.clone()
                };

            Ok(emb
                .forward(&t_idx)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for conv1d.
        fn conv1d<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            w: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _b: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            stride: usize,
            padding: usize,
            dilation: usize,
            groups: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.conv1d(w, padding, stride, dilation, groups)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for conv2d.
        fn conv2d<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            weight: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _bias: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            stride: usize,
            padding: usize,
            dilation: usize,
            groups: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.conv2d(weight, padding, stride, dilation, groups)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Auto-generated documentation for conv_transpose2d.
        fn conv_transpose2d<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            weight: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _bias: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            stride: usize,
            padding: usize,
            output_padding: usize,
            dilation: usize,
            _groups: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                t.conv_transpose2d(weight, padding, output_padding, stride, dilation)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for max_pool2d.
        fn max_pool2d<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            kernel_size: (usize, usize),
            stride: (usize, usize),
            _padding: (usize, usize),
            _dilation: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                t.max_pool2d_with_stride((kernel_size.0, kernel_size.1), (stride.0, stride.1))
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Auto-generated documentation for avg_pool2d.
        fn avg_pool2d<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            kernel_size: (usize, usize),
            stride: (usize, usize),
            _padding: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                t.avg_pool2d_with_stride((kernel_size.0, kernel_size.1), (stride.0, stride.1))
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::LossOps<Self> for CandleBackend<T, D>
    {
        /// Auto-generated documentation for l1_loss.
        fn l1_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("l1_loss not implemented for CandleBackend")
        }

        /// Auto-generated documentation for bce_with_logits_loss.
        fn bce_with_logits_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("bce_with_logits_loss not implemented for CandleBackend")
        }

        /// Auto-generated documentation for mse_loss.
        fn mse_loss<K: kindle_core::prelude::DType>(
            pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let loss = candle_nn::loss::mse(pred, target)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(loss)
        }

        /// Auto-generated documentation for cross_entropy_loss.
        fn cross_entropy_loss<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            target: &<Self as kindle_core::prelude::Backend>::Storage<KInt>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let target_u32 = target
                .to_dtype(candle_core::DType::U32)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            candle_nn::loss::cross_entropy(pred, &target_u32)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }
    }

    #[cfg(test)]
    /// Auto-generated documentation for tests.
    mod tests {
        use super::*;

        #[test]
        /// Auto-generated documentation for test_to_candle_dtype.
        fn test_to_candle_dtype() {
            assert_eq!(to_candle_dtype(KindleDType::F32), candle::DType::F32);
            assert_eq!(to_candle_dtype(KindleDType::I64), candle::DType::I64);
        }

        #[test]
        /// Auto-generated documentation for test_to_candle_device.
        fn test_to_candle_device() {
            let cpu = KindleDevice::cpu();
            let c_dev = to_candle_device(&cpu).unwrap();
            assert!(matches!(c_dev, candle::Device::Cpu));
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::OptimizerOps<Self> for CandleBackend<T, D>
    {
    }
}

// ----------------------------------------------------------------------------
// NdarrayBackend
// ----------------------------------------------------------------------------

/// Auto-generated documentation for ndarray_backend.
#[cfg(any())]
pub mod ndarray_backend {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// # Backend Float Element Limitation (B-4)
    /// **Known Limitation:** `NdarrayBackend` ignores its compile-time `T` generic
    /// for inner allocation precision, hardcoding its backing tensor storage to `f32`
    /// (`ndarray::ArrayD<f32>`). Using this backend with `f64` types will silently
    /// downcast or allocate `f32` internally.
    pub struct NdarrayBackend<T, D>(core::marker::PhantomData<(T, D)>);

    #[derive(Clone, Debug)]
    /// Auto-generated documentation for NdarrayVar.
    pub struct NdarrayVar(pub alloc::sync::Arc<spin::RwLock<ndarray::ArrayD<f32>>>);
    #[derive(Clone, Debug)]
    /// Auto-generated documentation for NdarrayGrads.
    pub struct NdarrayGrads;

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::Backend for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for Device.
        type Device = D;
        /// Auto-generated documentation for FloatElem.
        type FloatElem = T;
        /// Auto-generated documentation for IntElem.
        type IntElem = i64;
        /// Auto-generated documentation for BackendWithDevice.
        type BackendWithDevice<NewD: kindle_core::prelude::Device> = NdarrayBackend<T, NewD>;

        /// Auto-generated documentation for Storage.
        type Storage<K: kindle_core::prelude::DType> = ndarray::ArrayD<f32>;
        /// Auto-generated documentation for RawVar.
        type RawVar = NdarrayVar;
        /// Auto-generated documentation for Grads.
        type Grads = NdarrayGrads;
        /// Auto-generated documentation for InnerBackend.
        type InnerBackend = Self;

        /// Auto-generated documentation for to_bytes.
        fn to_bytes<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<u8>> {
            let slice = t.as_slice().ok_or_else(|| {
                let err: kindle_core::prelude::Error =
                    anyhow::anyhow!("Ndarray is not contiguous").into();
                err
            })?;
            let bytes = unsafe {
                core::slice::from_raw_parts(
                    slice.as_ptr() as *const u8,
                    core::mem::size_of_val(slice),
                )
            };
            Ok(bytes.to_vec())
        }

        /// Auto-generated documentation for from_bytes.
        fn from_bytes<K: kindle_core::prelude::DType>(
            bytes: &[u8],
            shape: &[usize],
            dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            if dtype != KindleDType::F32 {
                let err: kindle_core::prelude::Error =
                    anyhow::anyhow!("NdarrayBackend only supports f32").into();
                return Err(err);
            }
            let floats = unsafe {
                core::slice::from_raw_parts(
                    bytes.as_ptr() as *const f32,
                    bytes.len() / core::mem::size_of::<f32>(),
                )
            };
            let arr = ndarray::Array::from_vec(floats.to_vec())
                .into_shape_with_order(shape)
                .map_err(|e: ndarray::ShapeError| anyhow::anyhow!(e))?
                .into_dyn();
            Ok(arr)
        }

        /// Auto-generated documentation for shape.
        fn shape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Vec<usize> {
            t.shape().to_vec()
        }

        /// Auto-generated documentation for format_tensor_display.
        fn format_tensor_display<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("{}", t)
        }
        /// Auto-generated documentation for format_tensor_debug.
        fn format_tensor_debug<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("Raw Tensor: {:?}, Strides: {:?}", t, t.strides())
        }

        /// Auto-generated documentation for var_as_tensor.
        fn var_as_tensor<K: kindle_core::prelude::DType>(
            v: &<Self as kindle_core::prelude::Backend>::RawVar,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(v.0.read().clone())
        }
        /// Auto-generated documentation for var_from_tensor.
        fn var_from_tensor<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(NdarrayVar(alloc::sync::Arc::new(spin::RwLock::new(
                t.clone(),
            ))))
        }

        /// Auto-generated documentation for assign_var.
        fn assign_var<K: kindle_core::prelude::DType>(
            var: &mut <Self as kindle_core::prelude::Backend>::RawVar,
            tensor: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<()> {
            let mut w = var.0.write();
            *w = tensor.clone();
            Ok(())
        }

        /// Auto-generated documentation for var_to_device.
        fn var_to_device(
            var: &<Self as kindle_core::prelude::Backend>::RawVar,
            _device: &kindle_core::prelude::KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(var.clone())
        }
        /// Auto-generated documentation for backward.
        fn backward<K: kindle_core::prelude::DType>(
            _loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            Err(Error::UnsupportedBackendOperation {
                op: "backward",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for backward_with_nan_check.
        fn backward_with_nan_check<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            Self::backward::<K>(loss)
        }

        /// Auto-generated documentation for get_grad.
        fn get_grad<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _grads: &<Self as kindle_core::prelude::Backend>::Grads,
        ) -> Result<Option<<Self as kindle_core::prelude::Backend>::Storage<K>>> {
            Ok(None)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::QuantizedOps<Self> for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for quantize.
        fn quantize<K: kindle_core::prelude::FloatDType, Q: kindle_core::prelude::QuantDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<Q>> {
            Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for dequantize.
        fn dequantize<Q: kindle_core::prelude::QuantDType, K: kindle_core::prelude::FloatDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "dequantize",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for quantized_matmul.
        fn quantized_matmul<Q: kindle_core::prelude::QuantDType>(
            _lhs: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
            _rhs: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<f32>> {
            Err(Error::UnsupportedBackendOperation {
                op: "quantized_matmul",
                backend: "Ndarray",
            })
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::CreationOps<Self> for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for zeros.
        fn zeros<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(ndarray::ArrayD::<f32>::zeros(shape))
        }
        /// Auto-generated documentation for ones.
        fn ones<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(ndarray::ArrayD::<f32>::ones(shape))
        }
        /// Auto-generated documentation for rand.
        fn rand<K: kindle_core::prelude::DType>(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "rand",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for randn.
        fn randn<K: kindle_core::prelude::DType>(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "randn",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for var_zeros.
        fn var_zeros<K: kindle_core::prelude::DType>(
            s: &[usize],
            _dt: KindleDType,
            _dev: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(NdarrayVar(alloc::sync::Arc::new(spin::RwLock::new(
                ndarray::ArrayD::<f32>::zeros(s),
            ))))
        }
        /// Auto-generated documentation for var_ones.
        fn var_ones<K: kindle_core::prelude::DType>(
            s: &[usize],
            _dt: KindleDType,
            _dev: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(NdarrayVar(alloc::sync::Arc::new(spin::RwLock::new(
                ndarray::ArrayD::<f32>::ones(s),
            ))))
        }
        /// Auto-generated documentation for var_rand.
        fn var_rand<K: kindle_core::prelude::DType>(
            _s: &[usize],
            _dt: KindleDType,
            _dev: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Err(Error::UnsupportedBackendOperation {
                op: "var_rand",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for var_randn.
        fn var_randn<K: kindle_core::prelude::DType>(
            _s: &[usize],
            _dt: KindleDType,
            _dev: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Err(Error::UnsupportedBackendOperation {
                op: "var_randn",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for tensor_to_device.
        fn tensor_to_device<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.clone())
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::NumericOps<Self> for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for add.
        fn add<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs + rhs)
        }
        /// Auto-generated documentation for sub.
        fn sub<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs - rhs)
        }
        /// Auto-generated documentation for mul.
        fn mul<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs * rhs)
        }
        /// Auto-generated documentation for div.
        fn div<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs / rhs)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::FloatOps<Self> for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for add_scalar_float.
        fn add_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mapv(|x| x + scalar as f32))
        }
        /// Auto-generated documentation for mul_scalar_float.
        fn mul_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mapv(|x| x * scalar as f32))
        }
        /// Auto-generated documentation for relu.
        fn relu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mapv(|x| if x > 0.0 { x } else { 0.0 }))
        }
        /// Auto-generated documentation for gelu.
        fn gelu<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "gelu",
                backend: "Ndarray",
            })
        }

        fn step<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("step not implemented for NdarrayBackend")
        }

        fn mish<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("mish not implemented for NdarrayBackend")
        }

        fn elu<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("elu not implemented for NdarrayBackend")
        }
        /// Auto-generated documentation for softmax.
        fn softmax<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "softmax",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for swish.
        fn swish<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "swish",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for abs.
        fn abs<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mapv(|x: f32| x.abs()))
        }
        /// Auto-generated documentation for neg.
        fn neg<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "neg",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for sqrt.
        fn sqrt<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sqrt",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for exp.
        fn exp<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "exp",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for log.
        fn log<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "log",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for tanh.
        fn tanh<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "tanh",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for sigmoid.
        fn sigmoid<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sigmoid",
                backend: "Ndarray",
            })
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ReductionOps<Self> for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for sum_all.
        fn sum_all<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_all",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for mean_all.
        fn mean_all<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_all",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for max_all.
        fn max_all<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_all",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for min_all.
        fn min_all<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_all",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for sum_dim.
        fn sum_dim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_dim",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for sum_keepdim.
        fn sum_keepdim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_keepdim",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for mean_dim.
        fn mean_dim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_dim",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for mean_keepdim.
        fn mean_keepdim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_keepdim",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for max_dim.
        fn max_dim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_dim",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for max_keepdim.
        fn max_keepdim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_keepdim",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for min_dim.
        fn min_dim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_dim",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for min_keepdim.
        fn min_keepdim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_keepdim",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for argmax.
        fn argmax<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            Err(Error::UnsupportedBackendOperation {
                op: "argmax",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for argmin.
        fn argmin<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            Err(Error::UnsupportedBackendOperation {
                op: "argmin",
                backend: "Ndarray",
            })
        }

        fn topk<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _k: usize,
            _dim: usize,
            _largest: bool,
        ) -> Result<(<Self as kindle_core::prelude::Backend>::Storage<K>, <Self as kindle_core::prelude::Backend>::Storage<KInt>)> {
            unimplemented!("topk not implemented for NdarrayBackend")
        }

        fn argsort<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: usize,
            _descending: bool,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            unimplemented!("argsort not implemented for NdarrayBackend")
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::TensorOps<Self> for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for matmul.
        fn matmul<K: kindle_core::prelude::DType>(
            _lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "matmul",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for stack.
        fn stack<K: kindle_core::prelude::DType>(
            _tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            _dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "stack",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for concat.
        fn concat<K: kindle_core::prelude::DType>(
            _tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            _dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "concat",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for broadcast_as.
        fn broadcast_as<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "broadcast_as",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for broadcast_left.
        fn broadcast_left<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "broadcast_left",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for reshape.
        fn reshape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            t.to_owned()
                .into_shape_with_order(shape)
                .map_err(|e: ndarray::ShapeError| anyhow::anyhow!(e).into())
        }
        /// Auto-generated documentation for transpose.
        fn transpose<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d1: usize,
            _d2: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "transpose",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for slice.
        fn slice<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            ranges: &[(usize, usize)],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let mut out = t.clone();
            for (dim, &(start, end)) in ranges.iter().enumerate() {
                out = out
                    .slice_axis(ndarray::Axis(dim), ndarray::Slice::from(start..end))
                    .to_owned();
            }
            Ok(out)
        }

        /// Auto-generated documentation for flatten.
        fn flatten<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _s: usize,
            _e: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "flatten",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for narrow.
        fn narrow<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: usize,
            _start: usize,
            _len: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "narrow",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for squeeze.
        fn squeeze<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "squeeze",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for float_to_scalar.
        fn float_to_scalar<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<f64> {
            Err(Error::UnsupportedBackendOperation {
                op: "float_to_scalar",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for float_to_vec1.
        fn float_to_vec1<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<Vec<f64>> {
            Err(Error::UnsupportedBackendOperation {
                op: "float_to_vec1",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for int_to_scalar.
        fn int_to_scalar<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<i64> {
            Err(Error::UnsupportedBackendOperation {
                op: "int_to_scalar",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for int_to_vec1.
        fn int_to_vec1<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<Vec<i64>> {
            Err(Error::UnsupportedBackendOperation {
                op: "int_to_vec1",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for tensor_to_dtype.
        fn tensor_to_dtype<K: kindle_core::prelude::DType, K2: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dtype: KindleDType,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K2>> {
            Err(Error::UnsupportedBackendOperation {
                op: "tensor_to_dtype",
                backend: "Ndarray",
            })
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ModuleOps<Self> for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for adaptive_avg_pool2d.
        fn adaptive_avg_pool2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _output_size: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("adaptive_avg_pool2d not implemented for NdarrayBackend")
        }

        /// Auto-generated documentation for layer_norm.
        fn layer_norm<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _w: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _b: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            _e: f32,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "layer_norm",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for batch_norm.
        fn batch_norm<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _w: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            _b: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            _rm: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            _rv: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            _e: f32,
            _momentum: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "batch_norm",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for embedding.
        fn embedding<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<KInt>,
            _w: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "embedding",
                backend: "Ndarray",
            })
        }
        /// Auto-generated documentation for conv2d.
        fn conv2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _w: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _b: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            _s: usize,
            _p: usize,
            _d: usize,
            _groups: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "conv2d",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for conv1d.
        fn conv1d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _w: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _b: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            _s: usize,
            _p: usize,
            _d: usize,
            _groups: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "conv1d",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for conv_transpose2d.
        fn conv_transpose2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _w: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _b: Option<&<Self as kindle_core::prelude::Backend>::Storage<K>>,
            _s: usize,
            _p: usize,
            _op: usize,
            _d: usize,
            _groups: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "conv_transpose2d",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for max_pool2d.
        fn max_pool2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _k: (usize, usize),
            _s: (usize, usize),
            _p: (usize, usize),
            _d: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_pool2d",
                backend: "Ndarray",
            })
        }

        /// Auto-generated documentation for avg_pool2d.
        fn avg_pool2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _k: (usize, usize),
            _s: (usize, usize),
            _p: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "avg_pool2d",
                backend: "Ndarray",
            })
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::LossOps<Self> for NdarrayBackend<T, D>
    {
        /// Auto-generated documentation for l1_loss.
        fn l1_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("l1_loss not implemented for NdArrayBackend")
        }

        /// Auto-generated documentation for bce_with_logits_loss.
        fn bce_with_logits_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("bce_with_logits_loss not implemented for NdarrayBackend")
        }

        /// Auto-generated documentation for mse_loss.
        fn mse_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("mse_loss not implemented for NdArrayBackend")
        }
        /// Auto-generated documentation for cross_entropy_loss.
        fn cross_entropy_loss<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<KInt>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "cross_entropy_loss",
                backend: "Ndarray",
            })
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::OptimizerOps<Self> for NdarrayBackend<T, D>
    {
    }
}

// ----------------------------------------------------------------------------
// BurnBackend
// ----------------------------------------------------------------------------

/// Auto-generated documentation for burn_backend.
#[cfg(any())]
pub mod burn_backend {
    use super::*;

    /// Auto-generated documentation for BurnBackend.
    pub struct BurnBackend<B: burn::tensor::backend::Backend>(core::marker::PhantomData<B>);

    macro_rules! impl_burn_backend {
        ($n:expr, $($D:ident),*) => {
            impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> Backend<($($D,)*)> for BurnBackend<B> {
                /// Auto-generated documentation for RawTensor.
                type RawTensor = burn::tensor::Tensor<B, $n>;
                /// Auto-generated documentation for RawVar.
                type RawVar = burn::tensor::Tensor<B, $n>;
                /// Auto-generated documentation for Grads.
                type Grads = (); // TODO: update when Burn supports this
                /// Auto-generated documentation for InnerBackend.
                type InnerBackend = Self;

                /// Auto-generated documentation for var_as_tensor.
                fn var_as_tensor(var: &<Self as kindle_core::prelude::Backend>::RawVar) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(var.clone())
                }

                /// Auto-generated documentation for tensor_to_device.
                fn tensor_to_device(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "tensor_to_device", backend: "Burn" }) }
                /// Auto-generated documentation for var_to_device.
                fn var_to_device(_var: &<Self as kindle_core::prelude::Backend>::RawVar, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_to_device", backend: "Burn" }) }
                /// Auto-generated documentation for to_dtype.
                fn to_dtype(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _dtype: KindleDType) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "to_dtype", backend: "Burn" }) }
                /// Auto-generated documentation for backward.
                fn backward(_loss: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::Grads> { Err(Error::UnsupportedBackendOperation { op: "backward", backend: "Burn" }) }
                /// Auto-generated documentation for get_grad.
                fn get_grad(_var: &<Self as kindle_core::prelude::Backend>::RawVar, _grads: &<Self as kindle_core::prelude::Backend>::Grads) -> Result<Option<<Self as kindle_core::prelude::Backend>::RawTensor>> { Ok(None) }

    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::CreationOps<Self> for BurnBackend<B> {
                /// Auto-generated documentation for zeros.
                fn zeros(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    let d: [usize; $n] = shape.try_into().map_err(|_| Error::ShapeMismatch { expected: alloc::vec![$n], got: shape.to_vec() })?;
                    Ok(burn::tensor::Tensor::<B, $n>::zeros(d, &B::Device::default()))
                }
                /// Auto-generated documentation for ones.
                fn ones(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    let d: [usize; $n] = shape.try_into().map_err(|_| Error::ShapeMismatch { expected: alloc::vec![$n], got: shape.to_vec() })?;
                    Ok(burn::tensor::Tensor::<B, $n>::ones(d, &B::Device::default()))
                }
                /// Auto-generated documentation for rand.
                fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "rand", backend: "Burn" })
                }
                /// Auto-generated documentation for randn.
                fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "randn", backend: "Burn" })
                }

                /// Auto-generated documentation for var_zeros.
                fn var_zeros(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_zeros", backend: "Burn" }) }
                /// Auto-generated documentation for var_ones.
                fn var_ones(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_ones", backend: "Burn" }) }
                /// Auto-generated documentation for var_rand.
                fn var_rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_rand", backend: "Burn" }) }
                /// Auto-generated documentation for var_randn.
                fn var_randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_randn", backend: "Burn" }) }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::NumericOps<Self> for BurnBackend<B> {
                /// Auto-generated documentation for mul_scalar.
                fn mul_scalar(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _scalar: kindle_core::prelude::ScalarValue) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mul_scalar", backend: "Burn" }) }
                /// Auto-generated documentation for add_scalar.
                fn add_scalar(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _scalar: kindle_core::prelude::ScalarValue) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "add_scalar", backend: "Burn" }) }
                /// Auto-generated documentation for add.
                fn add(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone() + rhs.clone())
                }
                /// Auto-generated documentation for sub.
                fn sub(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone() - rhs.clone())
                }
                /// Auto-generated documentation for mul.
                fn mul(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone() * rhs.clone())
                }
                /// Auto-generated documentation for div.
                fn div(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone() / rhs.clone())
                }
                /// Auto-generated documentation for matmul.
                fn matmul(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone().matmul(rhs.clone()))
                }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::FloatOps<Self> for BurnBackend<B> {
                /// Auto-generated documentation for relu.
                fn relu(t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(burn::tensor::activation::relu(t.clone()))
                }
                /// Auto-generated documentation for gelu.
                fn gelu(t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(burn::tensor::activation::gelu(t.clone()))
                }
                /// Auto-generated documentation for softmax.
                fn softmax(t: &<Self as kindle_core::prelude::Backend>::RawTensor, dim: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(burn::tensor::activation::softmax(t.clone(), dim))
                }
                /// Auto-generated documentation for swish.
                fn swish(t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(burn::tensor::activation::silu(t.clone()))
                }
                /// Auto-generated documentation for abs.
                fn abs(t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(t.clone().abs())
                }
                /// Auto-generated documentation for neg.
                fn neg(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "neg", backend: "Burn" }) }
                /// Auto-generated documentation for sqrt.
                fn sqrt(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sqrt", backend: "Burn" }) }
                /// Auto-generated documentation for exp.
                fn exp(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "exp", backend: "Burn" }) }
                /// Auto-generated documentation for log.
                fn log(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "log", backend: "Burn" }) }
                /// Auto-generated documentation for tanh.
                fn tanh(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "tanh", backend: "Burn" }) }
                /// Auto-generated documentation for sigmoid.
                fn sigmoid(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sigmoid", backend: "Burn" }) }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::ReductionOps<Self> for BurnBackend<B> {
                /// Auto-generated documentation for sum_all.
                fn sum_all(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_all", backend: "Burn" }) }
                /// Auto-generated documentation for mean_all.
                fn mean_all(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_all", backend: "Burn" }) }
                /// Auto-generated documentation for max_all.
                fn max_all(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_all", backend: "Burn" }) }
                /// Auto-generated documentation for min_all.
                fn min_all(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_all", backend: "Burn" }) }
                /// Auto-generated documentation for sum_dim.
                fn sum_dim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_dim", backend: "Burn" }) }
                /// Auto-generated documentation for sum_keepdim.
                fn sum_keepdim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_keepdim", backend: "Burn" }) }
                /// Auto-generated documentation for mean_dim.
                fn mean_dim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_dim", backend: "Burn" }) }
                /// Auto-generated documentation for mean_keepdim.
                fn mean_keepdim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_keepdim", backend: "Burn" }) }
                /// Auto-generated documentation for max_dim.
                fn max_dim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_dim", backend: "Burn" }) }
                /// Auto-generated documentation for max_keepdim.
                fn max_keepdim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_keepdim", backend: "Burn" }) }
                /// Auto-generated documentation for min_dim.
                fn min_dim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_dim", backend: "Burn" }) }
                /// Auto-generated documentation for min_keepdim.
                fn min_keepdim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_keepdim", backend: "Burn" }) }
                /// Auto-generated documentation for argmax.
                fn argmax(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "argmax", backend: "Burn" }) }
                /// Auto-generated documentation for argmin.
                fn argmin(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "argmin", backend: "Burn" }) }

    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::TensorOps<Self> for BurnBackend<B> {
                /// Auto-generated documentation for stack.
                fn stack(_tensors: &[&<Self as kindle_core::prelude::Backend>::RawTensor], _dim: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "stack", backend: "Burn" }) }
                /// Auto-generated documentation for concat.
                fn concat(_tensors: &[&<Self as kindle_core::prelude::Backend>::RawTensor], _dim: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "concat", backend: "Burn" }) }
                /// Auto-generated documentation for broadcast_as.
                fn broadcast_as(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _shape: &[usize]) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "broadcast_as", backend: "Burn" }) }
                /// Auto-generated documentation for broadcast_left.
                fn broadcast_left(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _shape: &[usize]) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "broadcast_left", backend: "Burn" }) }
                /// Auto-generated documentation for reshape.
                fn reshape(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _shape: &[usize]) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "reshape", backend: "Burn" })
                }
                /// Auto-generated documentation for transpose.
                fn transpose(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d1: usize, _d2: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "transpose", backend: "Burn" }) }
                /// Auto-generated documentation for flatten.
                fn flatten(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _s: usize, _e: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "flatten", backend: "Burn" }) }
                /// Auto-generated documentation for narrow.
                fn narrow(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _dim: usize, _start: usize, _len: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "narrow", backend: "Burn" }) }
                /// Auto-generated documentation for squeeze.
                fn squeeze(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _dim: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "squeeze", backend: "Burn" }) }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::ModuleOps<Self> for BurnBackend<B> {
                /// Auto-generated documentation for layer_norm.
                fn layer_norm(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _w: &<Self as kindle_core::prelude::Backend>::RawTensor, _b: &<Self as kindle_core::prelude::Backend>::RawTensor, _e: f32) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "layer_norm", backend: "Burn" }) }
                /// Auto-generated documentation for batch_norm.
                fn batch_norm(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _w: &<Self as kindle_core::prelude::Backend>::RawTensor, _b: &<Self as kindle_core::prelude::Backend>::RawTensor, _rm: &<Self as kindle_core::prelude::Backend>::RawTensor, _rv: &<Self as kindle_core::prelude::Backend>::RawTensor, _e: f32) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "batch_norm", backend: "Burn" }) }
                /// Auto-generated documentation for embedding.
                fn embedding(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _w: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "embedding", backend: "Burn" }) }
                /// Auto-generated documentation for conv2d.
                fn conv2d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: &<Self as kindle_core::prelude::Backend>::RawTensor, _: Option<&<Self as kindle_core::prelude::Backend>::RawTensor>, _: usize, _: usize, _: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv2d", backend: "Burn" })
                }
                /// Auto-generated documentation for conv1d.
                fn conv1d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: &<Self as kindle_core::prelude::Backend>::RawTensor, _: Option<&<Self as kindle_core::prelude::Backend>::RawTensor>, _: usize, _: usize, _: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv1d", backend: "Burn" })
                }
                /// Auto-generated documentation for conv_transpose2d.
                fn conv_transpose2d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: &<Self as kindle_core::prelude::Backend>::RawTensor, _: Option<&<Self as kindle_core::prelude::Backend>::RawTensor>, _: usize, _: usize, _: usize, _: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv_transpose2d", backend: "Burn" })
                }
                /// Auto-generated documentation for max_pool2d.
                fn max_pool2d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: (usize, usize), _: (usize, usize)) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "max_pool2d", backend: "Burn" })
                }
                /// Auto-generated documentation for avg_pool2d.
                fn avg_pool2d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: (usize, usize), _: (usize, usize)) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "avg_pool2d", backend: "Burn" })
                }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::LossOps<Self> for BurnBackend<B> {
                /// Auto-generated documentation for l1_loss.
                fn l1_loss(_pred: &<Self as kindle_core::prelude::Backend>::RawTensor, _target: &<Self as kindle_core::prelude::Backend>::RawTensor, _reduction: kindle_core::prelude::Reduction) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "l1_loss", backend: "Burn" }) }
                /// Auto-generated documentation for bce_with_logits_loss.
                fn bce_with_logits_loss(_pred: &<Self as kindle_core::prelude::Backend>::RawTensor, _target: &<Self as kindle_core::prelude::Backend>::RawTensor, _reduction: kindle_core::prelude::Reduction) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "bce_with_logits_loss", backend: "Burn" }) }
                /// Auto-generated documentation for mse_loss.
                fn mse_loss(_pred: &<Self as kindle_core::prelude::Backend>::RawTensor, _target: &<Self as kindle_core::prelude::Backend>::RawTensor, _reduction: kindle_core::prelude::Reduction) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mse_loss", backend: "Burn" }) }
                /// Auto-generated documentation for cross_entropy_loss.
                fn cross_entropy_loss(_pred: &<Self as kindle_core::prelude::Backend>::RawTensor, _target: &<Self as kindle_core::prelude::Backend>::RawTensor, _reduction: kindle_core::prelude::Reduction) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "cross_entropy_loss", backend: "Burn" }) }

    }


        };
    }

    impl_burn_backend!(1, D0);
    impl_burn_backend!(2, D0, D1);
    impl_burn_backend!(3, D0, D1, D2);
    impl_burn_backend!(4, D0, D1, D2, D3);
    impl_burn_backend!(5, D0, D1, D2, D3, D4);
}
