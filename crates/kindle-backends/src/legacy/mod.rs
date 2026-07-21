// ----------------------------------------------------------------------------
// Legacy Wrappers (Burn, Candle, Ndarray)
// ----------------------------------------------------------------------------

pub use kindle_core::prelude::*;

// ----------------------------------------------------------------------------
// CandleBackend
// ----------------------------------------------------------------------------

/// `candle`.
pub mod candle {
    use super::*;
    use candle_core as candle;

    /// # Backend Float Element Limitation (B-4)
    /// **Known Limitation:** `CandleBackend` ignores its compile-time `T` generic
    /// for inner allocation precision and relies on the dynamic `KindleDType`
    /// supplied to creation methods.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CandleBackend<T, D>(core::marker::PhantomData<(T, D)>);

    /// `to_candle_device`.
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

    /// `to_candle_dtype`.
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
        /// `Device`.
        type Device = D;
        /// `FloatElem`.
        type FloatElem = T;
        /// `IntElem`.
        type IntElem = i64;
        /// `BackendWithDevice`.
        type BackendWithDevice<NewD: kindle_core::prelude::Device> = CandleBackend<T, NewD>;

        /// `Storage`.
        type Storage<K: kindle_core::prelude::DType> = candle_core::Tensor;
        /// `RawVar`.
        type RawVar = candle_core::Var;
        /// `Grads`.
        type Grads = candle_core::backprop::GradStore;
        /// `InnerBackend`.
        type InnerBackend = Self;

        /// `shape`.
        fn shape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Vec<usize> {
            t.dims().to_vec()
        }

        /// `format_tensor_display`.
        fn format_tensor_display<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("{}", t)
        }
        /// `format_tensor_debug`.
        fn format_tensor_debug<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("Raw Tensor: {:?}, Strides: {:?}", t, t.stride())
        }

        /// `var_as_tensor`.
        fn var_as_tensor<K: kindle_core::prelude::DType>(
            var: &<Self as kindle_core::prelude::Backend>::RawVar,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(var.as_tensor().clone())
        }
        /// `var_from_tensor`.
        fn var_from_tensor<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(candle::Var::from_tensor(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `var_to_device`.
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

        /// `assign_var`.
        fn assign_var<K: kindle_core::prelude::DType>(
            var: &mut <Self as kindle_core::prelude::Backend>::RawVar,
            tensor: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<()> {
            var.set(tensor)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        /// `backward`.
        fn backward<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            loss.backward()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        /// `backward_with_nan_check`.
        fn backward_with_nan_check<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            Self::backward::<K>(loss)
        }

        /// `get_grad`.
        fn get_grad<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            grads: &<Self as kindle_core::prelude::Backend>::Grads,
        ) -> Result<Option<<Self as kindle_core::prelude::Backend>::Storage<K>>> {
            Ok(grads.get(t).cloned())
        }

        /// `to_bytes`.
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

        /// `from_bytes`.
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
        /// `quantize`.
        fn quantize<K: kindle_core::prelude::FloatDType, Q: kindle_core::prelude::QuantDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<Q>> {
            Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Candle",
            })
        }
        /// `dequantize`.
        fn dequantize<Q: kindle_core::prelude::QuantDType, K: kindle_core::prelude::FloatDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "dequantize",
                backend: "Candle",
            })
        }
        /// `quantized_matmul`.
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
        /// `zeros`.
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

        /// `ones`.
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

        /// `rand`.
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

        /// `randn`.
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

        /// `var_zeros`.
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

        /// `var_ones`.
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

        /// `var_rand`.
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

        /// `var_randn`.
        fn var_randn<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            let dev = to_candle_device(device)?;
            Ok(candle::Var::randn(0f32, 1f32, shape, &dev)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `tensor_to_device`.
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
        /// `add`.
        fn add<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_add(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `sub`.
        fn sub<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_sub(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `mul`.
        fn mul<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_mul(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `div`.
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
        /// `matmul`.
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

        /// `stack`.
        fn stack<K: kindle_core::prelude::DType>(
            tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_core::Tensor::stack(tensors, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `concat`.
        fn concat<K: kindle_core::prelude::DType>(
            tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_core::Tensor::cat(tensors, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `broadcast_as`.
        fn broadcast_as<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.broadcast_as(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `broadcast_left`.
        fn broadcast_left<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.broadcast_left(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `reshape`.
        fn reshape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.reshape(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `transpose`.
        fn transpose<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim1: usize,
            dim2: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.transpose(dim1, dim2)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `slice`.
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

        /// `flatten`.
        fn flatten<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            start_dim: usize,
            end_dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.flatten(start_dim, end_dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `narrow`.
        fn narrow<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
            start: usize,
            len: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.narrow(dim, start, len)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `squeeze`.
        fn squeeze<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.squeeze(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `float_to_scalar`.
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
        /// `float_to_vec1`.
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
        /// `int_to_scalar`.
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
        /// `int_to_vec1`.
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
        /// `tensor_to_dtype`.
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
        /// `add_scalar_float`.
        fn add_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok((t + scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `mul_scalar_float`.
        fn mul_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok((t * scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `relu`.
        fn relu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.relu()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `gelu`.
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

        /// `softmax`.
        fn softmax<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_nn::ops::softmax(t, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `swish`.
        fn swish<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            // swish is x * sigmoid(x)
            Ok(candle_nn::ops::silu(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `abs`.
        fn abs<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.abs()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `neg`.
        fn neg<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.neg()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `sqrt`.
        fn sqrt<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sqrt()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `exp`.
        fn exp<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.exp()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `log`.
        fn log<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.log()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `tanh`.
        fn tanh<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.tanh()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `sigmoid`.
        fn sigmoid<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(::candle_nn::ops::sigmoid(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ReductionOps<Self> for CandleBackend<T, D>
    {
        /// `sum_all`.
        fn sum_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `mean_all`.
        fn mean_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `max_all`.
        fn max_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `min_all`.
        fn min_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `sum_dim`.
        fn sum_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `sum_keepdim`.
        fn sum_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `mean_dim`.
        fn mean_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `mean_keepdim`.
        fn mean_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `max_dim`.
        fn max_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `max_keepdim`.
        fn max_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `min_dim`.
        fn min_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// `min_keepdim`.
        fn min_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// `argmax`.
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

        /// `argmin`.
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
        ) -> Result<(
            <Self as kindle_core::prelude::Backend>::Storage<K>,
            <Self as kindle_core::prelude::Backend>::Storage<KInt>,
        )> {
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
        /// `adaptive_avg_pool2d`.
        fn adaptive_avg_pool2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _output_size: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("adaptive_avg_pool2d not implemented for CandleBackend")
        }

        /// `layer_norm`.
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

        /// `batch_norm`.
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

        /// `embedding`.
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

        /// `conv1d`.
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

        /// `conv2d`.
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

        /// `conv_transpose2d`.
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

        /// `max_pool2d`.
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

        /// `avg_pool2d`.
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
        /// `l1_loss`.
        fn l1_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("l1_loss not implemented for CandleBackend")
        }

        /// `bce_with_logits_loss`.
        fn bce_with_logits_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("bce_with_logits_loss not implemented for CandleBackend")
        }

        /// `mse_loss`.
        fn mse_loss<K: kindle_core::prelude::DType>(
            pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let loss = candle_nn::loss::mse(pred, target)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(loss)
        }

        /// `cross_entropy_loss`.
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

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::OptimizerOps<Self> for CandleBackend<T, D>
    {
    }

    #[cfg(test)]
    /// `tests`.
    mod tests {
        use super::*;

        #[test]
        /// `test_to_candle_dtype`.
        fn test_to_candle_dtype() {
            assert_eq!(to_candle_dtype(KindleDType::F32), candle::DType::F32);
            assert_eq!(to_candle_dtype(KindleDType::I64), candle::DType::I64);
        }

        #[test]
        /// `test_to_candle_device`.
        fn test_to_candle_device() {
            let cpu = KindleDevice::cpu();
            let c_dev = to_candle_device(&cpu).unwrap();
            assert!(matches!(c_dev, candle::Device::Cpu));
        }
    }
}
