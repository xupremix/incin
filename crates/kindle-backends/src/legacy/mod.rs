// ----------------------------------------------------------------------------
// Legacy Wrappers (Burn, Candle, Ndarray)
// ----------------------------------------------------------------------------

pub use kindle_core::prelude::*;

// ----------------------------------------------------------------------------
// CandleBackend
// ----------------------------------------------------------------------------

/// Wraps the `candle_core` crate, providing `CandleBackend` as a `Backend`
/// implementation backed by Candle's own tensor type.
pub mod candle {
    use super::*;
    use candle_core as candle;

    /// # Backend Float Element Limitation (B-4)
    /// **Known Limitation:** `CandleBackend` ignores its compile-time `T` generic
    /// for inner allocation precision and relies on the dynamic `DTypeId`
    /// supplied to creation methods.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CandleBackend<T, D>(core::marker::PhantomData<(T, D)>);

    /// Converts a kindle `DeviceId` into a candle `Device`, mapping CPU/CUDA/wgpu
    /// device kinds and erroring on any device kind Candle doesn't support.
    pub fn to_candle_device(dev: &DeviceId) -> Result<candle::Device> {
        use kindle_core::prelude::DeviceKind;
        match dev.kind() {
            DeviceKind::Cpu => Ok(candle::Device::Cpu),
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => Ok(candle::Device::new_cuda(dev.ordinal())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            #[cfg(feature = "wgpu")]
            DeviceKind::Wgpu => Ok(candle::Device::new_metal(dev.ordinal())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            _ => Err(Error::UnsupportedBackendOperation {
                op: "to_candle_device",
                backend: "Candle",
            }),
        }
    }

    /// Maps a kindle `DTypeId` to the corresponding candle `DType`, panicking on
    /// dtypes Candle has no native representation for (e.g. `Q8_0`).
    pub fn to_candle_dtype(dtype: DTypeId) -> candle::DType {
        match dtype {
            DTypeId::U8 => candle::DType::U8,
            DTypeId::U32 => candle::DType::U32,
            DTypeId::I64 => candle::DType::I64,
            DTypeId::BF16 => candle::DType::BF16,
            DTypeId::F16 => candle::DType::F16,
            DTypeId::F32 => candle::DType::F32,
            DTypeId::F64 => candle::DType::F64,
            DTypeId::Q8_0 => unimplemented!("Q8_0 is not natively supported in candle yet"),
            _ => unimplemented!("Unsupported dtype in candle"),
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::Backend for CandleBackend<T, D>
    {
        /// The device type, forwarded unchanged from the `D` generic parameter.
        type Device = D;
        /// The floating-point element type, forwarded unchanged from the `T`
        /// generic parameter.
        type FloatElem = T;
        /// Integer elements are always represented as `i64`, regardless of `T`.
        type IntElem = i64;
        /// Tensor storage is a raw `candle_core::Tensor`; the `K` dtype marker
        /// is not reflected in the storage type itself.
        type Storage<K: kindle_core::prelude::DType> = candle_core::Tensor;
        /// A trainable variable is backed by candle's `Var`.
        type RawVar = candle_core::Var;
        /// Gradients are accumulated in candle's `GradStore`, keyed by tensor.
        type Grads = candle_core::backprop::GradStore;
        /// `CandleBackend` has no further inner-backend indirection; it is its
        /// own inner backend.
        type InnerBackend = Self;

        /// Returns the tensor's dimensions as a `Vec<usize>`.
        fn shape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Vec<usize> {
            t.dims().to_vec()
        }

        /// Formats the tensor using candle's own `Display` implementation.
        fn format_tensor_display<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("{}", t)
        }
        /// Formats the tensor's raw contents together with its strides, for
        /// debugging.
        fn format_tensor_debug<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("Raw Tensor: {:?}, Strides: {:?}", t, t.stride())
        }

        /// Clones the variable's underlying tensor out as plain storage.
        fn var_as_tensor<K: kindle_core::prelude::DType>(
            var: &<Self as kindle_core::prelude::Backend>::RawVar,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(var.as_tensor().clone())
        }
        /// Wraps a tensor in a new candle `Var`, cloning its data.
        fn var_from_tensor<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(candle::Var::from_tensor(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Overwrites the variable's contents in place with `tensor`.
        fn assign_var<K: kindle_core::prelude::DType>(
            var: &mut <Self as kindle_core::prelude::Backend>::RawVar,
            tensor: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<()> {
            var.set(tensor)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        /// Runs backpropagation from `loss`, returning the resulting gradient
        /// store.
        fn backward<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            loss.backward()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        /// Identical to `backward`; candle has no separate NaN-checking
        /// backward pass, so this simply delegates.
        fn backward_with_nan_check<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            Self::backward::<K>(loss)
        }

        /// Looks up the accumulated gradient for `t` in `grads`, if one was
        /// recorded during backward.
        fn get_grad<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            grads: &<Self as kindle_core::prelude::Backend>::Grads,
        ) -> Result<Option<<Self as kindle_core::prelude::Backend>::Storage<K>>> {
            Ok(grads.get(t).cloned())
        }

        /// Flattens the tensor, casts it to `f32`, and returns its raw byte
        /// representation.
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

        /// Reinterprets `bytes` as `f32` values, builds a tensor with `shape`
        /// on `device`, and casts the result to `dtype`.
        fn from_bytes<K: kindle_core::prelude::DType>(
            bytes: &[u8],
            shape: &[usize],
            dtype: DTypeId,
            device: &DeviceId,
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

    impl<
        T: kindle_core::prelude::DType,
        D: kindle_core::prelude::Device,
        K: kindle_core::prelude::DType,
    > kindle_core::prelude::SupportsDType<K> for CandleBackend<T, D>
    {
    }

    impl<T, D, NewD> kindle_core::prelude::TransferTo<NewD> for CandleBackend<T, D>
    where
        T: kindle_core::prelude::DType,
        D: kindle_core::prelude::Device,
        NewD: kindle_core::prelude::Device,
    {
        type Output = CandleBackend<T, NewD>;

        fn transfer_storage<K: kindle_core::prelude::DType>(
            storage: &Self::Storage<K>,
            dtype: &K::Field,
            device: &NewD::Field,
        ) -> Result<<Self::Output as Backend>::Storage<K>>
        where
            Self::Output: SupportsDType<K>,
        {
            let destination = NewD::to_kindle(device)?;
            <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &destination)?;
            let target = to_candle_device(&destination)?;
            storage
                .to_device(&target)
                .map_err(|error| anyhow::anyhow!(error).into())
        }

        fn transfer_var(
            variable: &Self::RawVar,
            dtype: &<T as kindle_core::prelude::DType>::Field,
            device: &NewD::Field,
        ) -> Result<<Self::Output as Backend>::RawVar>
        where
            Self::Output: SupportsDType<T>,
        {
            let storage = <Self as Backend>::var_as_tensor::<T>(variable)?;
            let transferred = Self::transfer_storage(&storage, dtype, device)?;
            <Self::Output as Backend>::var_from_tensor::<T>(&transferred)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::QuantizedOps<Self> for CandleBackend<T, D>
    {
        /// Not supported by candle; always returns `UnsupportedBackendOperation`.
        fn quantize<K: kindle_core::prelude::FloatDType, Q: kindle_core::prelude::QuantDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<Q>> {
            Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Candle",
            })
        }
        /// Not supported by candle; always returns `UnsupportedBackendOperation`.
        fn dequantize<Q: kindle_core::prelude::QuantDType, K: kindle_core::prelude::FloatDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "dequantize",
                backend: "Candle",
            })
        }
        /// Not supported by candle; always returns `UnsupportedBackendOperation`.
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
        /// Allocates a tensor of `shape` filled with zeros on `device` with the
        /// given dtype.
        fn zeros<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: DTypeId,
            device: &DeviceId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                candle::Tensor::zeros(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Allocates a tensor of `shape` filled with ones on `device` with the
        /// given dtype.
        fn ones<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: DTypeId,
            device: &DeviceId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                candle::Tensor::ones(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Samples a uniform `[0, 1)` tensor of `shape` on `device`, then casts
        /// it to `dtype`.
        fn rand<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: DTypeId,
            device: &DeviceId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                candle::Tensor::rand(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                    .to_dtype(to_candle_dtype(dtype))
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Samples a standard-normal tensor of `shape` on `device`, then casts
        /// it to `dtype`.
        fn randn<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: DTypeId,
            device: &DeviceId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(
                candle::Tensor::randn(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                    .to_dtype(to_candle_dtype(dtype))
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Allocates a zero-initialized trainable `Var` of `shape` on `device`.
        fn var_zeros<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: DTypeId,
            device: &DeviceId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(
                candle::Var::zeros(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Allocates a one-initialized trainable `Var` of `shape` on `device`.
        fn var_ones<K: kindle_core::prelude::DType>(
            shape: &[usize],
            dtype: DTypeId,
            device: &DeviceId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(
                candle::Var::ones(shape, to_candle_dtype(dtype), &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Allocates a trainable `Var` of `shape` on `device`, sampled from a
        /// uniform `[0, 1)` distribution.
        fn var_rand<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: DTypeId,
            device: &DeviceId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(
                candle::Var::rand(0f32, 1f32, shape, &to_candle_device(device)?)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
            )
        }

        /// Allocates a trainable `Var` of `shape` on `device`, sampled from a
        /// standard-normal distribution.
        fn var_randn<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: DTypeId,
            device: &DeviceId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            let dev = to_candle_device(device)?;
            Ok(candle::Var::randn(0f32, 1f32, shape, &dev)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::NumericOps<Self> for CandleBackend<T, D>
    {
        /// Element-wise addition with broadcasting.
        fn add<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_add(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Element-wise subtraction with broadcasting.
        fn sub<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_sub(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Element-wise multiplication with broadcasting.
        fn mul<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_mul(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Element-wise division with broadcasting.
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
        /// Matrix-multiplies `lhs` and `rhs`. For operands with more than 3
        /// dimensions (which candle's `broadcast_matmul` can't handle directly),
        /// manually broadcasts the leading batch dimensions, flattens them into
        /// a single batch axis, multiplies, and reshapes back.
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

        /// Stacks `tensors` along a new dimension `dim`.
        fn stack<K: kindle_core::prelude::DType>(
            tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_core::Tensor::stack(tensors, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Concatenates `tensors` along an existing dimension `dim`.
        fn concat<K: kindle_core::prelude::DType>(
            tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_core::Tensor::cat(tensors, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Broadcasts `t` to `shape`.
        fn broadcast_as<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.broadcast_as(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Broadcasts `t` by prepending dimensions from `shape` on the left.
        fn broadcast_left<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.broadcast_left(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Reshapes `t` to `shape` without changing its underlying data.
        fn reshape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.reshape(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Swaps dimensions `dim1` and `dim2`.
        fn transpose<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim1: usize,
            dim2: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.transpose(dim1, dim2)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies a per-dimension `[start, end)` narrow for each entry in
        /// `ranges`, sequentially, one dimension at a time.
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

        /// Flattens dimensions `start_dim..=end_dim` into a single dimension.
        fn flatten<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            start_dim: usize,
            end_dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.flatten(start_dim, end_dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Takes a contiguous sub-range of length `len` starting at `start`
        /// along `dim`.
        fn narrow<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
            start: usize,
            len: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.narrow(dim, start, len)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Removes dimension `dim` if it has size 1.
        fn squeeze<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.squeeze(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Casts `t` to `f32` and extracts its single element as an `f64`
        /// scalar.
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
        /// Casts `t` to `f32` and collects it into a flat `Vec<f64>`.
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
        /// Casts `t` to `i64` and extracts its single element.
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
        /// Casts `t` to `i64` and collects it into a flat `Vec<i64>`.
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
        /// Casts `t` to the candle dtype corresponding to `dtype`.
        fn tensor_to_dtype<K: kindle_core::prelude::DType, K2: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dtype: DTypeId,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K2>> {
            Ok(t.to_dtype(to_candle_dtype(dtype))
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::FloatOps<Self> for CandleBackend<T, D>
    {
        /// Adds a scalar to every element of `t`.
        fn add_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok((t + scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Multiplies every element of `t` by a scalar.
        fn mul_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok((t * scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies the ReLU activation element-wise.
        fn relu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.relu()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies GELU using candle's exact erf-based formulation
        /// (`gelu_erf`), used here as the general-purpose GELU.
        fn gelu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.gelu_erf()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        } // using gelu_erf as fallback for general

        /// Implements Heaviside step function: H(x) = 0 if x < 0, else 1.
        /// Computed as: mask = (x >= 0), cast mask to float.
        fn step<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            // (t >= 0.0) gives a bool mask; cast to float dtype
            let zero = t
                .zeros_like()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let mask = t
                .ge(&zero)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(mask
                .to_dtype(t.dtype())
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Implements Mish activation: `x * tanh(softplus(x))`
        /// where `softplus(x) = ln(1 + exp(x))`.
        fn mish<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            // softplus(x) = ln(1 + exp(x))
            let exp_x = t
                .exp()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let softplus = (exp_x + 1.0f64)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .log()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            // mish(x) = x * tanh(softplus(x))
            let tanh_sp = softplus
                .tanh()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(t.broadcast_mul(&tanh_sp)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Implements Exponential Linear Unit: ELU(x) = x if x >= 0, else 1*(e^x - 1).
        fn elu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.elu(1.0f64)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Applies softmax along `dim`.
        fn softmax<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_nn::ops::softmax(t, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Applies the swish/SiLU activation (`x * sigmoid(x)`) via candle's
        /// `silu` op.
        fn swish<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            // swish is x * sigmoid(x)
            Ok(candle_nn::ops::silu(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies absolute value element-wise.
        fn abs<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.abs()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Negates every element of `t`.
        fn neg<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.neg()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies element-wise square root.
        fn sqrt<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sqrt()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies element-wise exponential.
        fn exp<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.exp()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies element-wise natural logarithm.
        fn log<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.log()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies element-wise hyperbolic tangent.
        fn tanh<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.tanh()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Applies the sigmoid activation element-wise.
        fn sigmoid<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(::candle_nn::ops::sigmoid(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ReductionOps<Self> for CandleBackend<T, D>
    {
        /// Sums all elements into a scalar tensor.
        fn sum_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Averages all elements into a scalar tensor.
        fn mean_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Reduces to the maximum element as a scalar tensor.
        fn max_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Reduces to the minimum element as a scalar tensor.
        fn min_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Sums along `dim`, removing it from the shape.
        fn sum_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Sums along `dim`, keeping it as a size-1 dimension.
        fn sum_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Averages along `dim`, removing it from the shape.
        fn mean_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Averages along `dim`, keeping it as a size-1 dimension.
        fn mean_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Reduces to the maximum along `dim`, removing it from the shape.
        fn max_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Reduces to the maximum along `dim`, keeping it as a size-1
        /// dimension.
        fn max_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Reduces to the minimum along `dim`, removing it from the shape.
        fn min_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        /// Reduces to the minimum along `dim`, keeping it as a size-1
        /// dimension.
        fn min_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        /// Returns the index of the maximum element, along `dim` if given,
        /// otherwise over the flattened tensor.
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

        /// Returns the index of the minimum element, along `dim` if given,
        /// otherwise over the flattened tensor.
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

        /// `topk` is not natively available in candle; returns an error
        /// instead of panicking so callers can handle the unsupported case.
        fn topk<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _k: usize,
            _dim: usize,
            _largest: bool,
        ) -> Result<(
            <Self as kindle_core::prelude::Backend>::Storage<K>,
            <Self as kindle_core::prelude::Backend>::Storage<KInt>,
        )> {
            Err(Error::UnsupportedBackendOperation {
                op: "topk",
                backend: "CandleBackend",
            })
        }

        /// Sorts indices along the last dimension using candle's native
        /// `argsort_last_dim`. For non-last dimensions, transposes to last
        /// and back.
        fn argsort<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
            descending: bool,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            let rank = t.rank();
            let last = rank.saturating_sub(1);
            // Candle's arg_sort_last_dim takes `asc: bool`; our API takes `descending: bool`.
            let asc = !descending;
            if dim == last {
                Ok(t.arg_sort_last_dim(asc)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
            } else {
                // Transpose target dim to last, sort, transpose back.
                // arg_sort_last_dim requires a contiguous tensor, so make it so.
                let t_swap = t
                    .transpose(dim, last)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                    .contiguous()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                let sorted = t_swap
                    .arg_sort_last_dim(asc)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
                Ok(sorted
                    .transpose(dim, last)
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
            }
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ModuleOps<Self> for CandleBackend<T, D>
    {
        /// Candle has no native adaptive pooling; returns an error
        /// instead of panicking so callers can handle the unsupported case.
        fn adaptive_avg_pool2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _output_size: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "adaptive_avg_pool2d",
                backend: "CandleBackend",
            })
        }

        /// Applies layer normalization over the last dimension, substituting a
        /// zero bias when none is provided.
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

        /// Applies batch normalization using the running mean/variance (or
        /// defaults of 0/1 when not provided) and an optional affine
        /// weight/bias, reshaping all of them to broadcast over the channel
        /// dimension.
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

        /// Looks up rows of embedding table `w` for each index in `t`, first
        /// casting indices to `U32` if they aren't already `U32`/`I64` (candle
        /// requires one of those two dtypes for embedding lookups).
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

        /// 1-D convolution of `t` with kernel `w`; the bias argument is
        /// ignored.
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

        /// 2-D convolution of `t` with kernel `weight`; the bias argument is
        /// ignored.
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

        /// 2-D transposed convolution of `t` with kernel `weight`; the bias
        /// and groups arguments are ignored.
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

        /// 2-D max pooling with the given kernel size and stride; padding and
        /// dilation are ignored (not supported by candle's pooling op).
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

        /// 2-D average pooling with the given kernel size and stride; padding
        /// is ignored (not supported by candle's pooling op).
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
        /// Computes L1 (Mean Absolute Error) loss: `|pred - target|` with
        /// the given `reduction` (Mean, Sum, or None).
        fn l1_loss<K: kindle_core::prelude::DType>(
            pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let diff = pred
                .broadcast_sub(target)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let abs_diff = diff
                .abs()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            match reduction {
                kindle_core::prelude::Reduction::Mean => Ok(abs_diff
                    .mean_all()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
                kindle_core::prelude::Reduction::Sum => Ok(abs_diff
                    .sum_all()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
                kindle_core::prelude::Reduction::None => Ok(abs_diff),
            }
        }

        /// Computes Binary Cross-Entropy from logits:
        /// `max(x, 0) - x*y + log(1 + exp(-|x|))`
        /// with the given `reduction` (Mean, Sum, or None).
        fn bce_with_logits_loss<K: kindle_core::prelude::DType>(
            pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            // Numerically stable: max(x, 0) - x*y + log(1 + exp(-|x|))
            let zero = pred
                .zeros_like()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let relu_x = pred
                .maximum(&zero)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let x_y = pred
                .broadcast_mul(target)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let abs_x = pred
                .abs()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let neg_abs_x = abs_x
                .neg()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let exp_neg_abs = neg_abs_x
                .exp()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let one = (exp_neg_abs + 1.0f64)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let log_term = one
                .log()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            let elementwise = relu_x
                .broadcast_sub(&x_y)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
                .broadcast_add(&log_term)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            match reduction {
                kindle_core::prelude::Reduction::Mean => Ok(elementwise
                    .mean_all()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
                kindle_core::prelude::Reduction::Sum => Ok(elementwise
                    .sum_all()
                    .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
                kindle_core::prelude::Reduction::None => Ok(elementwise),
            }
        }

        /// Computes mean squared error between `pred` and `target`; the
        /// reduction argument is ignored since candle's `mse` always averages.
        fn mse_loss<K: kindle_core::prelude::DType>(
            pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::prelude::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let loss = candle_nn::loss::mse(pred, target)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(loss)
        }

        /// Computes cross-entropy loss between `pred` logits and `target`
        /// class indices, casting `target` to `U32` as candle requires; the
        /// reduction argument is ignored.
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
    /// Unit tests for the candle dtype and device conversion helpers.
    mod tests {
        use super::*;

        #[test]
        /// Checks that `to_candle_dtype` maps `F32` and `I64` to the
        /// corresponding candle dtypes.
        fn test_to_candle_dtype() {
            assert_eq!(to_candle_dtype(DTypeId::F32), candle::DType::F32);
            assert_eq!(to_candle_dtype(DTypeId::I64), candle::DType::I64);
        }

        #[test]
        /// Checks that `to_candle_device` maps the CPU device kind to
        /// `candle::Device::Cpu`.
        fn test_to_candle_device() {
            let cpu = DeviceId::cpu();
            let c_dev = to_candle_device(&cpu).unwrap();
            assert!(matches!(c_dev, candle::Device::Cpu));
        }
    }
}
