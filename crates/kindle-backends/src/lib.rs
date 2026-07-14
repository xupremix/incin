//! # Kindle Backends
//!
//! `kindle-backends` provides concrete implementations of the `Backend` trait defined in `kindle-core`.
//! It acts as the bridge between Kindle's high-level strongly-typed abstractions and the low-level compute engines that actually perform tensor operations.
//!
//! ## Available Backends
//!
//! * **`candle`**: Integrates with [Hugging Face's Candle](https://github.com/huggingface/candle), a minimalist machine learning framework for Rust. It supports CUDA, Metal, and CPU acceleration. Enable this with the `candle` feature.
//! * **`ndarray`**: Integrates with the [ndarray](https://github.com/rust-ndarray/ndarray) ecosystem for pure Rust, CPU-bound multi-dimensional array operations. Enable this with the `ndarray` feature.
//! * **`dummy`**: A mock backend strictly used for testing compile-time shape verification and basic operation traversal without executing real compute.
//!
//! ## Creating Custom Backends
//!
//! If you wish to plug in your own compute engine, you simply need to implement the `kindle_core::tensor::backend::Backend` trait and the associated operation traits (`MatMulOp`, `Conv2dOp`, etc.).

#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
pub(crate) extern crate alloc;

pub use kindle_core::prelude::*;

pub mod prelude {
    #[cfg(feature = "burn")]
    pub use super::burn_backend::*;
    #[cfg(feature = "candle")]
    pub use super::candle::*;
    #[cfg(feature = "ndarray")]
    pub use super::ndarray_backend::*;
    /// `NativeBackend<T, D>`, `NativeStorage`, `NativeVar`, `NativeGrads`, and
    /// all re-exported `kindle_core::prelude::*` items are available when the
    /// `native` feature is enabled. Unlike `candle`/`ndarray`/`burn` which are
    /// defined as inline submodules of this file, `kindle-native` is a separate
    /// crate, so this is a direct `pub use` rather than `super::native_module::*`.
    #[cfg(feature = "native")]
    pub use kindle_native::*;
}

// ----------------------------------------------------------------------------
// CandleBackend
// ----------------------------------------------------------------------------
#[cfg(feature = "candle")]
pub mod candle {
    use super::*;
    use candle_core as candle;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CandleBackend<T, D>(core::marker::PhantomData<(T, D)>);

    pub fn to_candle_device(dev: &KindleDevice) -> Result<candle::Device> {
        use kindle_core::tensor::device::DeviceVariant;
        match dev.variant() {
            DeviceVariant::Cpu => Ok(candle::Device::Cpu),
            #[cfg(feature = "cuda")]
            DeviceVariant::Cuda(ord) => Ok(candle::Device::new_cuda(ord)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
            #[cfg(feature = "metal")]
            DeviceVariant::Metal(ord) => Ok(candle::Device::new_metal(ord)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
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
            KindleDType::Q8_0 => unimplemented!("Q8_0 is not natively supported in candle yet"),
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::Backend for CandleBackend<T, D>
    {
        type Device = D;
        type FloatElem = f32;
        type IntElem = i64;
        type BackendWithDevice<NewD: kindle_core::prelude::Device> = CandleBackend<T, NewD>;

        type Storage<K: kindle_core::prelude::DType> = candle_core::Tensor;
        type RawVar = candle_core::Var;
        type Grads = candle_core::backprop::GradStore;
        type InnerBackend = Self;

        fn shape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Vec<usize> {
            t.dims().to_vec()
        }

        fn format_tensor_display<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("{}", t)
        }
        fn format_tensor_debug<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("Raw Tensor: {:?}, Strides: {:?}", t, t.stride())
        }

        fn var_as_tensor<K: kindle_core::prelude::DType>(
            var: &<Self as kindle_core::prelude::Backend>::RawVar,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(var.as_tensor().clone())
        }
        fn var_from_tensor<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(candle::Var::from_tensor(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

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

        fn assign_var<K: kindle_core::prelude::DType>(
            var: &mut <Self as kindle_core::prelude::Backend>::RawVar,
            tensor: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<()> {
            var.set(tensor)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        fn backward<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            loss.backward()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
        }

        fn backward_with_nan_check<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            Self::backward::<K>(loss)
        }

        fn get_grad<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            grads: &<Self as kindle_core::prelude::Backend>::Grads,
        ) -> Result<Option<<Self as kindle_core::prelude::Backend>::Storage<K>>> {
            Ok(grads.get(t).cloned())
        }

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
        fn quantize<K: kindle_core::prelude::FloatDType, Q: kindle_core::prelude::QuantDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<Q>> {
            Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Candle",
            })
        }
        fn dequantize<Q: kindle_core::prelude::QuantDType, K: kindle_core::prelude::FloatDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "dequantize",
                backend: "Candle",
            })
        }
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

        fn var_randn<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: KindleDType,
            device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            let dev = to_candle_device(device)?;
            Ok(candle::Var::randn(0f32, 1f32, shape, &dev)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

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
        fn add<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_add(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn sub<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_sub(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn mul<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs
                .broadcast_mul(rhs)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
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

        fn stack<K: kindle_core::prelude::DType>(
            tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_core::Tensor::stack(tensors, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        fn concat<K: kindle_core::prelude::DType>(
            tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_core::Tensor::cat(tensors, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        fn broadcast_as<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.broadcast_as(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn broadcast_left<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.broadcast_left(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        fn reshape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.reshape(shape)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn transpose<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim1: usize,
            dim2: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.transpose(dim1, dim2)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
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

        fn flatten<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            start_dim: usize,
            end_dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.flatten(start_dim, end_dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        fn narrow<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
            start: usize,
            len: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.narrow(dim, start, len)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        fn squeeze<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.squeeze(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

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
        fn add_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok((t + scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn mul_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok((t * scalar).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn relu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.relu()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn gelu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.gelu_erf()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        } // using gelu_erf as fallback for general

        fn softmax<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(candle_nn::ops::softmax(t, dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        fn swish<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            // swish is x * sigmoid(x)
            Ok(candle_nn::ops::silu(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn abs<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.abs()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn neg<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.neg()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn sqrt<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sqrt()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn exp<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.exp()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn log<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.log()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn tanh<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.tanh()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn sigmoid<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(::candle_nn::ops::sigmoid(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ReductionOps<Self> for CandleBackend<T, D>
    {
        fn sum_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn mean_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn max_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn min_all<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min_all()
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        fn sum_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn sum_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.sum_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

        fn mean_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn mean_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mean_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn max_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn max_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.max_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn min_dim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }
        fn min_keepdim<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.min_keepdim(dim)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
        }

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
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::ModuleOps<Self> for CandleBackend<T, D>
    {
        fn adaptive_avg_pool2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _output_size: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("adaptive_avg_pool2d not implemented for CandleBackend")
        }

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
        fn l1_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::nn::loss::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("l1_loss not implemented for CandleBackend")
        }

        fn bce_with_logits_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::nn::loss::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("bce_with_logits_loss not implemented for CandleBackend")
        }

        fn mse_loss<K: kindle_core::prelude::DType>(
            pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::nn::loss::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let loss = candle_nn::loss::mse(pred, target)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            Ok(loss)
        }

        fn cross_entropy_loss<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            target: &<Self as kindle_core::prelude::Backend>::Storage<KInt>,
            _reduction: kindle_core::nn::loss::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            let target_u32 = target
                .to_dtype(candle_core::DType::U32)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
            candle_nn::loss::cross_entropy(pred, &target_u32)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
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

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::tensor::backend::OptimizerOps<Self> for CandleBackend<T, D> {}
}

// ----------------------------------------------------------------------------
// NdarrayBackend
// ----------------------------------------------------------------------------
#[cfg(feature = "ndarray")]
pub mod ndarray_backend {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NdarrayBackend<T, D>(core::marker::PhantomData<(T, D)>);

    #[derive(Clone, Debug)]
    pub struct NdarrayVar(pub alloc::sync::Arc<spin::RwLock<ndarray::ArrayD<f32>>>);
    #[derive(Clone, Debug)]
    pub struct NdarrayGrads;

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::Backend for NdarrayBackend<T, D>
    {
        type Device = D;
        type FloatElem = f32;
        type IntElem = i64;
        type BackendWithDevice<NewD: kindle_core::prelude::Device> = NdarrayBackend<T, NewD>;

        type Storage<K: kindle_core::prelude::DType> = ndarray::ArrayD<f32>;
        type RawVar = NdarrayVar;
        type Grads = NdarrayGrads;
        type InnerBackend = Self;

        fn to_bytes<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<u8>> {
            let slice = t.as_slice().ok_or_else(|| {
                let err: kindle_core::err::Error =
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

        fn from_bytes<K: kindle_core::prelude::DType>(
            bytes: &[u8],
            shape: &[usize],
            dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            if dtype != KindleDType::F32 {
                let err: kindle_core::err::Error =
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

        fn shape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Vec<usize> {
            t.shape().to_vec()
        }

        fn format_tensor_display<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("{}", t)
        }
        fn format_tensor_debug<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> alloc::string::String {
            std::format!("Raw Tensor: {:?}, Strides: {:?}", t, t.strides())
        }

        fn var_as_tensor<K: kindle_core::prelude::DType>(
            v: &<Self as kindle_core::prelude::Backend>::RawVar,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(v.0.read().clone())
        }
        fn var_from_tensor<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(NdarrayVar(alloc::sync::Arc::new(spin::RwLock::new(
                t.clone(),
            ))))
        }

        fn assign_var<K: kindle_core::prelude::DType>(
            var: &mut <Self as kindle_core::prelude::Backend>::RawVar,
            tensor: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<()> {
            let mut w = var.0.write();
            *w = tensor.clone();
            Ok(())
        }

        fn var_to_device(
            var: &<Self as kindle_core::prelude::Backend>::RawVar,
            _device: &kindle_core::prelude::KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(var.clone())
        }
        fn backward<K: kindle_core::prelude::DType>(
            _loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            Err(Error::UnsupportedBackendOperation {
                op: "backward",
                backend: "Ndarray",
            })
        }

        fn backward_with_nan_check<K: kindle_core::prelude::DType>(
            loss: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Grads> {
            Self::backward::<K>(loss)
        }

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
        fn quantize<K: kindle_core::prelude::FloatDType, Q: kindle_core::prelude::QuantDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<Q>> {
            Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Ndarray",
            })
        }
        fn dequantize<Q: kindle_core::prelude::QuantDType, K: kindle_core::prelude::FloatDType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<Q>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "dequantize",
                backend: "Ndarray",
            })
        }
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
        fn zeros<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(ndarray::ArrayD::<f32>::zeros(shape))
        }
        fn ones<K: kindle_core::prelude::DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(ndarray::ArrayD::<f32>::ones(shape))
        }
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

        fn var_zeros<K: kindle_core::prelude::DType>(
            s: &[usize],
            _dt: KindleDType,
            _dev: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(NdarrayVar(alloc::sync::Arc::new(spin::RwLock::new(
                ndarray::ArrayD::<f32>::zeros(s),
            ))))
        }
        fn var_ones<K: kindle_core::prelude::DType>(
            s: &[usize],
            _dt: KindleDType,
            _dev: &KindleDevice,
        ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
            Ok(NdarrayVar(alloc::sync::Arc::new(spin::RwLock::new(
                ndarray::ArrayD::<f32>::ones(s),
            ))))
        }
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
        fn add<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs + rhs)
        }
        fn sub<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs - rhs)
        }
        fn mul<K: kindle_core::prelude::DType>(
            lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(lhs * rhs)
        }
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
        fn add_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mapv(|x| x + scalar as f32))
        }
        fn mul_scalar_float<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            scalar: f64,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mapv(|x| x * scalar as f32))
        }
        fn relu<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mapv(|x| if x > 0.0 { x } else { 0.0 }))
        }
        fn gelu<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "gelu",
                backend: "Ndarray",
            })
        }
        fn softmax<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "softmax",
                backend: "Ndarray",
            })
        }
        fn swish<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "swish",
                backend: "Ndarray",
            })
        }
        fn abs<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Ok(t.mapv(|x: f32| x.abs()))
        }
        fn neg<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "neg",
                backend: "Ndarray",
            })
        }
        fn sqrt<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sqrt",
                backend: "Ndarray",
            })
        }
        fn exp<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "exp",
                backend: "Ndarray",
            })
        }
        fn log<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "log",
                backend: "Ndarray",
            })
        }
        fn tanh<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "tanh",
                backend: "Ndarray",
            })
        }
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
        fn sum_all<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_all",
                backend: "Ndarray",
            })
        }
        fn mean_all<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_all",
                backend: "Ndarray",
            })
        }
        fn max_all<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_all",
                backend: "Ndarray",
            })
        }
        fn min_all<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_all",
                backend: "Ndarray",
            })
        }
        fn sum_dim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_dim",
                backend: "Ndarray",
            })
        }
        fn sum_keepdim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "sum_keepdim",
                backend: "Ndarray",
            })
        }
        fn mean_dim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_dim",
                backend: "Ndarray",
            })
        }
        fn mean_keepdim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "mean_keepdim",
                backend: "Ndarray",
            })
        }
        fn max_dim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_dim",
                backend: "Ndarray",
            })
        }
        fn max_keepdim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "max_keepdim",
                backend: "Ndarray",
            })
        }
        fn min_dim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_dim",
                backend: "Ndarray",
            })
        }
        fn min_keepdim<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _d: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "min_keepdim",
                backend: "Ndarray",
            })
        }
        fn argmax<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            Err(Error::UnsupportedBackendOperation {
                op: "argmax",
                backend: "Ndarray",
            })
        }
        fn argmin<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<KInt>> {
            Err(Error::UnsupportedBackendOperation {
                op: "argmin",
                backend: "Ndarray",
            })
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::prelude::TensorOps<Self> for NdarrayBackend<T, D>
    {
        fn matmul<K: kindle_core::prelude::DType>(
            _lhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _rhs: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "matmul",
                backend: "Ndarray",
            })
        }
        fn stack<K: kindle_core::prelude::DType>(
            _tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            _dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "stack",
                backend: "Ndarray",
            })
        }
        fn concat<K: kindle_core::prelude::DType>(
            _tensors: &[&<Self as kindle_core::prelude::Backend>::Storage<K>],
            _dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "concat",
                backend: "Ndarray",
            })
        }
        fn broadcast_as<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "broadcast_as",
                backend: "Ndarray",
            })
        }
        fn broadcast_left<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "broadcast_left",
                backend: "Ndarray",
            })
        }
        fn reshape<K: kindle_core::prelude::DType>(
            t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            shape: &[usize],
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            t.to_owned()
                .into_shape_with_order(shape)
                .map_err(|e: ndarray::ShapeError| anyhow::anyhow!(e).into())
        }
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
        fn squeeze<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "squeeze",
                backend: "Ndarray",
            })
        }

        fn float_to_scalar<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<f64> {
            Err(Error::UnsupportedBackendOperation {
                op: "float_to_scalar",
                backend: "Ndarray",
            })
        }
        fn float_to_vec1<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<Vec<f64>> {
            Err(Error::UnsupportedBackendOperation {
                op: "float_to_vec1",
                backend: "Ndarray",
            })
        }
        fn int_to_scalar<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<i64> {
            Err(Error::UnsupportedBackendOperation {
                op: "int_to_scalar",
                backend: "Ndarray",
            })
        }
        fn int_to_vec1<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<Vec<i64>> {
            Err(Error::UnsupportedBackendOperation {
                op: "int_to_vec1",
                backend: "Ndarray",
            })
        }
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
        fn adaptive_avg_pool2d<K: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _output_size: (usize, usize),
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("adaptive_avg_pool2d not implemented for NdarrayBackend")
        }

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

        fn embedding<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _t: &<Self as kindle_core::prelude::Backend>::Storage<KInt>,
            _w: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "embedding",
                backend: "Ndarray",
            })
        }
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
        fn l1_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::nn::loss::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("l1_loss not implemented for NdArrayBackend")
        }

        fn bce_with_logits_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::nn::loss::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("bce_with_logits_loss not implemented for NdarrayBackend")
        }

        fn mse_loss<K: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _reduction: kindle_core::nn::loss::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            unimplemented!("mse_loss not implemented for NdArrayBackend")
        }
        fn cross_entropy_loss<K: kindle_core::prelude::DType, KInt: kindle_core::prelude::DType>(
            _pred: &<Self as kindle_core::prelude::Backend>::Storage<K>,
            _target: &<Self as kindle_core::prelude::Backend>::Storage<KInt>,
            _reduction: kindle_core::nn::loss::Reduction,
        ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
            Err(Error::UnsupportedBackendOperation {
                op: "cross_entropy_loss",
                backend: "Ndarray",
            })
        }
    }

    impl<T: kindle_core::prelude::DType, D: kindle_core::prelude::Device>
        kindle_core::tensor::backend::OptimizerOps<Self> for NdarrayBackend<T, D> {}
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
                type InnerBackend = Self;

                fn var_as_tensor(var: &<Self as kindle_core::prelude::Backend>::RawVar) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(var.clone())
                }

                fn tensor_to_device(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "tensor_to_device", backend: "Burn" }) }
                fn var_to_device(_var: &<Self as kindle_core::prelude::Backend>::RawVar, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_to_device", backend: "Burn" }) }
                fn to_dtype(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _dtype: KindleDType) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "to_dtype", backend: "Burn" }) }
                fn backward(_loss: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::Grads> { Err(Error::UnsupportedBackendOperation { op: "backward", backend: "Burn" }) }
                fn get_grad(_var: &<Self as kindle_core::prelude::Backend>::RawVar, _grads: &<Self as kindle_core::prelude::Backend>::Grads) -> Result<Option<<Self as kindle_core::prelude::Backend>::RawTensor>> { Ok(None) }

    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::CreationOps<Self> for BurnBackend<B> {
                fn zeros(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    let d: [usize; $n] = shape.try_into().map_err(|_| Error::ShapeMismatch { expected: alloc::vec![$n], got: shape.to_vec() })?;
                    Ok(burn::tensor::Tensor::<B, $n>::zeros(d, &B::Device::default()))
                }
                fn ones(shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    let d: [usize; $n] = shape.try_into().map_err(|_| Error::ShapeMismatch { expected: alloc::vec![$n], got: shape.to_vec() })?;
                    Ok(burn::tensor::Tensor::<B, $n>::ones(d, &B::Device::default()))
                }
                fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "rand", backend: "Burn" })
                }
                fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "randn", backend: "Burn" })
                }

                fn var_zeros(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_zeros", backend: "Burn" }) }
                fn var_ones(_s: &[usize], _dt: KindleDType, _dev: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_ones", backend: "Burn" }) }
                fn var_rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_rand", backend: "Burn" }) }
                fn var_randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> { Err(Error::UnsupportedBackendOperation { op: "var_randn", backend: "Burn" }) }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::NumericOps<Self> for BurnBackend<B> {
                fn mul_scalar(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _scalar: kindle_core::tensor::backend::ScalarValue) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mul_scalar", backend: "Burn" }) }
                fn add_scalar(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _scalar: kindle_core::tensor::backend::ScalarValue) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "add_scalar", backend: "Burn" }) }
                fn add(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone() + rhs.clone())
                }
                fn sub(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone() - rhs.clone())
                }
                fn mul(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone() * rhs.clone())
                }
                fn div(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone() / rhs.clone())
                }
                fn matmul(lhs: &<Self as kindle_core::prelude::Backend>::RawTensor, rhs: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(lhs.clone().matmul(rhs.clone()))
                }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::FloatOps<Self> for BurnBackend<B> {
                fn relu(t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(burn::tensor::activation::relu(t.clone()))
                }
                fn gelu(t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(burn::tensor::activation::gelu(t.clone()))
                }
                fn softmax(t: &<Self as kindle_core::prelude::Backend>::RawTensor, dim: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(burn::tensor::activation::softmax(t.clone(), dim))
                }
                fn swish(t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(burn::tensor::activation::silu(t.clone()))
                }
                fn abs(t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Ok(t.clone().abs())
                }
                fn neg(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "neg", backend: "Burn" }) }
                fn sqrt(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sqrt", backend: "Burn" }) }
                fn exp(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "exp", backend: "Burn" }) }
                fn log(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "log", backend: "Burn" }) }
                fn tanh(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "tanh", backend: "Burn" }) }
                fn sigmoid(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sigmoid", backend: "Burn" }) }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::ReductionOps<Self> for BurnBackend<B> {
                fn sum_all(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_all", backend: "Burn" }) }
                fn mean_all(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_all", backend: "Burn" }) }
                fn max_all(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_all", backend: "Burn" }) }
                fn min_all(_t: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_all", backend: "Burn" }) }
                fn sum_dim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_dim", backend: "Burn" }) }
                fn sum_keepdim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "sum_keepdim", backend: "Burn" }) }
                fn mean_dim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_dim", backend: "Burn" }) }
                fn mean_keepdim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mean_keepdim", backend: "Burn" }) }
                fn max_dim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_dim", backend: "Burn" }) }
                fn max_keepdim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "max_keepdim", backend: "Burn" }) }
                fn min_dim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_dim", backend: "Burn" }) }
                fn min_keepdim(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "min_keepdim", backend: "Burn" }) }
                fn argmax(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "argmax", backend: "Burn" }) }
                fn argmin(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "argmin", backend: "Burn" }) }

    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::TensorOps<Self> for BurnBackend<B> {
                fn stack(_tensors: &[&<Self as kindle_core::prelude::Backend>::RawTensor], _dim: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "stack", backend: "Burn" }) }
                fn concat(_tensors: &[&<Self as kindle_core::prelude::Backend>::RawTensor], _dim: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "concat", backend: "Burn" }) }
                fn broadcast_as(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _shape: &[usize]) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "broadcast_as", backend: "Burn" }) }
                fn broadcast_left(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _shape: &[usize]) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "broadcast_left", backend: "Burn" }) }
                fn reshape(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _shape: &[usize]) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "reshape", backend: "Burn" })
                }
                fn transpose(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _d1: usize, _d2: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "transpose", backend: "Burn" }) }
                fn flatten(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _s: usize, _e: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "flatten", backend: "Burn" }) }
                fn narrow(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _dim: usize, _start: usize, _len: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "narrow", backend: "Burn" }) }
                fn squeeze(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _dim: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "squeeze", backend: "Burn" }) }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::ModuleOps<Self> for BurnBackend<B> {
                fn layer_norm(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _w: &<Self as kindle_core::prelude::Backend>::RawTensor, _b: &<Self as kindle_core::prelude::Backend>::RawTensor, _e: f32) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "layer_norm", backend: "Burn" }) }
                fn batch_norm(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _w: &<Self as kindle_core::prelude::Backend>::RawTensor, _b: &<Self as kindle_core::prelude::Backend>::RawTensor, _rm: &<Self as kindle_core::prelude::Backend>::RawTensor, _rv: &<Self as kindle_core::prelude::Backend>::RawTensor, _e: f32) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "batch_norm", backend: "Burn" }) }
                fn embedding(_t: &<Self as kindle_core::prelude::Backend>::RawTensor, _w: &<Self as kindle_core::prelude::Backend>::RawTensor) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "embedding", backend: "Burn" }) }
                fn conv2d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: &<Self as kindle_core::prelude::Backend>::RawTensor, _: Option<&<Self as kindle_core::prelude::Backend>::RawTensor>, _: usize, _: usize, _: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv2d", backend: "Burn" })
                }
                fn conv1d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: &<Self as kindle_core::prelude::Backend>::RawTensor, _: Option<&<Self as kindle_core::prelude::Backend>::RawTensor>, _: usize, _: usize, _: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv1d", backend: "Burn" })
                }
                fn conv_transpose2d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: &<Self as kindle_core::prelude::Backend>::RawTensor, _: Option<&<Self as kindle_core::prelude::Backend>::RawTensor>, _: usize, _: usize, _: usize, _: usize) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "conv_transpose2d", backend: "Burn" })
                }
                fn max_pool2d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: (usize, usize), _: (usize, usize)) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "max_pool2d", backend: "Burn" })
                }
                fn avg_pool2d(_: &<Self as kindle_core::prelude::Backend>::RawTensor, _: (usize, usize), _: (usize, usize)) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> {
                    Err(Error::UnsupportedBackendOperation { op: "avg_pool2d", backend: "Burn" })
                }
    }

    impl<B: burn::tensor::backend::Backend, $($D: kindle_core::prelude::Dim),*> kindle_core::prelude::LossOps<Self> for BurnBackend<B> {
                fn l1_loss(_pred: &<Self as kindle_core::prelude::Backend>::RawTensor, _target: &<Self as kindle_core::prelude::Backend>::RawTensor, _reduction: kindle_core::nn::loss::Reduction) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "l1_loss", backend: "Burn" }) }
                fn bce_with_logits_loss(_pred: &<Self as kindle_core::prelude::Backend>::RawTensor, _target: &<Self as kindle_core::prelude::Backend>::RawTensor, _reduction: kindle_core::nn::loss::Reduction) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "bce_with_logits_loss", backend: "Burn" }) }
                fn mse_loss(_pred: &<Self as kindle_core::prelude::Backend>::RawTensor, _target: &<Self as kindle_core::prelude::Backend>::RawTensor, _reduction: kindle_core::nn::loss::Reduction) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "mse_loss", backend: "Burn" }) }
                fn cross_entropy_loss(_pred: &<Self as kindle_core::prelude::Backend>::RawTensor, _target: &<Self as kindle_core::prelude::Backend>::RawTensor, _reduction: kindle_core::nn::loss::Reduction) -> Result<<Self as kindle_core::prelude::Backend>::RawTensor> { Err(Error::UnsupportedBackendOperation { op: "cross_entropy_loss", backend: "Burn" }) }

    }


        };
    }

    impl_burn_backend!(1, D0);
    impl_burn_backend!(2, D0, D1);
    impl_burn_backend!(3, D0, D1, D2);
    impl_burn_backend!(4, D0, D1, D2, D3);
    impl_burn_backend!(5, D0, D1, D2, D3, D4);
}



