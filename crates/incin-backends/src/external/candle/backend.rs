//! Core `Backend`, `SupportsDType`, and `TransferTo` implementations.

use crate::external::candle::CandleBackend;
use crate::external::candle::convert::{to_candle_device, to_candle_dtype};
use crate::external::*;
use candle_core as candle;

impl<T: incin_core::prelude::DType, D: incin_core::prelude::Device> incin_core::prelude::Backend
    for CandleBackend<T, D>
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
    type Storage<K: incin_core::prelude::DType> = candle_core::Tensor;
    /// A trainable variable is backed by candle's `Var`.
    type RawVar = candle_core::Var;
    /// Gradients are accumulated in candle's `GradStore`, keyed by tensor.
    type Grads = candle_core::backprop::GradStore;
    /// `CandleBackend` has no further inner-backend indirection; it is its
    /// own inner backend.
    type InnerBackend = Self;

    /// Returns the tensor's dimensions as a `Vec<usize>`.
    fn shape<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Vec<usize> {
        t.dims().to_vec()
    }

    /// Formats the tensor using candle's own `Display` implementation.
    fn format_tensor_display<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> alloc::string::String {
        std::format!("{}", t)
    }
    /// Formats the tensor's raw contents together with its strides, for
    /// debugging.
    fn format_tensor_debug<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> alloc::string::String {
        std::format!("Raw Tensor: {:?}, Strides: {:?}", t, t.stride())
    }

    /// Clones the variable's underlying tensor out as plain storage.
    fn var_as_tensor<K: incin_core::prelude::DType>(
        var: &<Self as incin_core::prelude::Backend>::RawVar,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        Ok(var.as_tensor().clone())
    }
    /// Wraps a tensor in a new candle `Var`, cloning its data.
    fn var_from_tensor<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::RawVar> {
        Ok(candle::Var::from_tensor(t).map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }

    /// Overwrites the variable's contents in place with `tensor`.
    fn assign_var<K: incin_core::prelude::DType>(
        var: &mut <Self as incin_core::prelude::Backend>::RawVar,
        tensor: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<()> {
        var.set(tensor)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
    }

    /// Runs backpropagation from `loss`, returning the resulting gradient
    /// store.
    fn backward<K: incin_core::prelude::DType>(
        loss: &<Self as incin_core::prelude::Backend>::Storage<K>,
    ) -> Result<<Self as incin_core::prelude::Backend>::Grads> {
        loss.backward()
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e).into())
    }

    /// Looks up the accumulated gradient for `t` in `grads`, if one was
    /// recorded during backward.
    fn get_grad<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
        grads: &<Self as incin_core::prelude::Backend>::Grads,
    ) -> Result<Option<<Self as incin_core::prelude::Backend>::Storage<K>>> {
        Ok(grads.get(t).cloned())
    }

    /// Flattens the tensor, casts it to `f32`, and returns its raw byte
    /// representation.
    fn to_bytes<K: incin_core::prelude::DType>(
        t: &<Self as incin_core::prelude::Backend>::Storage<K>,
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
    fn from_bytes<K: incin_core::prelude::DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        let floats = unsafe {
            core::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                bytes.len() / core::mem::size_of::<f32>(),
            )
        };
        let d = to_candle_device(device)?;
        let c_dtype = to_candle_dtype(dtype)?;
        let t = candle_core::Tensor::from_slice(floats, shape, &d)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        let t = t
            .to_dtype(c_dtype)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
        Ok(t)
    }
}

macro_rules! impl_candle_storage_dtype {
    ($($dtype:ty),+ $(,)?) => {
        $(
            impl<T: DType, D: Device> SupportsDType<$dtype> for CandleBackend<T, D> {
                fn resolve_dtype(
                    field: &<$dtype as DType>::Field,
                    _device: &DeviceId,
                ) -> Result<DTypeId> {
                    let dtype = <$dtype as DType>::to_incin(field);
                    to_candle_dtype(dtype)?;
                    Ok(dtype)
                }
            }
        )+
    };
}

impl_candle_storage_dtype!(f32, f64, f16, bf16, u8, u32, i64);

impl<T: DType, D: Device> SupportsDType<Dyn> for CandleBackend<T, D> {
    fn resolve_dtype(field: &DTypeId, _device: &DeviceId) -> Result<DTypeId> {
        to_candle_dtype(*field)?;
        Ok(*field)
    }
}

impl<T, D, NewD> incin_core::prelude::TransferTo<NewD> for CandleBackend<T, D>
where
    T: incin_core::prelude::DType,
    D: incin_core::prelude::Device,
    NewD: incin_core::prelude::Device,
{
    type Output = CandleBackend<T, NewD>;

    fn transfer_storage<K: incin_core::prelude::DType>(
        storage: &Self::Storage<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as Backend>::Storage<K>>
    where
        Self::Output: SupportsDType<K>,
    {
        let destination = NewD::to_incin(device)?;
        <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &destination)?;
        let target = to_candle_device(&destination)?;
        storage
            .to_device(&target)
            .map_err(|error| anyhow::anyhow!(error).into())
    }

    fn transfer_var(
        variable: &Self::RawVar,
        dtype: &<T as incin_core::prelude::DType>::Field,
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
