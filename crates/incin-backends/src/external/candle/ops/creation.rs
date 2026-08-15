//! Tensor creation operations for the Candle adapter.

use crate::external::candle::CandleBackend;
use crate::external::candle::convert::{to_candle_device, to_candle_dtype};
use crate::external::candle::executor::CandleStorage;
use crate::external::*;
use candle_core as candle;

pub(crate) fn zeros_storage(
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CandleStorage> {
    let t = candle::Tensor::zeros(shape, to_candle_dtype(dtype)?, &to_candle_device(device)?)
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
    CandleStorage::try_new(t)
}

pub(crate) fn ones_storage(
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CandleStorage> {
    let t = candle::Tensor::ones(shape, to_candle_dtype(dtype)?, &to_candle_device(device)?)
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
    CandleStorage::try_new(t)
}

pub(crate) fn uniform_random_storage(
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CandleStorage> {
    let t = candle::Tensor::rand(0f32, 1f32, shape, &to_candle_device(device)?)
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
        .to_dtype(to_candle_dtype(dtype)?)
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
    CandleStorage::try_new(t)
}

pub(crate) fn normal_random_storage(
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CandleStorage> {
    let t = candle::Tensor::randn(0f32, 1f32, shape, &to_candle_device(device)?)
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?
        .to_dtype(to_candle_dtype(dtype)?)
        .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?;
    CandleStorage::try_new(t)
}

impl<D: incin_core::tensor::device::Device> CandleBackend<D> {
    // This adapter does not route candle's fill or sequence constructors yet.
    crate::unsupported::unsupported_creation_ops! {
        fill: full;
        sequence: arange, linspace;
    }

    /// Allocates a tensor of `shape` filled with zeros on `device` with the
    /// given dtype.
    pub fn zeros<K: incin_core::tensor::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        zeros_storage(shape, dtype, device)
    }

    /// Allocates a tensor of `shape` filled with ones on `device` with the
    /// given dtype.
    pub fn ones<K: incin_core::tensor::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        ones_storage(shape, dtype, device)
    }

    /// Samples a uniform `[0, 1)` tensor of `shape` on `device`, then casts
    /// it to `dtype`.
    pub fn rand<K: incin_core::tensor::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        uniform_random_storage(shape, dtype, device)
    }

    /// Samples a standard-normal tensor of `shape` on `device`, then casts
    /// it to `dtype`.
    pub fn randn<K: incin_core::tensor::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        normal_random_storage(shape, dtype, device)
    }

    /// Allocates a zero-initialized trainable `Var` of `shape` on `device`.
    fn var_zeros<K: incin_core::tensor::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::backend_authoring::VariableBackend>::Var<K>> {
        Ok(
            candle::Var::zeros(shape, to_candle_dtype(dtype)?, &to_candle_device(device)?)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
        )
    }

    /// Allocates a one-initialized trainable `Var` of `shape` on `device`.
    fn var_ones<K: incin_core::tensor::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::backend_authoring::VariableBackend>::Var<K>> {
        Ok(
            candle::Var::ones(shape, to_candle_dtype(dtype)?, &to_candle_device(device)?)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
        )
    }

    /// Allocates a trainable `Var` of `shape` on `device`, sampled from a
    /// uniform `[0, 1)` distribution.
    fn var_rand<K: incin_core::tensor::dtype::DType>(
        shape: &[usize],
        _dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::backend_authoring::VariableBackend>::Var<K>> {
        Ok(
            candle::Var::rand(0f32, 1f32, shape, &to_candle_device(device)?)
                .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?,
        )
    }

    /// Allocates a trainable `Var` of `shape` on `device`, sampled from a
    /// standard-normal distribution.
    fn var_randn<K: incin_core::tensor::dtype::DType>(
        shape: &[usize],
        _dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::backend_authoring::VariableBackend>::Var<K>> {
        let dev = to_candle_device(device)?;
        Ok(candle::Var::randn(0f32, 1f32, shape, &dev)
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?)
    }
}
