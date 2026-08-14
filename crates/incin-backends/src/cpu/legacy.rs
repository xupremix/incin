//! Compatibility adapter for the pre-canonical CPU creation surface.

use super::creation::{
    arange_with_total, full_with_total, linspace_with_total, ones_with_total,
    rand_with_total, randn_with_total, var_ones_with_total, var_rand_with_total,
    var_randn_with_total, var_zeros_with_total, zeros_with_total,
};
use super::CpuBackendImpl;
use incin_core::__backend_compat::legacy::CreationOps;
use incin_core::prelude::{Device, DeviceId, DType, DTypeDescriptor, Result, StorageBackend};

impl<D: Device> CreationOps<Self> for CpuBackendImpl<D> {
    fn zeros<K: DType>(shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as StorageBackend>::Storage<K>> {
        zeros_with_total(super::stride::checked_numel(shape)?, shape, dtype, device)
    }
    fn ones<K: DType>(shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as StorageBackend>::Storage<K>> {
        ones_with_total(super::stride::checked_numel(shape)?, shape, dtype, device)
    }
    fn rand<K: DType>(shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as StorageBackend>::Storage<K>> {
        rand_with_total(super::stride::checked_numel(shape)?, shape, dtype, device)
    }
    fn randn<K: DType>(shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as StorageBackend>::Storage<K>> {
        randn_with_total(super::stride::checked_numel(shape)?, shape, dtype, device)
    }
    fn full<K: DType>(value: f64, shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as StorageBackend>::Storage<K>> {
        full_with_total(super::stride::checked_numel(shape)?, value, shape, dtype, device)
    }
    fn arange<K: DType>(start: f64, step: f64, shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as StorageBackend>::Storage<K>> {
        arange_with_total(super::stride::checked_numel(shape)?, start, step, shape, dtype, device)
    }
    fn linspace<K: DType>(start: f64, end: f64, shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as StorageBackend>::Storage<K>> {
        linspace_with_total(super::stride::checked_numel(shape)?, start, end, shape, dtype, device)
    }
    fn var_zeros<K: DType>(shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as incin_core::prelude::VariableBackend>::Var<K>> {
        var_zeros_with_total(super::stride::checked_numel(shape)?, shape, dtype, device)
    }
    fn var_ones<K: DType>(shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as incin_core::prelude::VariableBackend>::Var<K>> {
        var_ones_with_total(super::stride::checked_numel(shape)?, shape, dtype, device)
    }
    fn var_rand<K: DType>(shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as incin_core::prelude::VariableBackend>::Var<K>> {
        var_rand_with_total(super::stride::checked_numel(shape)?, shape, dtype, device)
    }
    fn var_randn<K: DType>(shape: &[usize], dtype: DTypeDescriptor, device: &DeviceId) -> Result<<Self as incin_core::prelude::VariableBackend>::Var<K>> {
        var_randn_with_total(super::stride::checked_numel(shape)?, shape, dtype, device)
    }
}
