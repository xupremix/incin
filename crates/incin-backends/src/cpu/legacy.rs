//! Compatibility adapters for the pre-canonical CPU backend surface.
//!
//! Canonical CPU execution lives in [`super::canonical`] and calls the
//! operation-local helpers in [`super::creation`] directly.  This module is
//! deliberately the only CPU home for the old grouped creation family; it is
//! a bounded deletion target while downstream callers move to `Execute<O>`.

use super::creation::{
    arange_with_total, full_with_total, linspace_with_total, ones_with_total,
    rand_with_total, randn_with_total, var_ones_with_total, var_rand_with_total,
    var_randn_with_total, var_zeros_with_total, zeros_with_total,
};
use super::CpuBackendImpl;
use incin_core::__backend_compat::legacy::CreationOps;
use incin_core::prelude::{Device, DeviceId, DType, DTypeDescriptor, Result, StorageBackend};

impl<D: Device> CreationOps<Self> for CpuBackendImpl<D> {
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total = super::stride::checked_numel(shape)?;
        zeros_with_total(total, shape, dtype, device)
    }

    fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total = super::stride::checked_numel(shape)?;
        ones_with_total(total, shape, dtype, device)
    }

    fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total = super::stride::checked_numel(shape)?;
        rand_with_total(total, shape, dtype, device)
    }

    fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total = super::stride::checked_numel(shape)?;
        randn_with_total(total, shape, dtype, device)
    }

    fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total = super::stride::checked_numel(shape)?;
        full_with_total(total, val, shape, dtype, device)
    }

    fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total = super::stride::checked_numel(shape)?;
        arange_with_total(total, start, step, shape, dtype, device)
    }

    fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let total = super::stride::checked_numel(shape)?;
        linspace_with_total(total, start, end, shape, dtype, device)
    }

    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::VariableBackend>::Var<K>> {
        let total = super::stride::checked_numel(shape)?;
        var_zeros_with_total(total, shape, dtype, device)
    }

    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::VariableBackend>::Var<K>> {
        let total = super::stride::checked_numel(shape)?;
        var_ones_with_total(total, shape, dtype, device)
    }

    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::VariableBackend>::Var<K>> {
        let total = super::stride::checked_numel(shape)?;
        var_rand_with_total(total, shape, dtype, device)
    }

    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::VariableBackend>::Var<K>> {
        let total = super::stride::checked_numel(shape)?;
        var_randn_with_total(total, shape, dtype, device)
    }
}
