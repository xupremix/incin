//! Zero-operand and generator CUDA operations: fills, ranges, and random
//! sampling, for both storage and variable construction.

use super::*;

impl<D: Device> CudaBackendImpl<D> {
    // No kernel fills an arbitrary value or generates a sequence yet.
    /// `full`. Same host-fill-then-upload pattern `zeros`/`ones` above
    /// already use — `cuda_from_f32` reinterprets a `Vec<f32>`'s bytes as
    /// `dtype`'s native representation, so like those two this only
    /// actually succeeds for `dtype == F32`; any other dtype fails the byte
    /// length check inside `cuda_from_bytes` rather than misreading, the
    /// same pre-existing behavior `zeros`/`ones`/`rand`/`randn` already
    /// have (not something this pass changes).
    pub(crate) fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaStorage> {
        cuda_from_f32(
            shape,
            dtype,
            device,
            vec![val as f32; checked_numel(shape)?],
            "full",
        )
    }
    /// `arange`.
    pub(crate) fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaStorage> {
        let n = checked_numel(shape)?;
        let values: Vec<f32> = (0..n).map(|i| (start + (i as f64) * step) as f32).collect();
        cuda_from_f32(shape, dtype, device, values, "arange")
    }
    /// `linspace`.
    pub(crate) fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaStorage> {
        let n = checked_numel(shape)?;
        let step = if n > 1 {
            (end - start) / ((n - 1) as f64)
        } else {
            0.0
        };
        let values: Vec<f32> = (0..n)
            .map(|i| if i == n - 1 { end } else { start + (i as f64) * step } as f32)
            .collect();
        cuda_from_f32(shape, dtype, device, values, "linspace")
    }

    pub(crate) fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaStorage> {
        cuda_from_f32(
            shape,
            dtype,
            device,
            vec![0.0; checked_numel(shape)?],
            "zeros",
        )
    }

    pub(crate) fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaStorage> {
        cuda_from_f32(
            shape,
            dtype,
            device,
            vec![1.0; checked_numel(shape)?],
            "ones",
        )
    }

    pub(crate) fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaStorage> {
        use rand::RngExt as _;
        let mut rng = rand::rng();
        let values = (0..checked_numel(shape)?).map(|_| rng.random()).collect();
        cuda_from_f32(shape, dtype, device, values, "rand")
    }

    pub(crate) fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaStorage> {
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand::rng();
        let values = (0..checked_numel(shape)?)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        cuda_from_f32(shape, dtype, device, values, "randn")
    }

    pub(crate) fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaVar> {
        Self::zeros::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    pub(crate) fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaVar> {
        Self::ones::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    pub(crate) fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaVar> {
        Self::rand::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }

    pub(crate) fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<CudaVar> {
        Self::randn::<K>(shape, dtype, device).map(|storage| CudaVar { storage })
    }
}
