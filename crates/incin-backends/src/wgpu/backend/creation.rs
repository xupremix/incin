//! Zero-operand and generator WGPU operations: fills, ranges, and random
//! sampling, for both storage and variable construction.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Concrete creation helpers
// ─────────────────────────────────────────────────────────────────────────────
impl<D: Device> WgpuBackendImpl<D> {
    /// `full`. WGPU storage is always physically f32 (`zeros`/`ones` above
    /// build a `Vec<f32>` regardless of the requested `dtype`, which
    /// `validate_wgpu` restricts to what the dtype policy allows), so this
    /// fills a host-side `Vec<f32>` and uploads it exactly like they do.
    pub(crate) fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "full")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![val as f32; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }
    /// `arange`.
    pub(crate) fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "arange")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = (0..n).map(|i| (start + (i as f64) * step) as f32).collect();
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `linspace`.
    pub(crate) fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "linspace")?;
        let n = num_elements(shape)?;
        let step = if n > 1 {
            (end - start) / ((n - 1) as f64)
        } else {
            0.0
        };
        let data: Vec<f32> = (0..n)
            .map(|i| if i == n - 1 { end } else { start + (i as f64) * step } as f32)
            .collect();
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `zeros`.
    pub(crate) fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "zeros")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![0.0; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `ones`.
    pub(crate) fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Fill, "ones")?;
        let n = num_elements(shape)?;
        let data: Vec<f32> = vec![1.0; n];
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `rand`.
    pub(crate) fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Random, "rand")?;
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = num_elements(shape)?;
        // Simple LCG for now – GPU-side random generation would need more infrastructure
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let mut state = seed as u64;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (u32::MAX as f32)
            })
            .collect();
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `randn`.
    pub(crate) fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        validate_wgpu(dtype, device, OperationKind::Random, "randn")?;
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = num_elements(shape)?;
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let mut state = seed as u64;
        let lcg = |s: &mut u64| -> f32 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as f32) / (u32::MAX as f32)
        };
        // Box-Muller transform
        let data: Vec<f32> = (0..n.div_ceil(2))
            .flat_map(|_| {
                let u1 = lcg(&mut state).max(1e-7);
                let u2 = lcg(&mut state);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                [r * theta.cos(), r * theta.sin()]
            })
            .take(n)
            .collect();
        let buf = WgpuBuffer::try_from_slice(&data)?;
        Ok(WgpuStorage::new(buf, shape.to_vec()))
    }

    /// `var_zeros`.
    pub(crate) fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::Var<K>> {
        let s = Self::zeros::<K>(shape, dtype, device)?;
        Ok(WgpuVar::new(s))
    }

    /// `var_ones`.
    pub(crate) fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::Var<K>> {
        let s = Self::ones::<K>(shape, dtype, device)?;
        Ok(WgpuVar::new(s))
    }

    /// `var_rand`.
    pub(crate) fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::Var<K>> {
        let s = Self::rand::<K>(shape, dtype, device)?;
        Ok(WgpuVar::new(s))
    }

    /// `var_randn`.
    pub(crate) fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as VariableBackend>::Var<K>> {
        let s = Self::randn::<K>(shape, dtype, device)?;
        Ok(WgpuVar::new(s))
    }
}
