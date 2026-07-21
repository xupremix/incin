//! `CreationOps` for `CpuBackend<T, D>`: `zeros`/`ones`/`rand`/`randn` and
//! their `var_*` counterparts, plus `tensor_to_device` (a CPU-only no-op this
//! phase).
//!
//! `rand` uses `rand::distributions::Uniform` (uniform `[0.0, 1.0)`);
//! `randn` uses `rand_distr::StandardNormal` (standard-normal samples) — per
//! the "Don't Hand-Roll" guidance, never a hand-written Box-Muller.

use kindle_core::prelude::Error;
use kindle_core::prelude::{CreationOps, DType, KindleDType, KindleDevice, Result};
#[allow(unused_imports)]
use rand::Rng;
#[allow(unused_imports)]
use rand::SeedableRng;
#[allow(unused_imports)]
use rand_distr::{Distribution, StandardNormal};
#[allow(unused_imports)]
use rayon::prelude::*;

use crate::cpu::CpuBackend;
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::var;

/// Auto-generated documentation for fill_buffer.
fn fill_buffer(
    total: usize,
    value: f64,
    dtype: KindleDType,
    device: &KindleDevice,
) -> Result<CpuBuffer> {
    let host_buf = match dtype {
        KindleDType::F32 => CpuBuffer::F32(vec![value as f32; total]),
        KindleDType::F64 => CpuBuffer::F64(vec![value; total]),
        KindleDType::U8 => CpuBuffer::U8(vec![value as u8; total]),
        KindleDType::U32 => CpuBuffer::U32(vec![value as u32; total]),
        KindleDType::I64 => CpuBuffer::I64(vec![value as i64; total]),
        KindleDType::F16 => CpuBuffer::F16(vec![half::f16::from_f64(value); total]),
        KindleDType::BF16 => CpuBuffer::BF16(vec![half::bf16::from_f64(value); total]),
        KindleDType::Q8_0 => {
            return Err(Error::UnsupportedBackendOperation {
                op: "fill",
                backend: "Cpu Q8_0",
            });
        }
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "fill",
                backend: "Cpu unknown dtype",
            });
        }
    };

    match device.variant() {
        kindle_core::prelude::DeviceVariant::Cpu => Ok(host_buf),
        #[cfg(feature = "cuda")]
        kindle_core::prelude::DeviceVariant::Cuda(id) => {
            let ctx = crate::cpu::gpu::cuda_cache::get_cuda_device(id);
            let stream = ctx.default_stream();
            let bytes = host_buf.as_bytes();
            let dev_slice = stream
                .clone_htod(bytes)
                .map_err(|e| Error::Msg(format!("CUDA alloc/copy failed: {:?}", e)))?;
            Ok(CpuBuffer::Cuda(crate::cpu::storage::CpuCudaBuffer {
                len: total,
                data: alloc::sync::Arc::new(dev_slice),
                device: ctx.clone(),
                device_id: id,
            }))
        }
        _ => Err(Error::UnsupportedBackendOperation {
            op: "fill",
            backend: "Cpu unknown device",
        }),
    }
}

impl<T: DType, D: kindle_core::prelude::Device> CreationOps<Self> for CpuBackend<T, D> {
    /// Auto-generated documentation for zeros.
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
        let total: usize = shape.iter().product();
        let buffer = fill_buffer(total, 0.0, dtype, device)?;
        Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
    }

    /// Auto-generated documentation for ones.
    fn ones<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
        let total: usize = shape.iter().product();
        let buffer = fill_buffer(total, 1.0, dtype, device)?;
        Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
    }

    /// Auto-generated documentation for rand.
    fn rand<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
        if dtype != KindleDType::F32 {
            return Err(Error::UnsupportedBackendOperation {
                op: "rand",
                backend: "Cpu",
            });
        }
        let total: usize = shape.iter().product();
        #[cfg(feature = "std")]
        let mut rng = rand::thread_rng();
        #[cfg(not(feature = "std"))]
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1337);
        let data: Vec<f32> = (0..total).map(|_| rng.gen_range(0.0f32..1.0f32)).collect();
        let buffer = CpuBuffer::F32(data);

        let final_buffer = match device.variant() {
            kindle_core::prelude::DeviceVariant::Cpu => buffer,
            #[cfg(feature = "cuda")]
            kindle_core::prelude::DeviceVariant::Cuda(id) => {
                let ctx = crate::cpu::gpu::cuda_cache::get_cuda_device(id);
                let stream = ctx.default_stream();
                let bytes = buffer.as_bytes();
                let dev_slice = stream
                    .clone_htod(bytes)
                    .map_err(|e| Error::Msg(format!("CUDA alloc/copy failed: {:?}", e)))?;
                CpuBuffer::Cuda(crate::cpu::storage::CpuCudaBuffer {
                    len: total,
                    data: alloc::sync::Arc::new(dev_slice),
                    device: ctx.clone(),
                    device_id: id,
                })
            }
            _ => panic!("Unsupported device"),
        };

        Ok(CpuStorage::from_contiguous(final_buffer, shape.to_vec()))
    }

    /// Auto-generated documentation for randn.
    fn randn<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
        if dtype != KindleDType::F32 {
            return Err(Error::UnsupportedBackendOperation {
                op: "randn",
                backend: "Cpu",
            });
        }
        let total: usize = shape.iter().product();
        #[cfg(feature = "std")]
        let mut rng = rand::thread_rng();
        #[cfg(not(feature = "std"))]
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1337);
        let data: Vec<f32> = (0..total).map(|_| rng.sample(StandardNormal)).collect();
        let buffer = CpuBuffer::F32(data);

        let final_buffer = match device.variant() {
            kindle_core::prelude::DeviceVariant::Cpu => buffer,
            #[cfg(feature = "cuda")]
            kindle_core::prelude::DeviceVariant::Cuda(id) => {
                let ctx = crate::cpu::gpu::cuda_cache::get_cuda_device(id);
                let stream = ctx.default_stream();
                let bytes = buffer.as_bytes();
                let dev_slice = stream
                    .clone_htod(bytes)
                    .map_err(|e| Error::Msg(format!("CUDA alloc/copy failed: {:?}", e)))?;
                CpuBuffer::Cuda(crate::cpu::storage::CpuCudaBuffer {
                    len: total,
                    data: alloc::sync::Arc::new(dev_slice),
                    device: ctx.clone(),
                    device_id: id,
                })
            }
            _ => panic!("Unsupported device"),
        };

        Ok(CpuStorage::from_contiguous(final_buffer, shape.to_vec()))
    }

    /// Auto-generated documentation for var_zeros.
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
        let t = <Self as CreationOps<Self>>::zeros::<K>(shape, dtype, device)?;
        var::var_from_tensor(&t)
    }

    /// Auto-generated documentation for var_ones.
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
        let t = <Self as CreationOps<Self>>::ones::<K>(shape, dtype, device)?;
        var::var_from_tensor(&t)
    }

    /// Auto-generated documentation for var_rand.
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
        let t = <Self as CreationOps<Self>>::rand::<K>(shape, dtype, device)?;
        var::var_from_tensor(&t)
    }

    /// Auto-generated documentation for var_randn.
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::RawVar> {
        let t = <Self as CreationOps<Self>>::randn::<K>(shape, dtype, device)?;
        var::var_from_tensor(&t)
    }

    /// Auto-generated documentation for tensor_to_device.
    fn tensor_to_device<K: DType>(
        t: &<Self as kindle_core::prelude::Backend>::Storage<K>,
        device: &KindleDevice,
    ) -> Result<<Self as kindle_core::prelude::Backend>::Storage<K>> {
        let is_cpu = matches!(
            t.buffer.as_ref(),
            CpuBuffer::F32(_)
                | CpuBuffer::F64(_)
                | CpuBuffer::U8(_)
                | CpuBuffer::U32(_)
                | CpuBuffer::I64(_)
                | CpuBuffer::F16(_)
                | CpuBuffer::BF16(_)
                | CpuBuffer::Q8_0(_)
        );

        match (is_cpu, device.variant()) {
            (true, kindle_core::prelude::DeviceVariant::Cpu) => Ok(t.clone()),
            #[cfg(feature = "cuda")]
            (true, kindle_core::prelude::DeviceVariant::Cuda(id)) => {
                let ctx = crate::cpu::gpu::cuda_cache::get_cuda_device(id);
                let stream = ctx.default_stream();
                let bytes = t.buffer.as_bytes();
                let dev_slice = stream
                    .clone_htod(bytes)
                    .map_err(|e| Error::Msg(format!("CUDA alloc/copy failed: {:?}", e)))?;

                let mut cloned = t.clone();
                cloned.buffer =
                    alloc::sync::Arc::new(CpuBuffer::Cuda(crate::cpu::storage::CpuCudaBuffer {
                        len: t.buffer.len(),
                        data: alloc::sync::Arc::new(dev_slice),
                        device: ctx.clone(),
                        device_id: id,
                    }));
                Ok(cloned)
            }
            (false, kindle_core::prelude::DeviceVariant::Cpu) => {
                #[cfg(feature = "cuda")]
                {
                    if let CpuBuffer::Cuda(b) = t.buffer.as_ref() {
                        let stream = b.device.default_stream();
                        // Account for u8 size when allocating the device slice
                        // we need the number of bytes.
                        let mut bytes = vec![0u8; b.data.len()];
                        stream
                            .memcpy_dtoh(b.data.as_ref(), &mut bytes)
                            .map_err(|e| Error::Msg(format!("CUDA dtoh failed: {:?}", e)))?;
                        if core::any::TypeId::of::<K>() == core::any::TypeId::of::<f32>() {
                            let floats: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
                            let mut cloned = t.clone();
                            cloned.buffer = alloc::sync::Arc::new(CpuBuffer::F32(floats));
                            return Ok(cloned);
                        } else {
                            return Err(Error::UnsupportedBackendOperation {
                                op: "dtoh only supports F32 for now",
                                backend: "Cpu",
                            });
                        }
                    }
                }
                Err(Error::UnsupportedBackendOperation {
                    op: "tensor_to_device (unknown device source)",
                    backend: "Cpu",
                })
            }
            _ => Err(Error::UnsupportedBackendOperation {
                op: "tensor_to_device",
                backend: "Cpu",
            }),
        }
    }
}

#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use super::*;
    use kindle_core::prelude::Backend;

    /// Auto-generated documentation for TestBackend.
    type TestBackend = CpuBackend<f32, kindle_core::prelude::Cpu>;

    /// Auto-generated documentation for dev.
    fn dev() -> KindleDevice {
        KindleDevice::cpu()
    }

    /// Auto-generated documentation for f32_vec.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    #[test]
    /// Auto-generated documentation for zeros_produces_correct_shape_and_all_zero_values.
    fn zeros_produces_correct_shape_and_all_zero_values() {
        let t = TestBackend::zeros::<f32>(&[2, 3], KindleDType::F32, &dev()).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert!(f32_vec(&t).iter().all(|&v| v == 0.0));
    }

    #[test]
    /// Auto-generated documentation for ones_produces_correct_shape_and_all_one_values.
    fn ones_produces_correct_shape_and_all_one_values() {
        let t = TestBackend::ones::<f32>(&[2, 3], KindleDType::F32, &dev()).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert!(f32_vec(&t).iter().all(|&v| v == 1.0));
    }

    #[test]
    /// Auto-generated documentation for rand_produces_values_in_zero_one_range.
    fn rand_produces_values_in_zero_one_range() {
        let t = TestBackend::rand::<f32>(&[100], KindleDType::F32, &dev()).unwrap();
        assert_eq!(t.shape, vec![100]);
        let data = f32_vec(&t);
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    /// Auto-generated documentation for randn_produces_statistically_plausible_standard_normal_samples.
    fn randn_produces_statistically_plausible_standard_normal_samples() {
        let t = TestBackend::randn::<f32>(&[1000], KindleDType::F32, &dev()).unwrap();
        let data = f32_vec(&t);
        assert_eq!(data.len(), 1000);
        let n = data.len() as f64;
        let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance: f64 = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        // Wide tolerance — statistical smoke test, not exact.
        assert!(mean.abs() < 0.2, "sample mean {mean} not near 0.0");
        assert!(
            (variance - 1.0).abs() < 0.3,
            "sample variance {variance} not near 1.0"
        );
    }

    #[test]
    /// Auto-generated documentation for var_zeros_wraps_equivalent_zeros_result.
    fn var_zeros_wraps_equivalent_zeros_result() {
        let var = TestBackend::var_zeros::<f32>(&[2, 2], KindleDType::F32, &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert!(f32_vec(&t).iter().all(|&v| v == 0.0));
    }

    #[test]
    /// Auto-generated documentation for var_ones_wraps_equivalent_ones_result.
    fn var_ones_wraps_equivalent_ones_result() {
        let var = TestBackend::var_ones::<f32>(&[2, 2], KindleDType::F32, &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert!(f32_vec(&t).iter().all(|&v| v == 1.0));
    }

    #[test]
    /// Auto-generated documentation for var_rand_wraps_equivalent_rand_result.
    fn var_rand_wraps_equivalent_rand_result() {
        let var = TestBackend::var_rand::<f32>(&[10], KindleDType::F32, &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![10]);
        assert!(f32_vec(&t).iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    /// Auto-generated documentation for var_randn_wraps_equivalent_randn_result.
    fn var_randn_wraps_equivalent_randn_result() {
        let var = TestBackend::var_randn::<f32>(&[50], KindleDType::F32, &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![50]);
    }

    #[test]
    /// Auto-generated documentation for tensor_to_device_is_a_noop_returning_equivalent_storage.
    fn tensor_to_device_is_a_noop_returning_equivalent_storage() {
        let t = TestBackend::zeros::<f32>(&[3], KindleDType::F32, &dev()).unwrap();
        let t2 = TestBackend::tensor_to_device::<f32>(&t, &dev()).unwrap();
        assert_eq!(t2.shape, t.shape);
        assert_eq!(f32_vec(&t2), f32_vec(&t));
    }
}
