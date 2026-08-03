//! `CreationOps` for `CpuBackendImpl<T, D>`: `zeros`/`ones`/`rand`/`randn` and
//! their `var_*` counterparts.
//!
//! `rand` uses `rand::distributions::Uniform` (uniform `[0.0, 1.0)`);
//! `randn` uses `rand_distr::StandardNormal` (standard-normal samples) — per
//! the "Don't Hand-Roll" guidance, never a hand-written Box-Muller.

use crate::cpu::CpuBackendImpl;
use incin_core::backend_authoring::CreationOps;
use incin_core::prelude::*;
use incin_core::prelude::{DType, DTypeId, DeviceId, Error, Result};
#[allow(unused_imports)]
use rand::Rng;
#[allow(unused_imports)]
use rand::SeedableRng;
#[allow(unused_imports)]
use rand_distr::{Distribution, StandardNormal};
#[allow(unused_imports)]
use rayon::prelude::*;

use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::var;
use crate::dtype_policy::{BackendFamily, OperationKind, resolve_dtype_policy};

fn exact_integer(value: f64, dtype: DTypeId, operation: &'static str) -> Result<i64> {
    let value = convert_f64_to_i64(operation, DTypeId::F64, value, FloatToIntPolicy::Exact)?;
    let in_range = match dtype {
        DTypeId::U8 => u8::try_from(value).is_ok(),
        DTypeId::U32 => u32::try_from(value).is_ok(),
        DTypeId::I64 => true,
        _ => false,
    };
    if !in_range {
        return Err(Error::InvalidConversion {
            operation,
            from: DTypeId::F64,
            to: dtype,
            reason: ConversionFailure::OutOfRange,
        });
    }
    Ok(value)
}

/// `fill_buffer`.
fn fill_buffer(total: usize, value: f64, dtype: DTypeId, device: &DeviceId) -> Result<CpuBuffer> {
    resolve_dtype_policy(BackendFamily::Cpu, OperationKind::Fill, dtype, "fill")?;
    let host_buf = match dtype {
        DTypeId::F32 => CpuBuffer::F32(vec![value as f32; total]),
        DTypeId::F64 => CpuBuffer::F64(vec![value; total]),
        DTypeId::U8 => CpuBuffer::U8(vec![
            u8::try_from(exact_integer(value, dtype, "fill")?)
                .map_err(|_| Error::InternalInvariant {
                    operation: "fill",
                    reason: "validated U8 conversion became unrepresentable",
                })?;
            total
        ]),
        DTypeId::U32 => CpuBuffer::U32(vec![
            u32::try_from(exact_integer(value, dtype, "fill")?)
                .map_err(|_| Error::InternalInvariant {
                    operation: "fill",
                    reason: "validated U32 conversion became unrepresentable",
                })?;
            total
        ]),
        DTypeId::I64 => CpuBuffer::I64(vec![exact_integer(value, dtype, "fill")?; total]),
        DTypeId::F16 => CpuBuffer::F16(vec![half::f16::from_f64(value); total]),
        DTypeId::BF16 => CpuBuffer::BF16(vec![half::bf16::from_f64(value); total]),
        DTypeId::Q8_0 => {
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

    match device.kind() {
        incin_core::prelude::DeviceKind::Cpu => Ok(host_buf),

        _ => Err(Error::UnsupportedBackendOperation {
            op: "fill",
            backend: "Cpu unknown device",
        }),
    }
}

impl<T: DType, D: Device> CreationOps<Self> for CpuBackendImpl<T, D> {
    /// `zeros`.
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        let total: usize = crate::cpu::stride::checked_numel(shape)?;
        let buffer = fill_buffer(total, 0.0, dtype, device)?;
        Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
    }

    /// `ones`.
    fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        let total: usize = crate::cpu::stride::checked_numel(shape)?;
        let buffer = fill_buffer(total, 1.0, dtype, device)?;
        Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
    }

    /// `rand`.
    fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        resolve_dtype_policy(BackendFamily::Cpu, OperationKind::Random, dtype, "rand")?;
        let total: usize = crate::cpu::stride::checked_numel(shape)?;
        #[cfg(feature = "std")]
        let mut rng = rand::thread_rng();
        #[cfg(not(feature = "std"))]
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1337);
        let data: Vec<f64> = (0..total).map(|_| rng.gen_range(0.0f64..1.0f64)).collect();
        let buffer = match dtype {
            DTypeId::F32 => CpuBuffer::F32(data.iter().map(|&x| x as f32).collect()),
            DTypeId::F64 => CpuBuffer::F64(data),
            DTypeId::F16 => CpuBuffer::F16(data.iter().map(|&x| half::f16::from_f64(x)).collect()),
            DTypeId::BF16 => {
                CpuBuffer::BF16(data.iter().map(|&x| half::bf16::from_f64(x)).collect())
            }
            _ => unreachable!(),
        };

        let final_buffer = match device.kind() {
            incin_core::prelude::DeviceKind::Cpu => buffer,

            _ => {
                return Err(Error::DeviceInitializationError {
                    expected: "cpu".into(),
                    got: alloc::format!("{:?}", device.kind()),
                });
            }
        };

        Ok(CpuStorage::from_contiguous(final_buffer, shape.to_vec()))
    }

    /// `randn`.
    fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        resolve_dtype_policy(BackendFamily::Cpu, OperationKind::Random, dtype, "randn")?;
        let total: usize = crate::cpu::stride::checked_numel(shape)?;
        #[cfg(feature = "std")]
        let mut rng = rand::thread_rng();
        #[cfg(not(feature = "std"))]
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1337);
        let data: Vec<f64> = (0..total).map(|_| rng.sample(StandardNormal)).collect();
        let buffer = match dtype {
            DTypeId::F32 => CpuBuffer::F32(data.iter().map(|&x| x as f32).collect()),
            DTypeId::F64 => CpuBuffer::F64(data),
            DTypeId::F16 => CpuBuffer::F16(data.iter().map(|&x| half::f16::from_f64(x)).collect()),
            DTypeId::BF16 => {
                CpuBuffer::BF16(data.iter().map(|&x| half::bf16::from_f64(x)).collect())
            }
            _ => unreachable!(),
        };

        let final_buffer = match device.kind() {
            incin_core::prelude::DeviceKind::Cpu => buffer,

            _ => {
                return Err(Error::DeviceInitializationError {
                    expected: "cpu".into(),
                    got: alloc::format!("{:?}", device.kind()),
                });
            }
        };

        Ok(CpuStorage::from_contiguous(final_buffer, shape.to_vec()))
    }

    /// `full`.
    fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        let total: usize = crate::cpu::stride::checked_numel(shape)?;
        let buffer = fill_buffer(total, val, dtype, device)?;
        Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
    }

    /// `arange`.
    fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        if device.kind() != incin_core::prelude::DeviceKind::Cpu {
            return Err(Error::DeviceInitializationError {
                expected: "cpu".into(),
                got: alloc::format!("{:?}", device.kind()),
            });
        }
        let total: usize = crate::cpu::stride::checked_numel(shape)?;
        resolve_dtype_policy(BackendFamily::Cpu, OperationKind::Fill, dtype, "arange")?;
        let data: Vec<f64> = (0..total).map(|i| start + (i as f64) * step).collect();
        let buffer = match dtype {
            DTypeId::F32 => CpuBuffer::F32(data.iter().map(|&x| x as f32).collect()),
            DTypeId::F64 => CpuBuffer::F64(data),
            DTypeId::U8 => CpuBuffer::U8(
                data.iter()
                    .map(|&x| {
                        exact_integer(x, dtype, "arange").and_then(|x| {
                            u8::try_from(x).map_err(|_| Error::InternalInvariant {
                                operation: "arange",
                                reason: "validated U8 conversion became unrepresentable",
                            })
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            DTypeId::U32 => CpuBuffer::U32(
                data.iter()
                    .map(|&x| {
                        exact_integer(x, dtype, "arange").and_then(|x| {
                            u32::try_from(x).map_err(|_| Error::InternalInvariant {
                                operation: "arange",
                                reason: "validated U32 conversion became unrepresentable",
                            })
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            DTypeId::I64 => CpuBuffer::I64(
                data.iter()
                    .map(|&x| exact_integer(x, dtype, "arange"))
                    .collect::<Result<Vec<_>>>()?,
            ),
            DTypeId::F16 => CpuBuffer::F16(data.iter().map(|&x| half::f16::from_f64(x)).collect()),
            DTypeId::BF16 => {
                CpuBuffer::BF16(data.iter().map(|&x| half::bf16::from_f64(x)).collect())
            }
            _ => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "arange",
                    backend: "Cpu",
                });
            }
        };
        Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
    }

    /// `linspace`.
    fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::Storage<K>> {
        if device.kind() != incin_core::prelude::DeviceKind::Cpu {
            return Err(Error::DeviceInitializationError {
                expected: "cpu".into(),
                got: alloc::format!("{:?}", device.kind()),
            });
        }
        let total: usize = crate::cpu::stride::checked_numel(shape)?;
        resolve_dtype_policy(BackendFamily::Cpu, OperationKind::Fill, dtype, "linspace")?;
        let step = if total > 1 {
            (end - start) / ((total - 1) as f64)
        } else {
            0.0
        };
        let data: Vec<f64> = (0..total)
            .map(|i| {
                if i == total - 1 {
                    end
                } else {
                    start + (i as f64) * step
                }
            })
            .collect();
        let buffer = match dtype {
            DTypeId::F32 => CpuBuffer::F32(data.iter().map(|&x| x as f32).collect()),
            DTypeId::F64 => CpuBuffer::F64(data),
            DTypeId::U8 => CpuBuffer::U8(
                data.iter()
                    .map(|&x| {
                        exact_integer(x, dtype, "linspace").and_then(|x| {
                            u8::try_from(x).map_err(|_| Error::InternalInvariant {
                                operation: "linspace",
                                reason: "validated U8 conversion became unrepresentable",
                            })
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            DTypeId::U32 => CpuBuffer::U32(
                data.iter()
                    .map(|&x| {
                        exact_integer(x, dtype, "linspace").and_then(|x| {
                            u32::try_from(x).map_err(|_| Error::InternalInvariant {
                                operation: "linspace",
                                reason: "validated U32 conversion became unrepresentable",
                            })
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            DTypeId::I64 => CpuBuffer::I64(
                data.iter()
                    .map(|&x| exact_integer(x, dtype, "linspace"))
                    .collect::<Result<Vec<_>>>()?,
            ),
            DTypeId::F16 => CpuBuffer::F16(data.iter().map(|&x| half::f16::from_f64(x)).collect()),
            DTypeId::BF16 => {
                CpuBuffer::BF16(data.iter().map(|&x| half::bf16::from_f64(x)).collect())
            }
            _ => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "linspace",
                    backend: "Cpu",
                });
            }
        };
        Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
    }

    /// `var_zeros`.
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::RawVar> {
        let t = <Self as CreationOps<Self>>::zeros::<K>(shape, dtype, device)?;
        var::var_from_tensor(&t)
    }

    /// `var_ones`.
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::RawVar> {
        let t = <Self as CreationOps<Self>>::ones::<K>(shape, dtype, device)?;
        var::var_from_tensor(&t)
    }

    /// `var_rand`.
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::RawVar> {
        let t = <Self as CreationOps<Self>>::rand::<K>(shape, dtype, device)?;
        var::var_from_tensor(&t)
    }

    /// `var_randn`.
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<<Self as incin_core::prelude::Backend>::RawVar> {
        let t = <Self as CreationOps<Self>>::randn::<K>(shape, dtype, device)?;
        var::var_from_tensor(&t)
    }
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use incin_core::prelude::{Backend, Cpu};

    /// `TestBackend`.
    type TestBackend = CpuBackendImpl<f32, Cpu>;

    /// `dev`.
    fn dev() -> DeviceId {
        DeviceId::cpu()
    }

    /// `f32_vec`.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    #[test]
    /// `zeros_produces_correct_shape_and_all_zero_values`.
    fn zeros_produces_correct_shape_and_all_zero_values() {
        let t = TestBackend::zeros::<f32>(&[2, 3], DTypeId::F32, &dev()).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert!(f32_vec(&t).iter().all(|&v| v == 0.0));
    }

    #[test]
    /// `ones_produces_correct_shape_and_all_one_values`.
    fn ones_produces_correct_shape_and_all_one_values() {
        let t = TestBackend::ones::<f32>(&[2, 3], DTypeId::F32, &dev()).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert!(f32_vec(&t).iter().all(|&v| v == 1.0));
    }

    #[test]
    /// `Dyn` carries its runtime dtype through backend creation.
    fn dyn_dtype_uses_runtime_buffer_variant() {
        let t = TestBackend::ones::<Dyn>(&[2], DTypeId::F64, &dev()).unwrap();
        assert!(matches!(&*t.buffer, CpuBuffer::F64(values) if values == &vec![1.0, 1.0]));
    }

    #[test]
    fn integer_fill_rejects_implicit_truncation_and_saturation() {
        for (value, dtype) in [
            (1.5, DTypeId::I64),
            (f64::NAN, DTypeId::I64),
            (f64::INFINITY, DTypeId::U32),
            (-1.0, DTypeId::U8),
            (256.0, DTypeId::U8),
        ] {
            assert!(matches!(
                TestBackend::full::<Dyn>(value, &[1], dtype, &dev()),
                Err(Error::InvalidConversion { .. })
            ));
        }
    }

    #[test]
    fn integer_ranges_reject_fractional_values() {
        assert!(matches!(
            TestBackend::arange::<Dyn>(0.5, 1.0, &[2], DTypeId::I64, &dev()),
            Err(Error::InvalidConversion {
                operation: "arange",
                reason: ConversionFailure::Fractional,
                ..
            })
        ));
        assert!(matches!(
            TestBackend::linspace::<Dyn>(0.0, 1.0, &[3], DTypeId::I64, &dev()),
            Err(Error::InvalidConversion {
                operation: "linspace",
                reason: ConversionFailure::Fractional,
                ..
            })
        ));
    }

    #[test]
    /// `rand_produces_values_in_zero_one_range`.
    fn rand_produces_values_in_zero_one_range() {
        let t = TestBackend::rand::<f32>(&[100], DTypeId::F32, &dev()).unwrap();
        assert_eq!(t.shape, vec![100]);
        let data = f32_vec(&t);
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    /// `randn_produces_statistically_plausible_standard_normal_samples`.
    fn randn_produces_statistically_plausible_standard_normal_samples() {
        let t = TestBackend::randn::<f32>(&[1000], DTypeId::F32, &dev()).unwrap();
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
    /// `var_zeros_wraps_equivalent_zeros_result`.
    fn var_zeros_wraps_equivalent_zeros_result() {
        let var = TestBackend::var_zeros::<f32>(&[2, 2], DTypeId::F32, &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert!(f32_vec(&t).iter().all(|&v| v == 0.0));
    }

    #[test]
    /// `var_ones_wraps_equivalent_ones_result`.
    fn var_ones_wraps_equivalent_ones_result() {
        let var = TestBackend::var_ones::<f32>(&[2, 2], DTypeId::F32, &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert!(f32_vec(&t).iter().all(|&v| v == 1.0));
    }

    #[test]
    /// `var_rand_wraps_equivalent_rand_result`.
    fn var_rand_wraps_equivalent_rand_result() {
        let var = TestBackend::var_rand::<f32>(&[10], DTypeId::F32, &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![10]);
        assert!(f32_vec(&t).iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    /// `var_randn_wraps_equivalent_randn_result`.
    fn var_randn_wraps_equivalent_randn_result() {
        let var = TestBackend::var_randn::<f32>(&[50], DTypeId::F32, &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![50]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    /// Same-device transfer returns equivalent destination-native storage.
    fn transfer_to_cpu_returns_equivalent_storage() {
        let t = TestBackend::zeros::<f32>(&[3], DTypeId::F32, &dev()).unwrap();
        let t2 = <TestBackend as TransferTo<Cpu>>::transfer_storage::<f32>(
            &t,
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(t2.shape, t.shape);
        assert_eq!(f32_vec(&t2), f32_vec(&t));
    }
}
