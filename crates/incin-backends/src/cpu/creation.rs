//! CPU-local allocation kernels: `zeros`/`ones`/`rand`/`randn` and their
//! `var_*` counterparts.
//!
//! `rand` uses `rand::distributions::Uniform` (uniform `[0.0, 1.0)`);
//! `randn` uses `rand_distr::StandardNormal` (standard-normal samples) — per
//! the "Don't Hand-Roll" guidance, never a hand-written Box-Muller.

use incin_core::prelude::{
    ConversionFailure, DeviceId, DTypeDescriptor, DTypeId, Error, FloatToIntPolicy, Result,
    convert_f64_to_i64,
};
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

fn exact_integer(value: f64, dtype: DTypeId, operation: &'static str) -> Result<i64> {
    let value = convert_f64_to_i64(
        operation,
        DTypeId::F64.descriptor(),
        value,
        FloatToIntPolicy::Exact,
    )?;
    let in_range = match dtype {
        DTypeId::U8 => u8::try_from(value).is_ok(),
        DTypeId::U32 => u32::try_from(value).is_ok(),
        DTypeId::I64 => true,
        _ => false,
    };
    if !in_range {
        return Err(Error::InvalidConversion {
            operation,
            from: DTypeId::F64.descriptor(),
            to: dtype.descriptor(),
            reason: ConversionFailure::OutOfRange,
        });
    }
    Ok(value)
}

/// Allocate a contiguous buffer with every element set to `value`.
///
/// `operation` is the name the caller was invoked under, not this helper's.
/// `zeros`, `ones` and `full` are all one fill on the CPU, which is an
/// implementation choice and not something the caller agreed to: reporting
/// every failure as `fill` told a reader who wrote `zeros::<f64>()` that an
/// operation they never called was the one that refused, and left them with no
/// way to connect the message back to their own line. The dtype policy is
/// still queried under [`OperationKind::Fill`], because that is the work
/// actually being attempted and the capability table is keyed by the work.
fn fill_buffer(
    total: usize,
    value: f64,
    dtype: DTypeDescriptor,
    device: &DeviceId,
    operation: &'static str,
) -> Result<CpuBuffer> {
    let builtin_id = super::validate_cpu_dtype(dtype, operation)?;
    let host_buf = match builtin_id {
        DTypeId::F32 => CpuBuffer::F32(vec![value as f32; total]),
        DTypeId::F64 => CpuBuffer::F64(vec![value; total]),
        DTypeId::U8 => CpuBuffer::U8(vec![
            u8::try_from(exact_integer(
                value, builtin_id, operation
            )?)
            .map_err(|_| Error::InternalInvariant {
                operation,
                reason: "validated U8 conversion became unrepresentable",
            })?;
            total
        ]),
        DTypeId::U32 => CpuBuffer::U32(vec![
            u32::try_from(exact_integer(
                value, builtin_id, operation
            )?)
            .map_err(|_| Error::InternalInvariant {
                operation,
                reason: "validated U32 conversion became unrepresentable",
            })?;
            total
        ]),
        DTypeId::I64 => CpuBuffer::I64(vec![exact_integer(value, builtin_id, operation)?; total]),
        DTypeId::F16 => CpuBuffer::F16(vec![half::f16::from_f64(value); total]),
        DTypeId::BF16 => CpuBuffer::BF16(vec![half::bf16::from_f64(value); total]),
        DTypeId::Bool => CpuBuffer::Bool(vec![if value != 0.0 { 1u8 } else { 0u8 }; total]),
        DTypeId::Q8_0 => {
            return Err(Error::UnsupportedBackendOperation {
                op: operation,
                backend: "Cpu Q8_0",
            });
        }
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: operation,
                backend: "Cpu unknown dtype",
            });
        }
    };

    match device.kind() {
        incin_core::prelude::DeviceKind::Cpu => Ok(host_buf),

        _ => Err(Error::DeviceInitializationError {
            expected: "cpu".into(),
            got: alloc::format!("{:?}", device.kind()),
        }),
    }
}

/// `zeros`, given an already-known element count.
///
/// The `CreationOps` trait method below derives `total` itself so the direct,
/// non-descriptor API keeps working from a bare shape slice. The canonical
/// executor in `cpu::canonical` calls this form directly with a total it may
/// already hold as a compile-time constant (`Shape::STATIC_NUMEL`), so it
/// does not pay to re-derive here what `cpu::stride::numel_for` already
/// resolved for it.
pub(crate) fn zeros_with_total(
    total: usize,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CpuStorage> {
    let buffer = fill_buffer(total, 0.0, dtype, device, "zeros")?;
    Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
}

/// `ones`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn ones_with_total(
    total: usize,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CpuStorage> {
    let buffer = fill_buffer(total, 1.0, dtype, device, "ones")?;
    Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
}

/// `rand`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn rand_with_total(
    total: usize,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CpuStorage> {
    let builtin_id = super::validate_cpu_dtype(dtype, "rand")?;
    #[cfg(feature = "std")]
    let mut rng = rand::thread_rng();
    #[cfg(not(feature = "std"))]
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1337);
    let data: Vec<f64> = (0..total).map(|_| rng.gen_range(0.0f64..1.0f64)).collect();
    let buffer = match builtin_id {
        DTypeId::F32 => CpuBuffer::F32(data.iter().map(|&x| x as f32).collect()),
        DTypeId::F64 => CpuBuffer::F64(data),
        DTypeId::F16 => CpuBuffer::F16(data.iter().map(|&x| half::f16::from_f64(x)).collect()),
        DTypeId::BF16 => CpuBuffer::BF16(data.iter().map(|&x| half::bf16::from_f64(x)).collect()),
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

/// `randn`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn randn_with_total(
    total: usize,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CpuStorage> {
    let builtin_id = super::validate_cpu_dtype(dtype, "randn")?;
    #[cfg(feature = "std")]
    let mut rng = rand::thread_rng();
    #[cfg(not(feature = "std"))]
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x1337);
    let data: Vec<f64> = (0..total).map(|_| rng.sample(StandardNormal)).collect();
    let buffer = match builtin_id {
        DTypeId::F32 => CpuBuffer::F32(data.iter().map(|&x| x as f32).collect()),
        DTypeId::F64 => CpuBuffer::F64(data),
        DTypeId::F16 => CpuBuffer::F16(data.iter().map(|&x| half::f16::from_f64(x)).collect()),
        DTypeId::BF16 => CpuBuffer::BF16(data.iter().map(|&x| half::bf16::from_f64(x)).collect()),
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

/// `full`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn full_with_total(
    total: usize,
    val: f64,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CpuStorage> {
    let buffer = fill_buffer(total, val, dtype, device, "full")?;
    Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
}

/// `arange`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn arange_with_total(
    total: usize,
    start: f64,
    step: f64,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CpuStorage> {
    if device.kind() != incin_core::prelude::DeviceKind::Cpu {
        return Err(Error::DeviceInitializationError {
            expected: "cpu".into(),
            got: alloc::format!("{:?}", device.kind()),
        });
    }
    let builtin_id = super::validate_cpu_dtype(dtype, "arange")?;
    let data: Vec<f64> = (0..total).map(|i| start + (i as f64) * step).collect();
    let buffer = match builtin_id {
        DTypeId::F32 => CpuBuffer::F32(data.iter().map(|&x| x as f32).collect()),
        DTypeId::F64 => CpuBuffer::F64(data),
        DTypeId::U8 => CpuBuffer::U8(
            data.iter()
                .map(|&x| {
                    exact_integer(x, builtin_id, "arange").and_then(|x| {
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
                    exact_integer(x, builtin_id, "arange").and_then(|x| {
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
                .map(|&x| exact_integer(x, builtin_id, "arange"))
                .collect::<Result<Vec<_>>>()?,
        ),
        DTypeId::F16 => CpuBuffer::F16(data.iter().map(|&x| half::f16::from_f64(x)).collect()),
        DTypeId::BF16 => CpuBuffer::BF16(data.iter().map(|&x| half::bf16::from_f64(x)).collect()),
        DTypeId::Bool => CpuBuffer::Bool(
            data.iter()
                .map(|&x| if x != 0.0 { 1u8 } else { 0u8 })
                .collect(),
        ),
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "arange",
                backend: "Cpu",
            });
        }
    };
    Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
}

/// `linspace`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn linspace_with_total(
    total: usize,
    start: f64,
    end: f64,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<CpuStorage> {
    if device.kind() != incin_core::prelude::DeviceKind::Cpu {
        return Err(Error::DeviceInitializationError {
            expected: "cpu".into(),
            got: alloc::format!("{:?}", device.kind()),
        });
    }
    let builtin_id = super::validate_cpu_dtype(dtype, "linspace")?;
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
    let buffer = match builtin_id {
        DTypeId::F32 => CpuBuffer::F32(data.iter().map(|&x| x as f32).collect()),
        DTypeId::F64 => CpuBuffer::F64(data),
        DTypeId::U8 => CpuBuffer::U8(
            data.iter()
                .map(|&x| {
                    exact_integer(x, builtin_id, "linspace").and_then(|x| {
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
                    exact_integer(x, builtin_id, "linspace").and_then(|x| {
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
                .map(|&x| exact_integer(x, builtin_id, "linspace"))
                .collect::<Result<Vec<_>>>()?,
        ),
        DTypeId::F16 => CpuBuffer::F16(data.iter().map(|&x| half::f16::from_f64(x)).collect()),
        DTypeId::BF16 => CpuBuffer::BF16(data.iter().map(|&x| half::bf16::from_f64(x)).collect()),
        DTypeId::Bool => CpuBuffer::Bool(
            data.iter()
                .map(|&x| if x != 0.0 { 1u8 } else { 0u8 })
                .collect(),
        ),
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "linspace",
                backend: "Cpu",
            });
        }
    };
    Ok(CpuStorage::from_contiguous(buffer, shape.to_vec()))
}

/// `var_zeros`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn var_zeros_with_total(
    total: usize,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<var::CpuVar> {
    var::var_from_tensor(&zeros_with_total(total, shape, dtype, device)?)
}

/// `var_ones`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn var_ones_with_total(
    total: usize,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<var::CpuVar> {
    var::var_from_tensor(&ones_with_total(total, shape, dtype, device)?)
}

/// `var_rand`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn var_rand_with_total(
    total: usize,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<var::CpuVar> {
    var::var_from_tensor(&rand_with_total(total, shape, dtype, device)?)
}

/// `var_randn`, given an already-known element count. See [`zeros_with_total`].
pub(crate) fn var_randn_with_total(
    total: usize,
    shape: &[usize],
    dtype: DTypeDescriptor,
    device: &DeviceId,
) -> Result<var::CpuVar> {
    var::var_from_tensor(&randn_with_total(total, shape, dtype, device)?)
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::CpuBackendImpl;
    use incin_core::backend_authoring::VariableBackend;
    use incin_core::prelude::{
        Backend, ConversionFailure, Cpu, Dyn, Error, StorageTransfer,
    };

    /// `TestBackend`.
    type TestBackend = CpuBackendImpl<Cpu>;

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
        let t = zeros_with_total(6, &[2, 3], DTypeId::F32.descriptor(), &dev()).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert!(f32_vec(&t).iter().all(|&v| v == 0.0));
    }

    #[test]
    /// `ones_produces_correct_shape_and_all_one_values`.
    fn ones_produces_correct_shape_and_all_one_values() {
        let t = ones_with_total(6, &[2, 3], DTypeId::F32.descriptor(), &dev()).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert!(f32_vec(&t).iter().all(|&v| v == 1.0));
    }

    #[test]
    /// `Dyn` carries its runtime dtype through backend creation.
    fn dyn_dtype_uses_runtime_buffer_variant() {
        let t = ones_with_total(2, &[2], DTypeId::F64.descriptor(), &dev()).unwrap();
        assert!(matches!(&*t.buffer, CpuBuffer::F64(values) if values == &vec![1.0, 1.0]));
    }

    #[test]
    fn integer_fill_rejects_implicit_truncation_and_saturation() {
        for (value, dtype) in [
            (1.5, DTypeId::I64.descriptor()),
            (f64::NAN, DTypeId::I64.descriptor()),
            (f64::INFINITY, DTypeId::U32.descriptor()),
            (-1.0, DTypeId::U8.descriptor()),
            (256.0, DTypeId::U8.descriptor()),
        ] {
            assert!(matches!(
                full_with_total(1, value, &[1], dtype, &dev()),
                Err(Error::InvalidConversion { .. })
            ));
        }
    }

    #[test]
    fn integer_ranges_reject_fractional_values() {
        assert!(matches!(
            arange_with_total(2, 0.5, 1.0, &[2], DTypeId::I64.descriptor(), &dev()),
            Err(Error::InvalidConversion {
                operation: "arange",
                reason: ConversionFailure::Fractional,
                ..
            })
        ));
        assert!(matches!(
            linspace_with_total(3, 0.0, 1.0, &[3], DTypeId::I64.descriptor(), &dev()),
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
        let t = rand_with_total(100, &[100], DTypeId::F32.descriptor(), &dev()).unwrap();
        assert_eq!(t.shape, vec![100]);
        let data = f32_vec(&t);
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    /// `randn_produces_statistically_plausible_standard_normal_samples`.
    fn randn_produces_statistically_plausible_standard_normal_samples() {
        let t = randn_with_total(1000, &[1000], DTypeId::F32.descriptor(), &dev()).unwrap();
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
        let var =
            var_zeros_with_total(4, &[2, 2], DTypeId::F32.descriptor(), &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert!(f32_vec(&t).iter().all(|&v| v == 0.0));
    }

    #[test]
    /// `var_ones_wraps_equivalent_ones_result`.
    fn var_ones_wraps_equivalent_ones_result() {
        let var = var_ones_with_total(4, &[2, 2], DTypeId::F32.descriptor(), &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert!(f32_vec(&t).iter().all(|&v| v == 1.0));
    }

    #[test]
    /// `var_rand_wraps_equivalent_rand_result`.
    fn var_rand_wraps_equivalent_rand_result() {
        let var = var_rand_with_total(10, &[10], DTypeId::F32.descriptor(), &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![10]);
        assert!(f32_vec(&t).iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    /// `var_randn_wraps_equivalent_randn_result`.
    fn var_randn_wraps_equivalent_randn_result() {
        let var = var_randn_with_total(50, &[50], DTypeId::F32.descriptor(), &dev()).unwrap();
        let t = TestBackend::var_as_tensor::<f32>(&var).unwrap();
        assert_eq!(t.shape, vec![50]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    /// Same-device transfer returns equivalent destination-native storage.
    fn transfer_to_cpu_returns_equivalent_storage() {
        let t = zeros_with_total(3, &[3], DTypeId::F32.descriptor(), &dev()).unwrap();
        let t2 = <TestBackend as StorageTransfer<Cpu>>::transfer_storage::<f32>(
            &t,
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(t2.shape, t.shape);
        assert_eq!(f32_vec(&t2), f32_vec(&t));
    }
}
