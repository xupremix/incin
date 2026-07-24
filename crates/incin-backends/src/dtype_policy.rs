//! Authoritative backend dtype capability and compute-policy resolution.
//!
//! Storage support, tensor initialization, and operation support are separate
//! capabilities. Keeping them distinct prevents a backend from advertising a
//! dtype merely because it can copy that dtype's bytes.

use incin_core::prelude::{DTypeId, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendFamily {
    Cpu,
    Cuda,
    Wgpu,
}

impl BackendFamily {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "Cpu",
            Self::Cuda => "Cuda",
            Self::Wgpu => "Wgpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// Individual variants are activated by backend features. Cargo may also build
// a no-backend instance of this crate while checking the full workspace.
#[allow(dead_code)]
pub(crate) enum OperationFamily {
    Storage,
    Fill,
    Random,
    Pointwise,
    Reduction,
    Normalization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DTypePolicy {
    pub(crate) storage: DTypeId,
    pub(crate) compute: DTypeId,
    pub(crate) accumulator: DTypeId,
    pub(crate) output: DTypeId,
}

fn is_float(dtype: DTypeId) -> bool {
    matches!(
        dtype,
        DTypeId::F16 | DTypeId::BF16 | DTypeId::F32 | DTypeId::F64
    )
}

fn supports(backend: BackendFamily, family: OperationFamily, dtype: DTypeId) -> bool {
    match backend {
        BackendFamily::Cpu => match family {
            OperationFamily::Storage => true,
            OperationFamily::Fill => dtype != DTypeId::Q8_0,
            OperationFamily::Random
            | OperationFamily::Pointwise
            | OperationFamily::Reduction
            | OperationFamily::Normalization => is_float(dtype),
        },
        BackendFamily::Cuda => match family {
            // I64 storage is index-only (embedding lookup indices) — it never
            // reaches Pointwise/Reduction/Normalization compute, which stay
            // float-only below.
            OperationFamily::Storage => is_float(dtype) || dtype == DTypeId::I64,
            OperationFamily::Pointwise
            | OperationFamily::Reduction
            | OperationFamily::Normalization => is_float(dtype),
            OperationFamily::Fill | OperationFamily::Random => dtype == DTypeId::F32,
        },
        BackendFamily::Wgpu => dtype == DTypeId::F32,
    }
}

pub(crate) fn resolve_dtype_policy(
    backend: BackendFamily,
    family: OperationFamily,
    storage: DTypeId,
    op: &'static str,
) -> Result<DTypePolicy> {
    if !supports(backend, family, storage) {
        return Err(Error::UnsupportedDType {
            dtype: storage,
            backend: backend.name(),
            op,
        });
    }

    let compute = match storage {
        DTypeId::F16 | DTypeId::BF16 => DTypeId::F32,
        _ => storage,
    };
    let accumulator = match family {
        OperationFamily::Reduction | OperationFamily::Normalization
            if matches!(storage, DTypeId::F16 | DTypeId::BF16) =>
        {
            DTypeId::F32
        }
        _ => compute,
    };

    Ok(DTypePolicy {
        storage,
        compute,
        accumulator,
        output: storage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DTYPES: [DTypeId; 8] = [
        DTypeId::U8,
        DTypeId::U32,
        DTypeId::I64,
        DTypeId::BF16,
        DTypeId::F16,
        DTypeId::F32,
        DTypeId::F64,
        DTypeId::Q8_0,
    ];

    fn accepted(backend: BackendFamily, family: OperationFamily) -> alloc::vec::Vec<DTypeId> {
        DTYPES
            .into_iter()
            .filter(|&dtype| resolve_dtype_policy(backend, family, dtype, "test").is_ok())
            .collect()
    }

    #[test]
    fn capability_matrix_separates_storage_creation_and_operations() {
        assert_eq!(
            accepted(BackendFamily::Cuda, OperationFamily::Storage),
            vec![
                DTypeId::I64,
                DTypeId::BF16,
                DTypeId::F16,
                DTypeId::F32,
                DTypeId::F64
            ]
        );
        assert_eq!(
            accepted(BackendFamily::Cuda, OperationFamily::Pointwise),
            vec![DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64]
        );
        assert_eq!(
            accepted(BackendFamily::Cuda, OperationFamily::Reduction),
            vec![DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64]
        );
        assert_eq!(
            accepted(BackendFamily::Cuda, OperationFamily::Normalization),
            vec![DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64]
        );
        assert_eq!(
            accepted(BackendFamily::Wgpu, OperationFamily::Storage),
            vec![DTypeId::F32]
        );
        assert_eq!(
            accepted(BackendFamily::Cpu, OperationFamily::Random),
            vec![DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64]
        );
    }

    #[test]
    fn low_precision_storage_uses_f32_compute_and_accumulation() {
        for dtype in [DTypeId::F16, DTypeId::BF16] {
            let pointwise = resolve_dtype_policy(
                BackendFamily::Cuda,
                OperationFamily::Pointwise,
                dtype,
                "test",
            )
            .unwrap();
            assert_eq!(pointwise.storage, dtype);
            assert_eq!(pointwise.compute, DTypeId::F32);
            assert_eq!(pointwise.accumulator, DTypeId::F32);
            assert_eq!(pointwise.output, dtype);
        }
    }

    #[test]
    fn unsupported_error_preserves_backend_dtype_and_operation() {
        assert!(matches!(
            resolve_dtype_policy(
                BackendFamily::Wgpu,
                OperationFamily::Pointwise,
                DTypeId::F64,
                "add"
            ),
            Err(Error::UnsupportedDType {
                dtype: DTypeId::F64,
                backend: "Wgpu",
                op: "add"
            })
        ));
    }
}
