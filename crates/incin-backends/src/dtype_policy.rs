//! Authoritative backend dtype capability and compute-policy resolution.
//!
//! Storage support, tensor initialization, and operation support are separate
//! capabilities. Keeping them distinct prevents a backend from advertising a
//! dtype merely because it can copy that dtype's bytes.

use incin_core::exec::{CapabilityQuery, LayoutClass, MathMode};
use incin_core::prelude::{DTypeId, DeviceKind, Error, Result};

/// The operation vocabulary, owned by `incin-core`.
///
/// `EXE-001` deleted the crate-private `OperationFamily` that used to live here
/// and re-exports [`OperationKind`] in its place. Decision `D-008`: a proposed
/// type that duplicates an existing one promotes it, because ending with two
/// operation vocabularies would be worse than changing nothing.
///
/// Policy still *resolves* at the coarse granularity the old enum had — a
/// backend supports floating-point reduction, not `sum` specifically — so every
/// caller's [`OperationKind`] is folded through [`OperationKind::family`]
/// before it reaches the table below.
pub(crate) use incin_core::prelude::OperationKind;

pub(crate) use incin_core::prelude::DeviceKind as BackendFamily;

fn backend_name(backend: BackendFamily) -> &'static str {
    match backend {
        DeviceKind::Cpu => "Cpu",
        DeviceKind::Cuda => "Cuda",
        DeviceKind::Wgpu => "Wgpu",
        _ => "Unknown",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DTypePolicy {
    pub(crate) storage: DTypeId,
    pub(crate) compute: DTypeId,
    pub(crate) accumulator: DTypeId,
    pub(crate) output: DTypeId,
}

fn supports(backend: BackendFamily, operation: OperationKind, dtype: DTypeId) -> bool {
    crate::capability::support(
        backend,
        &CapabilityQuery {
            operation,
            dtype,
            layout: LayoutClass::Contiguous,
            rank: match operation {
                OperationKind::Normalization => 1,
                OperationKind::MatMul => 2,
                OperationKind::Conv2d | OperationKind::Pool2d => 3,
                _ => 0,
            },
            training: false,
            math_mode: MathMode::Precise,
        },
    )
    .is_supported()
}

pub(crate) fn resolve_dtype_policy(
    backend: BackendFamily,
    operation: OperationKind,
    storage: DTypeId,
    op: &'static str,
) -> Result<DTypePolicy> {
    // Dtype support is a property of the operation's family, not of the
    // individual operation: a backend supports floating-point reduction, not
    // `sum`. Callers name the precise operation so diagnostics can; the fold
    // happens once, here.
    let family = operation.family();
    if !supports(backend, family, storage) {
        return Err(Error::UnsupportedDType {
            dtype: storage,
            backend: backend_name(backend),
            op,
        });
    }

    let compute = match storage {
        DTypeId::F16 | DTypeId::BF16 => DTypeId::F32,
        _ => storage,
    };
    let accumulator = match family {
        OperationKind::Reduction | OperationKind::Normalization
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

    fn accepted(backend: BackendFamily, family: OperationKind) -> alloc::vec::Vec<DTypeId> {
        DTYPES
            .into_iter()
            .filter(|&dtype| resolve_dtype_policy(backend, family, dtype, "test").is_ok())
            .collect()
    }

    #[test]
    fn capability_matrix_separates_storage_creation_and_operations() {
        assert_eq!(
            accepted(BackendFamily::Cuda, OperationKind::Storage),
            vec![
                DTypeId::I64,
                DTypeId::BF16,
                DTypeId::F16,
                DTypeId::F32,
                DTypeId::F64
            ]
        );
        assert_eq!(
            accepted(BackendFamily::Cuda, OperationKind::Pointwise),
            vec![DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64]
        );
        assert_eq!(
            accepted(BackendFamily::Cuda, OperationKind::Reduction),
            vec![DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64]
        );
        assert_eq!(
            accepted(BackendFamily::Cuda, OperationKind::Normalization),
            vec![DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64]
        );
        assert_eq!(
            accepted(BackendFamily::Wgpu, OperationKind::Storage),
            vec![DTypeId::F32]
        );
        assert_eq!(
            accepted(BackendFamily::Cpu, OperationKind::Random),
            vec![DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64]
        );
    }

    #[test]
    fn low_precision_storage_uses_f32_compute_and_accumulation() {
        for dtype in [DTypeId::F16, DTypeId::BF16] {
            let pointwise =
                resolve_dtype_policy(BackendFamily::Cuda, OperationKind::Pointwise, dtype, "test")
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
                OperationKind::Pointwise,
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
