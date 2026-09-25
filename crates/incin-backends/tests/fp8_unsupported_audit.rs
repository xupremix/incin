//! Issue #94 unsupported-hardware audit: no accelerator registry claims
//! fp8, and the refusal is checked rather than assumed.
//!
//! The CUDA/WGPU/Metal capability tables are compiled constants; until a
//! backend lands fp8 kernels (#85), descriptor machinery (#90), and the
//! conditional-claim machinery to state them, every row must stay free of
//! `F8E4M3`/`F8E5M2` — a single fp8 entry in any rule would be a claim no
//! kernel here honours. Mirrors `wgpu_f16_audit.rs`.
//!
//! Like that audit, these tests need no hardware: `capability::support`
//! answers from the compiled table alone.

use incin_backends::capability::{
    CUDA_CAPABILITIES, METAL_CAPABILITIES, WGPU_CAPABILITIES, support,
};
use incin_core::exec::{CapabilityQuery, OperationIdentity, SupportLevel, UnsupportedReason};
use incin_core::shapes::error::OperationKind as K;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::DTypeId;

/// The global audit: not one rule of any accelerator backend lists fp8
/// among the dtypes it advertises. If a future change genuinely lands fp8
/// execution, this assertion is the one that must be updated deliberately
/// rather than silently passed.
#[test]
fn no_accelerator_rule_advertises_fp8() {
    let tables: &[(&str, &[incin_core::exec::CapabilityRule])] = &[
        ("CUDA", CUDA_CAPABILITIES),
        ("WGPU", WGPU_CAPABILITIES),
        ("METAL", METAL_CAPABILITIES),
    ];
    for dtype in [DTypeId::F8E4M3, DTypeId::F8E5M2] {
        for (backend, rules) in tables {
            for rule in *rules {
                assert!(
                    !rule.dtypes.contains(&dtype.descriptor()),
                    "{backend} rule for {:?} advertises {dtype:?}, but no accelerator kernel reads fp8 (#94)",
                    rule.operation,
                );
            }
        }
    }
}

/// Spot checks across row families an fp8 query would have to clear to be
/// useful: pointwise arithmetic, the matmul family, a creation row, and the
/// storage row itself. Every one must answer `Unsupported(DType)` for fp8
/// and admit plain `f32` so the refusal is by dtype and not an accident of
/// a missing row.
#[test]
fn representative_accelerator_rows_refuse_fp8_by_dtype() {
    let cases: &[(DeviceKind, K, usize)] = &[
        (DeviceKind::Cuda, K::Add, 1),
        (DeviceKind::Cuda, K::MatMulExact, 2),
        (DeviceKind::Cuda, K::Zeros, 1),
        (DeviceKind::Cuda, K::Storage, 1),
        (DeviceKind::Wgpu, K::Add, 1),
        (DeviceKind::Wgpu, K::Zeros, 1),
        (DeviceKind::Wgpu, K::Storage, 1),
        (DeviceKind::Metal, K::Add, 1),
        (DeviceKind::Metal, K::Zeros, 1),
        (DeviceKind::Metal, K::Storage, 1),
    ];
    for &(device, operation, rank) in cases {
        let admits = |dtype: DTypeId| {
            support(
                device,
                &CapabilityQuery {
                    operation: OperationIdentity::Builtin(operation),
                    dtype: dtype.descriptor(),
                    layout: incin_core::exec::LayoutClass::Contiguous,
                    rank,
                    training: false,
                    math_mode: incin_core::exec::MathMode::Precise,
                },
            )
        };
        for fp8 in [DTypeId::F8E4M3, DTypeId::F8E5M2] {
            assert!(
                matches!(
                    admits(fp8),
                    SupportLevel::Unsupported(UnsupportedReason::DType { .. })
                ),
                "{device:?} {operation:?} must refuse {fp8:?} by dtype, got {:?}",
                admits(fp8),
            );
        }
        assert!(
            !matches!(admits(DTypeId::F32), SupportLevel::Unsupported(_)),
            "{device:?} {operation:?} must still admit F32 -- the fp8 refusal has to be a dtype answer, not a missing row"
        );
    }
}
