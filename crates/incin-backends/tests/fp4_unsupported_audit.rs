//! Issue #95 unsupported-hardware audit: no accelerator registry claims
//! block FP4, and the refusal is checked rather than assumed.
//!
//! Mirrors `fp8_unsupported_audit.rs`: the CUDA/WGPU/Metal capability tables
//! are compiled constants, and until #85 lands a cuBLASLt path (plus a
//! Blackwell device to run it on), every row must stay free of
//! `NVFP4`/`MXFP4` — a single FP4 entry in any rule would claim execution
//! (block-scaled matmul) no kernel here honours. Blackwell-only
//! advertisement stays off by the same token: there are no device rows at
//! all, compile-time only.
//!
//! Like the fp8 audit, these tests need no hardware: `capability::support`
//! answers from the compiled table alone.

use incin_backends::capability::{
    CUDA_CAPABILITIES, METAL_CAPABILITIES, WGPU_CAPABILITIES, support,
};
use incin_core::exec::{CapabilityQuery, OperationIdentity, SupportLevel, UnsupportedReason};
use incin_core::shapes::error::OperationKind as K;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::DTypeId;

/// The global audit: not one rule of any accelerator backend lists block
/// FP4 among the dtypes it advertises. If a future change genuinely lands
/// Blackwell FP4 execution, this assertion is the one that must be updated
/// deliberately rather than silently passed.
#[test]
fn no_accelerator_rule_advertises_fp4() {
    let tables: &[(&str, &[incin_core::exec::CapabilityRule])] = &[
        ("CUDA", CUDA_CAPABILITIES),
        ("WGPU", WGPU_CAPABILITIES),
        ("METAL", METAL_CAPABILITIES),
    ];
    for dtype in [DTypeId::NVFP4, DTypeId::MXFP4] {
        for (backend, rules) in tables {
            for rule in *rules {
                assert!(
                    !rule.dtypes.contains(&dtype.descriptor()),
                    "{backend} rule for {:?} advertises {dtype:?}, but no accelerator kernel reads block FP4 (#95)",
                    rule.operation,
                );
            }
        }
    }
}

/// Spot checks across row families an FP4 query would have to clear to be
/// useful: pointwise arithmetic, the matmul family, a creation row, and the
/// storage row itself. Every one must answer `Unsupported(DType)` for FP4
/// and admit plain `f32` so the refusal is by dtype and not an accident of
/// a missing row.
#[test]
fn representative_accelerator_rows_refuse_fp4_by_dtype() {
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
        for fp4 in [DTypeId::NVFP4, DTypeId::MXFP4] {
            assert!(
                matches!(
                    admits(fp4),
                    SupportLevel::Unsupported(UnsupportedReason::DType { .. })
                ),
                "{device:?} {operation:?} must refuse {fp4:?} by dtype, got {:?}",
                admits(fp4),
            );
        }
        assert!(
            !matches!(admits(DTypeId::F32), SupportLevel::Unsupported(_)),
            "{device:?} {operation:?} must still admit F32 -- the fp4 refusal has to be a dtype answer, not a missing row"
        );
    }
}
