//! Issue #91 f16/bf16 audit: the WGPU registry cannot claim half
//! precision, and the refusal is checked rather than assumed.
//!
//! The story this pins, in the order `wgpu/capability.rs`'s
//! `validate_wgpu_dtype` doc tells it: `WGPU_CAPABILITIES` is a compiled
//! constant (rows cannot vary per adapter), `device.rs` requests
//! `Features::empty()` (no `SHADER_F16` query), no shader declares
//! `enable f16`, and `bf16` has no WGPU feature bit at all. Until those
//! change together, every row must stay free of `f16`/`bf16` — a single
//! half-dtype entry in any rule would be a claim no kernel here honours.
//!
//! Like `cuda_indexing_admission`, these tests need no hardware:
//! `capability::support` answers from the compiled table alone.
#![cfg(feature = "wgpu")]

use incin_backends::capability::{WGPU_CAPABILITIES, support};
use incin_core::exec::{CapabilityQuery, OperationIdentity, SupportLevel, UnsupportedReason};
use incin_core::shapes::error::OperationKind as K;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::DTypeId;

/// The global audit: not one WGPU rule, of any group, lists `f16` or
/// `bf16` among the dtypes it advertises. This is the claim the doc on
/// `validate_wgpu_dtype` makes about the whole registry; if a future
/// change genuinely lands the feature query, the shader variants and the
/// conditional row machinery, this assertion is the one that must be
/// updated deliberately rather than silently passed.
#[test]
fn no_wgpu_rule_advertises_f16_or_bf16() {
    for dtype in [DTypeId::F16, DTypeId::BF16] {
        for rule in WGPU_CAPABILITIES {
            assert!(
                !rule.dtypes.contains(&dtype.descriptor()),
                "WGPU rule for {:?} advertises {dtype:?}, but no WGPU shader reads half \
                 precision and device.rs requests Features::empty() -- see the \
                 validate_wgpu_dtype audit (#91)",
                rule.operation,
            );
        }
    }
}

/// Spot checks across the row families a half-dtype query would have to
/// clear to be useful: an arithmetic pointwise op, the matmul family
/// (#90's note), a normalization, a reduction, the new comparison rows
/// (#91), and the storage/creation legacy rows themselves. Every one must
/// answer `Unsupported(DType)` for `f16`/`bf16` and admit plain `f32`
/// (or `bool`, for the logicals) so the refusal is by dtype and not an
/// accident of the row being absent.
#[test]
fn representative_wgpu_rows_refuse_half_precision_by_dtype() {
    let cases: &[(K, DTypeId, usize)] = &[
        (K::Add, DTypeId::F32, 1),
        (K::MatMulExact, DTypeId::F32, 2),
        (K::Softmax, DTypeId::F32, 1),
        (K::SumAll, DTypeId::F32, 1),
        (K::CmpEq, DTypeId::F32, 1),
        (K::LogicalAnd, DTypeId::Bool, 1),
        (K::Zeros, DTypeId::F32, 1),
        (K::Storage, DTypeId::F32, 1),
    ];
    for &(operation, admitted_dtype, rank) in cases {
        let admits = |dtype: DTypeId| {
            support(
                DeviceKind::Wgpu,
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
        for half in [DTypeId::F16, DTypeId::BF16] {
            assert!(
                matches!(
                    admits(half),
                    SupportLevel::Unsupported(UnsupportedReason::DType { .. })
                ),
                "WGPU {operation:?} must refuse {half:?} by dtype, got {:?}",
                admits(half),
            );
        }
        assert!(
            !matches!(admits(admitted_dtype), SupportLevel::Unsupported(_)),
            "WGPU {operation:?} must still admit {admitted_dtype:?} -- the half-dtype \
             refusal has to be a dtype answer, not a missing row"
        );
    }
}
