//! Coverage and admission consistency checks over the four capability
//! tables, cross-referenced against `OPERATION_CATALOG`.

use super::query::{coverage_report, support};
use super::rules::descriptor_min_rank;
use super::tables::{CPU_CAPABILITIES, CUDA_CAPABILITIES, METAL_CAPABILITIES, WGPU_CAPABILITIES};
use alloc::collections::BTreeSet;
use incin_core::exec::{
    CapabilityQuery, CapabilityRule, GradientRule, ImplementationKind, LayoutClass, MathMode,
    OPERATION_CATALOG, SupportLevel, catalog_entry,
};
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::DTypeId;

#[test]
fn coverage_has_one_explicit_decision_per_catalog_operation() {
    let report = coverage_report();
    assert_eq!(report.len(), OPERATION_CATALOG.len());
    let identities: BTreeSet<_> = report.iter().map(|row| row.operation).collect();
    assert_eq!(identities.len(), report.len());
}

#[test]
fn exact_capability_rows_and_executor_admission_share_one_declaration() {
    for (device, rules) in [
        (DeviceKind::Cpu, CPU_CAPABILITIES),
        (DeviceKind::Cuda, CUDA_CAPABILITIES),
        (DeviceKind::Wgpu, WGPU_CAPABILITIES),
        (DeviceKind::Metal, METAL_CAPABILITIES),
    ] {
        for entry in OPERATION_CATALOG {
            let registered = rules.iter().find(|rule| rule.operation == entry.operation);
            if let Some(rule) = registered {
                let query = CapabilityQuery {
                    operation: incin_core::exec::OperationIdentity::Builtin(entry.operation),
                    dtype: rule.dtypes[0],
                    layout: rule.layouts[0],
                    rank: rule.min_rank,
                    training: rule.training,
                    math_mode: rule.math_modes[0],
                };
                assert!(
                    !matches!(support(device, &query), SupportLevel::Unsupported(_)),
                    "{device:?}: {}",
                    entry.operation
                );
            } else {
                let query = CapabilityQuery {
                    operation: incin_core::exec::OperationIdentity::Builtin(entry.operation),
                    dtype: DTypeId::F32.descriptor(),
                    layout: LayoutClass::Contiguous,
                    rank: descriptor_min_rank(entry.operation),
                    training: false,
                    math_mode: MathMode::Precise,
                };
                assert!(
                    matches!(support(device, &query), SupportLevel::Unsupported(_)),
                    "{device:?}: {}",
                    entry.operation
                );
            }
        }
    }

    for rules in [
        CPU_CAPABILITIES,
        CUDA_CAPABILITIES,
        WGPU_CAPABILITIES,
        METAL_CAPABILITIES,
    ] {
        assert!(rules.iter().any(|rule| rule.operation.is_exact()));
    }
}

/// The class of bug this guards against: `CUDA_CAPABILITIES` once carried
/// a hand-written `native(OperationKind::Normalization, FLOAT_DTYPES, ...)`
/// row from before the typed catalog existed, and nothing checked it
/// against reality after the CUDA normalization kernels were deleted. It
/// shipped in `docs/capabilities.md` claiming native LayerNorm/BatchNorm
/// support no CUDA kernel has ever executed.
///
/// `exact_capability_rows_and_executor_admission_share_one_declaration`
/// above only walks `OPERATION_CATALOG`, i.e. the exact typed rows; the
/// coarse rows a family query like `incin doctor`'s `spec(OperationKind::
/// Normalization, ...)` actually resolves against are a second,
/// hand-written channel `assert_every_advertised_*_row_executes!` never
/// sees, because they carry no `op::X` type to bind an `Execute<O>`
/// bound to. This test is that channel's check: coarser than a compile
/// time proof, but load-bearing where one is not possible.
#[test]
fn every_coarse_family_row_is_backed_by_a_native_exact_row() {
    use incin_core::shapes::error::OperationKind as K;

    // The exact identities `incin doctor` and the `legacy` capability
    // rows mean by each coarse family name, at the granularity those two
    // callers actually use (finer than `OperationKind::family()`, which
    // collapses `MatMul`/`Conv2d`/`Pool2d` into one `Reduction` bucket
    // and would not have caught this bug either). `Storage` is excluded:
    // every backend that compiles implements `StorageBackend`, so there
    // is no exact op whose absence could make that specific claim false.
    const FAMILIES: &[(K, &[K])] = &[
        (
            K::Pointwise,
            &[
                K::Add,
                K::Sub,
                K::Mul,
                K::Div,
                K::Relu,
                K::Exp,
                K::Sqrt,
                K::Log,
                K::Tanh,
                K::Sigmoid,
            ],
        ),
        (K::Broadcast, &[K::BroadcastAs]),
        (K::Reshape, &[K::ReshapeExact]),
        (
            K::Fill,
            &[K::Zeros, K::Ones, K::Full, K::Arange, K::Linspace],
        ),
        (K::Random, &[K::UniformRandom, K::NormalRandom]),
        (
            K::Reduction,
            &[K::SumAll, K::MeanAll, K::MaxAll, K::MinAll, K::SumDim],
        ),
        (K::MatMul, &[K::MatMulExact]),
        (K::Conv2d, &[K::Conv2dExact]),
        (K::Pool2d, &[K::MaxPool2d, K::AvgPool2d]),
        (
            K::Normalization,
            &[K::Softmax, K::LayerNorm, K::BatchNorm, K::RmsNorm],
        ),
    ];

    for (device, rules) in [
        (DeviceKind::Cpu, CPU_CAPABILITIES),
        (DeviceKind::Cuda, CUDA_CAPABILITIES),
        (DeviceKind::Wgpu, WGPU_CAPABILITIES),
        (DeviceKind::Metal, METAL_CAPABILITIES),
    ] {
        for rule in rules {
            if rule.implementation != ImplementationKind::Native {
                continue;
            }
            let Some((_, members)) = FAMILIES
                .iter()
                .find(|(family, _)| *family == rule.operation)
            else {
                continue;
            };
            let backed = members.iter().any(|member| {
                rules.iter().any(|r| {
                    r.operation == *member && r.implementation == ImplementationKind::Native
                })
            });
            assert!(
                backed,
                "{device:?} advertises native {} but none of {:?} has a \
                 matching native row in the same table; either the coarse \
                 row is stale or the family list above is out of date",
                rule.operation, members
            );
        }
    }
}

/// `crates/incin-core/src/exec/dispatch.rs`'s `admit_invocation` checks
/// *every* operand's dtype against the one capability row an operation
/// resolves to, not just a primary operand's. A mixed-operand op like
/// `where_cond`/`masked_fill` (a `bool` mask beside `f32` data) whose row
/// declares only the data dtype makes the mask operand fail admission
/// before either kernel ever launches - the exact bug this regression
/// guards: CUDA's `where_cond`/`masked_fill` rows briefly declared
/// `F32_ONLY`, which the single-dtype checks above never catch because
/// they only ever query `rule.dtypes[0]`.
#[test]
fn mixed_mask_and_data_operations_admit_both_operand_dtypes_on_every_backend() {
    use incin_core::shapes::error::OperationKind as K;

    for (device, rules) in [
        (DeviceKind::Cpu, CPU_CAPABILITIES),
        (DeviceKind::Cuda, CUDA_CAPABILITIES),
        (DeviceKind::Wgpu, WGPU_CAPABILITIES),
        (DeviceKind::Metal, METAL_CAPABILITIES),
    ] {
        for operation in [K::WhereCond, K::MaskedFill] {
            let Some(rule) = rules.iter().find(|rule| rule.operation == operation) else {
                // Not every backend has migrated this identity yet; a
                // missing row is a separate, already-covered gap.
                continue;
            };
            for dtype in [DTypeId::F32.descriptor(), DTypeId::Bool.descriptor()] {
                let query = CapabilityQuery {
                    operation: incin_core::exec::OperationIdentity::Builtin(operation),
                    dtype,
                    layout: rule.layouts[0],
                    rank: rule.min_rank,
                    training: rule.training,
                    math_mode: rule.math_modes[0],
                };
                assert!(
                    !matches!(support(device, &query), SupportLevel::Unsupported(_)),
                    "{device:?}: {operation} refuses a {dtype:?} operand, so a real \
                     invocation (mask=bool, data=f32) would fail admission on \
                     whichever operand this row does not list",
                );
            }

            // `where_cond`'s mask can legitimately arrive at a lower rank
            // than the data it broadcasts against (a per-column mask
            // selecting between two 2-D operands, for instance), and
            // `admit_invocation` checks each operand's *own* rank against
            // this one row before the executor's own broadcast ever
            // runs. `descriptor_min_rank(WhereCond)`/`MaskedFill` fall to
            // that function's `_ => 0` default (neither has a match arm
            // there), so the row's floor is already 0 - this asserts
            // that stays true rather than trusting the default silently.
            let low_rank_query = CapabilityQuery {
                operation: incin_core::exec::OperationIdentity::Builtin(operation),
                dtype: DTypeId::Bool.descriptor(),
                layout: rule.layouts[0],
                rank: 1,
                training: rule.training,
                math_mode: rule.math_modes[0],
            };
            assert!(
                !matches!(
                    support(device, &low_rank_query),
                    SupportLevel::Unsupported(_)
                ),
                "{device:?}: {operation} refuses a rank-1 bool mask, which a \
                 lower-rank-than-data mask broadcast would send",
            );
        }
    }
}

/// Issue #90: the CPU matmul rows widened from `F32_ONLY` to
/// `FLOAT_DTYPES`. The exact rows (`MatMulExact`/`BatchedMatMul`/`Addmm`/
/// `Linear`) and the coarse legacy `MatMul` row all have to agree on that
/// set - an exact row that understates the coarse one beside it (or the
/// reverse) would make `doctor`'s family probe and a real `matmul` call
/// disagree about what runs. The non-float dtypes stay refused, and the
/// three rows that deliberately did *not* widen
/// (`ScaledDotProductAttention`/`Dot`/`Outer`, which moved to
/// `composed_reduction` so they keep the F32_ONLY `$reduction` claim)
/// refuse an f16 query by dtype rather than by accident.
#[test]
fn the_cpu_matmul_rows_admit_every_float_and_refuse_the_rest() {
    use super::constants::FLOAT_DTYPES;
    use incin_core::exec::{LayoutClass, MathMode, OperationIdentity, UnsupportedReason};
    use incin_core::shapes::error::OperationKind as K;

    // Every float storage dtype across the four rows that widened.
    // Rank clears each row's floor (2 for `matmul`, 3 for `bmm`, 1 for
    // `addmm`/`linear`); Contiguous and training are admitted by all.
    let admitted: &[(K, DTypeId, usize)] = &[
        (K::MatMulExact, DTypeId::BF16, 2),
        (K::MatMulExact, DTypeId::F16, 2),
        (K::MatMulExact, DTypeId::F32, 2),
        (K::MatMulExact, DTypeId::F64, 2),
        (K::BatchedMatMul, DTypeId::BF16, 3),
        (K::BatchedMatMul, DTypeId::F16, 3),
        (K::BatchedMatMul, DTypeId::F32, 3),
        (K::BatchedMatMul, DTypeId::F64, 3),
        (K::Addmm, DTypeId::BF16, 2),
        (K::Addmm, DTypeId::F16, 2),
        (K::Addmm, DTypeId::F32, 2),
        (K::Addmm, DTypeId::F64, 2),
        (K::Linear, DTypeId::BF16, 2),
        (K::Linear, DTypeId::F16, 2),
        (K::Linear, DTypeId::F32, 2),
        (K::Linear, DTypeId::F64, 2),
    ];
    for &(operation, dtype, rank) in admitted {
        let query = CapabilityQuery {
            operation: OperationIdentity::Builtin(operation),
            dtype: dtype.descriptor(),
            layout: LayoutClass::Contiguous,
            rank,
            training: true,
            math_mode: MathMode::Precise,
        };
        assert!(
            !matches!(
                support(DeviceKind::Cpu, &query),
                SupportLevel::Unsupported(_)
            ),
            "CPU must admit {operation:?} at {dtype:?} rank {rank}, got {:?}",
            support(DeviceKind::Cpu, &query)
        );
    }

    // FLOAT_DTYPES is the whole widening: integer and boolean operands
    // stay refused rather than riding the moved rows.
    let refused: &[(K, DTypeId, usize)] = &[
        (K::MatMulExact, DTypeId::I64, 2),
        (K::MatMulExact, DTypeId::Bool, 2),
        (K::BatchedMatMul, DTypeId::I64, 3),
        (K::Addmm, DTypeId::I64, 2),
        (K::Linear, DTypeId::I64, 2),
    ];
    for &(operation, dtype, rank) in refused {
        let query = CapabilityQuery {
            operation: OperationIdentity::Builtin(operation),
            dtype: dtype.descriptor(),
            layout: LayoutClass::Contiguous,
            rank,
            training: true,
            math_mode: MathMode::Precise,
        };
        assert!(
            matches!(
                support(DeviceKind::Cpu, &query),
                SupportLevel::Unsupported(_)
            ),
            "CPU must still refuse {operation:?} at {dtype:?}, got {:?}",
            support(DeviceKind::Cpu, &query)
        );
    }

    // The rows that did not widen stay honest: SDPA sits on f32-only
    // `softmax`, Dot's `sum_all` always returns f32 storage, and Outer
    // stays on the same narrow set CUDA advertises - an f16 query is
    // refused by dtype, not by rank or layout.
    for operation in [K::ScaledDotProductAttention, K::Dot, K::Outer] {
        let query = CapabilityQuery {
            operation: OperationIdentity::Builtin(operation),
            dtype: DTypeId::F16.descriptor(),
            layout: LayoutClass::Contiguous,
            rank: descriptor_min_rank(operation),
            training: true,
            math_mode: MathMode::Precise,
        };
        assert!(
            matches!(
                support(DeviceKind::Cpu, &query),
                SupportLevel::Unsupported(UnsupportedReason::DType { .. })
            ),
            "CPU {operation:?} must stay f32-only (moved to composed_reduction), got {:?}",
            support(DeviceKind::Cpu, &query)
        );
    }

    // The coarse legacy `MatMul` row has to match the exact row it stands
    // beside: both claim FLOAT_DTYPES, neither trails the other at
    // F32_ONLY.
    let coarse = CPU_CAPABILITIES
        .iter()
        .find(|rule| rule.operation == K::MatMul)
        .expect("CPU carries a coarse MatMul row");
    assert_eq!(
        coarse.dtypes, FLOAT_DTYPES,
        "the coarse CPU MatMul row must match the exact MatMulExact row's FLOAT_DTYPES"
    );
}

/// Issue #93: the quantize boundary's `training` flag is a per-backend fact
/// about which kernels record a tape entry, so it is stated per operation
/// rather than per group.
///
/// CPU's `quantize`/`dequantize` kernels push the straight-through entry and
/// the catalog backs that with [`GradientRule::StraightThrough`], so those two
/// rows admit training. `quantized_matmul` has [`GradientRule::None`] and
/// records nothing on any backend, so its row stays `false` everywhere.
/// CUDA's executor pushes no tape for the boundary either, and WGPU/Metal do
/// not advertise the groups at all - fail-closed in both directions: a row
/// claims training only where the implementation that must answer for it
/// exists, and the catalog's gradient rule and the row never disagree.
#[test]
fn the_quantize_boundary_claims_training_only_where_a_kernel_records_it() {
    use incin_core::shapes::error::OperationKind as K;

    let training_of = |rules: &[CapabilityRule], operation: K| {
        rules
            .iter()
            .find(|rule| rule.operation == operation)
            .map(|rule| rule.training)
    };
    let gradient_of = |operation: K| catalog_entry(operation).map(|entry| entry.gradient);

    // The two halves of the boundary record on CPU, and say so in both
    // places the contract is written down.
    for operation in [K::Quantize, K::Dequantize] {
        assert_eq!(
            training_of(CPU_CAPABILITIES, operation),
            Some(true),
            "CPU records the STE tape entry for {operation:?}, so its row must admit training"
        );
        assert_eq!(
            gradient_of(operation),
            Some(GradientRule::StraightThrough),
            "{operation:?} records the STE tape entry, so its catalog row must claim the approximation"
        );
    }

    // The product over compressed blocks inherits nothing from the boundary.
    assert_eq!(
        training_of(CPU_CAPABILITIES, K::QuantizedMatMul),
        Some(false),
        "quantized_matmul records no tape, so its row must refuse training"
    );
    assert_eq!(
        gradient_of(K::QuantizedMatMul),
        Some(GradientRule::None),
        "quantized_matmul has no backward rule and must not inherit the boundary's claim"
    );

    // Every other backend: whatever it advertises of the three stays
    // training-false until an executor there records a tape entry. A backend
    // that does not advertise one of them at all is fine - `None` is the
    // fail-closed answer too.
    for (device, rules) in [
        (DeviceKind::Cuda, CUDA_CAPABILITIES),
        (DeviceKind::Wgpu, WGPU_CAPABILITIES),
        (DeviceKind::Metal, METAL_CAPABILITIES),
    ] {
        for operation in [K::Quantize, K::Dequantize, K::QuantizedMatMul] {
            if let Some(training) = training_of(rules, operation) {
                assert!(
                    !training,
                    "{device:?} advertises {operation:?} with training = true, but its \
                     executor records no tape for it"
                );
            }
        }
    }
}
