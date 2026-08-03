//! Authoritative native-backend capability registrations.

use incin_core::exec::{
    Capabilities, CapabilityQuery, CapabilityRegistry, CapabilityRule, ImplementationKind,
    LayoutClass, MathMode, OPERATION_CATALOG, SupportLevel,
};
use incin_core::prelude::{DTypeId, DeviceKind, MAX_RANK, OperationKind};

const ALL_DTYPES: &[DTypeId] = &[
    DTypeId::U8,
    DTypeId::U32,
    DTypeId::I64,
    DTypeId::BF16,
    DTypeId::F16,
    DTypeId::F32,
    DTypeId::F64,
    DTypeId::Q8_0,
];
const FLOAT_DTYPES: &[DTypeId] = &[DTypeId::BF16, DTypeId::F16, DTypeId::F32, DTypeId::F64];
const CUDA_STORAGE_DTYPES: &[DTypeId] = &[
    DTypeId::I64,
    DTypeId::BF16,
    DTypeId::F16,
    DTypeId::F32,
    DTypeId::F64,
];
const F32_ONLY: &[DTypeId] = &[DTypeId::F32];
const NON_QUANTIZED: &[DTypeId] = &[
    DTypeId::U8,
    DTypeId::U32,
    DTypeId::I64,
    DTypeId::BF16,
    DTypeId::F16,
    DTypeId::F32,
    DTypeId::F64,
];
const CONTIGUOUS: &[LayoutClass] = &[LayoutClass::Contiguous];
const CPU_LAYOUTS: &[LayoutClass] = &[LayoutClass::Contiguous, LayoutClass::Strided];
const PRECISE: &[MathMode] = &[MathMode::Precise];

const fn native(
    operation: OperationKind,
    dtypes: &'static [DTypeId],
    layouts: &'static [LayoutClass],
    training: bool,
) -> CapabilityRule {
    native_ranked(operation, dtypes, layouts, 0, MAX_RANK, training)
}

const fn native_ranked(
    operation: OperationKind,
    dtypes: &'static [DTypeId],
    layouts: &'static [LayoutClass],
    min_rank: usize,
    max_rank: usize,
    training: bool,
) -> CapabilityRule {
    CapabilityRule::new(
        operation,
        dtypes,
        layouts,
        min_rank,
        max_rank,
        training,
        PRECISE,
        ImplementationKind::Native,
    )
}

// Single declaration consumed by capability generation below, by the grouped
// legacy descriptor executors, and by the canonical per-identity executors in
// `cpu::canonical`. Adding an identity here changes what execution admits and
// what capability queries report in the same edit, and the canonical module
// turns the third consumer into a compile-time obligation: a row advertised
// without an `Execute<Descriptor<op::...>>` implementation does not build.
macro_rules! cpu_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            pointwise = [Add, Sub, Mul, Div],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll, ProdAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim, ProdDim
            ],
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            matmul = [MatMulExact]
        }
    };
}

// Re-exported crate-internally so the CPU executor module can prove, at
// compile time, that it implements every identity this declaration advertises.
pub(crate) use cpu_descriptor_operations;

macro_rules! cuda_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            pointwise = [Add, Sub, Mul, Div],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim
            ],
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            matmul = [MatMulExact]
        }
    };
}

macro_rules! wgpu_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            pointwise = [Add, Sub, Mul, Div],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim
            ],
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            matmul = [MatMulExact]
        }
    };
}

macro_rules! metal_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            pointwise = [Add, Sub, Mul, Div],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            reduction = [
                SumAll, MeanAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim
            ],
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            matmul = [MatMulExact]
        }
    };
}

macro_rules! descriptor_capability_rules {
    (
        pointwise = $pointwise:expr,
        broadcast = $broadcast:expr,
        reshape = $reshape:expr,
        reduction = $reduction:expr,
        spatial = $spatial:expr,
        matmul = $matmul:expr,
        broadcast_training = $broadcast_training:expr,
        reshape_training = $reshape_training:expr,
        pointwise_layouts = $pointwise_layouts:expr,
        broadcast_layouts = $broadcast_layouts:expr,
        reshape_layouts = $reshape_layouts:expr,
        reduction_layouts = $reduction_layouts:expr,
        spatial_layouts = $spatial_layouts:expr,
        matmul_layouts = $matmul_layouts:expr,
        legacy = [$($legacy:expr),* $(,)?];
        pointwise = [$($pointwise_op:ident),* $(,)?],
        broadcast = [$($broadcast_op:ident),* $(,)?],
        reshape = [$($reshape_op:ident),* $(,)?],
        reduction = [$($reduction_op:ident),* $(,)?],
        spatial = [$($spatial_op:ident),* $(,)?],
        matmul = [$($matmul_op:ident),* $(,)?]
    ) => {
        &[
            $($legacy,)*
            $(native(OperationKind::$pointwise_op, $pointwise, $pointwise_layouts, true),)*
            $(native(OperationKind::$broadcast_op, $broadcast, $broadcast_layouts, false),)*
            $(native(OperationKind::$reshape_op, $reshape, $reshape_layouts, false),)*
            // A view operation records a gradient, so it is usable while
            // training over the dtypes a gradient exists for. Without these
            // rows the exact identities are strictly narrower than the legacy
            // family rows they replace, and a training reshape would stop
            // resolving the moment those family rows are removed.
            $(native(OperationKind::$broadcast_op, $broadcast_training, $broadcast_layouts, true),)*
            $(native(OperationKind::$reshape_op, $reshape_training, $reshape_layouts, true),)*
            $(native_ranked(OperationKind::$matmul_op, $matmul, $matmul_layouts, 2, MAX_RANK, true),)*
            $(native_ranked(
                OperationKind::$reduction_op,
                $reduction,
                $reduction_layouts,
                descriptor_min_rank(OperationKind::$reduction_op),
                descriptor_max_rank(OperationKind::$reduction_op),
                true,
            ),)*
            $(native_ranked(
                OperationKind::$spatial_op,
                $spatial,
                $spatial_layouts,
                descriptor_min_rank(OperationKind::$spatial_op),
                descriptor_max_rank(OperationKind::$spatial_op),
                true,
            ),)*
        ]
    };
}

const fn descriptor_min_rank(operation: OperationKind) -> usize {
    match operation {
        OperationKind::Conv2dExact | OperationKind::MaxPool2d | OperationKind::AvgPool2d => 3,
        OperationKind::SumDim
        | OperationKind::SumKeepDim
        | OperationKind::MeanDim
        | OperationKind::MeanKeepDim
        | OperationKind::MaxDim
        | OperationKind::MaxKeepDim
        | OperationKind::MinDim
        | OperationKind::MinKeepDim
        | OperationKind::ProdDim => 1,
        _ => 0,
    }
}

const fn descriptor_max_rank(operation: OperationKind) -> usize {
    match operation {
        OperationKind::Conv2dExact | OperationKind::MaxPool2d | OperationKind::AvgPool2d => 4,
        _ => MAX_RANK,
    }
}

pub static CPU_CAPABILITIES: &[CapabilityRule] = cpu_descriptor_operations!(
    descriptor_capability_rules,
    pointwise = FLOAT_DTYPES,
    broadcast = ALL_DTYPES,
    reshape = ALL_DTYPES,
    reduction = F32_ONLY,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    broadcast_training = FLOAT_DTYPES,
    reshape_training = FLOAT_DTYPES,
    pointwise_layouts = CPU_LAYOUTS,
    broadcast_layouts = CPU_LAYOUTS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CPU_LAYOUTS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CPU_LAYOUTS,
    legacy = [
        CapabilityRule::new(
            OperationKind::ReshapeExact,
            ALL_DTYPES,
            &[LayoutClass::Strided],
            0,
            MAX_RANK,
            false,
            PRECISE,
            ImplementationKind::Composed,
        ),
        native(OperationKind::Storage, ALL_DTYPES, CPU_LAYOUTS, false),
        native(OperationKind::Fill, NON_QUANTIZED, CONTIGUOUS, false),
        native(OperationKind::Random, FLOAT_DTYPES, CONTIGUOUS, false),
        native(OperationKind::Pointwise, FLOAT_DTYPES, CPU_LAYOUTS, true),
        native(OperationKind::Reduction, F32_ONLY, CPU_LAYOUTS, true),
        native_ranked(
            OperationKind::Normalization,
            F32_ONLY,
            CPU_LAYOUTS,
            1,
            MAX_RANK,
            true,
        ),
        native(OperationKind::Broadcast, ALL_DTYPES, CPU_LAYOUTS, false),
        native(OperationKind::Broadcast, FLOAT_DTYPES, CPU_LAYOUTS, true),
        native(OperationKind::Reshape, ALL_DTYPES, CONTIGUOUS, false),
        native(OperationKind::Reshape, FLOAT_DTYPES, CONTIGUOUS, true),
        CapabilityRule::new(
            OperationKind::Reshape,
            NON_QUANTIZED,
            &[LayoutClass::Strided],
            0,
            MAX_RANK,
            false,
            PRECISE,
            ImplementationKind::Composed,
        ),
        CapabilityRule::new(
            OperationKind::Reshape,
            FLOAT_DTYPES,
            &[LayoutClass::Strided],
            0,
            MAX_RANK,
            true,
            PRECISE,
            ImplementationKind::Composed,
        ),
        CapabilityRule::new(
            OperationKind::MatMul,
            F32_ONLY,
            CPU_LAYOUTS,
            2,
            MAX_RANK,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Conv2d,
            F32_ONLY,
            CONTIGUOUS,
            3,
            4,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Pool2d,
            F32_ONLY,
            CONTIGUOUS,
            3,
            4,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
    ]
);

pub static CUDA_CAPABILITIES: &[CapabilityRule] = cuda_descriptor_operations!(
    descriptor_capability_rules,
    pointwise = FLOAT_DTYPES,
    broadcast = CUDA_STORAGE_DTYPES,
    reshape = CUDA_STORAGE_DTYPES,
    reduction = FLOAT_DTYPES,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    broadcast_training = FLOAT_DTYPES,
    reshape_training = FLOAT_DTYPES,
    pointwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    legacy = [
        native(
            OperationKind::Storage,
            CUDA_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Fill, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Random, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Pointwise, FLOAT_DTYPES, CONTIGUOUS, true),
        native(OperationKind::Reduction, FLOAT_DTYPES, CONTIGUOUS, true),
        native_ranked(
            OperationKind::Normalization,
            FLOAT_DTYPES,
            CONTIGUOUS,
            1,
            MAX_RANK,
            true,
        ),
        native(
            OperationKind::Broadcast,
            CUDA_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Broadcast, FLOAT_DTYPES, CONTIGUOUS, true),
        native(
            OperationKind::Reshape,
            CUDA_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Reshape, FLOAT_DTYPES, CONTIGUOUS, true),
        CapabilityRule::new(
            OperationKind::MatMul,
            F32_ONLY,
            CONTIGUOUS,
            2,
            MAX_RANK,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Conv2d,
            F32_ONLY,
            CONTIGUOUS,
            3,
            4,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Pool2d,
            F32_ONLY,
            CONTIGUOUS,
            3,
            4,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
    ]
);

pub static WGPU_CAPABILITIES: &[CapabilityRule] = wgpu_descriptor_operations!(
    descriptor_capability_rules,
    pointwise = F32_ONLY,
    broadcast = F32_ONLY,
    reshape = F32_ONLY,
    reduction = F32_ONLY,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    broadcast_training = F32_ONLY,
    reshape_training = F32_ONLY,
    pointwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    legacy = [
        native(OperationKind::Storage, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Fill, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Random, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Pointwise, F32_ONLY, CONTIGUOUS, true),
        native(OperationKind::Reduction, F32_ONLY, CONTIGUOUS, true),
        native_ranked(
            OperationKind::Normalization,
            F32_ONLY,
            CONTIGUOUS,
            1,
            MAX_RANK,
            true,
        ),
        CapabilityRule::new(
            OperationKind::Broadcast,
            F32_ONLY,
            CONTIGUOUS,
            0,
            MAX_RANK,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Reshape,
            F32_ONLY,
            CONTIGUOUS,
            0,
            MAX_RANK,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::MatMul,
            F32_ONLY,
            CONTIGUOUS,
            2,
            MAX_RANK,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Conv2d,
            F32_ONLY,
            CONTIGUOUS,
            3,
            4,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Pool2d,
            F32_ONLY,
            CONTIGUOUS,
            3,
            4,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
    ]
);

pub static METAL_CAPABILITIES: &[CapabilityRule] = metal_descriptor_operations!(
    descriptor_capability_rules,
    pointwise = FLOAT_DTYPES,
    broadcast = CUDA_STORAGE_DTYPES,
    reshape = CUDA_STORAGE_DTYPES,
    reduction = FLOAT_DTYPES,
    spatial = F32_ONLY,
    matmul = FLOAT_DTYPES,
    broadcast_training = FLOAT_DTYPES,
    reshape_training = FLOAT_DTYPES,
    pointwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    legacy = [
        native(
            OperationKind::Storage,
            CUDA_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Fill, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Random, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Pointwise, FLOAT_DTYPES, CONTIGUOUS, true),
        native(OperationKind::Reduction, FLOAT_DTYPES, CONTIGUOUS, true),
        native_ranked(
            OperationKind::Normalization,
            FLOAT_DTYPES,
            CONTIGUOUS,
            1,
            MAX_RANK,
            true,
        ),
        native(
            OperationKind::Broadcast,
            CUDA_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Broadcast, FLOAT_DTYPES, CONTIGUOUS, true),
        native(
            OperationKind::Reshape,
            CUDA_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Reshape, FLOAT_DTYPES, CONTIGUOUS, true),
        CapabilityRule::new(
            OperationKind::MatMul,
            FLOAT_DTYPES,
            CONTIGUOUS,
            2,
            MAX_RANK,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Conv2d,
            F32_ONLY,
            CONTIGUOUS,
            3,
            4,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Pool2d,
            F32_ONLY,
            CONTIGUOUS,
            3,
            4,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
    ]
);

static EMPTY_CAPABILITIES: &[CapabilityRule] = &[];

#[must_use]
pub fn registry(device: DeviceKind) -> CapabilityRegistry {
    let rules = match device {
        DeviceKind::Cpu => CPU_CAPABILITIES,
        DeviceKind::Cuda => CUDA_CAPABILITIES,
        DeviceKind::Wgpu => WGPU_CAPABILITIES,
        DeviceKind::Metal => METAL_CAPABILITIES,
        _ => EMPTY_CAPABILITIES,
    };
    CapabilityRegistry::new(rules)
}

#[must_use]
pub fn support(device: DeviceKind, query: &CapabilityQuery) -> SupportLevel {
    registry(device).support(query)
}

/// Exact support decisions generated by joining backend declarations to the
/// canonical operation catalog. A zero count is an explicit unsupported or
/// migration-blocked decision, never an omitted row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCoverageRow {
    pub operation: OperationKind,
    pub cpu_rules: usize,
    pub cuda_rules: usize,
    pub wgpu_rules: usize,
    pub metal_rules: usize,
}

#[must_use]
pub fn coverage_report() -> alloc::vec::Vec<BackendCoverageRow> {
    OPERATION_CATALOG
        .iter()
        .map(|entry| BackendCoverageRow {
            operation: entry.operation,
            cpu_rules: CPU_CAPABILITIES
                .iter()
                .filter(|rule| rule.operation == entry.operation)
                .count(),
            cuda_rules: CUDA_CAPABILITIES
                .iter()
                .filter(|rule| rule.operation == entry.operation)
                .count(),
            wgpu_rules: WGPU_CAPABILITIES
                .iter()
                .filter(|rule| rule.operation == entry.operation)
                .count(),
            metal_rules: METAL_CAPABILITIES
                .iter()
                .filter(|rule| rule.operation == entry.operation)
                .count(),
        })
        .collect()
}

#[cfg(test)]
mod catalog_tests {
    use super::*;
    use alloc::collections::BTreeSet;

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
                        operation: entry.operation,
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
                        operation: entry.operation,
                        dtype: DTypeId::F32,
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
}
