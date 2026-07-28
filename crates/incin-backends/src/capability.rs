//! Authoritative native-backend capability registrations.

use incin_core::exec::{
    Capabilities, CapabilityQuery, CapabilityRegistry, CapabilityRule, ImplementationKind,
    LayoutClass, MathMode, SupportLevel,
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

pub static CPU_CAPABILITIES: &[CapabilityRule] = &[
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
];

pub static CUDA_CAPABILITIES: &[CapabilityRule] = &[
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
];

pub static WGPU_CAPABILITIES: &[CapabilityRule] = &[
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
];

static EMPTY_CAPABILITIES: &[CapabilityRule] = &[];

#[must_use]
pub fn registry(device: DeviceKind) -> CapabilityRegistry {
    let rules = match device {
        DeviceKind::Cpu => CPU_CAPABILITIES,
        DeviceKind::Cuda => CUDA_CAPABILITIES,
        DeviceKind::Wgpu => WGPU_CAPABILITIES,
        _ => EMPTY_CAPABILITIES,
    };
    CapabilityRegistry::new(rules)
}

#[must_use]
pub fn support(device: DeviceKind, query: &CapabilityQuery) -> SupportLevel {
    registry(device).support(query)
}
