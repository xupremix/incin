//! The four backend capability tables.
//!
//! Each `pub static ..._CAPABILITIES` is unconditional: a capability claim is
//! data, and `registry`/`coverage_report` report every backend's regardless
//! of which backends are compiled in. Only the *executor* module that checks
//! a table's macro against its own `Execute<op::...>` implementations is
//! feature-gated (`super::declarations`'s re-exports carry that gate; these
//! tables invoke the ungated internal path instead).

use super::constants::{
    ALL_DTYPES, BOOL_ONLY, CONTIGUOUS, CPU_LAYOUTS, CUDA_BOOL_SAFE_STORAGE_DTYPES,
    CUDA_STORAGE_DTYPES, F32_AND_BOOL, F32_ONLY, FLOAT_DTYPES, INDEX_AND_F32_DTYPES, NON_QUANTIZED,
    PRECISE, Q8_ONLY,
};
use super::declarations::{
    cpu_descriptor_operations, cuda_descriptor_operations, metal_descriptor_operations,
    wgpu_descriptor_operations,
};
use super::rules::{
    composed_ranked, descriptor_capability_rules, descriptor_max_rank, descriptor_min_rank, native,
    native_ranked,
};
use incin_core::exec::{CapabilityRule, ImplementationKind, LayoutClass};
use incin_core::shapes::error::OperationKind;

/// CPU capability rules, generated from the CPU descriptor operation list.
pub static CPU_CAPABILITIES: &[CapabilityRule] = cpu_descriptor_operations!(
    descriptor_capability_rules,
    elementwise = FLOAT_DTYPES,
    broadcast = ALL_DTYPES,
    reshape = ALL_DTYPES,
    reduction = F32_ONLY,
    filling_dtypes = NON_QUANTIZED,
    sampling_dtypes = FLOAT_DTYPES,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
    embedding_dtypes = INDEX_AND_F32_DTYPES,
    broadcast_training = FLOAT_DTYPES,
    reshape_training = FLOAT_DTYPES,
    elementwise_layouts = CPU_LAYOUTS,
    broadcast_layouts = CPU_LAYOUTS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CPU_LAYOUTS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CPU_LAYOUTS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = NON_QUANTIZED,
    tensor_layouts = CPU_LAYOUTS,
    logical_dtypes = BOOL_ONLY,
    legacy = [
        CapabilityRule::new(
            OperationKind::ReshapeExact,
            ALL_DTYPES,
            &[LayoutClass::Strided],
            0,
            usize::MAX,
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
            usize::MAX,
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
            usize::MAX,
            false,
            PRECISE,
            ImplementationKind::Composed,
        ),
        CapabilityRule::new(
            OperationKind::Reshape,
            FLOAT_DTYPES,
            &[LayoutClass::Strided],
            0,
            usize::MAX,
            true,
            PRECISE,
            ImplementationKind::Composed,
        ),
        CapabilityRule::new(
            OperationKind::MatMul,
            F32_ONLY,
            CPU_LAYOUTS,
            2,
            usize::MAX,
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

/// CUDA capability rules, generated from the CUDA descriptor operation list.
pub static CUDA_CAPABILITIES: &[CapabilityRule] = cuda_descriptor_operations!(
    descriptor_capability_rules,
    elementwise = FLOAT_DTYPES,
    // `broadcast_as` launches `kernels/shape.cu`'s `shape_op`, whose kernel
    // signature is `const float*`/`float*` unconditionally - there is no
    // dtype-width parameter anywhere in the launch, so every element is read
    // and written at a 4-byte stride regardless of the storage's declared
    // dtype. For a 2-byte dtype (`f16`/`bf16`) that reads and writes past
    // the buffer `crate::bytes::byte_len` actually allocated; for an 8-byte
    // one (`f64`/`i64`) it silently touches only every other 4-byte half.
    // `f32` is the only dtype in `CUDA_STORAGE_DTYPES` this kernel is
    // byte-compatible with, so it is the only one the row may honestly
    // claim - narrowed here rather than in `shape.cu` itself, since fixing
    // the kernel to be dtype-parametric is separate, larger work.
    // `reshape` does not share this: it never launches `shape_op` at all,
    // only rewraps the same buffer under a new shape, so it stays byte-exact
    // for every dtype `CUDA_STORAGE_DTYPES` names.
    broadcast = F32_ONLY,
    reshape = CUDA_BOOL_SAFE_STORAGE_DTYPES,
    reduction = FLOAT_DTYPES,
    // `zeros`/`ones`/`full`/`arange`/`linspace`/`rand`/`randn` compute in
    // `f32` and hand the bit pattern to `cuda_from_f32`, which reinterprets
    // it as raw bytes rather than converting: every dtype whose element size
    // differs from 4 bytes fails `checked_storage_byte_len` before it could
    // return the wrong value, and `f32` is the only one both accepted by
    // `validate_cuda_storage_dtype` and byte-compatible. `NON_QUANTIZED` and
    // `FLOAT_DTYPES` were live but unused here until this session populated
    // the `filling`/`sampling` identity lists above; advertising them now
    // would repeat the exact mistake the coarse `Normalization` row made.
    filling_dtypes = F32_ONLY,
    sampling_dtypes = F32_ONLY,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
    embedding_dtypes = INDEX_AND_F32_DTYPES,
    // Same `shape_op` byte-width limit as the `broadcast` row above.
    broadcast_training = F32_ONLY,
    reshape_training = FLOAT_DTYPES,
    elementwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = F32_ONLY,
    tensor_layouts = CONTIGUOUS,
    logical_dtypes = BOOL_ONLY,
    legacy = [
        native(
            OperationKind::Storage,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        // Standalone rather than in the `filling = [...]` list above: both
        // route through `HostInterop::from_bytes`, verified safe for every
        // dtype `CUDA_BOOL_SAFE_STORAGE_DTYPES` names (see that constant's
        // own doc), which is wider than the `F32_ONLY` the group's other
        // five members are held to. No tape entry either way - a fresh
        // host-uploaded allocation records nothing to differentiate.
        native(
            OperationKind::TensorFromData,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(
            OperationKind::TensorFromBytes,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Fill, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Random, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Pointwise, FLOAT_DTYPES, CONTIGUOUS, true),
        native(OperationKind::Reduction, FLOAT_DTYPES, CONTIGUOUS, true),
        // No coarse `Normalization` row: the four exact identities below do
        // not share one rule shape, so a single family row could not state
        // them honestly, and `every_coarse_family_row_is_backed_by_a_native_
        // exact_row` does not require one - CPU's own Softmax member of the
        // family is itself `training = true` there only because CPU's kernel
        // pushes a real backward; the coarse row is not a promise every
        // backend has to fill.
        //
        // `layer_norm` and `batch_norm` are dedicated fused kernels (Welford
        // reduction; precomputed-statistics affine transform), so `Native`.
        // Neither pushes a tape entry yet, so `training = false`: a caller
        // inside a gradient-tracked context that reached either would get a
        // silently missing gradient rather than an error, which is what
        // `training` on this row exists to prevent.
        native_ranked(
            OperationKind::LayerNorm,
            F32_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::LayerNorm),
            descriptor_max_rank(OperationKind::LayerNorm),
            false,
        ),
        native_ranked(
            OperationKind::BatchNorm,
            F32_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::BatchNorm),
            descriptor_max_rank(OperationKind::BatchNorm),
            false,
        ),
        // `softmax` and `rms_norm` are answered by rewriting into other
        // catalog operations (subtract-max, exp, sum, divide; square, mean,
        // add, sqrt, divide, multiply) rather than a dedicated kernel, so
        // `Composed`. Every step in both rewrites already pushes its own
        // correct tape entry, so the composite's backward is the tape replay
        // over those entries, not new hand-derived math - `training = true`
        // is a verified claim here, not the conservative default the other
        // two rows above take.
        composed_ranked(
            OperationKind::Softmax,
            F32_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Softmax),
            descriptor_max_rank(OperationKind::Softmax),
            true,
        ),
        composed_ranked(
            OperationKind::RmsNorm,
            F32_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::RmsNorm),
            descriptor_max_rank(OperationKind::RmsNorm),
            true,
        ),
        // `F32_ONLY`, not `CUDA_STORAGE_DTYPES`/`FLOAT_DTYPES`: see the
        // `shape_op` byte-width comment on `CUDA_CAPABILITIES`'s own
        // `broadcast` field above. The coarse row has to match the exact
        // `BroadcastAs` row it stands beside, or `doctor`'s coarse probe
        // and a real `broadcast_as` call would disagree about what runs.
        native(OperationKind::Broadcast, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Broadcast, F32_ONLY, CONTIGUOUS, true),
        native(
            OperationKind::Reshape,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Reshape, FLOAT_DTYPES, CONTIGUOUS, true),
        CapabilityRule::new(
            OperationKind::MatMul,
            F32_ONLY,
            CONTIGUOUS,
            2,
            usize::MAX,
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
        // `where_cond`/`masked_fill` (`cuda/ops/select.rs`) are the
        // consumers a `bool` mask needs to be reachable at all: without them
        // a `cmp_*` result could be produced and reshaped but never fed back
        // into a float computation. `F32_AND_BOOL` rather than `F32_ONLY`
        // because both take a `bool` mask alongside `f32` data and
        // `dispatch::execute` checks every operand against this one row -
        // see that constant's own doc for why a shared-group row could not
        // state this.
        native_ranked(
            OperationKind::WhereCond,
            F32_AND_BOOL,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::WhereCond),
            descriptor_max_rank(OperationKind::WhereCond),
            true,
        ),
        native_ranked(
            OperationKind::MaskedFill,
            F32_AND_BOOL,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::MaskedFill),
            descriptor_max_rank(OperationKind::MaskedFill),
            true,
        ),
        // `logical_and`/`logical_or`/`logical_not` (`cuda/ops/logical.rs`):
        // dedicated kernels over `bool` throughout, `BOOL_ONLY` rather than
        // `F32_AND_BOOL` since there is no mixed-dtype operand here to union
        // against.
        native_ranked(
            OperationKind::LogicalAnd,
            BOOL_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::LogicalAnd),
            descriptor_max_rank(OperationKind::LogicalAnd),
            true,
        ),
        native_ranked(
            OperationKind::LogicalOr,
            BOOL_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::LogicalOr),
            descriptor_max_rank(OperationKind::LogicalOr),
            true,
        ),
        native_ranked(
            OperationKind::LogicalNot,
            BOOL_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::LogicalNot),
            descriptor_max_rank(OperationKind::LogicalNot),
            true,
        ),
    ]
);

/// WGPU capability rules, generated from the WGPU descriptor operation list.
pub static WGPU_CAPABILITIES: &[CapabilityRule] = wgpu_descriptor_operations!(
    descriptor_capability_rules,
    elementwise = F32_ONLY,
    broadcast = F32_ONLY,
    reshape = F32_ONLY,
    reduction = F32_ONLY,
    // `validate_wgpu_dtype` rejects anything but `f32` outright, and the
    // creation methods never pass `dtype` into the buffer they build, so
    // `f32` is not just the safe claim here, it is the only one that can
    // ever succeed. Same reasoning as the CUDA table above.
    filling_dtypes = F32_ONLY,
    sampling_dtypes = F32_ONLY,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
    embedding_dtypes = INDEX_AND_F32_DTYPES,
    broadcast_training = F32_ONLY,
    reshape_training = F32_ONLY,
    elementwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = F32_ONLY,
    tensor_layouts = CONTIGUOUS,
    logical_dtypes = BOOL_ONLY,
    legacy = [
        native(OperationKind::Storage, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Fill, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Random, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Pointwise, F32_ONLY, CONTIGUOUS, true),
        native(OperationKind::Reduction, F32_ONLY, CONTIGUOUS, true),
        // No legacy Normalization row: no WGPU kernel backs the family. The
        // typed `normalization = []` list above still advertises none, and a
        // coarse row here would claim native LayerNorm/BatchNorm support this
        // backend has never executed.
        //
        // `softmax` is the one member that does run, so it takes a standalone
        // row rather than joining a family row that would drag the other four
        // in with it. It is answered by rewriting into `max_keepdim`, `sub`,
        // `exp`, `sum_keepdim` and `log` rather than by a kernel of its own,
        // so `Composed`; every one of those steps already pushes its own
        // correct tape entry, so the composite's backward is the tape replay
        // over them rather than new hand-derived math, which is what makes
        // `training = true` a verified claim here instead of a hopeful one.
        // Same reasoning, and the same row shape, as CUDA's `softmax` above.
        composed_ranked(
            OperationKind::Softmax,
            F32_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Softmax),
            descriptor_max_rank(OperationKind::Softmax),
            true,
        ),
        // `rms_norm` on the same basis, and for the same reason it is
        // `Composed` on CUDA: it rewrites into `mul`, `mean_keepdim`,
        // `add_scalar`, `sqrt` and `div`, each of which pushes its own tape
        // entry, so the backward is the replay rather than new math.
        composed_ranked(
            OperationKind::RmsNorm,
            F32_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::RmsNorm),
            descriptor_max_rank(OperationKind::RmsNorm),
            true,
        ),
        CapabilityRule::new(
            OperationKind::Broadcast,
            F32_ONLY,
            CONTIGUOUS,
            0,
            usize::MAX,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Reshape,
            F32_ONLY,
            CONTIGUOUS,
            0,
            usize::MAX,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::MatMul,
            F32_ONLY,
            CONTIGUOUS,
            2,
            usize::MAX,
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

/// Metal capability rules, generated from the Metal descriptor operation list.
pub static METAL_CAPABILITIES: &[CapabilityRule] = metal_descriptor_operations!(
    descriptor_capability_rules,
    elementwise = F32_ONLY,
    broadcast = CUDA_STORAGE_DTYPES,
    reshape = CUDA_STORAGE_DTYPES,
    reduction = F32_ONLY,
    // `zeros`/`full`/`ones`/`arange`/`linspace` compute in `f32` and hand
    // the bit pattern to `MetalStorage::from_bytes` (or, for `zeros`,
    // `MetalStorage::zeros`) under whatever `dtype` was requested, without a
    // numeric conversion. Same reasoning as the CUDA table above.
    filling_dtypes = F32_ONLY,
    sampling_dtypes = F32_ONLY,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
    embedding_dtypes = INDEX_AND_F32_DTYPES,
    broadcast_training = F32_ONLY,
    reshape_training = F32_ONLY,
    elementwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = F32_ONLY,
    tensor_layouts = CONTIGUOUS,
    logical_dtypes = BOOL_ONLY,
    legacy = [
        native(
            OperationKind::Storage,
            CUDA_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Fill, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Random, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Pointwise, F32_ONLY, CONTIGUOUS, true),
        native(OperationKind::Reduction, F32_ONLY, CONTIGUOUS, true),
        // No legacy Normalization row: no Metal kernel backs it. The typed
        // `normalization = []` list above already advertises none, honestly;
        // a coarse row here claimed native LayerNorm/BatchNorm support this
        // backend has never executed.
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
            usize::MAX,
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
