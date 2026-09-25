//! The four backend capability tables.
//!
//! Each `pub static ..._CAPABILITIES` is unconditional: a capability claim is
//! data, and `registry`/`coverage_report` report every backend's regardless
//! of which backends are compiled in. Only the *executor* module that checks
//! a table's macro against its own `Execute<op::...>` implementations is
//! feature-gated (`super::declarations`'s re-exports carry that gate; these
//! tables invoke the ungated internal path instead).

use super::constants::{
    ALL_DTYPES, BOOL_ONLY, CONTIGUOUS, CPU_LAYOUTS, CUDA_BOOL_SAFE_STORAGE_DTYPES, CUDA_LAYOUTS,
    CUDA_STORAGE_DTYPES, F32_AND_BOOL, F32_AND_F64, F32_ONLY, FLOAT_DTYPES, INDEX_AND_F32_DTYPES,
    NON_QUANTIZED, PRECISE, Q8_ONLY, WGPU_STORAGE_DTYPES,
};
use super::declarations::{
    cpu_descriptor_operations, cuda_descriptor_operations, metal_descriptor_operations,
    rocm_descriptor_operations, wgpu_descriptor_operations,
};
use super::rules::{
    accelerator_max_rank, composed_ranked, descriptor_capability_rules, descriptor_max_rank,
    descriptor_min_rank, descriptor_training, native, native_ranked,
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
    // Issue #90: `matmul_forward`/`batched_gemm` read every float storage
    // dtype through the stride-aware accessor and write the result back in
    // the operand's own, so the exact `MatMulExact` row and the composed
    // rows that rewrite into it (`bmm`/`addmm`/`linear`) honestly claim
    // FLOAT_DTYPES. The executor's own refusal is narrower than this row
    // in the one direction a single row cannot state: both operands must
    // carry the *same* float dtype (a mixed pair fails `DTypeMismatch`
    // host-side in `ensure_matmul_dtypes` before any kernel runs, and a
    // non-float fails `UnsupportedDType`), which `dispatch::execute`'s
    // per-operand application of this one union cannot express - the same
    // split CUDA's coarse `MatMul` comment documents.
    matmul = FLOAT_DTYPES,
    normalization_dtypes = F32_ONLY,
    embedding_dtypes = INDEX_AND_F32_DTYPES,
    // Issue #104: the native fused-attention kernel serves f32 and f64.
    fused_attention_dtypes = F32_AND_F64,
    broadcast_training = FLOAT_DTYPES,
    reshape_training = FLOAT_DTYPES,
    elementwise_layouts = CPU_LAYOUTS,
    broadcast_layouts = CPU_LAYOUTS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CPU_LAYOUTS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CPU_LAYOUTS,
    fused_attention_layouts = CPU_LAYOUTS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = NON_QUANTIZED,
    tensor_layouts = CPU_LAYOUTS,
    logical_dtypes = BOOL_ONLY,
    // Host loops are rank-agnostic: the descriptor bound stands everywhere.
    max_rank = descriptor_max_rank,
    legacy = [
        // Composed, meaning it materializes the strided operand and reshapes
        // the copy. Materializing walks the logical index space one value at a
        // time, and a block encoding has no per-value access: thirty-two
        // logical values share one scale, so `CpuStorage`'s copy refuses `q8_0`
        // by name. `NON_QUANTIZED` rather than `ALL_DTYPES` for that reason.
        // The contiguous reshape above keeps every dtype, because it rewrites
        // metadata and never reads a value.
        CapabilityRule::new(
            OperationKind::ReshapeExact,
            NON_QUANTIZED,
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
        // Issue #90: `matmul.cu`-equivalent host kernels now run one GEMM
        // path per float storage dtype, so the coarse row matches the exact
        // `MatMulExact` row (which sits in `declarations`'s `matmul` group
        // on `FLOAT_DTYPES`) rather than trailing it at `F32_ONLY` - a
        // coarse row that understates the exact row beside it refuses
        // reachable work just as a wider one would over-advertise. The
        // executor's own refusals are narrower than this row in the one
        // direction a single row cannot state: `ensure_matmul_dtypes`
        // requires both operands to carry the *same* dtype (a mixed
        // `f16`/`f32` pair fails `DTypeMismatch` host-side before any
        // kernel runs) and refuses every non-float dtype
        // (`i64`/`bool`/`u8`/`u32`/`q8_0`). `dispatch::execute` applies
        // this one set to every operand in turn, so like `F32_AND_BOOL`
        // and `INDEX_AND_F32_DTYPES` the row states the union of what the
        // operands may carry, and the equality/dtype-split it cannot
        // express is enforced fail-closed inside the executor.
        CapabilityRule::new(
            OperationKind::MatMul,
            FLOAT_DTYPES,
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
    // `broadcast_as` launches `kernels/shape.cu`'s `shape_op` through
    // width-parametric entry points (`shape_op_8bit`/`16bit`/`32bit`/`64bit`,
    // selected by element width at launch), and the launcher passes byte
    // buffers through instead of reinterpreting them as `f32`. Measured on
    // hardware across transpose, broadcast, narrow and concat: `bool`,
    // `f16`, `bf16`, `f32`, `f64` and `i64` all land byte-exact, so the row
    // claims the dense storage set. `q8_0` stays out: a block encoding has
    // no element width to move by, and the kernels refuse it rather than
    // reinterpret block bytes as scalars (pinned by test).
    // `reshape` does not share this: it never launches `shape_op` at all,
    // only rewraps the same buffer under a new shape, so it stays byte-exact
    // for every dtype `CUDA_STORAGE_DTYPES` names.
    broadcast = CUDA_BOOL_SAFE_STORAGE_DTYPES,
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
    // Empty `fused_attention` group on CUDA; the dtype set rides along
    // unused, per the file's convention for empty groups.
    fused_attention_dtypes = F32_AND_F64,
    // Same measured set as the `broadcast` row above, for the reason stated
    // there: the training path moves the same bytes.
    broadcast_training = CUDA_BOOL_SAFE_STORAGE_DTYPES,
    reshape_training = FLOAT_DTYPES,
    // The one row widened past `CONTIGUOUS`: the strided elementwise kernel
    // exists, is benchmarked in `view_cost_bench`, and beats materialising for
    // a single consumer. It was unreachable through the descriptor path because
    // this row refused it. The others stay narrow until each has the same
    // evidence; a row is widened by a test that fails without it.
    elementwise_layouts = CUDA_LAYOUTS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    fused_attention_layouts = CONTIGUOUS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = F32_ONLY,
    tensor_layouts = CONTIGUOUS,
    logical_dtypes = BOOL_ONLY,
    // The shape-kernel identities (`broadcast_as`, transpose/narrow, the
    // comparison broadcasts, the `bool`-mask broadcasts) refuse rank 7+ at
    // launch; `reshape`/`concat`/the fused families keep descriptor bounds.
    max_rank = accelerator_max_rank,
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
        // reduction; per-channel affine), so `Native`. `layer_norm` pushes a
        // real tape entry replaying the forward's saved statistics, so
        // `training = true` is a verified claim there, proven against the
        // CPU reference on hardware. `batch_norm` claims the same since
        // #123: its training forward saves per-channel *batch* statistics
        // (never the running estimates) whenever the grad mode records, its
        // fused backward replays them into input/weight/bias gradients, and
        // `Execute<op::BatchNorm>` delegates `attributes.training` to that
        // tape-tracked method - so a training query answered `no` here
        // would understate what the executor now admits. The inference path
        // (running statistics) is unchanged and records nothing, which is
        // correct there: its statistics are constants w.r.t. the input.
        native_ranked(
            OperationKind::LayerNorm,
            F32_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::LayerNorm),
            descriptor_max_rank(OperationKind::LayerNorm),
            true,
        ),
        native_ranked(
            OperationKind::BatchNorm,
            F32_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::BatchNorm),
            descriptor_max_rank(OperationKind::BatchNorm),
            true,
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
        // The dense storage set, matching the exact `BroadcastAs` rows above:
        // the coarse row has to match the exact row it stands beside, or
        // `doctor`'s coarse probe and a real `broadcast_as` call would
        // disagree about what runs.
        native_ranked(
            OperationKind::Broadcast,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            0,
            6,
            false,
        ),
        native_ranked(
            OperationKind::Broadcast,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            0,
            6,
            true,
        ),
        // The movement operations, measured dtype by dtype on hardware across
        // transpose, broadcast, narrow and concat (see
        // `tests/cuda_shape_dtypes.rs`): each moves bytes by element width
        // with no arithmetic, pushes a real tape entry, and refuses block
        // encodings. One row per training mode, like the coarse `Broadcast`
        // pair above.
        //
        // TRACKED DEVIATION for the `TransposeExact` pair below (issue #113,
        // orchestrator decision 2026-09-25): the operation contract is views
        // everywhere (`LayoutRule::ViewWhenPossible`), and this backend still
        // copies through `launch_transpose` because its matmul/reduce
        // consumers refuse strided operands and there is no device here to
        // verify strided support on. Pending hardware-verified strided
        // support; the copy is recorded here, not blessed.
        native_ranked(
            OperationKind::TransposeExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::TransposeExact),
            accelerator_max_rank(OperationKind::TransposeExact),
            false,
        ),
        native_ranked(
            OperationKind::TransposeExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::TransposeExact),
            accelerator_max_rank(OperationKind::TransposeExact),
            true,
        ),
        native_ranked(
            OperationKind::TransposeView,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::TransposeView),
            accelerator_max_rank(OperationKind::TransposeView),
            false,
        ),
        native_ranked(
            OperationKind::TransposeView,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::TransposeView),
            accelerator_max_rank(OperationKind::TransposeView),
            true,
        ),
        native_ranked(
            OperationKind::Narrow,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Narrow),
            accelerator_max_rank(OperationKind::Narrow),
            false,
        ),
        native_ranked(
            OperationKind::Narrow,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Narrow),
            accelerator_max_rank(OperationKind::Narrow),
            true,
        ),
        native_ranked(
            OperationKind::ConcatExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::ConcatExact),
            descriptor_max_rank(OperationKind::ConcatExact),
            false,
        ),
        native_ranked(
            OperationKind::ConcatExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::ConcatExact),
            descriptor_max_rank(OperationKind::ConcatExact),
            true,
        ),
        native(
            OperationKind::Reshape,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Reshape, FLOAT_DTYPES, CONTIGUOUS, true),
        // Issue: close the remaining narrow CUDA dtype capability rows.
        // Each row here widens one identity past the `F32_ONLY` its shared
        // declaration-group row still states. Legacy rows render first and
        // `support()` is any-row-match, so a real invocation resolves
        // against this wider claim while the group row keeps its narrower
        // documented floor. Evidence per family:
        // - The eight composed identities rewrite into the measured wide
        //   byte-movement kernels (`tests/cuda_shape_dtypes.rs`'s
        //   byte-exact matrix across transpose, broadcast, narrow and
        //   concat): `flatten`/`squeeze`/`unsqueeze` are metadata-only
        //   buffer rewraps (same as the wide `reshape` row above), `stack`/
        //   `slice`/`chunk`/`split` rewrite through narrow+concat, and
        //   `broadcast_left` through `broadcast_as`. Every step pushes a
        //   real tape entry, so `training = true` is verified, not the
        //   conservative default. `ImplementationKind::Composed` matches
        //   what the group row already reports. Rank and training follow
        //   `descriptor_min_rank`/`accelerator_max_rank`, the same helpers
        //   the group row uses.
        // - `PixelShuffle`/`Unfold` are native: their executors are pure
        //   reshape/transpose and narrow+unsqueeze+concat+transpose chains
        //   (`cuda/executor.rs`) over those same byte-movement kernels,
        //   each pushing tape. `native_ranked` with `ImplementationKind::
        //   Native` matches the group row; ranks follow the descriptor
        //   (`PixelShuffle` is exactly 4-D, `Unfold` is rank 1..).
        // - `TensorToBytes` widens per the `readback` comment in
        //   `declarations.rs`: `HostInterop::to_bytes` only length-checks
        //   against `checked_storage_byte_len` and never reinterprets
        //   elements, so every dtype CUDA can hold round-trips. No tape
        //   entry - a device-to-host copy records nothing to differentiate.
        composed_ranked(
            OperationKind::FlattenExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::FlattenExact),
            accelerator_max_rank(OperationKind::FlattenExact),
            true,
        ),
        composed_ranked(
            OperationKind::SqueezeExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::SqueezeExact),
            accelerator_max_rank(OperationKind::SqueezeExact),
            true,
        ),
        composed_ranked(
            OperationKind::UnsqueezeExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::UnsqueezeExact),
            accelerator_max_rank(OperationKind::UnsqueezeExact),
            true,
        ),
        composed_ranked(
            OperationKind::StackExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::StackExact),
            accelerator_max_rank(OperationKind::StackExact),
            true,
        ),
        composed_ranked(
            OperationKind::SliceExact,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::SliceExact),
            accelerator_max_rank(OperationKind::SliceExact),
            true,
        ),
        composed_ranked(
            OperationKind::BroadcastLeft,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::BroadcastLeft),
            accelerator_max_rank(OperationKind::BroadcastLeft),
            true,
        ),
        composed_ranked(
            OperationKind::Chunk,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Chunk),
            accelerator_max_rank(OperationKind::Chunk),
            true,
        ),
        composed_ranked(
            OperationKind::Split,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Split),
            accelerator_max_rank(OperationKind::Split),
            true,
        ),
        native_ranked(
            OperationKind::PixelShuffle,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            4,
            4,
            true,
        ),
        native_ranked(
            OperationKind::Unfold,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            1,
            usize::MAX,
            true,
        ),
        native(
            OperationKind::TensorToBytes,
            CUDA_BOOL_SAFE_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        // Issue #90 (98df2b6c): `matmul.cu` exports one GEMM entry per float
        // storage dtype, so the coarse row now matches the exact
        // `MatMulExact` row (which sits in `declarations`'s `reduction`
        // group on `FLOAT_DTYPES`) rather than trailing it at `F32_ONLY` -
        // a coarse row that understates the exact row beside it refuses
        // reachable work just as a wider one would over-advertise. The
        // executor's own refusals are narrower than this row in the one
        // direction a single row cannot state: `launch_matmul` requires
        // both operands to carry the *same* dtype (a mixed `f16`/`f32`
        // pair fails `DTypeMismatch` host-side before any kernel launch)
        // and refuses every dtype without a `matmul.cu` entry
        // (`i64`/`bool`/`u8`/`u32`/`q8_0`). `dispatch::execute` applies
        // this one set to every operand in turn, so like `F32_AND_BOOL`
        // and `INDEX_AND_F32_DTYPES` the row states the union of what the
        // operands may carry, and the equality/dtype-split it cannot
        // express is enforced fail-closed inside the executor.
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
            accelerator_max_rank(OperationKind::WhereCond),
            true,
        ),
        native_ranked(
            OperationKind::MaskedFill,
            F32_AND_BOOL,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::MaskedFill),
            accelerator_max_rank(OperationKind::MaskedFill),
            true,
        ),
        // `gather`/`scatter`/`index_select`: each takes an integer index
        // operand beside `f32` data, and `dispatch::execute`'s
        // `admit_invocation` checks every operand against the one resolved
        // row in turn - so a row narrower than the union of what the
        // operands actually carry makes the operation unreachable, the index
        // operand failing dtype admission before either kernel ever launches.
        // That is the exact bug class `F32_AND_BOOL` and
        // `INDEX_AND_F32_DTYPES` document, and it is what these standalone
        // rows fix: the shared `native_tensor` group's `tensor_dtypes` is
        // `F32_ONLY`, which cannot state the integer index half.
        // `INDEX_AND_F32_DTYPES` is the union, not a claim either operand
        // may be *either* dtype - the descriptor's own per-operand contract
        // (`exec/catalog`'s `index_input` slot requires the index to be
        // integer) and `cuda::ops::shape`'s f32-only kernels enforce the
        // real, tighter split this row cannot state on its own. `CONTIGUOUS`
        // rather than the group's wider layouts because `launch_gather` and
        // `launch_scatter` compute their input strides through
        // `contiguous_strides` and never read the operand's actual
        // `meta.strides`: admitting `strided` here would let a
        // `transpose_view` reach a kernel that reads it as dense and
        // silently return wrong values. Rank and training follow the
        // descriptor, as the group row did.
        native_ranked(
            OperationKind::Gather,
            INDEX_AND_F32_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Gather),
            descriptor_max_rank(OperationKind::Gather),
            true,
        ),
        native_ranked(
            OperationKind::Scatter,
            INDEX_AND_F32_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Scatter),
            descriptor_max_rank(OperationKind::Scatter),
            true,
        ),
        native_ranked(
            OperationKind::IndexSelect,
            INDEX_AND_F32_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::IndexSelect),
            descriptor_max_rank(OperationKind::IndexSelect),
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
            accelerator_max_rank(OperationKind::LogicalAnd),
            false,
        ),
        native_ranked(
            OperationKind::LogicalOr,
            BOOL_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::LogicalOr),
            accelerator_max_rank(OperationKind::LogicalOr),
            false,
        ),
        native_ranked(
            OperationKind::LogicalNot,
            BOOL_ONLY,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::LogicalNot),
            accelerator_max_rank(OperationKind::LogicalNot),
            false,
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
    // Issue #90 audit: unlike CPU/CUDA, this backend's whole matmul family
    // (`MatMulExact`/`bmm`/`addmm`/`linear`/`dot`/`outer`/SDPA) stays
    // `F32_ONLY` honestly - `shaders/matmul.wgsl` is `array<f32>`
    // throughout and every `Execute` impl hardcodes `::<f32>`.
    // `device.rs` requests `Features::empty()` (no `SHADER_F16` feature
    // query) and `validate_wgpu_dtype` refuses `f16`/`bf16`/`f64`
    // outright, so nothing wider can reach the kernel. An `f16` claim
    // would need the adapter feature queried *and* required at
    // `request_device`, an `f16` shader variant, and a widened validator
    // first - none of which exist today.
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
    embedding_dtypes = INDEX_AND_F32_DTYPES,
    // Empty `fused_attention` group here; the dtype set rides along
    // unused, per the file's convention for empty groups.
    fused_attention_dtypes = F32_AND_F64,
    broadcast_training = F32_ONLY,
    reshape_training = F32_ONLY,
    elementwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    fused_attention_layouts = CONTIGUOUS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = F32_ONLY,
    tensor_layouts = CONTIGUOUS,
    logical_dtypes = BOOL_ONLY,
    // `broadcast_as`/`transpose` route through `prepare_shape_params`' fixed
    // rank-6 block, so the accelerator bound applies; `reshape` stays
    // unbounded (a metadata-only buffer rewrap on this backend).
    max_rank = accelerator_max_rank,
    legacy = [
        // Storage (allocation / `to_bytes` / `from_bytes`) admits every
        // dtype `validate_wgpu_dtype` now holds: `f32`, `bool` as physical
        // `f32`, and the integer index widths `u8`/`u32`/`i64`. Narrower
        // than the CPU row (no `f64`/`f16`/`bf16`/`q8_0`) because no WGPU
        // kernel here reads them and `from_bytes`/`to_bytes` size by
        // `dtype.size_bytes` for exactly these five.
        native(
            OperationKind::Storage,
            WGPU_STORAGE_DTYPES,
            CONTIGUOUS,
            false
        ),
        // Fill/Random stay F32-only: `creation.rs`'s `full`/`zeros`/`ones`/
        // `arange`/`linspace`/`rand`/`randn` build a host `Vec<f32>`
        // regardless of the requested dtype, so admitting anything else
        // would upload `f32` bits under a non-`f32` meta.
        native(OperationKind::Fill, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Random, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Pointwise, F32_ONLY, CONTIGUOUS, true),
        native(OperationKind::Reduction, F32_ONLY, CONTIGUOUS, true),
        // `TensorFromData`/`TensorFromBytes` ride the fill group's
        // `filling_dtypes` above, which is `F32_ONLY`. These standalone
        // rows widen only the two data-creation identities whose payload
        // path (`impl_data_creation_executors!` → `HostInterop::from_bytes`)
        // now genuinely round-trips every dtype in `WGPU_STORAGE_DTYPES`:
        // the fill group's `Zeros`/`Ones`/`Full`/`Arange`/`Linspace` and
        // the `var_*` siblings still build `Vec<f32>` and stay narrow.
        // Multiple rows for one operation are fine: capability resolution is
        // "any row matches", so a `bool` `TensorFromBytes` is admitted by
        // this row while an `f32` `Zeros` is admitted by the group row.
        native(
            OperationKind::TensorFromData,
            WGPU_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(
            OperationKind::TensorFromBytes,
            WGPU_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        // `masked_fill`/`where_cond`: the consumers a `bool` mask needs to
        // be reachable at all. `F32_AND_BOOL` rather than `F32_ONLY`
        // because both take a `bool` mask alongside `f32` data and
        // `dispatch::execute` checks every operand against this one row —
        // see that constant's own doc for why a shared-group row could not
        // state this. Rank-capped through `accelerator_max_rank` because
        // the mask broadcast routes through `prepare_shape_params`' fixed
        // rank-6 block. CUDA carries the identical pair at lines 441-456
        // above; these rows are their WGPU twins, `Native` over the new
        // `select.wgsl` modes.
        native_ranked(
            OperationKind::MaskedFill,
            F32_AND_BOOL,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::MaskedFill),
            accelerator_max_rank(OperationKind::MaskedFill),
            true,
        ),
        native_ranked(
            OperationKind::WhereCond,
            F32_AND_BOOL,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::WhereCond),
            accelerator_max_rank(OperationKind::WhereCond),
            true,
        ),
        // `gather`/`index_select`: host-walk forwards whose backward is the
        // same scatter-add CPU's `gather_storage`/`index_select_storage`
        // push (ONE TapeEntry, `input_ids = vec![t.id]`, integer index
        // off-tape). `INDEX_AND_F32_DTYPES` is the union of the integer
        // index operand and the f32 data operand for the same reason
        // `embedding`'s row uses it. No `accelerator_max_rank` cap: the
        // host walk is rank-agnostic, so `descriptor_max_rank` (unbounded)
        // stands. CUDA registers neither (its kernels live under
        // `embedding`'s group); these are standalone because WGPU's
        // `embedding` group is the only place the shared macro would place
        // them and it is already committed to `EmbeddingExact` alone.
        native_ranked(
            OperationKind::Gather,
            INDEX_AND_F32_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::Gather),
            descriptor_max_rank(OperationKind::Gather),
            true,
        ),
        native_ranked(
            OperationKind::IndexSelect,
            INDEX_AND_F32_DTYPES,
            CONTIGUOUS,
            descriptor_min_rank(OperationKind::IndexSelect),
            descriptor_max_rank(OperationKind::IndexSelect),
            true,
        ),
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
            // `broadcast_storage` routes through `prepare_shape_params`' fixed
            // rank-6 block; the typed `BroadcastAs` row above is capped the
            // same way, so the coarse row has to match it.
            6,
            false,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::Broadcast,
            F32_ONLY,
            CONTIGUOUS,
            0,
            // `broadcast_storage` routes through `prepare_shape_params`' fixed
            // rank-6 block; the typed `BroadcastAs` row above is capped the
            // same way, so the coarse row has to match it.
            6,
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
            false,
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
    // `binary_op_metal` reinterprets both operands' bytes as `f32`
    // (`backend.rs`/`executor.rs` hardcode `::<f32>` throughout), so only
    // `f32` is honest here. `reshape` keeps the wider set deliberately:
    // `reshape_metal` rewraps bytes without reading a value, so it is
    // byte-exact for every dtype the validator admits.
    broadcast = F32_ONLY,
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
    // Empty `fused_attention` group here; the dtype set rides along
    // unused, per the file's convention for empty groups.
    fused_attention_dtypes = F32_AND_F64,
    broadcast_training = F32_ONLY,
    reshape_training = F32_ONLY,
    elementwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    fused_attention_layouts = CONTIGUOUS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = F32_ONLY,
    tensor_layouts = CONTIGUOUS,
    logical_dtypes = BOOL_ONLY,
    // Host loops are rank-agnostic: the descriptor bound stands everywhere.
    max_rank = descriptor_max_rank,
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
        // No legacy Normalization row: no Metal kernel backs the whole
        // coarse family. The typed rows below cover only what this backend
        // executes — `softmax`/`log_softmax`/`layer_norm`/`rms_norm` — and a
        // coarse row here would also claim native BatchNorm/GroupNorm
        // support this backend has never executed.
        // `broadcast_as` computes through `binary_op_metal`, which reinterprets
        // both operands as `f32`: the wider claim used to bless silent
        // misreads of `f16`/`bf16`/`f64`/`i64` bytes. `reshape` keeps the
        // wider set: `reshape_metal` rewraps bytes without reading a value.
        native(OperationKind::Broadcast, F32_ONLY, CONTIGUOUS, false),
        native(OperationKind::Broadcast, F32_ONLY, CONTIGUOUS, true),
        native(
            OperationKind::Reshape,
            CUDA_STORAGE_DTYPES,
            CONTIGUOUS,
            false,
        ),
        native(OperationKind::Reshape, FLOAT_DTYPES, CONTIGUOUS, true),
        // `matmul_metal` is the same `f32`-reinterpretation story as
        // `broadcast_as` above: `F32_ONLY`, matching the typed `MatMulExact`
        // row, not the `FLOAT_DTYPES` this row used to claim.
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
        // No `Conv2d`/`Pool2d` rows: `MetalBackendImpl::conv2d`/`max_pool2d`/
        // `avg_pool2d` always return `Err(unsupported(..))`, so any row here
        // would advertise an operation that can never execute. The
        // `Execute` impls stay as loud errors behind the registry's refusal.
    ]
);

/// ROCm capability rules: exactly none until HIP kernels land (issue #6).
///
/// Invoked through the same macro as the other backends so the declaration
/// stays the single source of group membership; every group is empty, so
/// this static claims nothing. Dtype/layout parameters ride along unused,
/// per the file's convention for empty groups.
pub static ROCM_CAPABILITIES: &[CapabilityRule] = rocm_descriptor_operations!(
    descriptor_capability_rules,
    elementwise = F32_ONLY,
    broadcast = F32_ONLY,
    reshape = F32_ONLY,
    reduction = F32_ONLY,
    filling_dtypes = F32_ONLY,
    sampling_dtypes = F32_ONLY,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
    embedding_dtypes = INDEX_AND_F32_DTYPES,
    fused_attention_dtypes = F32_AND_F64,
    broadcast_training = F32_ONLY,
    reshape_training = F32_ONLY,
    elementwise_layouts = CONTIGUOUS,
    broadcast_layouts = CONTIGUOUS,
    reshape_layouts = CONTIGUOUS,
    reduction_layouts = CONTIGUOUS,
    spatial_layouts = CONTIGUOUS,
    matmul_layouts = CONTIGUOUS,
    fused_attention_layouts = CONTIGUOUS,
    quantized_dtypes = Q8_ONLY,
    quantized_layouts = CONTIGUOUS,
    tensor_dtypes = F32_ONLY,
    tensor_layouts = CONTIGUOUS,
    logical_dtypes = BOOL_ONLY,
    max_rank = descriptor_max_rank,
    legacy = []
);
