//! Backend-specific descriptor-operation macros.
//!
//! Each `*_descriptor_operations!` macro is that backend's declaration of the
//! identities it advertises, grouped by rule shape rather than operation
//! family (see the doc comment on `cpu_descriptor_operations!` below). The
//! macro is always defined, feature or not: a capability claim is data the
//! registry reports regardless of which backends are compiled in, so the
//! four `pub(crate)` re-exports below carry no `#[cfg]` and `super::tables`
//! can invoke every macro unconditionally. Only the *checking* side is
//! feature-gated: each backend's own executor module consumes its macro
//! through `crate::capability::{cpu,cuda,wgpu,metal}_descriptor_operations!`,
//! re-exported with that backend's `#[cfg]` in `super` (`capability::mod`),
//! so the coverage assertion only exists when the backend is compiled.

// Single declaration consumed by capability generation below, by the grouped
// legacy descriptor executors, and by the canonical per-identity executors in
// `cpu::canonical`. Adding an identity here changes what execution admits and
// what capability queries report in the same edit, and the canonical module
// turns the third consumer into a compile-time obligation: a row advertised
// without an `Execute<op::...>` implementation does not build.
//
// A group is a *rule shape*, not an operation family. Two identities belong to
// the same group when they produce an identical `CapabilityRule` apart from the
// operation name and the per-identity rank bounds, whichever trait their kernel
// happens to live on today. Grouping by family instead would mean a new group,
// and therefore a matching arm in every consumer of this declaration, for each
// family migrated; grouping by rule shape means a migrated identity is one more
// name in an existing list.
macro_rules! cpu_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            elementwise = [
                Add, Sub, Mul, Div,
                Relu, Step, Mish, Elu, Gelu, Abs, Exp, Neg, Sqrt, Log,
                Tanh, Sigmoid, Swish, Sign, Floor, Ceil, Round, Log2, Log10,
                Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh,
                Atanh, Erf, Rsqrt, Trunc, Frac,
                AddScalar, MulScalar, Powf, Clamp,
                // `sub_scalar` and `div_scalar` sit here rather than with the
                // tensor operations because their descriptor contract requires
                // floating-point input metadata, which is exactly this group's
                // dtype set and not the wider one the tensor group declares.
                SubScalar, DivScalar,
                Atan2, Fmod, Remainder,
                // `dropout` walks its operand once and writes one result of the
                // same shape, which is this group exactly. That it consults a
                // random draw on the way changes nothing the row states.
                Dropout
            ],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            // Allocation. These take no operand, which is why they are the one
            // group whose capability row is queried against the descriptor's
            // inferred output rather than against an input: there is no input.
            // Both groups are `training = false` because a fresh allocation
            // records nothing on the tape; the `var_*` forms that do are not
            // here, because they return a variable rather than storage.
            filling = [
                TensorFromData, TensorFromBytes, Zeros, Ones, Full, Arange, Linspace,
                // The variable forms produce the same allocation and differ
                // only in what they hand back, which the row does not describe.
                VariableZeros, VariableOnes
            ],
            sampling = [
                UniformRandom, NormalRandom,
                VariableUniformRandom, VariableNormalRandom
            ],
            // Reading a value back to the host. One rule shape: any
            // non-quantized dtype, any layout the accessor handles, any rank,
            // and no gradient, because a host value is off the tape by
            // definition and nothing downstream of one can be differentiated.
            readback = [
                ToHostFloatScalar, ToHostFloatVec,
                ToHostIntScalar, ToHostIntVec,
                TensorToBytes
            ],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll, ProdAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim, ProdDim,
                // The exponential and the logarithm underneath `logsumexp` are
                // f32 arithmetic, which is this group's dtype anyway, and the
                // shift by the axis maximum keeps every intermediate inside the
                // range f32 can hold. Both spellings are here for the same
                // reason `sum_dim` and `sum_keepdim` both are: whether the
                // reduced axis survives is the caller's choice, not a property
                // the row can decide for them.
                LogSumExpDim, LogSumExpKeepDim
            ],
            spatial = [
                Conv2dExact, Conv1dExact, ConvTranspose2d,
                MaxPool2d, AvgPool2d, AdaptiveAvgPool2dExact
            ],
            matmul = [MatMulExact],
            // `layer_norm` and `batch_norm` join `softmax` here because they
            // share its rule shape exactly: f32-only, axis-bearing, gradient
            // recording. They are a different operation family and a different
            // trait method, which is precisely why the group is named for the
            // shape rather than for the family.
            // `rms_norm` scales by a root mean square without subtracting a
            // mean, which is what separates it from `layer_norm`, but the row
            // the two produce is identical.
            // `group_norm` joins them for the same reason `layer_norm` and
            // `batch_norm` did: it divides by a per-group standard deviation,
            // which is an f32 computation behind a descriptor that refuses a
            // non-float operand, and the row that states is this one. It sat
            // with the shape operations, which re-address bytes of any dtype
            // the backend can hold and share nothing with it but an accessor.
            // `log_softmax` is the same row as `softmax` because it is the same
            // computation stopped one step earlier: the CPU kernel `softmax`
            // calls already produces log-probabilities and then exponentiates
            // them. It is declared separately rather than left to callers to
            // compose because `log(softmax(x))` sends every entry far below its
            // row maximum through an exponential that underflows to zero, and
            // the logarithm of zero is not a number a router can act on.
            normalization = [Softmax, LogSoftmax, LayerNorm, BatchNorm, RmsNorm, GroupNorm],
            // `embedding`'s two operands have different dtypes by construction:
            // an integer index and an f32 weight table (`embedding_impl` always
            // reads and writes f32, so a wider float claim here would be the
            // same over-claim FND-005 fixed for `conv1d`/`conv_transpose2d`/
            // `adaptive_avg_pool2d`). One row cannot state "operand 0 is
            // integer, operand 1 is f32" - `dispatch::execute` applies the same
            // dtype set to every operand in turn - so `INDEX_AND_F32_DTYPES` is the
            // union of both, the loosest set the row can honestly claim, the
            // same trick `descriptor_min_rank` already uses for rank. The
            // descriptor's own per-operand contract refuses an integer weight
            // or a non-integer index before this row is ever consulted, and
            // `cpu::canonical`'s `f32_only` enforces the real, tighter weight
            // constraint the row cannot state.
            //
            // `grouped_matmul` joins for the same mixed-operand reason: an i64
            // offsets tile beside two f32 matrices, one row cannot state the
            // split, and the union is what `dispatch::execute` checks every
            // operand against. The descriptor refuses a non-integer offsets
            // operand and a non-float matrix before this row is consulted, and
            // the executor re-checks the f32 constraint the row cannot carry.
            embedding = [EmbeddingExact, GroupedMatMul],
            // `fused_attention` is a group of its own (issue #104): the
            // CPU kernel is native single-pass online softmax over f32/f64
            // with one tape entry, so neither the composed-reduction
            // F32_ONLY row nor the matmul FLOAT_DTYPES row states it
            // honestly. Other backends leave this empty until they ship a
            // kernel behind it.
            fused_attention = [FusedAttention],
            native_tensor = [
                // The order statistics sit here rather than in the f32-only
                // reduction group above because each builds its value buffer
                // from the operand's own, so a result comes back in the dtype
                // it was read in rather than relabelled. `topk` was in that
                // group, for a reason its kernel stopped giving when the fixed
                // f32 buffer became the operand's: the row stayed behind and
                // advertised one dtype for a kernel that handles eight, which
                // is why CUDA already declared four of them where CPU declared
                // one.
                ArgMax, ArgMin, Argsort, Sort, TopK, Cumsum,
                Maximum, Minimum, AbsDiff, Lerp, MaskedFill, WhereCond,
                CmpEq, CmpNe, CmpLt, CmpLe, CmpGt, CmpGe,
                TransposeExact, TransposeView, Narrow, Triu, Tril, Diag,
                ConcatExact, Gather, Scatter, IndexSelect, Repeat, RepeatInterleave, Pad, Unfold,
                // Same operands and the same row as `scatter` beside it, and
                // declared on this backend only. The rule it advertises is a
                // fixed summation order, which a CUDA kernel built on atomics
                // could not honour, so the accelerator groups are deliberately
                // left to claim it once they have a kernel that can.
                ScatterAdd,
                // One integer operand in, one boolean tensor out, and the depth
                // that separates them is an attribute rather than an operand.
                // The row states the union the rule shape allows, the way
                // `embedding`'s does: the descriptor refuses a non-integer
                // operand before this row is consulted, and the kernel reads
                // every integer width through the same accessor.
                OneHot,
                // One integer operand in, one i64 histogram out, and the bin
                // count is an attribute. The same union the `one_hot` note
                // above explains, for the same reason.
                Bincount,
                // The operand is read through the same stride-aware f64
                // accessor as `bincount`/`one_hot` (any non-quantized dtype
                // in, i64 coordinate rows out), and the result's extent is
                // the count of non-zero positions — a property of the values
                // no metadata inference can name (#102). Same rule shape as
                // the histogram beside it.
                NonZero,
                PixelShuffle,
                // `to_dtype` reads through the same stride-aware accessor and
                // writes a fresh contiguous buffer, which is this group's shape
                // exactly. Its target dtype is an attribute rather than an
                // operand, so the row constrains what it reads and the executor
                // constrains what it is asked to write.
                ToDType
            ],
            // Boolean on every operand and on the result. See the `logical`
            // arm of `descriptor_capability_rules!` for why this cannot be a
            // name in the group above.
            logical = [LogicalAnd, LogicalOr, LogicalNot],
            composed_tensor = [
                FlattenExact, SqueezeExact, UnsqueezeExact,
                StackExact, SliceExact, BroadcastLeft,
                // Both answer with a sequence of narrows along one axis. They
                // are the first rows whose executor returns more than one
                // storage, which the contract carries because `Execute` names
                // its output as an associated type.
                Chunk, Split
            ],
            // Issue #90: `BatchedMatMul` and `Addmm` rewrite into the widened
            // matmul kernel and inherit FLOAT_DTYPES through `$matmul`.
            // `ScaledDotProductAttention`/`Dot`/`Outer` moved to
            // `composed_reduction` so they keep the F32_ONLY row
            // `$reduction` states - see that group's note. Leaving them here
            // would have advertised half-precision attention (whose second
            // matmul meets f32 `softmax` scores against the value operand and
            // fails the same-dtype guard) and a `Dot` whose `sum_all` always
            // returns f32 storage under an f16 label.
            composed_matmul = [BatchedMatMul, Addmm],
            // `linear` rewrites into a transpose and a matmul, so it inherits
            // the matmul constraint. It is a group of its own rather than a
            // name in the one above because the operations there carry no bias,
            // and the rank bound has to admit the rank-one one this has.
            composed_matmul_bias = [Linear],
            // Two groups rather than one, because the compression and the
            // operations over compressed storage read opposite dtype sets and a
            // row states one. `quantize` reads f32 and writes blocks;
            // `dequantize` and `quantized_matmul` read blocks. Both refuse a
            // strided operand: the kernels index the block buffer directly and
            // never consult a stride.
            //
            // The second element of each pair is this backend's training
            // claim, and it is stated per operation because the three
            // disagree. CPU's `quantize`/`dequantize` kernels record the
            // issue #93 straight-through tape entry whenever `GradMode`
            // enables recording (`cpu/canonical/linalg.rs`), so their rows
            // advertise training. `quantized_matmul` records nothing and has
            // `GradientRule::None` in the catalog, so its row stays `false` -
            // fail-closed, until a kernel behind it records a gradient.
            quantizing = [(Quantize, true)],
            quantized = [(Dequantize, true), (QuantizedMatMul, false)],
            // The losses supplied as real composed defaults
            // rather than as stubs: each rewrites into `sub`, `mul`, `abs` and
            // an all-reduce. They inherit the reduction group's f32-only claim
            // because their `Mean` and `Sum` forms end in `mean_all`/`sum_all`,
            // and the reduction mode is an attribute rather than part of the
            // identity, so the row has to hold for the narrowest of the three.
            composed_reduction = [
                MseLoss, L1Loss, BceWithLogitsLoss,
                // `instance_norm` is here rather than with the shape
                // operations it used to sit beside. It shares nothing with
                // them: they re-address bytes of any dtype the backend can
                // hold, while this subtracts a per-channel mean and divides by
                // a per-channel standard deviation, which is an f32
                // computation on this backend and a descriptor that refuses a
                // non-float operand outright. The row it needs is this group's,
                // named as ever for its shape rather than for its family.
                InstanceNorm,
                // Variance, standard deviation and the p-norm have no kernel of
                // their own on any backend: each is a subtract, a square, a
                // reduce and a scale over primitives already migrated above.
                // Same rule shape as the losses, for the same reason: they end
                // in an all-reduce or an axis reduce.
                VarianceAll, VarianceDim, VarianceKeepDim,
                StdAll, StdDim, StdKeepDim,
                Norm,
                // Issue #90: these three share this group's rule shape exactly
                // (F32_ONLY via `$reduction`, CPU_LAYOUTS, Composed, training)
                // and only sat in `composed_matmul` while that group also
                // carried F32_ONLY. Widening `$matmul` to FLOAT_DTYPES would
                // have over-advertised them: `ScaledDotProductAttention`
                // rewrites through f32-only `softmax`, so its second matmul
                // meets f32 scores against the value operand and fails the
                // same-dtype guard; `Dot`'s `sum_all` always returns f32
                // storage, so an f16 result would be a mislabel; `Outer` stays
                // on the same narrow set CUDA advertises for it.
                ScaledDotProductAttention,
                Dot, Outer
            ],
            // The composed reductions whose operands split into a float and an
            // integer index, which is the one thing keeping them out of the
            // group above: `cross_entropy_loss` takes f32 logits and integer
            // class targets, so its row carries `INDEX_AND_F32_DTYPES` - the
            // union of the two - for exactly the reason `embedding`'s does.
            // The descriptor's per-operand contract (`operand_ranks` gives
            // logits rank 2 and targets rank 1, and `index_input` names
            // operand 1 as the integer one) refuses a swapped or mistyped pair
            // before this row is consulted, and `cpu::canonical`'s `f32_only`
            // enforces the logits' real f32-only constraint the row cannot
            // state. Composed rather than native because the kernel rewrites
            // into `log_softmax`, `mul`, `sum_dim`, `neg` and an all-reduce.
            composed_reduction_indexed = [CrossEntropyLoss]
        }
    };
}

// Re-exported crate-internally so the CPU executor module can prove, at
// compile time, that it implements every identity this declaration advertises.
// Gated on the consumer's own feature: the table below is always compiled (a
// capability claim is data, and the registry reports every backend's), but the
// module that checks this one is not.
pub(crate) use cpu_descriptor_operations;

macro_rules! cuda_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            // `Maximum`/`Minimum`/`AbsDiff` sit on the same
            // `cuda_pointwise!` binary arms as `Add`/`Sub`
            // (`backend/elementwise.rs`'s `cuda_maximum_storage`/
            // `cuda_minimum_storage`/`cuda_abs_diff_storage`), and `Lerp`
            // composes sub + mul_scalar + add through `cuda_lerp_storage` -
            // one traversal, one dtype-parametric kernel family, the rule
            // shape this group already encodes (`FLOAT_DTYPES` +
            // `CUDA_LAYOUTS`, strided elementwise kernel included). They
            // left `native_tensor` because the `F32_ONLY` there overstated
            // nothing the kernels could not honour: #86 closed elementwise
            // widening, and these four were simply filed under the wrong
            // group. CPU's wider `NON_QUANTIZED` rides its own accessor;
            // CUDA's pointwise kernels have no `i64`/`bool` mode, so the
            // claim stops at `FLOAT_DTYPES` exactly like `Add`'s.
            elementwise = [
                Add, Sub, Mul, Div,
                Relu, Step, Mish, Elu, Gelu, Abs, Exp, Neg, Sqrt, Log,
                Tanh, Sigmoid, Swish, Sign, Floor, Ceil, Round, Log2, Log10,
                Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh,
                Atanh, Erf, Rsqrt, Trunc, Frac,
                AddScalar, MulScalar, Powf, Clamp,
                SubScalar, DivScalar,
                Atan2, Fmod, Remainder,
                Dropout,
                Maximum, Minimum, AbsDiff, Lerp
            ],
            // `ToDType` rides this group since #106(a): the cast kernel is
            // dtype-parametric over the source, and `broadcast`'s
            // `CUDA_BOOL_SAFE_STORAGE_DTYPES` is exactly that source set
            // (f32/f64/f16/bf16/i64/bool) on contiguous layouts with both
            // training arms. Capability admits the *input* dtype; the
            // executor refuses targets outside {f32,f64,f16,bf16,i64}.
            broadcast = [BroadcastAs, ToDType],
            reshape = [ReshapeExact],
            filling = [
                TensorFromData, TensorFromBytes, Zeros, Ones, Full, Arange, Linspace,
                VariableZeros, VariableOnes
            ],
            sampling = [
                UniformRandom, NormalRandom,
                VariableUniformRandom, VariableNormalRandom
            ],
            // The four `ToHost*` identities stay `F32_ONLY` through this
            // group row: their executors funnel into `float_to_vec1`/
            // `int_to_vec1`, which gate on `cuda_require_f32`
            // (`backend/contract.rs`) before any bytes leave the device.
            // `TensorToBytes` widens instead via its standalone
            // `CUDA_BOOL_SAFE_STORAGE_DTYPES` row in `tables.rs` - it
            // downloads raw bytes through `HostInterop::to_bytes`, which
            // only length-checks against `checked_storage_byte_len` and
            // never reinterprets the elements, so every dtype CUDA can
            // hold round-trips. Legacy rows render first and `support()` is
            // any-row-matches, so the wide claim is the one a real
            // `tensor_to_bytes` call resolves against while `ToHostFloatVec`
            // still refuses an `f16` operand by dtype.
            readback = [
                ToHostFloatScalar, ToHostFloatVec,
                ToHostIntScalar, ToHostIntVec,
                TensorToBytes
            ],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll, ProdAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim, ProdDim,
                LogSumExpDim, LogSumExpKeepDim,
                // Issue #90: `matmul.cu` now exports one GEMM entry per
                // float storage dtype, so `MatMulExact` sits in this
                // group's FLOAT_DTYPES row rather than the `matmul`
                // group's f32-only one. The rule shapes are identical
                // apart from that dtype set - Native, Contiguous,
                // training, the same rank bounds - and a group is a rule
                // shape, so this is where it honestly sits now. The
                // `matmul` group's F32_ONLY row (`tables.rs`) is
                // intentional: it feeds the composed `SDPA`/`Dot`/`Outer`
                // rows, which must stay narrow.
                MatMulExact
            ],
            spatial = [
                Conv2dExact, Conv1dExact, ConvTranspose2d,
                MaxPool2d, AvgPool2d, AdaptiveAvgPool2dExact
            ],
            // Empty since #90: `MatMulExact` moved to `reduction`, whose
            // rule shape it matches exactly once `matmul` widened to
            // FLOAT_DTYPES. The identities that remain f32-only
            // (`ScaledDotProductAttention`/`Dot`/`Outer`) never sat here;
            // they are in `composed_matmul` below, on purpose.
            matmul = [],
            normalization = [Softmax, LogSoftmax, LayerNorm, BatchNorm, RmsNorm, GroupNorm],
            // `OneHot`/`Bincount`/`ScatterAdd`/`GroupedMatMul` ride this
            // group because their index/offsets operand is an integer dtype
            // their value operand is not: the union of integer index dtypes
            // and f32 weights is exactly the admission `embedding`'s comment
            // already documents. The first three take no view-incompatible
            // path their `elementwise_layouts` would mis-describe.
            // `GroupedMatMul`'s CUDA narrow/matmul path is flat-buffer only,
            // so its executor re-checks contiguity fail-closed — the layout
            // twin of the dtype split this row also cannot state (issue #103).
            // `ScatterAdd`'s index stays off the tape the way
            // `EmbeddingExact`'s does; its f64-accumulated value operands
            // and dropped out-of-range writes live in the executor, as do
            // `GroupedMatMul`'s i64 offsets tile.
            embedding = [EmbeddingExact, OneHot, Bincount, ScatterAdd, GroupedMatMul],
            // No fused-attention kernel on CUDA yet (issue #104); empty
            // until one ships.
            fused_attention = [],
            // Issue #87: `TopK` leaves `reduction` and the six Welford
            // `var`/`std` rows leave `composed_reduction` because both now
            // sit on f32-only kernels (`incin_cuda_topk`,
            // `incin_cuda_welford` take `const float* input`). A group is a
            // rule shape, so they belong with the other f32-only native
            // tensor rows here - dtype set is the only difference from the
            // groups they left, and that is exactly the shape this group
            // already encodes. `descriptor_training` still answers true for
            // the `var`/`std` rows (they record a tape entry from the
            // backend method) and false for `TopK`/`Argsort`, same as before.
            //
            // The `F32_ONLY` this group's `tensor_dtypes` states is the
            // honest floor for every member that has no wider standalone row
            // in `tables.rs`, and the kernels name the reason one by one:
            // - the order statistics and Welford rows - `argmax`/`argmin`,
            //   `cumsum`, `sort`/`argsort` (both rewrite through `topk`),
            //   `topk`, `var`/`std` - sit on `incin_cuda_argmax_argmin`,
            //   `incin_cuda_cumsum`, `incin_cuda_topk` and
            //   `incin_cuda_welford`, each of which takes `const float*`
            //   input (`cuda/ops/reduce.rs`);
            // - `triu`/`tril`/`diag`/`pad`/`repeat` allocate their output
            //   buffer as `DTypeId::F32` no matter what the operand carries
            //   (`cuda/ops/shape.rs`'s `launch_triangular`, `launch_diag`,
            //   `launch_pad`, `launch_repeat`);
            // - `repeat_interleave` refuses any non-`f32` operand outright
            //   (`launch_repeat_interleave`'s `UnsupportedDType`);
            // - the comparisons gate on `cuda_require_f32`
            //   (`cuda/ops/compare.rs`).
            // Members whose claim is wider than `F32_ONLY` say so with a
            // standalone row in `tables.rs` that renders before this group
            // row: the measured movement identities
            // (`transpose*`/`narrow`/`concat`), the indexed ones
            // (`gather`/`scatter`/`index_select`), the mask consumers
            // (`where_cond`/`masked_fill` at `F32_AND_BOOL`), and -
            // like the composed identities below - `pixel_shuffle`/`unfold`,
            // whose executors are pure reshape/transpose/narrow chains over
            // those same byte-movement kernels.
            native_tensor = [
                ArgMax, ArgMin, Argsort, Cumsum, Sort,
                TopK, VarianceAll, VarianceDim, VarianceKeepDim,
                StdAll, StdDim, StdKeepDim,
                MaskedFill, WhereCond,
                CmpEq, CmpNe, CmpLt, CmpLe, CmpGt, CmpGe,
                TransposeExact, TransposeView, Narrow, Triu, Tril, Diag,
                ConcatExact, Gather, Scatter, IndexSelect, Repeat, RepeatInterleave,
                Pad, Unfold,
                PixelShuffle
            ],
            logical = [LogicalAnd, LogicalOr, LogicalNot],
            // Same shape, reported as composed: these answer by rewriting
            // into another operation rather than by running a kernel of
            // their own. The group row stays `F32_ONLY` through the shared
            // `tensor_dtypes`; each identity's honest claim is the wider
            // standalone `composed_ranked` row in `tables.rs` - the rewrite
            // targets are the measured wide byte-movement kernels
            // (`tests/cuda_shape_dtypes.rs`'s byte-exact matrix across
            // transpose, broadcast, narrow and concat), so
            // `CUDA_BOOL_SAFE_STORAGE_DTYPES` there is a verified claim
            // rather than an extrapolation from the rewrite shape alone.
            composed_tensor = [
                FlattenExact, SqueezeExact, UnsqueezeExact,
                StackExact, SliceExact, BroadcastLeft,
                Chunk, Split
            ],
            composed_matmul = [
                // Nothing moved out with `MatMulExact` in #90: these three
                // still inherit the unchanged f32-only `$matmul` argument
                // (SDPA sits on f32-only `softmax`; `Dot`/`Outer` compose
                // through mul + all-reduce under this group's dtype row).
                ScaledDotProductAttention,
                Dot, Outer
            ],
            // Empty since #90: `Linear` moved to `composed_reduction`,
            // whose rule shape it matches exactly once the product widened
            // to FLOAT_DTYPES - same Composed kind, Contiguous layouts,
            // rank bounds and training; only the dtype set differed.
            composed_matmul_bias = [],
            // All three `training` flags are `false` here, unlike CPU's: the
            // CUDA executor's quantize/dequantize paths
            // (`cuda/executor.rs` around the `Execute<op::Quantize>` impls,
            // down to `cuda/ops/quant.rs`) launch a kernel and return without
            // pushing a `cuda::tape` entry, so no training invocation could
            // ever be answered with a gradient. Fail-closed: the row claims
            // training only when the recording implementation exists.
            quantizing = [(Quantize, false)],
            quantized = [(Dequantize, false), (QuantizedMatMul, false)],
            composed_reduction = [
                MseLoss, L1Loss, BceWithLogitsLoss,
                InstanceNorm,
                // Issue #87: the six `Variance*`/`Std*` rows moved to
                // `native_tensor` - the Welford kernel is f32-only, so the
                // FLOAT_DTYPES inheritance here overstated the claim.
                Norm,
                // Issue #90: these three matched this group's rule shape
                // exactly and only sat in the matmul groups while those
                // groups carried the same dtype set. They now inherit
                // FLOAT_DTYPES honestly: `bmm`/`addmm`/`linear` all
                // rewrite into the widened `MatMulExact` kernel.
                BatchedMatMul, Addmm, Linear
            ],
            composed_reduction_indexed = [CrossEntropyLoss]
        }
    };
}

macro_rules! wgpu_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            // The unary activations here are not new kernels. `wgpu/executor.rs`
            // has implemented `Execute` for every one of them, against the op
            // modes in `shaders/unary.wgsl`, since the executor was written -
            // they were simply never listed here, so the capability query
            // answered `Unsupported` and no caller could reach them. A shader
            // with no capability row is dead code that reads as coverage, which
            // is what `assert_wgpu_unary_operations_are_advertised` now prevents
            // from recurring.
            elementwise = [
                Add, Sub, Mul, Div,
                Relu, Step, Mish, Elu, Gelu, Abs, Exp, Neg, Sqrt, Log,
                Tanh, Sigmoid, Swish,
                // The twenty-one unary floats and the scalar/binary floats
                // below each have a WGSL mode, an `Execute` impl and a tape
                // recipe in `wgpu/backend/elementwise.rs`; they sat
                // unadvertised until the gap in #91 was closed, exactly as
                // the activations above once did.
                Sign, Floor, Ceil, Round, Log2, Log10,
                Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh,
                Atanh, Erf, Rsqrt, Trunc, Frac,
                AddScalar, MulScalar, SubScalar, DivScalar, Powf, Clamp,
                Atan2, Fmod, Remainder,
                // Batch-C host composite: keep-mask (LCG) + scale; identity
                // when eval or p<=0, zero when p>=1 — CPU's recipe.
                Dropout
            ],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            // `impl_creation_executors!` gives WGPU real `UniformRandom`/
            // `NormalRandom` executors too, and `impl_data_creation_executors!`
            // gives it real `TensorFromData`/`TensorFromBytes` ones; none of
            // the four were ever listed here, same as the unary activations
            // above.
            // The nine rows below are wrappers over paths each backend
            // already has, which is why they arrive without new kernel source.
            // The `var_*` forms are their plain sibling's allocation plus
            // `VariableBackend::var_from_tensor`, so they belong in exactly the
            // groups their siblings do and inherit `F32_ONLY`/`CONTIGUOUS`
            // honestly. The readback rows take `tensor_dtypes`/`tensor_layouts`,
            // which this backend declares as `F32_ONLY`/`CONTIGUOUS`: both are
            // real constraints here, not conservatism. `float_to_vec1` requires
            // f32, and every accelerator readback downloads the whole
            // allocation without walking strides, so `readback_operand` refuses
            // a strided operand rather than returning the wrong window.
            filling = [
                TensorFromData, TensorFromBytes, Zeros, Ones, Full, Arange, Linspace,
                VariableZeros, VariableOnes
            ],
            sampling = [
                UniformRandom, NormalRandom,
                VariableUniformRandom, VariableNormalRandom
            ],
            readback = [
                ToHostFloatScalar, ToHostFloatVec,
                ToHostIntScalar, ToHostIntVec,
                TensorToBytes
            ],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll, ProdAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim, ProdDim,
                // Batch-C: host stable recipes over already-taped primitives
                // (max → sub → exp → sum/keepdim → log → add/squeeze).
                LogSumExpDim, LogSumExpKeepDim
            ],
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            // Issue #90 audit: this group and the two composed matmul groups
            // below keep the table's `F32_ONLY` `$matmul` while CPU/CUDA
            // widened to `FLOAT_DTYPES`. `matmul.wgsl` is `array<f32>`
            // throughout and `device.rs` requests `Features::empty()` (no
            // `SHADER_F16` query), so a wider claim here would advertise
            // half/double storage no kernel reads; see the `matmul`
            // parameter's note in `tables.rs`.
            matmul = [MatMulExact],
            // The whole normalization family WGPU can answer by rewriting into
            // taped primitives: `softmax` (already had an Execute impl via the
            // axis macro), `rms_norm` (mul/mean/sqrt/div), `layer_norm` (add
            // the mean-center step and an optional affine bias), `group_norm`
            // (reshape to runs, same statistical path, reshape back),
            // `batch_norm` (inference rides the running-statistics path;
            // training rides `sum_keepdim` over every non-channel axis so
            // the batch-statistics path reaches the gradient, CPU's
            // `batch_norm_training_impl` composition) and `instance_norm`
            // (group_norm with one group per channel). `LogSoftmax` joins
            // in batch C: the stable max/sub/exp/sum/log chain over the
            // same axis-macro request shape Softmax already rides.
            normalization = [Softmax, LogSoftmax, LayerNorm, BatchNorm, RmsNorm, GroupNorm],
            // `embedding` runs through a host-walk forward and a scatter-add
            // backward in `wgpu/backend/indexing.rs`, matching
            // `cpu::ops::embedding`'s recipe exactly (ONE TapeEntry,
            // `input_ids = vec![w.id]`, accumulate-not-overwrite for repeated
            // indices). The row states `INDEX_AND_F32_DTYPES` via the shared
            // `embedding_dtypes` parameter — the union of the integer index
            // operand and the f32 weight table — because
            // `dispatch::execute` applies one dtype set to every operand;
            // the descriptor's own per-operand contract refuses a non-integer
            // index or a non-f32 weight before this row is consulted.
            embedding = [EmbeddingExact],
            // No fused-attention kernel on WGPU yet (issue #104); empty
            // until one ships.
            fused_attention = [],
            // Advertised now that each has an executor and a gradient path.
            // The six comparisons join this group (#91): `binary.wgsl` has
            // carried their modes since the shader was written, and they sit
            // here rather than in `logical` because the row must state the
            // *operand* dtype set this group's `tensor_dtypes` already holds
            // (`F32_ONLY`, re-checked by name in `wgpu/backend/compare.rs`).
            // The result is `bool` — physical 0.0/1.0 under a `Bool` storage
            // label, the representation `wgpu/storage.rs::
            // physical_element_bytes` documents and `masked_fill`/
            // `where_cond` already consume as a mask; the row cannot state
            // an output dtype (admission checks operands only, see
            // `incin-core`'s `admit_invocation`), and the catalog's
            // `trace_output_dtype` is what types the output `Bool`.
            // `descriptor_training` resolves these rows `false` — a
            // comparison has nowhere to send a gradient — and
            // `wgpu/backend/compare.rs` pushes no tape entry, matching CPU
            // and CUDA walk for walk. Rank caps at 6 through
            // `accelerator_max_rank` because a stretched operand rides
            // `shape.wgsl`'s packed block, the same bound CUDA's
            // comparison broadcasts claim.
            // `transpose` has its own WGSL kernel and its own tape entry (a
            // transpose is its own inverse), so it is native rather than
            // composed. It sat unregistered until now: the kernel existed,
            // nothing advertised it, and dispatch refused it.
            // `TransposeView` is deliberately absent. It returns a
            // non-contiguous view, and this backend's pointwise shaders
            // (`unary.wgsl`, `binary.wgsl`) address linearly -- neither
            // mentions a stride -- so handing them a view would read the wrong
            // elements silently rather than fail. It can be advertised once
            // those shaders take strides, which is the same work the CUDA
            // strided kernels already do.
            native_tensor = [
                Maximum, Minimum, AbsDiff, TransposeExact,
                // Structural windows and joins, each with a `shape.wgsl`
                // slice/paste path and a tape entry: `narrow`/`slice` are one
                // mode-0 launch forward and one mode-1 paste backward;
                // `concat`/`stack` are one zeroed buffer plus one paste per
                // operand; `tril`/`triu` mask rank 1–2 storage host-side the
                // way CPU's `triangular_storage` does.
                Narrow, SliceExact, ConcatExact, StackExact, Tril, Triu,
                // Batch-C host walks: pad fills the outside window and extracts
                // it on the backward; repeat tiles by modulo; repeat_interleave
                // copies contiguous blocks; cumsum is a host scan whose reverse
                // suffix-sum is its gradient. Each carries its own tape entry.
                Pad, Repeat, RepeatInterleave, Cumsum,
                // The six numeric comparisons (#91), operands `F32_ONLY`
                // via `$tensor_dtypes`, result `Bool` — see the group's
                // note above for why the row states only the operands.
                CmpEq, CmpNe, CmpLt, CmpLe, CmpGt, CmpGe,
            ],
            // Boolean on every operand and on the result (#91):
            // `logical_and`/`logical_or` ride `binary.wgsl` modes 13/14 and
            // `logical_not` rides `unary.wgsl` mode 34, all over the
            // physical f32 0/1 encoding `wgpu/storage.rs` documents. The
            // descriptor already refuses a non-bool operand
            // (`catalog/inference.rs`'s logical contract), `BOOL_ONLY`
            // refuses it at admission, and `wgpu/backend/compare.rs`
            // re-checks by name. `descriptor_training` resolves these rows
            // `false` and no tape entry is pushed — same as CPU and CUDA.
            logical = [LogicalAnd, LogicalOr, LogicalNot],
            // All three rewrite into `reshape` rather than running a kernel of
            // their own: the elements are already in the right order and only
            // the shape changes. They push no tape entry of their own, so the
            // backward is `reshape`'s, which is what makes their `training`
            // claim true without new hand-derived math.
            // `broadcast_left` prepends the target prefix and reuses
            // `broadcast_storage`; `chunk`/`split` are sequences of `narrow`
            // (the first multi-output WGPU rows, matching CPU's Vec output).
            composed_tensor = [
                FlattenExact, SqueezeExact, UnsqueezeExact,
                BroadcastLeft, Chunk, Split
            ],
            // `bmm` is matmul under its own name; `addmm` is matmul plus two
            // scalar scales and an add; `dot` is mul + all-sum; each is CPU's
            // composition on taped primitives. `Linear` rides its own bias
            // group because the rank-one input/bias path needs the wider rank
            // bound that group carries. Batch C adds `outer` (two unsqueezes
            // and a broadcast multiply) and `scaled_dot_product_attention`
            // (transpose-k, matmul, scale, optional additive mask, softmax on
            // the last axis, matmul with v) — both pure taped compositions.
            // Issue #90: unlike CPU/CUDA, none of these moved to
            // `composed_reduction` - this backend's `$matmul` never widened
            // past `F32_ONLY`, so the group already states the honest row
            // for all five (and for `Linear` in its bias group beside it).
            composed_matmul = [
                BatchedMatMul, Addmm, Dot,
                Outer, ScaledDotProductAttention
            ],
            composed_matmul_bias = [Linear],
            quantizing = [],
            quantized = [],
            // Losses and moments composed from sub/mul/abs and an all- or
            // axis-reduce, exactly CPU's recipes, so each inherits this
            // backend's f32-only contiguous reduction claim honestly.
            // Batch C adds `bce_with_logits_loss` (max(x,0) - x*z + softplus
            // with the custom 0.5 slope at the kink) and `instance_norm`
            // (group_norm with one group per channel).
            // `CrossEntropyLoss` rides `composed_reduction_indexed`: its
            // integer class-target operand means the row states the union
            // `INDEX_AND_F32_DTYPES` (f32 logits + integer targets), the
            // same widened claim `embedding` carries. The path is CPU's
            // composition — `log_softmax`, a tape-tracked `gather` of the
            // target class, negate, reduce — and the gather's scatter-based
            // backward is what carries the gradient into the logits.
            composed_reduction = [
                MseLoss, L1Loss, BceWithLogitsLoss,
                VarianceAll, VarianceDim, VarianceKeepDim,
                StdAll, StdDim, StdKeepDim,
                Norm,
                InstanceNorm,
            ],
            composed_reduction_indexed = [CrossEntropyLoss]
        }
    };
}

macro_rules! metal_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            // Batch-A pointwise on top of the four arithmetic binaries:
            // every unary float with a host-side `Self::$method` and tape
            // recipe in `metal/pointwise.rs`, the four scalar forms plus
            // `powf`, `clamp`, and the three binary floats (`atan2`/
            // `fmod`/`remainder`). Each has an `Execute` impl in
            // `metal/executor.rs` — the compile-time assert at the bottom
            // of that file is what makes advertising them safe.
            elementwise = [
                Add, Sub, Mul, Div,
                Relu, Step, Mish, Elu, Gelu, Abs, Exp, Neg, Sqrt, Log,
                Tanh, Sigmoid, Swish, Sign, Floor, Ceil, Round, Log2, Log10,
                Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh,
                Atanh, Erf, Rsqrt, Trunc, Frac,
                AddScalar, SubScalar, MulScalar, DivScalar, Powf, Clamp,
                Atan2, Fmod, Remainder,
                // `dropout` walks its operand once and writes one result of
                // the same shape, which is this group exactly — CUDA's and
                // WGPU's comment, with Metal's counter-based keep-mask
                // (`metal/pointwise.rs`) behind it. That it consults a
                // random draw on the way changes nothing the row states.
                Dropout,
            ],
            // `ToDType` rides this group since #92: the cast walks the shared
            // host bytes dtype-parametrically over the source (`metal/convert.rs`),
            // and `broadcast`'s F32_ONLY rows admit the source set the
            // capability actually checks (inputs only — the target tag is an
            // attribute, never inspected by `admit_invocation`). Both training
            // arms of the group (`broadcast` and `broadcast_training`) cover
            // the float-to-float tape entry and the integer-target no-tape path.
            broadcast = [BroadcastAs, ToDType],
            reshape = [ReshapeExact],
            // `impl_creation_executors!` gives Metal real `UniformRandom`/
            // `NormalRandom` executors too, and `impl_data_creation_executors!`
            // gives it real `TensorFromData`/`TensorFromBytes` ones; none of
            // the four were ever listed here.
            // The nine rows below are wrappers over paths each backend
            // already has, which is why they arrive without new kernel source.
            // The `var_*` forms are their plain sibling's allocation plus
            // `VariableBackend::var_from_tensor`, so they belong in exactly the
            // groups their siblings do and inherit `F32_ONLY`/`CONTIGUOUS`
            // honestly. The readback rows take `tensor_dtypes`/`tensor_layouts`,
            // which this backend declares as `F32_ONLY`/`CONTIGUOUS`: both are
            // real constraints here, not conservatism. `float_to_vec1` requires
            // f32, and every accelerator readback downloads the whole
            // allocation without walking strides, so `readback_operand` refuses
            // a strided operand rather than returning the wrong window.
            filling = [
                TensorFromData, TensorFromBytes, Zeros, Ones, Full, Arange, Linspace,
                VariableZeros, VariableOnes
            ],
            sampling = [
                UniformRandom, NormalRandom,
                VariableUniformRandom, VariableNormalRandom
            ],
            readback = [
                ToHostFloatScalar, ToHostFloatVec,
                ToHostIntScalar, ToHostIntVec,
                TensorToBytes
            ],
            reduction = [
                SumAll, MeanAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim
            ],
            // Empty on purpose: `MetalBackendImpl::conv2d`/`max_pool2d`/
            // `avg_pool2d` always return `Err(unsupported(..))`, and the
            // legacy table carries no coarse `Conv2d`/`Pool2d` rows either.
            // Listing them here would advertise operations that can never
            // execute; the `Execute` impls in `metal/executor.rs` stay as
            // loud errors. Do NOT refill this without real Metal kernels.
            spatial = [],
            matmul = [MatMulExact],
            // #92 attention-block closure: the normalization ops Metal can
            // rewrite into taped primitives — `softmax`/`log_softmax` (the
            // stable max/sub/exp/sum/log chain, `max_keepdim` included),
            // `rms_norm` (mul/mean/sqrt/div over the last axis) and
            // `layer_norm` (add the mean-center step and an optional affine
            // bias). Each has an `Execute` impl in `metal/executor.rs`; the
            // compile-time assert at the bottom of that file is what makes
            // advertising them safe. BatchNorm rides the same group: batch
            // statistics are `mean_keepdim` walks plus a scale/shift, and
            // GroupNorm is a per-group reshape into the layer-norm recipe —
            // both composed from taped primitives in `metal/normalization.rs`.
            normalization = [Softmax, LogSoftmax, LayerNorm, RmsNorm, BatchNorm, GroupNorm],
            // Index-gather family: each is a host-side row/column walk with
            // a scatter-based tape entry in `metal/indexing.rs`. The row
            // claims `INDEX_AND_F32` because the index operands are i64
            // (Metal refuses u8/u32 storage, so i64 is the only admitted
            // index dtype here) while the value operands stay f32 —
            // `tensor_dtypes = F32_ONLY` would refuse the index operand.
            // `Scatter` joins with its ternary (input, index, source) walk
            // and `OneHot` with its bool-mask host walk (#92); `OneHot`
            // records no tape entry even though this group hardcodes
            // `training = true` (the same claim CUDA's embedding group
            // makes for it).
            embedding = [EmbeddingExact, Gather, IndexSelect, Scatter, OneHot],
            // No fused-attention kernel on Metal yet (issue #104); empty
            // until one ships.
            fused_attention = [],
            // The layout half of #92, each with a host-side walk and a tape
            // entry in `metal/layout.rs`: `transpose` materializes the swap
            // (a transpose is its own inverse, so backward reapplies it),
            // `narrow` copies one window and scatters its cotangent back,
            // `concat` assembles one output per operand offset and splits
            // the cotangent with `narrow`, and `tril`/`triu` mask rank 1–2
            // storage host-side the way CPU's `triangular_storage` does
            // (zeroing is its own transpose, so backward reapplies the mask).
            // `cumsum` is a host-side prefix scan with a suffix-sum backward
            // in `metal/reduction.rs`; it rides this group because it maps
            // an axis without collapsing it, same request shape as Softmax.
            // #92 adds `Pad`/`Repeat` (taped host walks in `metal/layout.rs`)
            // and the three forward-only index reductions `ArgMax`/`Sort`/
            // `TopK` (i64 results, no tape — `descriptor_training` resolves
            // those to `false`), all on the same F32_ONLY contiguous claim.
            native_tensor = [
                TransposeExact, Narrow, ConcatExact, Tril, Triu, Cumsum,
                Pad, Repeat, ArgMax, Sort, TopK,
            ],
            logical = [],
            // The rewrites: `slice` is one `narrow` per axis, `stack` is
            // `unsqueeze` per operand then `concat`, and the two axis views
            // are `reshape`s — each inherits the tape entry of the primitive
            // it rewrites into rather than pushing math of its own, which is
            // what makes `training = true` on these rows true.
            //
            // The matmul compositions follow: `scaled_dot_product_attention`
            // (transpose-k, matmul, scale, optional additive mask, softmax on
            // the last axis, matmul with v) rides `composed_matmul` — same
            // f32-only contiguous constraint as the product it wraps — and
            // `linear` (promote, weight transpose, matmul, optional bias
            // add, demote) rides `composed_matmul_bias`, whose rank bound
            // admits the rank-one bias beside it. Each has an `Execute` impl
            // in `metal/executor.rs`; the compile-time assert at the bottom
            // of that file is what makes advertising them safe.
            //
            // The loss and moment compositions follow: the three pairwise
            // losses (`mse`/`l1`/`bce_with_logits`) are taped arithmetic plus
            // a reduction mode, the variance/std family is mean-center →
            // square → sum → scale (optional sqrt), and `norm` is the
            // order-aware abs/pow/sum/root chain — all composed from taped
            // primitives in `metal/loss.rs`/`metal/reduction.rs`. The
            // indexed loss (`cross_entropy`) rides
            // `composed_reduction_indexed` because its target-class gather
            // is what carries the gradient into the logits.
            //
            // Everything still empty stays empty on purpose: an empty group
            // is a truthful claim, a copied one would not be. `logical`
            // needs the boolean-result representation settled first (no
            // comparison executor exists here), the quantization groups need
            // kernels that do not exist, and `composed_matmul`'s remaining
            // members (`addmm`/`dot`/`outer`) are compositions this
            // backend has not written yet (`bmm`/`BatchedMatMul` joins
            // with `ScaledDotProductAttention` in #92, rewriting into the
            // already-taped `Self::matmul`).
            composed_tensor = [SliceExact, StackExact, SqueezeExact, UnsqueezeExact],
            composed_matmul = [ScaledDotProductAttention, BatchedMatMul],
            composed_matmul_bias = [Linear],
            quantizing = [],
            quantized = [],
            composed_reduction = [
                MseLoss, L1Loss, BceWithLogitsLoss,
                VarianceAll, VarianceDim, VarianceKeepDim,
                StdAll, StdDim, StdKeepDim,
                Norm,
                InstanceNorm,
            ],
            composed_reduction_indexed = [CrossEntropyLoss]
        }
    };
}

pub(crate) use cuda_descriptor_operations;
pub(crate) use metal_descriptor_operations;
pub(crate) use wgpu_descriptor_operations;

macro_rules! rocm_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            // Issue #6 scaffolding: no HIP kernels exist — every group
            // empty on purpose, so the table claims nothing and the
            // registry answers every query with a typed Unsupported.
            elementwise = [],
            broadcast = [],
            reshape = [],
            filling = [],
            sampling = [],
            readback = [],
            reduction = [],
            spatial = [],
            matmul = [],
            normalization = [],
            embedding = [],
            fused_attention = [],
            native_tensor = [],
            logical = [],
            composed_tensor = [],
            composed_matmul = [],
            composed_matmul_bias = [],
            quantizing = [],
            quantized = [],
            composed_reduction = [],
            composed_reduction_indexed = []
        }
    };
}

pub(crate) use rocm_descriptor_operations;
