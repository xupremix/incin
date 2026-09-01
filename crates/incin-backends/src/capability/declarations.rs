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
                LogSumExpDim, LogSumExpKeepDim,
                // `topk` is here rather than with the other index reductions
                // because its value buffer is built as f32 whatever the operand
                // held. f32 is the only operand dtype whose result it labels
                // correctly, and this group is the f32-only one.
                TopK
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
            embedding = [EmbeddingExact],
            native_tensor = [
                ArgMax, ArgMin, Argsort, Cumsum,
                Maximum, Minimum, AbsDiff, Lerp, MaskedFill, WhereCond,
                CmpEq, CmpNe, CmpLt, CmpLe, CmpGt, CmpGe,
                TransposeExact, Narrow, Triu, Tril, Diag,
                ConcatExact, Gather, Scatter, IndexSelect, Repeat, Pad, Unfold,
                // Same operands and the same row as `scatter` beside it, and
                // declared on this backend only. The rule it advertises is a
                // fixed summation order, which a CUDA kernel built on atomics
                // could not honour, so the accelerator groups are deliberately
                // left to claim it once they have a kernel that can.
                ScatterAdd,
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
            composed_matmul = [
                BatchedMatMul, Addmm, ScaledDotProductAttention,
                // A dot is a multiply and an all-reduce; an outer product is
                // two unsqueezes and a broadcast multiply. Neither has a kernel,
                // and both inherit the matmul constraint rather than the wider
                // tensor one because that is what the reduce behind them holds.
                Dot, Outer
            ],
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
            quantizing = [Quantize],
            quantized = [Dequantize, QuantizedMatMul],
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
                Norm
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
            elementwise = [
                Add, Sub, Mul, Div,
                Relu, Step, Mish, Elu, Gelu, Abs, Exp, Neg, Sqrt, Log,
                Tanh, Sigmoid, Swish, Sign, Floor, Ceil, Round, Log2, Log10,
                Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh,
                Atanh, Erf, Rsqrt, Trunc, Frac,
                AddScalar, MulScalar, Powf, Clamp,
                SubScalar, DivScalar,
                Atan2, Fmod, Remainder,
                Dropout
            ],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
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
                TopK
            ],
            spatial = [
                Conv2dExact, Conv1dExact, ConvTranspose2d,
                MaxPool2d, AvgPool2d, AdaptiveAvgPool2dExact
            ],
            matmul = [MatMulExact],
            normalization = [Softmax, LayerNorm, BatchNorm, RmsNorm, GroupNorm],
            embedding = [EmbeddingExact],
            native_tensor = [
                ArgMax, ArgMin, Argsort, Cumsum,
                Maximum, Minimum, AbsDiff, Lerp, MaskedFill, WhereCond,
                CmpEq, CmpNe, CmpLt, CmpLe, CmpGt, CmpGe,
                TransposeExact, Narrow, Triu, Tril, Diag,
                ConcatExact, Gather, Scatter, IndexSelect, Repeat, Pad, Unfold,
                PixelShuffle,
                ToDType
            ],
            logical = [LogicalAnd, LogicalOr, LogicalNot],
            composed_tensor = [
                FlattenExact, SqueezeExact, UnsqueezeExact,
                StackExact, SliceExact, BroadcastLeft,
                Chunk, Split
            ],
            composed_matmul = [
                BatchedMatMul, Addmm, ScaledDotProductAttention,
                Dot, Outer
            ],
            composed_matmul_bias = [Linear],
            quantizing = [Quantize],
            quantized = [Dequantize, QuantizedMatMul],
            composed_reduction = [
                MseLoss, L1Loss, BceWithLogitsLoss,
                InstanceNorm,
                VarianceAll, VarianceDim, VarianceKeepDim,
                StdAll, StdDim, StdKeepDim,
                Norm
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
                Tanh, Sigmoid, Swish
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
                MaxDim, MaxKeepDim, MinDim, MinKeepDim, ProdDim
            ],
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            matmul = [MatMulExact],
            // No canonical executor was written for this backend beyond the
            // groups above, so it advertises none. An empty group is a truthful
            // claim; a copied one would not be.
            normalization = [],
            embedding = [],
            // Advertised now that each has an executor and a gradient path.
            // The comparison and logical modes of the same shader stay
            // unadvertised on purpose: they write 0.0/1.0 into an f32 buffer,
            // and this group's `tensor_dtypes` would claim an f32 result for
            // an operation the catalog types as boolean. Registering them needs
            // that representation settled first, not just a row.
            // `transpose` has its own WGSL kernel and its own tape entry (a
            // transpose is its own inverse), so it is native rather than
            // composed. It sat unregistered until now: the kernel existed,
            // nothing advertised it, and dispatch refused it.
            native_tensor = [Maximum, Minimum, AbsDiff, TransposeExact],
            // `rms_norm` only, not the whole normalization family: it is the
            // one member WGPU can answer, by rewriting into `mul`,
            // `mean_keepdim`, `add_scalar`, `sqrt` and `div`. `layer_norm`,
            // `batch_norm`, `group_norm` and `instance_norm` have no WGPU
            // path, and a family row would claim all five.
            logical = [],
            // All three rewrite into `reshape` rather than running a kernel of
            // their own: the elements are already in the right order and only
            // the shape changes. They push no tape entry of their own, so the
            // backward is `reshape`'s, which is what makes their `training`
            // claim true without new hand-derived math.
            composed_tensor = [FlattenExact, SqueezeExact, UnsqueezeExact],
            composed_matmul = [],
            composed_matmul_bias = [],
            quantizing = [],
            quantized = [],
            composed_reduction = [],
            composed_reduction_indexed = []
        }
    };
}

macro_rules! metal_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            elementwise = [Add, Sub, Mul, Div],
            broadcast = [BroadcastAs],
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
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            matmul = [MatMulExact],
            // No canonical executor was written for this backend beyond the
            // groups above, so it advertises none. An empty group is a truthful
            // claim; a copied one would not be.
            normalization = [],
            embedding = [],
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

pub(crate) use cuda_descriptor_operations;
pub(crate) use metal_descriptor_operations;
pub(crate) use wgpu_descriptor_operations;
