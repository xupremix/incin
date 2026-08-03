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
/// The only quantized representation any backend implements today.
const Q8_ONLY: &[DTypeId] = &[DTypeId::Q8_0];
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

/// An operation the backend answers by rewriting it into other operations.
///
/// `flatten`, `squeeze` and `unsqueeze` compute a target shape and call
/// `reshape`; `bmm` calls `matmul`. Reporting those as `native` would tell a
/// caller the backend has a dedicated kernel behind them, which is exactly the
/// question `ImplementationKind` exists to answer.
const fn composed_ranked(
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
        ImplementationKind::Composed,
    )
}

// Single declaration consumed by capability generation below, by the grouped
// legacy descriptor executors, and by the canonical per-identity executors in
// `cpu::canonical`. Adding an identity here changes what execution admits and
// what capability queries report in the same edit, and the canonical module
// turns the third consumer into a compile-time obligation: a row advertised
// without an `Execute<Descriptor<op::...>>` implementation does not build.
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
                Atan2, Fmod, Remainder,
                // `dropout` walks its operand once and writes one result of the
                // same shape, which is this group exactly. That it consults a
                // random draw on the way changes nothing the row states.
                Dropout
            ],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll, ProdAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim, ProdDim,
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
            normalization = [Softmax, LayerNorm, BatchNorm, RmsNorm],
            // `embedding` is the module-family operation deliberately absent
            // from every group above, and it is absent because of the contract
            // rather than because nobody wrote the executor. Its two operands
            // have different dtypes by construction: a float table and an
            // integer index. `dispatch::execute` resolves one capability row and
            // applies it to each operand in turn, so the row would have to
            // admit both dtype sets at once. Stating only f32 refuses every
            // legal call, and widening it to the non-quantized set claims f64
            // support the kernel answers by narrowing to f32. Per-operand dtype
            // sets are the change that unblocks it.
            native_tensor = [
                ArgMax, ArgMin, Argsort, Cumsum,
                Maximum, Minimum, AbsDiff, Lerp, MaskedFill, WhereCond,
                CmpEq, CmpNe, CmpLt, CmpLe, CmpGt, CmpGe,
                LogicalAnd, LogicalOr, LogicalNot,
                SubScalar, DivScalar,
                TransposeExact, Narrow, Triu, Tril, Diag,
                ConcatExact, Gather, Scatter, IndexSelect, Repeat, Pad, Unfold,
                PixelShuffle, GroupNorm,
                // `to_dtype` reads through the same stride-aware accessor and
                // writes a fresh contiguous buffer, which is this group's shape
                // exactly. Its target dtype is an attribute rather than an
                // operand, so the row constrains what it reads and the executor
                // constrains what it is asked to write.
                ToDType
            ],
            composed_tensor = [
                FlattenExact, SqueezeExact, UnsqueezeExact,
                StackExact, SliceExact, InstanceNorm, BroadcastLeft,
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
            // The losses that `LossOps` supplies as real composed defaults
            // rather than as stubs: each rewrites into `sub`, `mul`, `abs` and
            // an all-reduce. They inherit the reduction group's f32-only claim
            // because their `Mean` and `Sum` forms end in `mean_all`/`sum_all`,
            // and the reduction mode is an attribute rather than part of the
            // identity, so the row has to hold for the narrowest of the three.
            //
            // `cross_entropy_loss` is absent for the reason `embedding` is: its
            // logits are float and its targets are class indices, and one row
            // states one dtype set.
            // Two groups rather than one, because the compression and the
            // operations over compressed storage read opposite dtype sets and a
            // row states one. `quantize` reads f32 and writes blocks;
            // `dequantize` and `quantized_matmul` read blocks. Both refuse a
            // strided operand: the kernels index the block buffer directly and
            // never consult a stride.
            quantizing = [Quantize],
            quantized = [Dequantize, QuantizedMatMul],
            composed_reduction = [
                MseLoss, L1Loss, BceWithLogitsLoss,
                // Variance, standard deviation and the p-norm have no kernel of
                // their own on any backend: each is a subtract, a square, a
                // reduce and a scale over primitives already migrated above.
                // Same rule shape as the losses, for the same reason: they end
                // in an all-reduce or an axis reduce.
                VarianceAll, VarianceDim, VarianceKeepDim,
                StdAll, StdDim, StdKeepDim,
                Norm
            ]
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
            elementwise = [Add, Sub, Mul, Div],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim
            ],
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            matmul = [MatMulExact],
            // No canonical executor was written for this backend beyond the
            // groups above, so it advertises none. An empty group is a truthful
            // claim; a copied one would not be.
            normalization = [],
            native_tensor = [],
            composed_tensor = [],
            composed_matmul = [],
            composed_matmul_bias = [],
            quantizing = [],
            quantized = [],
            composed_reduction = []
        }
    };
}

macro_rules! wgpu_descriptor_operations {
    ($callback:ident, $($args:tt)*) => {
        $callback! {
            $($args)*;
            elementwise = [Add, Sub, Mul, Div],
            broadcast = [BroadcastAs],
            reshape = [ReshapeExact],
            reduction = [
                SumAll, MeanAll, MaxAll, MinAll,
                SumDim, SumKeepDim, MeanDim, MeanKeepDim,
                MaxDim, MaxKeepDim, MinDim, MinKeepDim
            ],
            spatial = [Conv2dExact, MaxPool2d, AvgPool2d],
            matmul = [MatMulExact],
            // No canonical executor was written for this backend beyond the
            // groups above, so it advertises none. An empty group is a truthful
            // claim; a copied one would not be.
            normalization = [],
            native_tensor = [],
            composed_tensor = [],
            composed_matmul = [],
            composed_matmul_bias = [],
            quantizing = [],
            quantized = [],
            composed_reduction = []
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
            native_tensor = [],
            composed_tensor = [],
            composed_matmul = [],
            composed_matmul_bias = [],
            quantizing = [],
            quantized = [],
            composed_reduction = []
        }
    };
}

macro_rules! descriptor_capability_rules {
    (
        elementwise = $elementwise:expr,
        broadcast = $broadcast:expr,
        reshape = $reshape:expr,
        reduction = $reduction:expr,
        spatial = $spatial:expr,
        matmul = $matmul:expr,
        normalization_dtypes = $normalization_dtypes:expr,
        broadcast_training = $broadcast_training:expr,
        reshape_training = $reshape_training:expr,
        elementwise_layouts = $elementwise_layouts:expr,
        broadcast_layouts = $broadcast_layouts:expr,
        reshape_layouts = $reshape_layouts:expr,
        reduction_layouts = $reduction_layouts:expr,
        spatial_layouts = $spatial_layouts:expr,
        matmul_layouts = $matmul_layouts:expr,
        quantized_dtypes = $quantized_dtypes:expr,
        quantized_layouts = $quantized_layouts:expr,
        tensor_dtypes = $tensor_dtypes:expr,
        tensor_layouts = $tensor_layouts:expr,
        legacy = [$($legacy:expr),* $(,)?];
        elementwise = [$($elementwise_op:ident),* $(,)?],
        broadcast = [$($broadcast_op:ident),* $(,)?],
        reshape = [$($reshape_op:ident),* $(,)?],
        reduction = [$($reduction_op:ident),* $(,)?],
        spatial = [$($spatial_op:ident),* $(,)?],
        matmul = [$($matmul_op:ident),* $(,)?],
        normalization = [$($normalization_op:ident),* $(,)?],
        native_tensor = [$($native_tensor_op:ident),* $(,)?],
        composed_tensor = [$($composed_tensor_op:ident),* $(,)?],
        composed_matmul = [$($composed_matmul_op:ident),* $(,)?],
        composed_matmul_bias = [$($composed_matmul_bias_op:ident),* $(,)?],
        quantizing = [$($quantizing_op:ident),* $(,)?],
        quantized = [$($quantized_op:ident),* $(,)?],
        composed_reduction = [$($composed_reduction_op:ident),* $(,)?]
    ) => {
        &[
            $($legacy,)*
            // Everything that walks its operands elementwise and writes one
            // result of the same shape: the arithmetic binaries, the whole
            // unary float set, the scalar-parametrised forms, and clamp. They
            // share one dtype and layout declaration because they share one
            // traversal, and every rank is a legal operand rank for all of them.
            $(native(OperationKind::$elementwise_op, $elementwise, $elementwise_layouts, true),)*
            $(native(OperationKind::$broadcast_op, $broadcast, $broadcast_layouts, false),)*
            $(native(OperationKind::$reshape_op, $reshape, $reshape_layouts, false),)*
            // A view operation records a gradient, so it is usable while
            // training over the dtypes a gradient exists for. Without these
            // rows the exact identities are strictly narrower than the legacy
            // family rows they replace, and a training reshape would stop
            // resolving the moment those family rows are removed.
            $(native(OperationKind::$broadcast_op, $broadcast_training, $broadcast_layouts, true),)*
            $(native(OperationKind::$reshape_op, $reshape_training, $reshape_layouts, true),)*
            $(native_ranked(
                OperationKind::$matmul_op,
                $matmul,
                $matmul_layouts,
                descriptor_min_rank(OperationKind::$matmul_op),
                descriptor_max_rank(OperationKind::$matmul_op),
                true,
            ),)*
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
            // `softmax` normalizes along an axis, so it needs one, and it does
            // not share the elementwise dtype set: the CPU kernel computes in
            // f32 and returns f32 storage, so advertising the half and double
            // types for it would be a claim execution does not honour.
            $(native_ranked(
                OperationKind::$normalization_op,
                $normalization_dtypes,
                $elementwise_layouts,
                descriptor_min_rank(OperationKind::$normalization_op),
                descriptor_max_rank(OperationKind::$normalization_op),
                true,
            ),)*
            // The tensor family reads its operands through the stride-aware
            // accessor and writes a fresh contiguous result, so its dtype and
            // layout sets are one declaration rather than one per operation.
            // The rank bounds are not: each identity states its own, because a
            // transpose and an unsqueeze do not accept the same ranks.
            $(native_ranked(
                OperationKind::$native_tensor_op,
                $tensor_dtypes,
                $tensor_layouts,
                descriptor_min_rank(OperationKind::$native_tensor_op),
                descriptor_max_rank(OperationKind::$native_tensor_op),
                true,
            ),)*
            // Same shape, reported as composed: these answer by rewriting into
            // another operation rather than by running a kernel of their own.
            $(composed_ranked(
                OperationKind::$composed_tensor_op,
                $tensor_dtypes,
                $tensor_layouts,
                descriptor_min_rank(OperationKind::$composed_tensor_op),
                descriptor_max_rank(OperationKind::$composed_tensor_op),
                true,
            ),)*
            // `bmm`, `addmm` and attention all rewrite into `matmul`, so they
            // inherit its dtype and layout constraint rather than the wider
            // tensor one.
            $(composed_ranked(
                OperationKind::$composed_matmul_op,
                $matmul,
                $matmul_layouts,
                descriptor_min_rank(OperationKind::$composed_matmul_op),
                descriptor_max_rank(OperationKind::$composed_matmul_op),
                true,
            ),)*
            // Same constraint as the product they wrap, with the rank bound
            // widened to admit the rank-one bias that travels beside it.
            $(composed_ranked(
                OperationKind::$composed_matmul_bias_op,
                $matmul,
                $matmul_layouts,
                descriptor_min_rank(OperationKind::$composed_matmul_bias_op),
                descriptor_max_rank(OperationKind::$composed_matmul_bias_op),
                true,
            ),)*
            // The compression reads the float set its kernel accepts, which is
            // narrower than the elementwise one: it matches on the buffer
            // variant rather than converting.
            $(native_ranked(
                OperationKind::$quantizing_op,
                $reduction,
                $quantized_layouts,
                descriptor_min_rank(OperationKind::$quantizing_op),
                descriptor_max_rank(OperationKind::$quantizing_op),
                false,
            ),)*
            // Operations over compressed storage. `training` is false on both:
            // quantization is not differentiable and neither kernel pushes a
            // tape entry, so advertising them for training would promise a
            // gradient that never arrives.
            $(native_ranked(
                OperationKind::$quantized_op,
                $quantized_dtypes,
                $quantized_layouts,
                descriptor_min_rank(OperationKind::$quantized_op),
                descriptor_max_rank(OperationKind::$quantized_op),
                false,
            ),)*
            // Same relationship to the reduction rows: a loss that ends in an
            // all-reduce cannot claim a dtype the all-reduce refuses.
            $(composed_ranked(
                OperationKind::$composed_reduction_op,
                $reduction,
                $reduction_layouts,
                descriptor_min_rank(OperationKind::$composed_reduction_op),
                descriptor_max_rank(OperationKind::$composed_reduction_op),
                true,
            ),)*
        ]
    };
}

/// The lowest rank any *operand* of `operation` may have.
///
/// Not the lowest rank of its primary operand, which is the reading that looks
/// right and is wrong. `dispatch::execute` resolves one capability row per
/// operation and then applies it to every operand in turn, so a bound derived
/// from the activation alone is also asserted against the bias. `conv2d` was
/// written that way and consequently refused its own rank-one bias for the
/// whole of its migration; `a_convolution_with_a_bias_is_not_refused_by_its_own_rank_bound`
/// in `cpu::canonical` is the regression test for it.
///
/// Nothing is lost by taking the minimum here. The primary operand's real bound
/// is enforced by the descriptor's `AttributeContract::validate`, which runs
/// before any capability query and can see which operand is which. This table
/// only has to avoid contradicting it.
const fn descriptor_min_rank(operation: OperationKind) -> usize {
    match operation {
        // Single-operand windows, so the activation's bound is the whole
        // operation's bound.
        OperationKind::MaxPool2d
        | OperationKind::AvgPool2d
        | OperationKind::AdaptiveAvgPool2dExact => 3,
        // The convolutions want three or four axes from the activation and
        // four or fewer from the weight, but each also takes an optional
        // rank-one bias, and that is the minimum the row can state. The
        // activation bound lives in `spatial_contract!` instead.
        OperationKind::Conv1dExact
        | OperationKind::Conv2dExact
        | OperationKind::ConvTranspose2d => 1,
        // A product needs two axes; batching it needs a third to batch over.
        // `softmax` needs one axis to normalize along, and `layer_norm`
        // normalizes over a trailing suffix, so it needs one too.
        OperationKind::MatMulExact => 2,
        // The quantized product reads its right operand as a two-axis [N, K]
        // weight and refuses a left operand with fewer than two axes.
        OperationKind::QuantizedMatMul => 2,
        OperationKind::BatchedMatMul => 3,
        OperationKind::Softmax | OperationKind::LayerNorm | OperationKind::RmsNorm => 1,
        // The weight is a matrix and the input has at least the feature axis,
        // but the bias beside them is rank one and this row speaks for it too.
        OperationKind::Linear => 1,
        // `BatchNormAttributes::validate` refuses an input without a channel
        // axis, but the weight, bias and running statistics are per-channel
        // vectors, so rank one is what this row has to admit.
        OperationKind::BatchNorm => 1,
        OperationKind::SumDim
        | OperationKind::SumKeepDim
        | OperationKind::MeanDim
        | OperationKind::MeanKeepDim
        | OperationKind::MaxDim
        | OperationKind::MaxKeepDim
        | OperationKind::MinDim
        | OperationKind::MinKeepDim
        | OperationKind::ProdDim
        // An axis-bearing scan or ordering needs an axis to run along.
        // `argmax` and `argmin` do not appear here: their axis is optional and
        // the flattened form is defined for a scalar, so their minimum is zero.
        | OperationKind::Cumsum
        | OperationKind::Argsort
        | OperationKind::TopK => 1,
        // A transpose needs two axes to swap; every other view here needs at
        // least the one axis its attributes name. `unsqueeze` is the exception
        // that keeps this table honest: it inserts an axis, so a scalar is a
        // legitimate operand and its minimum stays at zero.
        OperationKind::TransposeExact => 2,
        OperationKind::Narrow
        | OperationKind::FlattenExact
        | OperationKind::SqueezeExact
        | OperationKind::Triu
        | OperationKind::Tril
        | OperationKind::Diag => 1,
        // Measured, not assumed: each of these was run against ranks zero
        // through four and this is the lowest one that executed. The indexing
        // operations need an axis to index; `unfold` needs one to slide along;
        // `concat` needs an existing axis to join on.
        OperationKind::ConcatExact
        | OperationKind::Gather
        | OperationKind::Scatter
        | OperationKind::IndexSelect
        | OperationKind::Unfold => 1,
        // `addmm` broadcasts its addend against the product, so a per-column
        // rank-one addend is a legal operand of a rank-two operation, and the
        // same per-operand rule that governs the convolution biases applies.
        OperationKind::Addmm => 1,
        OperationKind::ScaledDotProductAttention => 2,
        // Matched to the descriptor, not to the kernel. `group_norm` needs a
        // channel axis and `instance_norm` needs the full [N, C, H, W] layout;
        // the CPU kernels accept less than that, but a row wider than what the
        // descriptor validates advertises requests that can never reach it.
        OperationKind::GroupNorm => 2,
        OperationKind::InstanceNorm | OperationKind::PixelShuffle => 4,
        _ => 0,
    }
}

const fn descriptor_max_rank(operation: OperationKind) -> usize {
    match operation {
        OperationKind::Conv2dExact
        | OperationKind::ConvTranspose2d
        | OperationKind::MaxPool2d
        | OperationKind::AvgPool2d
        | OperationKind::AdaptiveAvgPool2dExact => 4,
        // A one-dimensional convolution reads [N, C, L] or the unbatched [C, L];
        // there is no fourth axis for it to interpret.
        OperationKind::Conv1dExact => 3,
        // `DiagonalAttributes` refuses anything outside rank one or two, so a
        // wider row would advertise ranks the descriptor rejects before the
        // backend is ever reached.
        OperationKind::Triu | OperationKind::Tril | OperationKind::Diag => 2,
        // `pixel_shuffle` reads a four-axis (N, C, H, W) layout by name; no
        // other rank has an interpretation for it.
        OperationKind::PixelShuffle | OperationKind::InstanceNorm => 4,
        _ => MAX_RANK,
    }
}

pub static CPU_CAPABILITIES: &[CapabilityRule] = cpu_descriptor_operations!(
    descriptor_capability_rules,
    elementwise = FLOAT_DTYPES,
    broadcast = ALL_DTYPES,
    reshape = ALL_DTYPES,
    reduction = F32_ONLY,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
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
    elementwise = FLOAT_DTYPES,
    broadcast = CUDA_STORAGE_DTYPES,
    reshape = CUDA_STORAGE_DTYPES,
    reduction = FLOAT_DTYPES,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
    broadcast_training = FLOAT_DTYPES,
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
    elementwise = F32_ONLY,
    broadcast = F32_ONLY,
    reshape = F32_ONLY,
    reduction = F32_ONLY,
    spatial = F32_ONLY,
    matmul = F32_ONLY,
    normalization_dtypes = F32_ONLY,
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
    elementwise = FLOAT_DTYPES,
    broadcast = CUDA_STORAGE_DTYPES,
    reshape = CUDA_STORAGE_DTYPES,
    reduction = FLOAT_DTYPES,
    spatial = F32_ONLY,
    matmul = FLOAT_DTYPES,
    normalization_dtypes = F32_ONLY,
    broadcast_training = FLOAT_DTYPES,
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
