//! Assembly of `CapabilityRule` entries from a backend's per-group
//! declarations.
//!
//! `native`, `native_ranked` and `composed_ranked` are the row constructors;
//! `descriptor_capability_rules!` is the callback every `*_descriptor_operations!`
//! macro in `super::declarations` feeds its per-group dtype, layout and
//! operation-name arguments into, one row per operation. `descriptor_min_rank`
//! and `descriptor_max_rank` supply the rank bounds for the identities whose
//! legal rank is not `0..=usize::MAX`.
//!
//! All five stay `pub(super)` rather than fully private: `super::tables`
//! calls them directly too, both inside its own `legacy = [...]` blocks
//! (ordinary expressions, resolved where they are written) and, for `native`/
//! `native_ranked`/`composed_ranked`/`CONTIGUOUS`/`PRECISE`, from *inside*
//! `descriptor_capability_rules!`'s own body - a `$callback:ident` forwarded
//! through `cpu_descriptor_operations!` and friends carries the invocation's
//! syntax context, not this file's, so a bare identifier written in the
//! macro's definition here still resolves against whatever the ultimate
//! call site (`tables.rs`) has imported, not against this file's own `use`
//! declarations. `super::constants::PRECISE` is imported here anyway because
//! `native`/`native_ranked`/`composed_ranked` are ordinary `const fn`s, not
//! macro-forwarded text, and so resolve normally at their own definition site.

use super::constants::PRECISE;
use incin_core::exec::LayoutClass;
use incin_core::exec::{CapabilityRule, ImplementationKind};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeDescriptor;

pub(super) const fn native(
    operation: OperationKind,
    dtypes: &'static [DTypeDescriptor],
    layouts: &'static [LayoutClass],
    training: bool,
) -> CapabilityRule {
    native_ranked(operation, dtypes, layouts, 0, usize::MAX, training)
}

pub(super) const fn native_ranked(
    operation: OperationKind,
    dtypes: &'static [DTypeDescriptor],
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
pub(super) const fn composed_ranked(
    operation: OperationKind,
    dtypes: &'static [DTypeDescriptor],
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

macro_rules! descriptor_capability_rules {
    (
        elementwise = $elementwise:expr,
        broadcast = $broadcast:expr,
        reshape = $reshape:expr,
        reduction = $reduction:expr,
        filling_dtypes = $filling_dtypes:expr,
        sampling_dtypes = $sampling_dtypes:expr,
        spatial = $spatial:expr,
        matmul = $matmul:expr,
        normalization_dtypes = $normalization_dtypes:expr,
        embedding_dtypes = $embedding_dtypes:expr,
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
        logical_dtypes = $logical_dtypes:expr,
        legacy = [$($legacy:expr),* $(,)?];
        elementwise = [$($elementwise_op:ident),* $(,)?],
        broadcast = [$($broadcast_op:ident),* $(,)?],
        reshape = [$($reshape_op:ident),* $(,)?],
        filling = [$($filling_op:ident),* $(,)?],
        sampling = [$($sampling_op:ident),* $(,)?],
        readback = [$($readback_op:ident),* $(,)?],
        reduction = [$($reduction_op:ident),* $(,)?],
        spatial = [$($spatial_op:ident),* $(,)?],
        matmul = [$($matmul_op:ident),* $(,)?],
        normalization = [$($normalization_op:ident),* $(,)?],
        embedding = [$($embedding_op:ident),* $(,)?],
        native_tensor = [$($native_tensor_op:ident),* $(,)?],
        logical = [$($logical_op:ident),* $(,)?],
        composed_tensor = [$($composed_tensor_op:ident),* $(,)?],
        composed_matmul = [$($composed_matmul_op:ident),* $(,)?],
        composed_matmul_bias = [$($composed_matmul_bias_op:ident),* $(,)?],
        quantizing = [$($quantizing_op:ident),* $(,)?],
        quantized = [$($quantized_op:ident),* $(,)?],
        composed_reduction = [$($composed_reduction_op:ident),* $(,)?],
        composed_reduction_indexed = [$($composed_reduction_indexed_op:ident),* $(,)?]
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
            // Allocation, which admits every rank the shape contract allows and
            // is contiguous by construction.
            $(native(OperationKind::$filling_op, $filling_dtypes, CONTIGUOUS, false),)*
            $(native(OperationKind::$sampling_op, $sampling_dtypes, CONTIGUOUS, false),)*
            $(native(OperationKind::$readback_op, $tensor_dtypes, $tensor_layouts, false),)*
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
            // The union of the index operand's integer dtypes and the weight
            // operand's f32-only one - see `INDEX_AND_F32_DTYPES`'s own doc for why
            // one row cannot state the tighter, per-operand pair directly.
            $(native_ranked(
                OperationKind::$embedding_op,
                $embedding_dtypes,
                $elementwise_layouts,
                descriptor_min_rank(OperationKind::$embedding_op),
                descriptor_max_rank(OperationKind::$embedding_op),
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
            // Boolean throughout, on every operand and on the result, which
            // is why these cannot sit in the tensor group even though they
            // read through the same accessor. The tensor group's dtype set is
            // whatever the backend can hold, and `dispatch::execute` checks
            // every operand against the one resolved row, so a wider row here
            // advertises `logical_and` over `f32` while the descriptor
            // contract refuses it before any kernel is reached. Unlike
            // `where_cond`'s `F32_AND_BOOL`, no union is needed: there is no
            // mixed-dtype operand to union against.
            $(native_ranked(
                OperationKind::$logical_op,
                $logical_dtypes,
                $tensor_layouts,
                descriptor_min_rank(OperationKind::$logical_op),
                descriptor_max_rank(OperationKind::$logical_op),
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
            // The same rule, widened to the union of a float operand's dtypes
            // and an integer index operand's - see `INDEX_AND_F32_DTYPES`.
            $(composed_ranked(
                OperationKind::$composed_reduction_indexed_op,
                $embedding_dtypes,
                $reduction_layouts,
                descriptor_min_rank(OperationKind::$composed_reduction_indexed_op),
                descriptor_max_rank(OperationKind::$composed_reduction_indexed_op),
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
pub(super) const fn descriptor_min_rank(operation: OperationKind) -> usize {
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
        OperationKind::Softmax
        | OperationKind::LogSoftmax
        | OperationKind::LayerNorm
        | OperationKind::RmsNorm => 1,
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
        | OperationKind::LogSumExpDim
        | OperationKind::LogSumExpKeepDim
        // An axis-bearing scan or ordering needs an axis to run along.
        // `argmax` and `argmin` do not appear here: their axis is optional and
        // the flattened form is defined for a scalar, so their minimum is zero.
        | OperationKind::Cumsum
        | OperationKind::Argsort
        | OperationKind::TopK
        // `dot` contracts one axis away and `outer` expands two vectors into a
        // matrix; neither has anything to do on a scalar. Both reached this
        // function's zero default until the conformance harness posed a
        // rank-zero `dot` and the descriptor's own extent check indexed into an
        // empty shape.
        | OperationKind::Dot
        | OperationKind::Outer => 1,
        // A transpose needs two axes to swap; every other view here needs at
        // least the one axis its attributes name. `unsqueeze` is the exception
        // that keeps this table honest: it inserts an axis, so a scalar is a
        // legitimate operand and its minimum stays at zero.
        OperationKind::TransposeExact | OperationKind::TransposeView => 2,
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
        | OperationKind::ScatterAdd
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
        // The weight table is always rank two; the index operand carries
        // whatever batch geometry addresses it, down to a single-axis vector
        // of indices - a scalar index is not accepted. One is therefore the
        // loosest bound the row can honestly state, matching the per-operand
        // rank contract the descriptor validates separately.
        OperationKind::EmbeddingExact => 1,
        _ => 0,
    }
}

pub(super) const fn descriptor_max_rank(operation: OperationKind) -> usize {
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
        _ => usize::MAX,
    }
}

pub(super) use descriptor_capability_rules;
