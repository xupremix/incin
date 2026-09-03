//! The tables themselves: one entry per operation, grouped by operand contract.
//!
//! The shims come first and the group tables after, but a shim serving exactly
//! one family is written beside that family rather than up here, because the
//! attribute it fixes is only meaningful next to the operands it is fixed for.
//!
//! The groups of `cpu_descriptor_operations!` are where the operand contracts
//! are already written down, and their comments in `declarations.rs` explain
//! each one, so the families below are read off those groups. They are finer
//! than the groups rather than equal to them: a group exists to share a
//! capability row, and a row says nothing about arity, so a single group can
//! hold a unary and a binary operation that need different operands.

use incin_core::backend_authoring::{Execute, op};
use incin_core::exec::catalog::{
    AddmmAttributes, ArgsortAttributes, AttentionAttributes, AxisAttributes,
    AxisVarianceAttributes, ChunkAttributes, ClampAttributes, DTypeAttributes, DiagonalAttributes,
    DropoutAttributes, DuplicateIndexRule, EpsilonAttributes, FlattenAttributes,
    GroupNormAttributes, IndexReductionAttributes, LerpAttributes, LinearAttributes,
    LossAttributes, LossReduction, NarrowAttributes, NoAttributes, NormAttributes, PadAttributes,
    QuantizationAttributes, RepeatAttributes, ScalarAttributes, ScatterAttributes, ShapeAttributes,
    SliceAttributes, SplitAttributes, TopKAttributes, TransposeAttributes, VarianceAttributes,
};
use incin_core::exec::{CanonicalError, Capabilities, ExecutionContext, Operation, TensorHandle};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;

use crate::conformance::operands::materialized_extents;
use crate::conformance::plan::AdvertisedTuple;

use super::contracts::{Fixture, Operands, Role, Route, Subject};

// ============================================================================
// Typed shims
// ============================================================================

fn plain<O>(
    context: &ExecutionContext<Subject>,
    _tuple: &AdvertisedTuple,
    route: Route,
    inputs: &[TensorHandle<'_>],
) -> Result<(), CanonicalError>
where
    O: Operation<Attributes = NoAttributes> + incin_core::exec::CanonicalOperation,
    Subject: Execute<O> + Capabilities,
{
    route.run::<O>(context, NoAttributes, inputs)
}

/// Axis-bearing operations are driven on axis zero.
///
/// Zero rather than the last axis because it is the only index every rank in a
/// rule's range shares. Driving the last axis would silently change which axis
/// is under test as the rank boundary moves, and a kernel that mishandles axis
/// zero at rank one is exactly the boundary this harness exists to reach.
pub(crate) fn on_axis_zero<O>(
    context: &ExecutionContext<Subject>,
    _tuple: &AdvertisedTuple,
    route: Route,
    inputs: &[TensorHandle<'_>],
) -> Result<(), CanonicalError>
where
    O: Operation<Attributes = AxisAttributes> + incin_core::exec::CanonicalOperation,
    Subject: Execute<O> + Capabilities,
{
    route.run::<O>(context, AxisAttributes { axis: 0 }, inputs)
}

/// Scalar-bearing operations are driven with two.
///
/// Not one, and not zero. One is the identity for `mul_scalar` and `powf`, and
/// zero is the identity for `add_scalar`, so either would let a kernel that
/// ignores its scalar entirely produce the right answer.
fn with_scalar_two<O>(
    context: &ExecutionContext<Subject>,
    _tuple: &AdvertisedTuple,
    route: Route,
    inputs: &[TensorHandle<'_>],
) -> Result<(), CanonicalError>
where
    O: Operation<Attributes = ScalarAttributes> + incin_core::exec::CanonicalOperation,
    Subject: Execute<O> + Capabilities,
{
    route.run::<O>(context, ScalarAttributes { value: 2.0 }, inputs)
}

/// The index reductions report a position rather than a value, so their
/// attributes carry the index dtype alongside the axis. `i64` is the dtype the
/// CPU kernels write, and stating anything else here would test the harness's
/// preference rather than the row.
fn index_on_axis_zero<O>(
    context: &ExecutionContext<Subject>,
    tuple: &AdvertisedTuple,
    route: Route,
    inputs: &[TensorHandle<'_>],
) -> Result<(), CanonicalError>
where
    O: Operation<Attributes = IndexReductionAttributes> + incin_core::exec::CanonicalOperation,
    Subject: Execute<O> + Capabilities,
{
    route.run::<O>(
        context,
        IndexReductionAttributes {
            // `None` is the flattened form, and it is the only one a scalar
            // has. The rows for these two operations reach down to rank zero
            // because that form is defined there; asking for axis zero instead
            // would report the harness's choice as the backend's defect.
            axis: (tuple.rank > 0).then_some(0),
            dtype: DTypeId::I64.descriptor(),
        },
        inputs,
    )
}

/// A shim for an attribute type with one sensible constant value.
///
/// Every attribute below is chosen so that a kernel ignoring it cannot pass:
/// a clamp range that actually clips the operand ladder, a lerp weight that is
/// neither endpoint, a variance that is the biased form so the unbiased one
/// would differ, a dropout probability that is not zero or one.
macro_rules! constant_attribute_shim {
    ($name:ident, $attributes:ty, $value:expr) => {
        pub(crate) fn $name<O>(
            context: &ExecutionContext<Subject>,
            _tuple: &AdvertisedTuple,
            route: Route,
            inputs: &[TensorHandle<'_>],
        ) -> Result<(), CanonicalError>
        where
            O: Operation<Attributes = $attributes> + incin_core::exec::CanonicalOperation,
            Subject: Execute<O> + Capabilities,
        {
            route.run::<O>(context, $value, inputs)
        }
    };
}

constant_attribute_shim!(
    clamped,
    ClampAttributes,
    ClampAttributes {
        min: -1.0,
        max: 1.0
    }
);
constant_attribute_shim!(
    interpolated,
    LerpAttributes,
    LerpAttributes { weight: 0.25 }
);
constant_attribute_shim!(
    transposing_first_two,
    TransposeAttributes,
    TransposeAttributes {
        first: 0,
        second: 1
    }
);
constant_attribute_shim!(
    on_main_diagonal,
    DiagonalAttributes,
    DiagonalAttributes { offset: 0 }
);
constant_attribute_shim!(p_norm, NormAttributes, NormAttributes { order: 2.0 });
constant_attribute_shim!(
    biased_variance,
    VarianceAttributes,
    VarianceAttributes { unbiased: false }
);
constant_attribute_shim!(
    biased_axis_variance,
    AxisVarianceAttributes,
    AxisVarianceAttributes {
        axis: 0,
        unbiased: false
    }
);
constant_attribute_shim!(
    with_epsilon,
    EpsilonAttributes,
    EpsilonAttributes { epsilon: 1e-5 }
);
constant_attribute_shim!(
    dropping_half,
    DropoutAttributes,
    DropoutAttributes {
        probability: 0.5,
        training: true
    }
);
constant_attribute_shim!(
    mean_reduced,
    LossAttributes,
    LossAttributes {
        reduction: LossReduction::Mean
    }
);

/// A shim whose attributes are computed from the tuple.
///
/// Anything naming an extent or an axis per dimension has to be, since the rank
/// moves as the harness walks a rule's boundary and a fixed attribute would be
/// valid at one end and nonsense at the other.
macro_rules! derived_attribute_shim {
    ($name:ident, $attributes:ty, |$tuple:ident| $value:expr) => {
        pub(crate) fn $name<O>(
            context: &ExecutionContext<Subject>,
            $tuple: &AdvertisedTuple,
            route: Route,
            inputs: &[TensorHandle<'_>],
        ) -> Result<(), CanonicalError>
        where
            O: Operation<Attributes = $attributes> + incin_core::exec::CanonicalOperation,
            Subject: Execute<O> + Capabilities,
        {
            route.run::<O>(context, $value, inputs)
        }
    };
}

constant_attribute_shim!(
    to_f32,
    DTypeAttributes,
    DTypeAttributes {
        dtype: DTypeId::F32.descriptor()
    }
);
constant_attribute_shim!(
    narrowing_to_one,
    NarrowAttributes,
    NarrowAttributes {
        axis: 0,
        start: 0,
        length: 1
    }
);
constant_attribute_shim!(
    one_chunk,
    ChunkAttributes,
    ChunkAttributes { chunks: 1, axis: 0 }
);
constant_attribute_shim!(
    splitting_by_one,
    SplitAttributes,
    SplitAttributes {
        split_size: 1,
        axis: 0
    }
);
constant_attribute_shim!(
    largest_one,
    TopKAttributes,
    TopKAttributes {
        k: 1,
        axis: 0,
        largest: true,
        index_dtype: DTypeId::I64.descriptor()
    }
);
constant_attribute_shim!(
    ascending,
    ArgsortAttributes,
    ArgsortAttributes {
        axis: 0,
        descending: false,
        index_dtype: DTypeId::I64.descriptor()
    }
);
constant_attribute_shim!(
    single_group,
    GroupNormAttributes,
    GroupNormAttributes {
        groups: 1,
        epsilon: 1e-5
    }
);

// The operand's own extents, which makes reshape and broadcast identities.
// An identity is the weakest form of both, and deliberately: this harness asks
// whether the tuple runs at all, and a reshape that genuinely reassociates
// extents needs a factor pair the rank ladder does not always have.
derived_attribute_shim!(identity_shape, ShapeAttributes, |tuple| ShapeAttributes {
    shape: materialized_extents(tuple)
});

derived_attribute_shim!(flatten_fully, FlattenAttributes, |tuple| {
    FlattenAttributes {
        start_axis: 0,
        end_axis: tuple.rank.saturating_sub(1),
    }
});

derived_attribute_shim!(first_along_each_axis, SliceAttributes, |tuple| {
    SliceAttributes {
        ranges: alloc::vec![(0, 1); tuple.rank],
    }
});

derived_attribute_shim!(padding_by_one, PadAttributes, |tuple| PadAttributes {
    padding: alloc::vec![(1, 1); tuple.rank],
    value: 0.0
});

derived_attribute_shim!(repeating_twice, RepeatAttributes, |tuple| {
    RepeatAttributes {
        repeats: alloc::vec![2; tuple.rank],
    }
});

// ============================================================================
// Group tables
// ============================================================================

macro_rules! family {
    ($name:ident, $operands:expr, $shim:ident, [$($operation:ident),* $(,)?]) => {
        pub(crate) fn $name(operation: OperationKind) -> Option<Fixture> {
            match operation {
                $(OperationKind::$operation => Some(Fixture {
                    operands: $operands,
                    roles: &[],
                    run: $shim::<op::$operation>,
                }),)*
                _ => None,
            }
        }
    };
}

/// A family whose operands do not all carry the tuple's dtype.
macro_rules! typed_family {
    ($name:ident, $operands:expr, $roles:expr, $shim:ident, [$($operation:ident),* $(,)?]) => {
        pub(crate) fn $name(operation: OperationKind) -> Option<Fixture> {
            match operation {
                $(OperationKind::$operation => Some(Fixture {
                    operands: $operands,
                    roles: $roles,
                    run: $shim::<op::$operation>,
                }),)*
                _ => None,
            }
        }
    };
}

// `macro_rules!` is textual and scoped to the rest of its own file, so a macro
// is not an item another module can name until this line makes it one. `shaped`
// is the module that needs them, because it holds the families whose attributes
// name extents and so has to build shims of its own; the re-export it actually
// imports from is in `super`.
pub(crate) use {constant_attribute_shim, derived_attribute_shim, family, typed_family};

family!(
    unary_float,
    Operands::Unary,
    plain,
    [
        Relu, Step, Mish, Elu, Gelu, Abs, Exp, Neg, Sqrt, Log, Tanh, Sigmoid, Swish, Sign, Floor,
        Ceil, Round, Log2, Log10, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh,
        Erf, Rsqrt, Trunc, Frac,
    ]
);

family!(
    binary_elementwise,
    Operands::Binary,
    plain,
    [
        Add, Sub, Mul, Div, Atan2, Fmod, Remainder, Maximum, Minimum, AbsDiff, CmpEq, CmpNe, CmpLt,
        CmpLe, CmpGt, CmpGe, LogicalAnd, LogicalOr,
    ]
);

family!(unary_logical, Operands::Unary, plain, [LogicalNot]);

family!(
    scalar_elementwise,
    Operands::Unary,
    with_scalar_two,
    [AddScalar, MulScalar, SubScalar, DivScalar, Powf,]
);

family!(
    reduce_all,
    Operands::Unary,
    plain,
    [SumAll, MeanAll, MaxAll, MinAll, ProdAll,]
);

family!(
    reduce_axis,
    Operands::UnaryAxis,
    on_axis_zero,
    [
        SumDim,
        SumKeepDim,
        MeanDim,
        MeanKeepDim,
        MaxDim,
        MaxKeepDim,
        MinDim,
        MinKeepDim,
        ProdDim,
        LogSumExpDim,
        LogSumExpKeepDim,
        Cumsum,
        Softmax,
        // Same operand shape and the same single `axis` attribute as `softmax`
        // beside it, because it is that operation stopped before its final
        // exponential. The oracle asks whether the advertised tuple executes,
        // not whether the numbers are right, so it needs nothing narrower here;
        // the value the log form must produce is pinned by the shared
        // conformance vector and by `tensor_ops.rs`.
        LogSoftmax,
    ]
);

family!(
    index_reduce_axis,
    Operands::Unary,
    index_on_axis_zero,
    [ArgMax, ArgMin]
);

family!(
    readback,
    Operands::Unary,
    plain,
    [ToHostFloatVec, ToHostIntVec, TensorToBytes,]
);

family!(
    readback_scalar,
    Operands::UnaryScalar,
    plain,
    [ToHostFloatScalar, ToHostIntScalar,]
);

family!(clamping, Operands::Unary, clamped, [Clamp]);

family!(interpolating, Operands::Binary, interpolated, [Lerp]);

family!(
    transposing,
    Operands::UnaryAxis,
    transposing_first_two,
    [TransposeExact, TransposeView]
);

family!(
    diagonal_shape,
    Operands::Unary,
    on_main_diagonal,
    [Triu, Tril, Diag]
);

family!(norm_reduce, Operands::Unary, p_norm, [Norm]);

family!(
    variance_all,
    Operands::Unary,
    biased_variance,
    [VarianceAll, StdAll]
);

family!(
    variance_axis,
    Operands::UnaryAxis,
    biased_axis_variance,
    [VarianceDim, VarianceKeepDim, StdDim, StdKeepDim]
);

family!(epsilon_unary, Operands::Unary, with_epsilon, [InstanceNorm]);

family!(dropping, Operands::Unary, dropping_half, [Dropout]);

// `where_cond` selects between two float operands with a bool mask, and
// `masked_fill` overwrites a float operand where its mask is set. Both rows
// declare `F32_AND_BOOL`, which is the union of the two operand dtypes and not
// a claim that either operand may be either, so the roles state the split the
// row cannot.
typed_family!(
    selecting,
    Operands::Triple,
    &[Role::Mask, Role::Float, Role::Float],
    plain,
    [WhereCond]
);

typed_family!(
    masking,
    Operands::Binary,
    &[Role::Float, Role::Mask],
    with_scalar_two,
    [MaskedFill]
);

// `gather` reads an index of the operand's own rank; `index_select` reads a
// vector of them. Both are filled with zeros, the one value in range for every
// table extent the ladder produces.
typed_family!(
    gathering,
    Operands::Binary,
    &[Role::Tuple, Role::Index],
    on_axis_zero,
    [Gather]
);

typed_family!(
    selecting_rows,
    Operands::Binary,
    &[Role::Tuple, Role::IndexVector],
    on_axis_zero,
    [IndexSelect]
);

// `embedding` is the other `INDEX_AND_F32_DTYPES` row: an integer index vector
// against a rank-two float table, which is the shape pair the union cannot
// state any more than it can state the dtypes.
typed_family!(
    embedding_lookup,
    Operands::Binary,
    &[Role::IndexVector, Role::FloatMatrix],
    plain,
    [EmbeddingExact]
);

// The three extents a matrix product names: `[..batch, M, K]` against
// `[..batch, K, N]`, agreeing on `K` and on nothing else. Unequal on purpose,
// as the rank ladder is: a kernel that reads an extent off the wrong axis
// produces the right answer on a square product and the wrong one here.
const M: usize = 2;
const K: usize = 3;
const N: usize = 2;

typed_family!(
    matrix_product,
    Operands::Binary,
    &[
        Role::Paired {
            rows: M,
            columns: K
        },
        Role::Paired {
            rows: K,
            columns: N
        }
    ],
    plain,
    [MatMulExact, BatchedMatMul]
);

// `addmm` is the same product with an addend broadcast against it, so the
// addend is the product's own shape. It comes first because that is the order
// the inference reads: addend, then the two factors.
constant_attribute_shim!(
    unscaled_sum,
    AddmmAttributes,
    AddmmAttributes {
        alpha: 1.0,
        beta: 1.0
    }
);

typed_family!(
    fused_product_sum,
    Operands::Triple,
    &[
        Role::Paired {
            rows: M,
            columns: N
        },
        Role::Paired {
            rows: M,
            columns: K
        },
        Role::Paired {
            rows: K,
            columns: N
        }
    ],
    unscaled_sum,
    [Addmm]
);

// Attention agrees on two extents rather than one: the query and the key share
// their width, and the key and the value share their sequence length. Four
// distinct numbers, none of which the row can state.
constant_attribute_shim!(
    unmasked_attention,
    AttentionAttributes,
    AttentionAttributes {
        scale: None,
        has_mask: false
    }
);

typed_family!(
    attending,
    Operands::Triple,
    &[
        Role::Paired {
            rows: 2,
            columns: 3
        },
        Role::Paired {
            rows: 4,
            columns: 3
        },
        Role::Paired {
            rows: 4,
            columns: 2
        }
    ],
    unmasked_attention,
    [ScaledDotProductAttention]
);

// `linear` projects the input's final extent through a `[out, in]` weight and
// adds one value per output feature, which is the same bias shape a
// convolution reads.
constant_attribute_shim!(
    biased_projection,
    LinearAttributes,
    LinearAttributes { has_bias: true }
);

typed_family!(
    projecting,
    Operands::Triple,
    &[Role::Tuple, Role::LinearWeight, Role::OutputVector],
    biased_projection,
    [Linear]
);

// `cross_entropy_loss` is the third `INDEX_AND_F32_DTYPES` row, and the only
// one whose float operand is pinned to rank two: `operand_rank_window` gives
// the logits `[batch, classes]` and the targets `[batch]`. Both shapes the
// harness already builds for `embedding`, in the other order.
typed_family!(
    class_loss,
    Operands::Binary,
    &[Role::FloatMatrix, Role::IndexVector],
    mean_reduced,
    [CrossEntropyLoss]
);

// `scatter` writes a source into a target at an index, and `validated.rs`
// requires the index and the source to share a shape and to fit the target
// along every axis but the scattered one. Three operands of the tuple's own
// shape satisfy all of that, with the index reading zeros so every write lands
// in range.
constant_attribute_shim!(
    scattering_at_zero,
    ScatterAttributes,
    ScatterAttributes {
        axis: 0,
        duplicate_indices: DuplicateIndexRule::LastWriteWins,
    }
);

typed_family!(
    scattering,
    Operands::Triple,
    &[Role::Tuple, Role::Index, Role::Tuple],
    scattering_at_zero,
    [Scatter]
);

// Same operands as `scattering` above, but it cannot share that shim. The
// executor refuses every rule but its own, so posing it with the
// last-write-wins attribute would record a refusal the operation is entitled to
// give rather than a tuple that failed to execute.
constant_attribute_shim!(
    accumulating_at_zero,
    ScatterAttributes,
    ScatterAttributes {
        axis: 0,
        duplicate_indices: DuplicateIndexRule::Accumulate,
    }
);

typed_family!(
    accumulating,
    Operands::Triple,
    &[Role::Tuple, Role::Index, Role::Tuple],
    accumulating_at_zero,
    [ScatterAdd]
);

// The two halves of the block compression. Each names the representation it
// *produces* rather than the one it reads, which is what `verify_outputs`
// checks the attribute against, so the two dtypes are opposite ways round: the
// row's dtype set describes the operand and the attribute describes the result.
//
// Both are stated outright rather than read off the tuple. `quantize` advertises
// `F32_ONLY` and `dequantize` advertises `Q8_ONLY`, so in each case the tuple
// carries the operand's dtype and the other end of the conversion is fixed by
// the CPU executor, which refuses any compression target but `q8_0` and any
// expansion target that is not a float.
constant_attribute_shim!(
    compressing_to_blocks,
    QuantizationAttributes,
    QuantizationAttributes {
        dtype: DTypeId::Q8_0.descriptor()
    }
);

constant_attribute_shim!(
    expanding_to_floats,
    QuantizationAttributes,
    QuantizationAttributes {
        dtype: DTypeId::F32.descriptor()
    }
);

family!(
    compressing,
    Operands::Unary,
    compressing_to_blocks,
    [Quantize]
);
family!(
    expanding,
    Operands::Unary,
    expanding_to_floats,
    [Dequantize]
);

// `dot` contracts two vectors to a scalar and `outer` expands them to a matrix.
// Both read two operands of one shape, so the plain binary ladder serves.
family!(vector_product, Operands::Binary, plain, [Dot, Outer]);

family!(converting_dtype, Operands::Unary, to_f32, [ToDType]);

family!(unsqueezing, Operands::Unary, on_axis_zero, [UnsqueezeExact]);

// `concat` and `stack` accept one operand upwards; two of the same shape is the
// smallest case that actually joins anything.
family!(
    joining,
    Operands::Binary,
    on_axis_zero,
    [ConcatExact, StackExact]
);

family!(
    reshaping,
    Operands::Unary,
    identity_shape,
    [ReshapeExact, BroadcastAs, BroadcastLeft]
);

family!(narrowing, Operands::UnaryAxis, narrowing_to_one, [Narrow]);

family!(
    flattening,
    Operands::UnaryAxis,
    flatten_fully,
    [FlattenExact]
);

family!(
    slicing,
    Operands::UnaryAxis,
    first_along_each_axis,
    [SliceExact]
);

family!(padding, Operands::UnaryAxis, padding_by_one, [Pad]);

family!(repeating, Operands::UnaryAxis, repeating_twice, [Repeat]);

family!(chunking, Operands::UnaryAxis, one_chunk, [Chunk]);

family!(splitting, Operands::UnaryAxis, splitting_by_one, [Split]);

family!(order_statistic, Operands::UnaryAxis, largest_one, [TopK]);

family!(sorting, Operands::UnaryAxis, ascending, [Argsort]);

family!(grouped_norm, Operands::UnaryAxis, single_group, [GroupNorm]);

// The three losses that read two operands of the same dtype.
// `cross_entropy_loss` is not here: its second operand is an integer class
// index against float logits, which is the per-operand split a shared row
// cannot state and this operand builder cannot yet produce.
family!(
    same_dtype_loss,
    Operands::Binary,
    mean_reduced,
    [MseLoss, L1Loss, BceWithLogitsLoss]
);
