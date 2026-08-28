//! Operands and typed execution shims, keyed by the declaration group.
//!
//! The harness drives runtime data: an [`AdvertisedTuple`] names an operation
//! as an `OperationKind`. Execution is typed: `dispatch::execute` is generic
//! over the `op::X` marker and over that marker's attribute type. Something has
//! to cross between the two, and this module is it.
//!
//! The crossing is keyed on operand contract, not on attribute type and not on
//! semantic profile. Those two look like the obvious keys and are both wrong.
//! Seventy-eight catalog rows declare `NoAttributes`, and that set contains
//! unary floats, binary broadcasts, comparisons that return `bool` whatever
//! they read, logical operations that are `bool` on both sides, and a quantized
//! matmul that reads `Q8_0` blocks. One key cannot serve all of those.
//!
//! The groups of `cpu_descriptor_operations!` are where the operand contracts
//! are already written down, and their comments in `declarations.rs` explain
//! each one, so the families below are read off those groups. They are finer
//! than the groups rather than equal to them: a group exists to share a
//! capability row, and a row says nothing about arity, so a single group can
//! hold a unary and a binary operation that need different operands.
//!
//! An operation with no fixture is not a failure and not a pass. It is
//! [`Coverage::Unfixtured`] with a reason, counted, and held against a floor by
//! `crates/incin-backends/tests/conformance_oracle.rs` so the number can only
//! go up.

use alloc::string::String;

use incin_core::backend_authoring::{Descriptor, Execute, ExecutionRequest, op};
use incin_core::exec::catalog::{
    AddmmAttributes, ArgsortAttributes, AttentionAttributes, AxisAttributes,
    AxisVarianceAttributes, ChunkAttributes, ClampAttributes, DTypeAttributes, DiagonalAttributes,
    DropoutAttributes, DuplicateIndexRule, EpsilonAttributes, FlattenAttributes,
    GroupNormAttributes, IndexReductionAttributes, LerpAttributes, LinearAttributes,
    LossAttributes, LossReduction, NarrowAttributes, NoAttributes, NormAttributes, PadAttributes,
    RepeatAttributes, ScalarAttributes, ScatterAttributes, ShapeAttributes, SliceAttributes,
    SplitAttributes, TopKAttributes, TransposeAttributes, VarianceAttributes,
};
use incin_core::exec::{CanonicalError, Capabilities, ExecutionContext, Operation, TensorHandle};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;

use crate::conformance::operands::materialized_extents;
use crate::conformance::shaped;
use crate::cpu::CpuBackendImpl;

use super::plan::AdvertisedTuple;

/// The subject and the oracle are the same backend for the self-check, so the
/// harness names one type rather than two.
pub(crate) type Subject = CpuBackendImpl;

/// A typed execution shim, erased to a function pointer.
///
/// A plain `fn` rather than a boxed closure: there is nothing to capture, since
/// everything a shim needs arrives as an argument, and the family tables are
/// then simple enough to read as the lists they are.
pub(crate) type Run = fn(
    &ExecutionContext<Subject>,
    &AdvertisedTuple,
    Route,
    &[TensorHandle<'_>],
) -> Result<(), CanonicalError>;

/// Which of the two paths into a backend to take.
///
/// `dispatch::execute` validates the descriptor, then asks the capability
/// registry, then calls the executor. That middle step is what makes the
/// positive direction meaningful and the negative direction impossible: a
/// tuple the table does not advertise never reaches the kernel, so going
/// through the dispatcher could only ever prove that the dispatcher works.
///
/// [`Route::PastAdmission`] builds the same validated descriptor and calls the
/// executor with it directly. It is the only way to ask a kernel what it would
/// do with a tuple its own table never promised, which is one half of what an
/// advertisement means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Route {
    /// Through `dispatch::execute`, capability admission included.
    Dispatched,
    /// Straight to the executor, with the capability query skipped.
    PastAdmission,
}

impl Operands {
    /// How many operands to build.
    pub(crate) const fn arity(self) -> usize {
        match self {
            Self::Nullary => 0,
            Self::Unary | Self::UnaryAxis | Self::UnaryScalar => 1,
            Self::Binary => 2,
            Self::Triple => 3,
        }
    }
}

impl Route {
    pub(crate) fn run<O>(
        self,
        context: &ExecutionContext<Subject>,
        attributes: O::Attributes,
        inputs: &[TensorHandle<'_>],
    ) -> Result<(), CanonicalError>
    where
        O: Operation + incin_core::exec::CanonicalOperation,
        O::Attributes: incin_core::exec::AttributeContract,
        Subject: Execute<O> + Capabilities,
    {
        match self {
            Self::Dispatched => {
                incin_core::exec::dispatch::execute::<O, Subject>(context, attributes, inputs)
                    .map(|_| ())
            }
            Self::PastAdmission => {
                let logical = inputs
                    .iter()
                    .map(|handle| incin_core::exec::dispatch::logical_meta(handle.metadata()))
                    .collect();
                let validated = Descriptor::<O>::infer_runtime(attributes, logical)
                    .map_err(CanonicalError::Descriptor)?;
                context
                    .backend()
                    .execute(ExecutionRequest {
                        operation: &validated,
                        inputs,
                        context,
                        payload: None,
                    })
                    .map(|_| ())
                    .map_err(CanonicalError::Backend)
            }
        }
    }
}

/// How many operands to build, and what each one holds.
///
/// Every variant here states an operand contract that a capability row cannot.
/// A row carries one dtype set applied to every operand in turn, which is why
/// `declarations.rs` documents `INDEX_AND_F32_DTYPES` and `F32_AND_BOOL` as
/// unions rather than as per-operand claims. The fixture is the only place that
/// knows the split, so the split lives here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operands {
    /// No operand at all.
    ///
    /// The creation rows take their shape and dtype from their attributes
    /// rather than from an input, so the capability row is queried against the
    /// inferred output. Nothing has to be built, and the tuple still reaches
    /// the invocation because every creation attribute carries a dtype.
    Nullary,
    /// One operand carrying the tuple's dtype.
    Unary,
    /// Two operands of identical shape and dtype.
    Binary,
    /// Three operands, each read through its own role.
    ///
    /// Arity only. Shape authority belongs to [`Role`]: `where_cond` reads
    /// three operands of one shape, while `batch_norm` reads an activation
    /// against two per-channel vectors, and both arrive here.
    Triple,
    /// One operand, for an operation that names an axis.
    ///
    /// Separate from [`Operands::Unary`] only so that rank zero can be turned
    /// away with a reason. A scalar has no axis, and an attribute type with a
    /// plain `usize` axis field has no way to say so, unlike
    /// `IndexReductionAttributes` whose `Option` carries the flattened form.
    UnaryAxis,
    /// One operand holding exactly one element, at the tuple's rank.
    ///
    /// The scalar readbacks are advertised across the whole rank range because
    /// a one-element tensor exists at every rank. Their contract is about the
    /// element count, not the rank, which is the one thing a capability row has
    /// no column for.
    UnaryScalar,
}

/// What one operand carries, when the row cannot say.
///
/// A capability row applies one dtype set to every operand in turn, so a row
/// whose operands genuinely differ has to state the *union* of what they carry.
/// `declarations.rs` says so twice at length, for `INDEX_AND_F32_DTYPES` and
/// for `F32_AND_BOOL`. The union is the loosest honest claim the row can make
/// and it is not a claim that either operand may be either dtype, so walking it
/// and handing every operand the same dtype poses invocations the operation was
/// never meant to accept.
///
/// The split lives here because the fixture is the only place that knows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// Carries the tuple's dtype, at the tuple's rank and layout.
    Tuple,
    /// A boolean mask at the tuple's shape.
    Mask,
    /// A float payload, whatever the row's union says.
    Float,
    /// A float matrix, for an operation whose table operand is rank two.
    FloatMatrix,
    /// Integer indices at the tuple's shape, all zero so every one is in range.
    Index,
    /// Integer indices as a vector, all zero for the same reason.
    IndexVector,
    /// The tuple's batch extents followed by two named ones.
    ///
    /// The one operand shape a single ladder cannot produce: a matrix product
    /// reads `[..batch, m, k]` against `[..batch, k, n]`, and the two operands
    /// agree on `k` while differing on everything else. Naming both trailing
    /// extents per operand states that agreement, and it states the two harder
    /// ones above it as well: `addmm` adds a `[..batch, m, n]` addend to the
    /// same product, and attention reads a query, a key and a value that agree
    /// pairwise on two different extents.
    ///
    /// The strided form is built by transposing the *last* two axes rather
    /// than the first, or the batch extents of one operand stop agreeing with
    /// the batch extents of the next.
    Paired {
        /// Second-to-last extent.
        rows: usize,
        /// Last extent.
        columns: usize,
    },
    /// A convolution filter bank: `[out, in / groups, ..unit spatial]`.
    ///
    /// `spatial` is how many trailing kernel axes the operation has. It fixes
    /// the weight's rank, which `inference.rs` pins exactly, and it also fixes
    /// which input axis is the channel, read there at `input[len - 1 -
    /// spatial]`. One number decides both because they are the same fact.
    ConvWeight {
        /// Trailing kernel axes: one for `conv1d`, two for `conv2d`.
        spatial: usize,
    },
    /// A transposed convolution filter bank: `[in, out / groups, ..unit
    /// spatial]`.
    ///
    /// The two channel extents trade places against [`Role::ConvWeight`],
    /// which is the whole difference between the two roles and the reason a
    /// single one with a flag would read worse than two.
    ConvTransposeWeight {
        /// Trailing kernel axes, as in [`Role::ConvWeight`].
        spatial: usize,
    },
    /// One value per output channel or output feature.
    ///
    /// A convolution bias is read against `weight[0]` forward and
    /// `weight[1] * groups` transposed, and a linear bias against `weight[0]`.
    /// All three are the same extent while groups stay at one, which every
    /// fixture here keeps them at.
    OutputVector,
    /// One value per channel, as batch norm's affine and running state carry.
    ///
    /// Axis one absolutely, not counted back from the end.
    /// `BatchNormAttributes::validate` reads `input[1]`, which agrees with a
    /// convolution's channel axis at rank four and disagrees below it.
    ChannelVector,
    /// The input's final extent, as an RMS norm weight or as a layer norm
    /// parameter over a one-axis normalized shape.
    TrailingVector,
    /// A `[out, in]` projection, where `in` is the input's final extent.
    LinearWeight,
}

impl Role {
    /// Whether this operand carries the tuple's shape, and so its layout.
    ///
    /// A role that fixes its own shape is not the operand the layout claim is
    /// about. The tuple's layout describes the operand carrying its dtype, and
    /// transposing a per-channel vector or a unit kernel says nothing about
    /// whether the backend handles a strided activation.
    pub(crate) const fn follows_tuple_shape(self) -> bool {
        matches!(self, Self::Tuple | Self::Mask | Self::Float | Self::Index)
    }

    /// Whether this operand carries the tuple's dtype.
    ///
    /// False only for the roles that pin a dtype of their own. A fixture with
    /// no such operand never lets the tuple's dtype reach the invocation, which
    /// is what [`varies_with_tuple_dtype`] needs to know: posing an
    /// unadvertised dtype at a fixture like that changes nothing about the call
    /// and would report the row executing something it never advertised.
    pub(crate) const fn carries_tuple_dtype(self) -> bool {
        !matches!(
            self,
            Self::Mask | Self::Float | Self::FloatMatrix | Self::Index | Self::IndexVector
        )
    }
}

/// One operation's fixture: what to feed it and how to call it.
#[derive(Clone, Copy)]
pub(crate) struct Fixture {
    pub(crate) operands: Operands,
    /// Per-operand roles, or empty when every operand carries the tuple's dtype.
    pub(crate) roles: &'static [Role],
    pub(crate) run: Run,
}

/// Why a tuple could not be executed, when the reason is the harness rather
/// than the backend.
///
/// Kept distinct from a failure throughout. A harness that cannot build an
/// operand and reports that as a backend defect is worse than no harness,
/// because it spends a reader's attention on its own gaps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coverage {
    /// No fixture yet for this operation, with the reason it is outstanding.
    Unfixtured(&'static str),
    /// A fixture exists but this particular tuple cannot be materialized.
    Unbuildable(String),
}

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

// Visible to `shaped`, which holds the families whose attributes name extents
// and so has to build shims of its own. `macro_rules!` is textual and scoped to
// the rest of the file without this.
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
        Cumsum,
        Softmax,
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
    [TransposeExact]
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

/// The fixture for `operation`, or the reason there is not one yet.
///
/// Order is arbitrary because the family lists are disjoint. Nothing enforces
/// that beyond their being written that way, which is worth knowing: naming an
/// operation twice would silently give whichever family is consulted first.
pub(crate) fn fixture(operation: OperationKind) -> Result<Fixture, &'static str> {
    unary_float(operation)
        .or_else(|| binary_elementwise(operation))
        .or_else(|| unary_logical(operation))
        .or_else(|| scalar_elementwise(operation))
        .or_else(|| reduce_all(operation))
        .or_else(|| reduce_axis(operation))
        .or_else(|| index_reduce_axis(operation))
        .or_else(|| readback(operation))
        .or_else(|| readback_scalar(operation))
        .or_else(|| clamping(operation))
        .or_else(|| interpolating(operation))
        .or_else(|| transposing(operation))
        .or_else(|| diagonal_shape(operation))
        .or_else(|| norm_reduce(operation))
        .or_else(|| variance_all(operation))
        .or_else(|| variance_axis(operation))
        .or_else(|| epsilon_unary(operation))
        .or_else(|| dropping(operation))
        .or_else(|| same_dtype_loss(operation))
        .or_else(|| selecting(operation))
        .or_else(|| masking(operation))
        .or_else(|| gathering(operation))
        .or_else(|| selecting_rows(operation))
        .or_else(|| embedding_lookup(operation))
        .or_else(|| matrix_product(operation))
        .or_else(|| vector_product(operation))
        .or_else(|| converting_dtype(operation))
        .or_else(|| unsqueezing(operation))
        .or_else(|| joining(operation))
        .or_else(|| reshaping(operation))
        .or_else(|| narrowing(operation))
        .or_else(|| flattening(operation))
        .or_else(|| slicing(operation))
        .or_else(|| padding(operation))
        .or_else(|| repeating(operation))
        .or_else(|| chunking(operation))
        .or_else(|| splitting(operation))
        .or_else(|| order_statistic(operation))
        .or_else(|| sorting(operation))
        .or_else(|| grouped_norm(operation))
        .or_else(|| shaped::creating(operation))
        .or_else(|| shaped::creating_full(operation))
        .or_else(|| shaped::creating_arange(operation))
        .or_else(|| shaped::creating_linspace(operation))
        .or_else(|| shaped::squeezing(operation))
        .or_else(|| shaped::pooling_max(operation))
        .or_else(|| shaped::pooling_average(operation))
        .or_else(|| shaped::pooling_adaptive(operation))
        .or_else(|| shaped::sliding(operation))
        .or_else(|| shaped::shuffling(operation))
        .or_else(|| shaped::convolving_1d(operation))
        .or_else(|| shaped::convolving_2d(operation))
        .or_else(|| shaped::convolving_transposed(operation))
        .or_else(|| shaped::normalizing_layer(operation))
        .or_else(|| shaped::normalizing_rms(operation))
        .or_else(|| shaped::normalizing_batch(operation))
        .or_else(|| fused_product_sum(operation))
        .or_else(|| attending(operation))
        .or_else(|| projecting(operation))
        .or_else(|| class_loss(operation))
        .or_else(|| scattering(operation))
        .ok_or_else(|| unfixtured_reason(operation))
}

/// Whether the tuple's dtype reaches any operand of `operation`'s fixture.
///
/// False for a fixture whose every operand has a fixed role, as `embedding`'s
/// index vector and float table both do. The row's dtype set for such an
/// operation is a union describing operands the fixture pins, so posing a
/// different dtype changes nothing about the invocation and the
/// unadvertised-dtype probe would report the row executing something it never
/// advertised when in fact the dtype was never used.
pub(crate) fn varies_with_tuple_dtype(operation: OperationKind) -> bool {
    fixture(operation).is_ok_and(|fixture| {
        fixture.roles.is_empty() || fixture.roles.iter().any(|role| role.carries_tuple_dtype())
    })
}

/// Why an operation has no fixture, in the terms a contributor closing the gap
/// would need.
fn unfixtured_reason(operation: OperationKind) -> &'static str {
    use OperationKind::*;
    match operation {
        Quantize | Dequantize | QuantizedMatMul => {
            "block-encoded storage: the logical extent and the buffer length              differ by a block size this harness does not know, and inventing              one here would hard-code a backend detail into the enumeration"
        }
        TensorFromData | TensorFromBytes => {
            "the payload rides on the execution request rather than the              attributes, and this harness poses every tuple with no payload              at all; describing the byte length is not enough, the bytes have              to be supplied"
        }
        coarse if !coarse.is_exact() => {
            "a coarse family row rather than an exact identity: it has no              descriptor and nothing to execute, and the exact rows beneath it              carry the coverage"
        }
        // Every exact identity the CPU registry advertises today reaches one
        // of the families above it, so this arm is unreachable in practice. It
        // stays because removing a fixture must produce a counted gap with a
        // reason rather than an empty string, which
        // `every_uncovered_operation_carries_a_reason` asserts.
        _ => "no fixture yet",
    }
}
