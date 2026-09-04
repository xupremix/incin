/// Broad classification only. A family is never a capability identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticProfile {
    /// Allocates or fills storage without reading tensor inputs.
    Creation,
    /// One floating-point input, element-wise compute.
    UnaryFloat,
    /// Two inputs combined under right-aligned broadcasting.
    BinaryBroadcast,
    /// Element-wise predicate producing booleans.
    Comparison,
    /// Boolean combination of mask inputs.
    Logical,
    /// Re-addresses dimensions without touching element values.
    Shape,
    /// Picks a sub-range or sub-set of existing elements.
    Selection,
    /// Gathers or scatters through integer index tensors.
    Indexing,
    /// Contracted multiply over inner axes.
    MatMul,
    /// Scaled dot-product attention family.
    Attention,
    /// Collapses one or more axes into aggregates.
    Reduction,
    /// Reduction producing indices rather than values.
    IndexReduction,
    /// Prefix/cumulative pass along an axis.
    Scan,
    /// Normalizes over a channel or feature set.
    Normalization,
    /// Layer-level operation owning parameters.
    Module,
    /// Built from other catalog operations.
    Composite,
    /// Compares predictions against targets into a scalar.
    Loss,
    /// Operates on block-quantized representations.
    Quantized,
    /// Applies gradient-derived parameter updates.
    Optimizer,
    /// Moves storage between devices or hosts.
    Transfer,
    /// Tape and gradient bookkeeping.
    Autograd,
}

/// How an operation combines operand extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastingRule {
    /// No convention applies.
    None,
    /// Inputs combine under NumPy right-aligned broadcasting.
    Numpy,
    /// One input explicitly names the result target.
    ExplicitTarget,
    /// Fixed by typed inference rather than a listed rule.
    TypedContract,
}

/// Dtype legality and promotion policy recorded by the semantic contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeRule {
    /// Output dtype equals the input dtype.
    Preserve,
    /// Output is floating point regardless of input.
    Floating,
    /// Output matches inputs and must be numeric.
    NumericSame,
    /// Output is always boolean.
    Boolean,
    /// Boolean output derived from a comparison.
    BooleanResult,
    /// Output is an integer index type.
    IndexResult,
    /// Output dtype spelled explicitly in the row.
    ExplicitOutput,
    /// Operates on block-quantized representations.
    Quantized,
    /// Fixed by typed inference rather than a listed rule.
    TypedContract,
}

/// Output metadata inference category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputRule {
    /// Output shape freshly created from attributes.
    Created,
    /// Output dtype equals the input dtype.
    Preserve,
    /// Output follows broadcasting of the inputs.
    Broadcast,
    /// Output shape derives from shape attributes.
    ShapeAttributes,
    /// Collapses one or more axes into aggregates.
    Reduction,
    /// Contracted multiply over inner axes.
    MatMul,
    /// Gathers or scatters through integer index tensors.
    Indexing,
    /// Row carries an explicit output dtype entry.
    ExplicitDType,
    /// Resolved by typed inference.
    TypedInference,
    /// The output extent depends on tensor values, not only metadata.
    DataDependent,
    /// Gradient flows to host-side values only.
    HostValue,
}

impl OutputRule {
    /// Whether this profile reads runtime values static proofs cannot precompute.
    pub const fn is_data_dependent(self) -> bool {
        matches!(self, Self::DataDependent)
    }
}

/// Empty tensor behavior. Empty dimensions are never silently rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmptyRule {
    /// Gradient permitted; identity or well-defined adjoint exists.
    Allowed,
    /// Identity or well-defined adjoint exists.
    IdentityOrDefined,
    /// Refused when the reduction consumes every element.
    RejectedWhenReductionIsEmpty,
    /// Fixed by typed inference rather than a listed rule.
    TypedContract,
}

/// Non-finite and arithmetic behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericRule {
    /// No floating-point compute in this operation.
    NotApplicable,
    /// IEEE propagation semantics (NaN in, NaN out).
    IeeePropagate,
    /// Integer path: checked conversions replace NaN reasoning.
    CheckedInteger,
    /// Accumulation uses a stable order with defined NaN handling.
    StableAccumulation,
    /// Fixed by typed inference rather than a listed rule.
    TypedContract,
}

/// Gradient contract for an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradientRule {
    /// No convention applies.
    None,
    /// Derivative defined everywhere on the supported domain.
    Defined,
    /// Derivative piecewise with stated kinks.
    Piecewise,
    /// Derivative undefined at stated points; kernels may emit NaN.
    Undefined,
    /// Fixed by typed inference rather than a listed rule.
    TypedContract,
}

/// Aliasing/layout contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutRule {
    /// Outputs are freshly allocated contiguous buffers.
    FreshContiguous,
    /// Views returned whenever strides allow it.
    ViewWhenPossible,
    /// Preserve layout when legal, materialize otherwise.
    PreserveOrMaterialize,
    /// Fixed by typed inference rather than a listed rule.
    TypedContract,
    /// Operation requires host-resident storage.
    HostOnly,
}

/// Where an operation's result is produced, and therefore what shape of
/// contract can carry it.
///
/// This says nothing about whether an operation is implemented. It says what
/// kind of implementation is even possible, and it exists because the CPU
/// migration's remainder used to be one number. One number implies every
/// unmigrated operation is the same kind of missing work. It is not: most are a
/// kernel nobody has routed yet, but sixteen of them cannot be an
/// `Execute<O>` implementation as that trait is currently written,
/// so counting them beside a missing kernel describes a task that does not
/// exist and hides one that does.
///
/// [`ExecutionSite::is_backend_executable`] is the predicate that separates the
/// two. Every variant states its own reason rather than deferring to prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExecutionSite {
    /// Operands in, one allocation out. `Execute<O>` expresses this
    /// directly, so an unmigrated operation here is unfinished work rather than
    /// an unfinished contract.
    Kernel,
    /// No operand: the result is built from the descriptor's attributes alone.
    /// Still executable, because `dispatch::execute` accepts an empty operand
    /// slice and `Execute::Output` is an associated type, so the variable forms
    /// may return the backend's trainable-variable handle instead of storage.
    Creation,
    /// The result leaves the device as a host value rather than staying an
    /// allocation. Executable for the same reason: `Output` is an associated
    /// type and does not have to be storage.
    HostReadback,
    /// The frontend composes existing operations or owns the semantic
    /// execution payload. It is not a single backend allocation kernel and
    /// therefore does not belong behind `Execute<O>`.
    Composed,
    /// The operation writes through one of its operands instead of returning a
    /// fresh result. Not executable: `ExecutionRequest::inputs` is a slice of
    /// shared borrows, so an executor cannot reach a mutable operand at all.
    Mutation,
    /// The result is an allocation on a backend other than the one asked to run
    /// the operation. Not executable: the destination backend is reachable
    /// neither from `&self` nor from the descriptor, which names a `DeviceId`
    /// and not a backend value.
    DeviceTransfer,
    /// The operation's effect is on autograd state rather than on an
    /// allocation. Not executable, though for two different reasons:
    /// `require_grad` and `detach` change the tensor wrapper's gradient marker
    /// and make no backend call whatsoever, while `backward` is a real backend
    /// call whose result is the gradient map for a whole tape rather than an
    /// output derived from this descriptor's operands, so validating it against
    /// operand metadata would prove nothing.
    GraphState,
}

impl ExecutionSite {
    /// Whether `Execute<O>` on the executing backend can carry this
    /// operation's result.
    ///
    /// A `false` here is a contract gap, not a backlog item. Closing one means
    /// changing the execution trait, which is why they are counted separately
    /// from operations that merely have not been migrated yet.
    #[must_use]
    pub const fn is_backend_executable(self) -> bool {
        matches!(self, Self::Kernel | Self::Creation | Self::HostReadback)
    }

    /// Short reason a non-executable site cannot be reached, for reports.
    pub const fn blocking_reason(self) -> Option<&'static str> {
        match self {
            Self::Kernel | Self::Creation | Self::HostReadback => None,
            Self::Composed => Some("the frontend composition owns the execution semantics"),
            Self::Mutation => Some("writes through an operand; execution borrows operands shared"),
            Self::DeviceTransfer => {
                Some("produces storage on another backend, which the executor cannot name")
            }
            Self::GraphState => Some("acts on autograd state, not on an allocation"),
        }
    }
}
