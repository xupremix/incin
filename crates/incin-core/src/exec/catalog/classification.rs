/// Broad classification only. A family is never a capability identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticProfile {
    Creation,
    UnaryFloat,
    BinaryBroadcast,
    Comparison,
    Logical,
    Mutation,
    Shape,
    Selection,
    Indexing,
    MatMul,
    Attention,
    Reduction,
    IndexReduction,
    Scan,
    Normalization,
    Module,
    Composite,
    Loss,
    Quantized,
    Optimizer,
    Transfer,
    Autograd,
}

/// How an operation combines operand extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastingRule {
    None,
    Numpy,
    ExplicitTarget,
    TypedContract,
}

/// Dtype legality and promotion policy recorded by the semantic contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeRule {
    Preserve,
    Floating,
    NumericSame,
    Boolean,
    BooleanResult,
    IndexResult,
    ExplicitOutput,
    Quantized,
    TypedContract,
}

/// Output metadata inference category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputRule {
    Created,
    Preserve,
    Broadcast,
    ShapeAttributes,
    Reduction,
    MatMul,
    Indexing,
    ExplicitDType,
    TypedInference,
    /// The output extent depends on tensor values, not only metadata.
    DataDependent,
    HostValue,
}

impl OutputRule {
    pub const fn is_data_dependent(self) -> bool {
        matches!(self, Self::DataDependent)
    }
}

/// Empty tensor behavior. Empty dimensions are never silently rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmptyRule {
    Allowed,
    IdentityOrDefined,
    RejectedWhenReductionIsEmpty,
    TypedContract,
}

/// Non-finite and arithmetic behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericRule {
    NotApplicable,
    IeeePropagate,
    CheckedInteger,
    StableAccumulation,
    TypedContract,
}

/// Gradient contract for an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradientRule {
    None,
    Defined,
    Piecewise,
    Undefined,
    TypedContract,
}

/// Aliasing/layout contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutRule {
    FreshContiguous,
    ViewWhenPossible,
    PreserveOrMaterialize,
    TypedContract,
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
