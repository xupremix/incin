//! The vocabulary a fixture is written in: what to build, and how to call it.
//!
//! Every type here states something a capability row cannot. A row carries one
//! dtype set applied to every operand in turn, one rank range, and one layout
//! claim, and an operation whose operands genuinely differ has no way to say so
//! in that shape. [`Operands`] and [`Role`] are where the difference is
//! recorded, and [`Route`] is where the two ways of reaching a kernel are named.
//!
//! Kept apart from the tables in [`super::families`] because the tables are a
//! list and this is a definition: a reader adding an operation reads the list,
//! and a reader asking what `Role::Paired` means reads this.

use alloc::string::String;

use incin_core::backend_authoring::{Descriptor, Execute, ExecutionRequest};
use incin_core::exec::{CanonicalError, Capabilities, ExecutionContext, Operation, TensorHandle};

use crate::conformance::plan::AdvertisedTuple;
use crate::cpu::CpuBackendImpl;

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
        self.run_with_payload::<O>(context, attributes, inputs, None)
    }

    /// [`Route::run`], carrying the borrowed bytes a data-creation row needs.
    ///
    /// The payload rides on the execution request rather than on the
    /// attributes, which is why it has to be threaded down both routes instead
    /// of folded into the descriptor like every other fixture input. Posing a
    /// `DataAttributes` invocation with no bytes is rejected twice over: once
    /// by `validate_execution_payload` on the dispatched route, and again by
    /// the CPU executor, which refuses a data creation whose payload is
    /// `None`. Neither refusal is a finding against the backend, so a harness
    /// that could not supply bytes had to report the operation unfixtured
    /// rather than run it.
    pub(crate) fn run_with_payload<O>(
        self,
        context: &ExecutionContext<Subject>,
        attributes: O::Attributes,
        inputs: &[TensorHandle<'_>],
        payload: Option<&[u8]>,
    ) -> Result<(), CanonicalError>
    where
        O: Operation + incin_core::exec::CanonicalOperation,
        O::Attributes: incin_core::exec::AttributeContract,
        Subject: Execute<O> + Capabilities,
    {
        match self {
            Self::Dispatched => incin_core::exec::dispatch::execute_with_payload::<O, Subject>(
                context, attributes, inputs, payload,
            )
            .map(|_| ()),
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
                        payload,
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
