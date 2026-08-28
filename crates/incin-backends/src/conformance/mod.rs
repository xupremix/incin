//! Executing what a backend advertises, one tuple at a time, against the CPU.
//!
//! `docs/capabilities.md` opens by saying what it is: "A row here is a
//! canonical capability decision, not a claim about a machine." The compile
//! time obligation in `crates/incin-backends/src/cpu/canonical/mod.rs` closes
//! part of that gap, since a row advertising an operation with no `Execute`
//! implementation does not build. It closes only part. An implementation
//! existing for an operation says nothing about whether it accepts the dtypes,
//! the layouts, or the ranks the same row claims, and those are three of the
//! five columns.
//!
//! This module runs them. It enumerates the product a capability row describes
//! (see [`advertised_tuples`]), builds an operand for each point, executes it,
//! and reports what happened. Nothing here is hand-written per operation except the
//! operand contracts, which cannot be derived: a capability row applies one
//! dtype set to every operand in turn, so a row can never state that operand
//! zero is an integer index and operand one is a float table.
//!
//! # What this checks, and what it does not
//!
//! It checks that an advertised tuple executes, and that an unadvertised dtype
//! is refused. Both directions matter. A backend that quietly executes
//! something it did not advertise has broken the same contract as one that
//! refuses something it did, because in both cases the published table has
//! stopped describing the machine.
//!
//! The second direction is asked of the executor rather than of the
//! dispatcher. `dispatch::execute` queries the capability registry before it
//! calls anything, so a tuple posed through it can only ever demonstrate that
//! the dispatcher works. The CPU executors re-check their own row from inside
//! themselves (`cpu/canonical/common.rs`), and that re-check is what this half
//! of the harness is actually reading. A backend whose executors trusted the
//! dispatcher instead would fail here, which is the point.
//!
//! It does **not** check values, and it does not check gradients. Values are
//! meaningless while the oracle and the subject are the same backend, and they
//! become meaningful the moment a second backend runs through the same driver.
//! Gradients are a separate harness rather than another axis on this loop: the
//! `Training` column is a claim about a derivative, and checking a derivative
//! means finite differences or a recorded reference, not a second call to the
//! same dispatcher.
//!
//! # Coverage is a number, not a wall
//!
//! An operation with no fixture is reported as [`Coverage::Unfixtured`] with
//! the reason it is outstanding, and it is counted. It is deliberately not a failure. A
//! harness that opens with a hundred red rows is a harness that gets marked
//! ignored, which is the same outcome as one that silently passes. The floor
//! lives in `crates/incin-backends/tests/conformance_oracle.rs` and only moves
//! up.

// Private, and re-exported below rather than opened up. Two public paths to
// one type is the kind of surface that gets depended on by accident, and the
// split between enumeration and operands is how this module is organized
// rather than something a caller needs to navigate.
mod fixtures;
mod operands;
mod plan;
mod shaped;

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use incin_core::error::BackendError;
use incin_core::exec::{CanonicalError, ExecutionContext, SupportLevel, TensorHandle};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

pub use fixtures::Coverage;
pub use plan::{AdvertisedTuple, RANK_CAP, advertised_tuples};

use fixtures::{Route, Subject};

/// What running one advertised tuple concluded.
///
/// The three failing variants are separated by who is most likely at fault,
/// because a report that lumps them together costs a reader the triage the
/// harness already did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The tuple ran and produced a result. The advertisement holds.
    Executed,
    /// The backend refused a tuple it advertises. A capability defect: the
    /// table and the kernel disagree, and the table is what users read.
    Refused(String),
    /// The descriptor contract rejected the invocation before any kernel ran.
    /// Either the row claims a rank or dtype the contract forbids, or the
    /// fixture built operands the operation does not take.
    Rejected(String),
    /// The backend accepted the tuple and then failed executing it.
    Failed(String),
    /// The call unwound instead of returning. A defect on every route: the
    /// error contract says a bad invocation comes back as a value, and a
    /// panic past admission is as much a contract break as one before it.
    Panicked(String),
    /// Nothing was concluded, because the harness could not pose the question.
    NotCovered(Coverage),
}

impl Verdict {
    /// Whether this verdict is a finding against the backend.
    #[must_use]
    pub const fn is_finding(&self) -> bool {
        matches!(
            self,
            Self::Refused(_) | Self::Rejected(_) | Self::Failed(_) | Self::Panicked(_)
        )
    }
}

/// One tuple and what it concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// The advertised tuple that was posed.
    pub tuple: AdvertisedTuple,
    /// What posing it concluded.
    pub verdict: Verdict,
}

/// Everything one run concluded, in enumeration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleReport {
    /// The backend the tuples were drawn from and run against.
    pub device: DeviceKind,
    /// Every tuple posed, in the order the registry declares them.
    pub observations: Vec<Observation>,
}

impl OracleReport {
    /// Every observation that is a finding against the backend.
    pub fn findings(&self) -> impl Iterator<Item = &Observation> {
        self.observations
            .iter()
            .filter(|observation| observation.verdict.is_finding())
    }

    /// How many tuples actually executed.
    #[must_use]
    pub fn executed(&self) -> usize {
        self.observations
            .iter()
            .filter(|observation| observation.verdict == Verdict::Executed)
            .count()
    }

    /// The distinct operations that had a fixture and reached execution.
    #[must_use]
    pub fn covered_operations(&self) -> Vec<OperationKind> {
        let mut operations: Vec<OperationKind> = self
            .observations
            .iter()
            .filter(|observation| !matches!(observation.verdict, Verdict::NotCovered(_)))
            .map(|observation| observation.tuple.operation)
            .collect();
        // By the enum's own ordering, not by the display name. Several
        // identities share a name: the coarse family row `conv2d` and the exact
        // `Conv2dExact` both print as `conv2d`, so sorting by the string leaves
        // distinct variants adjacent and `dedup` no way to tell them apart.
        operations.sort_unstable();
        operations.dedup();
        operations
    }

    /// The distinct operations with no fixture, paired with the reason.
    #[must_use]
    pub fn unfixtured_operations(&self) -> Vec<(OperationKind, &'static str)> {
        let mut outstanding: Vec<(OperationKind, &'static str)> = self
            .observations
            .iter()
            .filter_map(|observation| match &observation.verdict {
                Verdict::NotCovered(Coverage::Unfixtured(reason)) => {
                    Some((observation.tuple.operation, *reason))
                }
                _ => None,
            })
            .collect();
        outstanding.sort_unstable_by_key(|(operation, _)| *operation);
        outstanding.dedup_by_key(|(operation, _)| *operation);
        outstanding
    }

    /// A one-line-per-finding rendering, for a failing test's message.
    #[must_use]
    pub fn findings_text(&self) -> String {
        let mut out = alloc::format!("conformance oracle: {:?}\n", self.device);
        for observation in self.findings() {
            let line = match &observation.verdict {
                Verdict::Refused(why) => {
                    alloc::format!("  REFUSED  {}: {why}\n", observation.tuple.label())
                }
                Verdict::Rejected(why) => {
                    alloc::format!("  REJECTED {}: {why}\n", observation.tuple.label())
                }
                Verdict::Failed(why) => {
                    alloc::format!("  FAILED   {}: {why}\n", observation.tuple.label())
                }
                Verdict::Panicked(why) => {
                    alloc::format!("  PANICKED {}: {why}\n", observation.tuple.label())
                }
                _ => String::new(),
            };
            out.push_str(&line);
        }
        out
    }
}

/// Classify a dispatch error by who it implicates.
fn classify(error: &CanonicalError) -> Verdict {
    match error {
        CanonicalError::Backend(BackendError::Unsupported { .. }) => {
            Verdict::Refused(error.to_string())
        }
        CanonicalError::Descriptor(_) => Verdict::Rejected(error.to_string()),
        _ => Verdict::Failed(error.to_string()),
    }
}

/// Whether the registry admits a tuple that was read off one of its own rules.
///
/// A tuple enumerated from a rule that the matcher reading that rule then calls
/// unsupported means the table and the reader of the table disagree, and no
/// amount of executing would find it: a tuple nothing admits never reaches a
/// kernel to fail in.
///
/// Deliberately separate from [`execute_tuple`] rather than a guard inside it.
/// The unadvertised-dtype check needs to reach the executor precisely *because*
/// the registry would turn it away, and a gate the negative check cannot get
/// past would report every unadvertised tuple as the contract holding whether
/// or not the kernel would have run it.
fn admitted(tuple: &AdvertisedTuple) -> Option<Verdict> {
    match crate::capability::support(DeviceKind::Cpu, &tuple.query()) {
        SupportLevel::Unsupported(reason) => Some(Verdict::Refused(alloc::format!(
            "the registry does not admit a tuple drawn from its own rule: {reason:?}"
        ))),
        _ => None,
    }
}

/// Build this tuple's operands and run it along `route`.
fn execute_tuple(
    context: &ExecutionContext<Subject>,
    tuple: &AdvertisedTuple,
    route: Route,
) -> Verdict {
    let fixture = match fixtures::fixture(tuple.operation) {
        Ok(fixture) => fixture,
        Err(reason) => return Verdict::NotCovered(Coverage::Unfixtured(reason)),
    };

    let mut storages = Vec::with_capacity(fixture.operands.arity());
    for index in 0..fixture.operands.arity() {
        let role = fixture
            .roles
            .get(index)
            .copied()
            .unwrap_or(fixtures::Role::Tuple);
        match operands::operand(tuple, fixture.operands, role) {
            Ok(storage) => storages.push(storage),
            Err(reason) => return Verdict::NotCovered(Coverage::Unbuildable(reason)),
        }
    }

    // `f32` in the turbofish whatever the operand's dtype is. `CpuStorage` is
    // one enum carrying its own dtype tag, so `Storage<K>` is the same type for
    // every `K` and the dtype a handle reports comes from the storage's
    // metadata rather than from this parameter.
    let handles: Vec<TensorHandle<'_>> = storages
        .iter()
        .map(TensorHandle::from_storage::<Subject, f32, _>)
        .collect();

    // A panic is a finding, not a reason to stop. The descriptor contract says
    // a rejected invocation comes back as an error, so a path that unwinds
    // instead has broken it, and a harness that dies on the first one reports
    // a single tuple where a reader wanted all of them. This caught
    // `dot`'s validation indexing into a rank-zero shape.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (fixture.run)(context, tuple, route, &handles)
    }));

    match outcome {
        Ok(Ok(())) => Verdict::Executed,
        Ok(Err(error)) => classify(&error),
        Err(_) => Verdict::Panicked(
            "panicked; the descriptor contract requires a returned error".to_string(),
        ),
    }
}

/// Pose one advertised tuple: the registry must admit it and the backend must
/// run it.
fn pose(context: &ExecutionContext<Subject>, tuple: &AdvertisedTuple) -> Verdict {
    admitted(tuple).unwrap_or_else(|| execute_tuple(context, tuple, Route::Dispatched))
}

/// Every dtype the rule for `operation` does not list.
///
/// The negative half of the contract needs tuples the backend did **not**
/// advertise. Drawing them from the complement of the row's own dtype set keeps
/// the choice honest: they are unadvertised by the same table that says what is.
///
/// All of them rather than the first. A single candidate can fail to execute
/// for a reason that has nothing to do with the row, since a `bool` operand
/// handed to an operation whose descriptor requires a float is refused two
/// layers before the kernel, and a check that stopped there would report the
/// contract holding while never having reached the kernel at all.
fn unadvertised_dtypes(operation: OperationKind) -> Vec<DTypeDescriptor> {
    const CANDIDATES: [DTypeId; 5] = [
        DTypeId::Bool,
        DTypeId::U8,
        DTypeId::I64,
        DTypeId::F32,
        DTypeId::F64,
    ];
    let registry = crate::capability::registry(DeviceKind::Cpu);
    let Some(rule) = registry
        .registrations()
        .iter()
        .find(|rule| rule.operation == operation)
    else {
        return Vec::new();
    };

    CANDIDATES
        .iter()
        .map(|id| id.descriptor())
        .filter(|candidate| {
            !rule
                .dtypes
                .iter()
                .any(|dtype| dtype.key() == candidate.key())
        })
        .collect()
}

/// Run every advertised CPU tuple, plus the unadvertised-dtype refusal check.
///
/// Never panics and never stops early. A run is read once, and a reader wants
/// every finding from it rather than the first.
#[must_use]
pub fn run_cpu_self_check() -> OracleReport {
    let context = ExecutionContext::new(Subject::new());
    let tuples = advertised_tuples(DeviceKind::Cpu);
    let mut observations = Vec::with_capacity(tuples.len());

    for tuple in tuples {
        let verdict = pose(&context, &tuple);
        observations.push(Observation { tuple, verdict });
    }

    let mut negatives = Vec::new();
    for operation in covered(&observations) {
        if !fixtures::varies_with_tuple_dtype(operation) {
            continue;
        }
        let executed = *observations
            .iter()
            .find(|observation| observation.tuple.operation == operation)
            .map(|observation| &observation.tuple)
            .expect("the operation was drawn from these observations");

        for dtype in unadvertised_dtypes(operation) {
            let tuple = AdvertisedTuple { dtype, ..executed };

            // Past admission on purpose. `dispatch::execute` would turn this
            // tuple away at the capability query, which proves the dispatcher
            // works and nothing about the kernel. What is under test is
            // whether the executor behind the row would have run something the
            // table never promised, and the only way to ask is to hand it the
            // descriptor directly.
            //
            // The verdict is inverted here: executing is the defect, and a
            // refusal is the contract holding.
            let verdict = match execute_tuple(&context, &tuple, Route::PastAdmission) {
                Verdict::Executed => Verdict::Failed(alloc::format!(
                    "executed {}, which the capability row does not advertise",
                    dtype.name()
                )),
                // A panic is a finding whichever route reached it, so it is
                // not inverted with the rest.
                panicked @ Verdict::Panicked(_) => panicked,
                Verdict::Refused(_) | Verdict::Rejected(_) | Verdict::Failed(_) => {
                    Verdict::Executed
                }
                not_covered => not_covered,
            };
            negatives.push(Observation { tuple, verdict });
        }
    }
    observations.extend(negatives);

    OracleReport {
        device: DeviceKind::Cpu,
        observations,
    }
}

/// The distinct operations that reached execution in `observations`.
fn covered(observations: &[Observation]) -> Vec<OperationKind> {
    let mut operations: Vec<OperationKind> = observations
        .iter()
        .filter(|observation| observation.verdict == Verdict::Executed)
        .map(|observation| observation.tuple.operation)
        .collect();
    operations.sort_unstable();
    operations.dedup();
    operations
}
