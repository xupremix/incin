//! The conformance suite a backend author runs against their own backend
//! (`EXE-010`).
//!
//! PROPOSALS.md sec. 2.9 lists what the backend-authoring surface contains and
//! closes with the property this module exists to enforce: "An external backend
//! implements only the operation descriptors it supports. Missing support is
//! visible through the capability registry rather than hundreds of default
//! trait methods."
//!
//! That sentence has a direct consequence for how the suite is built. A check
//! for an operation a backend never claimed must **skip**, not fail — a suite
//! that fails a backend for not implementing something it was never asked to
//! implement is a suite nobody runs. So every check asks
//! [`incin_core::exec::Capabilities::support`] first and reports what it found.
//!
//! The three things only the author can supply are on
//! [`Subject`](crate::external::conformance::Subject): their
//! backend, storage built from values, and values read back out. Everything
//! else — descriptors, expected results, tolerances, the verdict — belongs to
//! the harness, because those are the parts that must be the same for every
//! backend or the word "conformance" means nothing.
//!
//! The suite reports rather than aborts. A backend that panics where it should
//! have returned an error is a *finding*, and a harness that dies on it tells
//! the author less than one that names the check and keeps going.
//!
//! `crates/incin-backends/tests/conformance.rs` carries a complete minimal
//! backend that passes this suite. That is the template sec. 2.9 asks for:
//! prose describing the seven bullets would go stale, and a backend that
//! compiles and passes cannot.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::{format, vec};

use incin_core::backend_authoring::operations::{NoAttributes, ShapeAttributes};
use incin_core::backend_authoring::{Descriptor, Execute, ExecutionRequest, op};
use incin_core::exec::{
    Capabilities, CapabilityQuery, ExecutionContext, ExecutionDescriptor, LayoutClass,
    LogicalTensorMeta, MathMode, SupportLevel, TensorHandle, Validated,
};
use incin_core::prelude::{DTypeId, Local, OperationKind, ShapeBuf, StorageBackend};

// ============================================================================
// Numerical tolerance profiles
// ============================================================================

/// How far a backend's arithmetic may land from the reference.
///
/// Both bounds are needed and neither is sufficient. An absolute bound alone
/// rejects large values that are correct to every bit a float has; a relative
/// bound alone rejects values near zero, where the reference has no magnitude
/// to be relative to. A value passes if it is within *either*.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    pub absolute: f64,
    pub relative: f64,
}

impl Tolerance {
    /// Bit-exact. The right profile for anything that only moves bytes —
    /// reshape, transpose, a copy — where a difference of one ulp means the
    /// backend did arithmetic it was not asked to do.
    pub const EXACT: Self = Self {
        absolute: 0.0,
        relative: 0.0,
    };

    /// `f32` values summed over a short axis, as a small matmul or reduction
    /// does. Loose enough for a different accumulation order, tight enough that
    /// a wrong operand order fails.
    pub const F32_ACCUMULATED: Self = Self {
        absolute: 1e-5,
        relative: 1e-5,
    };

    /// `f32` values summed over a long axis, or through a fused kernel.
    pub const F32_WIDE: Self = Self {
        absolute: 1e-3,
        relative: 1e-4,
    };

    /// Whether `actual` is an acceptable answer for `expected`.
    #[must_use]
    pub fn accepts(self, expected: f64, actual: f64) -> bool {
        if expected == actual {
            return true;
        }
        if !expected.is_finite() || !actual.is_finite() {
            // A backend is allowed to produce a non-finite value only where the
            // reference did; comparing magnitudes of infinities says nothing.
            return false;
        }
        let difference = (expected - actual).abs();
        difference <= self.absolute || difference <= self.relative * expected.abs()
    }
}

// ============================================================================
// The subject
// ============================================================================

/// The backend under test, and the three things only its author can supply.
///
/// Deliberately small. Every method here is something the harness cannot
/// possibly know — how to build this backend's storage, how to read it back,
/// and what accuracy its author is claiming. Anything the harness *can* know it
/// does not ask for, because a conformance suite whose expectations come from
/// the subject is not testing the subject.
pub trait Subject {
    /// This backend's `f32` storage type.
    type Storage: core::any::Any;

    /// The backend itself.
    type Backend: StorageBackend<Storage<f32> = Self::Storage>
        + Capabilities
        + Execute<Descriptor<op::MatMulExact>, Output = Self::Storage>
        + Execute<Descriptor<op::ReshapeExact>, Output = Self::Storage>;

    /// A name for the report. Usually the backend's type name.
    fn name(&self) -> String;

    /// A fresh backend instance.
    fn backend(&self) -> Self::Backend;

    /// Storage holding `values` laid out as `dims`, row-major.
    ///
    /// Returns the author's own error text on failure; the harness reports it
    /// rather than interpreting it.
    fn storage(&self, dims: &[usize], values: &[f32]) -> Result<Self::Storage, String>;

    /// The values in `storage`, row-major.
    fn values(&self, storage: &Self::Storage) -> Result<Vec<f32>, String>;

    /// The accuracy this backend claims for an operation.
    ///
    /// Defaults are the ones most backends want: exact for operations that only
    /// re-address bytes, and a short-accumulation profile for the rest.
    fn tolerance(&self, operation: OperationKind) -> Tolerance {
        match operation {
            OperationKind::ReshapeExact => Tolerance::EXACT,
            _ => Tolerance::F32_ACCUMULATED,
        }
    }
}

// ============================================================================
// The report
// ============================================================================

/// What one check concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Passed,
    /// The backend does not claim this operation, so there was nothing to
    /// check. Not a failure: sec. 2.9 says a backend implements only the
    /// descriptors it supports.
    Skipped(String),
    Failed(String),
}

/// One named check and its outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: &'static str,
    pub outcome: Outcome,
}

impl Check {
    fn new(name: &'static str, outcome: Outcome) -> Self {
        Self { name, outcome }
    }
}

/// The result of running the suite against one backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub backend: String,
    pub checks: Vec<Check>,
}

impl Report {
    /// Whether nothing failed. A skipped check does not fail a report.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.failures().next().is_none()
    }

    /// Every check that failed, in the order they ran.
    pub fn failures(&self) -> impl Iterator<Item = &Check> {
        self.checks
            .iter()
            .filter(|check| matches!(check.outcome, Outcome::Failed(_)))
    }

    /// Every check that was skipped, and why.
    pub fn skipped(&self) -> impl Iterator<Item = &Check> {
        self.checks
            .iter()
            .filter(|check| matches!(check.outcome, Outcome::Skipped(_)))
    }

    /// A one-line-per-check rendering, for a failing test's message.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("conformance: {}\n", self.backend);
        for check in &self.checks {
            let line = match &check.outcome {
                Outcome::Passed => format!("  pass  {}\n", check.name),
                Outcome::Skipped(why) => format!("  skip  {} ({why})\n", check.name),
                Outcome::Failed(why) => format!("  FAIL  {}: {why}\n", check.name),
            };
            out.push_str(&line);
        }
        out
    }
}

// ============================================================================
// The suite
// ============================================================================

fn query(operation: OperationKind, rank: usize) -> CapabilityQuery {
    CapabilityQuery {
        operation: incin_core::exec::OperationIdentity::Builtin(operation),
        dtype: DTypeId::F32.descriptor(),
        layout: LayoutClass::Contiguous,
        rank,
        training: false,
        math_mode: MathMode::Precise,
    }
}

fn input_meta(dims: &[usize]) -> LogicalTensorMeta {
    LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(dims)),
        dtype: Some(DTypeId::F32.descriptor()),
        device: None,
    }
}

fn matmul_spec(
    lhs: &[usize],
    rhs: &[usize],
) -> Result<Validated<Descriptor<op::MatMulExact>>, String> {
    Descriptor::<op::MatMulExact>::infer_runtime(
        NoAttributes,
        vec![input_meta(lhs), input_meta(rhs)],
    )
    .map_err(|error| format!("the harness could not validate matmul: {error}"))
}

/// The suite's reshape fixture, `[2, 3]` reinterpreted as `[3, 2]`.
///
/// Statically shaped, unlike the matmul fixture, and not by preference:
/// `ReshapeShape` carries an `ElementCount` proof that `Dyn` cannot supply, so
/// a dynamically shaped reshape has nothing to lower. That is the rule working
/// — element count is what reshape has to preserve — and it means the harness
/// fixes this shape pair rather than taking it as arguments.
fn reshape_spec() -> Result<Validated<Descriptor<op::ReshapeExact>>, String> {
    Descriptor::<op::ReshapeExact>::infer_runtime(
        ShapeAttributes { shape: vec![3, 2] },
        vec![input_meta(&[2, 3])],
    )
    .map_err(|error| format!("the harness could not validate reshape: {error}"))
}

/// Run a check, turning a panic into a failure.
///
/// A backend that panics where the contract says to return an error is exactly
/// what this suite exists to catch, and a harness that dies on it reports one
/// check instead of all of them.
fn guarded(name: &'static str, body: impl FnOnce() -> Result<(), String>) -> Check {
    let outcome = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) => Outcome::Passed,
        Ok(Err(why)) => Outcome::Failed(why),
        Err(_) => Outcome::Failed(
            "panicked; the descriptor contract requires a returned BackendError".to_string(),
        ),
    };
    Check::new(name, outcome)
}

/// Run the whole suite against `subject`.
///
/// Never panics and never stops early: an author wants every finding from one
/// run, not the first one.
#[must_use]
pub fn run<S: Subject>(subject: &S) -> Report {
    let mut checks = vec![
        guarded("storage_round_trips_its_values", || round_trip(subject)),
        guarded("metadata_describes_the_tensor_it_came_from", || {
            metadata_agrees(subject)
        }),
    ];
    checks.extend(matmul_checks(subject));
    checks.extend(reshape_checks(subject));
    checks.push(guarded("the_registry_and_the_executor_agree", || {
        registry_agrees(subject)
    }));

    Report {
        backend: subject.name(),
        checks,
    }
}

fn round_trip<S: Subject>(subject: &S) -> Result<(), String> {
    let values = [1.0_f32, -2.5, 0.0, 4.25, 7.5, -0.125];
    let storage = subject.storage(&[2, 3], &values)?;
    let read = subject.values(&storage)?;
    if read.len() != values.len() {
        return Err(format!(
            "wrote {} values and read back {}",
            values.len(),
            read.len()
        ));
    }
    for (index, (wrote, read)) in values.iter().zip(&read).enumerate() {
        if wrote.to_bits() != read.to_bits() {
            return Err(format!(
                "value {index} changed on a round trip: wrote {wrote}, read {read}"
            ));
        }
    }
    Ok(())
}

fn metadata_agrees<S: Subject>(subject: &S) -> Result<(), String> {
    let storage = subject.storage(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let meta = <S::Backend as StorageBackend>::metadata::<f32>(&storage);

    if meta.shape().dims() != [2, 3] {
        return Err(format!(
            "storage built as [2, 3] reports shape {:?}",
            meta.shape().dims()
        ));
    }
    if meta.dtype() != DTypeId::F32.descriptor() {
        return Err(format!(
            "f32 storage reports dtype {:?}; the descriptor path keys on this",
            meta.dtype()
        ));
    }
    let numel = meta
        .shape()
        .checked_numel(OperationKind::Storage)
        .map_err(|error| format!("the reported shape has no element count: {error}"))?;
    if numel != 6 {
        return Err(format!(
            "shape {:?} has {numel} elements, not 6",
            meta.shape().dims()
        ));
    }
    if meta.strides().strides().len() != meta.shape().dims().len() {
        return Err(format!(
            "{} strides for a rank-{} shape",
            meta.strides().strides().len(),
            meta.shape().dims().len()
        ));
    }
    Ok(())
}

fn execute_matmul<S: Subject>(
    subject: &S,
    spec: &Validated<Descriptor<op::MatMulExact>>,
    lhs: &S::Storage,
    rhs: &S::Storage,
) -> Result<S::Storage, String> {
    let context = ExecutionContext::new(subject.backend());
    let inputs = [
        TensorHandle::from_storage::<S::Backend, f32, Local>(lhs),
        TensorHandle::from_storage::<S::Backend, f32, Local>(rhs),
    ];
    context
        .backend()
        .execute(ExecutionRequest {
            operation: spec,
            inputs: &inputs,
            context: &context,
        })
        .map_err(|error| format!("{error}"))
}

fn matmul_checks<S: Subject>(subject: &S) -> Vec<Check> {
    let support = subject
        .backend()
        .support(&query(OperationKind::MatMulExact, 2));
    if !support.is_supported() {
        let why = match support {
            SupportLevel::Unsupported(reason) => format!("{reason}"),
            _ => unreachable!("is_supported() is false only for Unsupported"),
        };
        return vec![
            Check::new("matmul_is_within_tolerance", Outcome::Skipped(why.clone())),
            Check::new(
                "matmul_output_matches_its_descriptor",
                Outcome::Skipped(why.clone()),
            ),
            Check::new("matmul_rejects_a_wrong_input_count", Outcome::Skipped(why)),
        ];
    }

    // [2, 3] x [3, 2] -> [2, 2], computed here so the expectation does not come
    // from the backend being tested.
    let lhs_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_values = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = [58.0_f64, 64.0, 139.0, 154.0];

    vec![
        guarded("matmul_is_within_tolerance", || {
            let spec = matmul_spec(&[2, 3], &[3, 2])?;
            let lhs = subject.storage(&[2, 3], &lhs_values)?;
            let rhs = subject.storage(&[3, 2], &rhs_values)?;
            let output = execute_matmul(subject, &spec, &lhs, &rhs)?;
            let actual = subject.values(&output)?;
            if actual.len() != expected.len() {
                return Err(format!(
                    "[2,3] x [3,2] produced {} values, not {}",
                    actual.len(),
                    expected.len()
                ));
            }
            let tolerance = subject.tolerance(OperationKind::MatMulExact);
            for (index, (want, got)) in expected.iter().zip(&actual).enumerate() {
                if !tolerance.accepts(*want, f64::from(*got)) {
                    return Err(format!(
                        "element {index} is {got}, outside {tolerance:?} of {want}"
                    ));
                }
            }
            Ok(())
        }),
        guarded("matmul_output_matches_its_descriptor", || {
            let spec = matmul_spec(&[2, 3], &[3, 2])?;
            let lhs = subject.storage(&[2, 3], &lhs_values)?;
            let rhs = subject.storage(&[3, 2], &rhs_values)?;
            let output = execute_matmul(subject, &spec, &lhs, &rhs)?;
            let meta = <S::Backend as StorageBackend>::metadata::<f32>(&output);
            let declared = spec
                .descriptor()
                .output_shape()
                .ok_or_else(|| "descriptor has no output shape".to_string())?
                .dims();
            if meta.shape().dims() != declared {
                return Err(format!(
                    "the validated descriptor says {declared:?} and the output reports {:?}; \
                     a proof the executor does not honour is worse than no proof",
                    meta.shape().dims()
                ));
            }
            Ok(())
        }),
        guarded("matmul_rejects_a_wrong_input_count", || {
            let spec = matmul_spec(&[2, 3], &[3, 2])?;
            let lhs = subject.storage(&[2, 3], &lhs_values)?;
            let context = ExecutionContext::new(subject.backend());
            let inputs = [TensorHandle::from_storage::<S::Backend, f32, Local>(&lhs)];
            let result = context.backend().execute(ExecutionRequest {
                operation: &spec,
                inputs: &inputs,
                context: &context,
            });
            if result.is_ok() {
                return Err(
                    "one operand was accepted for a two-operand matmul; input arity is \
                     the executor's to check, since Validated proves shapes and not handles"
                        .to_string(),
                );
            }
            Ok(())
        }),
    ]
}

fn reshape_checks<S: Subject>(subject: &S) -> Vec<Check> {
    let support = subject
        .backend()
        .support(&query(OperationKind::ReshapeExact, 2));
    if !support.is_supported() {
        let why = match support {
            SupportLevel::Unsupported(reason) => format!("{reason}"),
            _ => unreachable!("is_supported() is false only for Unsupported"),
        };
        return vec![
            Check::new(
                "reshape_preserves_element_order",
                Outcome::Skipped(why.clone()),
            ),
            Check::new(
                "reshape_rejects_a_mismatched_operand",
                Outcome::Skipped(why),
            ),
        ];
    }

    let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let execute = |spec: &Validated<Descriptor<op::ReshapeExact>>,
                   input: &S::Storage|
     -> Result<S::Storage, String> {
        let context = ExecutionContext::new(subject.backend());
        let inputs = [TensorHandle::from_storage::<S::Backend, f32, Local>(input)];
        context
            .backend()
            .execute(ExecutionRequest {
                operation: spec,
                inputs: &inputs,
                context: &context,
            })
            .map_err(|error| format!("{error}"))
    };

    vec![
        guarded("reshape_preserves_element_order", || {
            let spec = reshape_spec()?;
            let input = subject.storage(&[2, 3], &values)?;
            let output = execute(&spec, &input)?;
            let actual = subject.values(&output)?;
            let tolerance = subject.tolerance(OperationKind::ReshapeExact);
            if actual.len() != values.len() {
                return Err(format!(
                    "reshape changed the element count from {} to {}",
                    values.len(),
                    actual.len()
                ));
            }
            for (index, (want, got)) in values.iter().zip(&actual).enumerate() {
                if !tolerance.accepts(f64::from(*want), f64::from(*got)) {
                    return Err(format!(
                        "element {index} is {got} after a reshape of {want}; reshape \
                         re-addresses bytes and must not compute"
                    ));
                }
            }
            Ok(())
        }),
        guarded("reshape_rejects_a_mismatched_operand", || {
            // The descriptor was proved for a [2, 3] input. Handing it a [3, 2]
            // one is the case `Validated` cannot catch, because the proof is
            // about shapes and the handle arrives separately.
            let spec = reshape_spec()?;
            let wrong = subject.storage(&[3, 2], &values)?;
            if execute(&spec, &wrong).is_ok() {
                return Err(
                    "an operand whose shape disagrees with the validated descriptor was \
                     accepted; the executor is the only place that can notice"
                        .to_string(),
                );
            }
            Ok(())
        }),
    ]
}

/// What the registry claims and what the executor does must be the same claim.
///
/// This is sec. 2.9's "missing support is visible through the capability
/// registry" as a test. A backend that executes an operation its registry calls
/// unsupported has a registry nobody can plan against, and one that refuses an
/// operation its registry calls native has one nobody can trust.
fn registry_agrees<S: Subject>(subject: &S) -> Result<(), String> {
    let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let backend = subject.backend();

    let matmul_claimed = backend
        .support(&query(OperationKind::MatMulExact, 2))
        .is_supported();
    let spec = matmul_spec(&[2, 3], &[3, 2])?;
    let lhs = subject.storage(&[2, 3], &values)?;
    let rhs = subject.storage(&[3, 2], &values)?;
    let matmul_ran = execute_matmul(subject, &spec, &lhs, &rhs).is_ok();
    if matmul_claimed != matmul_ran {
        return Err(format!(
            "the registry says matmul is {} and executing it {}",
            if matmul_claimed {
                "supported"
            } else {
                "unsupported"
            },
            if matmul_ran { "succeeded" } else { "failed" }
        ));
    }

    let reshape_claimed = backend
        .support(&query(OperationKind::ReshapeExact, 2))
        .is_supported();
    let spec = reshape_spec()?;
    let input = subject.storage(&[2, 3], &values)?;
    let context = ExecutionContext::new(subject.backend());
    let inputs = [TensorHandle::from_storage::<S::Backend, f32, Local>(&input)];
    let reshape_ran = context
        .backend()
        .execute(ExecutionRequest {
            operation: &spec,
            inputs: &inputs,
            context: &context,
        })
        .is_ok();
    if reshape_claimed != reshape_ran {
        return Err(format!(
            "the registry says reshape is {} and executing it {}",
            if reshape_claimed {
                "supported"
            } else {
                "unsupported"
            },
            if reshape_ran { "succeeded" } else { "failed" }
        ));
    }

    // A dtype the adapter cannot represent must be refused by the registry
    // rather than discovered inside a kernel. Every backend has at least one.
    let unrepresentable = [
        DTypeId::Q8_0.descriptor(),
        DTypeId::F64.descriptor(),
        DTypeId::U8.descriptor(),
    ]
    .into_iter()
    .find(|dtype| {
        let mut probe = query(OperationKind::MatMulExact, 2);
        probe.dtype = *dtype;
        !backend.support(&probe).is_supported()
    });
    if unrepresentable.is_none() {
        return Err(
            "the registry claims matmul for every dtype probed, including Q8_0; a registry \
             that never says no cannot be planned against"
                .to_string(),
        );
    }
    Ok(())
}
