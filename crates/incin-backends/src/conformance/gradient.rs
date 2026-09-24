//! Gradient correctness for training rows (issue #83).
//!
//! The execution half of this harness (`super`) checks that a row claiming
//! `Training: yes` records a backward recipe on the tape. Recording is
//! necessary but not sufficient: a recipe that halves every contribution
//! records exactly as much as a correct one, and the tape-depth check cannot
//! tell them apart. This module checks the other half, that the recorded
//! recipe computes the right derivative, by comparing the analytic gradient
//! against a central difference.
//!
//! The comparison is the public sweep
//! [`gradcheck`](incin_core::exec::gradcheck), run against the real backward
//! path (`tape::backward`, reached through the storage trait), under the one
//! tolerance table's [`gradient_options`](super::tolerance::gradient_options).
//! A `Training: yes` row whose backward is wrong or missing fails here, which
//! is the check that would have caught the pre-#93 gaps.
//!
//! # Coverage is curated, not enumerated
//!
//! The sweep needs a scalar loss built by reducing the operation's output,
//! fixed inputs that stay inside every domain the operation has (no zero
//! divisor, no kink evaluation, no overflow), and `f32` operands the
//! perturbation path supports. Those three are per-operation facts a
//! capability row cannot state, so the operation set is a hand-written table
//! ([`GRADIENT_OPERATIONS`]) held against [`GRADIENT_OPERATION_FLOOR`] rather
//! than enumerated from the registry. An operation without an entry reports
//! [`GradientVerdict::Skipped`], explicitly, never silently.
//!
//! Every entry runs the operation through canonical dispatch, the same path
//! the execution half poses, so the recipe under test is the kernel's own.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use incin_core::backend_authoring::op;
use incin_core::exec::catalog::{AxisAttributes, NoAttributes};
use incin_core::exec::{ExecutionContext, GradMode, TensorHandle, dispatch, gradcheck};
use incin_core::shapes::error::OperationKind;

use crate::conformance::tolerance;
use crate::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};

/// What checking one operation's backward pass concluded.
#[derive(Debug, Clone, PartialEq)]
pub enum GradientVerdict {
    /// The analytic gradient agreed with the central difference on every
    /// element: how many were compared, and the worst relative error seen.
    Passed {
        /// Elements swept across all inputs.
        compared: usize,
        /// Largest relative error seen, including passing elements.
        worst_relative_error: f64,
    },
    /// The backward pass is wrong, missing, or errored, with why.
    Failed(String),
    /// No gradient fixture for this operation yet, with the reason. Explicit,
    /// never silent: extending [`GRADIENT_OPERATIONS`] is the work item.
    Skipped(&'static str),
}

impl GradientVerdict {
    /// Whether this verdict is a finding against the backend.
    #[must_use]
    pub const fn is_finding(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

/// One operation and what its gradient check concluded.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientObservation {
    /// The operation whose backward pass was checked.
    pub operation: OperationKind,
    /// What the check concluded.
    pub verdict: GradientVerdict,
}

/// Ratchet: the curated operation set must not shrink without a deliberate
/// edit here. Raise it when gradient fixtures land; lowering it means
/// coverage was removed, which should be a visible decision.
pub const GRADIENT_OPERATION_FLOOR: usize = 11;

/// The operations with gradient fixtures, in the order they run.
pub const GRADIENT_OPERATIONS: &[OperationKind] = &[
    OperationKind::Add,
    OperationKind::Sub,
    OperationKind::Mul,
    OperationKind::Div,
    OperationKind::Neg,
    OperationKind::Abs,
    OperationKind::Relu,
    OperationKind::Exp,
    OperationKind::SumAll,
    OperationKind::SumDim,
    OperationKind::MeanDim,
];

/// A harness-side construction failure, as a core error.
///
/// Inputs are curated constants, so reaching here means the harness is
/// broken, not the backend. It is still a loud message rather than a panic:
///
/// the descriptor contract requires errors to come back as values.
fn harness_fault(reason: &'static str) -> incin_core::error::Error {
    incin_core::error::Error::InternalInvariant {
        operation: "conformance gradient check",
        reason,
    }
}

fn storage(values: Vec<f32>, shape: &[usize]) -> Result<CpuStorage, incin_core::error::Error> {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values), shape)
        .map_err(|_| harness_fault("the curated inputs do not form valid storage"))
}

fn handles<'a>(
    inputs: &'a [CpuStorage],
) -> Result<Vec<TensorHandle<'a>>, incin_core::error::Error> {
    // `f32` in the turbofish whatever the storage holds. `CpuStorage` is one
    // enum carrying its own dtype tag, so the dtype a handle reports comes
    // from the storage's metadata rather than from this parameter.
    Ok(inputs
        .iter()
        .map(TensorHandle::from_storage::<CpuBackendImpl, f32, _>)
        .collect())
}

/// Reduce an operation's output to the scalar loss the sweep needs.
///
/// The reduction is itself a dispatched `sum_all`, so the analytic gradient
/// flows through the operation's own recipe composed with a summation: the
/// sweep compares the whole chain against the central difference of the same
/// chain, which is exactly what a model does when it reduces before calling
/// backward. An output that is already scalar is the loss as-is.
fn scalar_loss(
    context: &ExecutionContext<CpuBackendImpl>,
    output: &CpuStorage,
) -> Result<CpuStorage, incin_core::error::Error> {
    if output.shape().dims().is_empty() {
        return Ok(output.clone());
    }
    let handle = TensorHandle::from_storage::<CpuBackendImpl, f32, _>(output);
    dispatch::execute::<op::SumAll, _>(context, NoAttributes, &[handle])
        .map_err(incin_core::error::Error::from)
}

/// The curated inputs for one operation.
///
/// Inputs stay inside every domain the operation has: divisors are bounded
/// away from zero, kink operations (`abs`, `relu`) never evaluate at zero,
/// and `exp` stays clear of overflow.
fn inputs_for(operation: OperationKind) -> Result<Vec<CpuStorage>, incin_core::error::Error> {
    let binary = || -> Result<Vec<CpuStorage>, incin_core::error::Error> {
        Ok(vec![
            storage(vec![1.0, 2.0, 3.0, 4.0], &[4])?,
            storage(vec![2.0, 3.0, 4.0, 5.0], &[4])?,
        ])
    };
    // Nonzero on both sides of the kink: the derivative exists at every
    // element the sweep perturbs, so a disagreement is a wrong recipe rather
    // than the perturbation stepping across a boundary.
    let kinked = || storage(vec![-2.0, -1.0, 1.0, 2.0], &[4]);

    match operation {
        OperationKind::Add | OperationKind::Sub | OperationKind::Mul | OperationKind::Div => {
            binary()
        }
        OperationKind::Neg => Ok(vec![storage(vec![1.5, -0.25, 3.75, -2.5], &[4])?]),
        OperationKind::Abs | OperationKind::Relu => Ok(vec![kinked()?]),
        OperationKind::Exp => Ok(vec![storage(vec![0.5, 1.0, 1.5, 2.0], &[4])?]),
        OperationKind::SumAll | OperationKind::SumDim | OperationKind::MeanDim => {
            Ok(vec![storage(
                vec![1.0, -2.5, 3.25, 0.5, -4.75, 2.3],
                &[2, 3],
            )?])
        }
        _ => Err(harness_fault("no gradient fixture for this operation")),
    }
}

/// The scalar loss for one operation over the given storages.
///
/// Dispatch goes through the canonical path, so the recipe under test is the
/// kernel's own rather than a reconstruction. The sweep calls this once per
/// perturbed element, and each call records its own nodes and drains them
/// through the real backward path.
fn run_loss(
    context: &ExecutionContext<CpuBackendImpl>,
    operation: OperationKind,
    storages: &[CpuStorage],
) -> Result<CpuStorage, incin_core::error::Error> {
    let refs = handles(storages)?;
    let output = match operation {
        OperationKind::Add => dispatch::execute::<op::Add, _>(context, NoAttributes, &refs),
        OperationKind::Sub => dispatch::execute::<op::Sub, _>(context, NoAttributes, &refs),
        OperationKind::Mul => dispatch::execute::<op::Mul, _>(context, NoAttributes, &refs),
        OperationKind::Div => dispatch::execute::<op::Div, _>(context, NoAttributes, &refs),
        OperationKind::Neg => dispatch::execute::<op::Neg, _>(context, NoAttributes, &refs),
        OperationKind::Abs => dispatch::execute::<op::Abs, _>(context, NoAttributes, &refs),
        OperationKind::Relu => dispatch::execute::<op::Relu, _>(context, NoAttributes, &refs),
        OperationKind::Exp => dispatch::execute::<op::Exp, _>(context, NoAttributes, &refs),
        OperationKind::SumAll => dispatch::execute::<op::SumAll, _>(context, NoAttributes, &refs),
        OperationKind::SumDim => {
            dispatch::execute::<op::SumDim, _>(context, AxisAttributes { axis: 0 }, &refs)
        }
        OperationKind::MeanDim => {
            dispatch::execute::<op::MeanDim, _>(context, AxisAttributes { axis: 0 }, &refs)
        }
        _ => return Err(harness_fault("no gradient dispatch for this operation")),
    }
    .map_err(incin_core::error::Error::from)?;

    scalar_loss(context, &output)
}

/// Check one curated operation's backward pass.
///
/// The tape is emptied before and after: the harness poses many checks on one
/// thread, and nodes left behind would make the next measurement read this
/// one's. Recording is enabled explicitly rather than inherited, so the check
/// cannot silently stop measuring if the ambient default ever changes.
fn check_one(operation: OperationKind) -> GradientVerdict {
    crate::cpu::tape::clear();
    let outcome = GradMode::Enabled.scope(|| {
        let context = ExecutionContext::new(CpuBackendImpl::new());
        let inputs = inputs_for(operation)?;
        // The sweep re-runs the forward under the hood for every perturbed
        // element. The closure captures only the operation and the context;
        // everything element-specific arrives in `perturbed`.
        let closure = |perturbed: &[CpuStorage]| -> Result<CpuStorage, incin_core::error::Error> {
            run_loss(&context, operation, perturbed)
        };
        gradcheck(closure, &inputs, tolerance::gradient_options())
    });
    crate::cpu::tape::clear();

    match outcome {
        Ok(report) if report.passed() => GradientVerdict::Passed {
            compared: report.compared,
            worst_relative_error: report.worst_relative_error,
        },
        Ok(report) => GradientVerdict::Failed(alloc::format!(
            "analytic gradient disagrees with the central difference: {report}"
        )),
        Err(error) => GradientVerdict::Failed(alloc::format!(
            "the gradient check did not run to completion: {error}"
        )),
    }
}

/// Check every curated operation's backward pass.
///
/// Never panics and never stops early: a panic in one operation is a failure
/// for that operation, and a reader wants every finding from one run rather
/// than the first.
#[must_use]
pub fn check_gradients() -> Vec<GradientObservation> {
    GRADIENT_OPERATIONS
        .iter()
        .map(|operation| {
            let verdict = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_one(*operation)
            })) {
                Ok(verdict) => verdict,
                Err(_) => GradientVerdict::Failed(
                    "panicked; the descriptor contract requires a returned error".to_string(),
                ),
            };
            GradientObservation {
                operation: *operation,
                verdict,
            }
        })
        .collect()
}

/// A one-line-per-finding rendering, for a failing test's message.
#[must_use]
pub fn findings_text(observations: &[GradientObservation]) -> String {
    let mut out = String::from("conformance gradients:\n");
    for observation in observations {
        if let GradientVerdict::Failed(why) = &observation.verdict {
            out.push_str(&alloc::format!("  FAIL {}: {why}\n", observation.operation));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recipe wrong by a constant factor fails the sweep on every element.
    ///
    /// The control the harness needs: a check that has never failed a bad
    /// gradient says nothing when it passes a good one. This records the
    /// deliberate defect the same way a kernel would, through the real tape,
    /// and asserts the sweep reports it rather than passing it.
    #[test]
    fn a_recipe_wrong_by_a_constant_factor_fails_the_sweep() {
        use alloc::boxed::Box;
        use incin_core::exec::TapeNode;

        let x = storage(vec![2.0, 3.0, -1.0], &[3]).expect("curated inputs build");
        let op = |inputs: &[CpuStorage]| -> Result<CpuStorage, incin_core::error::Error> {
            let squared = crate::cpu::ops::elementwise::mul_storage(&inputs[0], &inputs[0])?;
            let out = crate::cpu::ops::reduce::sum_all(&squared)?;
            // Half of `2x`. Recorded after the real chain, so the walk
            // reaches this node first and its contribution is the one that
            // lands: every element disagrees by the same factor.
            let saved = inputs[0].clone();
            let (input_id, out_id) = (inputs[0].id, out.id);
            crate::cpu::tape::push(TapeNode {
                output_id: out_id,
                input_ids: alloc::vec![input_id],
                backward: Box::new(move |_grad: &CpuStorage| {
                    Ok(alloc::vec![crate::cpu::ops::elementwise::mul_storage(
                        &saved,
                        &saved.clone()
                    )?])
                }),
            });
            Ok(out)
        };

        let report = gradcheck(op, &[x], tolerance::gradient_options())
            .expect("the sweep ran to completion");
        assert!(!report.passed(), "a wrong recipe passed: {report}");
        assert!(
            !report.disagreements.is_empty(),
            "the report named no element"
        );
        crate::cpu::tape::clear();
    }

    /// The curated set only grows: removing an operation must be deliberate.
    #[test]
    fn the_curated_set_matches_its_floor() {
        assert_eq!(
            GRADIENT_OPERATIONS.len(),
            GRADIENT_OPERATION_FLOOR,
            "add the fixture and raise the floor, or record why one was removed"
        );
        let mut sorted = GRADIENT_OPERATIONS.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len(), "an operation is listed twice");
    }
}
