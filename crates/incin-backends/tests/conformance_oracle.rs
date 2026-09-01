//! The CPU backend answering for its own capability table.
//!
//! `docs/capabilities.md` publishes what each backend advertises, and states in
//! its own header that a row is "a canonical capability decision, not a claim
//! about a machine". This suite is the beginning of turning the CPU half of
//! that document into a claim about a machine: every tuple the CPU registry
//! advertises and the harness can build is executed, and a tuple the registry
//! does not advertise is checked to be refused.
//!
//! The floor below is a ratchet, not a target. It exists so that a fixture gap
//! is a visible number rather than a silent pass, and so that closing one
//! cannot be undone without the test saying so.

#![cfg(feature = "cpu")]

use std::collections::BTreeSet;

use incin_backends::conformance::{Coverage, Verdict, run_cpu_self_check};

/// Operations the harness reaches today, out of the 162 the CPU backend has an
/// executor for.
///
/// Exact rather than slack, because the point of a ratchet is that it cannot
/// slip. Raise it when a fixture family lands. Lowering it means fixtures were
/// removed or a capability row was narrowed, and either should be a deliberate
/// edit in the same commit rather than a number quietly absorbing it.
///
/// The one outstanding is `quantized_matmul`, and it is outstanding because two
/// contracts disagree rather than because the harness cannot build an operand.
/// The catalog gives it `OutputRule::MatMul`, which requires `lhs[-1] ==
/// rhs[-2]`; the CPU kernel reads the right operand as `[n, k]`, because a
/// block encoding shares one scale across thirty-two consecutive values and so
/// can only be contracted along its contiguous axis. A square operand satisfies
/// both and would report an agreement that is not there.
///
/// `quantize` and `dequantize` were counted with the same reason until this
/// floor and did not need one. `block_extents` widens the ladder's last extent
/// to a whole block, which is the shape a block encoding can hold, and the
/// block size is the dtype registry's rather than a number this harness
/// invented.
const COVERED_OPERATION_FLOOR: usize = 161;

#[test]
fn the_cpu_backend_executes_every_advertised_tuple_it_is_asked_about() {
    let report = run_cpu_self_check();
    assert!(
        report.findings().next().is_none(),
        "{}",
        report.findings_text()
    );
}

#[test]
fn fixture_coverage_does_not_regress() {
    let report = run_cpu_self_check();
    let covered = report.covered_operations();
    assert!(
        covered.len() >= COVERED_OPERATION_FLOOR,
        "fixture coverage fell to {} operations, below the recorded floor of \
         {COVERED_OPERATION_FLOOR}. Covered: {covered:?}",
        covered.len()
    );
}

/// Every operation the CPU registry advertises is posed at least once.
///
/// The harness enumerates tuples from the registry, so this is really a check
/// on the enumeration: a rule whose product came out empty would drop its
/// operation from the run silently, and a run that never asks about an
/// operation reports the same green as one that asks and passes.
#[test]
fn every_advertised_operation_is_posed_at_least_once() {
    use incin_backends::capability::{CPU_CAPABILITIES, registry};
    use incin_core::tensor::device::DeviceKind;

    let report = run_cpu_self_check();
    let posed: BTreeSet<_> = report
        .observations
        .iter()
        .map(|observation| observation.tuple.operation)
        .collect();

    let _ = registry(DeviceKind::Cpu);
    for rule in CPU_CAPABILITIES {
        assert!(
            posed.contains(&rule.operation),
            "{} has a capability rule but the enumeration posed no tuple for it",
            rule.operation
        );
    }
}

/// An unfixtured operation names why, so the list reads as work rather than as
/// an unexplained hole.
#[test]
fn every_uncovered_operation_carries_a_reason() {
    let report = run_cpu_self_check();
    for observation in &report.observations {
        if let Verdict::NotCovered(Coverage::Unfixtured(reason)) = &observation.verdict {
            assert!(
                !reason.is_empty(),
                "{} is unfixtured with no reason recorded",
                observation.tuple.operation
            );
        }
    }
}

/// Print the state of the run, so a contributor closing a gap can see the list
/// without reading the harness.
#[test]
fn report_the_current_coverage() {
    let report = run_cpu_self_check();
    println!(
        "advertised tuples: {}, executed: {}, covered operations: {}",
        report.observations.len(),
        report.executed(),
        report.covered_operations().len()
    );
    for (operation, reason) in report.unfixtured_operations() {
        println!("  unfixtured  {operation}: {reason}");
    }
    println!("{}", report.findings_text());
}

/// A union row is posed at its primary operand's floor as well as its own.
///
/// `admit_invocation` checks every operand against one resolved row, so a row
/// states the loosest rank across all of them. `conv2d` declares a floor of one
/// for the sake of its bias vector while the activation needs a channel axis
/// and two spatial ones, which puts the activation's real floor of three in the
/// *interior* of the declared range. Boundary enumeration does not look at the
/// interior, so the three convolution kernels panicked there unobserved until
/// the catalog's own `accepted_ranks` floor became a third boundary.
///
/// The negative half of the claim matters as much: a row whose floor already
/// agrees with the catalog's gains nothing and must not grow a third point, or
/// every rule in the registry pays for four rows.
#[test]
fn a_union_row_is_posed_at_the_catalog_floor_as_well_as_its_own() {
    use incin_backends::conformance::advertised_tuples;
    use incin_core::shapes::error::OperationKind;
    use incin_core::tensor::device::DeviceKind;

    let ranks = |operation: OperationKind| -> BTreeSet<usize> {
        advertised_tuples(DeviceKind::Cpu)
            .into_iter()
            .filter(|tuple| tuple.operation == operation)
            .map(|tuple| tuple.rank)
            .collect()
    };

    // `accepted_ranks` is 3..=4 for the two-dimensional convolutions and 2..=3
    // for `conv1d`, against a declared floor of one in all three.
    assert_eq!(
        ranks(OperationKind::Conv2dExact),
        BTreeSet::from([1, 3, 4]),
        "conv2d must be posed at the activation's floor, not only at the row's"
    );
    assert_eq!(ranks(OperationKind::Conv1dExact), BTreeSet::from([1, 2, 3]));
    assert_eq!(
        ranks(OperationKind::ConvTranspose2d),
        BTreeSet::from([1, 3, 4])
    );

    // A row that agrees with the catalog keeps two boundaries. `batch_norm` is
    // a union row too, but the tighter floor its activation needs is enforced
    // inside `BatchNormAttributes::validate` rather than declared in
    // `accepted_ranks`, so there is nothing here for the enumeration to read.
    assert_eq!(ranks(OperationKind::BatchNorm), BTreeSet::from([1, 4]));
    assert_eq!(ranks(OperationKind::Relu), BTreeSet::from([0, 4]));
}
