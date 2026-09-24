//! The registry-enumerated conformance matrix (issue #83).
//!
//! Three suites together deliver the issue's claim:
//!
//! * `tests/conformance_oracle.rs` runs what the CPU registry advertises and
//!   checks that it executes, that unadvertised dtypes are refused, and that
//!   training rows record a recipe.
//! * `tests/conformance_values.rs` checks CPU forward values against an
//!   independent `f64` oracle.
//! * This suite is the matrix around them: every backend family's registry
//!   enumerated, every cell carrying an explicit pass, fail, or skip; the
//!   training rows' recipes checked for correctness against central
//!   differences; and the whole run written out as the machine-readable
//!   artifact the documentation generator reads.
//!
//! # What this suite does NOT yet cover (honestly)
//!
//! * **Device legs do not execute.** CUDA, WGPU, and Metal enumerate their
//!   registries and report every cell as an explicit `NoDevice` skip. Running
//!   their kernels needs the #82 runner; executing them against the CPU
//!   backend would test nothing.
//! * **Gradient fixtures are curated, not enumerated.** Eleven operations
//!   have central-difference fixtures; a training row for any other operation
//!   is covered by the tape-recording check, not by a correctness sweep.
//!   Extending `GRADIENT_OPERATIONS` is the work item, and the floor below
//!   ratchets it.
//! * **Values are CPU-only.** Cross-backend value comparison reuses this
//!   matrix's enumeration the moment a device leg can execute.
//!
//! # Artifact schema (version [`SCHEMA_VERSION`])
//!
//! The normative schema reference is `src/conformance/artifact.rs`. In
//! short: the file holds `schema_version`, a per-leg `summary` of
//! `{advertised, passed, failed, skipped}` counts, `cells` with one object
//! per *executed* tuple (`status` `"passed"` or `"failed"`, `reason` only on
//! failures), and `skipped` with one object per `(backend, operation, skip)`
//! group (`skip` `"no-device"`, `"no-fixture"`, or `"unbuildable"`, with the
//! count it stands for, the reason once, and one example tuple). Counts
//! reconcile per leg: `advertised == passed + failed + skipped`. A consumer
//! that does not recognize the version refuses the file rather than guessing.

#![cfg(feature = "cpu")]

use std::collections::BTreeSet;

use incin_backends::conformance::{
    CellVerdict, GRADIENT_OPERATION_FLOOR, SCHEMA_VERSION, SkipReason, check_gradients,
    gradient_findings_text, run_matrix,
};
use incin_core::tensor::device::DeviceKind;

/// The CPU leg executes every advertised tuple it is asked about, with no
/// failures. If the oracle disagrees with itself, nothing downstream means
/// anything: this is the issue's first acceptance criterion.
#[test]
fn the_cpu_leg_has_no_failures() {
    let report = run_matrix();
    assert!(
        report.passed(),
        "the CPU leg disagrees with itself:\n{}",
        report.summary_text()
    );
}

/// No silent skips: every cell ends in exactly one of pass, fail, or an
/// explicit skip with a reason. A cell that was never executed and never
/// recorded why would be the failure mode this matrix exists to prevent.
#[test]
fn every_cell_carries_an_explicit_verdict() {
    let report = run_matrix();
    assert!(
        !report.cells.is_empty(),
        "the matrix posed no cells at all, which reports the same green as a pass"
    );
    for cell in &report.cells {
        match &cell.verdict {
            CellVerdict::Passed | CellVerdict::Failed(_) => {}
            CellVerdict::Skipped(reason) => {
                assert!(
                    !reason.reason().is_empty(),
                    "{} is skipped with no reason recorded",
                    cell.label()
                );
            }
        }
    }
}

/// Every operation the CPU registry advertises is posed at least once, and
/// every negative probe the run took is accounted for.
///
/// The harness enumerates tuples from the registry, so this is a check on the
/// enumeration: a rule whose product came out empty would drop its operation
/// silently. Probes are counted separately because they are not advertised
/// tuples; lumping them in would let a probe stand in for coverage.
#[test]
fn every_advertised_operation_is_posed_and_probes_are_counted() {
    use incin_backends::capability::CPU_CAPABILITIES;

    let report = run_matrix();
    let posed: BTreeSet<_> = report
        .cells
        .iter()
        .filter(|cell| cell.backend == DeviceKind::Cpu && !cell.negative_probe)
        .map(|cell| cell.operation)
        .collect();
    for rule in CPU_CAPABILITIES {
        assert!(
            posed.contains(&rule.operation),
            "{} has a capability rule but the matrix posed no tuple for it",
            rule.operation
        );
    }

    let probes: Vec<_> = report
        .cells
        .iter()
        .filter(|cell| cell.negative_probe)
        .collect();
    assert!(
        !probes.is_empty(),
        "no negative probes ran: unadvertised-but-executed would go unobserved"
    );
    // A passed probe is a refusal that held. A failed one is a backend
    // executing something it never advertised, which the first test already
    // fails the run for; this names the probe explicitly either way.
    for probe in &probes {
        assert!(
            !probe.verdict.is_finding(),
            "unadvertised-but-executed: {}",
            probe.label()
        );
    }
}

/// Device legs enumerate their registries and skip explicitly.
///
/// CUDA, WGPU, and Metal have nowhere to run on this machine (issue #82), so
/// every cell reports `NoDevice`: enumerated, posed nowhere, and recorded as
/// such. A leg that reported its tuples as passed would be lying, and one
/// that omitted them would be a silent skip.
#[test]
fn device_legs_enumerate_and_skip_without_a_device() {
    let report = run_matrix();
    for backend in [DeviceKind::Cuda, DeviceKind::Wgpu, DeviceKind::Metal] {
        let leg: Vec<_> = report.leg(backend).collect();
        assert!(
            !leg.is_empty(),
            "{backend:?} enumerated no tuples: a rule added there would extend no coverage"
        );
        for cell in &leg {
            assert_eq!(
                cell.verdict,
                CellVerdict::Skipped(SkipReason::NoDevice),
                "{} on {backend:?} is neither executed nor explicitly skipped",
                cell.label()
            );
        }
        assert_eq!(report.executed_on(backend), 0);
    }
}

/// Training rows with gradient fixtures agree with central differences.
///
/// This is the check that would have caught the pre-#93 gaps: a `Training:
/// yes` row whose backward is wrong or missing fails here, not silently. The
/// sweep compares every element of every input (`compared` is the proof it
/// ran), and the in-crate control
/// `conformance::gradient::tests::a_recipe_wrong_by_a_constant_factor_fails_the_sweep`
/// proves a wrong recipe fails it.
#[test]
fn training_gradients_match_central_differences() {
    let observations = check_gradients();
    assert!(
        observations.len() >= GRADIENT_OPERATION_FLOOR,
        "the gradient set shrank to {} operations, below the floor of {GRADIENT_OPERATION_FLOOR}",
        observations.len()
    );
    let mut failures = Vec::new();
    for observation in &observations {
        match &observation.verdict {
            incin_backends::conformance::GradientVerdict::Passed { compared, .. } => {
                assert!(
                    *compared > 0,
                    "{} passed with no comparisons: a sweep that compares nothing proves nothing",
                    observation.operation
                );
            }
            incin_backends::conformance::GradientVerdict::Failed(_) => {
                failures.push(observation.operation);
            }
            incin_backends::conformance::GradientVerdict::Skipped(_) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "gradient mismatches:\n{}",
        gradient_findings_text(&observations)
    );
}

/// The run writes the machine-readable artifact the documentation generator
/// reads: schema version, per-leg counts, and one compact object per cell.
#[test]
fn the_run_writes_a_machine_readable_artifact() {
    let report = run_matrix();
    let dir = tempfile::tempdir().expect("a scratch directory for the artifact");
    let path = dir.path().join("matrix.json");
    let written = report
        .write_json(&path)
        .expect("the artifact must be writable");
    assert_eq!(written, path.display().to_string());

    let text = std::fs::read_to_string(&path).expect("the artifact must be readable");
    // Small by construction: one compact object per cell, no tensors, no
    // values. Thousands of cells must stay well under a megabyte.
    assert!(
        text.len() < 1_000_000,
        "the artifact is {} bytes; it summarizes, it does not dump",
        text.len()
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&text).expect("the artifact must be valid JSON");
    assert_eq!(
        parsed["schema_version"], SCHEMA_VERSION,
        "a consumer that does not recognize the version must refuse the file"
    );
    for backend in ["Cpu", "Cuda", "Wgpu", "Metal"] {
        let leg = &parsed["summary"][backend];
        assert!(
            leg.is_object(),
            "summary has no leg for {backend}: {}",
            parsed["summary"]
        );
        for key in ["advertised", "passed", "failed", "skipped"] {
            assert!(
                leg[key].is_u64(),
                "summary[{backend}][{key}] is not a count: {leg}"
            );
        }
        let advertised = leg["advertised"].as_u64().expect("checked above");
        assert_eq!(
            advertised,
            report
                .leg(match backend {
                    "Cpu" => DeviceKind::Cpu,
                    "Cuda" => DeviceKind::Cuda,
                    "Wgpu" => DeviceKind::Wgpu,
                    _ => DeviceKind::Metal,
                })
                .count() as u64,
            "summary counts disagree with the report for {backend}"
        );
    }
    let cells = parsed["cells"].as_array().expect("cells is an array");
    let skipped = parsed["skipped"].as_array().expect("skipped is an array");
    // Executed cells are recorded per tuple; skips are grouped by
    // (backend, operation, skip) so repeated reason prose is stored once.
    let executed = report
        .cells
        .iter()
        .filter(|cell| cell.verdict.was_executed())
        .count();
    assert_eq!(
        cells.len(),
        executed,
        "the artifact dropped or duplicated executed cells"
    );
    for cell in cells {
        let status = cell["status"].as_str().expect("every cell has a status");
        assert!(
            ["passed", "failed"].contains(&status),
            "unknown status {status}: executed cells pass or fail"
        );
        if status == "passed" {
            assert!(
                cell.get("reason").is_none(),
                "a pass needs no explanation: {cell}"
            );
        } else {
            let reason = cell["reason"].as_str().expect("a failure names why");
            assert!(
                !reason.is_empty(),
                "an empty reason explains nothing: {cell}"
            );
        }
    }
    let mut skipped_total = 0_u64;
    for group in skipped {
        for key in ["backend", "operation", "skip", "reason", "example"] {
            assert!(
                group.get(key).is_some(),
                "a skipped group carries {key}: {group}"
            );
        }
        let skip = group["skip"].as_str().expect("a group names its skip");
        assert!(
            ["no-device", "no-fixture", "unbuildable"].contains(&skip),
            "unknown skip {skip}: there are three explicit reasons"
        );
        let count = group["count"].as_u64().expect("a group counts its tuples");
        assert!(count > 0, "an empty group stands for nothing: {group}");
        assert!(
            !group["reason"].as_str().unwrap_or_default().is_empty(),
            "an empty reason explains nothing: {group}"
        );
        skipped_total += count;
    }
    // Counts reconcile: advertised == passed + failed + skipped, per leg.
    for backend in ["Cpu", "Cuda", "Wgpu", "Metal"] {
        let leg = &parsed["summary"][backend];
        let reconciled = leg["passed"].as_u64().unwrap_or(0)
            + leg["failed"].as_u64().unwrap_or(0)
            + leg["skipped"].as_u64().unwrap_or(0);
        assert_eq!(
            leg["advertised"].as_u64().unwrap_or(0),
            reconciled,
            "summary counts do not reconcile for {backend}: a consumer must refuse this file"
        );
    }
    let report_skipped = report
        .cells
        .iter()
        .filter(|cell| matches!(cell.verdict, CellVerdict::Skipped(_)))
        .count() as u64;
    assert_eq!(
        skipped_total, report_skipped,
        "skipped groups stand for {skipped_total} tuples but the report skipped {report_skipped}"
    );
}

/// Print the state of the run, so a contributor closing a gap can see the
/// counts without reading the harness.
#[test]
fn report_the_matrix_counts() {
    let report = run_matrix();
    println!("{}", report.summary_text());
    let probes = report
        .cells
        .iter()
        .filter(|cell| cell.negative_probe)
        .count();
    println!("negative probes: {probes}");
    for reason in ["no-device", "no-fixture", "unbuildable"] {
        let count = report
            .cells
            .iter()
            .filter(|cell| match &cell.verdict {
                CellVerdict::Skipped(skip) => match skip {
                    SkipReason::NoDevice => reason == "no-device",
                    SkipReason::NoFixture(_) => reason == "no-fixture",
                    SkipReason::Unbuildable(_) => reason == "unbuildable",
                },
                _ => false,
            })
            .count();
        println!("skipped {reason}: {count}");
    }
    let gradients = check_gradients();
    let passed = gradients.iter().filter(|o| !o.verdict.is_finding()).count();
    println!(
        "gradients: {passed} of {} operations agree",
        gradients.len()
    );
}
