//! Verifies that the CUDA hardware suite actually ran (`EXE-004`, `EXE-005`,
//! `EXE-008`).
//!
//! A test run on a machine with no GPU reports `0 passed` and exits zero, which
//! is indistinguishable by exit code from a full pass. The hardware workflow
//! therefore needs a floor: a count that a device-less or misconfigured run
//! cannot reach.
//!
//! That floor used to be the literal `60` written into the workflow. Literals
//! do not track the tree. By the time this task was written the suite ran 65
//! tests, so the guard had five tests of slack -- enough to absorb the loss of
//! an entire suite (`cuda_reduce_ops` is five tests) and still report success.
//! A guard whose whole purpose is detecting tests that evaporate should not have
//! room for a suite to evaporate inside it.
//!
//! So the expectation is derived from the tree instead. Every `#[ignore]` in
//! `incin-backends` is classified by its reason string as either running on the
//! CUDA runner or deliberately excluded, and the expected count is simply how
//! many fall in the first group. Adding a hardware test raises the floor with no
//! edit here; deleting one lowers it, which is fine, because deleting a test is
//! a visible diff, while a test that silently stops running is not.
//!
//! An unrecognised reason string is a hard failure rather than a default, so a
//! new kind of ignored test cannot quietly land on either side of the line.

use std::path::Path;
use std::process::ExitCode;

const CRATE_ROOT: &str = "crates/incin-backends";

/// Reasons whose tests run on the CUDA hardware runner.
///
/// These are the ignored tests `cargo test -p incin-backends --features
/// std,cpu,cuda -- --ignored` is expected to execute on a machine with a device
/// and a CUDA toolchain.
const RUNS_ON_CUDA_RUNNER: &[&str] = &[
    "requires CUDA hardware",
    "requires one CUDA device",
    "requires a CUDA device and driver",
    "requires a locally installed NVRTC shared library",
];

/// Reasons deliberately excluded from the count, with the reason for excluding.
///
/// Each needs something the single-GPU runner does not have, so counting them
/// would make the floor unreachable and the guard useless.
const EXCLUDED: &[(&str, &str)] = &[
    (
        "requires two network-accessible CUDA hosts with NCCL",
        "multi-host; runs in the dist2-network job",
    ),
    (
        "probes the optional system NCCL shared library",
        "depends on an optional system library",
    ),
    (
        "run by tools/soundness.sh tsan",
        "needs a thread-sanitizer build",
    ),
    (
        "microbenchmark: run explicitly with --release --ignored --nocapture",
        "a measurement, not a pass/fail test",
    ),
];

/// One `#[ignore = "..."]` found in the tree.
struct Ignored {
    reason: String,
    file: String,
}

/// Collects every `#[ignore = "..."]` reason under `dir`.
fn collect(dir: &Path, found: &mut Vec<Ignored>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, found)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path)?;
            for line in text.lines() {
                let Some(rest) = line.trim_start().strip_prefix("#[ignore = \"") else {
                    continue;
                };
                if let Some(reason) = rest.split_once("\"]").map(|(reason, _)| reason) {
                    found.push(Ignored {
                        reason: reason.to_owned(),
                        file: path.display().to_string(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// The number of ignored tests the CUDA runner is expected to execute.
///
/// Returns `Err` with a description if any reason string is unclassified.
fn expected_count() -> Result<usize, String> {
    let mut found = Vec::new();
    collect(Path::new(CRATE_ROOT), &mut found)
        .map_err(|error| format!("could not scan {CRATE_ROOT}: {error}"))?;

    let mut expected = 0usize;
    let mut unknown = Vec::new();
    for ignored in &found {
        if RUNS_ON_CUDA_RUNNER.contains(&ignored.reason.as_str()) {
            expected += 1;
        } else if !EXCLUDED.iter().any(|(reason, _)| *reason == ignored.reason) {
            unknown.push(ignored);
        }
    }

    if !unknown.is_empty() {
        let mut message = String::from(
            "unclassified #[ignore] reason(s). Add each to RUNS_ON_CUDA_RUNNER or \
             EXCLUDED in xtask/src/hardware.rs so the hardware floor stays exact:\n",
        );
        for ignored in unknown {
            message.push_str(&format!("  {:?}  ({})\n", ignored.reason, ignored.file));
        }
        return Err(message);
    }
    Ok(expected)
}

/// Sums the `passed` field of every `test result: ok.` line in a cargo test log.
fn passed_in_log(log: &str) -> usize {
    log.lines()
        .filter(|line| line.starts_with("test result: ok."))
        .filter_map(|line| {
            let (count, _) = line.strip_prefix("test result: ok. ")?.split_once(' ')?;
            count.parse::<usize>().ok()
        })
        .sum()
}

pub fn check(log_path: Option<&str>) -> ExitCode {
    let expected = match expected_count() {
        Ok(expected) => expected,
        Err(message) => {
            eprintln!("error: {message}");
            return ExitCode::FAILURE;
        }
    };

    let Some(log_path) = log_path else {
        println!("{expected}");
        return ExitCode::SUCCESS;
    };

    let log = match std::fs::read_to_string(log_path) {
        Ok(log) => log,
        Err(error) => {
            eprintln!("error: could not read {log_path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    let actual = passed_in_log(&log);
    if actual < expected {
        eprintln!(
            "error: {actual} ignored CUDA tests ran, but {expected} are declared in \
             {CRATE_ROOT}.\n\
             A device-less run reports `0 passed` and exits zero, so a shortfall here \
             usually means the device, driver or NVRTC toolchain is missing -- not that \
             the tests passed."
        );
        return ExitCode::FAILURE;
    }
    println!("ignored CUDA tests executed: {actual} (declared: {expected})");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::{expected_count, passed_in_log};

    /// Every reason string in the tree must be classified.
    ///
    /// This is the check that keeps the floor honest: a new `#[ignore]` with an
    /// unfamiliar reason fails here rather than silently landing outside the
    /// count, which is how a hardware test would stop being guarded.
    #[test]
    fn every_ignore_reason_is_classified() {
        // The task runs from the workspace root; skip when invoked elsewhere.
        if !std::path::Path::new(super::CRATE_ROOT).is_dir() {
            return;
        }
        match expected_count() {
            Ok(expected) => assert!(expected > 0, "no hardware tests found; the scan is broken"),
            Err(message) => panic!("{message}"),
        }
    }

    #[test]
    fn sums_passed_across_binaries() {
        let log = "\
test result: ok. 52 passed; 0 failed; 0 ignored; 0 measured; 444 filtered out; finished in 29s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out; finished in 0.00s
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.6s
";
        assert_eq!(passed_in_log(log), 65);
    }

    /// A device-less run is the case the floor exists to catch.
    #[test]
    fn a_device_less_run_sums_to_zero() {
        let log = "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 65 filtered out; \
                   finished in 0.00s\n";
        assert_eq!(passed_in_log(log), 0);
    }

    /// A failing run must not be counted as passing tests.
    #[test]
    fn failed_results_are_not_counted() {
        let log = "test result: FAILED. 3 passed; 2 failed; 0 ignored; 0 measured; 0 filtered \
                   out; finished in 0.1s\n";
        assert_eq!(passed_in_log(log), 0);
    }
}
