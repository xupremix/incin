//! Shared checks for the trybuild suites.
//!
//! Separate feature and task-focused compile-fail directories keep each
//! acceptance command scoped to the rules it owns. A mesh case sitting in the
//! default suite, for example, would fail with "path does not resolve" when
//! the non-default feature is absent. Multiple directories are not multiple
//! properties, so the registry check lives here once.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Output that means the case failed on its own scaffolding.
///
/// Every entry here was found in `tests/compile_fail/` by `SHP-007`: a
/// `typenum::typenum::U2` import that never resolved, a file importing
/// `incin_core` twice, and a path through `incin_core::shapes`, which is
/// `pub(crate)`. Each produced a confident red test that proved nothing.
const SCAFFOLDING_FAILURES: &[(&str, &str)] = &[
    ("E0432", "an import that does not resolve"),
    ("E0254", "a name imported twice"),
    ("E0433", "a path that does not resolve"),
    ("E0603", "a path into a private module"),
    ("E0412", "a type that does not exist"),
];

/// Asserts every case in `dir` still fails for the reason `expected` records.
///
/// `trybuild` compares each case against its recorded `.stderr` and passes when
/// they match. It has no opinion about *what* the error was, so a case whose
/// source has rotted keeps passing while asserting nothing about the rule it
/// was written for.
///
/// Adding a case to `dir` without adding a row to `expected` fails here. That
/// is deliberate: writing down what the case proves is the point, and a case
/// nobody could name a reason for is a case that is not pinning one.
pub fn compile_fail_cases_name_their_reason(dir: &Path, expected: &BTreeMap<&str, &str>) {
    let mut problems = Vec::new();

    let mut stems: Vec<String> = fs::read_dir(dir)
        .unwrap_or_else(|_| panic!("{} must exist", dir.display()))
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .map(|path| {
            path.file_stem()
                .expect("a .rs file has a stem")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    stems.sort();

    for stem in &stems {
        let Some(reason) = expected.get(stem.as_str()) else {
            problems.push(format!(
                "{stem}: no row in the expected reasons; say what this case proves"
            ));
            continue;
        };

        let recorded = dir.join(format!("{stem}.stderr"));
        let Ok(output) = fs::read_to_string(&recorded) else {
            problems.push(format!("{stem}: no recorded .stderr next to the case"));
            continue;
        };

        if !output.contains(reason) {
            problems.push(format!(
                "{stem}: recorded output does not mention {reason:?}, so the case \
                 is no longer failing for the reason it claims"
            ));
        }
        for (marker, description) in SCAFFOLDING_FAILURES {
            if output.contains(marker) {
                problems.push(format!(
                    "{stem}: recorded output contains {marker} ({description}); \
                     the case fails on its own scaffolding, not on the rule"
                ));
            }
        }
        if output.lines().any(|line| line.starts_with("warning")) {
            problems.push(format!(
                "{stem}: recorded output carries a warning, which will churn the \
                 baseline on unrelated changes"
            ));
        }
    }

    for stem in expected.keys() {
        if !stems.iter().any(|present| present == stem) {
            problems.push(format!(
                "{stem}: has a row in the expected reasons but no case file"
            ));
        }
    }

    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
