//! The macro suite `PROPOSALS.md`'s macro policy requires (`CI-005`).
//!
//! Every public macro must "provide compile-pass, compile-fail, hygiene,
//! rename, and rustfmt tests". This crate had no `tests/` directory at all, so
//! `cargo test -p incin-macros` ran nothing, and it exited zero doing it.
//!
//! The three macros in scope are the ones that exist today: `s!`, `idx!`, and
//! `#[module]`. The cases themselves are files under `tests/compile_pass/` and
//! `tests/compile_fail/`; this module is the harness and the guards that keep
//! the cases honest.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

#[test]
fn compile_pass_and_compile_fail() {
    let t = trybuild::TestCases::new();
    t.pass("tests/compile_pass/*.rs");
    t.compile_fail("tests/compile_fail/*.rs");
}

/// Text that must appear in a case's recorded output, keyed by file stem.
///
/// A macro rejection carries no error code — it is a `syn::Error` the macro
/// wrote — so each row is the message itself. That is the stronger pin anyway:
/// it fails when the wording changes, and the wording is the diagnostic the
/// user reads.
///
/// Adding a case without adding a row here fails
/// [`compile_fail_cases_fail_for_their_stated_reason`]. `SHP-007` added the
/// same guard to `incin-core` after finding four cases whose sources had
/// rotted into passing while asserting nothing.
fn expected_reasons() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        (
            "idx_rejects_an_open_ended_range",
            "idx! currently only supports `..` or `start..end`",
        ),
        (
            "idx_rejects_a_qualified_named_dim",
            "idx! expects simple identifiers for NamedDyn",
        ),
        (
            "module_rejects_a_non_struct",
            "expected one of: `struct`, `enum`, `union`",
        ),
        (
            "module_rejects_an_unknown_argument",
            "unknown attribute argument for #[module]",
        ),
        ("s_rejects_a_non_path_dim", "expected identifier"),
        (
            "s_rejects_a_repeat_without_a_count",
            "expected integer literal",
        ),
        (
            "mesh_rejects_zero_degree",
            "mesh degree must be a non-zero positive integer",
        ),
        (
            "mesh_rejects_duplicate_axis",
            "duplicate `dp` / `data` axis in mesh!",
        ),
        (
            "mesh_rejects_unknown_axis",
            "unknown mesh axis key `foo`; expected `dp`, `tp`, or `pp`",
        ),
    ])
}

/// Output that means the case failed on its own scaffolding rather than on the
/// rule it was written for.
///
/// A macro case is unusually easy to break this way: the file has to import
/// the macro from somewhere, and an import that does not resolve produces a
/// confident red test that proves nothing about the macro.
const SCAFFOLDING_FAILURES: &[(&str, &str)] = &[
    ("E0432", "an import that does not resolve"),
    ("E0433", "a path that does not resolve"),
    ("E0603", "a path into a private module"),
    ("E0412", "a type that does not exist"),
];

#[test]
fn compile_fail_cases_fail_for_their_stated_reason() {
    let expected = expected_reasons();
    let dir = Path::new("tests/compile_fail");
    let mut problems = Vec::new();

    let mut stems: Vec<String> = fs::read_dir(dir)
        .expect("tests/compile_fail must exist")
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

    assert!(!stems.is_empty(), "tests/compile_fail is empty");

    for stem in &stems {
        let Some(reason) = expected.get(stem.as_str()) else {
            problems.push(format!(
                "{stem}: no row in expected_reasons(); say what this case proves"
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

        for (code, description) in SCAFFOLDING_FAILURES {
            if output.contains(code) {
                problems.push(format!(
                    "{stem}: recorded output contains {code} ({description}), which is \
                     the case's own scaffolding failing rather than the macro"
                ));
            }
        }
    }

    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Every category the macro policy lists must have a case.
///
/// The list is the requirement, so a suite that quietly stops covering one of
/// them is the failure worth catching. Hygiene and rename are the two nobody
/// notices missing: everything compiles without them, right up until a user
/// has an item named `incin`.
#[test]
fn every_policy_category_is_covered() {
    let dir = Path::new("tests/compile_pass");
    let names: Vec<String> = fs::read_dir(dir)
        .expect("tests/compile_pass must exist")
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .map(|path| {
            path.file_stem()
                .expect("a .rs file has a stem")
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    for required in ["hygiene", "rename", "rustfmt_fixture"] {
        assert!(
            names.iter().any(|n| n == required),
            "tests/compile_pass/{required}.rs is missing; the macro policy lists it"
        );
    }

    // One grammar case per macro in this row's scope.
    for macro_name in [
        "s_grammar",
        "idx_grammar",
        "module_arguments",
        "mesh_grammar",
    ] {
        assert!(
            names.iter().any(|n| n == macro_name),
            "no compile-pass case covers `{macro_name}`"
        );
    }
}

/// A file invoking all three macros must survive `rustfmt`.
///
/// This is the policy's rustfmt requirement, and it is not a formality. A
/// function-like macro whose invocation is not parseable as Rust makes
/// `rustfmt` skip the whole call, and an attribute macro on an item it cannot
/// parse can make it skip more. Either way the cost lands on every downstream
/// user's `cargo fmt`, and nothing else in this repository would notice,
/// because the macros' own sources format fine.
///
/// Asserted as a fixed point: formatting the already-formatted fixture returns
/// it unchanged.
#[test]
fn invoking_the_macros_leaves_a_file_formattable() {
    let fixture = Path::new("tests/compile_pass/rustfmt_fixture.rs");
    let source = fs::read_to_string(fixture).expect("the rustfmt fixture must exist");

    let output = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout", "--quiet"])
        .arg(fixture)
        .output();

    let Ok(output) = output else {
        // `rustfmt` is a declared toolchain component, but a contributor on a
        // partial toolchain should get a skipped check rather than a failure
        // about something they did not break. CI has it.
        eprintln!("rustfmt is not installed; skipping the formatting check");
        return;
    };

    assert!(
        output.status.success(),
        "rustfmt failed on a file invoking s!, idx! and #[module]:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let formatted = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        formatted.trim(),
        source.trim(),
        "rustfmt reformatted the fixture, so it is no longer the canonical form; \
         run rustfmt on tests/compile_pass/rustfmt_fixture.rs and commit the result"
    );
}
