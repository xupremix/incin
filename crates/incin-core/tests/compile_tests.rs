//! The compile-fail suite, plus the check that it fails for the right reasons.
//!
//! `trybuild` compares each case against its recorded `.stderr` and passes when
//! they match. It has no opinion about *what* the error was, so a case whose
//! source has rotted — a broken import, a path to a module that has since become
//! private — keeps passing while asserting nothing about the rule it was written
//! for. `SHP-007` found four such cases in this directory, one of which
//! (`macro_module_invalid`) produced no error from the macro at all.
//!
//! [`compile_fail_cases_fail_for_their_stated_reason`] is the guard against that
//! happening again.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[test]
/// Compile fail.
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}

/// Text that must appear in a case's recorded output, keyed by file stem.
///
/// An error code where one exists, because that is the specific thing the case
/// is pinning: `E0624` is "this associated function is private" and nothing
/// else. Macro rejections have no code, so those are matched on the message the
/// macro emits.
///
/// Adding a case to `tests/compile_fail/` without adding a row here fails this
/// test. That is deliberate: writing down what the case proves is the point,
/// and a case nobody could name a reason for is a case that is not pinning one.
fn expected_reasons() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        // Shape rules, all unsatisfied trait bounds.
        ("concat_static_mismatch", "E0277"),
        ("conv2d_invalid_shape", "E0277"),
        ("forward_batchnorm_mismatch", "E0277"),
        ("forward_broadcast_mismatch", "E0277"),
        ("forward_conv1d_static_mismatch", "E0277"),
        ("forward_conv2d_static_mismatch", "E0277"),
        ("forward_linear_partial_mismatch", "E0277"),
        ("forward_linear_static_mismatch", "E0277"),
        ("forward_model_building_mismatch", "E0277"),
        ("kernel_conv2d_channel_mismatch", "E0277"),
        ("layer_builder_invalid_count", "E0277"),
        ("layer_builder_invalid_order", "E0277"),
        ("matmul_static_mismatch", "E0277"),
        ("named_dim_concat_mismatch", "E0277"),
        ("reshape_static_mismatch", "E0277"),
        // Shape rules the compiler reports as a type mismatch rather than an
        // unmet bound, because the shape appears in the signature directly.
        ("device_mismatch", "E0308"),
        ("dtype_mismatch", "E0308"),
        ("loss_cross_entropy_mismatch", "E0308"),
        ("named_dim_identity_mismatch", "E0308"),
        ("named_dim_size_mismatch", "E0308"),
        ("shape_mismatch", "E0308"),
        ("stack_static_mismatch", "E0308"),
        // Proof lowering: the seal around `Validated` and the descriptor
        // taxonomy (`EXE-002`), and the frontend binding (`EXE-003`).
        ("operation_spec_is_sealed", "E0277"),
        ("shape_rule_needs_the_frontend_proof", "E0277"),
        ("validated_fields_are_private", "E0451"),
        ("validated_new_is_crate_private", "E0624"),
        // Macros, which emit their own diagnostics and carry no error code.
        ("macro_idx_invalid", "expected `..=`"),
        (
            "macro_module_invalid",
            "unknown attribute argument for #[module]",
        ),
        ("macro_s_invalid", "expected identifier"),
    ])
}

/// Output that means the case failed on its own scaffolding.
///
/// Every entry here was found in this directory: a `typenum::typenum::U2`
/// import that never resolved, a file importing `incin_core` twice, and a path
/// through `incin_core::shapes`, which is `pub(crate)`. Each produced a
/// confident red test that proved nothing about shapes.
const SCAFFOLDING_FAILURES: &[(&str, &str)] = &[
    ("E0432", "an import that does not resolve"),
    ("E0254", "a name imported twice"),
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
                "{stem}: has a row in expected_reasons() but no case file"
            ));
        }
    }

    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
