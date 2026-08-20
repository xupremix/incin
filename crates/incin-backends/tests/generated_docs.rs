//! `UX-013`: the checked-in support tables are the ones the registrations
//! generate, and every public example is compiled.
//!
//! §2.10 says not to maintain handwritten support tables that can drift from
//! code. Generating them is only half of that - a generated file that nobody
//! regenerates has drifted just as far as a handwritten one, and reads as
//! authoritative while doing it. This suite is the other half: it fails when the
//! checked-in text and the renderer disagree, so the drift cannot be committed.
//!
//! Run `INCIN_DOCS=overwrite cargo test -p incin-backends --test generated_docs`
//! to rewrite the generated files after changing a capability rule.

#![cfg(feature = "std")]

use std::path::{Path, PathBuf};

use incin_backends::capability_docs;

/// The repository root, from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the workspace root is two levels above this crate")
}

/// Compares `path` against `expected`, or rewrites it under `INCIN_DOCS`.
///
/// The rewrite path is deliberately the same code the assertion uses. A blessing
/// mode that writes something the check would not accept is worse than no
/// blessing mode, because it turns a red suite green without making the claim
/// true.
fn assert_generated(path: &Path, expected: &str) {
    let actual = std::fs::read_to_string(path).unwrap_or_default();
    if actual == expected {
        return;
    }
    if std::env::var("INCIN_DOCS").as_deref() == Ok("overwrite") {
        std::fs::write(path, expected).expect("the generated file must be writable");
        return;
    }
    panic!(
        "{} is stale.\n\nRegenerate it with:\n    INCIN_DOCS=overwrite cargo test \
         -p incin-backends --test generated_docs\n",
        path.display()
    );
}

#[test]
fn the_capability_document_matches_the_registrations() {
    assert_generated(
        &repo_root().join("docs").join("capabilities.md"),
        &capability_docs::document(),
    );
}

/// The summary table is generated, so the risk is not that it is wrong but that
/// it is empty and still renders as a well-formed table. This asserts it says
/// something.
#[test]
fn the_document_covers_every_registered_operation() {
    let document = capability_docs::document();
    for rules in [
        incin_backends::capability::CPU_CAPABILITIES,
        incin_backends::capability::CUDA_CAPABILITIES,
        incin_backends::capability::WGPU_CAPABILITIES,
    ] {
        assert!(!rules.is_empty());
        for rule in rules {
            assert!(
                document.contains(&format!("`{}`", rule.operation.name())),
                "{} is registered but absent from the generated document",
                rule.operation.name()
            );
            for dtype in rule.dtypes {
                assert!(
                    document.contains(&format!("`{}`", dtype.name())),
                    "{} is registered for {} but absent from the generated document",
                    dtype.name(),
                    rule.operation.name()
                );
            }
        }
    }
}

/// An operation no backend registers must not appear as a supported row.
///
/// Without this the summary table could be generated from `OperationKind`'s
/// variants rather than from the registrations, and would then advertise
/// everything the taxonomy can name.
#[test]
fn the_document_does_not_advertise_an_unregistered_operation() {
    use incin_core::prelude::OperationKind;

    let document = capability_docs::document();
    let registered = |operation: OperationKind| {
        [
            incin_backends::capability::CPU_CAPABILITIES,
            incin_backends::capability::CUDA_CAPABILITIES,
            incin_backends::capability::WGPU_CAPABILITIES,
        ]
        .iter()
        .any(|rules| rules.iter().any(|rule| rule.operation == operation))
    };

    // `Permute` is in the taxonomy and registered by nobody, which makes it the
    // control: a generator driven by the enum would list it.
    assert!(!registered(OperationKind::Permute));
    assert!(
        !document.contains("| `permute` |"),
        "the summary table lists an operation no backend registered"
    );
}

/// §2.10: "Every public example is compiled in the minimum documented feature
/// set."
///
/// A ` ```ignore ` fence is a Rust example the build was told not to check, so
/// it documents whatever the API used to be. This crate's workspace had 70 of
/// them before `UX-013` and `cargo test --workspace --doc` reported success
/// while compiling nine examples out of seventy-nine.
///
/// A snippet that genuinely cannot compile where it lives - the invocation form
/// of a `pub(crate)` macro, a `model!` that opens a file at build time - is
/// fenced ` ```text `, which says it is shown rather than run and, unlike
/// `ignore`, does not claim to be a checked example.
#[test]
fn no_public_example_is_excluded_from_compilation() {
    let mut offenders = Vec::new();
    let crates = repo_root().join("crates");
    let mut stack = vec![crates];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("crates/ is readable") {
            let path = entry.expect("a readable directory entry").path();
            if path.is_dir() {
                // Only `src`: `tests/compile_fail` holds deliberately
                // uncompilable inputs, which is a different mechanism.
                if path.file_name().is_some_and(|name| name != "target") {
                    stack.push(path);
                }
                continue;
            }
            if path.extension().is_some_and(|ext| ext == "rs")
                && path.components().any(|c| c.as_os_str() == "src")
            {
                let text = std::fs::read_to_string(&path).expect("a readable source file");
                for (number, line) in text.lines().enumerate() {
                    let trimmed = line.trim_start();
                    let Some(rest) = trimmed
                        .strip_prefix("///")
                        .or_else(|| trimmed.strip_prefix("//!"))
                    else {
                        continue;
                    };
                    let fence = rest.trim();
                    if fence == "```ignore" || fence == "```rust,ignore" {
                        offenders.push(format!("{}:{}", path.display(), number + 1));
                    }
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these examples are excluded from compilation; make them compile, or \
         fence them ```text if they cannot:\n  {}",
        offenders.join("\n  ")
    );
}
