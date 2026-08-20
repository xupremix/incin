//! `docs/FROZEN_FOUNDATIONS.md` names files, and files get moved.
//!
//! The document's whole value is that a reader can follow it to the code that
//! actually holds each contract. A list of paths that no longer exist is worse
//! than no list at all, because it reads as authoritative. This suite extracts
//! every repository path the document mentions and checks it resolves.
//!
//! It deliberately does not check the prose. Whether a foundation is genuinely
//! finished is a judgement, and a test that pretended to verify it would be
//! claiming more than it can.

#![cfg(feature = "std")]

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the workspace root is two levels above this crate")
}

fn document() -> String {
    std::fs::read_to_string(repo_root().join("docs").join("FROZEN_FOUNDATIONS.md"))
        .expect("docs/FROZEN_FOUNDATIONS.md must exist")
}

/// Every path the document names in backticks, in source order.
///
/// A backticked span is treated as a path when it starts with a known top-level
/// directory of this repository. That keeps Rust items such as
/// `Execute<O>` out of the list without needing to parse them, and
/// it means a newly mentioned path is picked up without editing this test.
fn mentioned_paths(document: &str) -> Vec<String> {
    const ROOTS: [&str; 3] = ["crates/", "docs/", "audit-evidence/"];
    document
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|span| ROOTS.iter().any(|root| span.starts_with(root)))
        .map(str::to_owned)
        .collect()
}

#[test]
fn every_path_the_frozen_foundations_document_names_exists() {
    let document = document();
    let paths = mentioned_paths(&document);
    assert!(
        paths.len() >= 10,
        "the document named only {} paths, which suggests the extraction broke rather \
         than that the document shrank",
        paths.len()
    );
    let root = repo_root();
    for path in paths {
        assert!(
            root.join(&path).exists(),
            "docs/FROZEN_FOUNDATIONS.md points at {path}, which does not exist. Either the \
             file moved and the document needs updating, or the foundation it described was \
             deleted and the document is now describing something that is not there"
        );
    }
}

/// The two mechanisms the document leans on hardest must still be named as
/// such in the code.
///
/// These are the macros that make "advertised" and "declared" the same edit. If
/// either is renamed, the document's account of why the foundation holds stops
/// being checkable by a reader, and this catches it at the same commit rather
/// than at the next audit.
#[test]
fn the_proof_mechanisms_the_document_cites_are_still_present() {
    let root = repo_root();
    for (path, symbol) in [
        (
            "crates/incin-core/src/operation_catalog.rs",
            "macro_rules! incin_operation_catalog",
        ),
        (
            "crates/incin-backends/src/cpu/canonical/mod.rs",
            "macro_rules! assert_every_advertised_row_executes",
        ),
        (
            "crates/incin-backends/src/capability.rs",
            "macro_rules! cpu_descriptor_operations",
        ),
    ] {
        let source = std::fs::read_to_string(root.join(path))
            .unwrap_or_else(|_| panic!("{path} must be readable"));
        assert!(
            source.contains(symbol),
            "docs/FROZEN_FOUNDATIONS.md credits `{symbol}` in {path} with keeping a \
             foundation true, and it is no longer there"
        );
    }
}
