//! FND-005: a machine-checked account of how much of the catalog the CPU has
//! actually migrated onto the canonical descriptor contract.
//!
//! The risk this suite exists to remove is a migration that reads as finished
//! because the interesting parts are done. The inventory it generates is
//! derived from the CPU capability registrations rather than written by hand,
//! so an identity cannot be described as migrated unless the backend advertises
//! it, and the compile-time proof in `cpu::canonical` separately guarantees
//! that anything advertised has an executor. Together those two make the
//! generated document's "migrated" column impossible to overstate.
//!
//! Run `INCIN_DOCS=overwrite cargo test -p incin-backends --test
//! cpu_migration_status` to rewrite the document after migrating an operation.

#![cfg(feature = "std")]

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use incin_backends::capability::CPU_CAPABILITIES;
use incin_core::exec::OPERATION_CATALOG;
use incin_core::prelude::OperationKind;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("the workspace root is two levels above this crate")
}

/// Every exact identity the CPU backend advertises.
///
/// A broad family row is deliberately excluded: FND-004 made families unable to
/// satisfy an exact query, so counting one here would inflate the migrated set
/// with rows that answer no canonical question.
fn migrated() -> BTreeSet<OperationKind> {
    CPU_CAPABILITIES
        .iter()
        .map(|rule| rule.operation)
        .filter(|operation| operation.is_exact())
        .collect()
}

fn document() -> String {
    let migrated = migrated();
    let total = OPERATION_CATALOG.len();
    let done = OPERATION_CATALOG
        .iter()
        .filter(|row| migrated.contains(&row.operation))
        .count();

    let mut out = String::from(
        "# CPU canonical migration status\n\nGenerated from `CPU_CAPABILITIES` and \
         `incin_core::exec::OPERATION_CATALOG`; the Rust source is authoritative. \
         \"Migrated\" means the CPU backend advertises the exact identity and therefore, \
         by the compile-time proof in `cpu::canonical`, implements \
         `Execute<Descriptor<op::...>>` for it. It does not mean the operation is \
         unreachable through the legacy operation-family traits: those remain the path \
         the stable tensor surface uses.\n\n",
    );
    let _ = writeln!(
        out,
        "**{done} of {total} catalog operations migrated.** The remaining {} are still \
         reachable only through the legacy operation-family traits.\n",
        total - done
    );
    out.push_str("| Operation | Migrated | Legacy source |\n|---|:--:|---|\n");
    for row in OPERATION_CATALOG {
        let _ = writeln!(
            out,
            "| `{}` | {} | `{}` |",
            row.name,
            if migrated.contains(&row.operation) {
                "yes"
            } else {
                "no"
            },
            row.legacy_source,
        );
    }
    out
}

#[test]
fn the_migration_status_document_matches_the_registrations() {
    let path = repo_root()
        .join("audit-evidence")
        .join("FND-005")
        .join("cpu-migration-status.md");
    let expected = document();
    let actual = std::fs::read_to_string(&path).unwrap_or_default();
    if actual == expected {
        return;
    }
    if std::env::var("INCIN_DOCS").as_deref() == Ok("overwrite") {
        std::fs::create_dir_all(path.parent().expect("the path has a parent"))
            .expect("the evidence directory must be creatable");
        std::fs::write(&path, expected).expect("the generated file must be writable");
        return;
    }
    panic!(
        "{} is stale.\n\nRegenerate it with:\n    INCIN_DOCS=overwrite cargo test \
         -p incin-backends --test cpu_migration_status\n",
        path.display()
    );
}

/// Every migrated identity is a real catalog row.
///
/// A capability row naming an identity the catalog does not declare would be a
/// support claim for an operation with no defined semantics.
#[test]
fn every_migrated_identity_is_a_catalog_operation() {
    let declared: BTreeSet<OperationKind> =
        OPERATION_CATALOG.iter().map(|row| row.operation).collect();
    for operation in migrated() {
        assert!(
            declared.contains(&operation),
            "{operation} is advertised by the CPU backend but is not a catalog operation"
        );
    }
}

/// The migration is incomplete, and the evidence must keep saying so.
///
/// This assertion is the opposite of the usual kind: it fails if the migrated
/// set ever reaches the whole catalog while this test still exists. That is
/// intentional. Whoever completes FND-005 has to come here and delete it, which
/// forces the completion claim to be a deliberate edit rather than a number
/// that quietly crossed a threshold nobody was watching.
#[test]
fn the_migration_is_recorded_as_incomplete() {
    let done = migrated().len();
    let total = OPERATION_CATALOG.len();
    assert!(
        done < total,
        "every catalog operation is now migrated ({done} of {total}); FND-005 is \
         complete, so delete this test and update audit-evidence/FND-005/summary.md \
         rather than leaving a stale partial claim in the tree"
    );
}
