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
    let executable: Vec<_> = OPERATION_CATALOG
        .iter()
        .filter(|row| row.site.is_backend_executable())
        .collect();
    let done = executable
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
         the stable tensor surface uses.\n\nThe denominator is the number of operations \
         that `Execute<Descriptor<O>>` can carry at all, not the whole catalog. An \
         operation whose `ExecutionSite` is not backend-executable is listed separately \
         with the reason: it is a gap in the execution trait rather than an unwritten \
         executor, and counting it here would describe work that cannot be done without \
         changing the contract first.\n\n",
    );
    let _ = writeln!(
        out,
        "**{done} of {} backend-executable operations migrated**, out of {total} catalog \
         operations in total. The remaining {} executable operations are still reachable \
         only through the legacy operation-family traits.\n",
        executable.len(),
        executable.len() - done
    );

    out.push_str("## Backend-executable operations\n\n");
    out.push_str("| Operation | Site | Migrated | Legacy source |\n|---|---|:--:|---|\n");
    for row in &executable {
        let _ = writeln!(
            out,
            "| `{}` | `{:?}` | {} | `{}` |",
            row.name,
            row.site,
            if migrated.contains(&row.operation) {
                "yes"
            } else {
                "no"
            },
            row.legacy_source,
        );
    }

    out.push_str(
        "\n## Operations the execution contract cannot carry\n\nThese are not pending \
         migrations. Each one needs a change to `Execute`/`ExecutionRequest` before an \
         executor for it could be written, and until then the stable tensor surface \
         reaches it through the legacy path by necessity rather than by omission.\n\n",
    );
    out.push_str("| Operation | Site | Why | Legacy source |\n|---|---|---|---|\n");
    for row in OPERATION_CATALOG
        .iter()
        .filter(|row| !row.site.is_backend_executable())
    {
        let _ = writeln!(
            out,
            "| `{}` | `{:?}` | {} | `{}` |",
            row.name,
            row.site,
            row.site
                .blocking_reason()
                .expect("a non-executable site states its reason"),
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

/// Nothing migrated may sit at a site the execution contract cannot carry.
///
/// A backend that advertises an exact identity has, by the compile-time proof
/// in `cpu::canonical`, an `Execute<Descriptor<op::X>>` for it. If the catalog
/// also says that operation's result cannot be carried by `Execute`, one of the
/// two is lying, and the classification is the more likely of the pair to be
/// wrong because it is the newer claim.
#[test]
fn no_migrated_operation_sits_at_a_non_executable_site() {
    for row in OPERATION_CATALOG {
        if !migrated().contains(&row.operation) {
            continue;
        }
        assert!(
            row.site.is_backend_executable(),
            "{} is advertised and therefore has an executor, but the catalog classifies \
             it as `{:?}`, which is documented as impossible to execute: {}",
            row.name,
            row.site,
            row.site.blocking_reason().unwrap_or("no reason recorded"),
        );
    }
}

/// The migration is incomplete, and the evidence must keep saying so.
///
/// This assertion is the opposite of the usual kind: it fails if the migrated
/// set ever covers every backend-executable operation while this test still
/// exists. That is intentional. Whoever completes FND-005 has to come here and
/// delete it, which forces the completion claim to be a deliberate edit rather
/// than a number that quietly crossed a threshold nobody was watching.
///
/// The bound is the executable subset, not the whole catalog. Against the whole
/// catalog this test could never fire, because thirteen operations cannot be
/// migrated without first changing `Execute`, so it would have guarded nothing.
#[test]
fn the_migration_is_recorded_as_incomplete() {
    let migrated = migrated();
    let executable: Vec<_> = OPERATION_CATALOG
        .iter()
        .filter(|row| row.site.is_backend_executable())
        .collect();
    let done = executable
        .iter()
        .filter(|row| migrated.contains(&row.operation))
        .count();
    assert!(
        done < executable.len(),
        "every backend-executable operation is now migrated ({done} of {}); FND-005's \
         migration step is complete, so delete this test and update \
         audit-evidence/FND-005/summary.md rather than leaving a stale partial claim \
         in the tree",
        executable.len()
    );
}
