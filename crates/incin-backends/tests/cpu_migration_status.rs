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

/// Why each still-unmigrated backend-executable operation has no executor.
///
/// Every entry here is a limit of the descriptor or capability contract rather
/// than an unwritten function, and each was established by reading the
/// declaration that blocks it rather than by trying and giving up. Two tests
/// below hold this honest from both directions: an operation that is unmigrated
/// and unlisted fails, and so does one that is listed and migrated.
///
/// The point is that "seven left" and "seven executors left to write" are very
/// different claims, and only the first is true.
fn blocking_reason(operation: OperationKind) -> Option<&'static str> {
    Some(match operation {
        OperationKind::Sample => {
            "`DistributionAttributes` names its distribution as a string and its \
             parameters as bytes. Executing one needs a registry that maps that pair \
             back to a sampler, and no such registry exists"
        }
        OperationKind::Rnn | OperationKind::Lstm => {
            "the descriptor carries no weights. Its operand arity admits an input and \
             the recurrent states only, and `RecurrentAttributes` holds sizes and \
             bias-presence flags, so the matrices the recurrence multiplies by cannot \
             be named"
        }
        _ => return None,
    })
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
         `Execute<op::...>` for it. The legacy operation-family traits remain only as \
         backend-local adapters for special kernels and compatibility tests; they are not \
         the stable tensor execution path.\n\nThe denominator is the number of operations \
         that `Execute<O>` can carry at all, not the whole catalog. An \
         operation whose `ExecutionSite` is not backend-executable is listed separately \
         with the reason: it is a gap in the execution trait rather than an unwritten \
         executor, and counting it here would describe work that cannot be done without \
         changing the contract first.\n\n",
    );
    let _ = writeln!(
        out,
        "**{done} of {} backend-executable operations migrated**, out of {total} catalog \
         operations in total.\n",
        executable.len(),
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
        "\n## Why the rest have no executor\n\nNone of these is an unwritten function. \
         Each names a limit of the descriptor or capability contract that has to change \
         before an executor for it could be written at all, so the remaining count and \
         the remaining work are not the same number.\n\n",
    );
    out.push_str("| Operation | What blocks it |\n|---|---|\n");
    for row in &executable {
        if migrated.contains(&row.operation) {
            continue;
        }
        let _ = writeln!(
            out,
            "| `{}` | {} |",
            row.name,
            blocking_reason(row.operation).unwrap_or("not recorded"),
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
/// in `cpu::canonical`, an `Execute<op::X>` for it. If the catalog
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

/// Every unmigrated backend-executable operation states what blocks it.
///
/// Without this, "unmigrated" collapses two different things: an executor
/// nobody has written, and one nobody can write yet. The second is the whole
/// remaining set, and a reader who could not tell them apart would plan work
/// that cannot be started.
#[test]
fn every_unmigrated_executable_operation_records_what_blocks_it() {
    let migrated = migrated();
    for row in OPERATION_CATALOG {
        if !row.site.is_backend_executable() || migrated.contains(&row.operation) {
            continue;
        }
        assert!(
            blocking_reason(row.operation).is_some(),
            "{} is backend-executable and unmigrated, but nothing here says why. Either \
             write its executor or record what stops you, so the remaining work stays \
             countable",
            row.name
        );
    }
}

/// A recorded reason must not outlive the thing it explains.
///
/// The obligation above only bites in one direction. This is the other: once an
/// operation is migrated, its entry is a stale claim that the contract still
/// blocks something it demonstrably does not, and stale claims in this file are
/// exactly what it exists to prevent.
#[test]
fn no_migrated_operation_still_claims_to_be_blocked() {
    for operation in migrated() {
        assert!(
            blocking_reason(operation).is_none(),
            "{operation} has an executor, so the reason recorded for why it cannot have \
             one is wrong. Delete it"
        );
    }
}
