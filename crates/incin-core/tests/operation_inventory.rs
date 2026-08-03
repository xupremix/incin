//! Machine-checked proof that the canonical catalog covers the operation
//! surface it claims to.
//!
//! The reviewed prose in `audit-evidence/FND-004/` records *why* each mapping is
//! correct. This file records *that* the mapping is still complete, by reading
//! the legacy operation-family traits out of the source that defines them and
//! comparing them against `OPERATION_CATALOG`. Adding a trait method without a
//! catalog row, or renaming one out from under a row, fails here rather than in
//! a later review.

#![cfg(feature = "std")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use incin_core::exec::OPERATION_CATALOG;

/// The operation-family traits FND-005 migrates onto the descriptor contract.
///
/// `Backend` and `StorageBackend` are deliberately absent: they carry identity
/// and lifecycle, not semantic operations, and FND-005 keeps them that way.
const OPERATION_TRAITS: &[&str] = &[
    "CreationOps",
    "FloatOps",
    "NumericOps",
    "TensorOps",
    "ReductionOps",
    "ModuleOps",
    "LossOps",
    "QuantizedOps",
    "OptimizerOps",
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root")
}

/// Extract `Trait::method` for every method declared in the operation-family
/// traits, by brace-matching each trait body.
fn declared_trait_methods(source: &str) -> BTreeMap<String, Vec<String>> {
    let mut declared = BTreeMap::new();
    for name in OPERATION_TRAITS {
        let needle = format!("pub trait {name}");
        let start = source
            .find(&needle)
            .unwrap_or_else(|| panic!("{name} is no longer declared in tensor/backend.rs"));
        let open = start + source[start..].find('{').expect("trait body");
        let mut depth = 0usize;
        let mut end = open;
        for (offset, byte) in source[open..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + offset;
                        break;
                    }
                }
                _ => {}
            }
        }

        // Methods are declared at one level of indentation inside the trait;
        // anything deeper belongs to a default body, not the contract.
        let methods: Vec<String> = source[open..end]
            .lines()
            .filter_map(|line| line.strip_prefix("    fn "))
            .filter_map(|rest| {
                rest.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .filter(|token| !token.is_empty())
                    .map(str::to_owned)
            })
            .collect();
        assert!(!methods.is_empty(), "{name} declares no methods");
        declared.insert((*name).to_owned(), methods);
    }
    declared
}

/// Catalog rows whose `legacy_source` names one of the operation-family traits.
fn catalog_trait_sources() -> BTreeMap<String, &'static str> {
    let mut sources = BTreeMap::new();
    for row in OPERATION_CATALOG {
        let Some((trait_name, _)) = row.legacy_source.split_once("::") else {
            continue;
        };
        if !OPERATION_TRAITS.contains(&trait_name) {
            continue;
        }
        let previous = sources.insert(row.legacy_source.to_owned(), row.name);
        assert!(
            previous.is_none(),
            "{} is claimed by both `{}` and `{}`; an alias must reuse an existing identity rather \
             than duplicate its legacy mapping",
            row.legacy_source,
            previous.unwrap_or_default(),
            row.name,
        );
    }
    sources
}

#[test]
fn every_legacy_operation_method_has_exactly_one_catalog_row() {
    let source =
        std::fs::read_to_string(repo_root().join("crates/incin-core/src/tensor/backend.rs"))
            .expect("tensor/backend.rs is readable");
    let declared = declared_trait_methods(&source);
    let mapped = catalog_trait_sources();

    let mut missing = Vec::new();
    let mut total = 0usize;
    for (trait_name, methods) in &declared {
        for method in methods {
            total += 1;
            let key = format!("{trait_name}::{method}");
            if !mapped.contains_key(&key) {
                missing.push(key);
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} operation-family methods have no canonical catalog row: {missing:#?}\n\
         Add the exact identity to `operation_catalog.rs` rather than letting the method execute \
         under a broad family.",
        missing.len(),
    );

    // The reverse direction: a row must not claim a method that no longer
    // exists, which is how a rename silently orphans a descriptor.
    let declared_keys: BTreeSet<String> = declared
        .iter()
        .flat_map(|(trait_name, methods)| {
            methods
                .iter()
                .map(move |method| format!("{trait_name}::{method}"))
        })
        .collect();
    let orphaned: Vec<&String> = mapped
        .keys()
        .filter(|key| !declared_keys.contains(*key))
        .collect();
    assert!(
        orphaned.is_empty(),
        "catalog rows name operation-family methods that no longer exist: {orphaned:#?}",
    );

    assert_eq!(
        total,
        mapped.len(),
        "the legacy operation surface and its catalog mapping have diverged",
    );
}

/// Every stable operation family the run promised to cover is represented.
///
/// A catalog that silently dropped, say, every loss or every optimizer entry
/// point would still satisfy the per-method check above if the traits were
/// edited in the same commit. This pins the families themselves.
#[test]
fn every_stable_operation_family_is_represented() {
    use incin_core::exec::catalog::SemanticProfile;

    let mut seen: BTreeSet<&'static str> = BTreeSet::new();
    for row in OPERATION_CATALOG {
        seen.insert(match row.profile {
            SemanticProfile::Creation => "creation",
            SemanticProfile::Transfer => "transfer",
            SemanticProfile::Autograd => "autograd",
            SemanticProfile::Module | SemanticProfile::Composite => "module",
            SemanticProfile::Loss => "loss",
            SemanticProfile::Optimizer => "optimizer",
            SemanticProfile::Quantized => "quantized",
            SemanticProfile::Reduction | SemanticProfile::IndexReduction => "reduction",
            _ => "tensor",
        });
    }

    for family in [
        "creation",
        "transfer",
        "autograd",
        "module",
        "loss",
        "optimizer",
        "quantized",
        "reduction",
        "tensor",
    ] {
        assert!(seen.contains(family), "no catalog row covers {family}");
    }
}

/// No exact operation may be represented only by a broad family identity.
#[test]
fn no_catalog_row_is_a_broad_family_identity() {
    for row in OPERATION_CATALOG {
        assert!(
            row.operation.is_exact(),
            "{} is a broad family, not an executable identity",
            row.operation,
        );
        assert_ne!(
            row.operation, row.family,
            "{} collapses its own family into its identity",
            row.operation,
        );
    }
}

/// Regenerate (or verify) the reviewed inventory document.
///
/// Run with `INCIN_DOCS=overwrite` to refresh it after a catalog change.
#[test]
fn operation_inventory_document_matches_the_catalog() {
    let path = repo_root().join("audit-evidence/FND-004/operation-inventory.md");
    let expected = render_inventory();
    let actual = std::fs::read_to_string(&path).unwrap_or_default();
    if actual == expected {
        return;
    }
    if std::env::var("INCIN_DOCS").as_deref() == Ok("overwrite") {
        std::fs::create_dir_all(path.parent().expect("evidence directory")).expect("mkdir");
        std::fs::write(&path, expected).expect("inventory document is writable");
        return;
    }
    panic!(
        "{} is stale; regenerate with `INCIN_DOCS=overwrite cargo test -p incin-core --test \
         operation_inventory`",
        path.display(),
    );
}

fn render_inventory() -> String {
    use std::fmt::Write as _;

    let mut out = String::from(
        "# FND-004 canonical operation inventory\n\n\
         Generated by `cargo test -p incin-core --test operation_inventory`. The Rust catalog in \
         `crates/incin-core/src/operation_catalog.rs` is authoritative; this file is a review \
         artifact and is verified against the catalog on every test run.\n\n\
         Each row is one exact executable identity. Families classify rows and never imply backend \
         support.\n\n",
    );

    let _ = writeln!(out, "Total exact operations: {}\n", OPERATION_CATALOG.len());

    let mut by_source: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for row in OPERATION_CATALOG {
        by_source
            .entry(row.legacy_source)
            .or_default()
            .push(row.name);
    }
    let trait_rows = OPERATION_CATALOG
        .iter()
        .filter(|row| {
            row.legacy_source
                .split_once("::")
                .is_some_and(|(name, _)| OPERATION_TRAITS.contains(&name))
        })
        .count();
    let _ = writeln!(
        out,
        "Legacy operation-family trait methods with a reviewed descriptor mapping: {trait_rows}\n",
    );

    let _ = writeln!(
        out,
        "| Exact identity | Family | Profile | Descriptor | Attributes | Inputs | Outputs | \
         Legacy source |",
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|");
    for row in OPERATION_CATALOG {
        let arity = |range: &core::ops::RangeInclusive<usize>| {
            if *range.end() == usize::MAX {
                format!("{}–many", range.start())
            } else {
                format!("{}–{}", range.start(), range.end())
            }
        };
        let _ = writeln!(
            out,
            "| `{}` | `{:?}` | `{:?}` | `{}` | `{}` | {} | {} | `{}` |",
            row.name,
            row.family,
            row.profile,
            row.descriptor,
            row.attributes,
            arity(&row.input_arity),
            arity(&row.output_arity),
            row.legacy_source,
        );
    }
    out
}
