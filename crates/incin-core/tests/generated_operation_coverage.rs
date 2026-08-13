#![cfg(feature = "std")]

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root")
}

#[test]
fn operation_coverage_matches_generated_document() {
    let path = repo_root().join("docs").join("operation-coverage.md");
    let expected = incin_core::exec::operation_coverage_document();
    let actual = std::fs::read_to_string(&path).unwrap_or_default();
    if actual == expected {
        return;
    }
    if std::env::var("INCIN_DOCS").as_deref() == Ok("overwrite") {
        std::fs::write(path, expected).expect("generated operation coverage is writable");
        return;
    }
    panic!(
        "{} is stale; regenerate with `INCIN_DOCS=overwrite cargo test -p incin-core --test generated_operation_coverage`",
        path.display()
    );
}

#[test]
fn operation_coverage_has_one_row_for_each_execution_site() {
    let coverage = incin_core::exec::operation_coverage();
    assert_eq!(coverage.by_site.len(), 7);
    assert_eq!(
        coverage
            .by_site
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>(),
        coverage.canonical
    );
}
