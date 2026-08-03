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
fn operation_semantics_matches_the_code_catalog() {
    let path = repo_root().join("docs").join("OPERATION_SEMANTICS.md");
    let expected = incin_core::exec::operation_semantics_document();
    let actual = std::fs::read_to_string(&path).unwrap_or_default();
    if actual == expected {
        return;
    }
    if std::env::var("INCIN_DOCS").as_deref() == Ok("overwrite") {
        std::fs::write(path, expected).expect("generated semantics document is writable");
        return;
    }
    panic!(
        "{} is stale; regenerate with `INCIN_DOCS=overwrite cargo test -p incin-core --test generated_operation_semantics`",
        path.display()
    );
}
