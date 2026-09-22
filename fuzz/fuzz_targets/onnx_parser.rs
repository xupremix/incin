//! ONNX importer fuzz target, issue #48.
//!
//! The threat model names malformed ONNX as an adversarial input. This drives
//! the same file-path entry point a caller uses -- `OnnxImporter::import` --
//! with arbitrary bytes, so the contract under test is the real one: accept a
//! coherent model, return `Err` for everything else, never panic. Bounds are
//! fail-closed (`ResourceLimits`, prost's decode recursion cap); this target
//! exists to search past where the hand-written suites stop.
#![no_main]

use std::path::PathBuf;

use libfuzzer_sys::fuzz_target;

const MAX_INPUT: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(MAX_INPUT)];
    let path = scratch("onnx-parser", "onnx");
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let _ = incin_core::onnx::OnnxImporter::new(&path).import();
});

/// One scratch path per process; each iteration rewrites it in place, the way
/// `tests/onnx_fuzz.rs` shares a path across a sweep.
fn scratch(tag: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "incin-fuzz-{tag}-{}.{extension}",
        std::process::id()
    ))
}
