//! GGUF reader fuzz target, issue #48.
//!
//! The threat model names malformed GGUF as adversarial input. `inspect_file`
//! is the file-path entry point that reads the header, tensor directory, and
//! metadata kv pairs before any payload is trusted; arbitrary bytes must come
//! back as `Err`, never as a panic or an out-of-bounds read.
#![no_main]

use std::path::PathBuf;

use incin_core::io::inspect_file;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(MAX_INPUT)];
    let path = scratch("gguf-reader", "gguf");
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let _ = inspect_file(&path);
});

/// One scratch path per process; each iteration rewrites it in place.
fn scratch(tag: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "incin-fuzz-{tag}-{}.{extension}",
        std::process::id()
    ))
}
