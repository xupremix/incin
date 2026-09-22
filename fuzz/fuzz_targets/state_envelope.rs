//! State-loader envelope fuzz target, issue #48.
//!
//! The threat model names malformed checkpoints as adversarial input. The
//! bytes go to `ModelExt::load(Format::Postcard, ..)` on a small CPU module
//! -- the same file-path entry point a restore takes -- so the envelope
//! parser, its schema-version gate, and the dtype/shape checks that follow
//! must all refuse a malformed file rather than index past it or panic.
#![no_main]

use std::path::PathBuf;

use incin_core::nn::linear::linear;
use incin_core::prelude::*;
use incin_core::tensor::device::Cpu;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(MAX_INPUT)];
    let path = scratch("state-envelope", "incin_meta");
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let mut module = linear(shape![2, 2])
        .init(&Cpu)
        .expect("a 2x2 linear initialises");
    let _ = module.load(Format::Postcard, &path);
});

/// One scratch path per process; each iteration rewrites it in place.
fn scratch(tag: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "incin-fuzz-{tag}-{}.{extension}",
        std::process::id()
    ))
}
