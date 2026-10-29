//! Example: export a layer to GGUF and re-read the file with `inspect_file`.
//!
//! Incin has no GGUF weight importer: this is an export-then-inspect round
//! trip over the file's headers and metadata, the same read path
//! `cargo incin inspect model.gguf` uses. Payload correctness against an
//! independent GGUF writer is covered by `tests/export_test.rs`.
extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::io::{GgufExporter, QuantScheme, inspect_file};
use incin_core::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    // The weight's last (fastest-varying) dimension is 64, a multiple of
    // the GGUF row width, so it quantizes to Q4_0. The rank-1 bias row has
    // width 33 and falls back to F32 — one file, mixed tensor headers.
    let layer = Linear::<s![64, 33], CpuBackendImpl>::build(())?;
    let path: PathBuf = std::env::temp_dir().join("incin_gguf_roundtrip.gguf");

    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::W4A16_Q4_0)
        .save(&path)?;

    let info = inspect_file(&path)?;
    println!(
        "round trip: {} ({} bytes), {} tensor(s)",
        info.format, info.file_size_bytes, info.tensor_count
    );
    for tensor in &info.tensors {
        println!(
            "  {:<8} shape={:?} {} bytes",
            tensor.dtype, tensor.shape, tensor.size_bytes
        );
    }

    assert!(info.format.starts_with("GGUF v"));
    assert_eq!(info.tensor_count, 2);
    assert!(info.tensors.iter().any(|tensor| tensor.dtype == "Q4_0"));
    assert!(info.tensors.iter().any(|tensor| tensor.dtype == "F32"));

    let _ = std::fs::remove_file(&path);
    Ok(())
}
