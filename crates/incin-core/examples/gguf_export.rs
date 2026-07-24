extern crate incin_core as incin;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::io::{GgufExporter, GgufValue, QuantScheme};
use incin_core::prelude::*;
use std::path::Path;

fn main() -> Result<()> {
    println!("🚀 Incin GGUF Export Example");

    let layer = Linear::<s![32, 64], CpuBackendImpl>::build(())?;
    let path = Path::new("example_model_q8_0.gguf");

    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::Q8_0)
        .with_metadata_entry("llama.context_length", GgufValue::Uint32(2048))
        .with_metadata_entry("llama.embedding_length", GgufValue::Uint32(64))
        .save(path)?;

    println!(
        "✅ Successfully exported model to GGUF format: {}",
        path.display()
    );

    let _ = std::fs::remove_file(path);

    Ok(())
}
