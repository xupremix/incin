extern crate incin_core as incin;
use incin_core::io::{GgufExporter, QuantScheme, inspect_file};
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use std::path::Path;

fn main() -> Result<()> {
    println!("🔍 Incin Model File Inspector Example");

    let layer = Linear::<s![16, 32], DummyBackend<f32, Cpu>>::build(())?;
    let path = Path::new("inspect_demo.gguf");

    GgufExporter::from_module(&layer)
        .with_quantization(QuantScheme::Q8_0)
        .save(path)?;

    let info = inspect_file(path)?;

    println!("--------------------------------------------------");
    println!("Format      : {}", info.format);
    println!("File Path   : {}", info.path);
    println!("Size        : {} bytes", info.file_size_bytes);
    println!("Tensor Count: {}", info.tensor_count);
    println!("--------------------------------------------------");

    let _ = std::fs::remove_file(path);

    Ok(())
}
