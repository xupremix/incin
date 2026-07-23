extern crate kindle_core as kindle;
use kindle_backends::cpu::CpuBackendImpl;
use kindle_core::io::MlxExporter;
use kindle_core::prelude::*;
use std::path::Path;

fn main() -> Result<()> {
    println!("🍎 Kindle Apple MLX Export Example");

    let layer = Linear::<s![32, 64], CpuBackendImpl>::build(())?;
    let dir_path = Path::new("example_mlx_model");
    let config_json = r#"{
    "model_type": "linear",
    "hidden_size": 64,
    "input_size": 32
}"#;

    MlxExporter::export_dir::<CpuBackendImpl, _, _>(&layer, dir_path, config_json)?;

    println!(
        "✅ Successfully exported model to Apple MLX format in: {}",
        dir_path.display()
    );

    let _ = std::fs::remove_dir_all(dir_path);

    Ok(())
}
