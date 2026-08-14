use crate::err::Result;
use crate::nn::VisitState;
use crate::nn::save::save_safetensors;
use crate::tensor::backend::Backend;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::Path;

/// Exporter for Apple Silicon MLX ecosystem bundles.
pub struct MlxExporter;

impl MlxExporter {
    /// Exports a `incin` module to an MLX-compatible directory structure containing:
    /// - `weights.safetensors`: Safetensors binary model weights
    /// - `config.json`: Model architecture configuration
    pub fn export_dir<B: Backend + crate::tensor::backend::VariableBackend, M: VisitState<B>, P: AsRef<Path>>(
        module: &M,
        dir_path: P,
        config_json: &str,
    ) -> Result<()> {
        let dir = dir_path.as_ref();
        create_dir_all(dir)?;

        let weights_path = dir.join("weights.safetensors");
        save_safetensors::<B, M, _>(module, &weights_path)?;

        let config_path = dir.join("config.json");
        let mut config_file = File::create(config_path)?;
        config_file.write_all(config_json.as_bytes())?;
        config_file.flush()?;

        Ok(())
    }
}
