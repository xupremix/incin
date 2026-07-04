use candle_core::safetensors::load;
use hf_hub::api::tokio::Api;
use kindle_backends::candle::CandleBackend;
use kindle_core::prelude::*;
use std::path::PathBuf;

/// Helper to easily download and load weights from HuggingFace Hub.
pub struct HuggingFaceHub;

impl HuggingFaceHub {
    /// Download a file from a HuggingFace repository and return its local path.
    pub async fn download(repo_id: &str, filename: &str) -> Result<PathBuf> {
        let api = Api::new().map_err(anyhow::Error::from)?;
        let repo = api.model(repo_id.to_string());

        let path = repo.get(filename).await.map_err(anyhow::Error::from)?;

        Ok(path)
    }

    /// Download and load a Safetensors file into a map of dynamic Kindle tensors.
    pub async fn load_safetensors(
        repo_id: &str,
        filename: &str,
        candle_device: &candle_core::Device,
    ) -> Result<std::collections::HashMap<String, Tensor<Dyn, CandleBackend<f32, Cpu>>>> {
        let path = Self::download(repo_id, filename).await?;

        let loaded = load(&path, candle_device).map_err(anyhow::Error::from)?;

        let mut result = std::collections::HashMap::new();

        for (name, tensor) in loaded {
            let dims = tensor.dims().to_vec();
            // Wrap in our dynamic Tensor
            let kindle_tensor = Tensor::<Dyn, CandleBackend<f32, Cpu>, Grad>::from_parts(
                tensor,
                dims,
                core::marker::PhantomData,
                core::marker::PhantomData,
                core::marker::PhantomData,
            );
            result.insert(name, kindle_tensor);
        }

        Ok(result)
    }
}
