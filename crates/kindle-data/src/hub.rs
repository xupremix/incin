use hf_hub::api::sync::Api;
use kindle_core::prelude::*;
use std::path::PathBuf;

/// Helper to ergonomically download and manage weights from the HuggingFace Hub.
pub struct HubApi {
    inner: Api,
}

impl HubApi {
    /// Creates a new Hub API instance.
    /// Can be configured via `KINDLE_HUB_CACHE_DIR` and `KINDLE_HUB_TOKEN` environment variables.
    pub fn new() -> Result<Self> {
        let mut builder = hf_hub::api::sync::ApiBuilder::new();

        if let Ok(dir) = std::env::var("KINDLE_HUB_CACHE_DIR") {
            builder = builder.with_cache_dir(PathBuf::from(dir));
        }

        if let Ok(token) = std::env::var("KINDLE_HUB_TOKEN") {
            builder = builder.with_token(Some(token));
        }

        let api = builder.build().map_err(anyhow::Error::from)?;
        Ok(Self { inner: api })
    }

    /// Access a specific repository (model) on the Hub.
    pub fn model(&self, repo_id: &str) -> HubRepo {
        HubRepo {
            inner: self.inner.model(repo_id.to_string()),
        }
    }
}

/// Hub repo.
pub struct HubRepo {
    inner: hf_hub::api::sync::ApiRepo,
}

impl HubRepo {
    /// Downloads a specific file from the repository, returning its local path.
    /// If the file is already cached, it will return the cached path immediately.
    pub fn get(&self, filename: &str) -> Result<PathBuf> {
        let path = self.inner.get(filename).map_err(anyhow::Error::from)?;
        Ok(path)
    }

    /// Downloads `model.safetensors` (or specified filename) from HuggingFace Hub
    /// and loads the state tensors directly.
    pub fn load_safetensors<B: Backend<FloatElem = f32>>(
        &self,
        filename: Option<&str>,
        device: &DeviceId,
    ) -> Result<alloc::collections::BTreeMap<String, B::Storage<f32>>> {
        let file = filename.unwrap_or("model.safetensors");
        let path = self.get(file)?;
        kindle_core::prelude::load_safetensors::<B>(&path, device)
    }
}

/// Helper shortcut function to quickly download a file from a repository.
pub fn download(repo_id: &str, filename: &str) -> Result<PathBuf> {
    HubApi::new()?.model(repo_id).get(filename)
}

/// Downloads a `safetensors` model file from HuggingFace `repo_id` and loads it directly into a state map.
pub fn from_pretrained<B: Backend<FloatElem = f32>>(
    repo_id: &str,
    filename: Option<&str>,
    device: &DeviceId,
) -> Result<alloc::collections::BTreeMap<String, B::Storage<f32>>> {
    HubApi::new()?.model(repo_id).load_safetensors::<B>(filename, device)
}

