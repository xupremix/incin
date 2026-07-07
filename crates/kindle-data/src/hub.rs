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
}

/// Helper shortcut function to quickly download a file from a repository.
pub fn download(repo_id: &str, filename: &str) -> Result<PathBuf> {
    HubApi::new()?.model(repo_id).get(filename)
}
