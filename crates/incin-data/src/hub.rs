use hf_hub::api::sync::Api;
use incin_core::error::Result;
use incin_core::nn::save::load_foreign_safetensors_snapshot;
use incin_core::nn::state::StateSnapshot;
use std::path::PathBuf;

/// Helper to ergonomically download and manage weights from the HuggingFace Hub.
pub struct HubApi {
    inner: Api,
}

impl HubApi {
    /// Creates a new Hub API instance.
    /// Can be configured via `INCIN_HUB_CACHE_DIR` and `INCIN_HUB_TOKEN` environment variables.
    pub fn new() -> Result<Self> {
        let mut builder = hf_hub::api::sync::ApiBuilder::new();

        if let Ok(dir) = std::env::var("INCIN_HUB_CACHE_DIR") {
            builder = builder.with_cache_dir(PathBuf::from(dir));
        }

        if let Ok(token) = std::env::var("INCIN_HUB_TOKEN") {
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

    /// Access a specific dataset repository on the Hub.
    ///
    /// Datasets and models are separate repository kinds on the Hub (distinct
    /// URL namespaces and API endpoints), even though the underlying file
    /// operations are identical. [`HubRepo::get`] downloads any named file
    /// from either kind; this method exists so a dataset repo ID resolves
    /// through the correct namespace rather than being silently misrouted
    /// through `model()`.
    pub fn dataset(&self, repo_id: &str) -> HubRepo {
        HubRepo {
            inner: self.inner.dataset(repo_id.to_string()),
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
    ///
    /// A Hub file was written by whatever produced the repository, never by
    /// an incin build, so it carries no `incin.format.version` key; this
    /// loads it through [`load_foreign_safetensors_snapshot`], which does
    /// not require one, rather than the stricter incin-own-format loader.
    pub fn load_safetensors(&self, filename: Option<&str>) -> Result<StateSnapshot> {
        let file = filename.unwrap_or("model.safetensors");
        let path = self.get(file)?;
        load_foreign_safetensors_snapshot(&path)
    }
}

/// Helper shortcut function to quickly download a file from a model repository.
pub fn download(repo_id: &str, filename: &str) -> Result<PathBuf> {
    HubApi::new()?.model(repo_id).get(filename)
}

/// Helper shortcut function to quickly download a file from a dataset repository.
pub fn download_dataset(repo_id: &str, filename: &str) -> Result<PathBuf> {
    HubApi::new()?.dataset(repo_id).get(filename)
}

/// Downloads a `safetensors` model file from HuggingFace `repo_id` and loads it directly into a state map.
pub fn from_pretrained(repo_id: &str, filename: Option<&str>) -> Result<StateSnapshot> {
    HubApi::new()?.model(repo_id).load_safetensors(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `HubApi::new` must not require network access: `ApiBuilder::build`
    /// only resolves cache-directory and token configuration, it does not
    /// contact the Hub. If this ever starts making a request (e.g. an
    /// eager whoami call), it stops being safe to construct in a
    /// non-network context and every caller of `HubApi::new` needs to know.
    #[test]
    fn hub_api_constructs_offline_with_env_configuration() {
        let tmp = std::env::temp_dir().join(format!(
            "incin-hub-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // SAFETY-equivalent: std::env::set_var is unsafe in this edition
        // because it is process-global and races with other threads reading
        // the environment; this test owns the variable for its duration and
        // restores it, and #[test] harness invocations of this specific test
        // do not run this env var concurrently with another reader of it.
        unsafe {
            std::env::set_var("INCIN_HUB_CACHE_DIR", &tmp);
        }
        let api = HubApi::new();
        unsafe {
            std::env::remove_var("INCIN_HUB_CACHE_DIR");
        }
        assert!(
            api.is_ok(),
            "HubApi::new() must succeed offline when only cache-dir configuration is supplied"
        );
    }

    /// `model()`/`dataset()` must resolve to the correct Hub namespace and
    /// must not themselves make a network call - only `HubRepo::get`
    /// (exercised by the `#[ignore]`d network tests below) touches the
    /// network. Constructing a `HubRepo` for a repo that doesn't exist must
    /// still succeed; only a subsequent `get` can fail.
    #[test]
    fn model_and_dataset_repo_handles_construct_offline() {
        let api = HubApi::new().expect("offline construction");
        let _model_repo = api.model("incin/does-not-need-to-exist-for-this-test");
        let _dataset_repo = api.dataset("incin/does-not-need-to-exist-for-this-test");
        // No assertion beyond "did not panic and did not touch the network":
        // ApiRepo construction is pure string/path composition in `hf-hub`.
    }

    /// End-to-end proof that `HubApi`/`HubRepo::get`/`load_safetensors`
    /// actually work against the real Hub, not just that they compile.
    /// `hf-internal-testing/tiny-random-gpt2` is a stable, widely-used
    /// tiny fixture repo in the ML ecosystem (small `model.safetensors`,
    /// maintained specifically for library test suites like this one).
    ///
    /// Combined into one test rather than split across several: every
    /// assertion here shares one `INCIN_HUB_CACHE_DIR` mutation, and
    /// `cargo test` runs tests in parallel threads by default, so two
    /// separate tests each setting/reading/clearing the same process-global
    /// env var race each other (this shipped broken that way once already:
    /// two split tests intermittently read each other's cache directory).
    /// One test means one thread owns the mutation for its whole duration.
    ///
    /// Ignored by default so the workspace test suite stays network-free
    /// and deterministic; run explicitly with `--ignored` to verify.
    #[test]
    #[ignore = "requires network access to huggingface.co"]
    fn hub_client_works_end_to_end_against_the_real_hub() {
        let tmp = std::env::temp_dir().join(format!("incin-hub-live-test-{}", std::process::id()));
        unsafe {
            std::env::set_var("INCIN_HUB_CACHE_DIR", &tmp);
        }

        let first = download("hf-internal-testing/tiny-random-gpt2", "config.json");
        let second = download("hf-internal-testing/tiny-random-gpt2", "config.json");
        let snapshot = from_pretrained("hf-internal-testing/tiny-random-gpt2", None);

        unsafe {
            std::env::remove_var("INCIN_HUB_CACHE_DIR");
        }

        let first = first.expect("first download must succeed");
        let second = second.expect("second (cached) download must succeed");
        assert_eq!(
            first, second,
            "cached download must resolve to the same path"
        );
        assert!(first.exists(), "downloaded file must exist on disk");

        let snapshot = snapshot.expect(
            "loading a real (unversioned, foreign) Hub safetensors file must succeed — \
             this is the exact case load_foreign_safetensors_snapshot exists for",
        );
        assert!(
            !snapshot.is_empty(),
            "a real model's safetensors file must yield at least one state entry"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
