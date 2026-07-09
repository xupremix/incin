# External Integrations

**Analysis Date:** 2026-07-09

## APIs & External Services

**Model Hub:**
- HuggingFace Hub - Downloading model weights/repos (safetensors, config files)
  - SDK/Client: `hf-hub` `0.5.0` (features: `tokio`, `rustls-tls`, `ureq`), wrapped by `HubApi`/`HubRepo` in `crates/kindle-data/src/hub.rs`
  - Auth: `KINDLE_HUB_TOKEN` env var (optional, for private/gated repos)
  - Cache location override: `KINDLE_HUB_CACHE_DIR` env var (defaults to `~/.cache/huggingface/hub`)
  - Usage: `HubApi::new()` builds an `hf_hub::api::sync::ApiBuilder`; `.model(repo_id).get(filename)` fetches/cache a file and returns its local `PathBuf`; convenience function `download(repo_id, filename)` at module root

**Generic Dataset/File Downloads:**
- Arbitrary HTTP(S) URLs - Used for downloading raw dataset files (e.g., vision datasets)
  - Client: `ureq` `3.3.0` (blocking HTTP), `crates/kindle-data/src/downloader.rs`
  - `Downloader::download(url, cache_dir, filename)` - fetches to a local cache path, skips re-download if file exists
  - `Downloader::download_and_extract_gz(...)` - downloads then gzip-decompresses via `flate2`
  - No authentication built in (plain GET requests)

## Data Storage

**Databases:**
- None detected — no SQL/NoSQL database client dependencies in any crate

**File Storage:**
- Local filesystem only. Model weights and datasets are cached under:
  - HuggingFace cache dir (`~/.cache/huggingface/hub` or `KINDLE_HUB_CACHE_DIR`)
  - Caller-specified `cache_dir` passed to `Downloader`
- Model weight formats read/written locally: `.safetensors` (via `safetensors` crate), `.onnx` (via generated protobuf structs from `crates/kindle-core/proto/onnx.proto`), plus a `.kindle_meta` JSON sidecar cache used by the `import_model!` macro to avoid re-parsing on every build

**Caching:**
- `.kindle_meta` JSON cache (build-time) — speeds up `import_model!` macro expansion by skipping full `.safetensors`/`.onnx` re-parsing unless `KINDLE_NO_META=1` is set
- HuggingFace Hub's own on-disk cache (via `hf-hub` crate), not a network cache service

## Authentication & Identity

**Auth Provider:**
- None for the library itself. The only "auth" concept is the HuggingFace Hub bearer token (`KINDLE_HUB_TOKEN`) passed through to `hf-hub`'s `ApiBuilder::with_token`.

## Monitoring & Observability

**Error Tracking:**
- None — no Sentry/Bugsnag or similar. Errors propagate as `anyhow::Result`/`anyhow::Error` and library-specific `thiserror`-derived error enums in `kindle-core`.

**Logs:**
- No structured logging framework (no `tracing`/`log` crate dependency detected in any `Cargo.toml`). Diagnostics are via `Result`/`anyhow::Error` and compiler-level diagnostics (proc-macro `syn::Error`/`compile_error!`) for the static-shape-verification failures.

## CI/CD & Deployment

**Hosting:**
- Not applicable — this is a Rust library workspace, not a hosted service. No Dockerfile, no cloud deployment config detected.

**CI Pipeline:**
- GitHub Actions, single workflow `.github/workflows/ci.yml` ("Kindle CI")
  - Triggers: `push`/`pull_request` on `main`
  - Steps: checkout → cache `~/.cargo/registry`, `~/.cargo/git`, `target` (keyed on `Cargo.lock` hash) → install Rust `stable` (`dtolnay/rust-toolchain@stable`) → `cargo fmt --all -- --check` → `cargo clippy --workspace --all-targets -- -D warnings` → `cargo test --workspace --all-targets` → `cargo build --examples --workspace`
  - Runs on `ubuntu-latest`; note CI does not install `protoc` explicitly, implying it must already be present on the GitHub-hosted runner image or the build could fail (no explicit `apt-get install protobuf-compiler` step observed)

## Environment Configuration

**Required env vars:**
- None strictly required for a basic build (all are optional overrides)
- Optional: `KINDLE_HUB_CACHE_DIR`, `KINDLE_HUB_TOKEN`, `KINDLE_NO_META`

**Secrets location:**
- No `.env` files or secrets directories present in the repository. `KINDLE_HUB_TOKEN` is expected to be supplied by the developer's shell/CI secret store at build or run time — never committed.

## Webhooks & Callbacks

**Incoming:**
- None — this is not a server application

**Outgoing:**
- None (no webhook dispatch logic); only outbound one-shot HTTP GET requests for model/dataset downloads described above

---

*Integration audit: 2026-07-09*
</content>
