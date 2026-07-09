# Technology Stack

**Analysis Date:** 2026-07-09

## Languages

**Primary:**
- Rust (edition 2024) - Entire workspace, all crates under `crates/`

**Secondary:**
- Python - One-off refactor/fix scripts at repo root (`fix_*.py`, `refactor_*.py`, `rewrite_backend.py`), not part of the build; used as developer tooling to mechanically rewrite Rust source during large refactors
- Protocol Buffers - `crates/kindle-core/proto/onnx.proto` defines the ONNX graph schema compiled via `prost-build`

## Runtime

**Environment:**
- Rust toolchain: `stable` (channel pinned in `rust-toolchain.toml`), with `rust-src` component; a commented-out `nightly` channel line exists for future use
- Installed compiler observed: `rustc 1.92.0`
- No `no_std`/embedded target evidenced; `std` is a default cargo feature across crates (`kindle`, `kindle-core`, `kindle-macros`)

**Package Manager:**
- Cargo (workspace with resolver `"2"`), root manifest `Cargo.toml`
- Lockfile: present (`Cargo.lock`, workspace-wide)

## Frameworks

**Core:**
- Not a web/app framework — this is a deep learning / tensor library. Core abstractions: `Backend` trait, statically-typed `Tensor<S, B, T, D, G>` (`crates/kindle-core/src/tensor/`)

**ML Backends (pluggable via Cargo features):**
- `candle-core` / `candle-nn` `0.9.1` (HuggingFace Candle) - default/primary backend, `crates/kindle-backends/`
- `ndarray` `0.17.2` - optional CPU array backend
- `burn` `0.21.0` (with `autodiff` feature) + `burn-ndarray` `0.21.0` - optional alternative backend
- Feature flags in `crates/kindle/Cargo.toml`: `candle` (default), `ndarray`, `burn`, `cuda`, `metal`

**Proc-Macro / Codegen:**
- `crates/kindle-macros` (proc-macro crate) - powers `#[kindle::module]`, `#[kindle::forward]`, `s!`, `idx!`, and `import_model!` macros
- Dependencies: `syn` `2.0.118` (full), `quote` `1.0.46`, `proc-macro2` `1.0.106`

**Testing:**
- Built-in `cargo test` / `#[test]` across crates
- `trybuild` `1.0.117` (dev-dependency of `kindle-core`) - compile-fail/UI testing for proc macros and static shape-checking guarantees

**Build/Dev:**
- `prost-build` `0.14.4` (build-dependency of `kindle-core`) - compiles `onnx.proto` into Rust structs at build time; requires system `protoc` (Protocol Buffers Compiler) to be installed
- `cargo fmt`, `cargo clippy` (workspace lints defined in root `Cargo.toml` under `[workspace.lints.clippy]`, currently empty/default)

## Key Dependencies

**Critical:**
- `candle-core`/`candle-nn` `0.9.1` - default tensor compute backend
- `safetensors` `0.4.1` / `0.4.5` - model weight (de)serialization format, used in `kindle-core` and `kindle-macros`
- `prost` `0.14.4` (kindle-core) / `0.6` (kindle-macros) - Protobuf runtime for ONNX graph representation (note version mismatch between crates)
- `onnx-pb` `0.1.4` - ONNX protobuf message definitions used by `kindle-macros`
- `typenum` `1.20.1` - compile-time type-level numerics, underpins the static shape-verification system
- `serde` `1.0.228`/`1.0` (derive) + `bincode` `1.3.3` + `serde_json` `1.0.150` - serialization (model metadata caching, `.kindle_meta` JSON cache mentioned in README)
- `thiserror` `2.0.18` - error type definitions
- `half` `2.7.1` - half-precision (f16/bf16) float support
- `anyhow` `1.0.103` - error handling/propagation across `kindle`, `kindle-backends`, `kindle-core`, `kindle-data`
- `bytes` `1.12.0` - byte buffer handling in `kindle-core`

**Infrastructure:**
- `hf-hub` `0.5.0` (features: `tokio`, `rustls-tls`, `ureq`) - HuggingFace Hub client, used by `kindle-data::hub`
- `ureq` `3.3.0` - blocking HTTP client for direct dataset/file downloads (`kindle-data::downloader`)
- `rayon` `1.12.0` - data-parallelism for `DataLoaderExt`/`into_par_loader()` in `kindle-data`
- `rand` `0.8` - randomness (data shuffling, weight init)
- `flate2` `1.1.9` - gzip decompression for downloaded dataset archives

## Configuration

**Environment:**
- `KINDLE_HUB_CACHE_DIR` - overrides HuggingFace Hub cache directory (default `~/.cache/huggingface/hub`)
- `KINDLE_HUB_TOKEN` - HuggingFace auth token for private/gated repos, read in `crates/kindle-data/src/hub.rs`
- `KINDLE_NO_META` - set to `1`/`true` to force `import_model!` macro to bypass its `.kindle_meta` JSON cache and fully re-parse `.safetensors`/`.onnx` graphs during `cargo build`
- No `.env` files present in the repo; configuration is read directly via `std::env::var` at macro-expansion/runtime

**Build:**
- Root `Cargo.toml` - workspace member list and shared `[workspace.dependencies]`/`[workspace.package]`
- `crates/kindle-core/build.rs` - invokes `prost-build` to compile `proto/onnx.proto`; requires system `protoc`
- `rust-toolchain.toml` - pins toolchain to `stable` with `rust-src`
- `.github/workflows/ci.yml` - CI pipeline (see INTEGRATIONS.md)

## Platform Requirements

**Development:**
- Rust stable toolchain (`rustup` recommended) with `rust-src` component
- System-installed Protocol Buffers Compiler (`protoc`) — required to build `kindle-core` due to ONNX proto compilation
- Optional: CUDA toolkit (for `cuda` feature) or Metal (macOS, for `metal` feature) if GPU acceleration is desired

**Production:**
- No deployment target detected — this is a library/framework crate (published to crates.io style workspace, version `0.1.0`), not a deployed service
- Consumers embed `kindle`/`kindle-core`/`kindle-data` as Rust dependencies in their own binaries

---

*Stack analysis: 2026-07-09*
</content>
