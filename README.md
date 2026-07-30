# Incin

Incin is a deep learning framework in Rust focused on compile-time shape verification, developer ergonomics, and native multi-backend execution.

Incin enforces shape and type bounds at compile time to prevent tensor shape mismatches and runtime out-of-bounds panics.

## Features

A normal `incin = "0.0.0"` dependency enables only the standard library and
the native CPU backend. CUDA, WGPU, telemetry, autotuning, nightly experiments,
and third-party backends are opt-in. Enabling an accelerator does not change
`DefaultBackend`, which remains CPU whenever the `cpu` feature is enabled.

- **Compile-Time Shape Verification**: static `s![]` shapes catch incompatible tensor operations during compilation.
- **Named Dimensions**: `dim!` makes semantic axis names part of the tensor type.
- **Python-Style Slicing**: `idx![]` provides checked range and index expressions.
- **ONNX Model Importing**: `import_model!` generates typed Rust modules at compile time. Runtime ONNX weight loading is not implemented; use safetensors for runtime loading.
- **Native Backends**: CPU is the default; CUDA and WGPU are explicit features.
- **External Backends**: Candle interoperability is available explicitly through `external-candle` and `incin::external::candle`.
- **Data and Tooling**: parallel loading, diagnostics, telemetry, visualization, and editor integrations live in focused workspace crates.

### Facade feature matrix

<!-- BEGIN GENERATED: facade-features -->
| Feature | Default | Purpose |
|---|:--:|---|
| `std` | yes | Enables standard-library functionality, serialization, and filesystem APIs. |
| `nightly` | no | Enables nightly-only APIs in the core and macro crates. |
| `cpu` | yes | Enables the built-in CPU backend. This is the only default backend. |
| `cpu-blas` | no | Hands large f32 CPU matmuls to a blocked GEMM. The CPU backend is complete without it; see incin-backends for what it does and does not change. |
| `cuda` | no | Enables the native CUDA backend. CUDA is never enabled implicitly. |
| `wgpu` | no | Enables the cross-platform WGPU backend. WGPU is never enabled implicitly. |
| `external-candle` | no | Enables the external Candle backend at `incin::external::candle`. |
| `autotune` | no | Enables CUDA launch autotuning. |
| `train` | no | Enables the automatic `Trainer` at `incin::train`. Preview tier: useful and tested, but the interface may change without a migration path. |
| `distributed` | no | Enables typed meshes, static/runtime tensor placements, and distributed lowering proofs. Transports remain separate opt-in backend features. |
| `distributed-reference` | no | Enables the deterministic in-process collective transport used by conformance tests and local distributed-plan development. |
| `distributed-nccl` | no | Two-host process-per-rank CUDA transport and its TCP bootstrap. |
| `telemetry` | no | Enables backend telemetry hooks. `cargo incin doctor` also reports the run directory under this feature, which is why the dependency is direct here and not only through incin-backends. |
<!-- END GENERATED: facade-features -->

Examples:

```toml
# Bare/default CPU installation
incin = "0.0.0"

# WGPU in addition to the default CPU backend
incin = { version = "0.0.0", features = ["wgpu"] }

# CUDA-only application (use explicit CUDA backend/device types)
incin = { version = "0.0.0", default-features = false, features = ["std", "cuda"] }

# Third-party Candle interoperability
incin = { version = "0.0.0", features = ["external-candle"] }
```

### Lower-level crate features

<!-- BEGIN GENERATED: crate-features -->
- `incin-backends`: defaults to `std,cpu`; optional `cpu-blas`, `cuda`, `wgpu`, `autotune`, `external-candle`, `telemetry`, `distributed`, `distributed-reference`, and `distributed-nccl`.
- `incin-core`: defaults to `std`; optional `nightly`, `paranoid-validation`, `distributed`, `cuda`, and `wgpu`.
- `incin-macros`: defaults to `std`; optional `nightly` and `distributed`.
- `incin-diagnostics`: defaults to `std`.
- `incin-data`, `incin-telemetry`, `incin-viz`, `incin-viz-plugin-api`, and `incin-lsp` expose no Cargo features.
<!-- END GENERATED: crate-features -->

The two-host launcher uses one process per rank. `DistributedContext::from_env`
checks `INCIN_RANK`, `INCIN_WORLD_SIZE`, `INCIN_RUN_ID`,
`INCIN_LOCAL_CUDA_DEVICE`, `INCIN_RENDEZVOUS_ADDR`, and
`INCIN_RENDEZVOUS_TIMEOUT_MS` before communicator creation. Both hosts may use
local CUDA ordinal zero; the address on rank zero is a bind address and the
address on rank one must reach rank zero over the network.

With `autotune`, `incin_backends::tuning` exposes stable UUID/compiler/topology
identities, an atomic bounded persistent cache, and statically typed or `Dyn`
disabled, heuristic, coordinated-warmup, and profile-guided services. Cached
winners are treated as untrusted hints and are reused only after the current
legal candidate set matches. Distributed warmup uses one bounded epoch permit
and requires every topology rank to accept the same result before commit.

`incin-core`'s `cuda` and `wgpu` flags expose the device metadata backend crates
need; they do not execute kernels themselves. Disabling `incin-diagnostics`'
defaults gives the allocation-only diagnostic core.

The per-backend support tables — which operations each backend registers, for
which element types, layouts and ranks — are generated from those registrations
into [docs/capabilities.md](docs/capabilities.md). `cargo incin doctor` reports
which of them this machine can actually reach.

`incin-backends` can also be used without default features when implementing a
backend-specific binary. At least one of `cpu`, `cuda`, or `wgpu` is needed for
`IncinBackend` execution; `external-candle` exposes its separate external adapter.

## Setup & Requirements

Incin serializes ONNX protocol graphs natively using Protocol Buffers. Building the workspace requires the Protocol Buffers compiler (`protoc`).

### Ubuntu / Debian
```bash
sudo apt-get install -y protobuf-compiler
```

### macOS (Homebrew)
```bash
brew install protobuf
```

### Environment Variables

- `INCIN_HUB_CACHE_DIR`: Specifies a custom cache directory for downloaded models (defaults to `~/.cache/huggingface/hub`).
- `INCIN_HUB_TOKEN`: Authorization token for downloading private HuggingFace Hub repositories.
- `INCIN_NO_META`: Set to `1` to bypass `.incin_meta` cache and force full ONNX graph re-parsing during macro compilation.

## Quick Start

### Tensor Creation & Shape Macro

```rust,ignore
use incin::prelude::*;

// Statically dimensioned 2x3 tensor on default CPU backend
let static_tensor: Tensor<s![2, 3]> = Tensor::zeros(())?;

// Dynamically dimensioned tensor
let dynamic_tensor: Tensor<Dyn> = Tensor::zeros([2, 3])?;
```

### Module Definition & Forward Pass

```rust,ignore
use incin::prelude::*;

#[module]
pub struct MLP<B: Backend> {
    pub fc1: Linear<s![784, 128], B>,
    pub fc2: Linear<s![128, 10], B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<s![dyn, 784], B>) -> Result<Tensor<s![dyn, 10], B>> {
        let x = self.fc1.forward(x)?.relu()?;
        self.fc2.forward(x)
    }
}
```

### Slicing with `idx!`

```rust,ignore
use incin::prelude::*;

let t = Tensor::<Dyn>::zeros([2, 3, 4])?;
let sliced = t.slice(idx![.., 1..3, 0])?;
```

### ONNX Model Import

```rust,ignore
use incin::prelude::*;

import_model!("resnet18.onnx", ResNet18);

fn main() -> Result<()> {
    let mut model = ResNet18::<CpuBackendImpl>::new();
    println!("Parsed ONNX graph into Rust AST!");
    Ok(())
}
```

## Workspace Crates

- `incin`: Primary facade crate providing unified imports and prelude, plus
  the `cargo-incin` CLI binary.
- `incin-core`: Statically-typed `Tensor` implementation, traits, and graph definitions.
- `incin-backends`: Native CPU, opt-in CUDA and WGPU execution engines, plus an external Candle adapter.
- `incin-macros`: Procedural macros (`s!`, `idx!`, `module`, `import_model!`).
- `incin-data`: Data loading utilities, dataset traits, and HuggingFace Hub support.
- `incin-telemetry`: Event emission, transport streams, and graph snapshot recording.
- `incin-viz`: Terminal UI (TUI) model graph visualizer.
- `incin-diagnostics`: Typenum-to-decimal shape diagnostic humanization,
  shared by the CLI and the editor LSP proxy — see [CLI](#cli--cargo-incin)
  and [Editor / IDE Support](#editor--ide-support) below.
- `incin-lsp`: Transparent LSP proxy that routes rust-analyzer through
  `incin-diagnostics` so shape errors and inlay hints are humanized live in
  your editor.

## CLI — `cargo incin`

`cargo-incin` wraps `cargo check`/`build`/`test`/`run`, rewriting the
compiler's typenum shape errors (`UInt<UInt<...>>` walls) into plain decimals
live in your terminal — the same translation the editor extensions use. It
also inspects exported model files and translates pasted error text ad hoc.

### Install

```bash
cargo install --path crates/incin
```

This builds and installs the `cargo-incin` binary (the crate's only binary
target); invoke it as `cargo incin <subcommand>` from any Incin project.

### Usage

```bash
cargo incin check                # cargo check, with humanized shape errors
cargo incin build --release      # any cargo subcommand + its normal args work
cargo incin test
cargo incin run
cargo incin inspect model.gguf   # metadata for .safetensors / .gguf / .onnx files
cargo incin translate "..."      # translate pasted error text (arg or stdin)
```

Flags: `--raw` (skip translation, show the compiler's raw output), `--explain`
(append a plain-English shape-rule explanation for common errors), `--help`.

## Editor / IDE Support

The same humanization is available live inside your editor via `incin-lsp`,
a thin proxy that sits between your editor and rust-analyzer and rewrites
diagnostics and shape inlay hints through `incin-diagnostics` before they
reach you — no forked rust-analyzer, no per-editor parsing logic. See
`docs/growth/02-ide-extensions.md` for the full architecture and verification
status of each client below.

| Editor | Status | Install guide |
|---|---|---|
| VS Code | Ships; activation + rust-analyzer config-rewriting verified by an automated test against a real VS Code (`npm test`). Full humanized-diagnostic pipeline (real `incin-lsp` + rust-analyzer) not yet exercised end-to-end | [`editors/vscode/README.md`](editors/vscode/README.md) |
| Neovim (0.11+, or nvim-lspconfig on older versions) | Verified against a real Neovim install | [`editors/nvim/README.md`](editors/nvim/README.md) |
| RustRover / IntelliJ | External-tool fallback verified; native LSP-mode integration unverified | [`editors/rustrover/README.md`](editors/rustrover/README.md) |

VS Code and Neovim both need the proxy on `PATH` first (RustRover's shipped
fallback only needs the CLI above):

```bash
cargo install --path crates/incin-lsp --bin incin-lsp
```

## Documentation

- **Growth & architecture plans** (`docs/growth/`): task-by-task execution
  plans for adoption-facing features — named dimensions, this CLI/IDE
  tooling, deployment, and more — each with its own dated status ledger.
- **The Book ("Incinnomicon"):** planned, not yet built. See
  `docs/growth/07-the-book.md` for the full chapter outline; it will ship as
  an mdBook at `docs/book/`, built with `mdbook build docs/book`, deployed to
  GitHub Pages. Until it lands, this README, the growth docs, and
  `crates/incin/examples/` are the best starting points.

## License

Dual-licensed under either MIT or Apache 2.0.
