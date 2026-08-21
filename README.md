<div align="center">

# incin

### A Rust deep learning framework with compile-time checks for tensor shapes and dtypes.

[![Incin CI](https://img.shields.io/badge/Incin_CI-passing-brightgreen?logo=github)](https://github.com/xupremix/incin/actions)
[![crates.io](https://img.shields.io/crates/v/incin.svg?color=orange)](https://crates.io/crates/incin)
[![docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://docs.rs/incin)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](./LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-orange)](./Cargo.toml)

[Quick Start](#quick-start) · [Features](#features) · [Architecture](#architecture) · [The Book](https://xupremix.github.io/incin) · [CLI](#cli-cargo-incin) · [Editors](#editor-support)

</div>

Incin encodes tensor shapes, dtypes, devices, and gradient capability in Rust's type system. A `matmul` with incompatible dimensions, an update to a frozen parameter, or an unsupported dtype becomes a compile error instead of a runtime failure on epoch 40 at 3 AM on a Saturday.

```rust,ignore
use incin::prelude::*;

// [4, 8] x [8, 2] is valid, so the result has shape [4, 2].
let x: Tensor<s![4, 8], DefaultBackend> = Tensor::zeros(())?;
let w: Tensor<s![8, 2], DefaultBackend> = Tensor::zeros(())?;
let y = x.matmul(&w)?; // Tensor<s![4, 2], ...>

// This one does not compile because the inner dimensions do not match:
let bad: Tensor<s![3, 8], DefaultBackend> = Tensor::zeros(())?;
let _ = x.matmul(&bad)?;
```

The last line does not compile. Plain `cargo check` shows Rust's underlying
typenum representation:

```text
error[E0277]: Cannot contract dimension `UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>` with `UInt<UInt<UTerm, B1>, B1>`
  --> src/main.rs:9:22
   |
 9 |     let _ = x.matmul(&bad)?;
   |               ------ ^^^^ inner dimensions do not match
```

[`cargo incin check`](#cli-cargo-incin) prints the same error with decimal
dimensions and readable shapes:

```text
error[E0277]: Cannot contract dimension `[4, 8]` with `MatMulShape<[3, 8]>`
  --> src/main.rs:9:22
   |
 9 |     let _ = x.matmul(&bad)?;
   |               ------ ^^^^ inner dimensions do not match

  └── 💡 [Typenum Translation Hints]:
      • 4  <= UInt<UInt<UInt<UTerm, B1>, B0>, B0>
      • 8  <= UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>
      • 3  <= UInt<UInt<UTerm, B1>, B1>
```

It also expands the `long-type-*.txt` files that rustc creates for very long
types.

> The CPU backend is the supported baseline for 0.1. CUDA, WGPU, and Metal are
> previews with smaller operation sets. The generated
> [capability table](https://github.com/xupremix/incin/blob/master/docs/capabilities.md)
> records the exact coverage. The Book also keeps a direct list of
> [unfinished work](https://github.com/xupremix/incin/blob/master/docs/book/src/whats_not_finished.md).

## Features

| | |
|---|---|
| **Compile-time shape checks** | Static `s![]` shapes reject incompatible tensor operations during compilation. |
| **Named dimensions** | `dim!` puts semantic axis names such as `Batch` and `Channels` in the tensor type. |
| **Checked slicing** | `i![]` provides runtime range and index expressions with bounds checks. |
| **Gradient state in types** | `Grad` and `NoGrad` prevent frozen tensors from receiving gradient updates. |
| **ONNX import** | `import_model!` expands supported `.onnx` graphs into typed Rust and rejects unsupported graphs. |
| **Backend support** | CPU is complete. CUDA, WGPU, and Metal are opt-in previews with support recorded in [`docs/capabilities.md`](https://github.com/xupremix/incin/blob/master/docs/capabilities.md). |
| **Hugging Face Hub** | The `data-hub` feature downloads model weights and dataset files. |
| **Static shape types** | Known dimensions live in types and need no separate runtime shape object. |

## Architecture

| Layer | Responsibility |
|---|---|
| `incin` | Facade, prelude, macros, and the `cargo-incin` CLI. |
| `incin-core` | Typed tensors, operations, and graph definitions. |
| `incin-backends` | Descriptor dispatch and backend executors. CPU is complete; CUDA, WGPU, and Metal are previews. |
| Supporting crates | Data loading, telemetry, visualization, diagnostics, and the `incin-lsp` editor proxy. |

`Tensor<S, B, K, G>` records the shape, backend, element dtype, and gradient
state. Tensor methods build validated operation descriptors, then dispatch to
an executor registered by the selected backend. A missing executor is a type
error; Incin does not silently switch backends.

## Quick Start

Version 0.1.0 has not been published to crates.io yet. Until it is published,
depend on the `master` branch:

```toml
[dependencies]
incin = { git = "https://github.com/xupremix/incin", branch = "master" }
```

After publication, replace that line with `incin = "0.1.0"`. The default
feature set uses the standard library and native CPU backend. CUDA, WGPU,
telemetry, autotuning, nightly experiments, and third-party backends are
opt-in. `DefaultBackend` remains CPU whenever the `cpu` feature is enabled.

### Concrete CPU quick start

```rust,ignore
use incin::prelude::*;

fn main() -> Result<()> {
    let a = Cpu.randn(shape![2, 3])?;
    let b = Cpu.randn(shape![3])?;
    let sum = &a + &b;
    let reshaped = sum.reshape(shape![3, 2])?;
    let reduced = reshaped.sum_keepdim(axis!(-1))?;
    let _ = reduced;
    Ok(())
}
```

The concrete API is the shortest route for a CPU application. Generic code can
name its backend and shape parameters explicitly:

```rust,ignore
use incin::prelude::*;

type B = DefaultBackend;

fn generic_example() -> Result<()> {
    let static_tensor: Tensor<s![2, 3], B> = Tensor::zeros(())?;
    let dynamic_tensor: Tensor<Dyn, B> = Tensor::zeros(shape![2, 3])?;
    let _sum = &static_tensor + &static_tensor;
    Ok(())
}
```

`shape!` creates runtime shapes, while `s![]` describes static shapes.
`axis!(-1)` selects the last axis. `i![]` accepts ordinary index and range
expressions without a fixed rank limit.

<details>
<summary>Module definition and forward pass</summary>

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

</details>

<details>
<summary>Slicing with <code>i!</code></summary>

```rust,ignore
use incin::prelude::*;

let t = Tensor::<Dyn>::zeros([2, 3, 4])?;
let sliced = t.get(i![.., 1..3, 0])?;
```

</details>

<details>
<summary>ONNX model import</summary>

```rust,ignore
use incin::prelude::*;

import_model!("resnet18.onnx", ResNet18);

let model = ResNet18::<CpuBackendImpl>::new();
```

Runtime ONNX *weight* loading is not implemented; use safetensors for
runtime loading, or download real-world weights straight from the
[Hugging Face Hub](#environment-variables) with the `data-hub` feature.

</details>

## Facade feature matrix

<!-- BEGIN GENERATED: facade-features -->
| Feature | Tier | Prerequisites | Default | Enables | Incompatibilities | Purpose |
|---|---|---|:--:|---|---|---|
| `std` | stable | none | yes | `incin-core/std`, `incin-macros/std`, `incin-backends/std` | none | Enables standard-library functionality, serialization, and filesystem APIs. |
| `nightly` | preview | none | no | `incin-core/nightly`, `incin-macros/nightly` | none | Enables nightly-only APIs in the core and macro crates. |
| `cpu` | stable | `std` | yes | `std`, `incin-backends/cpu` | none | Enables the built-in CPU backend. This is the only default backend. |
| `cpu-blas` | stable | `std`, `cpu` | no | `std`, `cpu`, `incin-backends/cpu-blas` | none | Hands large f32 CPU matmuls to a blocked GEMM. The CPU backend is complete without it; see incin-backends for what it does and does not change. |
| `cuda` | preview | `std` | no | `std`, `incin-core/cuda`, `incin-backends/cuda` | none | Preview: the native CUDA backend, covering the subset in docs/capabilities.md. Never enabled implicitly. |
| `wgpu` | preview | `std` | no | `std`, `incin-backends/wgpu` | none | Preview: the cross-platform WGPU backend, covering the subset in docs/capabilities.md. Never enabled implicitly. |
| `metal` | preview | `std` | no | `std`, `incin-backends/metal` | none | Preview: the native Metal backend for Apple Silicon. Its executors are stubs pending MTL-002/003; see docs/capabilities.md. Never enabled implicitly. |
| `metal-mps` | preview | `metal` | no | `metal`, `incin-backends/metal-mps` | none | Enables MPS and MPSGraph structured primitives for Apple Silicon. |
| `external-candle` | stable | `std` | no | `std`, `incin-backends/external-candle` | none | Enables the external Candle backend at `incin::external::candle`. |
| `autotune` | preview | `cuda` | no | `cuda`, `incin-backends/autotune` | none | Enables CUDA launch autotuning. |
| `train` | preview | `std` | no | `std` | none | Enables the preview trainer at `incin::experimental::training`. The interface may change without a migration path. |
| `distributed` | preview | none | no | `incin-core/distributed`, `incin-macros/distributed` | none | Preview: typed meshes, static/runtime tensor placements, and distributed lowering proofs. This is a planning and validation layer; there is no distributed execution path. Transports remain separate opt-in backend features. |
| `distributed-reference` | preview | `distributed` | no | `distributed`, `incin-backends/distributed-reference` | none | Enables the deterministic in-process collective transport used by conformance tests and local distributed-plan development. |
| `distributed-nccl` | preview | `distributed`, `cuda` | no | `distributed`, `cuda`, `incin-backends/distributed-nccl` | none | Two-host process-per-rank CUDA transport and its TCP bootstrap. |
| `telemetry` | stable | `std` | no | `std`, `incin-backends/telemetry`, `dep:incin-telemetry` | none | Enables backend telemetry hooks. `cargo incin doctor` also reports the run directory under this feature, which is why the dependency is direct here and not only through incin-backends. |
| `test-utils` | test-only | `std`, `cpu` | no | `std`, `cpu`, `incin-backends/test-utils` | none | Deterministic fault-injection hooks for tests. No stand-in backend: a test that needs a backend uses a real one. |
| `backend-authoring` | stable | none | no | none | none | Extension contracts for backend authors. |
| `data-hub` | stable | `std` | no | `std`, `incin-data/hub` | none | The Hugging Face Hub client at `incin::hub`. Off by default because it brings an async runtime and a second TLS stack into the dependency graph for an API most training code never calls; dataset downloading does not need it. |
| `compiled` | preview | `std`, `cpu` | no | `std`, `cpu`, `incin-core/compiled`, `incin-backends/compiled` | none | Preview-only CPU reference evaluator and plan-inspection types under `incin::experimental::compiled`. No stable API, deployment format, or portable artifact ABI is promised. |
| `hardware-tests` | hardware-only | `distributed-nccl` | no | `distributed-nccl` | none | Opt-in only: ignored multi-host CUDA runtime fixtures require actual hardware and are not part of compile-only feature coverage. |
<!-- END GENERATED: facade-features -->

<details>
<summary>Common feature combinations</summary>

These registry dependency forms apply after `0.1.0` is published. Until then,
use the `master` Git dependency from the Quick Start and add the same feature
fields.

```toml
# Bare/default CPU installation
incin = "0.1.0"

# WGPU in addition to the default CPU backend
incin = { version = "0.1.0", features = ["wgpu"] }

# CUDA-only application (use explicit CUDA backend/device types)
incin = { version = "0.1.0", default-features = false, features = ["std", "cuda"] }

# Third-party Candle interoperability
incin = { version = "0.1.0", features = ["external-candle"] }
```

</details>

<details>
<summary><strong>Lower-level crate features</strong></summary>

<!-- BEGIN GENERATED: crate-features -->
- `incin-backends`: defaults to `std,cpu`; optional `compiled`, `cpu-blas`, `cuda`, `cuda-vendor`, `wgpu`, `metal`, `metal-mps`, `autotune`, `external-candle`, `telemetry`, `distributed`, `distributed-reference`, `distributed-nccl`, and `test-utils`.
- `incin-core`: defaults to `std`; optional `nightly`, `paranoid-validation`, `distributed`, `cuda`, `wgpu`, `metal`, `compiled`, `postcard`, `safetensors`, and `serde_json`.
- `incin-macros`: defaults to `std`; optional `nightly` and `distributed`.
- `incin-diagnostics`: defaults to `std`.
- `incin-data`: defaults to `download`; optional `hub`.
- `incin-telemetry`, `incin-viz`, `incin-viz-plugin-api`, and `incin-lsp` expose no Cargo features.
<!-- END GENERATED: crate-features -->

The full feature contract, including hardware requirements and incompatible
combinations, is in [the feature matrix](https://github.com/xupremix/incin/blob/master/docs/FEATURE_MATRIX.md).
Backend operation coverage is generated separately in
[docs/capabilities.md](https://github.com/xupremix/incin/blob/master/docs/capabilities.md).
Use `cargo incin doctor` to see which backends are usable on the current
machine.

</details>

## Setup and requirements

The normal build needs only a Rust toolchain. The generated ONNX protobuf
module is checked into the repository, so `protoc` is needed only when a
maintainer intentionally regenerates it with `cargo xtask onnx`.

### Environment variables

| Variable | Purpose |
|---|---|
| `INCIN_HUB_CACHE_DIR` | Custom cache directory for Hub downloads (defaults to `~/.cache/huggingface/hub`). |
| `INCIN_HUB_TOKEN` | Authorization token for private Hugging Face Hub repositories. |
| `INCIN_NO_META` | Set to `1` to bypass the `.incin_meta` cache and force full ONNX graph re-parsing during macro compilation. |

## Workspace crates

| Crate | Role |
|---|---|
| [`incin`](https://github.com/xupremix/incin/tree/master/crates/incin) | Primary facade: unified imports, prelude, and the `cargo-incin` CLI binary. |
| [`incin-core`](https://github.com/xupremix/incin/tree/master/crates/incin-core) | Statically-typed `Tensor` implementation, traits, and graph definitions. |
| [`incin-backends`](https://github.com/xupremix/incin/tree/master/crates/incin-backends) | Native CPU (complete), opt-in CUDA/WGPU/Metal (preview) execution engines, plus an external Candle adapter. |
| [`incin-macros`](https://github.com/xupremix/incin/tree/master/crates/incin-macros) | Procedural macros: `s!`, `shape!`, `axis!`, `i!`, `module`, `import_model!`. |
| [`incin-data`](https://github.com/xupremix/incin/tree/master/crates/incin-data) | Data loading utilities, dataset traits, and Hugging Face Hub support. |
| [`incin-telemetry`](https://github.com/xupremix/incin/tree/master/crates/incin-telemetry) | Event emission, transport streams, and graph snapshot recording. |
| [`incin-viz`](https://github.com/xupremix/incin/tree/master/crates/incin-viz) | Terminal UI (TUI) model graph visualizer. |
| [`incin-diagnostics`](https://github.com/xupremix/incin/tree/master/crates/incin-diagnostics) | Typenum-to-decimal shape diagnostic humanization, shared by the CLI and the editor LSP proxy. |
| [`incin-lsp`](https://github.com/xupremix/incin/tree/master/crates/incin-lsp) | Transparent LSP proxy that routes rust-analyzer through `incin-diagnostics` so shape errors and inlay hints are humanized live in your editor. |

## CLI: `cargo incin`

`cargo-incin` wraps normal Cargo commands and rewrites typenum-heavy compiler
errors as they arrive. It can also inspect supported model files or translate
an error copied from another command.

```bash
# Install
cargo install --path crates/incin --bin cargo-incin --locked

# Use
cargo incin check                # cargo check with readable shape errors
cargo incin build --release      # Cargo flags pass through
cargo incin test
cargo incin run
cargo incin inspect model.gguf   # metadata for .safetensors / .gguf / .onnx files
cargo incin translate "..."      # translate pasted error text (arg or stdin)
```

Flags: `--raw` (skip translation, show the compiler's raw output), `--explain`
(append a plain-English shape-rule explanation for common errors), `--help`.

## Editor support

`incin-lsp` sits between an editor and rust-analyzer. It rewrites diagnostics
and shape inlay hints with the same code used by the CLI.
[`docs/growth/02-ide-extensions.md`](https://github.com/xupremix/incin/blob/master/docs/growth/02-ide-extensions.md) records
the design and test boundaries.

| Editor | Status | Install guide |
|---|---|---|
| VS Code | Verified with the local VSIX, `incin-lsp`, and rust-analyzer | [`editors/vscode/README.md`](https://github.com/xupremix/incin/blob/master/editors/vscode/README.md) |
| Neovim 0.11+ | Verified with `incin-lsp` and rust-analyzer | [`editors/nvim/README.md`](https://github.com/xupremix/incin/blob/master/editors/nvim/README.md) |
| RustRover / IntelliJ | External-tool fallback verified; native LSP mode is not yet verified | [`editors/rustrover/README.md`](https://github.com/xupremix/incin/blob/master/editors/rustrover/README.md) |

VS Code and Neovim need the proxy on `PATH`:

```bash
cargo install --path crates/incin-lsp --bin incin-lsp --locked
```

### VS Code

![A reshape error rewritten by incin-lsp in VS Code](https://raw.githubusercontent.com/xupremix/incin/master/docs/assets/editors/vscode-shape-diagnostic.png)

### Neovim

![The same reshape error rewritten by incin-lsp in Neovim](https://raw.githubusercontent.com/xupremix/incin/master/docs/assets/editors/neovim-shape-diagnostic.png)

## Documentation

- [The Book](https://xupremix.github.io/incin/) is the user guide.
- [What's not finished](https://github.com/xupremix/incin/blob/master/docs/book/src/whats_not_finished.md)
  lists known gaps directly.
- [Backend capabilities](https://github.com/xupremix/incin/blob/master/docs/capabilities.md)
  are generated from the executor registrations.
- [Architecture and growth notes](https://github.com/xupremix/incin/tree/master/docs/growth/)
  cover longer-term work.

Build the Book with `mdbook build docs/book`. Its Rust examples are checked by
`cargo test -p incin --features backend-authoring --doc`.

## License

Dual-licensed under either

[**MIT**](https://github.com/xupremix/incin/blob/master/LICENSE_MIT) or [**Apache 2.0**](https://github.com/xupremix/incin/blob/master/LICENSE_APACHE)

at your option.
