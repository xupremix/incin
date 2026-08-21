<div align="center">

# incin

**A Rust deep learning framework with compile-time checks for tensor shapes and dtypes.**

[![CI](https://github.com/xupremix/incin/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/xupremix/incin/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/incin.svg)](https://crates.io/crates/incin)
[![docs.rs](https://img.shields.io/docsrs/incin)](https://docs.rs/incin)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-orange.svg)](https://github.com/xupremix/incin/blob/master/Cargo.toml)

[Quick Start](#quick-start) ·
[Features](#features) ·
[Architecture](#architecture) ·
[The Book](https://github.com/xupremix/incin/blob/master/docs/book/src/SUMMARY.md) ·
[CLI](#cli--cargo-incin) ·
[Editors](#editor--ide-support)

</div>

<br>

Incin encodes tensor shapes, dtypes, devices, and gradient capability in Rust's
type system. A `matmul` with incompatible dimensions, an update to a frozen
parameter, or an unsupported dtype becomes a compile error instead of a
runtime failure during training.

```rust,ignore
use incin::prelude::*;

// The types describe the shapes. [4, 8] · [8, 2] is a legal matmul,
// and the compiler checks it before the program runs.
let x: Tensor<s![4, 8], DefaultBackend> = Tensor::zeros(())?;
let w: Tensor<s![8, 2], DefaultBackend> = Tensor::zeros(())?;
let y = x.matmul(&w)?;                    // Tensor<s![4, 2], ...>

// This one does not compile because the inner dimensions do not match:
let bad: Tensor<s![3, 8], DefaultBackend> = Tensor::zeros(())?;
let _ = x.matmul(&bad)?;
```

The last line produces a normal Rust compiler error. Plain `cargo check` shows
Rust's raw typenum encoding:

```text
error[E0277]: Cannot contract dimension `UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>` with `UInt<UInt<UTerm, B1>, B1>`
  --> src/main.rs:9:22
   |
 9 |     let _ = x.matmul(&bad)?;
   |               ------ ^^^^ inner dimensions do not match
```

[`cargo incin check`](#cli--cargo-incin) rewrites the error while the command
runs. It collapses the `UInt<UInt<...>>` representation and the shape's
`DimCons<H, DimCons<...>>` cons-list encoding to a plain `[4, 8]`, then adds a
translation key:

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

When rustc writes a deeply nested type to a `long-type-*.txt` file instead of
printing it, `cargo incin` reads that file as well. The translated output adds
an `[Expanded Full Type]` section, so you do not need to find and decode the
file separately.

> **CPU is the complete, verified backend.** Every operation in the canonical
> catalog that a backend can execute has a CPU executor, and training runs
> there today. CUDA, WGPU, and Metal are **previews**: each covers a
> documented subset (arithmetic, reductions, `matmul`, convolution/pooling,
> plus unary activations on WGPU) and none yet covers normalization, loss, or
> embedding. [`docs/capabilities.md`](https://github.com/xupremix/incin/blob/master/docs/capabilities.md) is generated
> straight from the backend registrations and is the authoritative answer for
> any given operation. See also [what isn't finished
> yet](https://github.com/xupremix/incin/blob/master/docs/book/src/whats_not_finished.md).

---

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
| **No runtime shape metadata** | Static shape information exists in `typenum` types and is erased during compilation. |

---

## Architecture

| Layer | Responsibility |
|---|---|
| `incin` | Facade, prelude, macros, and the `cargo-incin` CLI. |
| `incin-core` | Typed tensors, operations, and graph definitions. |
| `incin-backends` | Descriptor dispatch and backend executors. CPU is complete; CUDA, WGPU, and Metal are previews. |
| Supporting crates | Data loading, telemetry, visualization, diagnostics, and the `incin-lsp` editor proxy. |

Operations follow a typed path from the frontend to a validated descriptor.
A capability table determines whether the selected backend can execute the
operation. If no kernel is registered, compilation fails with a trait-bound
error instead of silently falling back. In `Tensor<S, B, K, G>`, `S`, `B`, `K`,
and `G` represent the shape, backend, element dtype, and gradient capability.

---

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

The concrete form is the shortest path for an application using the default
CPU backend. For generic code, make the backend and shape parameters
explicit:

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

`shape!` constructs runtime shape values and `s![]` describes compile-time
shape facts. Axis selectors accept signed values, so `axis!(-1)` selects the
last axis without constructing cursor types.

`i![]` expands to an unbounded index list, so indexing does not stop at a
fixed tuple arity. Static transpose and flatten selectors also support
arbitrary positive and negative axis positions; known static shapes retain
their exact output proof.

<details>
<summary><strong>Module definition & forward pass</strong></summary>

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
<summary><strong>Slicing with <code>i!</code></strong></summary>

```rust,ignore
use incin::prelude::*;

let t = Tensor::<Dyn>::zeros([2, 3, 4])?;
let sliced = t.get(i![.., 1..3, 0])?;
```

</details>

<details>
<summary><strong>ONNX model import</strong></summary>

```rust,ignore
use incin::prelude::*;

import_model!("resnet18.onnx", ResNet18);

fn main() -> Result<()> {
    let mut model = ResNet18::<CpuBackendImpl>::new();
    println!("Parsed ONNX graph into Rust AST!");
    Ok(())
}
```

Runtime ONNX *weight* loading is not implemented; use safetensors for
runtime loading, or download real-world weights straight from the
[Hugging Face Hub](#environment-variables) with the `data-hub` feature.

</details>

---

## Facade feature matrix

<!-- BEGIN GENERATED: facade-features -->
| Feature | Default | Purpose |
|---|:--:|---|
| `std` | yes | Enables standard-library functionality, serialization, and filesystem APIs. |
| `nightly` | no | Enables nightly-only APIs in the core and macro crates. |
| `cpu` | yes | Enables the built-in CPU backend. This is the only default backend. |
| `cpu-blas` | no | Hands large f32 CPU matmuls to a blocked GEMM. The CPU backend is complete without it; see incin-backends for what it does and does not change. |
| `cuda` | no | Preview: the native CUDA backend, covering the subset in docs/capabilities.md. Never enabled implicitly. |
| `wgpu` | no | Preview: the cross-platform WGPU backend, covering the subset in docs/capabilities.md. Never enabled implicitly. |
| `metal` | no | Preview: the native Metal backend for Apple Silicon. Its executors are stubs pending MTL-002/003; see docs/capabilities.md. Never enabled implicitly. |
| `metal-mps` | no | Enables MPS and MPSGraph structured primitives for Apple Silicon. |
| `external-candle` | no | Enables the external Candle backend at `incin::external::candle`. |
| `autotune` | no | Enables CUDA launch autotuning. |
| `train` | no | Enables the preview trainer at `incin::experimental::training`. The interface may change without a migration path. |
| `distributed` | no | Preview: typed meshes, static/runtime tensor placements, and distributed lowering proofs. This is a planning and validation layer; there is no distributed execution path. Transports remain separate opt-in backend features. |
| `distributed-reference` | no | Enables the deterministic in-process collective transport used by conformance tests and local distributed-plan development. |
| `distributed-nccl` | no | Two-host process-per-rank CUDA transport and its TCP bootstrap. |
| `telemetry` | no | Enables backend telemetry hooks. `cargo incin doctor` also reports the run directory under this feature, which is why the dependency is direct here and not only through incin-backends. |
| `test-utils` | no | Deterministic fault-injection hooks for tests. No stand-in backend: a test that needs a backend uses a real one. |
| `backend-authoring` | no | Extension contracts for backend authors. |
| `data-hub` | no | The Hugging Face Hub client at `incin::hub`. Off by default because it brings an async runtime and a second TLS stack into the dependency graph for an API most training code never calls; dataset downloading does not need it. |
| `compiled` | no | Preview: curated types for compiled execution. The interface may change without a migration path. |
| `hardware-tests` | no | Opt-in only: ignored multi-host CUDA runtime fixtures require actual hardware and are not part of compile-only feature coverage. |
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
- `incin-core`: defaults to `std`; optional `nightly`, `paranoid-validation`, `distributed`, `cuda`, `wgpu`, `metal`, and `compiled`.
- `incin-macros`: defaults to `std`; optional `nightly` and `distributed`.
- `incin-diagnostics`: defaults to `std`.
- `incin-data`: defaults to `download`; optional `hub`.
- `incin-telemetry`, `incin-viz`, `incin-viz-plugin-api`, and `incin-lsp` expose no Cargo features.
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

The per-backend support tables (which operations each backend registers, for
which element types, layouts, and ranks) are generated from those
registrations into [docs/capabilities.md](https://github.com/xupremix/incin/blob/master/docs/capabilities.md). `cargo incin
doctor` reports which of them this machine can actually reach.

`incin-backends` can also be used without default features when implementing a
backend-specific binary. At least one of `cpu`, `cuda`, or `wgpu` is needed for
`IncinBackend` execution; `external-candle` exposes its separate external
adapter.

</details>

---

## Setup & Requirements

The normal build needs only a Rust toolchain. The generated ONNX protobuf
module is checked into the repository, so `protoc` is needed only when a
maintainer intentionally regenerates it with `cargo xtask onnx`.

### Environment Variables

| Variable | Purpose |
|---|---|
| `INCIN_HUB_CACHE_DIR` | Custom cache directory for Hub downloads (defaults to `~/.cache/huggingface/hub`). |
| `INCIN_HUB_TOKEN` | Authorization token for private Hugging Face Hub repositories. |
| `INCIN_NO_META` | Set to `1` to bypass the `.incin_meta` cache and force full ONNX graph re-parsing during macro compilation. |

---

## Workspace Crates

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

---

## CLI: `cargo incin`

`cargo-incin` wraps `cargo check`/`build`/`test`/`run`, rewriting the
compiler's typenum shape errors (`UInt<UInt<...>>` walls) into plain decimals
live in your terminal, using the same translation the editor extensions use. It
also inspects exported model files and translates pasted error text ad hoc.

```bash
# Install
cargo install --path crates/incin --bin cargo-incin --locked

# Use
cargo incin check                # cargo check, with humanized shape errors
cargo incin build --release      # any cargo subcommand + its normal args work
cargo incin test
cargo incin run
cargo incin inspect model.gguf   # metadata for .safetensors / .gguf / .onnx files
cargo incin translate "..."      # translate pasted error text (arg or stdin)
```

Flags: `--raw` (skip translation, show the compiler's raw output), `--explain`
(append a plain-English shape-rule explanation for common errors), `--help`.

---

## Editor / IDE Support

`incin-lsp` runs between an editor and rust-analyzer. It rewrites diagnostics
and shape inlay hints using the same `incin-diagnostics` code as the CLI.
[`docs/growth/02-ide-extensions.md`](https://github.com/xupremix/incin/blob/master/docs/growth/02-ide-extensions.md) records
the design and test boundaries.

| Editor | Status | Install guide |
|---|---|---|
| VS Code | End-to-end check completed with the local VSIX, `incin-lsp`, and rust-analyzer | [`editors/vscode/README.md`](https://github.com/xupremix/incin/blob/master/editors/vscode/README.md) |
| Neovim (0.11+, or nvim-lspconfig on older versions) | End-to-end check completed with Neovim 0.12, `incin-lsp`, and rust-analyzer | [`editors/nvim/README.md`](https://github.com/xupremix/incin/blob/master/editors/nvim/README.md) |
| RustRover / IntelliJ | External-tool fallback verified; native LSP-mode integration unverified | [`editors/rustrover/README.md`](https://github.com/xupremix/incin/blob/master/editors/rustrover/README.md) |

VS Code and Neovim both need the proxy on `PATH` first (RustRover's shipped
fallback only needs the CLI above):

```bash
cargo install --path crates/incin-lsp --bin incin-lsp --locked
```

### VS Code

![A reshape error rewritten by incin-lsp in VS Code](https://raw.githubusercontent.com/xupremix/incin/master/docs/assets/editors/vscode-shape-diagnostic.png)

### Neovim

![The same reshape error rewritten by incin-lsp in Neovim](https://raw.githubusercontent.com/xupremix/incin/master/docs/assets/editors/neovim-shape-diagnostic.png)

---

## Documentation

- **[The Book ("Incinnomicon")](https://github.com/xupremix/incin/blob/master/docs/book/src/SUMMARY.md)**: the current
  user guide. Build it locally with `mdbook build docs/book`. Validate its
  Rust examples with `cargo test -p incin --features 'backend-authoring'
  --doc`; `mdbook test` is not the validation command because standalone
  mdBook rustdoc does not receive Cargo's dependency metadata.
- **[What's not finished yet](https://github.com/xupremix/incin/blob/master/docs/book/src/whats_not_finished.md)**: an
  honest, source-verified list of what still blocks real usage, kept
  separate so nothing above has to hedge every sentence.
- **[docs/capabilities.md](https://github.com/xupremix/incin/blob/master/docs/capabilities.md)**: generated per-backend,
  per-dtype operation support, straight from the registrations.
- **[Growth & architecture plans](https://github.com/xupremix/incin/tree/master/docs/growth/)**: task-by-task execution
  plans for adoption-facing features (named dimensions, CLI/IDE tooling,
  deployment, and more), each with its own dated status ledger. The older
  roadmap in `docs/growth/07-the-book.md` is historical and is not the
  source of current documentation status.

---

## License

<div align="center">

Dual-licensed under either

[**MIT**](https://github.com/xupremix/incin/blob/master/LICENSE_MIT) or [**Apache 2.0**](https://github.com/xupremix/incin/blob/master/LICENSE_APACHE)

at your option.

</div>
