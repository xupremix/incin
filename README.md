<div align="center">

# Incin

### A Rust deep learning framework with compile-time verification of tensor shapes, dtypes, devices, and gradient state.

[![Incin CI](https://github.com/xupremix/incin/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/xupremix/incin/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/incin.svg?color=orange)](https://crates.io/crates/incin)
[![docs.rs](https://img.shields.io/docsrs/incin?label=docs.rs)](https://docs.rs/incin)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](./LICENSE_MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-orange)](./Cargo.toml)

[Quick start](#quick-start) · [Why incin](#why-another-deep-learning-framework) · [The tooling](#the-tooling-is-the-feature) · [The Book](https://xupremix.github.io/incin)

</div>

Incin puts tensor shapes, dtypes, devices, and gradient state into Rust's type
system. A `matmul` between incompatible matrices is a compile error. Training a
frozen parameter is a compile error. Feeding `f32` where the kernel demands
`u32` indices? Compile error.

You find all of this out in `cargo check`, seconds after typing, instead of in
a runtime failure days into a training run.

## The demo

```rust,ignore
use incin::prelude::*;

let x: Tensor<s![4, 8], DefaultBackend> = Tensor::zeros(())?;
let w: Tensor<s![8, 2], DefaultBackend> = Tensor::zeros(())?;
let y = x.matmul(&w)?;        // [4, 8] x [8, 2] -> [4, 2]. Compiles.

let bad: Tensor<s![3, 8], DefaultBackend> = Tensor::zeros(())?;
let _ = x.matmul(&bad)?;      // inner dims: 8 vs 3. Does NOT compile.
```

Plain `cargo check` shows you the raw typenum soup:

```text
error[E0277]: Cannot contract dimension `UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>`
              with `UInt<UInt<UTerm, B1>, B1>`
```

[`cargo incin check`](#the-tooling-is-the-feature) intercepts that and prints what you
actually wanted:

```text
error[E0277]: Cannot contract dimension `[4, 8]` with `MatMulShape<[3, 8]>`
  --> src/main.rs:9:22

  └── 💡 [Typenum Translation Hints]:
      • 4  <= UInt<UInt<UInt<UTerm, B1>, B0>, B0>
      • 8  <= UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>
      • 3  <= UInt<UInt<UTerm, B1>, B1>
```

Same error, readable shapes, zero runtime cost. The types prove the shapes; the
numbers are covered too: every operation comes from one canonical catalog
backed by generated conformance vectors, and the differentiable ones carry
finite-difference gradcheck suites. And with
[incin-lsp](#the-tooling-is-the-feature), rust-analyzer shows shapes as decimal hints
directly in your editor while you type.

## Why another deep learning framework?

Because "shape mismatch at runtime" was never a necessary trade-off. Python
frameworks check shapes when the kernel launches; by then you've already paid
for the queue, the data load, and three minutes of your life. Incin moves that
check to the type system, which means:

- **If it compiles, the shapes agree.** Not "probably agree": the compiler
  proved it.
- **Dynamic shapes are still first-class.** Real models have runtime batch
  sizes and sequence lengths. `Tensor<Dyn, B>` composes with static shapes in
  the same program, and every dynamic transition is checked before a kernel
  sees bad metadata.
- **Every operation is declared once**, in a canonical catalog with generated
  semantics docs and conformance vectors. No drift between what the docs claim
  and what a backend does.
- **No silent fallbacks.** If a backend can't run an operation, you get a type
  error naming the gap, never a quiet transfer to another device.

Under the hood there's an autograd tape, AdamW/SGD, conv/pool/norm/loss layers,
safetensors checkpoints, ONNX import for supported graphs, and CPU executors for
all 158 backend-executable catalog operations, checked against generated
conformance vectors, with finite-difference gradchecks over the differentiable
kernels. The full honest status, including what's *not* done, lives in
[what's not finished](https://xupremix.github.io/incin/#/whats_not_finished).

## Quick start

Pull it from crates.io:

```toml
[dependencies]
incin = "0.1.0"
```

Define a model, train it, keep every shape known at compile time:

```rust,ignore
use incin::prelude::*;

#[module]
pub struct Mlp {
    fc1: Linear<s![784, 128]>,
    fc2: Linear<s![128, 10]>,
}

impl Mlp {
    pub fn new() -> Result<Self> {
        Ok(Self {
            fc1: Linear::build(())?,
            fc2: Linear::build(())?,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<s![dyn, 784]>,
    ) -> Result<Tensor<s![dyn, 10], DefaultBackend, f32, Grad>> {
        Ok(self.fc2.forward(self.fc1.forward(x)?.relu()?)?)
    }
}
```

`s![dyn, 784]` is the batch axis left dynamic; everything else is proven at
compile time. The output spells out `f32` and `Grad` because a forward pass
through parameters joins their gradient state into the result type. Swap in
`AdamW`, add a loss from
[`incin::nn`](https://docs.rs/incin/latest/incin/nn/index.html), iterate a
[`DataLoader`](https://xupremix.github.io/incin/#/data_loading),
and you have a training loop.

Prefer to explore first? The default build needs nothing but a Rust toolchain:

```rust,ignore
use incin::prelude::*;

fn main() -> Result<()> {
    let a = Cpu.randn(shape![2, 3])?;
    let b = Cpu.randn(shape![3])?;
    let sum = (&a + &b).reshape(shape![3, 2])?;
    println!("{}", sum.sum_keepdim(axis!(-1))?);
    Ok(())
}
```

More in [the Book's quickstart](https://xupremix.github.io/incin/#/quickstart),
including slicing with `i![]`, ONNX import via `import_model!`, and Hugging Face
downloads through the `data-hub` feature.

## The tooling *is* the feature

Type-level shape systems have a dirty secret: raw compiler diagnostics are
unreadable. Incin ships the fix as part of the framework, not as a plugin you
have to go find.

**`cargo incin`** wraps cargo itself. `check`, `build`, `test`, `run`, `clippy`
all pass through with typenum noise translated to decimal shapes as it streams.
It also inspects model files and translates pasted errors:

```bash
cargo install incin --bin cargo-incin --locked

cargo incin check                  # readable shape errors, live
cargo incin inspect model.safetensors
cargo incin translate "<paste an error>"
cargo incin doctor                 # what backends work on this machine?
```

**`incin-lsp`** proxies rust-analyzer and rewrites diagnostics, hover types,
and inlay hints with the same engine. Tensor types render as `[4, 8]`, not
`UInt<UInt<...>>`. Verified in VS Code and Neovim:

<p align="center">
  <img src="https://raw.githubusercontent.com/xupremix/incin/master/docs/assets/editors/vscode-shape-diagnostic.png" alt="A reshape error rewritten by incin-lsp in VS Code">
</p>

**`incin-viz`** is a terminal UI for watching training runs live (losses,
gradient norms, memory, graph snapshots) over a local telemetry socket.

Three tools, one diagnostic engine, no configuration required.

<details>
<summary><strong>All the ways to install</strong></summary>

```toml
# Default: std + native CPU backend
incin = "0.1.0"

# Add WGPU
incin = { version = "0.1.0", features = ["wgpu"] }

# CUDA-only application
incin = { version = "0.1.0", default-features = false,
          features = ["std", "cuda"] }

# Candle interop
incin = { version = "0.1.0", features = ["external-candle"] }
```

From a checkout instead of the registry, point the dependency at your clone or
install the binaries with `--path`.

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

</details>

## What's inside

```text
incin/
├── incin               Facade: prelude, macros, cargo-incin CLI
├── incin-core          Typed tensors, op catalog, autograd contracts, graphs
├── incin-backends      CPU (complete) · CUDA/WGPU/Metal (preview) · Candle adapter
├── incin-data          Data loading, datasets, Hugging Face Hub
├── incin-telemetry     Event emission and transport streams
├── incin-viz           Terminal training visualizer (+ plugin API)
├── incin-diagnostics   Typenum → decimal shape translation engine
└── incin-lsp           Editor proxy: rust-analyzer in, readable shapes out
```

A `Tensor<S, B, K, G>` carries its Shape, Backend, dtype Kind, and Gradient
state as type parameters. Methods build validated operation descriptors and
dispatch to the backend's executor. Missing executor? Type error. Wrong device?
Type error. Nothing switches backends behind your back.

<details>
<summary><strong>Facade feature matrix</strong></summary>

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
| `update-check` | stable | `std` | no | `std`, `dep:ureq` | none | Lets `cargo incin doctor --check-updates` ask crates.io whether a newer incin exists. Off by default: it is the only feature that can reach the network, and no build should gain that ability without asking for it. |
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

</details>

<details>
<summary><strong>Environment variables</strong></summary>

| Variable | Purpose |
|---|---|
| `INCIN_HUB_CACHE_DIR` | Custom cache directory for Hub downloads (defaults to `~/.cache/huggingface/hub`). |
| `INCIN_HUB_TOKEN` | Authorization token for private Hugging Face Hub repositories. |
| `INCIN_NO_META` | Set to `1` to bypass the `.incin_meta` cache and force full ONNX re-parsing during macro compilation. |

</details>

<details>
<summary><strong>Editor setup</strong></summary>

Install the proxy, point your editor at it, done:

```bash
cargo install incin-lsp --locked
```

| Editor | Status | Guide |
|---|---|---|
| VS Code | Verified end to end | [`editors/vscode/README.md`](editors/vscode/README.md) |
| Neovim 0.11+ | Verified end to end | [`editors/nvim/README.md`](editors/nvim/README.md) |
| RustRover | External-tool fallback verified | [`editors/rustrover/README.md`](editors/rustrover/README.md) |

Neovim screenshot for good measure:

![The same reshape error rewritten by incin-lsp in Neovim](https://raw.githubusercontent.com/xupremix/incin/master/docs/assets/editors/neovim-shape-diagnostic.png)

</details>

## Where to go next

- **[The Book](https://xupremix.github.io/incin/)** is the task-oriented guide,
  from five-line tensors up to a Transformer block.
- **[What's not finished](https://xupremix.github.io/incin/#/whats_not_finished)**
  is the honest list. GPU backends cover a subset; read it before planning a
  CUDA training run.
- **[Capability matrix](https://github.com/xupremix/incin/blob/master/docs/capabilities.md)**
  is per-operation, generated from executor registrations, never hand-edited.
- **[docs/GUIDE.md](https://github.com/xupremix/incin/blob/master/docs/GUIDE.md)**
  is the repository-side tour of the shape-proof system.
- **[The deep dive](https://xupremix.github.io/incin/#/deep_architecture)** runs
  five rendered chapters on the execution model: how a typed call becomes a
  kernel, what each stage proves, and how to add your own backends, devices,
  and dtypes.
- Found something? [Open an issue](https://github.com/xupremix/incin/issues/new/choose).
  The templates make it fast.

<div align="center">

<em>Because a shape mismatch belongs in <code>cargo check</code>, not on epoch 40, at 3 AM, on a Saturday.</em>

</div>

<div align="center">

## License

Dual-licensed under either [**MIT**](./LICENSE_MIT) or [**Apache 2.0**](./LICENSE_APACHE) at your option.

</div>
