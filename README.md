# Kindle

Kindle is a deep learning framework in Rust focused on compile-time shape verification, developer ergonomics, and native multi-backend execution.

Kindle enforces shape and type bounds at compile time to prevent tensor shape mismatches and runtime out-of-bounds panics.

## Features

- **Compile-Time Shape Verification**: Tensor shapes can be statically tracked using `s![]` macros, catching dimension and matrix multiplication mismatches at build time.
- **Named Dimensions**: `symbolic_dim!(Batch, Seq, Feature)` gives an axis a name that's part of its *type* — `Tensor<s![Batch, Feature]>` and `Tensor<s![Batch, Seq]>` are different types even when both axes are the same runtime size, so passing one where the other is expected is a compile error, not a silently-wrong result. Unlike PyTorch's named tensors (runtime, opt-in, easy to bypass), a Kindle name can't be forgotten — see `crates/kindle/examples/named_dims_safety`.
- **Python-Style Slicing**: Expressive index manipulation macros (`idx![]`) enable Python-like slicing and dynamic index selection.
- **ONNX Model Importing**: Convert ONNX models directly into fully-typed Rust module structs at compile time with `import_model!`.
- **Multi-Backend Support**: Abstract `Backend` design with native execution implementations for CPU, CUDA, and WGPU, plus Candle interop via `legacy`.
- **Parallel Data Loading**: Efficient multi-threaded batching and data loading (`kindle-data`).
- **Telemetry & Visualization**: Real-time graph extraction and terminal UI visualization tools (`kindle-telemetry` and `kindle-viz`).

## Setup & Requirements

Kindle serializes ONNX protocol graphs natively using Protocol Buffers. Building the workspace requires the Protocol Buffers compiler (`protoc`).

### Ubuntu / Debian
```bash
sudo apt-get install -y protobuf-compiler
```

### macOS (Homebrew)
```bash
brew install protobuf
```

### Environment Variables

- `KINDLE_HUB_CACHE_DIR`: Specifies a custom cache directory for downloaded models (defaults to `~/.cache/huggingface/hub`).
- `KINDLE_HUB_TOKEN`: Authorization token for downloading private HuggingFace Hub repositories.
- `KINDLE_NO_META`: Set to `1` to bypass `.kindle_meta` cache and force full ONNX graph re-parsing during macro compilation.

## Quick Start

### Tensor Creation & Shape Macro

```rust,ignore
use kindle::prelude::*;

// Statically dimensioned 2x3 tensor on default CPU backend
let static_tensor: Tensor<s![2, 3]> = Tensor::zeros(())?;

// Dynamically dimensioned tensor
let dynamic_tensor: Tensor<Dyn> = Tensor::zeros([2, 3])?;
```

### Module Definition & Forward Pass

```rust,ignore
use kindle::prelude::*;

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
use kindle::prelude::*;

let t = Tensor::<Dyn>::zeros([2, 3, 4])?;
let sliced = t.slice(idx![.., 1..3, 0])?;
```

### ONNX Model Import

```rust,ignore
use kindle::prelude::*;

import_model!("resnet18.onnx", ResNet18);

fn main() -> Result<()> {
    let mut model = ResNet18::<CpuBackendImpl>::new();
    println!("Parsed ONNX graph into Rust AST!");
    Ok(())
}
```

## Workspace Crates

- `kindle`: Primary facade crate providing unified imports and prelude, plus
  the `cargo-kindle` CLI binary.
- `kindle-core`: Statically-typed `Tensor` implementation, traits, and graph definitions.
- `kindle-backends`: Native CPU, CUDA, WGPU, and legacy Candle execution engines.
- `kindle-macros`: Procedural macros (`s!`, `idx!`, `module`, `import_model!`).
- `kindle-data`: Data loading utilities, dataset traits, and HuggingFace Hub support.
- `kindle-telemetry`: Event emission, transport streams, and graph snapshot recording.
- `kindle-viz`: Terminal UI (TUI) model graph visualizer.
- `kindle-diagnostics`: Typenum-to-decimal shape diagnostic humanization,
  shared by the CLI and the editor LSP proxy — see [CLI](#cli--cargo-kindle)
  and [Editor / IDE Support](#editor--ide-support) below.
- `kindle-lsp`: Transparent LSP proxy that routes rust-analyzer through
  `kindle-diagnostics` so shape errors and inlay hints are humanized live in
  your editor.

## CLI — `cargo kindle`

`cargo-kindle` wraps `cargo check`/`build`/`test`/`run`, rewriting the
compiler's typenum shape errors (`UInt<UInt<...>>` walls) into plain decimals
live in your terminal — the same translation the editor extensions use. It
also inspects exported model files and translates pasted error text ad hoc.

### Install

```bash
cargo install --path crates/kindle
```

This builds and installs the `cargo-kindle` binary (the crate's only binary
target); invoke it as `cargo kindle <subcommand>` from any Kindle project.

### Usage

```bash
cargo kindle check                # cargo check, with humanized shape errors
cargo kindle build --release      # any cargo subcommand + its normal args work
cargo kindle test
cargo kindle run
cargo kindle inspect model.gguf   # metadata for .safetensors / .gguf / .onnx files
cargo kindle translate "..."      # translate pasted error text (arg or stdin)
```

Flags: `--raw` (skip translation, show the compiler's raw output), `--explain`
(append a plain-English shape-rule explanation for common errors), `--help`.

## Editor / IDE Support

The same humanization is available live inside your editor via `kindle-lsp`,
a thin proxy that sits between your editor and rust-analyzer and rewrites
diagnostics and shape inlay hints through `kindle-diagnostics` before they
reach you — no forked rust-analyzer, no per-editor parsing logic. See
`docs/growth/02-ide-extensions.md` for the full architecture and verification
status of each client below.

| Editor | Status | Install guide |
|---|---|---|
| VS Code | Ships; activation + rust-analyzer config-rewriting verified by an automated test against a real VS Code (`npm test`). Full humanized-diagnostic pipeline (real `kindle-lsp` + rust-analyzer) not yet exercised end-to-end | [`editors/vscode/README.md`](editors/vscode/README.md) |
| Neovim (0.11+, or nvim-lspconfig on older versions) | Verified against a real Neovim install | [`editors/nvim/README.md`](editors/nvim/README.md) |
| RustRover / IntelliJ | External-tool fallback verified; native LSP-mode integration unverified | [`editors/rustrover/README.md`](editors/rustrover/README.md) |

VS Code and Neovim both need the proxy on `PATH` first (RustRover's shipped
fallback only needs the CLI above):

```bash
cargo install --path crates/kindle-lsp --bin kindle-lsp
```

## Documentation

- **Growth & architecture plans** (`docs/growth/`): task-by-task execution
  plans for adoption-facing features — named dimensions, this CLI/IDE
  tooling, deployment, and more — each with its own dated status ledger.
- **The Book ("Kindlenomicon"):** planned, not yet built. See
  `docs/growth/07-the-book.md` for the full chapter outline; it will ship as
  an mdBook at `docs/book/`, built with `mdbook build docs/book`, deployed to
  GitHub Pages. Until it lands, this README, the growth docs, and
  `crates/kindle/examples/` are the best starting points.

## License

Dual-licensed under either MIT or Apache 2.0.
