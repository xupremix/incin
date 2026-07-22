# Kindle

Kindle is a deep learning framework in Rust focused on compile-time shape verification, developer ergonomics, and native multi-backend execution.

Kindle enforces shape and type bounds at compile time to prevent tensor shape mismatches and runtime out-of-bounds panics.

## Features

- **Compile-Time Shape Verification**: Tensor shapes can be statically tracked using `s![]` macros, catching dimension and matrix multiplication mismatches at build time.
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

- `kindle`: Primary facade crate providing unified imports and prelude.
- `kindle-core`: Statically-typed `Tensor` implementation, traits, and graph definitions.
- `kindle-backends`: Native CPU, CUDA, WGPU, and legacy Candle execution engines.
- `kindle-macros`: Procedural macros (`s!`, `idx!`, `module`, `import_model!`).
- `kindle-data`: Data loading utilities, dataset traits, and HuggingFace Hub support.
- `kindle-telemetry`: Event emission, transport streams, and graph snapshot recording.
- `kindle-viz`: Terminal UI (TUI) model graph visualizer.

## License

Dual-licensed under either MIT or Apache 2.0.
