# Kindle 🔥

> "Lighting the way for static, compile-time verified deep learning."

**Kindle** is an experimental deep learning framework in Rust focusing on ergonomic syntax, uncompromised speed, and explicit statically verified mathematical correctness.

It wraps robust tensor engines (like `candle` and `burn`) and enforces strict mathematical type checking at compile time.

## 🚀 The Philosophy
- **Compile-Time Shape Verification**: Your matrix multiplications and broadcasts are verified when compiling, reducing out-of-bounds runtime panic overhead.
- **Python-like Ergonomics**: We use Rust `macro_rules!` (`s!`, `idx!`) to recreate the seamless slicing and shape specification syntax of PyTorch/NumPy.
- **Backend Agnostic API Design**: While natively using HuggingFace's `candle`, the abstract `Backend` trait enables plugging in Burn or wgpu engines effortlessly.
- **Zero-Cost Abstractions**: The static types `S`, `T`, `D`, `G` map entirely to `PhantomData` markers, imposing ZERO overhead on runtime buffers.

## 🛠️ Setup & Requirements

Because Kindle supports ONNX protocol graph serialization natively, you must have the **Protocol Buffers Compiler (`protoc`)** installed on your system to build the crate.

### Ubuntu / Debian
```bash
sudo apt-get install -y protobuf-compiler
```

### macOS (Homebrew)
```bash
brew install protobuf
```

### Environment Variables
You can configure internal macro and Hub behavior using the following environment variables:
- `KINDLE_HUB_CACHE_DIR`: Specifies a custom cache directory for downloaded models (overrides `~/.cache/huggingface/hub`).
- `KINDLE_HUB_TOKEN`: Sets your HuggingFace authorization token for accessing private or gated repositories.
- `KINDLE_DISABLE_META_CACHE`: Set to `1` or `true` to force `import_model!` to bypass the lightning-fast `.kindle_meta` JSON cache and do a full `.safetensors`/`.onnx` graph re-parse during `cargo build`.

## 🌟 Quick Tour

### Type-Safe ResNet Definition
```rust
use kindle::prelude::*;

#[kindle::module]
struct ResNetBlock {
    conv1: Conv2d<...>,
}

impl ResNetBlock {
    #[kindle::forward]
    fn forward(&self, x: Tensor<s![dyn, 64, 224, 224]>) -> Result<Tensor<s![dyn, 64, 224, 224]>> {
        // Rust's trait solver will completely verify the tensor mathematics!
        let x = self.conv1.forward(x)?;
        Ok(x)
    }
}
```

### Python-like Slicing
```rust
use kindle::prelude::*;

let t = Tensor::zeros([2, 3, 4]).unwrap();
// PyTorch equivalent: t[:, 1:3, 0]
let sliced = t.slice(idx![.., 1..3, 0]).unwrap();
```

### Multi-Threaded DataLoaders
```rust
use kindle_data::prelude::*;

let iterator = (0..100).into_iter();
// Magically utilizes all CPU cores for mapping!
let sum: i32 = iterator.into_par_loader().map(|x| x * 2).sum();
```

## 🏗️ Crates
- `kindle-core`: The underlying statically-typed `Tensor` implementation and operations.
- `kindle-macros`: Proc macros (`idx!`, `s!`, `module`, `forward`) that power the developer ergonomics.
- `kindle-data`: Utilities for data loading (`DataLoaderExt`) and simple HuggingFace Hub downloads.

## 🤝 Contribution
This project is an experimental prototype to study strict static ML guarantees. Contributions to macros and backend integrations are highly welcome!
