# PROJECT MEMORY: Kindle Deep Learning Framework

> **Instructions for Claude / AI Assistants**: Copy-paste this entire document into your initial prompt context when continuing work on another PC or session. It contains the complete architectural blueprint, workspace structure, autotuning design, and current status.

---

## 1. Project Overview & Architecture

`kindle` is a high-performance, strongly-typed deep learning framework written in Rust. It enforces shape safety at compile time using `typenum` and symbolic dimensions (`s![...]`), while backing computations with native high-performance hardware execution across CPU, CUDA, and WebGPU.

### Repository Crate Structure
```
kindle/
├── crates/
│   ├── kindle-core/           # Core traits (Backend, Tensor, Shape, DType, Autograd Tape)
│   ├── kindle-backends/       # Hardware backends (CPU, CUDA, WGPU) and kernel dispatches
│   ├── kindle-macros/         # Proc macros (s![], idx![], #[module], import_model!)
│   ├── kindle-telemetry/      # Real-time event logging and execution tracing
│   ├── kindle-data/           # DataLoader and Dataset utilities
│   └── kindle-viz/            # TUI & visualization dashboard (ratatui)
└── docs/                      # Documentation and architecture specs
```

---

## 2. Core Backend Architecture: `KindleBackend<T, D>`

### Unified Type System
Instead of separate backend structs, `kindle` uses a single unified `KindleBackend<T, D>` parameterized by element type `T` (`f32`, `f16`, `i32`) and hardware device `D` (`Cpu`, `Cuda`, `Wgpu`):

```rust
#[derive(Debug, Clone, Copy)]
pub struct KindleBackend<T: DType = f32, D: Device = Cpu> {
    _marker: core::marker::PhantomData<(T, D)>,
}

pub type CpuBackend<T = f32> = KindleBackend<T, Cpu>;
pub type CudaBackend<T = f32> = KindleBackend<T, Cuda>;
pub type WgpuBackend<T = f32> = KindleBackend<T, Wgpu>;
```

### Static Monomorphization (`BackendDevice<T>`)
The `BackendDevice<T>` trait maps `D` to its hardware-specific storage struct (`CpuStorage`, `CudaStorage`, `WgpuStorage`) at compile time. This guarantees **zero runtime dispatch overhead**.

```rust
pub trait BackendDevice<T: DType>: Device {
    type Storage: Clone;
    type Var: Clone;
    type Grads: Clone;
}
```

---

## 3. CUDA GPU Autotuning Engine

GPU kernel execution is automatically optimized using a **3-tier autotuning engine** without requiring any macros or launch boilerplate from the end-user:

1. **Tier 1: Proc-Macro (`cuda_launch_config!`)**: Pre-computes block/grid constants for static shapes at compile time ($0\text{ ns}$ runtime cost).
2. **Tier 2: Hardware Occupancy (`cudaOccupancyMaxPotentialBlockSize`)**: Computes optimal block sizes ($32 \dots 1024$) and grid sizes ($N_{\text{SM}} \times 8$) based on CUDA device attributes.
3. **Tier 3: Empirical Profiling (`features = ["autotune"]`)**: Benchmarks matrix/kernel shape candidates on iteration 1 and caches the fastest configuration in `LRUCache<KernelKey, LaunchConfig>`.

### Grid-Stride CUDA Kernels
All CUDA C++ kernels in `crates/kindle-backends/src/cuda/ops/kernels/` utilize **Grid-Stride Loops** to natively handle any tensor shape (1D to N-D, odd dimensions, small or massive memory sizes):

```cpp
extern "C" __global__ void elementwise_binary(
    const float* lhs, const float* rhs, float* out, unsigned int numel, unsigned int op_type
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < numel; i += stride) {
        // Compute operation
    }
}
```

---

## 4. End-User API Experience

End-users write standard high-level Rust code. All hardware selection, autotuning, and tape tracking happen automatically behind the scenes:

```rust
use kindle::prelude::*;
use kindle_backends::KindleBackend;

// Select CUDA backend:
type B = KindleBackend<f32, Cuda>;

fn main() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    // Create tensors with static shape bounds
    let a = Tensor::<s![32, 512], B>::ones(&device)?;
    let b = Tensor::<s![32, 512], B>::ones(&device)?;

    // High-level operations automatically use autotuned GPU launches:
    let c = a.add(&b)?;
    let z = c.relu()?;

    Ok(())
}
```

---

## 5. Development & Verification Commands

- **Build Workspace**: `cargo build --workspace`
- **Run All Unit Tests**: `cargo test --workspace`
- **Run CUDA Tests**: `cargo test -p kindle-backends --features cuda`
- **Check Workspace Examples**: `cargo check --workspace --examples`
- **Run Training Demo**: `cargo run --example native_training_demo`

---

## 6. Current Status & Next Milestones

- [x] Unify CPU, CUDA, and WGPU backends under `kindle-backends`.
- [x] Implement CUDA memory management (`CudaStorage`) via `cudarc`.
- [x] Create CUDA Grid-Stride kernels and autotuning engine specification.
- [x] Clean up legacy backend references (`metal`, `candle`) from active targets.
- [ ] Refactor `kindle-backends` to export `KindleBackend<T, D>` with `BackendDevice<T>` trait.
- [ ] Implement `autotune` LRU cache in `cuda/gpu.rs`.
- [ ] Build PyTorch & NumPy comparison benchmark suite in `benches/`.
