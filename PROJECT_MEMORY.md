# PROJECT MEMORY: Kindle Deep Learning Framework

> **Instructions for Claude / AI Assistants**: Copy-paste this entire document into your initial prompt context when continuing work on another PC or session. It contains the complete architectural blueprint, workspace structure, autotuning design, tensor shape ranges & indexing syntax, recent changelog, and current status.

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

## 3. Shape Syntax, Ranges, & Slicing (`s![]`, `idx![]`)

Kindle features compile-time type-checked shape bounds and multidimensional range slicing:

### A. Static Shape Declarations (`s![]`)
```rust
// Static 4D Tensor: [Batch=2, Channels=3, Height=224, Width=224]
type ImageBatch = s![2, 3, 224, 224];

// Dynamic shapes or symbolic dimensions:
type DynamicBatch = s![Dyn, 3, 224, 224];
```

### B. Multi-Dimensional Indexing & Slicing Ranges (`idx![]`)
Ranges translate into zero-allocation type bounds:
- `0..5`: Range slice along a dimension (`Slice<U0, U5>`).
- `..`: Take full dimension (`Full`).
- `...` or `..`: Fill missing dimensions (`Ellipsis`).
- `-1`: Inferred dimension size in `reshape`.

```rust
// Slice tensor `t` of shape [10, 20, 30]:
// Takes 0..5 on dim 0, full dim 1, and 15..30 on dim 2 -> output shape [5, 20, 15]
let view = t.slice::<idx![0..5, .., 15..30]>()?;
```

---

## 4. CUDA GPU Autotuning Engine

GPU kernel execution is automatically optimized using a **3-tier autotuning engine** without requiring any macros or launch boilerplate from the end-user:

1. **Tier 1: Proc-Macro (`cuda_launch_config!`)**: Pre-computes block/grid constants for static shapes at compile time ($0\text{ ns}$ runtime cost).
2. **Tier 2: Hardware Occupancy (`cudaOccupancyMaxPotentialBlockSize`)**: Computes optimal block sizes ($32 \dots 1024$) and grid sizes ($N_{\text{SM}} \times 8$) based on CUDA device attributes.
3. **Tier 3: Empirical Profiling (`features = ["autotune"]`)**: Benchmarks matrix/kernel shape candidates on iteration 1 and caches the fastest configuration in `LRUCache<KernelKey, LaunchConfig>`.

---

## 5. End-User API Experience

End-users write standard high-level Rust code. All hardware selection, autotuning, and tape tracking happen automatically behind the scenes:

```rust
use kindle::prelude::*;
use kindle_backends::KindleBackend;

// Select CUDA backend:
type B = KindleBackend<f32, Cuda>;

fn main() -> Result<()> {
    let device = CudaDevice::new(0)?;
    
    let a = Tensor::<s![32, 512], B>::ones(&device)?;
    let b = Tensor::<s![32, 512], B>::ones(&device)?;

    let c = a.add(&b)?;
    let z = c.relu()?;

    Ok(())
}
```

---

## 6. Recent Project Changelog

- **`c48c530`**: `docs: add PROJECT_MEMORY.md, autotuning spec, and fix training demo & doctests`.
- **`1c8cb35`**: `chore: clean up obsolete Claude info and legacy planning files`.
- **`a93ca32`**: `feat: complete CudaBackend integration and test/example fixes`.
- **`74f197b`**: `feat: complete Phase 4 and Phase 5`.
- **`243e6a5`**: `Fix compilation: remove orphaned CpuBuffer::Cuda references in cpu backend`.
- **`f854c18`**: `Phase 4: Replaced metal config with wgpu in core, backends, and app lib`.
- **`5c49e09`**: `Fix tests and examples after backend refactoring`.
- **`68bc1f6`**: `feat: complete wgpu conv implementations and telemetry wiring`.
- **Crate Consolidation**: Deleted `kindle-native` & `kindle-wgpu`, moving CPU, CUDA, and WGPU implementations under `kindle-backends`.
- **Legacy Cleanup**: Removed deprecated `metal` and `candle` dependencies from default build targets.

---

## 7. Development & Verification Commands

- **Build Workspace**: `cargo build --workspace`
- **Run All Unit Tests**: `cargo test --workspace`
- **Run CUDA Tests**: `cargo test -p kindle-backends --features cuda`
- **Check Workspace Examples**: `cargo check --workspace --examples`
- **Run Training Demo**: `cargo run --example native_training_demo`

---

## 8. Current Status & Next Milestones

- [x] Unify CPU, CUDA, and WGPU backends under `kindle-backends`.
- [x] Implement CUDA memory management (`CudaStorage`) via `cudarc`.
- [x] Create CUDA Grid-Stride kernels and autotuning engine specification.
- [x] Clean up legacy backend references (`metal`, `candle`) from active targets.
- [ ] Refactor `kindle-backends` to export `KindleBackend<T, D>` with `BackendDevice<T>` trait.
- [ ] Implement `autotune` LRU cache in `cuda/gpu.rs`.
- [ ] Build PyTorch & NumPy comparison benchmark suite in `benches/`.
