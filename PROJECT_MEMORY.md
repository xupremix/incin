# PROJECT MEMORY: Kindle Deep Learning Framework

> **Instructions for Claude / AI Assistants**: Copy-paste this entire document into your initial prompt context when continuing work on another PC or session. It contains the complete architectural blueprint, workspace structure, autotuning design, tensor shape ranges & indexing syntax, API encapsulation rules, recent changelog, and current status.

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
├── .agents/                   # Agent rules, workflows, and API guidelines
└── docs/                      # Documentation and architecture specs
```

---

## 2. Core Backend Architecture: `KindleBackend<T, D>`

As of 0.2.0, `KindleBackend<T, D>` maps static device markers to their concrete
backends and maps `Dyn` to an enum-backed runtime dispatcher. Tensor device
metadata comes only from the backend:

```rust
pub type KindleBackend<T, D> = <D as BackendFor<T>>::Backend;
pub struct Tensor<S, B, K = B::FloatElem, G = Grad> { /* private fields */ }
```

### Static monomorphization and runtime dispatch

`Cpu`, `Wgpu<N>`, and `Cuda<N>` retain concrete, zero-overhead backend types.
`KindleBackend<T, Dyn>` uses `DispatchBackend<T, Dyn>` and chooses CPU, WGPU,
or CUDA from a validated `DeviceId`. CPU supports all storage dtypes; WGPU and
CUDA advertise F32 only.

```rust
pub trait BackendFor<T: DType>: Device {
    type Backend: Backend<Device = Self, FloatElem = T, IntElem = i64>;
}
```

`BackendFor<T>` is sealed. Cross-device and cross-family movement is expressed by
`TransferTo<NewD>`; its `Output` backend is selected by `BackendFor`, and tensor or
variable payloads are validated and rebuilt in destination-native storage.

Allocating layers expose only `build(args)`. The tuple lists dynamic dimensions,
backend dtype, backend device, dynamic optional flags, and runtime parameters in
that order; compile-time-static positions are omitted.

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

The forward architecture for dtype-polymorphic kernels is tracked in
`docs/DTYPE_KERNEL_ARCHITECTURE.md`. Kernel bodies are defined by operation
family and specialized lazily from explicit storage, compute, accumulator,
layout, vector-width, hardware-feature, and math-mode policy. Structured
GEMM/conv/attention should prefer tuned libraries; generated templates are
primarily for pointwise, reduction, indexing, and fusion glue.

CPU elementwise arithmetic and unary operations now use one operation-family
dispatcher with typed F16/BF16/F32/F64 kernels for every layout. The shared
unary/binary iteration plans supply allocation-free zero-stride indexing for
general broadcast/views, remove unit axes, coalesce compatible dimensions, and
use a shared serial odometer with a tight inner loop. F32/F64 contiguous and
scalar-broadcast arithmetic uses runtime-selected AVX2 on x86-64 with scalar
tails. At 2 Mi elements, Rayon partitions dense work into measured 128
Ki-element chunks that reuse the same allocation-free AVX2 writers. A single
dtype-family projection macro also maps those writers onto normalized
dense-broadcast plans with contiguous/scalar inner strides; generic arbitrary-
stride loops use Rayon from 256 Ki.

The target GPU design uses a **3-tier autotuning engine** without requiring
launch boilerplate from the end-user. Candidate enumeration, typed problem
keys, synchronized CUDA-event median selection, compute-capability identity,
and a bounded device/workload cache are implemented and consumed by
pointwise/reduction dispatch. Tier 3 is therefore implemented for these two
families; static-shape and occupancy-driven pruning remain architectural
targets:

1. **Tier 1: Proc-Macro (`cuda_launch_config!`)**: Pre-computes block/grid constants for static shapes at compile time ($0\text{ ns}$ runtime cost).
2. **Tier 2: Hardware Occupancy (`cudaOccupancyMaxPotentialBlockSize`)**: Computes optimal block sizes ($32 \dots 1024$) and grid sizes ($N_{\text{SM}} \times 8$) based on CUDA device attributes.
3. **Tier 3: Empirical Profiling (`features = ["autotune"]`)**: Benchmarks matrix/kernel shape candidates on iteration 1 and caches the fastest configuration in `LRUCache<KernelKey, LaunchConfig>`.

The implemented CUDA foundation is narrower: buffers and host staging retain
F16/BF16/F32/F64 byte widths, one unary/binary source-template family renders
typed storage and compute conversions with dtype-specific keys, and the raw
launch ABI checks rendered dtype/width plus all 32-bit metadata before GPU
work. Dense pointwise dispatch distinguishes scalar ILP (`u1`/`u2`/`u4`) from
aligned packed access (`half2`/`bfloat162`/`float4`/`double2`), with masked
scalar tails for incomplete packets. `SupportsDType` remains F32-only because
normalization, shape, embedding, loss, quantization, and gradient paths are not
all dtype-safe yet. See `docs/DTYPE_KERNEL_ARCHITECTURE.md` for the physical
milestone order and performance gates.

CUDA reduction source is generated lazily by dtype, operation, layout, and
indexed/non-indexed ABI. F16/BF16 accumulate in F32; F32/F64 remain native.
Contiguous last-axis reductions use warp shuffles and one shared value per warp,
while arbitrary views use the checked strided template. Do not restore the
removed checked-in F32-only `reduce.cu` module.

CUDA normalization source is also generated lazily. Layer norm uses per-thread
Welford state, warp shuffles, one shared record per warp, and a fused affine
write; batch-norm inference shares the storage/compute conversion vocabulary.
The obsolete F32-only `norm.cu` module was removed. Keep non-contiguous views
on an explicit fallback/error path until a layout-aware implementation exists;
never reinterpret them as dense rows.

`kindle-backends/src/dtype_policy.rs` is the authoritative capability seam.
It distinguishes the ability to store bytes from initialization and operation-
family support and resolves storage, compute, accumulator, and output dtypes.
Do not add backend-local dtype allowlists; extend this table and its exhaustive
matrix tests instead.

The shared iteration planner classifies contiguous, scalar-left, scalar-right,
and strided pointwise layouts. CUDA uses distinct template/cache families for
these ABIs: dense/scalar launches pass only pointers, offsets, and element
count, while the correctness fallback alone uploads shape and stride arrays.

---

## 5. Agent Guidelines & Public API Encapsulation (`.agents`)

All AI agents working on this repository MUST follow these strict encapsulation rules:

1. **`pub(crate)` is Default**: Never expose internal dispatch functions, kernel state, or raw buffers as `pub`. Always default to `pub(crate)` unless an item is explicitly required by end-users.
2. **Private Struct Fields**: Even for public trait types (e.g. `CudaStorage`), internal fields like `buffer` or `shape` must be `pub(crate)` to prevent downstream mutation or memory safety leaks.
3. **Graphify Workflow**: Run `graphify update .` after code modifications to keep `graphify-out/` current. (Note: `graphify-out/` is ignored by `.gitignore`).

---

## 6. End-User API Experience

End-users write standard high-level Rust code. All hardware selection, autotuning, and tape tracking happen automatically behind the scenes:

```rust
use kindle::prelude::*;
use kindle_backends::KindleBackend;

// Select CUDA backend:
type B = KindleBackend<f32, Cuda>;

fn main() -> Result<()> {
    let a = Tensor::<s![32, 512], B>::ones(())?;
    let b = Tensor::<s![32, 512], B>::ones(())?;

    let c = a.add(&b)?;
    let z = c.relu()?;

    Ok(())
}
```

---

## 7. Recent Project Changelog

- **2026-07-22 tuning coordinator wiring + occupancy pruning**: committed the
  in-flight dtype/kernel specialization + CUDA autotuning WIP (dtype policy,
  iteration planner, typed CUDA kernel generation, tuning cache), after fixing
  a cpu-only test-build break it had introduced (`iteration.rs` test module
  using cuda-gated APIs without a matching `cfg`). Fixed pre-existing
  cuda-only/wgpu-only `clippy -D warnings` failures on `main` (mismatched
  feature gates in `backend_kind.rs`, `cpu/creation.rs`, `tests/ops.rs`,
  `tests/gradient_parity.rs`). Wired `tuning.rs`'s previously-dead in-flight
  suppression coordinator (`claim_tuning`/`TuningPermit`) into CUDA pointwise
  and reduction dispatch, so concurrent callers tuning the same key block on
  the in-progress measurement instead of redundantly benchmarking it; added
  Tier-2 occupancy pruning for pointwise autotune candidates via `cudarc`'s
  `cuOccupancyMaxActiveBlocksPerMultiprocessor`. All CUDA-side work in this
  entry is compile/clippy-verified only — no CUDA hardware in this
  environment.
- **2026-07-21 full-codebase audit**: reviewed every crate for correctness,
  security, and API-design issues. Found and documented in `ROADMAP.md`:
  CUDA ops panic on first call (`Arc::get_mut` refcount bug), CPU elementwise
  ops silently downcast all dtypes to f32, WGPU and CUDA autograd tapes are
  fully disconnected (no gradients ever produced), unchecked shape-multiplication
  overflow, dynamic-shape broadcast performs no compatibility check, and
  `pub` API leakage in `cuda`/`cpu` backends + a new `kindle-core` `TRACING_GRAPH`
  leak. Renamed `dev/refactor` → `main`; fixed a broken CI feature flag
  (`kindle-backends/native` didn't exist); linked `origin` to
  `github.com/xupremix/kindle` (pushed as a new branch, existing unrelated
  `master` history left untouched). See `ROADMAP.md` for the full findings and
  phased implementation plan.
- **`d3601a4`**: `docs: add tensor shape ranges, index syntax, and changelog to PROJECT_MEMORY.md`.
- **`c48c530`**: `docs: add PROJECT_MEMORY.md, autotuning spec, and fix training demo & doctests`.
- **`1c8cb35`**: `chore: clean up obsolete Claude info and legacy planning files`.
- **`a93ca32`**: `feat: complete CudaBackendImpl integration and test/example fixes`.
- **`74f197b`**: `feat: complete Phase 4 and Phase 5`.
- **`243e6a5`**: `Fix compilation: remove orphaned CpuBuffer::Cuda references in cpu backend`.
- **`f854c18`**: `Phase 4: Replaced metal config with wgpu in core, backends, and app lib`.
- **`5c49e09`**: `Fix tests and examples after backend refactoring`.
- **`68bc1f6`**: `feat: complete wgpu conv implementations and telemetry wiring`.
- **Crate Consolidation**: Deleted `kindle-native` & `kindle-wgpu`, moving CPU, CUDA, and WGPU implementations under `kindle-backends`.
- **Legacy Cleanup**: Removed deprecated `metal` and `candle` dependencies from default build targets.

---

## 8. Development & Verification Commands

- **Build Workspace**: `cargo build --workspace`
- **Run All Unit Tests**: `cargo test --workspace`
- **Run CUDA Tests**: `cargo test -p kindle-backends --features cuda`
- **Check Workspace Examples**: `cargo check --workspace --examples`
- **Run Training Demo**: `cargo run --example native_training_demo`

---

## 9. Current Status & Next Milestones

- [x] Unify CPU, CUDA, and WGPU backends under `kindle-backends`.
- [x] Implement CUDA memory management (`CudaStorage`) via `cudarc`.
- [x] Create CUDA Grid-Stride kernels and autotuning engine specification.
- [x] Clean up legacy backend references (`metal`, `candle`) from active targets.
- [ ] Refactor `kindle-backends` to export `KindleBackend<T, D>` with `BackendDevice<T>` trait.
- [x] Implement a bounded `autotune` launch-result cache with canonical typed
      problem keys and pointwise/reduction dispatch integration.
- [x] Measure pointwise/reduction candidates with synchronized CUDA events and
      scope results by compute capability.
- [x] Wire the concurrent first-use (in-flight) suppression coordinator into
      CUDA pointwise/reduction dispatch, so racing callers for the same
      tuning key block on the in-progress measurement instead of redundantly
      benchmarking it themselves.
- [x] Add occupancy pruning for CUDA pointwise autotune candidates
      (`cuOccupancyMaxActiveBlocksPerMultiprocessor` via `cudarc`), skipping
      block sizes the driver reports as non-viable before timing them.
      Reduction candidates are not pruned yet (block size there is a launch
      parameter, not a compiled-kernel axis, so the function handle isn't
      available at selection time without restructuring).
- [ ] Add device UUID/driver/compiler identity to the tuning cache key, and
      persistent/telemetrized winning launch plans.
- [ ] Build PyTorch & NumPy comparison benchmark suite in `benches/`.
