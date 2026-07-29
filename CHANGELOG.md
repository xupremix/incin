# Changelog

> The workspace intentionally remains at `0.0.0`. Dated version headings below
> are development snapshots retained for traceability, not published releases.

All notable changes to the Incin framework will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Structured backward failures and `NanPolicy` (`GRD-005`):** backward
  recipes return `Result` — the 115 `.expect("unbroadcast lhs (add)")` and
  `.unwrap()` sites inside them propagate now — and a failure arrives as
  `BackwardError`, naming the tensor and whether the non-finite value came from
  a recipe or from summing two contributions. NaN checking is an
  `ExecutionPolicy` axis (`incin_core::exec::check_gradients(|| ..)`), read by
  every backend's walk, and defaults to off because the check reads every
  element of every gradient. The CUDA backend had no check at all before this.
- **Backend-neutral autograd tape (`GRD-003`):** `incin_core::exec::tape` now
  owns the graph `PROPOSALS.md` §1.2.5 puts in the core — one `TensorId`, a
  `TapeNode` holding a node's inputs and backward recipe, a `Tape` owning the
  nodes, and the reverse walk that consumes them. `TapeStorage` is the whole of
  what a backend still supplies: identity, a ones seed, a fallible accumulate,
  and a non-finite predicate. The CPU backend runs on it; `GRD-004` moves WGPU
  and CUDA. The walk takes its nodes by value, so a backward recipe that itself
  records — every convolution backward does — cannot re-enter the tape it is
  draining.

- **`GradMode` and `no_grad` (`GRD-002`):** the type-level `Grad`/`NoGrad`
  markers now reach the layer that records. `GradMode` joins the other axes on
  `ExecutionPolicy`, is derived from `RequiresGrad::requires_grad` rather than
  declared beside it, and travels to the backends through the ambient policy
  `GRD-001` already installs. Every frontend operation runs its kernel under
  the mode its *result*'s marker derives, and the CPU, WGPU, and CUDA tapes
  refuse a push when that mode does not record — so a `NoGrad` chain creates no
  autograd node and retains no saved tensor, as `PROPOSALS.md` §1.2.5 requires.
  `incin_core::exec::no_grad(|| ..)` is the inference form and applies to
  `Grad` tensors too; an operand can only tighten the ambient mode, never raise
  it. `cpu::tape_depth()` (likewise `wgpu`, `cuda`) is newly public so the
  guarantee can be counted rather than assumed.
- **Dtype/kernel specialization architecture:** new `dtype_policy.rs` (single
  storage/compute/accumulator/output dtype resolver for CPU/CUDA/WGPU),
  `iteration.rs` (backend-neutral broadcast/layout iteration plan), and
  `kernel.rs` (typed CUDA source generation shared across pointwise,
  reduction, and normalization operation families). See `PROPOSALS.md` §3 for the consolidated design and phased roadmap.
- **CUDA autotuning foundation** (new `autotune` feature, `tuning.rs`): typed
  canonical launch-candidate keys, CUDA-event warmup/sample measurement,
  compute-capability-scoped caching, a Condvar-coordinated in-flight
  suppression claim so concurrent callers tuning the same problem/device/
  workload key block on the in-progress measurement instead of redundantly
  benchmarking it, and Tier-2 occupancy pruning for pointwise candidates
  (`cuOccupancyMaxActiveBlocksPerMultiprocessor`, conservative — only drops a
  candidate the driver confirms has zero active blocks). CUDA reductions and
  layer/batch norm are now generated from the dtype policy (replacing the
  checked-in F32-only `norm.cu`/`reduce.cu`) with warp/block cooperation and
  Welford accumulation; CUDA pointwise dispatch adds scalar-ILP and aligned
  packed (`half2`/`bfloat162`/`float4`/`double2`) access candidates. All CUDA
  work is compile/clippy-verified only — no CUDA hardware available in CI or
  local development at time of writing.
- **WGPU autograd, essentially complete:** `layer_norm`, `batch_norm`,
  `adaptive_avg_pool2d`/`avg_pool2d`/`max_pool2d`, `max_dim`/`min_dim`/
  `max_all`/`min_all`/`max_keepdim`/`min_keepdim`, and `cross_entropy_loss`
  are now gradient-correct on the WGPU backend, verified against a real
  software WGPU adapter (not just compile-checked) via finite-difference
  gradcheck tests. Pooling and the max/min-family reductions needed genuine
  new backward code (host-readback + recomputed-argmax/window scatter,
  mirroring the CPU backend's proven `cpu/ops/pool.rs`/`cpu/ops/reduce.rs`
  algorithms); `layer_norm`/`batch_norm`/`softmax`/`cross_entropy_loss` turned
  out to already be gradient-correct by composition from already-wired
  primitives and only needed verification. WGPU autograd coverage now
  matches CPU's, except `quantize`/`dequantize`/`quantized_matmul` (not wired
  on CPU either — not a WGPU-specific gap).
- **Cross-backend gradient parity:** extended `tests/gradient_parity.rs` with
  `max_pool2d` and `cross_entropy_loss` (non-zero target class) CPU-vs-WGPU
  checks, the permanent regression class this file exists to catch.

### Changed
- `incin_backends::{cpu,wgpu,cuda}::storage::TensorId` are re-exports of
  `incin_core::exec::TensorId`; three independent identity counters became one.

### Removed
- `Backend::backward_with_nan_check` and its four implementations. NaN checking
  is `NanPolicy` on `ExecutionPolicy`; wrap the ordinary `backward` in
  `incin_core::exec::check_gradients(|| ..)`, which returns an error where the
  old method panicked.

### Fixed
- **Feature isolation and naming:** a bare install now enables only `std` and `cpu`; CUDA, WGPU, Candle, autotuning, and telemetry are explicit opt-ins. The third-party Candle adapter moved from `legacy::candle` to `external::candle`, and accelerator-only builds no longer reference CPU-only dispatch variants. Candle dtype conversion now returns an error instead of panicking on unsupported types.
- **C-10:** `Tensor::to_scalar<E>`/`to_vec1<E>` could construct an invalid `bool`
  (Miri-confirmed undefined behavior) when reading non-0/1 byte values from
  storage. Fixed by special-casing `bool` `TypeId` checks and enforcing a
  safe non-zero element truthiness check without unsound transmutes.
- **C-9:** WGPU `embedding`'s backward and `cross_entropy_loss`'s one-hot
  construction bit-reinterpreted F32-stored index/target bytes as `u32`
  (`buffer.to_vec::<u32>()`) instead of converting the value, silently
  corrupting every gradient/loss contribution for any non-zero class or
  vocab index (only index `0.0` happened to survive, since its IEEE bit
  pattern is `0x00000000`). Existing tests never caught this — both only
  exercised index/class `0`. Fixed to read `to_vec::<f32>()` and convert,
  matching the WGSL forward kernel's own `u32(indices[i])` value conversion.
- Pre-existing (not introduced this cycle) `cargo clippy --features cuda,std`
  and `--features wgpu,std` failures on `main`, found while auditing the
  above: mismatched feature gates in `backend_kind.rs`'s test module,
  `cpu/creation.rs`'s `TransferTo<Cpu>` test, and `tests/ops.rs`/
  `tests/gradient_parity.rs` assuming `cpu` was always enabled alongside
  `cuda`/`wgpu`.

## Development snapshot — 2026-07-22

### Changed

- Tensor device metadata is derived exclusively from the backend; `Tensor` now has the
  four parameters `Tensor<S, B, K, G>`.
- Runtime metadata is named `DTypeId`, `DeviceId`, and `DeviceKind`; GPU families
  remain representable even when their feature is disabled.
- Tensor allocation uses one `zeros`, `ones`, `rand`, or `randn` entry point for
  static and dynamic metadata. Allocating layers expose `build`.
- `from_slice` accepts the element type associated with its static dtype.
- `IncinBackend<T, D>` is the only concrete backend spelling exported by the
  public prelude; the former CPU, WGPU, and CUDA backend type names were removed. Device changes now use `TransferTo`, rebuilding destination-native storage through checked, dtype-aware host staging.

### Fixed

- CUDA-only builds can access the shared layout and quantized-storage helpers.
- CPU dynamic dtype allocation preserves the physical buffer variant and floating random
  initialization supports F32, F64, F16, and BF16.
- WGPU creation rejects non-F32 dtypes, wrong device families, invalid ordinals, and
  malformed byte payloads with typed errors.
- Runtime dispatch preserves physical dtype/device metadata, delegates reductions,
  and performs dynamic device transfers through dtype-aware host staging.

### Added
- **WGPU Autograd:** Implemented backward passes for `gelu`, `elu`, and `mish`
  activations in `WgpuBackendImpl`, including WGSL gradient kernels (`gelu_grad`,
  `elu_grad`, `mish_grad`) and tape entries in the autograd system.
- **Cross-Backend Parity Tests:** New `crates/incin-backends/tests/gradient_parity.rs`
  test suite verifies numeric agreement (≤ 1e-4) between `CpuBackendImpl` and
  `WgpuBackendImpl` for elementwise add, matmul, layer_norm, softmax, and
  cross_entropy_loss forward+backward passes.
- **`DTypeId::element_size()`:** New method returns the byte width of each
  dtype, used by safety checks in `to_scalar`/`to_vec1`.
- **Activation `ToDevice` impls:** Stateless activation modules (`ReLU`, `GELU`,
  `Swish`, `Mish`, `ELU`, `Softmax`, `Sigmoid`, `Tanh`) now implement
  `ToDevice<B, NewD>`, enabling their use as fields in `#[module]`-derived
  structs that call `to_device`.
- **Docs:** All 2,541 filler doc comments (`/// Core abstraction for \`X\`…`)
  replaced with real one-line descriptions across the entire workspace
  (`incin-core`, `incin-backends`, `incin-data`, `incin-macros`,
  `incin-telemetry`, `incin-viz`, `incin-viz-plugin-api`, test and
  example crates).
- **Real Doctests:** `s![]`, `idx![]`, and `#[module]` macro doc examples in
  `incin-macros/src/lib.rs` are compiled doctests (not `ignore`) and pass
  `cargo test --doc -p incin-macros`.

### Fixed
- **Safety:** `to_scalar` and `to_vec1` now validate the raw byte slice length
  against `DTypeId::element_size()` before interpreting bytes, preventing
  potential undefined behaviour on malformed storage.
- **Error Handling:** Replaced `panic!`/`unimplemented!` calls in `serialize.rs`
  (Q8_0 quantization path), `onnx_exporter.rs` (Q8_0 ONNX export), and
  `shapes/idx.rs` (multiple inferred dims) with clean `Result::Err` returns.
- **Security:** `FileTransport::open` now sets Unix file permissions to `0o600`
  (owner read/write only) on newly created telemetry log files.
- **Test Isolation:** All integration tests in `crates/incin/tests/` now
  explicitly target `CpuBackendImpl<f32, Cpu>` rather than `DefaultBackend`,
  preventing failures when `--features cuda` is active on CPU-only CI hosts.
- **CPU Feature Gate (C-8):** `cpu::ops::elementwise` components were previously
  gated under the `cuda` feature flag rather than `cpu`; corrected.

### Changed
- **`DefaultBackend`:** Always resolves to `CpuBackendImpl<f32, Cpu>` regardless of
  active GPU feature flags, ensuring a safe default on non-GPU hosts.

---

## Development snapshot — Backend Refactoring Sprint

### Changed
- **Backend Crates:** Moved `native`, `wgpu`, and `cuda` backends into their own
  distinct crate (`incin-backends`), standardizing trait bounds
  (`NumericOps`, `ModuleOps`, `ReductionOps`, etc.) across devices.
- **WGPU Migration:** Transitioned core components, backends, and app libraries
  from Metal to WGPU for unified cross-platform execution.
- **External Adapter Cleanup:** Deleted obsolete, dead-code `ndarray` and `burn`
  compatibility wrappers.

### Added
- Complete WGPU convolution implementations and telemetry tracking features.
