# Changelog

All notable changes to the Kindle framework will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Dtype/kernel specialization architecture:** new `dtype_policy.rs` (single
  storage/compute/accumulator/output dtype resolver for CPU/CUDA/WGPU),
  `iteration.rs` (backend-neutral broadcast/layout iteration plan), and
  `kernel.rs` (typed CUDA source generation shared across pointwise,
  reduction, and normalization operation families). See
  `docs/DTYPE_KERNEL_ARCHITECTURE.md` for the full design and phased status.
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

### Fixed
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

## [0.2.0] - 2026-07-22

### Changed

- Tensor device metadata is derived exclusively from the backend; `Tensor` now has the
  four parameters `Tensor<S, B, K, G>`.
- Runtime metadata is named `DTypeId`, `DeviceId`, and `DeviceKind`; GPU families
  remain representable even when their feature is disabled.
- Tensor allocation uses one `zeros`, `ones`, `rand`, or `randn` entry point for
  static and dynamic metadata. Allocating layers expose `build`.
- `from_slice` accepts the element type associated with its static dtype.
- `KindleBackend<T, D>` is the only concrete backend spelling exported by the
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
- **Cross-Backend Parity Tests:** New `crates/kindle-backends/tests/gradient_parity.rs`
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
  (`kindle-core`, `kindle-backends`, `kindle-data`, `kindle-macros`,
  `kindle-telemetry`, `kindle-viz`, `kindle-viz-plugin-api`, test and
  example crates).
- **Real Doctests:** `s![]`, `idx![]`, and `#[module]` macro doc examples in
  `kindle-macros/src/lib.rs` are compiled doctests (not `ignore`) and pass
  `cargo test --doc -p kindle-macros`.

### Fixed
- **Safety:** `to_scalar` and `to_vec1` now validate the raw byte slice length
  against `DTypeId::element_size()` before interpreting bytes, preventing
  potential undefined behaviour on malformed storage.
- **Error Handling:** Replaced `panic!`/`unimplemented!` calls in `serialize.rs`
  (Q8_0 quantization path), `onnx_exporter.rs` (Q8_0 ONNX export), and
  `shapes/idx.rs` (multiple inferred dims) with clean `Result::Err` returns.
- **Security:** `FileTransport::open` now sets Unix file permissions to `0o600`
  (owner read/write only) on newly created telemetry log files.
- **Test Isolation:** All integration tests in `crates/kindle/tests/` now
  explicitly target `CpuBackendImpl<f32, Cpu>` rather than `DefaultBackend`,
  preventing failures when `--features cuda` is active on CPU-only CI hosts.
- **CPU Feature Gate (C-8):** `cpu::ops::elementwise` components were previously
  gated under the `cuda` feature flag rather than `cpu`; corrected.

### Changed
- **`DefaultBackend`:** Always resolves to `CpuBackendImpl<f32, Cpu>` regardless of
  active GPU feature flags, ensuring a safe default on non-GPU hosts.

---

## [0.1.0-alpha.1] - Backend Refactoring Sprint

### Changed
- **Backend Crates:** Moved `native`, `wgpu`, and `cuda` backends into their own
  distinct crate (`kindle-backends`), standardizing trait bounds
  (`NumericOps`, `ModuleOps`, `ReductionOps`, etc.) across devices.
- **WGPU Migration:** Transitioned core components, backends, and app libraries
  from Metal to WGPU for unified cross-platform execution.
- **Legacy Removal:** Deleted obsolete, dead-code `ndarray` and `burn`
  compatibility wrappers.

### Added
- Complete WGPU convolution implementations and telemetry tracking features.
