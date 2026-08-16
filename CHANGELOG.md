# Changelog

> The workspace intentionally remains at `0.0.0`. Dated version headings below
> are development snapshots retained for traceability, not published releases.

All notable changes to the Incin framework will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed
- **Advanced indexing facade:** Curated `incin::advanced` to export only the
  documented type-level indexing selectors and traits.
- **Core advanced indexing facade:** Applied the same explicit export boundary
  to `incin_core::advanced`, keeping hidden implementation traits out of the
  downstream namespace.
- **Public API guard:** Stable `incin` and `incin-core` facade files now fail
  validation if a wildcard re-export is reintroduced.
- **Shape root exports:** Replaced wildcard exports from private shape
  implementation modules with explicit scalar, storage, proof, and dimension
  items.
- **Shape storage boundary:** Kept the internal `InlineOrHeap` representation
  out of the public shape prelude and removed the unused public
  `fold_static_numel` helper.
- **CPU allocation imports:** Removed unused random and Rayon import
  suppressions from the CPU creation kernel.
- **Editor prose:** Replaced dash-heavy comments and user-facing text in the
  Neovim, VS Code, and RustRover integrations with ordinary punctuation.
- **Descriptor macro policy:** Removed an obsolete unused-macro suppression
  from the shared descriptor executor declarations.
- **Dispatch scaffolding:** Removed the unused multi-operand dispatch macro;
  all live routes use the module-specific helper or explicit routing path.
- **Unsupported-operation scaffolding:** Removed unused creation, reduction,
  and tensor-operation declaration macros; float-operation declarations remain
  because CUDA, Metal, WGPU, and Candle still use them.
- **Target feature gating:** Compiled the non-CPU target implementation macro
  only when one of its target backends is enabled, removing its unused-macro
  suppression in CPU-only builds.
- **Capability macro exports:** Feature-gated backend-specific capability macro
  re-exports so CPU-only builds no longer need unused-import suppressions.
- **Rust toolchain reproducibility:** Pinned the supported compiler and stable
  CI, hardware, and release jobs to Rust 1.97.1 to keep diagnostics and builds
  repeatable.
- **CPU test helper isolation:** The finite-difference gradient checker is now
  compiled only for CPU unit tests instead of shipping as dormant production
  code behind a module-wide dead-code allowance.
- **CPU operation test cleanup:** Removed unused backend aliases from pooling,
  convolution, and embedding tests, and removed an obsolete macro forwarding
  helper that had no callers.
- **Dispatch dead-code cleanup:** Removed four private variable-creation
  dispatch wrappers that were never called; the execution registry remains the
  active path for variable creation operations.
- **Hidden API inventory:** Refreshed the reviewed source locations for the
  descriptor transform, paranoid-validation, and macro-support hidden items so
  the mechanical inventory check matches the current source.
- **Dummy backend scope:** The shape-only dummy backend is now compiled only
  for unit tests or the explicit `test-utils` feature, matching its documented
  role and keeping its test-support suppressions out of normal core builds.
- **Backend documentation:** Repaired stale placeholder references in the
  unsupported-operation macro documentation so each explanation names the
  operation family it describes.
- **CI package gate:** The ledger job now validates locked Cargo metadata and
  every publishable package archive, catching omitted sources, binaries, and
  license metadata before release packaging.
- **Core rustdoc links:** Removed invalid `GradMode` scope links and clarified
  the no-`std` policy-scope wording so core rustdoc passes with warnings denied.
- **Dummy backend dead code:** Removed an unused family of private float
  operation shims and all stale dead-code allowances from the test backend.
- **Dynamic marker scope:** Restricted the private `Dyn::marker` test helper to
  unit-test builds, removing the last production dead-code allowance in core.
- **Compile-fail diagnostics:** Updated the `Dyn` privacy regression snapshot
  to reflect that the test-only marker helper is no longer suggested to users.
- **Metal tuning isolation:** Metal benchmark winner selection and cache-claim
  helpers are now test-only, with production builds retaining only the
  candidate conversion and fallback policy they use.
- **Facade API tiers:** Removed backend-authoring traits from the stable
  `incin` root and default prelude, and removed `Graph` from the core prelude.
  The data prelude now also uses an explicit allow-list. The supported
  migration paths are recorded in `docs/MIGRATION.md`; backend contracts
  remain under the explicit `backend-authoring` feature.
- **Editor documentation prose:** Replaced em-dash-heavy phrasing in the
  VS Code, Neovim, and RustRover integration READMEs with ordinary punctuation
  so current user-facing documentation follows the repository prose style.
- **Candle adapter cleanup:** Removed unused unsupported-operation stubs and
  quantization placeholders from the legacy inherent surface. Unsupported
  capabilities remain represented by the descriptor capability registry.
- **`must_use` signal:** Removed 40 redundant method and function annotations
  whose return types were already `Option` or `Result`, while retaining
  annotations on builders, constructors, and semantically important values.
- **CUDA lint and structure:** Grouped internal two-dimensional column-to-image
  parameters into `Col2Im2dSpec`, kept the shared transposed-convolution
  backend contract explicit, and moved CUDA backend trait implementations
  before the test module so the CUDA all-targets lint gate passes with
  warnings denied.
- **Dead-code audit:** Removed unused raw conversion and complement helpers
  from the private axis-mask implementation, and removed a redundant CUDA
  identity suppression while retaining feature-gated test and dummy-backend
  helpers.
- **Rustdoc coverage:** Documented the public plan-report exit status constants
  so trainer builds remain warning-free under the facade documentation lint.
- **Architecture and build hygiene:** The shape buffer helpers remain available
  through the documented `incin_core::shapes` facade while their implementation
  modules are private. Unreferenced WGPU dispatch paths and CUDA kernel sources
  were removed, and backend layout and quantized-storage modules are now gated
  by the features that use them. WGPU lifetime owners and CUDA tuning helpers
  no longer rely on broad dead-code allowances. The book CI job installs
  Chromium in the job that runs the browser checks.
- **Feature isolation:** Distributed context imports and protocol decoding are
  now gated with `std`, while compiled distributed plans retain their
  no-std-compatible ownership imports. The supported `compiled,distributed`
  and `distributed` feature contracts both compile cleanly.
- **ONNX export surface:** Removed the unreferenced captured-graph export
  helper from the private exporter module; the reviewed eager-graph exporter
  remains the supported ONNX path.
- **Release packaging:** The editor release job now uses pinned Node.js and
  VS Code packaging-tool versions, and names the IntelliJ-platform archive
  independently from the RustRover integration directory. Release assets
  include the book, editor integrations, `incin-lsp`, and `cargo-incin`; the
  VS Code manifest now identifies the repository for package consumers.
- **Rustdoc coverage:** The public `incin` facade now enables local
  `missing_docs` warnings, and CI runs a warning-free facade-only rustdoc gate
  in addition to the workspace link and warning check.
- **Tensor byte views:** `Tensor::from_slice` now uses the `bytemuck` checked
  byte-slice conversion already guaranteed by `TensorElement`, removing a raw
  pointer reinterpretation from the core tensor boundary.

### Added
- **Core Stabilization & Migration Guide (`REL-001`):** Completed comprehensive core stabilization review and added `docs/MIGRATION.md` detailing API migration pathways across backend storage decoupling (`EXE-006`..`EXE-009`), unified autograd graph engine (`GRD-001`..`GRD-006`), proof-carrying shape safety (`SHP-001`..`SHP-008`), and distributed placement proofs (`DST-001`..`DST-005`). `docs/MIGRATION.md` section 7 added for the compiled-graph subsystem.
- **Compiled graph capture (`CMP-001`, `incin-core::compiled::capture`):** `CapturedGraph` and `CapturedNode` provide a serializable IR snapshot of an eager `Graph` for offline analysis, inspection, and compilation passes.
- **Immutable compiled plans and dynamic guards (`CMP-002`, `incin-core::compiled::plan`):** `CompiledPlan` bundles a `CapturedGraph` with `CompileOptions` and per-input `ShapeGuard` entries for runtime dtype/shape verification. `DynamicShapePolicy` and `FusionPolicy` are the two knobs.
- **Constant folding, weight prepacking, and shape buckets (`CMP-004`, `incin-core::compiled::fold`):** `ConstantFolder` propagates compile-time-known values, `WeightPrepacker` tiles weights into contiguous layouts, and `ShapeBucket` bins dynamic shapes to reduce recompilation.
- **Liveness and allocation planner (`CMP-003`, `incin-core::compiled::alloc`):** `LivenessMap` computes per-node def/use intervals; `AllocationPlanner` assigns buffer slots with slot reuse; `MemoryPlan` reports peak live slot count and alias candidates for buffer aliasing.
- **Compiled-graph saved-tensor liveness (`GRD-007`, `incin-core::compiled::alloc`):** `SavedTensorSet` and `LivenessMap::extend_for_saved_tensors` extend forward liveness intervals through the backward pass, preventing premature buffer reuse for autograd-retained tensors.
- **Safe kernel fusion pass (`CMP-005`, `incin-core::compiled::fusion`):** `FusionPass` identifies adjacent pointwise chains (`FusionCandidate`) and applies them (`FusedKernel`), reducing launch count. `FusionBlocker` documents why two ops may not fuse.
- **Versioned compiled artifacts (`CMP-006`, `incin-core::compiled::artifact`):** `CompiledArtifact` wraps a `CompiledPlan` with an `ArtifactHeader` containing an `ArtifactVersion` and an Adler-32 integrity checksum. `serialize` / `deserialize` / `load` cover the full roundtrip; `verify_integrity` and `check_compatibility` guard against corruption and version skew.
- **Distributed placement proofs (`DST-003`, `incin-core`'s `distributed`
  feature):** `Replicated`, `Sharded`, `Partial`, and `PipelineStage` extend
  the existing `Local` placement typestate, with `PlacementKind` as their
  runtime projection. `ShardDivisible` proves an exact typenum quotient through
  a zero `Rem`; dynamic extents use the same rule through `validate_shard`.
  `LegalTransition` admits only identity, local shard, all-gather, all-reduce,
  and reduce-scatter, while `CompletePlacement` prevents an unreduced
  `Partial` from reaching an ordinary consumer. `PlacementTransitionRule`
  validates typed global shape, descriptor output, input placements, and every
  local shape against mesh-derived world, tensor, and pipeline degrees before
  minting the private-field, private-constructor
  `ValidatedDistributed`. Physical mesh identity remains supplied by
  `DeviceMesh`, and executable collective ordering remains `DST-007`.
- **Typed logical device meshes (`DST-001`, `incin-core`'s `distributed`
  feature):** `incin_core::dist::mesh` adds `MeshSpec<Data<DP>,
  TensorParallel<TP>, Pipeline<PP>>` and `ValidMesh`, the compile-time half of
  `PROPOSALS.md` §3.8. A mesh holds no `DeviceId` — the claim is logical device
  selection, never hardware existence — so `ValidMesh` proves only that the
  degrees are nonzero and that `DP × TP × PP` is countable, over the same
  `typenum` `Mul` the shape rules use. `World` is an associated type so a
  caller can write `M: ValidMesh<World = U3>`, which is how §3.8's "`DP=3`,
  `TP=3`, or `PP=3` are valid for three GPUs and a rectangular `2 × 2` is not"
  becomes a compile error. The axes are positional and each position accepts
  only its own marker, because swapping tensor and pipeline keeps the world
  size and changes the meaning. Omitted axes default to one. `DeviceMesh::bind`,
  the topology fingerprint, and the runtime guards are `DST-002`.
- **Automatic `Trainer` (`UX-001`, `train` feature):** `incin::train` builds
  `PROPOSALS.md` §2's level-1 workflow — pick devices, get a validated plan,
  train. The load-bearing property is a refusal: an unsatisfiable device request
  is an error, never a CPU fallback, and `NotCompiledIn` (fix your
  `Cargo.toml`) is a separate variant from `DeviceUnavailable` (fix your
  machine). `DeviceSet` and `DevicePreference` join
  `incin_core::tensor::device`; they are separate types so that "I asked for
  CUDA and got CPU" is something the API can refuse rather than express.
  `DevicePreference::Fastest` may resolve to the CPU — that is what asking for
  it means — and records every family it skipped. Availability is answered
  through a `Machine` trait, so a three-GPU plan is testable on a runner with
  none. Multi-device `fit` is an explicit `CollectivesUnavailable` naming
  `DST-005` rather than a quiet single-GPU run.
- **Generated capability and feature documentation (`UX-013`):**
  `docs/capabilities.md` is rendered from `CPU_CAPABILITIES`,
  `CUDA_CAPABILITIES` and `WGPU_CAPABILITIES` by
  `incin_backends::capability_docs`, and `README.md`'s two feature tables are
  rendered from the Cargo manifests by `cargo xtask docs` — including the
  `Purpose` column, which comes from the `#` comment above each feature in the
  manifest. Both have a check that runs in CI (`cargo xtask docs --check` and
  the `generated_docs` suite), because a generator nobody runs is a handwritten
  table with extra steps. `DTypeId::name`, `DeviceKind::name` and
  `ImplementationKind::name` give the enums one spelling each, so the tables,
  the conformance suite and `cargo incin doctor` cannot disagree about what to
  call `f32`.
- **External-backend SDK and conformance suite (`EXE-010`):**
  `incin_backends::external::conformance` is the backend-authoring surface from
  `PROPOSALS.md` §2.9 — a `Subject` trait carrying the three things only an
  author can supply, `Tolerance` profiles, and eight checks identical for every
  backend. Every check consults the capability registry first, so an operation
  a backend does not claim is *skipped*, not failed. `external` is no longer
  gated on `external-candle`: authoring a backend no longer requires enabling
  the Candle adapter. `crates/incin-backends/tests/conformance.rs` carries a
  complete minimal template backend to copy, four deliberately broken ones that
  each fail exactly one check, and the Candle adapter passing all eight.
- **`cargo incin doctor` (`UX-014`):** one command reporting toolchain and
  crate versions, enabled Cargo features, the CPU ISA extensions the kernels
  branch on, each backend family's compiled-in and available state, cache paths
  and writeability, and capability probes for eight representative operations
  on every device that answered. Stable `key: value` text by default, `--json`
  for CI and support reports with a `schema_version`. Findings carry stable
  codes — `no-backend-compiled`, `backend-unavailable`, `cache-not-writable`,
  `deprecated-feature`, `toolchain-unknown`, `isa-unavailable` — and only the
  first exits non-zero. The report is `incin::doctor`, a library module, so it
  is testable; every observation goes behind a `Host` trait, so a three-GPU
  machine can be put in front of it on a runner with none. The command is
  read-only: writeability is read from mode bits rather than probed by writing.
- **Macro test suite (`CI-005`):** `crates/incin-macros/tests/` now carries the
  compile-pass, compile-fail, hygiene, rename, and rustfmt cases the macro
  policy in `PROPOSALS.md` requires — twelve trybuild cases plus guards that
  fail when a case stops asserting what it claims or when one of the five
  categories disappears. `cargo test -p incin-macros` previously ran nothing.
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

### Fixed
- **Every public example is compiled (`UX-013`).** 70 of the workspace's 79 doc
  examples were fenced ```` ```rust,ignore ````, so `cargo test --workspace
  --doc` reported success having compiled nine — and CI never ran it at all,
  because `cargo test --all-targets` excludes doctests. Compiling them found
  the examples documenting an API that does not exist: `from_slice` shown with
  one argument where it takes two (fifteen examples), `Param<Tensor<S, B>>`
  where the type is `Param<S, B>`, a rank-1 reshape argument written `()` where
  it is `((),)`, `dims()` compared against a `Vec` where a static shape returns
  an array, `incin::symbolic_dim!` which does not resolve (`dim!` is the public
  macro; `symbolic_dim!` is a `#[doc(hidden)]` alias the facade does not
  re-export), and an `incin-data` front page built on a `DataLoader` builder API
  that was never written. A test now fails on any reintroduced `ignore` fence.
- **`IndexSpec`, `LSTM` and `LSTMCell` are reachable from the prelude
  (`UX-013`).** `Tensor::slice` takes `&[IndexSpec]` and `IndexSpec` was not
  exported, so the documented call could not be written by a user of the
  prelude; `RNN` and `RNNCell` were exported while `LSTM` and `LSTMCell` were
  not.
- **`DummyBackend`'s binary operations broadcast (`UX-013`).** They returned the
  left operand's shape unchanged, which disagrees with every real backend:
  `broadcast_add` and its siblings reach `Backend::add` with differently shaped
  operands and hand the result to `Tensor::from_parts` against the *broadcast*
  type. `incin_core::shapes::broadcast::broadcast_dim_slices` is the one
  right-aligned rule both paths now use.
- **`--features external-candle` failed `clippy -D warnings` (`EXE-010`):** the
  `bytes` module was gated on `external-candle` alongside `cuda` and `wgpu`,
  but the Candle adapter never allocates by byte length, so that feature set
  compiled a module whose only function was dead.
- **WGPU device detection crashed when probed more than once (`UX-014`):**
  `incin_backends::detect::probe` built a fresh `wgpu::Instance` per call and
  dropped it, and two threads each probing twice segfaulted inside adapter
  enumeration. The instance is shared for the process lifetime now, matching
  what the WGPU backend already did; detection is still performed per call.
- **Macro hygiene (`CI-005`):** `s!`, `idx!`, `#[module]`, `model!`, and
  `import_model!` expanded to a relative `incin::prelude::…`, so any caller
  item named `incin` captured the expansion. All five emit absolute `::incin`
  paths now; use `s![@ ..]` inside the workspace, which expands to
  `crate::prelude::…`. A package rename in a caller's `Cargo.toml` remains
  unsupported and is documented on each macro.
- **`#[module]` argument validation (`CI-005`):** struct-level arguments were
  matched as substrings, so `#[module(no_such_argument)]` was silently accepted
  as `#[module]` and `#[module(not_internal)]` as `#[module(internal)]`. The
  list is parsed against a closed vocabulary and unknown keys are rejected.

### Removed
- **Deprecated `candle` feature alias (`REL-002`, `D-014`):** Removed `candle` feature alias from `incin` and `incin-backends` in favor of explicit `external-candle`.
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
