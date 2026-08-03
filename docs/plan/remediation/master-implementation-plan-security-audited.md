# Incin 0.1 Stability and 1.0 Framework Completion — Master Implementation Plan

**Document role:** single implementation and audit handoff to be read together with `PROPOSALS.md`  
**Repository source:** supplied `incin(1).zip`  
**Audited branch:** `develop`  
**Audited commit:** `eb3633525ea74e56f7a6b2d5c5b57dc74a5d9b8d`  
**Latest commit subject:** `fix(macros): remove duplicate pub fn mesh export`  
**Repository state during audit:** clean  
**Primary milestone:** `0.1.0`, after which the selected public API is treated as non-breaking  
**Long-term target:** stable single-device framework for `1.0.0`; distributed execution remains preview  
**Team:** one human maintainer, Opus 5, GPT-5.6, and two Gemini agents, operating sequentially  
**Change policy before 0.1:** breaking changes and large internal restructuring are allowed  
**Change policy after 0.1:** no breaking user-visible changes to the stabilized public surface; private/internal changes remain unrestricted  
**Audit limitation:** this environment did not contain `rustc` or `cargo`; all code findings are static and the first implementation issue must capture real build/test evidence  
**Security review:** dedicated static security pass completed against the same commit; Part XXIII supersedes earlier release-readiness estimates where security is concerned  

---

# Part 0 — Decisions, unresolved inputs, and release interpretation

## 0.1 Decisions supplied by the maintainer

The following are requirements, not suggestions:

1. **`0.1.0` is the API-stability boundary.** The project may currently restructure anything. Once `0.1.0` ships, the explicitly stabilized user-facing API must evolve through additive changes and deprecation windows rather than removals or semantic breaks.
2. **Internal implementation is not automatically public contract.** Private modules, compiler IR, caches, generated kernels, backend internals, and unexposed serialization details may change without constituting a user-visible break.
3. **`1.0.0` is a stable single-device framework.** CPU and accelerator execution, model building, training, compiled inference, model interchange, documentation, and reliable packaging are in scope. Distributed capabilities may remain preview.
4. **Python interoperability is an adoption bridge, not an attempt to replace Python.** A Python developer should be able to move tensors, models, checkpoints, and exported graphs between Incin, PyTorch, NumPy, Hugging Face tooling, and related libraries with minimal friction.
5. **Safety remains differentiating.** Imported runtime models receive explicit runtime contracts; models with adequate metadata may generate typed Rust modules whose shape constraints become compile-time facts. No interoperability shortcut may silently pretend a runtime value has compile-time proof.
6. **First-class model sources and formats:** Hugging Face repositories, ONNX, safetensors, NumPy (`.npy`/`.npz`), PyTorch exports/checkpoints through a safe bridge, and later GGUF. GGUF remains low priority until core interoperability and compilation are trustworthy.
7. **Hugging Face loading must have two modes:** compile-time metadata/code generation for statically recoverable models, and runtime loading for flexible or user-provided model families.
8. **Model UX follows Rust best practices while using familiar vocabulary and behavior.** `Module`, parameters, buffers, state dictionaries, `train`/`eval`, device and dtype movement, named children, and strict loading should be immediately recognizable to PyTorch users without importing Python’s unsafe or implicit behavior.
9. **Performance target:** approach PyTorch-class performance and optimization for supported workloads and hardware. This is measured per model/backend/shape and may never be claimed from synthetic timings.
10. **Documentation target:** complete rustdoc, a maintained book, PyTorch-to-Incin comparisons, and executable examples covering every important workflow.
11. **Unsafe and panic policy:** minimize `unsafe`; every necessary block has a precise safety argument and dedicated tests. Avoid `unwrap`, `expect`, and panic in user-controlled/runtime paths.
12. **Contribution workflow:** create a focused issue first, then a small pull request linked to that issue. Do not send large undifferentiated agent dumps directly to `develop`.
13. **Execution order:** agents work sequentially to maximize quality and token/tool usage. Every handoff leaves the repository buildable and records exact next steps.
14. **Public positioning:** document Incin’s capabilities directly. Do not frame public documentation around another repository or product.

## 0.2 Remaining non-blocking questions

These were not answered and should be captured as decisions before their dependent release gates. They do **not** block the first architecture and correctness issues.

| Decision | Default used in this plan | Must be decided before |
|---|---|---|
| Calendar deadline | milestone-based, with 12/24/40-week projections | committing to a public release date |
| Tier-1 hardware | Linux x86-64 CPU and NVIDIA CUDA | publishing performance claims |
| Apple/AMD hardware availability | Tier 2, best-effort CI | declaring Metal/ROCm production support |
| Minimum Python | Python 3.11 | publishing `incin-python` wheels |
| Supported PyTorch range | latest stable plus previous minor | publishing the Python compatibility matrix |
| Managed compiler download | opt-in pinned download; system toolchain also supported | shipping compiled executable artifacts |
| Stable crates at 0.1 | `incin` facade plus explicitly named extension traits; internal crates remain pre-stable | 0.1 release candidate |
| Cloud/runner budget | no paid service assumed | hardware CI and broad wheel matrix |
| Hook/lazy-module support | preview/post-0.1 | finalizing the Module API |

Every issue depending on one of these decisions must either:

- open a short architecture decision issue and wait for the maintainer, or
- implement the documented default behind an unstable/preview feature without stabilizing it.

## 0.3 What IREE is, why it appears in this plan, and what users should see

IREE is an optional compiler/runtime layer that can take a tensor program expressed in MLIR dialects such as StableHLO and turn it into executable modules for CPU, CUDA, Vulkan, ROCm, and Metal targets. Incin should use it the same way a high-level language uses LLVM: as an implementation target, not as the user-facing programming model.

The recommended integration sequence is:

```text
Incin model
  -> canonical, verified Incin graph IR
  -> reference Incin compiled executor
  -> StableHLO/MLIR text or bytecode
  -> pinned `iree-compile` subprocess
  -> executable VMFB section in a versioned `.incin` artifact
  -> small runtime bridge
```

Why the subprocess first:

- it isolates compiler crashes and version skew;
- it is easy to pin, hash, log, and reproduce;
- IREE’s official compiler APIs are C/C++ and Python, while Rust compiler bindings are not an official stable surface;
- Incin can later bind the supported C API without changing the public `compile`/`run` interface.

What an Incin user should see:

```rust
let executable = model.compile(&context, &example, CompileOptions::release())?;
let output = executable.run(input)?;
```

What the user should **not** be required to understand:

- MLIR dialect names;
- VMFB internals;
- compiler pass flags;
- runtime driver names;
- compiler installation details when the managed toolchain is enabled.

## 0.4 Repository size and maturity snapshot

Static repository inventory at the audited commit:

| Metric | Value |
|---|---:|
| Rust source files under `crates/` | 514 |
| Approximate Rust LOC | 133,307 |
| Test Rust files | 221 |
| `incin-backends` LOC | 63,846 |
| `incin-core` LOC | 48,207 |
| `incin` facade LOC | 8,354 |
| Ledger rows | 101 |
| Ledger rows marked complete | 89 |
| Ledger rows partial | 6 |
| Ledger rows unchecked | 6 |
| Core-tier rows marked complete | 39/39 |

The ledger is valuable for task sequencing, but its 88% completion ratio is **not** product readiness. Several rows marked complete establish types, narrow vertical slices, or placeholder behavior rather than a production-quality feature. The compiled subsystem is the clearest example.

## 0.5 Honest progress toward 0.1 and 1.0

### 0.5.1 Scoring method

The percentages below are engineering estimates, not measurements of line count. Each category is scored against the maintainer’s stated release contract. A feature receives credit only when its public behavior, negative cases, evidence, and documentation are credible. Static inspection cannot prove runtime correctness, so confidence is medium-low until the baseline issue runs.

### 0.5.2 Estimated readiness for stability-targeted `0.1.0`: **about 47%**

| Category | Weight | Current credit | Rationale |
|---|---:|---:|---|
| Typed tensor/shape/proof foundation | 15 | 12 | Strong and broad; still needs public-surface review and compile-time budget evidence. |
| Eager backend correctness | 15 | 10 | Multiple native backends and conformance infrastructure exist; panic/unwrap, hardware coverage, and capability truth need audit. |
| Autograd and training fundamentals | 15 | 9 | Shared tape/error work is substantial; lifetime ownership, compiled training, optimizer/state stability, and end-to-end model tests remain. |
| Module/state/checkpoint correctness | 10 | 4 | Familiar abstractions exist, but strict loading, typed state values, metadata consistency, and sharded checkpoints are critical gaps. |
| Real compiled execution | 15 | 2 | Current modules are scaffolding and tests permit no-op or semantically invalid implementations. |
| Python/model ecosystem interoperability | 10 | 1 | ONNX/safetensors/HF pieces exist, but no seamless Python/DLPack/torch.export path and current dtype handling has correctness risks. |
| Public API stability and docs | 10 | 3 | API design principles and tooling exist, but no frozen surface, SemVer gate, comprehensive book, or migration matrix. |
| CI, safety, release, security | 10 | 6 | Good governance infrastructure; missing complete hardware/feature matrix, API compatibility gate, safety audit, and release soak. |
| **Total** | **100** | **47** | |

### 0.5.3 Estimated readiness for the requested `1.0.0`: **about 35%**

| Category | Weight | Current credit | Critical gap |
|---|---:|---:|---|
| Correct, stable single-device core | 20 | 12 | Public API freeze, hardware evidence, error and safety audits. |
| Compiled performance and deployment | 20 | 4 | Executable IR, real lowering/runtime, memory planning, artifacts, measured optimization. |
| Model building and training UX | 15 | 8 | State semantics, containers, tied parameters, strict loading, mature trainer/optimizer/checkpoint flow. |
| Python/PyTorch/NumPy interoperability | 15 | 2 | Python package, DLPack, graph import, backend registration, conversion tooling. |
| Formats and model hub | 10 | 3 | Shards, revisions, configs/tokenizers, architecture registry, runtime/static generation. |
| Documentation and education | 10 | 2 | Book, complete examples, PyTorch mapping, custom-op and deployment guides. |
| Release operations and support | 10 | 4 | MSRV, wheels, artifacts, security policy, long-running tests, compatibility guarantees. |
| **Total** | **100** | **35** | |

These percentages should change only when evidence changes. Do not increase them because an issue was opened, a type was declared, or a happy-path unit test was added.

## 0.6 Critical missing pieces in priority order

1. **A truthful public stability boundary.** Identify exactly what `0.1` freezes, remove accidental public APIs, and add SemVer compatibility checks.
2. **Correct model state semantics.** State dictionaries and safetensors must preserve dtype, shape, device, parameter/buffer identity, shared weights, and atomic strict loading.
3. **A real executable compiled path.** Canonical IR, scoped symbolic capture, reference execution, guards, memory planning, passes, executable artifacts, and a public `compile().run()` API.
4. **Measured accelerator performance.** Real kernel/vendor-library choices, real tuning, reproducible benchmarks, and hardware CI. Synthetic timing is prohibited.
5. **A Python bridge with honest guarantees.** DLPack/buffer interoperability, NumPy and PyTorch conversion, safe graph/checkpoint import, and optional generated typed Rust models.
6. **A model package and Hugging Face resolver.** Config, tokenizer metadata, sharded safetensors, revisions, offline/cache behavior, checksums, and architecture mapping.
7. **A final model API.** Stable parameter/buffer traversal, strict state loading, train/eval behavior, containers, shared parameters, device/dtype conversion, and custom-op extension points.
8. **Documentation as tested code.** Rustdoc and book examples compile in CI; Python comparisons are executable and versioned.
9. **Safety and reliability audits.** Categorize every production panic/unwrap/unsafe site; fuzz parsers and artifact loaders; use sanitizers/Miri where applicable.
10. **Release machinery.** MSRV, feature powerset, hardware matrix, API diff gate, migration/deprecation policy, security policy, release candidates and soak period.

## 0.7 Confirmed high-severity findings from static inspection

The references below are paths and line ranges at the audited commit. Agents must re-open the current file before editing because later PRs may shift lines.

### 0.7.1 Compiler/capture correctness

| Severity | Location | Finding | Required correction |
|---|---|---|---|
| Critical | `compiled/capture.rs:10-34` | `CapturedNode`/`CapturedGraph` retain IDs and op tags but discard shapes, dtypes, attributes, constants, descriptors, layouts, device, aliases, names, and provenance. | Replace with a canonical executable IR; do not extend this lossy shape piecemeal. |
| High | `compiled/capture.rs:53-60` | Topological validation checks `graph.values.contains_key`, so a value registered in the graph but produced by a later node is accepted as already valid. | Validate each input against graph inputs, initializers, or outputs of earlier nodes only. Add forward-reference and cycle mutants. |
| Critical | `compiled/plan.rs:112-117` | Every input guard is empty-shape F32, independent of graph metadata. | Derive guard programs from canonical input contracts and symbolic constraints. |
| High | `compiled/plan.rs:126-132` | Out-of-range input index silently succeeds. | Return structured arity/index error and validate all inputs atomically. |
| Critical | `compiled/fold.rs` | Constant folding and prepacking return the graph unchanged. | Either implement real passes with parity tests or mark APIs preview/unavailable; no-op cannot be described as optimization. |
| Critical | `compiled/fusion.rs:83-103` | Candidate discovery does not prove the producer result has one use, despite the comment. | Build a use-def graph and require safe region boundaries/effects. |
| Critical | `compiled/fusion.rs:133-148` | Fusion drops consumer-only inputs and stores no composite semantics. Overlapping candidates can skip nodes. | Represent a fused region/expression explicitly; execute and compare before/after for every candidate. |
| Critical | `compiled/artifact.rs:14-15,98-110` | Magic is declared but not serialized or checked; artifact is JSON plan data, not executable code. | Introduce a sectioned binary envelope with lengths, hashes, manifest, ABI, targets, and executable variants. |
| High | `compiled/artifact.rs:42-48` | Compatibility is only format + framework major, with no target/compiler/backend/schema contract. | Independently version format and public ABI; validate target/device/runtime requirements. |
| Critical | `compiled/tuning.rs` | Tuning results are simulated from node count. | Delete/rename synthetic behavior; measurements must execute real candidate plans with warmup, synchronization, and statistical policy. |
| Critical | public API | There is no complete normal-model `compile(...).run(...)` path. | Ship reference compiled execution before external compiler integration. |

### 0.7.2 Tracing and graph construction

| Severity | Location | Finding | Required correction |
|---|---|---|---|
| Critical | `tensor/tracing.rs` | Process-global mutex graph permits capture interference and prevents nested/reentrant capture. | Session-scoped `CaptureContext` in `ExecutionContext`, with RAII ownership and explicit graph builder. |
| Critical | `tensor/tracing.rs` | Multiple trace helpers hardcode F32 and omit operation attributes/scalars. | Capture typed metadata from `TensorMeta` and full validated descriptors. |
| High | tracing design | Capture delegates to real backend execution, causing allocations/compute and requiring hardware. | Introduce symbolic/meta storage and an executor that lowers without numerical work. |
| High | `graph.rs` to compiled bridge | Rich `Graph` metadata is discarded by `CapturedGraph`. | Define one canonical conversion with losslessness tests and round-trip invariants. |

### 0.7.3 Parameters, state dictionaries, and safetensors

| Severity | Location | Finding | Required correction |
|---|---|---|---|
| Critical | `nn/param.rs` state loading | Missing keys may be ignored and replacement storage can diverge from stored shape/dtype/device metadata. | Validate complete load plan first, then commit atomically; never leave metadata/storage disagreement. |
| High | `nn/param.rs` state export | A failed tensor/shape conversion can silently omit a parameter. | Make state export fallible and return exact path/context. |
| Critical | `nn/save.rs` | Loader returns backend float storage even for integer/boolean safetensors and can reinterpret physical bytes as `FloatElem`. | Introduce dynamically typed state values or dtype-erased storage with validated typed conversion. |
| Critical | `nn/save.rs` | I32 may map to U32 and BOOL to U8; default tensor dtype/device can misrepresent stored values. | Preserve exact logical dtype, reject unsupported types, and require target-device policy. |
| High | `nn/save.rs` | Save path may fall back to F32 when storage dtype lookup fails. | Return a structured unsupported-storage error; never lie in serialized metadata. |
| High | state API | No PyTorch-like strict report for missing, unexpected, shape-mismatched, dtype-mismatched, or alias-conflicting keys. | Add `LoadStateReport` and strict/default/custom policies. |

### 0.7.4 Hub and data reliability

| Severity | Location | Finding | Required correction |
|---|---|---|---|
| High | `incin-data/src/hub.rs` | Resolver is oriented around one `model.safetensors`; lacks revisions, subfolders, shard index, config/tokenizer package, offline policy, and integrity manifest. | Build a repository resolver and `ModelPackage` abstraction. |
| High | downloader | Direct-to-destination writes risk partial/corrupt cache entries. | Write temporary file, flush, verify size/hash when available, then atomic rename. |
| High | `MnistDataset` | Several filename/UTF-8 unwraps and insufficient magic/header validation. | Return structured dataset errors and validate sizes with checked arithmetic. |
| High | data loader | Worker mutex poisoning/panic propagation and cancellation are not robust; zero-worker behavior still creates overhead. | Explicit worker protocol, join handles, cancellation, deterministic seed, synchronous zero-worker path. |

### 0.7.5 Panic, unwrap, and unsafe inventory

A lexical count outside dedicated `tests/` files found approximately:

| Pattern | Count |
|---|---:|
| `.unwrap()` | 1,088 |
| `.expect(...)` | 293 |
| `panic!(...)` | 68 |
| `unsafe { ... }` blocks | 141 |
| `unsafe fn` | 28 |
| `unreachable!(...)` | 12 |

This count includes some `#[cfg(test)]` code inside source files and is therefore a triage input, not a release metric. Major hotspots include CUDA backend setup, CPU elementwise/shape/reduction paths, NCCL, pointwise code generation, and matmul SIMD code.

Required classification for every occurrence:

- **Class A — user/runtime controlled:** must return a structured error; panic/unwrap prohibited.
- **Class B — proof-established internal invariant:** encode in types where reasonable; otherwise use a documented invariant and a debug assertion, not an unqualified panic.
- **Class C — process initialization/build script:** may terminate only with an actionable diagnostic and exact cause.
- **Class D — tests:** unwrap/expect allowed for test readability.
- **Class E — FFI/unsafe kernel boundary:** every block gets a `// SAFETY:` argument and dedicated boundary tests.

The goal is not “zero unsafe at any cost.” The goal is a small, reviewed unsafe surface whose invariants are mechanically and dynamically tested.

## 0.8 The 0.1 public stability contract

### 0.8.1 What should be stable

By default, stabilize only:

- the `incin` facade crate’s documented public modules and prelude;
- tensor construction and core operations intentionally exported by the facade;
- shape/dimension syntax intended for users (`s!`, named dimensions, slicing syntax);
- the final `Module`, parameter, buffer, state dictionary, optimizer, trainer, device, dtype, error, and execution-context APIs chosen for 0.1;
- documented import/load APIs for formats declared stable;
- the public backend/custom-op extension traits explicitly marked stable;
- documented CLI commands and their machine-readable JSON schemas where promised;
- `.incin` artifact **manifest ABI** only if executable artifacts ship in 0.1; compiler cache contents remain unstable.

### 0.8.2 What should remain unstable/private

- canonical compiler IR structs;
- optimization pass types;
- tuning cache implementation and schema;
- generated kernel source and internal ABI;
- backend implementation types not required by users;
- distributed planning internals;
- experimental formats and model architectures;
- Python internals below the documented package API;
- private workspace crates.

### 0.8.3 SemVer rules after 0.1

Because the maintainer wants a non-breaking boundary earlier than Rust’s usual 1.0 convention, CI must enforce the policy explicitly:

- snapshot the selected public API at the 0.1 release tag;
- run `cargo-semver-checks` or an equivalent rustdoc-JSON comparison on every PR;
- require an explicit `breaking-approved-before-0.1` label before release, then disable that escape hatch after 0.1;
- additive trait requirements must use default methods or sealed/internal traits;
- public enums likely to grow should be `#[non_exhaustive]` before freeze;
- public structs should avoid exposed fields unless construction/update semantics are intentionally stable;
- behavior changes need regression tests and a migration/deprecation note even when signatures remain identical;
- deprecations remain for at least two minor releases and include a machine-applicable replacement where possible.

## 0.9 Definition of done for 0.1

`0.1.0` is not ready until all of these are true:

- selected public API has a reviewed inventory and SemVer baseline;
- no known critical correctness bug remains in tensor metadata, state loading, serialization, capture, or execution;
- default CPU path and at least one accelerator path have reproducible model-level forward/backward evidence;
- capability tables match executable reality;
- runtime/user-input paths have completed panic/unwrap triage;
- unsafe surface has safety comments and targeted tests;
- model state can round-trip exactly for supported dtypes and shared parameters;
- ONNX/safetensors/HF/NumPy claims are scoped and tested truthfully;
- compiled APIs are either genuinely executable and documented, or explicitly preview and excluded from the stable surface;
- every rustdoc and book example used as documentation compiles in CI;
- MSRV and supported platform/toolchain matrix are published;
- release artifacts, licenses, security policy, changelog, migration guide, and support expectations exist;
- two release candidates pass a soak period with no unresolved severity-1/2 issue.

---

# Part I — Current-state audit of `develop`

## 1. What Incin already has that must be preserved

The `develop` branch is materially more advanced than the earlier repository snapshot. It already contains the architectural foundations that make this program feasible:

### 1.1 Typed frontend and model system

- `Tensor<S, B, K, G, P>` carries shape, backend, dtype, gradient mode, and optional placement.
- Static, named, mixed, and dynamic shape paths already exist.
- `#[module]`, `Module<Input>`, parameter traversal, state dictionaries, device transfer, train/eval mode, named layers, and compute statistics provide a genuine model-building framework.
- The module and state-dictionary naming conventions are already close to PyTorch conventions, especially for sequential models.
- The repository exposes a broad operation vocabulary in `graph::OpType`—91 variants at this snapshot—including pointwise ops, reductions, matmul, convolution, pooling, indexing, normalization, attention, and shape transforms.

### 1.2 Proof-carrying operation execution

- `Validated<O>` is sealed.
- `ShapeRule` lowering exists for the core descriptor families.
- `BroadcastSpec`, `MatMulSpec`, `ReductionSpec`, `Conv2dSpec`, `Pool2dSpec`, and `ReshapeSpec` carry checked semantic geometry.
- `StorageBackend`, `Execute<O>`, `ExecutionRequest`, capabilities, normalized `TensorMeta`, and witnessed tensor construction provide the right trust boundary.
- CPU, CUDA, WGPU, Metal, and external backend surfaces exist in the branch.

### 1.3 Autograd and policy

- Execution policy is explicit.
- `GradMode` propagation exists.
- Backend-neutral tape nodes and structured backward errors exist.
- CUDA and WGPU gradient parity tests are represented in the repository evidence.

### 1.4 Backend code generation and tuning foundations

- Pointwise expression/codegen infrastructure exists for CUDA/WGSL/MSL.
- Device/compiler/topology-aware tuning infrastructure exists.
- Capability tables and `cargo incin doctor` provide a strong diagnostics base.
- Metal support and codegen landed recently on `develop`.

### 1.5 Tooling and UX

- `cargo-incin` already wraps Cargo and humanizes type-level diagnostics.
- LSP/editor integration exists.
- Model graph visualization and telemetry crates exist.
- ONNX and safetensors import paths exist, although they require hardening.
- A Trainer exists behind a feature.

These are Incin’s moat. The compiler program must consume them rather than bypassing or replacing them.

---

## 2. Critical correction: the current compiled stack is scaffolding, not production AOT-compiler execution

The ledger marks `CMP-001` through `CMP-006` complete. On the exact `develop` snapshot, those rows satisfy their narrow repository tests, but the implementation is not yet an executable graph compiler. The new work must be tracked as a **compiled-execution hardening and productization program**, not as cosmetic extensions.

The following findings are load-bearing.

### 2.1 Capture drops the data required to execute or lower a graph

`crates/incin-core/src/compiled/capture.rs` retains only:

- node ID;
- `OpType`;
- input IDs;
- output IDs.

It discards:

- value shape and dtype metadata;
- node attributes;
- constants and initializers;
- validated descriptors;
- layout and aliasing information;
- source locations;
- parameter identity;
- gradient/effect metadata;
- device/placement information.

This representation cannot correctly distinguish, for example:

- `sum(dim=1)` from `sum(dim=2)`;
- one convolution stride/padding from another;
- one transpose permutation from another;
- F16 from F32 values;
- constants from mutable parameters;
- views from materialized tensors.

### 2.2 Current guards are placeholders

`CompiledPlan::compile` creates every input guard with:

```rust
expected_shape = Vec::new()
expected_dtype = DTypeId::F32
```

This is not a dynamic-shape policy. It is a placeholder that rejects legitimate non-empty inputs or reports incorrect expectations.

### 2.3 Current allocation planning is slot counting, not memory planning

`alloc.rs` computes simple liveness intervals and reusable slot numbers. It does not model:

- byte sizes;
- dtype;
- alignment;
- storage class;
- device;
- alias/view relationships;
- input/output ownership;
- saved tensors;
- workspace requirements;
- mutation;
- asynchronous lifetimes.

It also assumes a relationship between node IDs and topological indices that must not be relied upon.

### 2.4 Constant folding and weight prepacking are no-ops

`ConstantFolder::fold` returns an unchanged clone and an empty folded set. `WeightPrepacker::prepack` returns an unchanged clone. Tests that only verify stability cannot establish compiler behavior.

### 2.5 Current fusion is not semantically valid execution

The fusion pass recognizes adjacent pointwise operation names, but replacement retains the producer operation, producer inputs, and consumer outputs. It does not create a composite expression or executable kernel. For a chain such as `relu(neg(x))`, merely reducing node count can silently remove part of the computation.

### 2.6 Artifacts contain plans, not executable programs

The artifact module serializes a plan to JSON with an Adler checksum. It does not contain:

- VMFB or native executable bytes;
- backend variants;
- function ABI metadata;
- external parameter sections;
- compiler/runtime fingerprints;
- target requirements;
- real entry points;
- binary section bounds;
- robust integrity hashes.

### 2.7 Current plan tuning is simulated

The compiled tuning path derives synthetic latency from node count and applies simulated percentage improvements. This must never be exposed as real tuning or benchmark evidence.

### 2.8 Current tracing is unsuitable as the canonical compiler frontend

`tensor/tracing.rs` uses a process-global graph mutex, delegates to real backend execution, hardcodes F32 in multiple places, and often records empty attributes. This creates problems with:

- nested capture;
- concurrent capture;
- deterministic isolation;
- backend-independent symbolic capture;
- dtype fidelity;
- operation-attribute fidelity;
- compile-only environments.

### 2.9 There is no public executable `model.compile().run()` product path

Compiled types are primarily exercised through narrow tests. There is no complete public path that:

1. accepts a normal Incin model;
2. captures it without real execution;
3. builds a correct plan;
4. lowers it to executable code;
5. validates runtime inputs;
6. executes it;
7. returns a typed output;
8. demonstrates eager/compiled parity.

This is the first product milestone.

---

## 3. Audit rule for all agents

No existing `[x]` ledger status may be used as proof that a feature is production-ready. For this program:

- code is authoritative;
- end-to-end executable tests are authoritative;
- mutation tests are authoritative;
- real benchmark output is authoritative;
- a row that only serializes data, counts nodes, or checks that an object exists is not sufficient.

Every new task must include a **semantic mutant**: a deliberately broken implementation that would fail the test. A test suite that passes the mutant is inadequate.

---

# Part II — Definition of “framework-completion” success

## 4. Minimum complete AOT workflow

Before Incin describes compiled execution as complete, it must provide all of the following user-visible capabilities:

1. Rust-authored tensor graph compilation.
2. Static typed input/output validation.
3. Build-time AOT compilation.
4. IREE VMFB generation.
5. Reusable runtime engine.
6. Multiple target/backend variants in one logical artifact.
7. CPU, CUDA, Vulkan/WGPU-class, and Metal-capable deployment paths where IREE supports the target/runtime combination.
8. Embedded or external artifact loading.
9. Stable artifact metadata and compatibility checks.
10. Clear compile-time diagnostics for unsupported operations.

## 5. Unified-framework requirements

The framework is compelling only when it offers all of the following from the same model definition:

### 5.1 Framework UX

- no mandatory `build.rs` authoring;
- no separate tracing tensor language;
- eager execution and debugging;
- `model.summary()` and graph inspection;
- custom modules through ordinary Rust;
- train/eval mode;
- parameter/state traversal;
- optimizer and Trainer integration;
- straightforward unit tests.

### 5.2 Dynamic flexibility

- exact static dimensions;
- named symbolic dimensions;
- bounded dynamic dimensions;
- equality and affine constraints;
- bucketed specialization;
- generic fallback where legal;
- safe recompile when explicitly allowed;
- no stale-specialization execution.

### 5.3 PyTorch interoperability

- strict safetensors/state-dict import report;
- PyTorch-compatible parameter naming;
- `torch.export` import;
- ONNX import using the modern exporter path;
- DLPack CPU and CUDA exchange;
- `torch.compile(backend="incin")` inference;
- AOTAutograd forward/backward compilation;
- custom-op fallback/registration;
- parity tools that compare Rust and PyTorch outputs and gradients.

### 5.4 Deployment and observability

- artifact cache;
- explicit target selection;
- reproducibility manifest;
- capability report;
- compiler explanation report;
- memory plan report;
- graph/fusion visualization;
- benchmark command;
- no hidden host transfers or silent fallback.

### 5.5 Native and distributed differentiation

- retain Incin native CPU/CUDA/WGPU/Metal execution;
- use native paths for unsupported IREE regions only when an explicit partition policy permits it;
- preserve placement and collective semantics;
- later place local IREE regions inside Incin’s distributed plan rather than asking IREE to replace the mesh/collective layer.

---

## 6. Public framework-completion gate

Do not make broad framework-completeness or performance claims until all twelve gates pass:

| Gate | Required evidence |
|---|---|
| O-1 | Same Incin `Module` runs eagerly and through IREE without model rewrite. |
| O-2 | Static MLP/CNN/Transformer examples compile and run on at least LLVM CPU and CUDA. |
| O-3 | One artifact can contain at least two target variants and selects a compatible one with an explanation. |
| O-4 | Build-time AOT and runtime cached compilation both work. |
| O-5 | Dynamic batch and sequence-length examples run under documented guards/buckets. |
| O-6 | `torch.export` model imports into the same canonical IR. |
| O-7 | `torch.compile(..., backend="incin")` passes an inference suite. |
| O-8 | DLPack CPU and CUDA paths demonstrate no mandatory host copy. |
| O-9 | Eager/compiled/PyTorch numerical parity passes for representative models and dtypes. |
| O-10 | Benchmark suite shows Incin executable performance within the declared tolerance of direct IREE execution and the selected PyTorch baseline on equivalent graphs, shapes, dtypes, and targets. |
| O-11 | Compiler diagnostics identify unsupported node, source/module path, reason, and remediation. |
| O-12 | A new-user tutorial reaches compiled inference without writing `build.rs`, MLIR, or backend-specific code. |

A suggested performance tolerance for the first public comparison is:

- warm execution latency: no worse than 5% versus direct execution of the same compiled artifact, and reported against PyTorch eager/compiled baselines for the same model, unless a difference is explained by runtime configuration;
- cold build/compile latency: no worse than 20%, with cache hits faster;
- artifact size: no worse than 15% after equivalent metadata/variant normalization;
- zero numerical regressions outside the declared dtype tolerance.

These are starting budgets, not permanent promises. Record the exact environment.

---

# Part III — Target architecture

## 7. Architectural invariant

The compiler is an additional consumer of Incin’s semantic operation lowering:

```text
                 ordinary Incin model / tensor API
                              |
                              v
                typed shape and operation traits
                              |
                              v
                ShapeRule::lower -> Validated<Spec>
                              |
          +-------------------+--------------------+
          |                   |                    |
          v                   v                    v
      eager native       capture canonical     distributed
       executor                IR                lowering
                                  |
                                  v
                        optimization pipeline
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
          Incin native plan                  IREE compiler target
                 |                                 |
                 v                                 v
        native kernels/library                 VMFB variants
                 +----------------+----------------+
                                  |
                                  v
                         typed CompiledModel
```

No compiler path may independently recompute shape semantics already established by `ShapeRule` unless running paranoid verification.

---

## 8. Crate and dependency layout

Use minimal disruption to the existing public API while separating no-std IR from std-only compiler orchestration.

### 8.1 Keep in `incin-core`

`incin-core::compiled` should own:

- canonical IR data types;
- graph verifier;
- shape constraints and guards;
- compile options and policies;
- backend-neutral executable plan schema;
- artifact manifest schema types;
- graph hashing/canonicalization interfaces;
- input/output pytree contracts;
- source/provenance metadata.

It must remain compatible with `no_std + alloc` where the existing crate promises it.

### 8.2 Add `crates/incin-compiler`

Responsibilities:

- pass manager;
- canonicalization;
- constant folding;
- partitioning;
- fusion planning;
- liveness and buffer planning;
- compiler target registry;
- target-independent compilation reports;
- filesystem cache;
- subprocess and embedded compiler hosts;
- artifact packager;
- build-time integration.

This crate is `std`-only and optional from the facade.

### 8.3 Add `crates/incin-iree`

Responsibilities:

- Incin IR to StableHLO/standard MLIR emission;
- IREE compiler discovery and version probing;
- `iree-compile` subprocess integration first;
- IREE C runtime binding or vetted Rust wrapper integration;
- device/driver selection;
- VMFB module loading;
- parameter provider binding;
- typed runtime invocation;
- IREE-specific target/capability reporting.

Features:

```toml
iree = []
iree-compiler = ["iree"]
iree-runtime = ["iree"]
iree-c-api = ["iree-runtime"]
iree-python-toolchain = ["iree-compiler"]
```

Do not put IREE dependencies in default Incin builds.

### 8.4 Add `crates/incin-interop`

Responsibilities:

- DLPack C structures and safe ownership wrappers;
- CPU buffer protocol integration where useful;
- CUDA stream/event bridge abstractions;
- state-dict key mapping and conversion reports;
- common ABI descriptors shared with Python bindings.

This can be split later; one focused crate is faster initially.

### 8.5 Add `python/incin_torch`

Python package responsibilities:

- `torch.export` import;
- `torch.compile` backend registration;
- AOTAutograd wrapper;
- PyTorch pytree-to-Incin ABI conversion;
- DLPack exchange;
- canonical `torch.export`/ATen-to-Incin-IR integration;
- artifact generation and load;
- parity/diagnostic tools.

Use `maturin`/PyO3 only for the pieces that need a native bridge. Keep graph extraction and user-facing APIs in Python for rapid iteration.

### 8.6 Facade feature additions

Suggested facade features:

```toml
compile = ["dep:incin-compiler"]
iree = ["compile", "dep:incin-iree"]
interop = ["dep:incin-interop"]
dlpack = ["interop"]
```

Do not enable them by default.

---

## 9. Canonical compiler IR

The current `Graph` remains useful for ONNX import, visualization, and generic graph interchange. It must not be the executable compiler IR unless upgraded to satisfy every requirement below. The fastest low-risk path is to introduce a richer canonical IR under `incin-core::compiled::ir` and provide conversion to/from the existing graph where fidelity allows.

### 9.1 Stable identifiers

Use dense IDs that are independent of vector position:

```rust
#[repr(transparent)]
pub struct FunctionId(u32);
#[repr(transparent)]
pub struct BlockId(u32);
#[repr(transparent)]
pub struct NodeId(u32);
#[repr(transparent)]
pub struct ValueId(u32);
#[repr(transparent)]
pub struct SymbolId(u32);
#[repr(transparent)]
pub struct ParameterId(u64);
```

Never use `node.id == topological_index` as an invariant.

### 9.2 Tensor type

```rust
pub struct TensorType {
    pub dtype: DTypeId,
    pub shape: ShapeExpr,
    pub layout: LayoutRequirement,
    pub placement: PlacementExpr,
    pub mutability: ValueMutability,
}
```

`ShapeExpr`:

```rust
pub struct ShapeExpr {
    pub dims: ShapeBuf<DimExpr>,
}

pub enum DimExpr {
    Static(usize),
    Symbol(SymbolId),
    Affine {
        symbol: SymbolId,
        multiplier: i64,
        offset: i64,
    },
    Runtime(RuntimeDimId),
}
```

Initial constraint vocabulary:

```rust
pub enum ShapeConstraint {
    Eq(DimExpr, DimExpr),
    Range { dim: DimExpr, min: usize, max: Option<usize> },
    Divisible { dim: DimExpr, divisor: usize },
    NonZero(DimExpr),
    ProductEq { lhs: SmallVec<DimExpr>, rhs: SmallVec<DimExpr> },
}
```

Do not add a general symbolic algebra engine initially. Support constants, symbols, univariate affine expressions, ranges, equality, divisibility, and checked products. This aligns with practical dynamic batch/sequence use cases and maps well to PyTorch export constraints.

### 9.3 Value metadata

```rust
pub struct IrValue {
    pub id: ValueId,
    pub ty: TensorType,
    pub origin: ValueOrigin,
    pub alias: AliasInfo,
    pub debug_name: Option<InternedString>,
    pub source: Option<SourceLocation>,
}

pub enum ValueOrigin {
    Input { index: u16 },
    Parameter { id: ParameterId },
    Constant { section: ConstantRef },
    NodeOutput { node: NodeId, index: u16 },
}
```

### 9.4 Operation descriptor

Do not compile from `OpType` plus stringly attributes. Create one canonical, serializable, type-erased descriptor enum whose variants wrap the same checked information used by eager execution.

```rust
#[non_exhaustive]
pub enum OpDescriptor {
    Pointwise(PointwiseSpec),
    Broadcast(BroadcastSpec),
    MatMul(MatMulSpec),
    Reduction(ReductionSpec),
    Conv1d(Conv1dSpec),
    Conv2d(Conv2dSpec),
    Pool2d(Pool2dSpec),
    Reshape(ReshapeSpec),
    Transpose(TransposeSpec),
    Slice(SliceSpec),
    Concat(ConcatSpec),
    Gather(GatherSpec),
    Scatter(ScatterSpec),
    Softmax(SoftmaxSpec),
    Norm(NormSpec),
    Attention(AttentionSpec),
    Cast(CastSpec),
    Custom(CustomOpSpec),
}
```

Each public typed descriptor must convert into this enum only through a sealed/witnessed path:

```rust
impl FromValidated<MatMulSpec> for OpDescriptor { ... }
```

The canonical descriptor must include semantic parameters, not just geometry. For example, `ReductionSpec` must identify reduction kind; `Pool2dSpec` must identify max versus average and padding/count policy.

### 9.5 Node and function

```rust
pub struct IrNode {
    pub id: NodeId,
    pub descriptor: OpDescriptor,
    pub inputs: SmallVec<ValueId>,
    pub outputs: SmallVec<ValueId>,
    pub effects: EffectSet,
    pub source: Option<SourceLocation>,
    pub module_path: Option<InternedString>,
}

pub struct IrFunction {
    pub name: InternedString,
    pub inputs: SmallVec<ValueId>,
    pub outputs: SmallVec<ValueId>,
    pub values: DenseMap<ValueId, IrValue>,
    pub nodes: Vec<IrNode>,
    pub constraints: Vec<ShapeConstraint>,
}
```

Initial `EffectSet`:

- pure;
- reads parameter;
- writes state;
- random;
- synchronization;
- collective;
- host callback;
- custom/unknown.

Pure-only graphs are the first IREE milestone. Random/dropout, mutation, and host effects must be rejected or explicitly legalized.

### 9.6 Parameters and constants

Use an external parameter archive by default, not giant Rust byte arrays.

```rust
pub struct ParameterRecord {
    pub id: ParameterId,
    pub name: InternedString,
    pub dtype: DTypeId,
    pub shape: ShapeBuf<usize>,
    pub checksum: Digest,
    pub section: ParameterSection,
}
```

Support:

- safetensors-backed archive;
- IREE parameter archive/provider where useful;
- embedded small constants;
- external large parameters;
- deduplication for shared weights;
- deterministic parameter IDs derived from global name plus declared shape/dtype, with collision checks.

### 9.7 IR verifier

`IrModule::verify()` must check at least:

1. unique IDs;
2. input/output existence;
3. topological dominance;
4. node arity;
5. descriptor/input/output compatibility;
6. dtype legality;
7. shape constraints are well-formed;
8. every parameter record exists and matches type;
9. aliases refer to valid storage roots;
10. effect ordering is legal;
11. graph outputs are defined;
12. no stale descriptor schema;
13. no duplicate names where names are declared unique;
14. checked cardinality and byte size;
15. placement consistency.

Every pass must call the verifier in test/debug mode. Artifact creation always verifies.

### 9.8 Canonical hashing

Graph hashes must be independent of incidental allocation/order where semantics are unchanged. Hash:

- schema version;
- canonical function order;
- stable opcode IDs;
- descriptor fields in stable field order;
- tensor types;
- constraints;
- parameter IDs and checksums;
- compilation-relevant policies.

Do not hash debug names unless requested for debug identity. Do not rely on Rust `Hash` implementation stability across versions.

Use BLAKE3 or SHA-256. Store a hash algorithm ID.

---

# Part IV — Capture and public model compilation

## 10. Replace global tracing with session-scoped symbolic capture

### 10.1 Capture context

```rust
pub struct CaptureContext {
    module: IrModuleBuilder,
    policy: CapturePolicy,
    scope_stack: Vec<ModuleScope>,
    diagnostics: DiagnosticSink,
}
```

Requirements:

- no process-global mutable graph;
- nested capture rejected with a structured error unless explicitly supported;
- concurrent captures are independent;
- deterministic IDs within a capture;
- source/module paths tracked;
- no backend kernel execution required;
- no device required for pure compile-time capture;
- parameter reads are symbolic;
- random/effectful operations recorded explicitly.

### 10.2 Symbolic storage/backend

Introduce a capture backend whose storage is metadata plus `ValueId`, not real memory:

```rust
pub struct CaptureStorage<K: DType> {
    value: ValueId,
    meta: TensorMeta,
    _kind: PhantomData<K>,
}
```

It must implement the minimum storage/execution interfaces needed to let ordinary module code run symbolically.

The capture executor must receive `Validated<Spec>` and emit an IR node. It must not reconstruct semantics from runtime method names.

### 10.3 Input/output pytree

Support one tensor, tuples, arrays, and named structs through a trait:

```rust
pub trait TraceTree: Sized {
    type Spec: TreeSpec;
    fn make_symbolic(ctx: &mut CaptureContext, spec: &Self::Spec) -> Result<Self>;
    fn collect_values(&self, out: &mut Vec<ValueId>);
}
```

Generate tuple implementations to a documented arity. Allow `#[derive(TraceTree)]` for user structs.

### 10.4 Blanket compile trait

```rust
pub trait Compilable<I>: Module<I>
where
    I: TraceTree,
    Self::Output: TraceTree,
{
    fn compile<B: CompilationHost>(
        &self,
        context: &ExecutionContext<B>,
        input: <I as TraceTree>::Spec,
        options: CompileOptions,
    ) -> Result<CompiledModel<I, Self::Output>>;
}
```

Prefer a blanket implementation when trait coherence permits. Otherwise generate through `#[module]`.

### 10.5 Public API tiers

#### Runtime cached compilation

```rust
let compiled = model.compile(
    &context,
    input_spec![Tensor<s![Batch, 784], F32>],
    CompileOptions::default()
        .target(CompileTarget::Iree(IreeTarget::Auto))
        .dynamic_shapes(DynamicShapePolicy::Bucketed),
)?;
```

#### AOT build script convenience

```rust
fn main() -> incin_build::Result<()> {
    incin_build::compile::<Model, Inputs>(
        "model",
        CompileOptions::deployment()
            .targets([IreeTarget::LlvmCpu, IreeTarget::Cuda]),
    )
}
```

This is optional convenience, not the only way to use compiled execution.

#### CLI AOT

```text
cargo incin compile \
  --bin infer \
  --model model_factory \
  --input-spec specs/resnet.toml \
  --target llvm-cpu \
  --target cuda \
  --output model.incin
```

### 10.6 Capture acceptance tests

Required tests:

- dtype retention for F16/BF16/F32/F64/I32;
- transpose permutation retained;
- reduction axes and kind retained;
- convolution stride/padding/dilation/groups retained;
- parameter names and shared parameter identity retained;
- nested module source path retained;
- no actual storage allocation during capture;
- two concurrent captures do not interfere;
- dynamic/named dimension constraints retained;
- input/output pytrees round-trip;
- unsupported effect produces a structured diagnostic;
- current global tracing APIs either delegate to sessions or are deprecated.

Mutation examples:

- force every dtype to F32—tests must fail;
- drop reduction axis—tests must fail;
- reuse the same global capture—concurrency test must fail;
- erase parameter names—state-dict/artifact test must fail.

---

# Part V — Real compiled execution

## 11. Native reference executable before IREE

Before adding IREE, implement a slow but semantically correct backend-neutral reference executor for canonical IR. This is essential for differential testing and isolates frontend bugs from IREE lowering bugs.

### 11.1 `IrInterpreter`

The interpreter may dispatch each node through existing Incin `Execute<O>` paths. Performance is not the objective.

Responsibilities:

- bind typed runtime inputs;
- verify guards;
- bind parameters;
- execute nodes topologically;
- retain outputs;
- release dead temporaries according to liveness in debug mode only after parity is established;
- return type-erased runtime outputs that can be safely converted to typed tensors through witnesses.

### 11.2 Runtime value abstraction

```rust
pub struct RuntimeValue<B: StorageBackend> {
    pub ty: TensorType,
    pub storage: ErasedStorage<B>,
    pub placement: PlacementKind,
}
```

Do not use `Any` without a checked dtype/storage tag. Every downcast must be preceded by schema validation.

### 11.3 Executable plan

```rust
pub enum PlanStep {
    ExecuteNative(NativeStep),
    ExecuteFused(FusedStep),
    ExecuteIree(IreeStep),
    Copy(CopyStep),
    Materialize(MaterializeStep),
    Collective(CollectiveStep),
    Barrier(BarrierStep),
}

pub struct ExecutablePlan {
    pub function: FunctionId,
    pub guards: GuardProgram,
    pub buffers: BufferPlan,
    pub steps: Vec<PlanStep>,
    pub outputs: Vec<ValueBinding>,
    pub report: PlanReport,
}
```

The first milestone may only emit `ExecuteNative` steps. The same plan container later hosts IREE and distributed regions.

### 11.4 Correct guards

Replace `ShapeGuard { expected_shape: Vec<usize>, F32 }` with a guard bytecode/program:

```rust
pub enum GuardOp {
    CheckRank { input: u16, rank: u8 },
    CheckDType { input: u16, dtype: DTypeId },
    BindDim { symbol: SymbolId, input: u16, axis: u8 },
    CheckEq { lhs: DimOperand, rhs: DimOperand },
    CheckRange { value: DimOperand, min: usize, max: Option<usize> },
    CheckDivisible { value: DimOperand, divisor: usize },
    CheckProductEq { lhs: SmallVec<DimOperand>, rhs: SmallVec<DimOperand> },
}
```

Guard execution returns structured failures with input name, axis, expected relationship, actual value, source/module path, and allowed actions.

### 11.5 Buffer plan

```rust
pub struct BufferRequirement {
    pub value: ValueId,
    pub bytes: SizeExpr,
    pub alignment: Alignment,
    pub dtype: DTypeId,
    pub device: DeviceClass,
    pub lifetime: LiveRange,
    pub alias: AliasInfo,
    pub persistence: Persistence,
}
```

Allocation reuse requires compatibility of:

- non-overlapping lifetime;
- device/storage class;
- sufficient bytes;
- alignment;
- alias safety;
- mutability;
- saved-for-backward status;
- external observation.

The plan should report peak bytes before/after reuse and every prevented reuse reason in debug mode.

### 11.6 Passes required before IREE

Implement only correctness-preserving passes that make the IR stable:

1. verification;
2. canonical topological order;
3. dead-code elimination for pure nodes;
4. constant deduplication;
5. no-op view elimination;
6. explicit broadcast insertion;
7. explicit cast insertion;
8. effect ordering;
9. guard generation;
10. target support classification.

Do not spend the first month building a complete native optimizer. IREE will initially own whole-graph fusion, dispatch formation, scheduling, bufferization, and code generation.

---

## 12. Fix or supersede current CMP implementations

Track these as new hardening IDs so history remains honest.

### CMP2-001 — Canonical executable IR

**Depends on:** EXE-003, EXE-006  
**Target:** `incin-core/src/compiled/ir/*`  
**Deliverable:** full value metadata, descriptor enum, constants/parameters, constraints, verifier, stable hash.  
**Evidence:** semantic round-trip tests for all Tier-0 operations; malformed IR mutants rejected.

### CMP2-002 — Session capture

**Depends on:** CMP2-001  
**Target:** `compiled/capture.rs`, `tensor/tracing.rs`, capture backend  
**Deliverable:** symbolic capture through `Validated<Spec>`, no global graph, dtype/attribute fidelity.  
**Evidence:** concurrency, no-allocation, descriptor parity, source-path tests.

### CMP2-003 — Reference executable

**Depends on:** CMP2-002  
**Target:** `compiled/interpreter.rs`, runtime value bindings  
**Deliverable:** canonical IR executes through existing backend descriptors.  
**Evidence:** eager/interpreter parity for MLP, CNN, attention block.

### CMP2-004 — Real guard program

**Depends on:** CMP2-001  
**Target:** `compiled/guard.rs`  
**Deliverable:** static, named, bounded, affine, divisibility and product guards.  
**Evidence:** accepted/rejected shape matrix and remediation messages.

### CMP2-005 — Byte-aware buffer planner

**Depends on:** CMP2-003  
**Target:** `compiled/alloc.rs`  
**Deliverable:** bytes/alignment/device/alias/lifetime planning and report.  
**Evidence:** exact peak memory tests; alias/saved tensor mutants.

### CMP2-006 — Real constant folding/prepacking contract

**Depends on:** CMP2-003  
**Target:** `compiled/fold.rs`  
**Deliverable:** evaluator-backed constant folding and target-owned prepack records.  
**Evidence:** outputs equal eager; folded node removal; checksum and invalidation tests.

### CMP2-007 — Semantic fusion regions

**Depends on:** CMP2-001  
**Target:** `compiled/fusion.rs`, existing pointwise AST  
**Deliverable:** fusion creates a real pointwise expression/subgraph, preserves semantics, handles broadcasts and multi-use.  
**Evidence:** launch/node reduction plus numerical parity; `relu(neg(x))` mutant.

### CMP2-008 — Artifact v2

**Depends on:** CMP2-004  
**Target:** `compiled/artifact.rs`  
**Deliverable:** sectioned binary container, robust hash, executable variants, parameter refs, ABI/fingerprint.  
**Evidence:** corruption, truncation, unknown-section, version, and target mismatch tests.

### CMP2-009 — Public `model.compile/run`

**Depends on:** CMP2-003, CMP2-004  
**Target:** facade and module API  
**Deliverable:** same model eager and compiled.  
**Evidence:** public examples and trybuild diagnostics.

### CMP2-010 — Real tuning host

**Depends on:** CMP2-003  
**Target:** `compiled/tuning.rs`  
**Deliverable:** injected measurement host, real samples, no synthetic timing in production path.  
**Evidence:** deterministic fake-host unit tests and actual benchmark integration tests.

---

# Part VI — IREE compiler target

## 13. Strategy

Use IREE because it already accepts MLIR input dialects including StableHLO, TOSA, and Linalg and performs global optimization, dispatch formation, scheduling, bufferization, and target code generation. The official IREE bindings list C/C++ and Python as supported; Rust bindings are unofficial/experimental. Therefore:

### Phase 1 integration

- emit textual StableHLO/standard MLIR;
- invoke a pinned `iree-compile` executable as a subprocess;
- probe and record exact compiler version and flags;
- use a runtime binding that is either generated from the official C API or a narrowly audited existing wrapper;
- isolate all IREE details in `incin-iree`.

### Phase 2 integration

- add the official compiler embedding C API for lower startup overhead and better diagnostics;
- retain subprocess mode as a reproducible/debug option;
- compare outputs between modes.

Do not block the beta on embedded compiler integration.

---

## 14. IREE lowering design

### 14.1 First input dialect: StableHLO plus standard MLIR

StableHLO is the best first target because:

- it has stable operation semantics;
- IREE accepts it as an input format;
- it represents common ML operations and dynamic tensor dimensions;
- it aligns with external framework interchange;
- unsupported low-level details can fall back to `linalg`, `tensor`, `arith`, and `math` dialects later.

Do not create a custom Incin MLIR dialect in the first release.

### 14.2 Emitter

```rust
pub trait CompilerTarget {
    fn name(&self) -> &'static str;
    fn classify(&self, node: &IrNode, module: &IrModule) -> SupportDecision;
    fn compile(&self, request: CompileRequest<'_>) -> Result<CompiledVariant>;
}

pub struct IreeTarget { ... }
```

`StableHloEmitter` must:

- emit deterministic symbol names;
- emit tensor types with static/dynamic dimensions;
- bind external parameters;
- preserve numeric policies;
- emit source locations where possible;
- produce a side-table mapping MLIR ops to Incin node IDs;
- validate emitted MLIR with an independent parser/tool in CI;
- preserve entry function ABI.

### 14.3 Tier-0 operation lowering

Implement in this order:

1. constants and parameters;
2. add/sub/mul/div and scalar variants;
3. abs/neg/exp/log/sqrt/tanh/sigmoid/relu/gelu;
4. cast;
5. reshape, squeeze, unsqueeze, transpose;
6. broadcast;
7. sum/mean/max/min reductions;
8. matmul and batched matmul;
9. concat;
10. comparisons and select/where;
11. softmax decomposed or direct;
12. conv2d;
13. max/average pool;
14. layer norm;
15. embedding/gather/index-select;
16. scaled dot-product attention decomposition.

This set is enough for useful MLP, CNN, transformer encoder, and many imported models.

### 14.4 Tier-1 lowering

- conv1d and transposed convolution;
- batch/group/instance norm;
- pad, repeat, slice, narrow;
- top-k/argsort where target support is adequate;
- scatter/masked fill;
- pixel shuffle/unfold;
- losses for compiled training;
- quantized primitives after dtype semantics are frozen.

### 14.5 Unsupported operation behavior

Default policy is whole-graph compilation failure with an actionable report:

```text
cannot compile function `Transformer::forward` for IREE/CUDA

unsupported node:
  module: blocks.3.attention
  operation: Scatter
  source: src/model.rs:118
  dtype/layout: f16, rank 4, strided

reason:
  Incin StableHLO lowering for Scatter does not support this reduction mode

options:
  - use `CompilePartitionPolicy::ExplicitFallback`
  - replace with `index_select` + `where`
  - run this model with the native CUDA executor
  - enable experimental `iree-linalg-fallback`
```

No silent eager fallback. No silent host round-trip.

### 14.6 Graph partitioning—later

After whole-graph IREE is reliable, add explicit partitioning:

```rust
pub enum CompilePartitionPolicy {
    WholeGraph,
    ExplicitFallback,
    PreferIree,
    PreferNative,
}
```

Partition boundaries must account for:

- device;
- layout/materialization;
- dtype;
- aliasing;
- transfer cost;
- gradient boundaries;
- placement;
- synchronization.

The plan report must list every region and boundary copy.

---

## 15. IREE compiler host

### IREE-001 — Toolchain discovery

Implement:

- explicit path option;
- environment variable;
- PATH search;
- Python package tool discovery;
- exact `--version` probe;
- supported-target probe;
- diagnostic report;
- cache identity.

Never download a compiler implicitly during a normal build. `cargo incin setup iree` may install with explicit user action.

### IREE-002 — Deterministic MLIR emission

Golden tests for every Tier-0 op. Parse/compile emitted files in CI where toolchain is available.

### IREE-003 — Subprocess compiler

Requirements:

- timeout;
- bounded stdout/stderr capture;
- temporary directory cleanup;
- atomic output move;
- command reproduction in diagnostics;
- source-map translation from MLIR diagnostics to Incin node/module/source;
- exact compiler flags in artifact;
- no shell string interpolation—pass arguments directly.

### IREE-004 — Runtime engine

```rust
pub struct IreeEngine {
    driver: IreeDriver,
    device: IreeDevice,
    modules: ModuleCache,
    parameters: ParameterProvider,
}
```

Requirements:

- reusable engine;
- module cache;
- typed input validation;
- runtime driver selection;
- external parameter binding;
- asynchronous invocation where supported;
- explicit synchronization API;
- zero-copy import/export hooks;
- structured runtime errors;
- no global singleton requirement.

### IREE-005 — Multi-variant artifact

At least:

- `llvm-cpu` + local-task/local-sync runtime;
- CUDA target/driver;
- Vulkan target/driver;
- Metal target/driver where supported by the pinned IREE build.

Variant selection uses actual capability/driver compatibility, not “first variant”. Return a rejection report for every candidate.

### IREE-006 — Build-time generation

Provide an ergonomic helper that writes generated typed wrappers to `OUT_DIR`, supporting conventional build-time deployment for users who prefer it, while retaining the normal runtime/CLI path.

### IREE-007 — Embedded compiler API

Post-beta. Bind official versioned compiler C API. Compare bytecode and diagnostics against subprocess mode.

---

# Part VII — Artifact format and cache

## 16. `.incin` artifact v2

Use a sectioned binary container:

```text
Header
  magic = "INCIN\0\x02"
  container schema
  endianness
  section count
  manifest offset/length
  root digest

Sections
  MANIFEST
  IR_DEBUG              optional
  SOURCE_MAP            optional
  PARAMETER_INDEX
  PARAMETERS            optional embedded
  VMFB_VARIANT[n]
  NATIVE_PLAN_VARIANT[n]
  TUNING_PROVENANCE     optional
  REPRODUCIBILITY
```

### 16.1 Manifest

Must include:

- container schema;
- IR schema;
- descriptor schema;
- Incin crate version and Git commit when available;
- compiler target kind;
- exact IREE compiler version/build identifier;
- exact runtime ABI requirement;
- function names;
- typed input/output tree schemas;
- dynamic constraints;
- parameter records and checksums;
- variant IDs;
- target/backend/driver;
- required CPU features or GPU capabilities;
- compile flags;
- precision, determinism, fallback and math policies;
- graph hash;
- source map checksum;
- build timestamp only in non-reproducible metadata, not semantic hash.

### 16.2 Integrity

- BLAKE3 or SHA-256 per section;
- root digest over canonical section table;
- checked offset/length arithmetic;
- maximum section and file size policy;
- duplicate section rejection;
- unknown optional section preservation/skipping;
- unknown required section rejection;
- atomic cache writes;
- no deserialization allocation from unbounded lengths.

### 16.3 Compatibility

Do not use semver major alone, especially while the project is `0.0.0`.

Compatibility checks:

- exact container schema support;
- compatible IR/descriptor schema range;
- runtime ABI ID;
- target/driver availability;
- compiler-generated variant requirements;
- parameter checksum;
- dtype/layout support;
- shape guard compatibility;
- device fingerprint for tuned artifacts.

### 16.4 Cache key

```text
hash(
  graph_semantic_hash,
  parameter_checksums_or_external_ids,
  input_shape_signature,
  dynamic_policy,
  target,
  target_features,
  precision,
  determinism,
  compiler_version,
  compiler_flags,
  IR_schema,
  descriptor_schema,
  runtime_ABI
)
```

Cache requirements:

- bounded LRU/size policy;
- single-flight compile lock;
- stale lock recovery;
- corruption quarantine;
- negative cache only for bounded time;
- explain cache hit/miss;
- explicit clean command;
- no reuse across incompatible device/toolchain fingerprints.

---

# Part VIII — Dynamic shapes without losing safety

## 17. Dynamic shape policy

Replace the two-state `Guarded/Strict` enum with:

```rust
pub enum DynamicShapePolicy {
    Exact,
    Guarded,
    Bucketed(ShapeBucketPolicy),
    Generic,
    Recompile(RecompilePolicy),
}
```

Semantics:

- **Exact:** all dimensions fixed; mismatch fails.
- **Guarded:** one symbolic artifact with runtime constraints where target supports it.
- **Bucketed:** bounded set of compiled signatures; select smallest compatible bucket.
- **Generic:** use target’s generic dynamic kernels and no specialization beyond required guards.
- **Recompile:** on mismatch, compile a new bounded signature under explicit cache/budget policy.

### 17.1 Shape signature

```rust
pub struct ShapeSignature {
    pub inputs: Vec<InputShapeSignature>,
    pub symbol_bindings: Vec<SymbolBinding>,
    pub bucket_classes: Vec<BucketClass>,
}
```

Bucket examples:

- exact small dimensions;
- powers of two for sequence length;
- batch classes 1, 2–4, 5–8, 9–16;
- tensor-core multiples;
- generic overflow bucket.

### 17.2 Recompile safety

- compile outside execution locks;
- single-flight per key;
- bound compilation count and wall-time;
- retain a valid old/generic path while compiling;
- never run stale artifact after guard failure;
- expose recompile count/latency;
- allow disabling at deployment.

### 17.3 PyTorch constraint mapping

Map PyTorch export constraints into Incin symbols:

- static dimension -> `Static`;
- named `Dim` -> `Symbol` plus range;
- affine relation -> `Affine` plus equality;
- divisibility/assertion -> matching guard;
- unsupported nonlinear relation -> runtime guard/custom constraint or reject with explanation.

### 17.4 Dynamic acceptance suite

- variable batch MLP;
- variable sequence transformer;
- equality between Q/K hidden dimensions;
- affine relation `y = 2*x + 4`;
- divisibility for heads/sharding;
- invalid range;
- zero dimension;
- overflow boundary;
- bucket selection;
- recompile stampede;
- stale cache rejection;
- generic fallback parity.

---

# Part IX — PyTorch interoperability program

## 18. Principles

Do not parse arbitrary pickle-based `.pt`/`.pth` files directly in Rust. That is insecure, version-sensitive, and unnecessary.

Use supported interchange surfaces:

1. safetensors/state dictionaries for parameters;
2. `torch.export.ExportedProgram` for normalized functional ATen graphs and shape constraints;
3. modern ONNX export as a broad interchange fallback;
4. DLPack for tensor memory exchange;
5. `torch.compile` custom backend for live integration;
6. AOTAutograd for training graphs.

All Python graph paths must lower through the canonical Incin IR. A separate bypass would duplicate operation coverage, diagnostics, constraints, artifacts, and custom-op behavior, undermining seamless interchange.

---

## 19. State-dict and safetensors hardening

### TORCH-001 — Typed state archive

Fix the current loader so storage dtype is not silently represented as `B::FloatElem` regardless of source dtype.

Introduce:

```rust
pub struct StateArchive {
    entries: BTreeMap<String, StateTensor>,
}

pub struct StateTensor {
    pub dtype: DTypeId,
    pub shape: ShapeBuf<usize>,
    pub bytes: Arc<[u8]>,
    pub byte_order: ByteOrder,
}
```

Loading into a module applies an explicit conversion policy:

```rust
pub enum StateLoadPolicy {
    Exact,
    CastFloating,
    CastCompatible,
}
```

### TORCH-002 — Strict load report

```rust
pub struct StateLoadReport {
    pub loaded: Vec<LoadedParameter>,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub shape_mismatches: Vec<...>,
    pub dtype_conversions: Vec<...>,
    pub renamed: Vec<...>,
}
```

Default is strict. Provide PyTorch-like `strict=false` explicitly.

### TORCH-003 — Key mapping

Support:

- exact keys;
- prefix strip/add;
- regex mapping;
- common PyTorch module aliases;
- user callback;
- generated mapping report;
- collision rejection.

### TORCH-004 — Device-aware loading

- CPU staging is explicit;
- direct GPU load where backend supports it;
- asynchronous copies with lifecycle tracking;
- no hidden CPU-only assumption;
- load report includes destination and conversions.

---

## 20. Python package

Directory:

```text
python/incin_torch/
  pyproject.toml
  src/incin_torch/
    __init__.py
    backend.py
    export.py
    artifact.py
    dlpack.py
    diagnostics.py
    ops.py
    parity.py
    _native.*
  tests/
```

Public API:

```python
incin_torch.compile(model, example_inputs, options=...)
incin_torch.export(model, example_inputs, output=..., dynamic_shapes=...)
incin_torch.load_artifact(path)
incin_torch.compare(model, artifact, samples=...)
incin_torch.explain(model, example_inputs)
```

Register the backend through the `torch_dynamo_backends` entry point so users can write:

```python
torch.compile(model, backend="incin")
```

---

## 21. `torch.export` importer

### TORCH-005 — ExportedProgram schema bridge

Python extracts:

- graph nodes;
- normalized ATen targets;
- args/kwargs pytrees;
- tensor metadata;
- range constraints;
- graph signature;
- parameters/buffers/constants;
- module call graph/source stacks where available.

Serialize to a stable, versioned bridge schema consumed by Rust. Prefer Cap’n Proto/FlatBuffers/Postcard-style bounded binary over ad hoc Python pickle. JSON is acceptable only for the first internal prototype and must not become the artifact ABI.

### TORCH-006 — ATen-to-Incin operation registry

Create a generated registry:

```text
aten.add.Tensor            -> Pointwise(Add)
aten.mm.default             -> MatMulSpec
aten.bmm.default            -> MatMulSpec(batch)
aten.view.default           -> ReshapeSpec
aten.permute.default        -> TransposeSpec
aten.sum.dim_IntList        -> ReductionSpec(Sum)
aten.convolution.default    -> ConvSpec
...
```

Each registration includes:

- supported dtypes;
- rank/layout constraints;
- static/dynamic support;
- lowering function;
- decomposition option;
- gradient support;
- parity tolerance;
- documentation.

Generate support docs and tests from this registry.

### TORCH-007 — Decomposition policy

Use PyTorch decompositions/AOTAutograd core ATen set where helpful, but record the exact decomposition table/version in the artifact. Avoid a huge frontend opset.

### TORCH-008 — Constraint importer

Translate export shape constraints to Incin `ShapeConstraint`. Reject unsupported relations with a precise source stack and suggested shape policy.

### TORCH-009 — Parameter import

Export parameters to safetensors/IREE parameter archive and map them to stable Incin `ParameterId`s. Never duplicate multi-gigabyte weights in bridge metadata.

---

## 22. `torch.compile(backend="incin")`

### 22.1 Inference backend

The backend contract receives an FX `GraphModule` and example inputs, then returns a callable. Implementation path:

1. obtain/export normalized graph;
2. lower the normalized graph to canonical Incin IR;
3. build/load `.incin` artifact in cache;
4. return callable using DLPack to invoke Incin/IREE runtime;
5. preserve pytree contract;
6. map runtime errors back to Python exceptions with graph/source context.

### 22.2 Canonical staging

The first backend release must already use the same canonical Incin IR as Rust-authored models. Stage functionality by operation coverage and target support, not by introducing a second compiler path:

1. FX/`torch.export` normalization and constraint extraction;
2. ATen-to-Incin descriptor lowering;
3. canonical IR verification;
4. reference compiled execution for correctness;
5. optional IREE target for performance;
6. one artifact/cache/runtime/diagnostic path for both Rust and Python models.

This is initially slower to implement than a bypass but substantially reduces long-term complexity and ensures anything built in one ecosystem can use the same graph, passes, artifacts, custom operations, and debugging tools.

### 22.3 Graph breaks

- whole-graph mode available and used in conformance tests;
- normal mode reports graph-break count and reasons;
- no hidden host copies;
- minifier integration documented;
- backend registered by string for PyTorch’s minifier workflow.

---

## 23. AOTAutograd and training

### TORCH-010 — AOTAutograd wrapper

Use the official AOTAutograd custom-backend path to receive smaller core ATen forward/backward graphs.

First milestone:

- compile forward and backward regions;
- leave optimizer step eager;
- return boxed callables as required by PyTorch;
- preserve saved tensor contract;
- compare gradients against eager PyTorch.

### INCIN-GRAD2-001 — Compile native Incin forward/backward

Do not write a second symbolic autodiff engine first. Use existing backend-neutral autograd recipes to capture a combined forward/backward IR:

1. capture forward;
2. materialize backward recipe graph symbolically;
3. mark saved values;
4. run liveness/min-cut-like save/recompute policy later;
5. compile both functions;
6. optimizer remains eager initially.

### INCIN-GRAD2-002 — Saved tensor planner

Integrate with buffer liveness:

- must-save values;
- recomputable pure values;
- recompute cost;
- peak memory;
- alias safety;
- checkpoint policy.

### Training acceptance

- linear regression one-step parity;
- MLP forward/loss/gradient parity;
- CNN gradient parity;
- transformer block gradient parity;
- F16/BF16 mixed precision after precision policy lands;
- optimizer eager step parity;
- no saved tensor use-after-free;
- no unexpected graph recording under `NoGrad`.

---

# Part X — DLPack and zero-copy interoperability

## 24. DLPack implementation

### 24.1 Ownership

Create RAII wrappers for `DLManagedTensorVersioned` and legacy form during transition. The deleter must retain the producing tensor/storage owner until the consumer releases it.

### 24.2 Layout validation

Validate:

- dtype mapping;
- device kind/id;
- rank;
- shape;
- strides or contiguous-null convention;
- byte offset;
- checked bounds;
- alignment;
- negative strides policy;
- read-only/mutation policy where available.

### 24.3 CPU

Demonstrate shared memory by mutating only in tests with explicit ownership expectations. Expose `copy` policy.

### 24.4 CUDA/ROCm streams

The consumer must communicate the stream it will use and establish correct ordering. Implement:

- producer current stream query;
- consumer stream handoff;
- event/wait when streams differ;
- lifetime retention until asynchronous work completes;
- device identity validation beyond ordinal where possible;
- explicit sync fallback only when required and reported.

### 24.5 Python protocol

Implement `__dlpack__` and `__dlpack_device__` on Incin Python tensor wrappers. Accept PyTorch tensors directly through the protocol rather than legacy one-use capsules when possible.

### 24.6 Tests

- CPU zero-copy;
- non-contiguous strided view;
- offset view;
- zero-size tensor;
- dtype matrix;
- ownership after original Python reference deletion;
- one-use legacy capsule behavior;
- CUDA same stream;
- CUDA different stream with event ordering;
- wrong device;
- explicit copy request;
- mutation warning/documentation.

---

# Part XI — Model-building and testing UX

## 25. Model API goals

Incin’s model layer should be the reason users do not author graphs in a build script.

### 25.1 Preserve ordinary Rust modules

```rust
#[module]
pub struct MLP<B: Backend> {
    fc1: Linear<s![784, 128], B>,
    fc2: Linear<s![128, 10], B>,
}

impl<B: Backend> Module<Tensor<s![Batch, 784], B>> for MLP<B> {
    type Output = Tensor<s![Batch, 10], B>;
    fn forward(&self, x: Tensor<s![Batch, 784], B>) -> Result<Self::Output> {
        self.fc2.forward(self.fc1.forward(x)?.gelu()?)
    }
}
```

The same code must support eager, test capture, compiled execution, export, and distributed planning.

### 25.2 Model builder convenience

Add optional builders for users who prefer dynamic construction:

```rust
let model = ModelBuilder::new()
    .linear("fc1", 784, 128)
    .gelu()
    .linear("fc2", 128, 10)
    .build()?;
```

This must lower to ordinary module/graph abstractions; it is not a parallel execution model.

### 25.3 Custom model support

- custom module derives;
- custom operation registration;
- shape rule;
- eager implementation;
- optional IREE lowering;
- optional PyTorch custom-op mapping;
- gradient recipe;
- capability registration;
- generated conformance test template.

One command should scaffold it:

```text
cargo incin new-op rotary_embedding
```

Generated files:

- descriptor;
- shape rule;
- eager reference;
- backend capability registration;
- IREE lowering stub;
- PyTorch mapping stub;
- forward parity test;
- gradient test;
- compile-fail shape test;
- docs section.

---

## 26. Testing toolkit

Create `incin::testing`:

```rust
assert_tensor_eq!(a, b);
assert_tensor_close!(a, b, atol = 1e-5, rtol = 1e-4);
assert_eager_compiled_parity!(model, input, options);
assert_backend_parity!(model, input, [Cpu, Cuda, Wgpu]);
assert_grad_close!(...);
assert_torch_parity!(...); // integration feature
```

### 26.1 Required test layers

1. shape compile-pass/fail;
2. descriptor unit tests;
3. IR verifier tests;
4. eager/reference interpreter differential tests;
5. IREE differential tests;
6. PyTorch differential tests;
7. gradient tests;
8. artifact corruption tests;
9. dynamic-shape property tests;
10. hardware tests;
11. performance budgets;
12. user-facing compile tests.

### 26.2 Golden snapshots

Snapshot:

- canonical IR;
- StableHLO MLIR;
- compile report;
- source map;
- artifact manifest;
- graph visualization.

Normalize incidental IDs and paths.

### 26.3 Fuzz/property tests

Targets:

- shape guard interpreter;
- IR deserialization/verifier;
- artifact section parser;
- small random pointwise graphs;
- broadcast/reduction/matmul shape combinations;
- DLPack metadata;
- state-dict key mapping;
- dynamic bucket selection.

### 26.4 Mutation policy

For each critical feature, record one known mutant and the test that kills it. Examples:

- drop consumer op during fusion;
- force F32 during capture;
- ignore transpose axes;
- permit wrong parameter checksum;
- skip guard after cache hit;
- use wrong CUDA stream;
- silently copy to CPU;
- accept unknown required artifact section;
- report simulated benchmark as measured.

---

# Part XII — CLI and diagnostics

## 27. Commands

Extend the existing `cargo-incin` dispatcher.

### 27.1 `cargo incin compile`

- select model factory/binary/example;
- input spec;
- target variants;
- dynamic policy;
- output artifact;
- external parameters;
- cache policy;
- precision/determinism;
- emit IR/MLIR/report.

### 27.2 `cargo incin explain compile`

Report:

- captured model inputs/outputs;
- dynamic symbols and constraints;
- unsupported operations;
- decompositions;
- partitions;
- fusions;
- target selection;
- buffer peak;
- parameter size;
- cache identity;
- expected runtime requirements.

### 27.3 `cargo incin diff`

```text
cargo incin diff --eager --compiled model.incin --samples inputs/
cargo incin diff --torch script.py::model --compiled model.incin
```

Produces numerical mismatch report by node when debug IR is available.

### 27.4 `cargo incin benchmark`

Metrics:

- capture time;
- MLIR emission;
- compiler time;
- artifact size;
- cold load;
- warm latency;
- throughput;
- peak memory;
- dispatch/launch count;
- host/device copies;
- cache hit/miss;
- dynamic recompiles.

### 27.5 `cargo incin inspect`

Extend artifact inspection to `.incin`:

- variants;
- ABI;
- parameters;
- shapes;
- constraints;
- compiler/runtime versions;
- checksums;
- compatibility with current host.

### 27.6 `cargo incin import torch`

Invokes Python package explicitly and records exact PyTorch/exporter/compiler/runtime versions.

### 27.7 Diagnostic quality gate

Every failure should answer:

1. what failed;
2. where in model/user source;
3. actual values;
4. expected rule;
5. why the target cannot handle it;
6. available alternatives;
7. command to reproduce or inspect.

---

# Part XIII — Performance and comparison program

## 28. Benchmark matrix

Compare:

- Incin eager CPU;
- Incin native compiled/reference;
- Incin IREE;
- direct IREE compilation/runtime baseline;
- PyTorch eager;
- PyTorch Inductor where available;
- PyTorch `backend="incin"`;
- ONNX Runtime only as an optional ecosystem reference.

Models:

1. pointwise chain;
2. MLP;
3. batched matmul;
4. small CNN;
5. ResNet18;
6. transformer encoder block;
7. GPT-style decode block with dynamic sequence/cache shape;
8. embedding + layernorm + attention;
9. custom fused op example;
10. training MLP/CNN after AOTAutograd.

Shapes:

- static small;
- static large;
- dynamic batch;
- dynamic sequence;
- non-contiguous input where supported.

Dtypes:

- F32 mandatory;
- F16/BF16 on capable targets;
- F64 CPU correctness;
- quantized later.

Targets:

- x86-64 LLVM CPU;
- AArch64 LLVM CPU when CI exists;
- CUDA;
- Vulkan;
- Metal on Apple hardware.

### 28.1 Fair compiler and framework comparison

- same mathematical graph;
- same IREE compiler build;
- same target backend and flags;
- same runtime driver;
- same parameter embedding policy;
- same warmup/sample count;
- same synchronization;
- report generated VMFB size separately from wrapper metadata;
- report whether graph compilation includes extra decompositions.

The goal is not to fabricate a win. If the VMFB and runtime are equivalent, execution should be near-equivalent. Incin wins primarily through integration and UX; performance parity is sufficient at first.

### 28.2 Hard performance gates

- no hidden host readback in compiled device-native graph;
- no per-run compiler invocation;
- no per-run module reload;
- no per-run parameter upload unless input changes require it;
- no redundant semantic shape validation inside each node;
- bounded guard overhead;
- no unbounded artifact cache;
- no simulated timings in user reports;
- compiled path must reduce launch count for pointwise chains where IREE fuses them;
- peak memory report must be within a declared tolerance of measured memory.

### 28.3 Benchmark evidence format

Every result records:

- Git commit;
- dirty state;
- Rust version;
- compiler/runtime version;
- OS/kernel;
- CPU model/features;
- GPU UUID/architecture/driver;
- target flags;
- model/parameter hash;
- input signature;
- precision/determinism;
- warmup/samples;
- median, p90, p99;
- synchronization method;
- artifact/cache state.

---

# Part XIV — Operation coverage roadmap

## 29. Coverage tiers

### Tier 0 — public beta

- Input/Constant/Parameter
- Add/Sub/Mul/Div and scalar forms
- Maximum/Minimum
- Relu/Gelu/Tanh/Sigmoid/Exp/Log/Sqrt/Abs/Neg
- MatMul/Bmm/Addmm/Linear
- Reshape/Squeeze/Unsqueeze/Transpose
- Broadcast
- Sum/Mean/Max/Min reductions
- Cast
- Concat
- comparisons/logical/select/where
- Softmax
- Conv2d
- MaxPool2d/AvgPool2d/AdaptiveAvgPool2d
- LayerNorm
- Embedding/Gather/IndexSelect
- Pad/Slice/Narrow
- SDPA through decomposition

### Tier 1 — release candidate

- Conv1d
- ConvTranspose2d
- BatchNorm/GroupNorm/InstanceNorm
- Repeat
- MaskedFill
- Scatter
- TopK/Argsort
- Unfold/PixelShuffle
- losses
- richer attention/masking
- random/dropout policy

### Tier 2 — post-1.0 compiler maturity

- quantized op families;
- sparse ops;
- ragged tensors;
- complex dtype;
- custom control flow;
- mutation-heavy models;
- full optimizer compilation;
- elastic/distributed compiler regions.

## 30. Coverage registry

One generated source of truth:

| Field | Meaning |
|---|---|
| Incin opcode | Canonical semantic op |
| Descriptor schema | Required checked descriptor |
| Eager CPU/CUDA/WGPU/Metal | Support |
| IREE lowering | Support/decomposition |
| PyTorch ATen mappings | Supported overloads |
| Static/mixed/dynamic | Shape modes |
| Dtypes/layouts/ranks | Capability |
| Forward/backward | Gradient support |
| Tolerance profile | Parity thresholds |
| Tests | Exact evidence IDs |

Generate:

- docs;
- compiler classifier;
- PyTorch support report;
- conformance tests;
- CLI capability output.

---

# Part XV — Agent execution protocol

## 31. Sequential roles and handoff order

Only one implementation agent is active at a time. “Owner” means the model that receives the issue after prerequisites merge; it does not authorize concurrent edits. Reviews may be performed after the implementation agent stops, but the next implementation issue does not start until the previous PR is merged or formally abandoned.

### Human maintainer — product and release authority

Primary responsibilities:

- approve public API, release scope, priorities, and architecture decisions with user-facing consequences;
- create/approve GitHub issues and merge pull requests;
- supply credentials, hardware access, and release signing;
- decide unresolved trade-offs rather than allowing agents to invent product policy;
- own the 0.1 compatibility promise and final release tags.

### Opus 5 — architecture and adversarial review owner

Primary responsibilities:

- freeze interfaces and invariants through ADRs;
- inspect cross-crate dependency direction;
- design canonical IR, state, artifact, dynamic-shape, model, and Python boundaries;
- review safety and semantic correctness;
- design negative tests and semantic mutants;
- integrate conflicting work;
- reject placeholder completion;
- perform final review of high-risk GPT/Gemini PRs.

Assign Opus first when a task changes contracts across crates, public APIs, serialization formats, memory ownership, compiler semantics, or safety boundaries. Opus should not spend its turn on repetitive fixtures once the contract is frozen.

### GPT-5.6 — primary implementation and integration owner

Primary responsibilities:

- implement bounded cross-crate tasks from approved ADRs;
- write unit/integration/property/fuzz tests;
- implement compiler/runtime paths, state loading, Python bindings, operation lowerings, and CLI/package glue;
- run and record exact commands;
- produce small reviewable PRs;
- stop and request an ADR when a contract is missing.

Use GPT-5.6 after Opus freezes a difficult contract and before delegating repetitive coverage to Gemini.

### Gemini Pro agent 1 — bounded subsystem implementation owner

Primary responsibilities:

- implement one well-specified subsystem with exact signatures and file boundaries;
- build fixtures, format readers/writers, deterministic manifests, and conformance matrices;
- expand positive and negative tests;
- perform mechanical migrations that still require code understanding;
- report ambiguity instead of broad redesign.

This agent must receive a task packet containing pseudocode, invariants, explicit files, expected errors, and exact test commands.

### Gemini Pro/Flash agent 2 — mechanical coverage, docs, and verification owner

Primary responsibilities:

- generated tables and support matrices;
- repetitive operator/format fixtures after one canonical example exists;
- rustdoc and book examples;
- compile-pass/fail cases;
- downstream example projects;
- safety inventory classification and evidence collection;
- mutation execution and regression verification.

Do not assign this agent new cross-cutting ownership or an underspecified compiler pass. It should copy a proven pattern and be required to demonstrate that its tests fail against a supplied mutant.

### Required sequence for a difficult feature

1. **Opus:** audit, ADR, public contract, invariants, task split.
2. **GPT-5.6:** minimal correct vertical slice and reference tests.
3. **Gemini agent 1:** broaden operation/backend/format coverage using the frozen pattern.
4. **Gemini agent 2:** docs, matrices, negative fixtures, downstream examples, and evidence replay.
5. **Opus:** adversarial final review.
6. **Human maintainer:** merge/reject and activate the next issue.

This sequence intentionally trades wall-clock parallelism for lower integration risk and better use of each model’s strengths.

---

## 32. Git worktree discipline

Start every task from the current remote `develop` head:

```bash
git fetch origin
git worktree add ../incin-CMP2-001 -b cmp2-001 origin/develop
```

Rules:

- one task per branch;
- one owner per file at a time;
- no sweeping formatting outside touched modules;
- rebase before handoff;
- commit tests with implementation;
- commit generated outputs only where repository policy requires them;
- no direct commits to `develop`;
- integration owner merges in dependency order;
- update ledger only after evidence passes on integrated head.

Maintain:

```text
docs/plan/compiler-program/
  STATUS.md
  DECISIONS.md
  RISKS.md
  SUPPORT_MATRIX.toml
  BENCHMARKS.md
  HANDOFF.md
```

`STATUS.md` contains current baseline SHA and active worktrees. `HANDOFF.md` contains no prose claims without command output or code references.

---

## 33. Completion contract for every task

Every task description must contain:

1. objective;
2. non-goals;
3. dependencies;
4. exact files/modules allowed to change;
5. public API contract;
6. invariants;
7. implementation steps;
8. positive tests;
9. negative tests;
10. semantic mutant;
11. commands to run;
12. performance/safety budget;
13. documentation changes;
14. evidence block;
15. remaining limitations.

A task is not complete when:

- code only compiles;
- a type exists but is unused;
- a test only counts nodes;
- a pass returns its input unchanged;
- a benchmark is simulated;
- a hardware test runs zero cases;
- an unsupported path silently falls back;
- the public example is ignored or non-compiling;
- the artifact cannot execute.

---

## 34. Standard agent prompt header

Give every agent this exact header:

```text
You are implementing one bounded task in the Incin compiler program.
Baseline from origin/develop: <SHA>.

Read first:
- docs/plan/compiler-program/DECISIONS.md
- docs/plan/compiler-program/STATUS.md
- the task specification
- the existing modules named by the task

Hard rules:
1. Do not redesign cross-cutting APIs without an ADR approved by the architecture owner.
2. Do not mark placeholder/no-op/simulated behavior complete.
3. Preserve no_std/default-feature contracts.
4. No silent fallback, host copy, dtype conversion, or device change.
5. Use checked arithmetic for shapes, bytes, offsets, and section lengths.
6. A semantic mutant must be killed by the test suite.
7. Run the exact evidence commands and paste concise output into the task evidence.
8. Keep the diff bounded. Report any required out-of-scope change instead of making it silently.
9. Treat current code and test output as authoritative over ledger prose.
10. Return: summary, files changed, invariants, tests, command outputs, mutant, risks, and follow-up IDs.
```

---

## 35. Review checklist

Architecture owner rejects a PR unless all applicable answers are yes:

- Does the implementation consume `Validated<Spec>` rather than reconstruct semantics?
- Is dtype preserved?
- Are all shape/byte calculations checked?
- Is runtime hardware validation separate from logical proof?
- Are errors structured and actionable?
- Is fallback explicit?
- Is host/device transfer visible?
- Are source/module paths retained?
- Are artifact/cache identities complete?
- Does the test fail if the core semantic field is dropped?
- Does the public API compile in its minimum feature set?
- Is no_std unaffected when compiler features are disabled?
- Are optional dependencies truly optional?
- Does capability advertisement match execution?
- Is the benchmark real and reproducible?
- Are generated docs/matrices synchronized?

---

# Part XVI — Aggressive implementation schedule

## 36. Twelve-effort-week aggressive preview schedule

These are **effort weeks**, not parallel calendar lanes. Execute every role block sequentially in the order written: architecture/review first, implementation second, coverage/documentation third. A single human maintainer should expect roughly 18–30 calendar weeks depending on review latency, hardware availability, and how many baseline defects appear.

### Week 0 — baseline and truth reset

**Opus**

- write ADRs for IR, IREE target, artifact v2, dynamic symbols, PyTorch strategy;
- add CMP2/IREE/TORCH task rows;
- freeze dependency graph;
- create support registry schema;
- add no-stub completion policy.

**GPT-5.6**

- build current branch on supported feature sets;
- record exact baseline tests/benchmarks;
- add compiler end-to-end test harness that initially fails;
- add semantic mutant tests exposing current fusion/guard/dtype problems.

**Exit:** repository truthfully reports current compiled path as preview scaffolding; failing end-to-end tests are tracked, not hidden.

### Weeks 1–2 — canonical IR and capture

**Opus:** CMP2-001 design/review, graph verifier, stable hash.  
**GPT-5.6:** descriptor conversions, value metadata, capture session, pytree support, tests.

**Exit:** MLP captures without allocation; all shapes/dtypes/attributes/parameters are preserved; IR verifies and snapshots deterministically.

### Week 3 — reference executable and public API skeleton

**Opus:** runtime value and witness contract.  
**GPT-5.6:** interpreter, guard generation, public `model.compile` with `Reference` target.

**Exit:** same MLP and CNN run eager and compiled-reference with parity.

### Week 4 — IREE MLIR emitter

**Opus:** StableHLO mapping rules and diagnostics.  
**GPT-5.6:** emitter for pointwise/shape/reduction/matmul; golden tests.

**Exit:** emitted modules compile with pinned IREE for LLVM CPU.

### Week 5 — IREE runtime and static inference

**Opus:** runtime ABI and artifact variant contract.  
**GPT-5.6:** subprocess compiler, VMFB loader, engine, typed invocation.

**Exit:** MLP runs through IREE LLVM CPU from ordinary Incin model.

### Week 6 — artifact v2 and cache

**Opus:** format/compatibility/security review.  
**GPT-5.6:** binary container, cache, CLI inspect/compile, corruption tests.

**Exit:** build-time and runtime compilation produce loadable `.incin` artifacts; cache hits skip compilation.

### Week 7 — CNN/transformer operation coverage and GPU

**Opus:** coverage registry and partition policy.  
**GPT-5.6:** conv/pool/layernorm/embedding/attention decomposition; CUDA target integration.

**Exit:** CNN and transformer block pass CPU/CUDA parity.

### Week 8 — dynamic shapes

**Opus:** symbols/constraints/bucket policy.  
**GPT-5.6:** guard bytecode, bucket cache, variable batch/sequence tests.

**Exit:** one public dynamic sequence example works without unsafe stale specialization.

### Week 9 — state dict and DLPack

**Opus:** ownership/stream safety review.  
**GPT-5.6:** typed state archive, strict report, CPU DLPack, CUDA DLPack if hardware available.

**Exit:** PyTorch tensors can feed Incin artifact without mandatory host copy; weights load with a complete report.

### Week 10 — PyTorch export and backend beta

**Opus:** ATen registry and bridge schema.  
**GPT-5.6:** Python package, canonical ATen-to-Incin lowering, `torch.compile` registration, and pytree ABI.

**Exit:** `torch.compile(model, backend="incin")` runs MLP/CNN/transformer inference.

### Week 11 — hardening, diagnostics, comparison

**Opus:** security/compatibility/mutation review.  
**GPT-5.6:** minifier integration, `cargo incin diff`, benchmark matrix, docs/examples.

**Exit:** reproducible compiler/framework performance report; unsupported-op errors identify source and remediation.

### Week 12 — beta release

- freeze schemas;
- run feature powerset;
- hardware CI;
- publish Rust/Python packages under preview versions;
- publish migration/tutorial/benchmark docs;
- tag beta only if O-1 through O-12 pass or clearly mark missing gates.

---

## 37. Production extension: weeks 13–24

### Weeks 13–15

- expand the canonical ATen-to-Incin IR path from Tier-0 to the release operator set;
- semantic fusion and byte-aware allocator;
- multi-variant artifact selection;
- Vulkan/Metal hardware validation;
- custom-op SDK.

### Weeks 16–18

- AOTAutograd forward/backward;
- native Incin compiled backward;
- saved tensor planner;
- mixed precision/loss scaling;
- training parity.

### Weeks 19–21

- explicit graph partitioning;
- native/IREE region boundaries;
- distributed local compiled regions;
- plan visualization;
- broader ONNX import.

### Weeks 22–24

- release candidate hardening;
- security fuzzing;
- long-run cache/concurrency tests;
- model zoo;
- docs/book;
- compatibility policy;
- release readiness review.

---

# Part XVII — First 30 pull requests in dependency order

## 38. PR sequence

1. `COMP-POLICY-001`: no-stub completion policy and program docs.
2. `COMP-AUDIT-001`: failing semantic tests for current guards/fusion/capture.
3. `CMP2-001A`: ID/type/value/constraint skeleton.
4. `CMP2-001B`: descriptor enum and checked conversions.
5. `CMP2-001C`: verifier and stable hash.
6. `CMP2-002A`: capture session and symbolic storage.
7. `CMP2-002B`: pytree input/output and module scopes.
8. `CMP2-002C`: operation coverage for six existing descriptors.
9. `CMP2-003A`: runtime value table and reference executor.
10. `CMP2-004A`: guard bytecode and evaluator.
11. `CMP2-009A`: public compile/reference run API.
12. `IREE-001`: toolchain discovery and diagnostic command.
13. `IREE-002A`: MLIR module/type emitter.
14. `IREE-002B`: pointwise/reshape/broadcast lowering.
15. `IREE-002C`: reduction/matmul lowering.
16. `IREE-003`: subprocess compiler and source-map diagnostics.
17. `IREE-004A`: runtime engine and LLVM CPU invocation.
18. `CMP2-008A`: artifact v2 parser/writer.
19. `CMP2-008B`: parameter archive and executable variants.
20. `CACHE-001`: compiler cache/single-flight.
21. `CLI-COMP-001`: compile/inspect/explain commands.
22. `IREE-LOWER-001`: conv/pool.
23. `IREE-LOWER-002`: norm/embedding/gather.
24. `IREE-LOWER-003`: attention/decompositions.
25. `IREE-CUDA-001`: CUDA target/runtime hardware slice.
26. `DYN-001`: symbolic guards and dynamic policy.
27. `DYN-002`: shape buckets/recompile cache.
28. `TORCH-001`: typed state archive/load report.
29. `DLPACK-001`: CPU protocol and ownership.
30. `TORCH-BACKEND-001`: Python package and canonical-IR inference backend.

Do not parallelize PRs that modify the same core contract before the previous contract merges.

---

# Part XVIII — Detailed acceptance models

## 39. Rust model suite

### 39.1 MLP

Proves:

- module capture;
- parameters;
- matmul/add/gelu;
- static/dynamic batch;
- artifact execution;
- state dict.

### 39.2 CNN

Proves:

- conv/pool/flatten;
- layout;
- weight parameters;
- CPU/CUDA;
- ONNX comparison.

### 39.3 Transformer block

Proves:

- layer norm;
- linear projections;
- reshape/transpose;
- batched matmul;
- softmax/mask;
- dynamic batch/sequence;
- attention decomposition.

### 39.4 Custom model

A user-defined module with one custom operation proves SDK and fallback diagnostics.

### 39.5 Stateful model

Batch norm or recurrent state proves explicit state/effect handling. It may be rejected in the first pure-graph milestone, but the error must be correct.

---

## 40. PyTorch suite

Use small models first, then standard models:

1. `nn.Linear` chain;
2. Conv/BatchNorm/ReLU;
3. Multihead attention or equivalent decomposed block;
4. ResNet18;
5. ViT/tiny transformer;
6. dynamic sequence model;
7. custom autograd function rejection or registration;
8. training MLP through AOTAutograd.

For every model:

- eager PyTorch output;
- exported program output;
- Incin imported/reference output;
- Incin IREE output;
- dtype tolerance;
- dynamic input verification;
- parameter checksum and key report;
- graph-break report;
- compile/load/warm metrics.

---

# Part XIX — Risks and mitigations

## 41. Risk register

### R-1: rebuilding too much of IREE

**Mitigation:** use StableHLO + `iree-compile`; postpone native compiler sophistication.

### R-2: two sources of graph semantics

**Mitigation:** canonical descriptors are produced from `Validated<Spec>`; existing generic `Graph` is not executable authority.

### R-3: current ledger masks scaffolding

**Mitigation:** CMP2 hardening track and no-stub completion contract.

### R-4: Rust/IREE binding instability

**Mitigation:** subprocess compiler first; official C API runtime/compiler interfaces; isolate crate; pin exact version.

### R-5: PyTorch frontend scope explosion

**Mitigation:** use `torch.export`, decompositions, AOTAutograd core ATen set, and a small, explicit decomposition policy.

### R-6: dynamic-shape complexity

**Mitigation:** bounded symbol/affine vocabulary; explicit policy; generic path; no general theorem prover.

### R-7: DLPack lifetime/stream bugs

**Mitigation:** RAII owner, event ordering, hardware tests, sanitizers, explicit copy fallback.

### R-8: artifact incompatibility/security

**Mitigation:** section bounds, robust digests, schema/ABI IDs, fuzzing, no untrusted pickle.

### R-9: performance claims without evidence

**Mitigation:** identical IREE build/flags and real benchmark metadata; simulated timing prohibited.

### R-10: operation coverage long tail

**Mitigation:** generated support registry, decompositions, explicit unsupported diagnostics, staged tiers.

### R-11: native/IREE fallback hides transfers

**Mitigation:** whole-graph default; explicit partition policy; report every boundary copy/sync.

### R-12: model API becomes generic-type-heavy

**Mitigation:** retain typed core, add builders/aliases/macros, improve diagnostic translator, do not expose IR types to ordinary users.

### R-13: Python package and Rust crate release skew

**Mitigation:** protocol/schema version, compatibility matrix, exact artifact producer metadata, coordinated release automation.

### R-14: distributed work distracts compiler beta

**Mitigation:** freeze new distributed features during weeks 1–8 except compatibility fixes; integrate compiled local regions later.

---

# Part XX — Documentation and positioning

## 42. Documentation set

Required before beta:

1. **Why Incin compiled execution**—same model eager and compiled.
2. **Five-minute compile tutorial**—no build script.
3. **Build-time AOT tutorial**—for conventional embedded/offline deployment.
4. **Dynamic shapes tutorial**—named batch/sequence.
5. **PyTorch import tutorial**.
6. **`torch.compile` tutorial**.
7. **Custom module and custom op tutorial**.
8. **Artifact deployment tutorial**.
9. **Compiler diagnostics guide**.
10. **Fair compiler and framework comparison methodology and results**.
11. **Support matrix generated from registry**.
12. **Known limitations**—explicit and current.

## 43. README comparison language

Use factual language:

> Incin provides static AOT IREE compilation from ordinary Incin modules, plus eager execution, named and dynamic shapes, state dictionaries, PyTorch import and `torch.compile` integration, autograd, testing utilities, native backends, and distributed planning. Build-script graph generation remains available as an optional deployment workflow rather than a separate authoring model.

Avoid attacking another project. The strongest comparison is a working example and a transparent matrix.

## 44. Demonstration repository

Create `examples/compiled-showcase` containing:

- one Rust model source;
- eager run;
- compiled reference run;
- IREE CPU/CUDA run;
- dynamic batch/sequence;
- safetensors load;
- PyTorch parity script;
- `torch.compile` use;
- artifact inspect output;
- benchmark command;
- custom op example.

A reviewer should be able to see the advantage without reading architecture prose.

---

# Part XXI — Exact definition of done

## 45. Program-level done

The program is complete when a new user can:

1. define or import a model;
2. run and debug it eagerly;
3. inspect its modules/parameters/shapes;
4. write normal Rust tests;
5. compile the same model to IREE;
6. choose static, guarded, bucketed, or generic dynamic behavior;
7. package CPU/GPU variants;
8. run the artifact with typed validation;
9. feed/receive PyTorch tensors without mandatory copies where supported;
10. invoke Incin as a `torch.compile` backend;
11. compare eager/PyTorch/compiled outputs;
12. get actionable errors for unsupported paths;
13. reproduce benchmark and compiler decisions;
14. use custom modules and eventually custom ops;
15. retain native/distributed options.

At that point Incin provides a coherent end-to-end workflow rather than requiring users to assemble separate tensor, compiler, model, and interoperability tools.

---




# Part XXII — Repository hardening, ecosystem, and release issue packets

This part supplements the compiler packets with correctness, API, format, documentation, and release work. Each packet is intentionally repetitive. Weaker agents should follow the order literally and must not infer missing steps.

## 46. Universal issue/PR protocol

For every task below:

1. The human maintainer or coordinator creates a GitHub issue using the exact title.
2. Copy the task’s objective, invariants, steps, tests, and acceptance criteria into the issue.
3. Assign one agent and one reviewer role. Only one implementation agent works at a time.
4. Create branch `issue/<number>-<short-slug>` from the latest green `develop`.
5. Before editing, run the baseline commands relevant to the crate and paste results in the issue.
6. Keep the PR limited to the issue. A prerequisite defect becomes a new issue unless it is a tiny local correction required by the same invariant.
7. PR title: `fix|feat|refactor(scope): description (closes #N)`.
8. PR body links `Closes #N`, lists public API changes, files changed, tests, benchmarks, safety impact, and documentation changes.
9. No `[x]` ledger update until every acceptance item and command has evidence.
10. Merge only after the coordinator independently reads the diff and runs or validates the reported commands.

## 47. GOV2-001 — Establish a reproducible truth baseline

**Suggested owner:** Gemini Pro 1  
**Reviewer:** Opus 5  
**Issue title:** `GOV2-001: capture the develop baseline and reconcile build evidence`

### Objective

Produce the first trustworthy machine-generated snapshot of repository health. This issue changes no product behavior unless a command cannot run because of a trivial script defect.

### Steps

1. Record host OS, kernel, architecture, CPU, memory, Rust toolchain, Cargo version, linker, Python version, GPU devices/drivers, and environment variables affecting builds.
2. Confirm `git status --short` is empty and record `git rev-parse HEAD`.
3. Install or select the repository’s intended Rust toolchain. Do not silently use a newer nightly to bypass stable failures.
4. Run formatting, Clippy, tests, no-default builds, feature-specific checks, rustdoc, ledger, shape audit, and budget commands.
5. For each optional hardware suite, distinguish `passed`, `failed`, and `not executed due to unavailable hardware`. Never record unavailable as pass.
6. Record duration and peak memory for workspace check/test/doc so later regressions are measurable.
7. Generate a machine-readable JSON baseline and human-readable Markdown report under `docs/plan/baselines/`.
8. Open separate issues for every failure. Classify severity and whether it blocks 0.1.
9. Add a CI check that validates the baseline schema, not the machine-specific numbers.
10. Update `PROPOSALS.md` only with observed evidence and links.

### Required commands

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test --workspace --no-default-features
cargo doc --workspace --all-features --no-deps
cargo xtask ledger
cargo xtask budgets
cargo xtask shape-audit
```

Feature subsets must be run separately because `--all-features` can mask invalid combinations.

### Acceptance criteria

- Every command has exit status, stdout/stderr artifact, duration, and environment metadata.
- Hardware suites state actual test count.
- No ledger status is changed merely to make validation pass.
- The report can be reproduced from a documented single command.

### Prohibited shortcuts

- deleting failing tests;
- relaxing warnings globally;
- marking unsupported hardware tests ignored without an issue;
- editing baseline numbers manually;
- using `|| true` in evidence scripts.

## 48. GOV2-002 — Reconcile ledger claims with implemented semantics

**Suggested owner:** Opus 5  
**Reviewer:** human maintainer  
**Issue title:** `GOV2-002: make the implementation ledger evidence-based and non-misleading`

### Objective

Separate “foundation landed,” “preview behavior exists,” and “production feature complete.” The compiler rows in particular must not be interpreted as an executable compiler until the hardening track lands.

### Steps

1. Parse every ledger row and map it to task document, source files, tests, and evidence output.
2. Add explicit maturity columns or a generated companion index: `scaffold`, `vertical-slice`, `preview`, `release-candidate`, `stable`.
3. For each completed row, state exactly what is and is not implemented.
4. Reclassify the existing compiler rows as foundation/scaffold while preserving their historical completion evidence.
5. Add the `CMP2`, `STATE2`, `PY`, `HF2`, `SAFE2`, `DOC2`, and `REL2` tracks from this document.
6. Detect missing task documents and status disagreement automatically.
7. Make the checker fail when a completed row lacks command evidence or its task checklist is still entirely unchecked.
8. Add tests against deliberately inconsistent fixture ledgers.
9. Generate a concise public roadmap that does not expose internal checkboxes as product claims.
10. Document how agents update the ledger in linked PRs.

### Acceptance criteria

- Running the checker surfaces the current compiler/task-document inconsistency.
- Historical work is not erased; maturity is clarified.
- README feature claims link to stable/preview capability data generated from evidence.

## 49. API2-001 — Inventory and minimize the 0.1 public API

**Suggested owner:** Opus 5  
**Reviewer:** GPT-5.6  
**Issue title:** `API2-001: define the Incin 0.1 public compatibility surface`

### Objective

Decide what users can rely on after 0.1 and hide everything else before the freeze.

### Steps

1. Generate rustdoc JSON/public API listing for every workspace crate.
2. Identify which items are re-exported by `incin` or documented for direct use.
3. Classify each as `stable-0.1`, `preview`, `internal-but-public-by-accident`, or `remove`.
4. Move accidental items behind private modules or `#[doc(hidden)]` only when semantically appropriate; `doc(hidden)` alone does not remove SemVer exposure, so prefer actual privacy.
5. Seal traits not intended for external implementation.
6. For extension traits intended for users, write conformance tests and evolution strategy.
7. Make extensible enums non-exhaustive before freeze.
8. Replace public struct fields with constructors/accessors where future invariants are expected.
9. Add `cargo-semver-checks` against an API baseline artifact.
10. Add an `API_STABILITY.md` describing guarantees, exceptions, deprecation window, and preview features.
11. Review all prelude exports; a prelude is a permanent ergonomic and naming commitment.
12. Run a downstream fixture crate that uses only stable APIs.

### Acceptance criteria

- A checked-in machine-readable allowlist names the stable public surface.
- CI fails on an unapproved removal/signature change.
- Every preview module is clearly labeled and excluded from the non-breaking promise.
- The downstream fixture builds on MSRV and current stable.

## 50. SAFE2-001 — Panic, unwrap, and unsafe triage

**Suggested owner:** Gemini Pro 2  
**Reviewer:** GPT-5.6  
**Issue title:** `SAFE2-001: classify and harden panic, unwrap, expect, and unsafe sites`

### Objective

Turn a raw lexical inventory into enforced safety policy without replacing clear invariant checks with unreadable error plumbing.

### Steps

1. Build an `xtask safety-audit` parser using `syn` rather than grep.
2. Record file, function, cfg context, macro origin where possible, and category for every site.
3. Exclude tests from the production release gate but keep their inventory.
4. Start with user-controlled parsers/loaders, state loading, device discovery, FFI setup, data loading, artifact loading, and network bootstrap.
5. Replace Class A unwrap/panic sites with structured errors carrying path, operation, expected/got, and source error.
6. For Class B invariants, attempt type-level elimination first. Otherwise add a named helper such as `validated_index` with one documented invariant boundary.
7. Add `#![deny(unsafe_op_in_unsafe_fn)]` at workspace crates that contain unsafe.
8. Add `// SAFETY:` immediately before every unsafe block; the comment must state pointer provenance, alignment, length, aliasing, initialization, lifetime, thread, and device-stream assumptions as applicable.
9. Add Miri tests for CPU memory/view code and fuzz targets for parsers/artifacts.
10. Add ASan/UBSan jobs for supported Linux targets and CUDA compute-sanitizer jobs where hardware permits.
11. Add a budget that allows no new unclassified production sites.
12. Do not require zero unsafe; require a shrinking, reviewed boundary.

### Semantic mutants

- corrupt shape produces length overflow;
- misaligned SIMD input takes safe fallback;
- foreign CUDA pointer/device is rejected;
- poisoned data-worker channel returns an error;
- malformed artifact length does not allocate attacker-controlled memory;
- unsupported dtype never reaches an unreachable branch.

### Acceptance criteria

- every production site is classified;
- no Class A site remains;
- every unsafe block has a safety comment and targeted test reference;
- CI prevents unclassified growth.

## 51. STATE2-001 — Introduce an exact typed state value

**Suggested owner:** GPT-5.6  
**Reviewer:** Opus 5  
**Issue title:** `STATE2-001: represent state dictionary tensors without float reinterpretation`

### Objective

Create a state value that preserves exact dtype, shape, layout, device/source-device metadata, byte order, and parameter/buffer identity without pretending every value is `B::FloatElem`.

### Proposed core types

```rust
pub struct StateTensor<B: StorageBackend> {
    name: StatePath,
    dtype: DTypeId,
    shape: ShapeBuf,
    layout: LayoutDescriptor,
    storage: ErasedStorage<B>,
    role: StateRole,
    alias_group: Option<AliasGroupId>,
}

#[non_exhaustive]
pub enum StateRole {
    Parameter { requires_grad: bool },
    Buffer { persistent: bool },
    Optimizer,
    Auxiliary,
}
```

The exact storage erasure design must use existing backend handles and witnessed construction rather than adding a second storage hierarchy.

### Steps

1. Audit `RawVar`, `Param`, `StateDict`, safetensors conversion, checkpoint code, and optimizer state.
2. Write an ADR selecting storage erasure and ownership semantics.
3. Add `StatePath` with validated components and canonical dot rendering.
4. Add exact dtype and checked byte-length invariants.
5. Add alias-group representation for tied/shared parameters.
6. Implement fallible typed extraction (`try_into_tensor<K,S>()`) that verifies all metadata.
7. Implement construction from a typed tensor without copying when ownership/layout permit.
8. Migrate state export to return `Result<StateArchive<B>>`.
9. Keep compatibility adapters private until the new path is complete.
10. Add round-trip tests for every supported dtype, scalar/rank-N, non-contiguous rejection/materialization policy, and shared parameters.

### Prohibited shortcuts

- storing all data as f32;
- silently casting integer/bool tensors;
- using a `Vec<u8>` without dtype/endianness/shape validation;
- dropping aliases;
- making export infallible by omitting values.

## 52. STATE2-002 — Strict, atomic, PyTorch-familiar state loading

**Suggested owner:** Gemini Pro 1  
**Reviewer:** GPT-5.6  
**Issue title:** `STATE2-002: add atomic strict state loading and detailed reports`

### Public behavior

```rust
let report = model.load_state_dict(
    archive,
    LoadStateOptions::strict()
        .map_keys(mapping)
        .target(device.clone()),
)?;
```

```rust
pub struct LoadStateReport {
    pub loaded: Vec<StatePath>,
    pub missing: Vec<StatePath>,
    pub unexpected: Vec<StatePath>,
    pub shape_mismatches: Vec<ShapeMismatch>,
    pub dtype_mismatches: Vec<DTypeMismatch>,
    pub alias_conflicts: Vec<AliasConflict>,
}
```

### Steps

1. Traverse module parameters and buffers into an expected manifest.
2. Apply deterministic key mapping before matching.
3. Build a complete load plan without mutating the model.
4. Validate missing/unexpected keys, exact shapes, allowed dtype conversions, device policy, aliases, and parameter/buffer role.
5. In strict mode, return the complete report as an error if any incompatibility exists.
6. In non-strict mode, load only explicitly allowed differences and return the report.
7. Allocate/transfer all replacements before committing.
8. Commit atomically; on any failure, leave the model byte-for-byte unchanged.
9. Preserve static type metadata and verify storage after commit.
10. Test rollback by injecting failure after several staged allocations.
11. Test tied weights: one source alias group maps to one destination sharing relationship.
12. Add PyTorch terminology comparison to docs while documenting intentional stricter behavior.

## 53. IO2-001 — Correct safetensors read/write and sharded checkpoints

**Suggested owner:** GPT-5.6  
**Reviewer:** Gemini Pro 2  
**Issue title:** `IO2-001: make safetensors dtype-exact, device-aware, sharded, and atomic`

### Steps

1. Route safetensors through `StateArchive`, never `B::FloatElem` storage.
2. Create a complete mapping table between safetensors dtypes and Incin dtypes.
3. Reject unsupported dtypes explicitly; do not remap signedness or bool silently.
4. Validate header length, tensor ranges, overlap, alignment assumptions, shape product, and byte count with checked arithmetic.
5. Add target-device policy: load CPU/mmap then transfer, direct device upload, or retain CPU.
6. Support `model.safetensors.index.json`, shard weight maps, and total-size metadata.
7. Support lazy/memory-mapped loading when storage lifetime can be safely represented.
8. Preserve shared/tied-weight metadata in the Incin manifest even though file slices may not encode aliasing directly.
9. Save to temporary files, fsync as configured, verify, and atomically rename.
10. Generate deterministic shard boundaries and index output.
11. Fuzz the header and range parser.
12. Compare round trips with the reference Python safetensors package across dtypes.

## 54. HF2-001 — Build a secure Hugging Face repository resolver

**Suggested owner:** Gemini Pro 1  
**Reviewer:** Opus 5  
**Issue title:** `HF2-001: resolve complete Hugging Face model packages with revisions and offline guarantees`

### Objective

Resolve a repository into a deterministic local package manifest containing config, graph/architecture information, tokenizer assets, weight shards, license/metadata, and hashes.

### Steps

1. Add `HubReference { repo, revision, subfolder }`; revision must resolve to an immutable commit for reproducibility.
2. Define `DownloadPolicy`: online, local-files-only, refresh, and offline-strict.
3. Resolve common config/tokenizer/generation files and safetensors shard index.
4. Use a content-addressed cache plus a human-readable repo/revision index.
5. Download into temporary files and atomically publish after size/hash verification.
6. Sanitize repository paths; reject traversal and symlink escape.
7. Support authenticated private repositories without logging tokens.
8. Support resumable range requests only when validator/ETag semantics make them safe.
9. Produce `ModelPackageManifest` with exact resolved commit and file hashes.
10. Add deterministic offline replay tests using a local HTTP fixture server.
11. Add cache corruption, interrupted download, stale ETag, missing shard, and token-redaction tests.
12. Expose one high-level resolver through the facade; keep HTTP implementation internal.

## 55. HF2-002 — Compile-time typed and runtime flexible model loading

**Suggested owner:** Opus 5  
**Reviewer:** GPT-5.6  
**Issue title:** `HF2-002: generate typed Rust models from metadata and support runtime model packages`

### Two required modes

#### Static/generated mode

```rust
incin::model_from_hub!(
    pub TinyLlama,
    repo = "org/model",
    revision = "immutable-commit",
    inputs = { tokens: s![Batch, Seq] },
);
```

The macro/build tool fetches only metadata needed to generate types unless weights are explicitly embedded. Generated code records a manifest hash and validates loaded weights.

#### Runtime mode

```rust
let package = ModelPackage::from_hub(reference, policy).await?;
let model = AutoModel::load(package, &context)?;
```

Runtime models use dynamic/erased module interfaces with explicit input contracts and structured validation.

### Steps

1. Define architecture config traits and a registry keyed by safe config fields, not arbitrary code from the repository.
2. Implement one small transformer architecture end to end before generic abstraction.
3. Separate metadata resolution, architecture construction, and weight loading.
4. Generate stable module paths matching checkpoint names.
5. Support tied weights, rotary settings, vocabulary size, hidden dimensions, layer count, head count, and dtype policy.
6. For static generation, turn known dimensions into type-level dimensions and unknown/bounded dimensions into named runtime constraints.
7. Never execute repository Python code or `trust_remote_code` in the Rust process.
8. Provide an explicit Python conversion tool for unsupported custom architectures; its output is a declarative Incin package.
9. Add compile-fail tests for contradictory metadata and runtime tests for missing/incorrect weights.
10. Add one Hugging Face parity example with deterministic logits against PyTorch.

## 56. NPY2-001 — NumPy `.npy`/`.npz` and zero-copy array exchange

**Suggested owner:** Gemini Pro 2  
**Reviewer:** GPT-5.6  
**Issue title:** `NPY2-001: add exact NumPy file and array interoperability`

### Steps

1. Implement `.npy` header parsing for supported versions with checked lengths and shape products.
2. Preserve endian marker, Fortran/C order, dtype, and scalar/rank semantics.
3. Define materialization policy for unsupported endianness/order.
4. Implement `.npz` as a named tensor archive with zip-bomb limits and duplicate-name rejection.
5. Add Python buffer protocol for contiguous CPU tensors.
6. Add DLPack protocol for zero-copy exchange where ownership/layout/device permit.
7. Make copy behavior explicit: `CopyPolicy::{Never,IfNeeded,Always}`.
8. Test ownership after producer deletion, read-only views, non-contiguous arrays, endian conversion, and writeability.
9. Differential-test files against NumPy.
10. Document static Rust conversion (`try_into_tensor<s![...], K>()`) and runtime `Dyn` conversion.

## 57. ONNX2-001 — Harden ONNX import into canonical model/graph contracts

**Suggested owner:** GPT-5.6  
**Reviewer:** Opus 5  
**Issue title:** `ONNX2-001: make ONNX import lossless, opset-aware, and shape-safe`

### Steps

1. Separate ONNX graph import from state-dictionary serialization APIs.
2. Parse and retain model IR version, opset imports by domain, graph/value metadata, initializers, attributes, functions, and external data references.
3. Run or reproduce shape inference without treating inferred dimensions as stronger than the source guarantees.
4. Map symbolic ONNX dimensions to named Incin symbols and retain equality relationships.
5. Build an operator registry keyed by domain/opset/operator with explicit version ranges.
6. Reject unknown attributes and unsupported semantic variants rather than choosing defaults silently.
7. Lower imported nodes to canonical Incin descriptors and verify the graph.
8. Support external tensor data with path sandboxing and hashes.
9. Generate typed Rust modules when ranks/dimensions/contracts are adequate; otherwise return a runtime model with explicit contracts.
10. Differential-test representative ops and models against ONNX Runtime/reference outputs.
11. Add malformed protobuf, cyclic graph, forward reference, duplicate SSA name, external-path traversal, and oversized initializer tests.
12. Publish a generated opset/operator support matrix.

## 58. MOD2-001 — Finalize the stable Module/state/container UX

**Suggested owner:** Opus 5  
**Reviewer:** human maintainer  
**Issue title:** `MOD2-001: finalize the Rust-idiomatic, PyTorch-familiar Module API for 0.1`

### Required familiar concepts

- `Module` and `forward`;
- parameters and named parameters;
- buffers and named buffers;
- children and named modules;
- stable hierarchical state paths;
- `train` and `eval` behavior;
- device/dtype transfer;
- freezing/unfreezing;
- strict state dictionaries;
- sequential and common containers;
- tied/shared parameters.

### Rust-specific principles

- fallible operations return `Result`;
- hidden mutation is minimized;
- execution policy remains explicit;
- statically known shapes remain in types;
- runtime-erased modules are a separate deployment/interoperability interface;
- macros generate ordinary inspectable Rust code and high-quality diagnostics;
- no Python-style stringly typed module surgery in the stable core.

### Steps

1. Inventory every public module trait/macro and all downstream examples.
2. Write the desired API as compile-pass fixtures before implementation.
3. Decide ownership convention for `forward` inputs and document clone/view costs.
4. Add parameter and buffer derive/macro annotations with stable path derivation.
5. Add train/eval propagation and explicit context interaction.
6. Add `Sequential`, `ModuleList`, `ModuleDict` equivalents only where they remain type-safe and ergonomic.
7. Add an erased `DynModule`/`Model` interface for imported runtime graphs without weakening generic native modules.
8. Add tied parameter registration and duplicate traversal protection.
9. Integrate `StateArchive` and strict loader.
10. Add custom derive diagnostics with `trybuild` compile-fail coverage.
11. Migrate all examples and predefined models.
12. Freeze only after the PyTorch comparison guide and three real model fixtures are reviewed.

## 59. DATA2-001 — Make DataLoader cancellation, errors, and determinism reliable

**Suggested owner:** Gemini Pro 1  
**Reviewer:** GPT-5.6  
**Issue title:** `DATA2-001: harden data loading worker lifecycle and deterministic shuffling`

### Steps

1. Define a worker protocol with request, batch, error, cancellation, and shutdown messages.
2. Keep worker join handles and join on loader drop/finish.
3. Convert worker panic into a structured `WorkerPanicked` error with worker ID.
4. Avoid poisoned shared mutex state by using channels/isolated worker state where practical.
5. Implement a truly synchronous path for `num_workers == 0`.
6. Define deterministic seed derivation by global seed, epoch, worker, and sample order.
7. Bound prefetch memory and expose backpressure.
8. Ensure early consumer drop cancels workers and releases files/buffers.
9. Add timeout only as explicit policy, not hidden behavior.
10. Add race/stress tests and Loom tests for the small synchronization core.
11. Validate dataset file headers/magic/counts with checked arithmetic.
12. Document reproducibility guarantees and unsupported nondeterministic transforms.

## 60. DOC2-001 — Build the tested documentation system

**Suggested owner:** Gemini Pro 2  
**Reviewer:** human maintainer  
**Issue title:** `DOC2-001: create the Incin book, API map, and executable example matrix`

### Deliverables

1. API rustdoc coverage gate for stabilized public items.
2. `mdBook` with:
   - installation and device setup;
   - tensors, static/dynamic/named shapes;
   - model definition;
   - training and evaluation;
   - saving/loading/state dictionaries;
   - Hugging Face, ONNX, safetensors, NumPy, and PyTorch bridges;
   - compiled execution and troubleshooting;
   - custom modules and custom operations;
   - performance methodology;
   - deployment;
   - safety model and error handling.
3. “PyTorch concept → Incin concept” pages with side-by-side executable examples.
4. Student path explaining tensors, gradients, layers, optimizers, and shape errors.
5. Rust ML researcher path emphasizing custom models, experiments, instrumentation, and reproducibility.
6. Generated capability and model-format support tables.
7. CI that extracts/runs every code block or references compiled example crates.
8. Version selector and migration guide per release.

### Acceptance criteria

- no important public API is documented only by generated signatures;
- every major workflow has a copy-paste example that CI executes;
- feature-gated examples state exact features and hardware;
- documentation never claims unexecuted hardware support;
- comparison pages explain intentional differences, not just matching names.

## 61. REL2-001 — Prepare and freeze the 0.1 release candidate

**Suggested owner:** human maintainer with Opus 5 checklist review  
**Issue title:** `REL2-001: cut the stability-targeted Incin 0.1 release candidate`

### Steps

1. Resolve every open critical/high blocker in the 0.1 milestone.
2. Regenerate public API baseline and review every stable item.
3. Confirm all preview APIs are visibly marked and outside the promise.
4. Pin MSRV and test MSRV/current stable/beta; nightly is advisory unless a nightly feature is enabled.
5. Run feature powerset or a curated complete matrix that catches mutually exclusive and additive features.
6. Run CPU and available accelerator model suites on clean machines.
7. Run parser/artifact fuzz corpora and sanitizer/Miri jobs for the agreed duration.
8. Build crates, docs, CLI binaries, Python wheels if included, and example applications from release artifacts rather than workspace paths.
9. Audit licenses, notices, dependencies, advisories, and bundled external tools.
10. Publish RC artifacts and a migration guide; run a minimum two-week soak.
11. Require one external/downstream fixture project to update from the previous preview using only documented APIs.
12. Repeat for RC2 after fixes. Tag 0.1 only with no unresolved severity-1/2 defects.

### 0.1 freeze rule after merge

Any later PR that changes the selected public API must include:

- SemVer report;
- compatibility rationale;
- deprecation/migration path if behavior changes;
- tests against the 0.1 downstream fixture;
- maintainer approval.

## 62. Post-0.1 path to 1.0

### 0.2–0.3: compiler and interoperability preview

- complete canonical compiled execution and external compiler target;
- release Python CPU/CUDA tensor bridge and NumPy/safetensors package;
- import `torch.export` inference graphs;
- harden model package/Hugging Face loader;
- publish benchmark dashboard with no broad performance claims.

### 0.4–0.6: model ecosystem and performance

- `torch.compile` backend preview;
- broader ATen/ONNX operator coverage;
- optimized CPU/CUDA structured operations and real autotuning;
- predefined transformer/vision models;
- dynamic shape buckets and artifact deployment;
- mature trainer/checkpoint/AMP path.

### 0.7–0.9: production hardening

- stable Python compatibility matrix;
- compiled backward/training preview with AOTAutograd bridge;
- broad hardware CI and long-running stress tests;
- complete book/cookbook;
- deployment/runtime packaging;
- API deprecation cleanup that remains compatible with 0.1.

### 1.0 gate

- supported model suite reaches competitive measured performance on declared Tier-1 hardware;
- stable APIs have survived real downstream use;
- no critical format/state/compiler correctness gaps;
- Python/Rust interchange is safe and documented;
- release/support/security processes are routine;
- distributed features may remain explicitly preview.

# Part XXIII — Security Audit, Exploitability Analysis, and Mandatory Remediation

**Security audit basis:** static review of the supplied `develop` branch at commit
`eb3633525ea74e56f7a6b2d5c5b57dc74a5d9b8d`.

**Interpretation rule:** this section distinguishes four categories:

- **confirmed unsoundness:** safe Incin code can reach undefined behavior or violate a
  type invariant without the caller writing `unsafe`;
- **confirmed security defect:** the source contains a directly identifiable trust-boundary
  failure such as path traversal, code injection, unauthenticated network control, or
  unbounded allocation;
- **likely exploit path:** exploitation depends on deployment conditions, permissions,
  network reachability, a malicious model/cache/checkpoint, or a later executable consumer;
- **hardening requirement:** no complete exploit is proven statically, but the design is
  unsuitable for accepting untrusted inputs or for a stable release.

This is not a claim that every item has been weaponized. It is a release-blocking source
audit. Dynamic validation is still required with Miri, sanitizers, fuzzers, dependency
scanners, hardware tests, and adversarial integration tests.

---

## 63. Security executive summary

### 63.1 Immediate conclusion

The audited revision must **not** be published as a security-stable `0.1.0` while the
critical items below remain reachable through safe public APIs or ordinary model-loading
workflows.

The most serious findings are:

1. safe tensor extraction can construct arbitrary invalid Rust values and perform an
   alignment-invalid typed copy;
2. the external Candle backend reinterprets arbitrary byte slices as aligned `f32` slices;
3. checkpoint loading can replace statically typed parameter storage without validating
   the static shape/dtype/device contract;
4. the ONNX procedural-macro sidecar cache stores and reparses Rust token strings, allowing
   a modified `.incin_meta` file to inject code into generated model source;
5. model, cache, dataset, and artifact parsers allocate or recurse from untrusted lengths
   without one shared resource-limit policy;
6. file download, telemetry, and cache-clearing paths contain traversal, symlink, partial
   file, or arbitrary deletion hazards;
7. distributed rendezvous and NCCL bootstrap use plaintext, unauthenticated TCP;
8. compiled artifacts use unbounded JSON plus Adler-32 and do not validate the declared
   magic or the plan's executable semantics;
9. representation-bearing dtype traits are safe and externally implementable even though
   downstream code treats their elements as plain initialized bytes.

### 63.2 Security-adjusted readiness

The earlier readiness estimates in this document were made before the dedicated security
pass. Until the critical and high user-controlled-input findings are closed with dynamic
evidence, use these conservative security-adjusted estimates:

| Milestone | Previous estimate | Security-adjusted estimate | Why |
|---|---:|---:|---|
| stability-targeted `0.1.0` | 47% | **about 39%** | Public safe APIs contain memory-unsound behavior; model/checkpoint/cache boundaries are not ready for untrusted input; no security CI exists. |
| requested single-device `1.0.0` | 35% | **about 31%** | The same defects affect interoperability, deployment artifacts, Python transition paths, and model-hub loading. |

These are planning estimates, not mathematical measurements. Restoring the lost credit
requires closing findings, adding permanent gates, and producing real tool output—not
merely documenting the risks.

### 63.3 Mandatory release rule

`0.1.0` is blocked unless all of the following are true:

- every **Critical** item in this section is fixed;
- every **High** item reachable from a local untrusted model, checkpoint, archive, cache,
  dataset, environment variable, or public safe API is fixed;
- multi-host networking is either authenticated or explicitly disabled/loopback-only by
  default and labelled insecure preview;
- all unsafe blocks reachable in normal library use have documented invariants and Miri or
  equivalent tests where Miri can exercise them;
- all public parsers and deserializers enforce explicit byte, count, rank, dimension,
  nesting, and allocation limits;
- malformed input returns a structured error and never panics, aborts, overflows, hangs, or
  performs a partial state mutation;
- `cargo audit`, `cargo deny`, action pinning checks, secret scanning, and security-focused
  tests run in CI;
- a `SECURITY.md` defines supported versions and a private reporting path;
- the security issue ledger matches the actual source and test evidence.

---

## 64. Threat model and trust boundaries

### 64.1 Assets Incin must protect

Incin may run inside developer workstations, training servers, model-serving processes,
CI builders, and multi-GPU workers. Security-sensitive assets include:

- process memory and Rust memory safety;
- model weights, gradients, prompts, training data, and telemetry;
- filesystem integrity under the current user account;
- build-host integrity when procedural macros or `build.rs` run;
- GPU memory and device stability;
- compiler and artifact cache integrity;
- distributed communicator identities and control messages;
- release credentials and repository integrity in CI;
- the credibility of static tensor guarantees.

### 64.2 Inputs that must be considered untrusted by default

The following must be treated as hostile unless a stronger trust policy is explicitly
selected:

- ONNX, safetensors, NumPy, GGUF, `.incin`, checkpoint, optimizer-state, and model-package
  bytes;
- Hugging Face repository contents, metadata, names, revisions, symlinks, and external-data
  references;
- `.incin_meta` sidecars;
- downloaded datasets and compressed archives;
- cache files and lock files in user-writable directories;
- environment variables controlling paths, devices, caches, rendezvous, and toolchains;
- Python/PyTorch checkpoint files, especially pickle-backed `.pt`/`.pth`;
- telemetry run names and socket clients;
- peer messages on any network socket;
- compiled artifacts produced elsewhere;
- external backend implementations and custom dtype implementations;
- dimensions, offsets, strides, counts, names, operation attributes, and shape constraints
  originating in imported formats.

### 64.3 Trust-policy API required for all model sources

Do not scatter booleans such as `trusted: true` through loaders. Introduce one explicit,
auditable policy:

```rust
#[non_exhaustive]
pub enum TrustPolicy {
    /// Local bytes explicitly trusted by the caller. Resource limits still apply.
    LocalTrusted,

    /// Bytes must match a caller-supplied cryptographic digest.
    VerifyDigest {
        algorithm: DigestAlgorithm,
        expected: Digest,
    },

    /// Package manifest and every referenced blob must verify under a trusted key.
    VerifySignature {
        keyring: Arc<dyn ModelKeyring>,
    },

    /// Treat all metadata and payloads as hostile. No executable serialization,
    /// no external paths outside the package, strict limits, and isolated conversion.
    Untrusted,
}
```

`LocalTrusted` must not disable memory-safety checks, integer overflow checks, shape
validation, or parser limits. It only changes authenticity/provenance decisions.

### 64.4 Security invariant hierarchy

Every implementation must preserve this order:

1. **bytes are bounded and framed;**
2. **bytes are parsed without panic or unchecked allocation;**
3. **parsed metadata is semantically validated;**
4. **physical storage is validated against metadata;**
5. **typed witnesses are created only after validation;**
6. **execution consumes only validated descriptors and storage;**
7. **artifacts, caches, and distributed messages are authenticated where authenticity
   matters.**

A checksum, a Rust type name, or a file extension cannot skip an earlier layer.

---

## 65. Finding index

| ID | Severity | Confidence | Area | Release effect |
|---|---|---|---|---|
| SEC-001 | **Critical** | Confirmed source-level unsoundness | `Tensor::to_scalar` / `to_vec1` | Blocks any release |
| SEC-002 | **Critical** | Confirmed source-level unsoundness | external Candle byte decoding | Blocks Candle feature and 0.1 while public |
| SEC-003 | **Critical** | Confirmed invariant violation; memory impact backend-dependent | parameter/buffer state loading | Blocks checkpoint/model loading |
| SEC-004 | **Critical** | Confirmed code-injection design | ONNX `.incin_meta` cache | Blocks compile-time ONNX import |
| SEC-005 | **Critical** | Confirmed unsafe-contract gap | public `DType` / `ConstDType` representation | Blocks stable dtype API |
| SEC-006 | **High** | Confirmed path traversal/symlink/partial-cache defect | dataset downloader | Blocks untrusted downloads |
| SEC-007 | **High** | Confirmed resource-exhaustion and malformed-input defects | model/dataset parsers | Blocks model hub and file inspection |
| SEC-008 | **High** | Confirmed arbitrary path/delete hazards | telemetry run IDs and tune clear | Blocks affected CLI/telemetry stabilization |
| SEC-009 | **High** | Confirmed unauthenticated plaintext protocol | distributed rendezvous/NCCL bootstrap | Blocks production network claims |
| SEC-010 | **High** | Confirmed weak framing/integrity; future execution amplification | compiled artifacts | Blocks portable executable artifacts |
| SEC-011 | **High** | Confirmed unchecked arithmetic/panic surface | backend byte/numel construction | Blocks hostile metadata execution |
| SEC-012 | **Medium–High** | Confirmed pre-limit whole-file read | tuning and other caches | Must fix before cache stabilization |
| SEC-013 | **Medium** | Deployment-dependent confidentiality/DoS | telemetry local socket | Fix or document before telemetry stable |
| SEC-014 | **High process risk** | Confirmed missing controls | CI and supply chain | Blocks release signing/publishing |
| SEC-015 | **High interop risk** | Known ecosystem hazard | PyTorch pickle and NumPy object arrays | Blocks “seamless” unsafe import claims |
| SEC-016 | **High future risk** | Design requirement | HF/ONNX/NPZ external paths and archives | Must land before first-class packages |
| SEC-017 | **Medium** | Documentation correctness | file transport durability claim | Correct before stable docs |
| SEC-018 | **Program-wide** | Audit incompleteness | unsafe/panic/dependency inventory | Permanent security workstream |

---

## 66. SEC-001 — Safe tensor extraction can create invalid Rust values

**Severity:** Critical  
**CWE mapping:** CWE-843 (type confusion), CWE-704 (incorrect type conversion), memory
safety/undefined behavior  
**Primary source:** `crates/incin-core/src/tensor/ops/manipulation.rs:360-457`

### 66.1 Evidence

The public safe methods are generic over any `E: Copy + 'static`:

```rust
pub fn to_scalar<E: Copy + 'static>(&self) -> Result<E>
pub fn to_vec1<E: Copy + 'static>(&self) -> Result<Vec<E>>
```

The scalar path checks only byte length and dtype element width, then executes:

```rust
core::ptr::read_unaligned(bytes.as_ptr() as *const E)
```

The vector path casts `Vec<u8>::as_ptr()` to `*const E` and executes
`copy_nonoverlapping` into an uninitialized `Vec<E>`.

### 66.2 Why this is unsound

Size equality does not prove that arbitrary bytes are a valid `E`.

Safe callers can choose types with invalid bit-pattern requirements, including references,
`char`, `NonZero*`, function pointers, enums with restricted discriminants, and user types
whose invariants are not represented by size. Constructing an invalid value is undefined
behavior even before it is dereferenced.

The vector path has a separate alignment defect. A byte allocation is not required to meet
the alignment of arbitrary `E`; `copy_nonoverlapping<T>` requires properly aligned source
and destination pointers for `T`, even when the bytes came from a valid numeric payload.

The current `bool` special case avoids arbitrary boolean bit patterns but does not make the
general generic API sound.

### 66.3 Safe replacement design

Remove arbitrary representation inference from the API.

Preferred stable API:

```rust
pub fn to_scalar(&self) -> Result<K::Elem>
where
    K: ConstDType,
    K::Elem: TensorElement;

pub fn to_vec1(&self) -> Result<Vec<K::Elem>>
where
    K: ConstDType,
    K::Elem: TensorElement;

pub fn scalar_cast<T>(&self) -> Result<T>
where
    K: ConstDType,
    K::Elem: TensorElement,
    T: CheckedNumericCast;
```

`scalar_cast` performs numeric conversion, not bit reinterpretation.

For dynamic dtypes:

```rust
pub enum ScalarValue {
    U8(u8),
    U32(u32),
    I64(i64),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    Q8Block(Q8BlockValue),
}

pub fn to_scalar_value(&self) -> Result<ScalarValue>;
```

### 66.4 Implementation steps

1. Open a security issue titled:
   **`security: remove arbitrary-value construction from tensor extraction`**.
2. Mark the current generic methods deprecated only if compatibility is temporarily needed;
   because pre-0.1 breakage is allowed, prefer removing/replacing them immediately.
3. Introduce a sealed `TensorElement` for built-in scalar representations. The safe public
   trait must not be implementable for arbitrary external types.
4. Decode each dtype with exact-width chunking:
   - `u8`: direct byte;
   - integers/floats: `from_ne_bytes` or an explicitly documented endian policy;
   - `f16`/`bf16`: decode `u16`, then construct through the half crate;
   - quantized blocks: a dedicated block decoder.
5. Use checked multiplication for `numel * element_size`.
6. Require exact payload length; reject trailing and truncated bytes.
7. Keep `bool` as an explicit checked conversion, not an implicit same-size reinterpret.
8. Search every call site and change type-directed calls to dtype-directed extraction.
9. Add a migration note explaining numeric conversion versus byte reinterpretation.
10. Add an internal unsafe helper only if a measured hot path requires it; its input must
    already be a proven `Pod` type and its safety contract must be local and testable.

### 66.5 Mandatory tests

- compile-fail: extraction cannot request `&'static u8`, `char`, `NonZeroU32`, a function
  pointer, or a custom enum;
- unit: every supported scalar dtype round-trips representative edge values;
- unit: NaN payloads and signed zero are preserved for exact extraction;
- unit: wrong dtype and wrong payload length return errors;
- Miri: scalar and vector extraction for every built-in dtype;
- Miri: deliberately create byte buffers at offsets that would be unaligned for `u32`/`f64`;
  the safe decoder must still work because it decodes chunks rather than dereferencing a
  typed source pointer;
- property test: arbitrary byte strings never panic; they either decode under the exact
  dtype contract or return an error.

### 66.6 Forbidden shortcuts

- do not replace `read_unaligned` with `transmute`;
- do not add `E: Default`;
- do not assume `Copy` means plain-old-data;
- do not rely on `size_of::<E>()`;
- do not make the method `unsafe` and leave it as the ordinary user path;
- do not create `&[E]` from `&[u8]` without an alignment and validity proof.

### 66.7 Completion evidence

```bash
cargo test -p incin-core tensor_extraction
cargo +nightly miri test -p incin-core --test tensor_extraction_miri
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The issue remains open until the old unsound route is absent from the public safe API.

---

## 67. SEC-002 — Candle backend performs alignment-invalid byte reinterpretation

**Severity:** Critical  
**CWE mapping:** CWE-704 / memory alignment and representation violation  
**Primary source:** `crates/incin-backends/src/external/candle/backend.rs:90-130`

### 67.1 Evidence

`from_bytes` performs:

```rust
core::slice::from_raw_parts(
    bytes.as_ptr() as *const f32,
    bytes.len() / size_of::<f32>(),
)
```

A `&[u8]` does not guarantee `align_of::<f32>()`. The function is public through a safe
backend contract. The implementation also truncates non-multiple-of-four lengths and
decodes every dtype as `f32` before casting.

`to_bytes` similarly converts every Candle tensor to `f32`, so integer and `f64` payloads
do not round-trip exact storage semantics.

### 67.2 Impact

- undefined behavior from an unaligned typed slice;
- silent truncation;
- dtype corruption;
- incorrect checkpoint/model conversion;
- false conformance claims for supported dtypes.

### 67.3 Correct implementation

1. Dispatch by exact `DTypeId`.
2. Validate `shape_numel` with checked arithmetic.
3. Compute the exact expected byte count for the requested dtype.
4. Reject any length mismatch before allocation.
5. Decode using safe chunk conversion or a proven POD helper that supports unaligned input.
6. Build the Candle tensor from a correctly typed `Vec<T>`.
7. `to_bytes` must export the tensor's actual logical dtype, not force F32.
8. Add a single conformance helper shared by all external backends:
   `decode_storage_payload(dtype, shape, bytes, limits)`.
9. Ensure unsupported quantized formats return a structured unsupported error.

### 67.4 Adversarial tests

- pass `&buffer[1..]` for a four-byte dtype;
- lengths 0, 1, 3, 5, and `expected ± 1`;
- f64/i64/u8/bf16/f16 round-trips;
- a shape whose checked byte count overflows;
- dynamic dtype mismatch;
- Miri execution of every decoding path;
- external-backend conformance must compare byte-exact values, shape, and dtype.

### 67.5 Merge policy

Disable or hide `external-candle` from release builds if this issue cannot be completed
before the release candidate. Documentation must not describe the adapter as conformant
while it reinterprets all payloads through F32.

---

## 68. SEC-003 — Checkpoint loading can corrupt statically typed parameter invariants

**Severity:** Critical  
**CWE mapping:** CWE-20 (improper input validation), CWE-670 (always-incorrect control flow
implementation), potential memory-safety amplification  
**Primary sources:**

- `crates/incin-core/src/nn/param.rs:105-117`
- `crates/incin-core/src/nn/param.rs:386-400`
- `crates/incin-core/src/nn/param.rs:445-457`
- `crates/incin-core/src/nn/param.rs:625-636`
- generated recursive loading in `crates/incin-macros/src/module.rs`

### 68.1 Evidence

`Param<S, B>::load_state_dict` replaces its raw variable from a `Tensor<Dyn, B>` without
checking that the tensor's physical metadata agrees with static `S`, expected dtype,
device, placement, or parameter alias contract. Missing keys are ignored. Fields are
mutated as recursion proceeds.

`Param::as_tensor` and `Buffer::as_tensor` construct `Tensor<S, ...>` directly from backend
storage, bypassing the checked storage constructor and therefore assuming the invariant
continues to hold.

### 68.2 Exploit scenario

A malicious or corrupted checkpoint supplies an undersized or differently shaped tensor
under a valid parameter name. The loader installs it behind a parameter whose Rust type
still states the original shape. Later operations may use static descriptor geometry,
backend kernels, or generated launch sizes under the assumption that the storage matches
the type. The immediate outcome may be a structured backend error, a panic, a GPU device
fault, or—where a backend trusts the shape—out-of-bounds access.

Even when no memory corruption occurs, partial mutation means a failed load can leave a
model in a mixed old/new state.

### 68.3 Required two-phase atomic loader

```rust
pub struct StateSchema {
    entries: BTreeMap<StatePath, StateContract>,
    aliases: AliasGroups,
}

pub struct StateContract {
    kind: StateKind, // parameter or buffer
    shape: ShapeContract,
    dtype: DTypeId,
    device_policy: DeviceLoadPolicy,
    placement: PlacementContract,
    requires_grad: bool,
}

pub struct ValidatedStateLoad<B> {
    replacements: BTreeMap<StatePath, B::RawVar>,
    report: LoadStateReport,
}
```

Process:

1. enumerate the entire target module schema;
2. normalize source keys without mutating the model;
3. detect duplicate, missing, unexpected, ambiguous, and alias-conflicting keys;
4. validate rank, dimensions, dtype conversion policy, byte count, device and placement;
5. stage converted storage/variables;
6. validate tied/shared parameters preserve alias expectations;
7. commit every replacement only after all validation and allocation succeeds;
8. return a deterministic report.

### 68.4 Stable behavior

- `strict = true` is the default;
- non-strict loading is explicit and returns all differences;
- shape mismatch is never silently reshaped;
- dtype conversion is explicit through a `DTypeLoadPolicy`;
- device movement is explicit through a `DeviceLoadPolicy`;
- no parameter is mutated if any strict validation fails;
- state export is fallible and cannot silently omit a parameter;
- `as_tensor` uses `try_from_storage` or a narrowly scoped construction witness.

### 68.5 Tests

- undersized payload under a correct name;
- same number of elements but wrong shape;
- wrong dtype with conversion disabled;
- wrong device/placement;
- missing and unexpected keys;
- duplicate normalized names;
- tied parameter supplied twice with different bytes;
- allocation failure on the final parameter proves earlier parameters remain unchanged;
- a backend test that records storage metadata before and after failed load;
- static parameter cannot be obtained from malformed dynamic storage;
- safetensors/PyTorch naming conversion feeds the same validated transaction.

---

## 69. SEC-004 — ONNX sidecar cache is a Rust code-injection boundary

**Severity:** Critical  
**CWE mapping:** CWE-94 (code injection), CWE-502 (unsafe deserialization), CWE-829
(inclusion of functionality from untrusted control sphere)  
**Primary source:** `crates/incin-macros/src/onnx.rs:18-27,283-325,403-456,501-568`

### 69.1 Evidence

`OnnxMeta` stores:

```rust
forward_stmts: Vec<String>,
last_output: String,
```

Cache freshness is based on file modification time. Cached strings are deserialized from
`<model>.<ext>.incin_meta`, parsed into `proc_macro2::TokenStream` with `unwrap`, and
inserted into the generated `forward` method.

A sidecar is therefore not metadata. It is cached source code.

### 69.2 Attack path

A repository, archive, model package, dependency, compromised cache, or local attacker
places a newer `.incin_meta` beside an ONNX file. When the downstream crate builds, the
procedural macro accepts the sidecar and emits its token strings into the user's crate.
The injected Rust code executes at runtime and may also use compile-time facilities such as
`include_bytes!`/`env!` to disclose build inputs or alter compilation.

No malformed ONNX graph is needed; the attacker only needs the sidecar to pass JSON
deserialization and the modification-time check.

### 69.3 Immediate containment

1. Remove support for token/source strings in `.incin_meta`.
2. Reject legacy sidecars containing `forward_stmts` or `last_output`.
3. Delete/rebuild them from the source ONNX file.
4. Do not parse arbitrary cached strings as Rust tokens.
5. Replace every `parse().unwrap()` with code generation from typed enums; malformed data
   must yield `compile_error!` with the model path and field context.
6. Bind cache identity to a cryptographic hash of:
   - complete source model bytes;
   - importer schema version;
   - Incin version;
   - enabled importer feature set;
   - relevant code-generation policy.
7. Write caches atomically with restrictive permissions and no symlink following.

### 69.4 Correct cache representation

Cache only data:

```rust
enum CachedOp {
    MatMul { a: ValueId, b: ValueId, output: ValueId },
    Conv2d { input: ValueId, weight: ValueId, bias: Option<ValueId>, attrs: ConvAttrs },
    Relu { input: ValueId, output: ValueId },
    // ...
}

struct OnnxCache {
    schema: u32,
    source_digest: [u8; 32],
    graph: CanonicalImportedGraph,
}
```

Token generation occurs after validating the data-only graph and is entirely controlled by
trusted Rust code.

### 69.5 Better long-term architecture

Procedural macros run inside the compiler process and are a poor place to parse large,
remote, or hostile model files. Prefer:

```text
cargo incin model generate model.onnx
  -> isolated/bounded importer process
  -> canonical signed/hash-bound manifest
  -> generated Rust source + state package
  -> normal compiler consumes generated source
```

The macro may remain as a convenience for small local trusted files, but the stable
model-hub path should use an explicit generation command that can apply CPU, memory, time,
and file-access limits.

### 69.6 Tests

- malicious legacy sidecar with syntactically valid Rust tokens is rejected;
- source changes with preserved/coerced mtime still invalidate cache because the hash
  changes;
- symlinked cache is rejected or safely replaced;
- malformed JSON and unknown schema return `compile_error!`, never macro panic;
- cache with a valid hash but invalid graph IDs/attributes is rejected;
- no serialized cache field is ever parsed as a token stream;
- compile test scans generated output and verifies only importer-generated constructs exist.

---

## 70. SEC-005 — Public safe dtype traits feed unsafe byte representations

**Severity:** Critical design defect  
**CWE mapping:** unsafe contract exposure / type confusion  
**Primary sources:**

- `crates/incin-core/src/tensor/dtype.rs:7-37`
- `crates/incin-core/src/tensor/base.rs:553-566`

### 70.1 Evidence

`DType` and `ConstDType` are public safe traits. `ConstDType::Elem` requires only
`Copy + Debug + Send + Sync`. `Tensor::from_slice` then views any `&[K::Elem]` as bytes
inside an unsafe block.

External safe code can implement the trait with an element type that has padding, a
nonportable representation, invalid byte exposure assumptions, or a `DTypeId` inconsistent
with its element width. Safe trait implementation must not be sufficient to satisfy an
unsafe representation contract.

### 70.2 Required API split

Separate three concepts:

1. **logical dtype identity**—safe and inspectable;
2. **plain storage element representation**—unsafe or sealed;
3. **backend/custom dtype codec**—explicit conversion contract.

Recommended core:

```rust
pub trait DType: sealed::DTypeSealed + 'static + ... {
    type Field;
    fn id(field: &Self::Field) -> DTypeId;
}

pub trait PlainDType: DType {
    type Elem: TensorElement;
}

mod sealed {
    pub trait TensorElementSealed {}
}

pub trait TensorElement:
    sealed::TensorElementSealed + bytemuck::Pod + bytemuck::Zeroable + Copy + 'static
{}
```

For an external custom dtype, expose an explicitly unsafe SDK contract:

```rust
pub unsafe trait ExternalElementCodec: Send + Sync + 'static {
    type Elem: Copy + 'static;
    const WIDTH: usize;
    fn encode_slice(...);
    fn decode_slice(...);
}
```

The unsafe trait documentation must state every invariant. An external implementation is
never automatically granted a built-in `DTypeId`.

### 70.3 Tests and API gate

- compile-fail external safe implementation of built-in representation traits;
- custom logical dtype can exist without gaining raw byte reinterpretation;
- padded element test cannot enter `Tensor::from_slice`;
- exact width consistency is tested for every built-in dtype;
- semver review treats these traits as part of the 0.1 stability boundary.

---

## 71. SEC-006 — Downloader permits path escape, symlink overwrite, and poisoned partial files

**Severity:** High  
**CWE mapping:** CWE-22 (path traversal), CWE-59 (link following), CWE-494 (download
without integrity check), CWE-400 (resource exhaustion)  
**Primary source:** `crates/incin-data/src/downloader.rs:9-46`

### 71.1 Evidence

`cache_dir.join(filename)` accepts absolute paths and `..` components. `File::create`
follows symlinks. Existing files are trusted solely because they exist. Network and gzip
streams are copied without byte, time, ratio, or digest limits. Writes go directly to the
final file, so interruption leaves a partial file that future calls accept as complete.

### 71.2 Required primitives

Create a shared `incin-io` or core I/O hardening module:

```rust
pub struct RelativeAssetPath(PathBuf);
pub struct ExpectedBlob {
    size: Option<u64>,
    digest: Option<Digest>,
}
pub struct DownloadLimits {
    max_compressed_bytes: u64,
    max_decompressed_bytes: u64,
    max_expansion_ratio: u32,
    connect_timeout: Duration,
    read_timeout: Duration,
    total_timeout: Duration,
    max_redirects: u8,
}
pub struct AtomicBlobWriter { ... }
```

`RelativeAssetPath::new` rejects:

- absolute paths;
- root/prefix components;
- `..`;
- empty final names;
- NUL/control characters where relevant;
- platform path separators inside remote logical names;
- reserved device names on Windows.

### 71.3 Atomic download algorithm

1. resolve and create the cache root with private permissions;
2. open the root as a directory handle where the platform permits;
3. resolve a validated relative asset path under that root;
4. acquire a per-blob lock;
5. if a final file exists, verify type, size, and digest—never trust `exists()`;
6. create a random temporary regular file in the same directory with `create_new`;
7. refuse symlinks and nonregular targets;
8. stream through a counting and hashing reader;
9. enforce timeout and maximum bytes;
10. for gzip, enforce compressed size, decompressed size, and expansion ratio;
11. flush and `sync_all` when durability is required;
12. atomically rename into place;
13. sync the parent directory where supported;
14. remove the temporary file on every failure;
15. return a verified blob descriptor, not merely a path.

### 71.4 Dataset requirements

Built-in datasets such as MNIST must pin:

- canonical URL;
- compressed SHA-256;
- expected compressed size;
- decompressed SHA-256 or exact decoded format constraints;
- IDX magic values;
- image/label counts and dimensions.

### 71.5 Tests

- `filename = "../../victim"`;
- absolute Unix and Windows-style paths;
- final path is a symlink;
- cache root component is a symlink;
- interrupted write leaves no trusted final file;
- two concurrent downloaders produce one verified result;
- gzip bomb exceeds output/ratio limit;
- endless/slow response reaches timeout;
- redirect loop and cross-scheme redirect policy;
- existing wrong-digest file is quarantined and redownloaded;
- error paths do not leave stale locks or temporary files.

---

## 72. SEC-007 — Parsers accept attacker-controlled sizes without a shared resource budget

**Severity:** High  
**CWE mapping:** CWE-400 (uncontrolled resource consumption), CWE-190 (integer overflow),
CWE-125/787 amplification risk from bad offsets  
**Primary sources include:**

- `crates/incin-core/src/io/inspect.rs`
- `crates/incin-core/src/nn/save.rs`
- `crates/incin-core/src/serialize.rs`
- `crates/incin-macros/src/onnx.rs`
- `crates/incin-macros/src/safetensors.rs`
- `crates/incin-data/src/vision/mnist.rs`
- `crates/incin-backends/src/tuning/cache.rs`

### 72.1 Confirmed examples

- safetensors inspection reads an untrusted `u64` header length and allocates that amount
  before checking it against the file size or a configured maximum;
- shape products and `element_count * bytes_per_element` use unchecked multiplication;
- GGUF strings, tensor counts, dimension counts, metadata arrays, and recursive array
  nesting are unbounded;
- GGUF consecutive offsets are subtracted without first proving monotonicity and file
  containment;
- ONNX and safetensors macros read complete files into memory;
- ONNX code indexes node inputs/outputs and attributes without complete arity validation;
- negative ONNX dimensions can be cast into very large unsigned dimensions;
- MNIST ignores the parsed magic value and allocates `count * rows * cols` unchecked;
- tuning cache performs `fs::read` before applying `max_bytes`;
- bincode-backed reads have no framework-wide input budget.

### 72.2 One mandatory limit type

Every parser must accept a `ResourceLimits` value. Do not define unrelated constants in
each importer.

```rust
#[derive(Clone)]
pub struct ResourceLimits {
    pub max_file_bytes: u64,
    pub max_header_bytes: u64,
    pub max_tensor_count: usize,
    pub max_metadata_entries: usize,
    pub max_name_bytes: usize,
    pub max_rank: usize,
    pub max_dimension: u64,
    pub max_tensor_bytes: u64,
    pub max_total_tensor_bytes: u64,
    pub max_graph_nodes: usize,
    pub max_graph_edges: usize,
    pub max_nesting_depth: usize,
    pub max_string_bytes: usize,
    pub max_archive_entries: usize,
    pub max_archive_expanded_bytes: u64,
}
```

Provide conservative defaults by workflow:

- `inspection_defaults`;
- `model_load_defaults`;
- `compile_time_defaults`;
- `trusted_local_large_model`;
- caller-defined values with validated upper bounds.

### 72.3 Bounded reader rules

- check filesystem metadata before allocation, but do not rely on it alone because streams
  and files may change;
- wrap input in `Take(max + 1)` and reject if the extra byte is observed;
- convert `u64 -> usize` with `try_from`;
- use `checked_mul`, `checked_add`, and checked range construction;
- check rank before allocating a dimension vector;
- check every offset range is monotonic, nonoverlapping where required, and within the
  actual payload;
- cap recursion or replace recursive metadata traversal with an explicit bounded stack;
- avoid `serde_json::Value` for huge headers when a typed streaming form is feasible;
- no parser may use direct indexing until arity is validated;
- parser errors include format, byte offset/field, limit, and observed value.

### 72.4 GGUF policy

GGUF remains low priority, but the inspector is already a parser and therefore needs
hardening now. Until complete:

- label GGUF inspection experimental;
- never allocate from raw tensor/metadata counts without limits;
- reject descending offsets;
- reject dimensions that cannot fit `usize`;
- reject data offsets beyond file length;
- cap metadata array nesting and count;
- fuzz the parser independently of full model execution.

### 72.5 ONNX policy

- decode through a bounded input;
- validate graph and tensor names;
- validate each operator's arity before indexing;
- reject negative fixed dimensions;
- cap initializers and total embedded tensor bytes;
- validate external-data references through `RelativeAssetPath`;
- reject control-flow graphs whose nesting exceeds policy;
- generate diagnostics, not proc-macro panics.

### 72.6 Fuzz targets

Create `fuzz/` targets for:

- safetensors header/loader;
- GGUF inspector;
- ONNX importer/cache;
- `.incin` artifact envelope;
- state dictionary manifest;
- NumPy `.npy` and `.npz`;
- dataset IDX;
- distributed wire messages;
- tuning cache envelope.

Invariant for every fuzz target:

> For every byte string under the harness limit, the parser returns success or a structured
> error within bounded time and memory. It never panics, aborts, loops indefinitely, or
> constructs a validated typed object from inconsistent metadata.

---

## 73. SEC-008 — User-controlled paths allow telemetry escape and dangerous cache deletion

**Severity:** High  
**CWE mapping:** CWE-22, CWE-59, CWE-73 (external control of file name/path)  
**Primary sources:**

- `crates/incin-telemetry/src/emitter.rs:134-158`
- `crates/incin-telemetry/src/transport/file.rs:34-43`
- `crates/incin/src/tune_report.rs:29-51`

### 73.1 Telemetry run path

`Emitter::to_run_dir(Some(name))` appends `"{run_id}.jsonl"` without applying the run-ID
validation already present in the socket transport. Names containing separators or parent
components can escape the default run directory. `FileTransport::open` follows symlinks.

Required fix:

- central `RunId` newtype shared by file and socket transports;
- generated UUID or a bounded safe character set;
- reject empty, `.`, `..`, separators, control characters, and platform-special names;
- open run directory privately;
- no-follow regular-file creation/append;
- ownership/type checks before appending;
- explicit API for caller-selected arbitrary output paths, separate from a run ID.

### 73.2 Autotune clear

`cargo incin tune --clear` turns `INCIN_AUTOTUNE_CACHE_DIR` into a path and passes it to
`remove_dir_all`, ignores deletion errors, and reports success.

An accidental or malicious environment value could point to a home directory, workspace,
mounted data directory, or other recursively deletable location.

Required fix:

1. clearing must call the tuning cache API, not recursively delete an arbitrary directory;
2. create an ownership sentinel/manifest when Incin creates a cache root;
3. canonicalize and reject root, home, current directory, ancestors, shallow paths, and
   symlinked roots;
4. a custom cache directory requires an explicit flag and printed resolved path;
5. destructive action errors return nonzero status;
6. do not report “cleared” when deletion failed;
7. tests cover `/`, home, `.`, `..`, symlink, nonowned directory, and a valid cache.

---

## 74. SEC-009 — Distributed control and NCCL bootstrap are unauthenticated plaintext

**Severity:** High; Critical for hostile/reachable networks  
**CWE mapping:** CWE-306 (missing authentication), CWE-319 (cleartext transmission),
CWE-345 (insufficient verification of data authenticity)  
**Primary sources:**

- `crates/incin-core/src/dist/context.rs`
- `crates/incin-backends/src/dist/nccl.rs` bootstrap protocol

### 74.1 Evidence

The run ID is explicitly documented as non-secret. Rank peers exchange fixed-format
messages over raw TCP. Control messages can abort or shut down the peer. NCCL bootstrap
transmits plan identifiers and the NCCL unique ID without authenticated encryption. Rank
zero accepts the first connector that presents the expected public fields.

### 74.2 Threats

- first-connector race and peer impersonation;
- forged abort/shutdown and denial of service;
- replay of an old handshake;
- tampering with rank, world, plan, topology, or communicator identity;
- disclosure of topology and bootstrap data;
- joining or disrupting the wrong communicator;
- silent downgrade to an insecure protocol.

### 74.3 Required product decision

Because the chosen `1.0` target is single-device and distributed remains preview, the
fastest safe policy is:

- default network rendezvous binds to loopback/private explicit interfaces;
- nonlocal plaintext mode is disabled unless the user passes an unmistakable
  `InsecureDevelopmentOnly` policy;
- logs and docs state that insecure mode provides no confidentiality or peer authenticity;
- production network claims wait for authenticated transport.

### 74.4 Production protocol design

Do not invent a cryptographic protocol from primitives. Use a maintained TLS or Noise
implementation.

Handshake must bind:

- protocol and schema version;
- run/session random nonce;
- both peer identities and roles;
- rank/world;
- endpoint identities;
- topology fingerprint;
- plan/graph hash;
- collective sequence contract;
- compiler/artifact identity where relevant;
- NCCL unique ID;
- expiry and replay protection.

Use either:

- mutual TLS with pinned certificates/CA and hostname or explicit peer identity; or
- an authenticated Noise protocol with a per-run secret/key pair provisioned by the
  launcher.

Control and NCCL bootstrap must share the authenticated session or transcript hash. A
public run ID is an identifier, not a key.

### 74.5 Tests

- fake peer wins connection race;
- wrong certificate/key;
- replay old transcript;
- tamper rank/world/plan/NCCL ID;
- forged abort/shutdown;
- protocol downgrade;
- wrong run ID with valid key;
- timeout and partial frame;
- oversized frame;
- reconnect after failure;
- secret redaction in logs and diagnostics.

---

## 75. SEC-010 — Compiled artifacts lack strong framing, bounded decoding, and semantic verification

**Severity:** High now; potentially Critical once artifacts execute native code  
**CWE mapping:** CWE-347 (improper verification), CWE-502, CWE-400  
**Primary source:** `crates/incin-core/src/compiled/artifact.rs:1-145`

### 75.1 Evidence

- `ARTIFACT_MAGIC` exists but is never serialized or checked;
- the entire artifact is unbounded JSON;
- Adler-32 detects accidental corruption but is trivial for an attacker to recompute;
- compatibility checks only format and framework major;
- loading verifies checksum and compatibility but not graph/plan semantics;
- the current format does not frame executable sections or bind target/runtime/compiler
  requirements.

### 75.2 Required envelope

```text
magic
fixed header version
header length
manifest length
section table length
payload length
canonical manifest
section descriptors
section bytes
content digest
optional signature block
```

Every length is bounded and checked before allocation.

Manifest includes:

- artifact schema and public ABI;
- Incin version;
- compiler target and compiler build identity;
- runtime/driver requirements;
- graph hash;
- input/output contracts;
- dynamic guard program;
- precision/determinism policies;
- required capabilities;
- section hashes;
- provenance and optional signer.

### 75.3 Integrity versus authenticity

- BLAKE3 or SHA-256: detects corruption and identifies content;
- Ed25519 or another reviewed signature scheme: authenticates publisher/package;
- neither replaces semantic plan validation.

Never label an unkeyed checksum as proof that an artifact came from a trusted source.

### 75.4 Semantic verifier

Before allocation or execution, verify:

- unique node/value IDs;
- valid DAG/topological order;
- every input and output definition;
- operation arity and attributes;
- dtype/shape/layout compatibility;
- checked memory sizes and alignments;
- allocation ranges and nonoverlap/alias rules;
- guard coverage for every specialization;
- target capability requirements;
- executable section kind and hash;
- no unknown mandatory section;
- no out-of-range offsets;
- bounded name and metadata lengths.

### 75.5 Tests

- wrong magic/version/length;
- truncation at every byte boundary;
- duplicate/overlapping sections;
- forged checksum/digest;
- valid digest but invalid plan;
- incompatible target/runtime;
- zip-bomb-like JSON nesting is impossible because decoding is bounded;
- signature valid/invalid/unknown key/revoked key;
- fuzz loader and verifier;
- execution cannot receive an artifact until the verifier returns a sealed
  `ValidatedArtifact`.

---

## 76. SEC-011 — Backend shape and byte arithmetic must be checked once and everywhere

**Severity:** High  
**CWE mapping:** CWE-190, CWE-400, backend memory-safety amplification  
**Affected areas:** CPU, WGPU, Metal, CUDA, external backends and imported storage paths

### 76.1 Required central API

```rust
pub struct CheckedNumel(usize);
pub struct CheckedByteLen(usize);

impl TensorMeta {
    pub fn checked_numel(&self, limits: &ResourceLimits) -> Result<CheckedNumel>;
    pub fn checked_byte_len(&self, limits: &ResourceLimits) -> Result<CheckedByteLen>;
}
```

No backend should repeat `shape.iter().product()` or `elements * size` in a user-controlled
path.

### 76.2 Migration

1. inventory every product/addition involved in shape, stride, allocation, launch, and byte
   offsets;
2. replace with checked core helpers;
3. return structured overflow/limit errors;
4. remove `expect` from storage constructors and imported metadata paths;
5. validate before any host or device allocation;
6. validate integer narrowing before kernel launch parameters;
7. fuzz descriptor-to-backend binding;
8. add boundary tests around `usize::MAX`, `u32::MAX`, device allocation limits, and
   quantized block rounding.

---

## 77. SEC-012 — Cache limits are applied after whole-file allocation

**Severity:** Medium–High  
**Primary source:** `crates/incin-backends/src/tuning/cache.rs:726-780`

`fs::read(path)` allocates the complete file before `limits.max_bytes` influences pruning.
A malicious or corrupted user-writable cache can therefore cause a large allocation before
the cache's own configured maximum is enforced.

### Required fix

- inspect metadata and reject files larger than `max_bytes`;
- read through `Take(max_bytes + 1)`;
- cap JSON nesting/records and string sizes;
- validate file is a regular file owned/acceptable under cache policy;
- no-follow lock, temp, and cache files;
- quarantine through an atomic rename within the cache root;
- custom checksum remains corruption detection only;
- preserve the current legal-candidate revalidation because that is an important defense.

Apply the same rule to every compiler, plan, model, telemetry index, and Python bridge
cache.

---

## 78. SEC-013 — Local telemetry socket has confidentiality and backpressure risks

**Severity:** Medium; deployment dependent  
**Primary source:** `crates/incin-telemetry/src/transport/socket.rs`

The socket name is based on a run ID and there is no application-layer client
authentication. Connected streams remain blocking; a client that connects and stops
reading may stall the writer thread on `write_all`.

### Required fix

- create a random per-run bearer token or authenticated local handshake;
- ensure namespace/file permissions restrict the current OS user where possible;
- do not print the token in ordinary logs;
- use bounded per-client queues or nonblocking writes with deadlines;
- drop slow clients without blocking other telemetry destinations;
- document telemetry data sensitivity;
- test an unauthorized client, a slow client, many clients, disconnect races, and shutdown.

Telemetry must never be allowed to stall training indefinitely.

---

## 79. SEC-014 — CI and dependency supply-chain controls are insufficient

**Severity:** High process/release risk  
**Primary sources:** `.github/workflows/ci.yml`, `.github/workflows/hardware.yml`,
`Cargo.lock`

### 79.1 Confirmed observations

- GitHub Actions are referenced by movable tags such as `actions/checkout@v4`;
- the main CI workflow does not declare one minimal top-level `permissions` policy;
- no `cargo-audit`, `cargo-deny`, Dependabot configuration, SBOM, or dedicated security
  workflow was found;
- the lockfile contains `bincode 1.3.3`;
- the current RustSec advisory database marks bincode as permanently unmaintained with no
  patched versions;
- no Git-sourced Rust dependency was found in the audited manifests/lockfile, which is a
  positive property to preserve.

### 79.2 Required controls

1. pin every third-party action to a full commit SHA with a comment recording the human
   version;
2. set `permissions: contents: read` by default and grant narrower job-specific write
   permissions only where needed;
3. prevent untrusted pull-request code from receiving release/hardware secrets;
4. add:
   - `cargo audit`;
   - `cargo deny check advisories bans licenses sources`;
   - OSV scan as a supplemental ecosystem check;
   - secret scanning;
   - actionlint;
   - lockfile diff review;
   - SBOM generation for releases;
5. enable Dependabot/Renovate for Cargo, GitHub Actions, Python, and any managed compiler;
6. pin the release Rust toolchain and keep moving stable as a separate compatibility job;
7. sign release tags and artifacts and publish checksums/provenance;
8. remove bincode from untrusted/public interchange. If retained temporarily for a trusted
   internal cache, bound it, version it, and document that it is not a stable format;
9. establish a dependency exception file with owner, reason, expiry, and compensating
   controls—never silent ignores.

### 79.3 Security references

- RustSec advisory: `RUSTSEC-2025-0141`, “bincode is unmaintained,” no patched versions:
  https://rustsec.org/advisories/RUSTSEC-2025-0141.html
- GitHub recommends pinning actions to a full-length commit SHA as the immutable form:
  https://docs.github.com/en/actions/reference/security/secure-use
- GitHub also recommends explicitly declaring minimal workflow permissions:
  https://docs.github.com/en/code-security/tutorials/secure-your-organization/protect-against-threats

---

## 80. SEC-015 — Python ecosystem interoperability must not import executable serialization

**Severity:** High  
**Area:** planned PyTorch/NumPy bridge and first-class formats

### 80.1 PyTorch checkpoint rule

Python pickle can execute code. Incin must never advertise arbitrary `.pt`/`.pth` loading
as a safe direct model format.

Stable policy:

- prefer safetensors for weights;
- prefer `torch.export` or a constrained graph interchange for executable structure;
- a Python conversion command runs in a separate process with an explicit trust warning;
- use `torch.load(..., weights_only=True)` where compatible, but still enforce process,
  memory, time, tensor-size, and output-format limits;
- never silently retry with `weights_only=False`;
- loading a full pickled module requires `TrustPolicy::LocalTrusted` and a loud
  confirmation outside library defaults;
- the converter emits data-only safetensors plus a canonical graph/config manifest;
- conversion output is revalidated by Rust before typed witnesses are created.

PyTorch's own documentation states that `weights_only=True` narrows remote-code-execution
surface but does not protect against denial of service and may not eliminate every memory
corruption risk. Treat it as defense in depth, not a sandbox:
https://docs.pytorch.org/docs/main/notes/serialization.html

### 80.2 NumPy policy

For `.npy`:

- reject object dtype;
- reject pickle-backed arrays;
- validate header length, shape, dtype, endian and exact data length;
- handle Fortran order explicitly;
- checked numel and byte count;
- memory mapping only after file identity and range validation.

For `.npz`:

- treat it as an archive;
- reject absolute/parent paths and duplicate normalized names;
- cap entry count, compressed bytes, expanded bytes and compression ratio;
- do not extract to arbitrary filesystem paths merely to read arrays;
- reject nested archives by policy.

### 80.3 Safetensors policy

Safetensors is data-only and preferable to pickle, but Incin must still validate:

- bounded header length;
- unique names;
- exact dtype support;
- checked shapes and offsets;
- no overlap where prohibited;
- exact payload range;
- total bytes and tensor count;
- state-schema compatibility.

The format's safety goal does not remove Incin's responsibility to validate its own typed
contracts.

---

## 81. SEC-016 — Future model-package paths and archives must be containment-safe

**Severity:** High future requirement  
**Area:** Hugging Face, ONNX external data, sharded safetensors, NPZ, GGUF side assets,
tokenizers and generated source

### 81.1 Package resolver rules

A `ModelPackage` resolver must:

- pin an immutable revision/commit for reproducibility;
- normalize every logical path once;
- reject path traversal, absolute paths, drive prefixes, NUL and ambiguous Unicode policy;
- refuse symlinks by default for untrusted packages;
- validate downloaded blob hash/size against a manifest;
- bind shard index entries to contained relative paths;
- cap number and total size of shards;
- use one content-addressed blob store and atomic materialization;
- separate trusted generated Rust source from downloaded data;
- record provenance in the generated manifest;
- never execute repository Python code merely to discover configuration.

### 81.2 ONNX external data

Before supporting ONNX external tensors:

- external `location` must be a contained `RelativeAssetPath`;
- offset and length must be checked against the referenced blob;
- duplicate/overlapping ranges follow explicit policy;
- URLs inside model metadata are not fetched automatically;
- path resolution cannot escape via symlink;
- every external blob participates in the package digest/signature.

### 81.3 Hugging Face custom code

Features analogous to `trust_remote_code=True` must be **off by default**. Incin should
parse standard config/tokenizer/model metadata and use its own architecture registry.
Executing downloaded Python/Rust/build scripts requires a separate explicit trusted
workflow, not a convenience fallback.

---

## 82. SEC-017 — Correct the file durability documentation

**Severity:** Medium documentation/correctness  
**Primary source:** `crates/incin-telemetry/src/transport/file.rs:1-12`

The documentation implies `write_all` prevents partial bytes from becoming externally
observable before an error. `write_all` is a retrying convenience and does not provide
transactional record atomicity or crash durability. A process can be interrupted after a
partial underlying write.

Fix the wording:

- prior complete records remain structurally independent because JSONL is append-only;
- the final record may be partial after process termination;
- concurrent writers need an explicit serialization/locking policy;
- power-loss durability requires sync policy;
- readers must tolerate and ignore/report an incomplete final line.

Add a fault-injection writer that fails after every byte position.

---

## 83. SEC-018 — Program-wide unsafe, panic, and dependency audit

**Severity:** Program-wide

A lexical scan of the audited tree found approximately:

| Pattern | All Rust files | Rough non-test/example paths | Files in rough production set |
|---|---:|---:|---:|
| `unsafe` | 176 | 167 | 22 |
| `.unwrap(` | 2,807 | 1,180 | 85 |
| `.expect(` | 628 | 317 | 50 |
| `panic!(` | 112 | 69 | 33 |

These counts include documentation, inline test modules, generated-style code, and
unreachable paths. They are triage signals, not proof that every occurrence is a bug.

### 83.1 Required unsafe ledger

Create `docs/security/unsafe-ledger.md` generated or checked by tooling. Every production
unsafe block records:

- ID;
- file/function;
- owner;
- reason safe Rust is insufficient;
- preconditions;
- invariants established;
- aliasing/alignment/initialization/lifetime assumptions;
- caller trust;
- Miri/sanitizer/test evidence;
- review date;
- whether removal is planned.

CI fails when a new unsafe block lacks a ledger entry and a `// SAFETY:` explanation.

### 83.2 Panic policy

Classify every panic/unwrap/expect:

- compile-time proof or impossible internal invariant;
- test-only;
- process startup/configuration;
- user-controlled runtime input;
- backend/hardware result;
- parser/deserializer;
- thread/lock poisoning.

Rules:

- user-controlled/runtime/parser/backend failures return structured errors;
- an “impossible” invariant must be enforced structurally and tested by a semantic mutant;
- poisoned locks must not silently continue with corrupt shared state;
- procedural macros emit diagnostics rather than panic;
- command-line tools return meaningful nonzero exit status;
- no broad `catch_unwind` is used to conceal invariant failures.

---

## 84. Cross-cutting security architecture

### 84.1 `incin-security` is not required as a new crate

Avoid creating a vague dumping-ground crate. Place primitives with their owners:

- `ResourceLimits`, checked tensor sizes: `incin-core`;
- `RelativeAssetPath`, atomic blob/cache transaction: `incin-data` or a focused
  `incin-io` crate only if reused by core, compiler and telemetry;
- artifact verifier: `incin-core::compiled`;
- distributed secure channel: distributed core/backend boundary;
- state schema/transaction: `incin-core::nn`;
- trust/provenance policy: facade/core model I/O;
- Python sandbox/process policy: `incin-python`.

### 84.2 Sealed validation tokens

Use nonforgeable types:

```rust
pub struct ValidatedBlob private;
pub struct ValidatedStateLoad<B> private;
pub struct ValidatedArtifact private;
pub struct AuthenticatedDistributedSession private;
pub struct VerifiedModelPackage private;
```

Execution APIs consume these values instead of raw parsed structs or paths.

### 84.3 Error taxonomy

Add structured errors with stable categories:

```rust
pub enum SecurityError {
    LimitExceeded { resource: ResourceKind, limit: u64, observed: u64 },
    PathEscape { logical: String },
    SymlinkRejected { path: PathBuf },
    DigestMismatch { expected: Digest, observed: Digest },
    SignatureInvalid { key_id: String },
    UntrustedExecutableFormat { format: String },
    AuthenticationFailed,
    ReplayDetected,
    UnsafeLegacyCache,
    InvalidRepresentation { dtype: DTypeId, detail: String },
}
```

Avoid leaking secrets, tokens, full private paths, or model contents into errors.

### 84.4 Secure defaults

- model-hub content: untrusted;
- pickle/custom code: rejected;
- network distributed mode: authenticated or explicit insecure preview;
- fallback device transfer: explicit;
- artifact signature: optional for local files, policy-enforced for remote deployment;
- cache and telemetry paths: private and contained;
- external backend/custom dtype: capability-limited and conformance-tested;
- compiler toolchain download: pinned digest and provenance.

---

## 85. Security issue and PR sequence

Agents work sequentially. Each row is one issue followed by one focused PR unless the issue
explicitly defines a small stack.

| Order | Issue | Primary agent | Reviewer | Why now |
|---:|---|---|---|---|
| 1 | SEC-001 tensor extraction soundness | GPT-5.6 | Opus 5 | Direct safe-API UB |
| 2 | SEC-005 dtype/element unsafe contract | Opus 5 design, GPT implementation | Opus 5 | Prevents reintroducing SEC-001 class |
| 3 | SEC-002 Candle exact byte codec | GPT-5.6 | Opus 5 | Direct backend UB |
| 4 | SEC-003 atomic validated state load | Opus 5 then GPT | Opus 5 | Protects typed model invariant |
| 5 | SEC-004 remove code-bearing ONNX cache | GPT-5.6 | Opus 5 | Build-host code injection |
| 6 | SEC-007 shared resource limits | Opus 5 | human + GPT | Foundation for all parsers |
| 7 | SEC-006 secure downloader/cache writer | GPT-5.6 | Gemini Pro + Opus | Path and cache integrity |
| 8 | parser migrations: safetensors/ONNX/GGUF/IDX/cache | Gemini agent 1 | GPT-5.6 | Bounded repetitive work |
| 9 | SEC-008 path/delete hardening | Gemini agent 1 | GPT-5.6 | Small focused filesystem fixes |
| 10 | SEC-010 artifact envelope/verifier | Opus 5 + GPT | human + Opus | Deployment foundation |
| 11 | SEC-011 checked backend arithmetic | Gemini agent 1 | GPT-5.6 | Mechanical broad migration |
| 12 | SEC-009 secure distributed policy/protocol | Opus 5 + GPT | human | Cryptographic/network design |
| 13 | SEC-013 telemetry auth/backpressure | GPT-5.6 | Opus 5 | Local privacy/availability |
| 14 | SEC-014 security CI/supply chain | Gemini agent 2 | GPT-5.6 | Permanent gates |
| 15 | fuzz/Miri/sanitizer campaign | Gemini agents | GPT + Opus | Verify entire closure |
| 16 | security docs/release review | Gemini agent 2 | human + Opus | 0.1 gate |

Do not begin broad interoperability work until orders 1–7 are merged. Otherwise new
loaders will duplicate unsafe primitives and require a second rewrite.

---

## 86. Standard security issue template

Every security issue must contain:

```markdown
## Threat boundary
What input or actor is untrusted?

## Security property
What must never happen?

## Source evidence
Exact current paths, functions, and commit.

## Exploitability
What preconditions are needed? Is impact confirmed or potential?

## Required design
Types, ownership, validation order, and error behavior.

## Non-goals
What this issue deliberately does not solve.

## Positive tests
Valid behavior.

## Adversarial tests
Malformed, boundary, race, replay, symlink, overflow, partial failure.

## Semantic mutants
Intentionally broken implementations that the tests must reject.

## Tool evidence
Miri, sanitizer, fuzz corpus, audit/deny, hardware, platform-specific tests.

## Documentation
Safety comments, threat model, migration, user warning.

## Completion gate
No unchecked legacy path remains.
```

The linked PR description includes `Fixes #<issue>`, exact commands/output, and a checklist
mapping every acceptance criterion to code/tests.

---

## 87. Ready-to-paste agent packets

### 87.1 Packet A — weaker-model instructions for memory-safety fixes

```text
You are implementing one Incin security issue. Do not redesign unrelated APIs.

1. Read the issue and every file it names.
2. Locate every caller of the unsafe function before editing.
3. Write failing tests first:
   - one valid case;
   - one invalid length;
   - one invalid type/representation compile-fail case;
   - one unaligned-byte case;
   - one overflow case.
4. Remove the unsafe operation from the safe public path.
5. Use checked arithmetic and exact dtype matching.
6. Do not add transmute, from_raw_parts, read_unaligned, or a new unsafe trait unless the
   issue explicitly authorizes it.
7. Run the focused test, workspace test, Clippy, and Miri command from the issue.
8. Search for the deleted pattern and include the result in the PR.
9. Update the unsafe ledger and PROPOSALS/security row.
10. Stop if the new design requires guessing a dtype, shape, byte order, or ownership rule.
```

### 87.2 Packet B — weaker-model instructions for parser hardening

```text
Implement only the named parser migration.

Inputs are hostile. Follow this exact order:
1. Reject file/stream larger than ResourceLimits before whole-file allocation.
2. Read fixed framing fields.
3. Convert integer widths with try_from.
4. Check count/rank/string limits before allocating.
5. Use checked_add and checked_mul for every size/range.
6. Verify ranges are inside the payload and satisfy ordering/overlap rules.
7. Parse into an unvalidated structure.
8. Run semantic validation.
9. Construct a sealed validated object only after success.
10. Return a structured error with field and observed/limit values.

Never:
- index a vector before checking arity;
- use unwrap/expect on input-derived data;
- recurse without a depth counter;
- trust an extension or checksum as semantic validation;
- allocate from a raw count before checking it.

Add corpus tests for empty, truncated, maximum-valid, one-over-limit, overflow, descending
offset, duplicate name, invalid UTF-8 policy, and trailing data.
```

### 87.3 Packet C — weaker-model instructions for filesystem hardening

```text
The target path is attacker-controlled until RelativeAssetPath validates it.

1. Reject absolute/root/prefix/parent components.
2. Resolve only under the opened cache/run root.
3. Refuse symlinks and nonregular files.
4. Use create_new for temporary files.
5. Stream with byte limits and a cryptographic digest.
6. Flush, sync when required, then atomic rename in the same directory.
7. Clean temporary files on all error paths.
8. Verify existing files; never trust exists().
9. Add concurrency and symlink tests.
10. Never call remove_dir_all on a path obtained directly from an environment variable.
```

### 87.4 Packet D — adversarial reviewer prompt

```text
Review this security PR as an attacker and as a Rust unsafe-code reviewer.

Attempt to disprove:
- alignment;
- initialization/valid bit patterns;
- aliasing and lifetimes;
- integer bounds;
- path containment and symlink resistance;
- atomicity after partial failure;
- parser memory/time limits;
- state rollback;
- artifact semantic validation;
- network peer authentication and replay resistance;
- secret redaction.

For every claimed property, identify the exact test that fails when the implementation is
mutated. Reject the PR if a property exists only in comments.
```

---

## 88. Security CI design

### 88.1 Per-pull-request gates

- formatting and Clippy with warnings denied;
- complete workspace tests under default and selected feature sets;
- `cargo audit`;
- `cargo deny check`;
- unsafe-ledger diff validation;
- no new production `unwrap`/`expect`/`panic` without an allow record;
- action pin validation;
- secret scan;
- parser seed corpus tests;
- API compatibility check for the stabilized surface after 0.1.

### 88.2 Scheduled gates

- fuzz each parser/artifact/state target with retained corpus;
- Miri for core byte/aliasing tests;
- AddressSanitizer/UndefinedBehaviorSanitizer where supported;
- ThreadSanitizer or Loom for cache/telemetry/control concurrency where practical;
- hardware adversarial tests for CUDA/WGPU/Metal storage binding;
- dependency update and advisory report;
- malformed model corpus;
- distributed authentication/replay tests on two processes/hosts.

### 88.3 Release gates

- clean advisory/deny report or documented, time-bounded exception;
- SBOM;
- signed tag, checksums, artifact provenance;
- reproducible compiler/toolchain manifest;
- no critical/high unresolved finding reachable in the released feature set;
- security changelog and migration notes;
- private vulnerability reporting route tested by the maintainer.

---

## 89. Security documentation deliverables

Create:

- `SECURITY.md`;
- `docs/security/threat-model.md`;
- `docs/security/unsafe-ledger.md`;
- `docs/security/model-trust.md`;
- `docs/security/artifact-signing.md`;
- `docs/security/distributed-security.md`;
- book chapter: “Loading models safely”;
- PyTorch comparison page explaining why pickle is not a safe interchange default;
- backend-author guide for unsafe representation contracts;
- incident-response and release-revocation procedure.

`SECURITY.md` should state:

- supported versions;
- private reporting method;
- expected acknowledgement and remediation process;
- request not to publish active exploit details before coordination;
- which features are preview and excluded from production security claims;
- how signed releases/advisories are published.

---

## 90. Final security definition of done for `0.1.0`

The human maintainer signs this checklist:

### Memory safety

- [ ] SEC-001 closed; arbitrary `E` extraction removed.
- [ ] SEC-002 closed or Candle feature excluded.
- [ ] SEC-005 closed; representation contracts sealed/unsafe and documented.
- [ ] Miri passes the byte/storage/state test suite.
- [ ] Every production unsafe block has a ledger entry and test evidence.

### Typed invariants

- [ ] State loading is strict, validated, and atomic.
- [ ] No malformed dynamic storage can become a statically typed tensor/parameter.
- [ ] Backend allocation/launch sizes use checked central helpers.

### Model and data inputs

- [ ] ONNX sidecars are data-only and hash-bound.
- [ ] Every parser accepts `ResourceLimits`.
- [ ] Downloader/cache writes are contained, atomic, limited, and digest-aware.
- [ ] Pickle/object-array execution is rejected by default.
- [ ] malformed-input fuzz corpus passes without panic/OOM.

### Artifacts and caches

- [ ] artifact framing, digest, compatibility and semantic verifier exist.
- [ ] cache size limits apply before allocation.
- [ ] destructive cache commands cannot delete arbitrary paths.
- [ ] remote artifacts/packages can require digest/signature policy.

### Networking and telemetry

- [ ] distributed nonlocal plaintext is disabled by default or authenticated.
- [ ] control/bootstrap replay and tamper tests exist.
- [ ] telemetry paths are validated and slow clients cannot stall training.

### Supply chain and response

- [ ] actions pinned to immutable SHAs.
- [ ] minimal workflow permissions.
- [ ] audit/deny/secret/action scans in CI.
- [ ] bincode removed from untrusted interchange.
- [ ] `SECURITY.md`, threat model, unsafe ledger, and model-trust docs merged.
- [ ] release artifacts signed with published checksums/provenance.

No checkbox may be satisfied by a comment, planned issue, ignored test, or feature that is
enabled in the release but excluded from the evidence command.

---

## 91. Audit limitations and next dynamic pass

This security review is intentionally conservative and source-based. The audit environment
did not contain `cargo`, `rustc`, Miri, cargo-audit, cargo-deny, Semgrep, CodeQL, or the
target GPUs. Therefore:

- dependency advisories beyond the specifically verified bincode advisory are unknown;
- unsafe behavior has not yet been confirmed under Miri;
- no exploit proof was executed;
- platform-specific symlink and permission behavior has not been tested;
- network interception/replay has not been performed;
- GPU kernel memory safety has not been dynamically assessed;
- generated macro output has not been compiled in this environment;
- timing, denial-of-service thresholds, and concurrency failures are not measured;
- absence of a finding is not evidence of absence.

The first real development-machine security session must install the toolchain, capture the
baseline, and run the gates from Sections 88 and 90 before changing the readiness score.

---

# Appendix A — Immediate instructions for the first agent session

## A.1 Opus 5 first prompt

```text
Work from origin/develop commit eb3633525ea74e56f7a6b2d5c5b57dc74a5d9b8d or a newer explicitly recorded develop head.

Goal: establish the compiler hardening program before implementation begins.

Tasks:
1. Audit incin-core/src/compiled, tensor/tracing.rs, graph.rs, exec/spec.rs, module/state APIs, CLI, and ledger.
2. Write ADRs for canonical IR, session capture, artifact v2, IREE target isolation, dynamic shape constraints, and PyTorch bridge.
3. Add CMP2-001..010, IREE-001..007, TORCH/DLPACK/GRAD2 tasks to the machine-readable ledger with dependencies and evidence commands.
4. Add failing semantic tests proving current capture drops dtype/attributes, current guards are placeholders, current fusion is not executable composition, and current artifacts contain no executable variant.
5. Do not implement the new compiler yet except for minimal test seams.
6. Preserve all default/no_std feature contracts.

Deliverables:
- bounded diff;
- ADRs;
- ledger changes;
- failing tests marked as tracked/ignored only with exact blocker, or tests against isolated current behavior that demonstrate the defect;
- dependency graph;
- first three task handoffs for GPT-5.6.
```

## A.2 GPT-5.6 first prompt

```text
After the ADR/ledger PR merges, implement CMP2-001A: canonical IR identity, tensor type, values, symbols, constraints, and verifier skeleton.

Do not modify capture, IREE, Python, or public compile API.

Required:
- no_std + alloc compatible;
- stable explicit IDs;
- checked shape/cardinality/byte arithmetic;
- deterministic canonical ordering;
- structured verification errors;
- positive and negative tests;
- one semantic mutant per verifier category;
- rustdoc for every public type;
- no serialization format commitment beyond versioned in-memory schema unless ADR says otherwise.

Return command evidence and exact follow-up seams for descriptor conversion.
```

---

# Appendix B — Evidence commands to establish on the real development machine

The environment used to prepare this plan did not contain `rustc` or `cargo`, so no claim is made that the supplied branch currently passes these commands. The first implementation session must run and record them.

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test -p incin-core --no-default-features
cargo check -p incin --no-default-features --features std,cpu
cargo check -p incin --no-default-features --features std,cuda
cargo check -p incin --no-default-features --features std,wgpu
cargo check -p incin --no-default-features --features std,metal
cargo test -p incin-backends --features external-candle --test conformance
cargo xtask ledger
cargo xtask budgets
```

Add compiler program commands as tasks land:

```bash
cargo test -p incin-core --test compiled_ir
cargo test -p incin-core --test compiled_capture_v2
cargo test -p incin-core --test compiled_reference
cargo test -p incin-compiler
cargo test -p incin-iree --features compiler,runtime
cargo test -p incin-interop --features dlpack
python -m pytest python/incin_torch/tests
cargo incin benchmark --suite compiler-parity
```

Hardware commands must assert that tests actually executed and record device/driver/compiler metadata.

---

# Appendix C — Decision summary

1. IREE is a target, not Incin’s identity.
2. `Validated<Spec>` is the semantic source of truth.
3. Current `Graph` remains an interchange/viz structure; canonical executable IR is richer.
4. Capture is session-scoped and symbolic.
5. Reference execution lands before IREE.
6. StableHLO/standard MLIR is the first IREE input path.
7. Subprocess compilation lands before embedded compiler API.
8. Artifact v2 is sectioned, executable, hashed, and versioned.
9. Dynamic shapes use bounded symbols/affine constraints and explicit policies.
10. PyTorch integration uses `torch.export`, DLPack, `torch.compile`, and AOTAutograd.
11. Every Rust and Python graph path uses the canonical Incin IR; no parallel bypass is accepted.
12. No silent fallback, copy, device change, dtype cast, or benchmark simulation.
13. Same model API serves eager, compiled, imported, tested, and distributed workflows.
14. Framework quality is demonstrated through executable examples and fair benchmarks, not claims.

---

# Appendix D — External technical basis checked for this plan

The implementation strategy was cross-checked against current official documentation available on 2026-08-01:

- **IREE API bindings documentation:** C/C++ and Python are official compiler/runtime APIs; Rust remains unofficial/experimental.
- **IREE C API documentation:** the compiler exposes a versioned embedding API through a shared library; the runtime has a supported C API.
- **IREE developer overview:** supported core input dialects include StableHLO, TOSA, and Linalg; `iree-compile` is the main binary compiler driver.
- **PyTorch custom backend documentation:** a `torch.compile` backend receives an FX `GraphModule` plus example inputs and returns an equivalent callable; backends can be registered through the `torch_dynamo_backends` entry point.
- **PyTorch AOTAutograd documentation:** custom backends may compile normalized forward and backward graphs over a smaller core ATen operation set.
- **PyTorch `torch.export` documentation:** exported programs contain normalized functional ATen graphs, remove most Python control flow, and record the shape constraints needed for future inputs.
- **DLPack Python specification:** exchange should be zero-copy when possible, ownership must be retained, and CUDA/ROCm stream handoff must establish correct ordering.

Pin exact versions during implementation; do not assume current nightly APIs remain stable.


# Appendix E — Sequential Agent Execution Pack

**Use with:** this master plan  
**Baseline:** `origin/develop` at `eb3633525ea74e56f7a6b2d5c5b57dc74a5d9b8d`, or a newer `develop` SHA explicitly recorded in `STATUS.md` before work starts.

This file is designed to be copied directly into Opus 5 and GPT-5.6 sessions. Do not give an agent the entire program and ask it to “implement everything.” Give one bounded task at a time.

---

## 1. Coordinator prompt for Opus 5

```text
You are the architecture and integration owner for the Incin compiler program.

Mission:
Make Incin one coherent framework in which the same ordinary model API supports eager execution, compiled execution, dynamic shapes, Python interoperability, testing, autograd, and distributed planning.

Repository baseline:
- Start from origin/develop.
- Record the exact SHA before planning.
- The supplied audit baseline was eb3633525ea74e56f7a6b2d5c5b57dc74a5d9b8d.

Read completely before changing code:
1. docs/plan/remediation/master-implementation-plan.md
2. PROPOSALS.md
3. docs/plan/ledger.toml
4. crates/incin-core/src/compiled/*
5. crates/incin-core/src/tensor/tracing.rs
6. crates/incin-core/src/graph.rs
7. crates/incin-core/src/exec/{spec,proof,rule,meta,capability,context}.rs
8. crates/incin-core/src/nn/{module,save}.rs
9. crates/incin-macros/src/{module,onnx,safetensors}.rs
10. crates/incin/src/bin/cargo-incin.rs

Known audit findings to verify, not blindly trust:
- current captured IR drops attrs/value metadata/initializers/descriptors;
- current input guards are empty-shape F32 placeholders;
- current allocator counts slots rather than bytes/alias/device/alignment;
- folding and prepacking are no-ops;
- current fusion reduces nodes without creating executable composite semantics;
- current artifact contains JSON plan, not executable variants;
- current compiled tuning uses simulated latency;
- current tracing is global and often F32-hardcoded;
- no complete public model.compile().run() path exists.

Your responsibilities:
- freeze cross-cutting contracts through ADRs;
- keep dependency directions acyclic and optional features isolated;
- define bounded tasks and evidence commands;
- approve or reject API changes;
- review semantic mutants;
- merge in dependency order;
- ensure no ledger row is marked complete on placeholder evidence;
- maintain STATUS.md, DECISIONS.md, RISKS.md, SUPPORT_MATRIX.toml, HANDOFF.md.

Hard rules:
1. IREE is an optional target; do not rebuild IREE.
2. Validated<Spec> is the semantic source of truth.
3. No separate compiler-only tensor language.
4. No global capture state.
5. No silent fallback, device change, dtype conversion, host transfer, or synchronization.
6. No simulated timings in product paths.
7. No unbounded artifact parsing or cache growth.
8. Preserve default std+cpu and no_std contracts.
9. Every critical test must kill a semantic mutant.
10. Stop an implementation task if an unapproved cross-cutting decision is required.

For each review return:
- verdict;
- architectural compatibility;
- invariant checklist;
- tests and mutant quality;
- feature/no_std impact;
- compatibility/security concerns;
- exact required changes;
- merge dependency and next task.
```

---

## 2. Implementer prompt header for GPT-5.6

Prepend this to every task:

```text
You are implementing one bounded Incin compiler task.

Baseline:
- Use the exact origin/develop SHA recorded in docs/plan/compiler-program/STATUS.md.
- Work in a dedicated Git worktree and branch.

Read first:
- docs/plan/remediation/master-implementation-plan.md sections relevant to this task
- docs/plan/compiler-program/DECISIONS.md
- docs/plan/compiler-program/STATUS.md
- task specification
- existing modules named by the task

Hard rules:
1. Do not redesign cross-cutting APIs without an approved ADR.
2. Do not mark no-op, placeholder, simulated, or unused code complete.
3. Preserve no_std/default feature contracts.
4. No silent fallback, copy, cast, device change, readback, or synchronization.
5. Use checked arithmetic for shape, cardinality, bytes, offsets, and lengths.
6. Add positive, negative, and mutation-sensitive tests.
7. Run exact evidence commands and report concise outputs.
8. Keep the diff bounded to allowed files. Report out-of-scope needs.
9. Treat code and current test output as authoritative over ledger prose.
10. Do not weaken privacy/sealing to make tests easier.

Final response format:
A. Summary
B. Files changed
C. Public/API changes
D. Invariants preserved
E. Tests added
F. Semantic mutant and test that kills it
G. Commands run and outcomes
H. Performance/allocation observations
I. Known limitations
J. Follow-up task IDs
```

---

## 2A. Implementer prompt header for Gemini agent 1

```text
You are implementing one narrowly specified Incin issue. Do not redesign architecture.

Before editing:
1. Read the linked GitHub issue completely.
2. Read the exact ADR and reference implementation named by the issue.
3. Confirm the branch starts from the recorded green develop SHA.
4. Run the issue's baseline command and paste its result.

Execution rules:
- Change only the files listed by the issue unless you stop and request permission.
- Follow the supplied types/signatures/pseudocode literally.
- Preserve every listed invariant.
- Return Result for user/runtime failures; do not add unwrap/expect/panic.
- Do not add unsafe. If the task appears to require unsafe, stop and explain why.
- Add every positive, negative, and mutant test listed in the issue.
- A test is insufficient unless you can explain what incorrect implementation it rejects.
- Do not mark the ledger complete. The reviewer does that after evidence.

Before finishing:
- run fmt, targeted Clippy, targeted tests, and the exact evidence command;
- inspect git diff for unrelated formatting;
- provide a file-by-file summary and unresolved questions.

Stop conditions:
- required type or ADR does not exist;
- current source differs materially from the issue;
- implementation requires a public API change not written in the issue;
- a prerequisite test is already failing;
- there is no way to satisfy an invariant with the allowed files.
```

## 2B. Verification/documentation prompt header for Gemini agent 2

```text
You are extending tests, generated coverage, examples, or documentation for an already implemented Incin contract. You are not the architecture owner.

Inputs you must receive:
- linked issue and merged implementation SHA;
- one canonical passing example;
- exact support registry/schema;
- exact semantic mutant or failure cases;
- commands to execute.

Tasks:
1. Reproduce the canonical example unchanged.
2. Add the requested matrix/fixtures/examples by following the same pattern.
3. Add negative cases before updating documentation claims.
4. Run the semantic mutant and prove the new test fails for the intended reason.
5. Regenerate documentation/tables through repository tools; do not edit generated blocks by hand.
6. Compile/run all code examples.
7. Report unsupported combinations explicitly; never label skipped hardware as passing.

Do not:
- introduce a new public type;
- broaden feature support without executable evidence;
- silence a failing test;
- use placeholder output;
- rewrite unrelated prose or code.
```

# 3. Ready-to-run task prompts

## Task 0 — Truth reset and semantic regression tests

**Owner:** Opus 5, with GPT-5.6 implementing tests after ADR approval.

```text
Task ID: COMP-AUDIT-001

Objective:
Establish tests that prove the current compiled stack is not executable/semantically complete, then create the CMP2 task track without deleting historical task evidence.

Allowed files:
- docs/plan/compiler-program/*
- docs/plan/ledger.toml
- xtask ledger validation files if required
- new tests under crates/incin-core/tests/
- minimal test-only seams in compiled modules

Do not:
- implement canonical IR;
- implement IREE;
- rewrite the current compiled modules;
- change public APIs except test-only visibility approved by ADR.

Required tests/findings:
1. Capture a non-F32 graph and prove current captured form cannot retain dtype.
2. Capture two reductions with different axes and prove current captured form cannot distinguish them.
3. Compile a real non-empty input and prove generated guard is empty-shape/F32 placeholder.
4. Create a two-op pointwise chain and prove current fusion replacement does not represent both operations.
5. Serialize artifact and prove it contains no executable variant bytes/function ABI.
6. Prove current tuning score is synthetic rather than measured.

Do not write tests that permanently assert broken behavior as desired behavior. Use:
- defect characterization tests;
- tests against a new expected contract marked with a tracked blocker;
- or isolated mutant tests that fail until the owning task lands.

Deliver:
- ADR directory;
- CMP2/IREE/TORCH/DLPACK task rows;
- dependency graph;
- evidence policy;
- first three GPT-5.6 handoffs.
```

---

## Task 1 — Canonical IR foundation

```text
Task ID: CMP2-001A

Objective:
Add no_std-compatible canonical compiler IR identity, tensor types, symbolic dimensions, shape constraints, values, nodes, and functions without changing capture or execution.

Allowed files:
- crates/incin-core/src/compiled/ir/* (new)
- crates/incin-core/src/compiled/mod.rs
- crates/incin-core/src/compiled/error.rs (new if ADR specifies)
- crates/incin-core/tests/compiled_ir.rs
- compile-fail fixtures if needed

Required types:
- stable explicit FunctionId/BlockId/NodeId/ValueId/SymbolId/ParameterId;
- TensorType;
- ShapeExpr and bounded DimExpr;
- ShapeConstraint;
- IrValue/ValueOrigin/AliasInfo;
- IrNode shell with descriptor field deferred or abstracted exactly as ADR specifies;
- IrFunction/IrModule;
- structured VerifyError.

Invariants:
- IDs are independent of vector index;
- checked cardinality and byte calculations;
- no HashMap iteration affects canonical order;
- no std dependency;
- no unbounded recursive expressions;
- all public types documented.

Tests:
- valid minimal function;
- duplicate IDs;
- missing input/output;
- use-before-definition;
- invalid symbol;
- overflow cardinality;
- malformed affine range;
- deterministic canonical order/hash skeleton.

Mutant:
Change verifier to treat node ID as vector index. A test with sparse/non-monotonic IDs must fail.
```

---

## Task 2 — Canonical descriptors

```text
Task ID: CMP2-001B

Objective:
Create a type-erased canonical OpDescriptor that preserves all semantic fields from validated operation specs.

Depends on:
- CMP2-001A

Allowed files:
- compiled/ir/op.rs
- exec/spec.rs only for narrowly approved conversion helpers
- new descriptor modules approved by ADR
- compiled_ir tests

Initial variants:
- Pointwise
- Broadcast
- MatMul
- Reduction with reduction kind
- Reshape
- Conv2d
- Pool2d with pool kind/policies
- Transpose
- Cast

Required:
- conversion only from validated/witnessed specs;
- stable opcode IDs;
- explicit schema version;
- no stringly attributes in executable IR;
- descriptor/input/output verifier hooks.

Tests:
- two reductions with different axes/kinds are unequal and serialize/hash differently;
- conv stride/pad/dilation/groups retained;
- transpose permutation retained;
- dtype/cast retained;
- forged descriptor construction rejected by privacy or verifier.

Mutant:
Drop reduction axes from canonicalization/hash. Equality/hash test must fail.
```

---

## Task 3 — Session-scoped symbolic capture

```text
Task ID: CMP2-002A

Objective:
Replace process-global execution-backed tracing as the canonical compiler frontend with session-scoped symbolic capture that receives Validated<Spec>.

Depends on:
- CMP2-001B

Allowed files:
- compiled/capture.rs
- new compiled/capture/*
- tensor/tracing.rs only for delegation/deprecation
- capture backend modules
- tests

Required:
- CaptureContext owns builder and diagnostics;
- symbolic storage is ValueId + TensorMeta, not real allocation;
- parameters become Parameter records;
- module/source scope stack;
- no global graph;
- concurrent captures independent;
- no real device required;
- dtype, shape, attrs, descriptor, and names preserved.

First supported operations:
- six existing ShapeRule descriptor families;
- pointwise unary/binary;
- cast/transpose if descriptor exists.

Tests:
- no allocation/backend kernel call;
- concurrent capture;
- nested capture error;
- F16 retained;
- axes/stride/pad retained;
- shared parameter identity;
- deterministic snapshot.

Mutant:
Force F32 in symbolic output. F16 capture test must fail.
```

---

## Task 4 — Pytree and public capture contract

```text
Task ID: CMP2-002B

Objective:
Support tensor/tuple/array/derived-struct inputs and outputs and expose a stable internal capture API for Module<Input>.

Depends on:
- CMP2-002A

Required:
- TraceTree/TreeSpec or ADR-approved equivalent;
- tuple support to documented arity;
- derive macro or module-generated implementation for named structs;
- stable leaf ordering and names;
- input/output schema serializable into artifact manifest;
- useful compile errors for unsupported leaf types.

Tests:
- single tensor;
- tuple input/multi-output;
- nested struct/list equivalent supported by Rust types;
- mismatched runtime tree rejected;
- names retained.

Mutant:
Reverse tuple leaf order in runtime binder. Round-trip test must fail.
```

---

## Task 5 — Reference interpreter

```text
Task ID: CMP2-003

Objective:
Execute canonical IR through existing validated descriptor/native backend paths as a correctness oracle.

Depends on:
- CMP2-002B

Required:
- RuntimeValue with checked type/storage tags;
- input binding;
- parameter binding;
- topological execution;
- output extraction through witnessed construction;
- structured unsupported descriptor error;
- debug trace of value IDs and source paths.

Do not optimize beyond obvious dead-value release after correctness is proven.

Tests:
- MLP parity;
- batched matmul/broadcast parity;
- CNN subset parity;
- multiple outputs;
- wrong runtime dtype/shape;
- missing parameter;
- backend capability mismatch.

Mutant:
Swap matmul operands while retaining output shape. Numerical parity test must fail.
```

---

## Task 6 — Real guard program

```text
Task ID: CMP2-004

Objective:
Generate and execute guards for static, named, bounded, affine, divisibility, and product constraints.

Depends on:
- CMP2-001A

Required:
- compact GuardProgram;
- deterministic symbol binding;
- checked evaluator;
- structured GuardFailure;
- no stale artifact execution after failure;
- explain output.

Tests:
- exact static;
- named equality across inputs;
- range;
- affine relation;
- divisibility;
- reshape product;
- overflow;
- zero dimension;
- wrong rank/dtype.

Mutant:
Skip a bound check on cache hit. Cache-hit invalid-shape test must fail.
```

---

## Task 7 — Public reference compile API

```text
Task ID: CMP2-009A

Objective:
Expose the first complete same-model eager/reference-compiled path.

Depends on:
- CMP2-003
- CMP2-004

Public target:
CompileTarget::Reference

Required API:
- model.compile(context, input_spec, options);
- CompiledModel::run;
- typed input/output tree;
- compile report;
- no build.rs requirement;
- ordinary Module implementation unchanged.

Examples:
- MLP static;
- MLP dynamic batch;
- CNN subset.

Tests:
- examples compile under minimum feature set;
- eager/reference parity;
- guard failure;
- module train/eval state captured intentionally;
- parameters loaded after capture policy documented.

Mutant:
Return eager model directly instead of executing plan. Test must inspect plan step execution or inject backend spy to fail.
```

---

## Task 8 — IREE toolchain and emitter

```text
Task IDs: IREE-001 and IREE-002A

Objective:
Add isolated optional IREE crate, exact toolchain probing, and deterministic StableHLO module/type emission.

Allowed files:
- new crates/incin-iree/*
- workspace Cargo.toml
- facade feature forwarding
- generated docs/feature inventory

Required:
- no IREE dependency in default build;
- explicit path/env/PATH/Python-package discovery;
- exact compiler version probe;
- target support probe;
- deterministic MLIR names/types/functions;
- source map side table;
- golden tests;
- no shell interpolation.

Do not yet execute VMFB.

Mutant:
Omit a dynamic dimension marker or dtype in emitted type. Golden/parse test must fail.
```

---

## Task 9 — Tier-0 StableHLO lowering

```text
Task ID: IREE-002B/C

Objective:
Lower pointwise, reshape/transpose/broadcast, reductions, matmul/bmm, concat, comparison/select, cast, and softmax to StableHLO/standard MLIR.

Required:
- registry-driven lowering;
- structured SupportDecision;
- source mapping;
- emitted module verifier/iree-compile smoke test when tool is present;
- decomposition IDs recorded.

Tests:
- one golden per op family;
- dynamic dimensions;
- dtype matrix;
- unsupported rank/layout/dtype diagnostics;
- reference vs IREE later-ready fixtures.

Mutant:
Emit sum for mean without division. IR/reference semantic test must fail.
```

---

## Task 10 — IREE subprocess compiler and runtime

```text
Task IDs: IREE-003, IREE-004A

Objective:
Compile emitted MLIR to VMFB and execute a typed function through a reusable IREE engine.

Required compiler host:
- timeout;
- bounded output;
- atomic file result;
- exact command/flags/version report;
- source-map diagnostic translation;
- cache-ready result.

Required runtime:
- reusable engine/device;
- module cache;
- function lookup;
- typed input/output validation;
- parameter provider;
- structured runtime errors;
- explicit synchronization.

First target:
- LLVM CPU.

Tests:
- MLP end-to-end;
- repeated run reuses module;
- wrong ABI;
- compiler failure maps node/source;
- runtime target mismatch;
- output parity.

Mutant:
Reload module every run. Spy/counter test must fail performance contract.
```

---

## Task 11 — Artifact v2

```text
Task ID: CMP2-008

Objective:
Replace JSON-plan artifact with a sectioned executable container.

Required sections:
- manifest;
- parameter index;
- VMFB variant;
- optional parameters;
- source map/debug IR;
- reproducibility.

Required:
- checked offsets/lengths;
- BLAKE3/SHA-256 section/root digests;
- schema and runtime ABI IDs;
- target/compiler metadata;
- typed input/output tree;
- dynamic constraints;
- atomic writes;
- bounded parser.

Tests:
- round-trip executable artifact;
- byte corruption;
- truncation;
- duplicate section;
- oversized length;
- unknown optional/required section;
- compiler/runtime mismatch;
- parameter checksum mismatch.

Mutant:
Skip VMFB digest validation. Corruption test must fail.
```

---

## Task 12 — Dynamic buckets and cache

```text
Task IDs: DYN-001, DYN-002, CACHE-001

Objective:
Implement explicit Exact/Guarded/Bucketed/Generic/Recompile policies and bounded single-flight compiler cache.

Required:
- shape signature;
- bucket selection;
- cache key completeness;
- guard before dispatch;
- bounded LRU/bytes;
- stampede prevention;
- stale lock recovery;
- corruption quarantine;
- compile budget;
- explain hit/miss/recompile.

Tests:
- dynamic batch/sequence;
- smallest compatible bucket;
- generic overflow bucket;
- concurrent identical compile once;
- invalid shape never uses cached VMFB;
- toolchain/device fingerprint invalidation.

Mutant:
Remove compiler version from key. Version-invalidation test must fail.
```

---

## Task 13 — Typed state archive and PyTorch naming

```text
Task IDs: TORCH-001..004

Objective:
Make safetensors/state-dict loading dtype-correct, device-aware, strict, and PyTorch-friendly.

Required:
- StateArchive/StateTensor;
- Exact/CastFloating/CastCompatible policy;
- strict load report;
- key mapping/prefix/regex/callback;
- collision rejection;
- destination device report;
- shared parameter handling.

Tests:
- F16/BF16/F32 weights;
- shape mismatch;
- missing/unexpected;
- key rename collision;
- CPU/GPU destination;
- sequential PyTorch naming parity;
- round-trip safetensors.

Mutant:
Load all tensors through FloatElem irrespective of source dtype. Dtype test must fail.
```

---

## Task 14 — DLPack CPU/CUDA

```text
Task IDs: DLPACK-001, DLPACK-002

Objective:
Implement safe zero-copy interchange with PyTorch-compatible DLPack protocol.

Required:
- versioned and legacy managed tensor wrappers;
- RAII owner/deleter;
- dtype/device/shape/stride/offset validation;
- CPU zero-copy;
- __dlpack__/__dlpack_device__ in Python wrapper;
- CUDA stream handoff/event ordering;
- explicit copy policy;
- no hidden host fallback.

Tests:
- ownership after original deletion;
- shared CPU memory;
- strided/offset/zero-size;
- dtype matrix;
- same/different CUDA streams;
- wrong device;
- explicit copy;
- legacy capsule consumed once.

Mutant:
Drop producer owner immediately after import. Lifetime test under stress/sanitizer must fail.
```

---

## Task 15 — PyTorch backend beta

```text
Task ID: TORCH-BACKEND-001

Objective:
Publish a Python package registering `torch.compile(backend="incin")` for inference through the canonical Incin IR.

Required:
- package entry point registration;
- FX/example input backend contract;
- normalized export-to-canonical-IR flow;
- Incin artifact cache/container;
- DLPack input/output;
- pytree preservation;
- graph break and compile report;
- source-rich Python exceptions;
- MLP/CNN/transformer tests.

The backend must use canonical Incin IR. The artifact manifest records source frontend, exporter version, operator set, constraints, and compiler/runtime versions.

Tests:
- torch.compile string backend registration;
- repeated callable cache reuse;
- dynamic batch where supported;
- output parity;
- unsupported op diagnostic;
- minifier-compatible registration import.

Mutant:
Convert inputs through CPU NumPy. Device/copy telemetry test must fail.
```

---

## Task 16 — Canonical `torch.export` importer

```text
Task ID: TORCH-EXPORT-001

Objective:
Import ExportedProgram normalized ATen graph and constraints into canonical Incin IR.

Required:
- versioned bounded bridge schema;
- ATen registry generated from one source;
- graph signature and pytree;
- parameters/buffers/constants externalized;
- source/module stack;
- dynamic constraints;
- decomposition provenance;
- support report.

Initial ATen coverage:
- pointwise;
- view/reshape/permute;
- reductions;
- mm/bmm/addmm;
- convolution;
- pooling;
- layer norm;
- embedding/gather;
- softmax/attention decomposition.

Tests:
- exported MLP/CNN/transformer;
- named/affine dynamic dims;
- unsupported relation;
- parameter mapping;
- reference/IREE parity.

Mutant:
Ignore one export range constraint. Invalid-input test must fail.
```

---

## Task 17 — AOTAutograd

```text
Task ID: TORCH-AOT-001

Objective:
Compile PyTorch forward/backward graphs through the Incin backend using AOTAutograd; keep optimizer eager initially.

Required:
- forward and backward compiler hooks;
- boxed callable contract;
- saved tensor ownership;
- gradient pytree;
- failure propagation;
- dtype/shape/stream correctness;
- exact decomposition/version record.

Tests:
- linear regression step;
- MLP;
- CNN;
- transformer block;
- eager PyTorch gradient parity;
- no use-after-free;
- dynamic batch where supported.

Mutant:
Release a saved activation after forward. Backward lifetime test must fail.
```

---

# 4. Integration/review prompt

```text
Review this task branch against origin/develop and the approved ADRs.

Required review procedure:
1. Read task specification and evidence.
2. Inspect diff for out-of-scope changes.
3. Verify no duplicate semantic source of truth was introduced.
4. Check no_std/default feature isolation.
5. Run or inspect exact commands.
6. Temporarily apply the declared semantic mutant and confirm a test fails.
7. Check error remediation quality.
8. Check checked arithmetic and bounded parsing/allocation.
9. Check cache/artifact identity completeness where relevant.
10. Check that tests execute nonzero cases on hardware-gated paths.

Return:
- Approve / request changes / reject;
- blocking findings;
- non-blocking follow-ups;
- merge order;
- ledger status recommendation;
- next task prompt updated to integrated SHA.
```

---

# 5. Daily operating cadence

1. Architecture owner updates `STATUS.md` with integrated SHA and task owners.
2. Each implementation agent rebases its worktree before coding.
3. Agent works on one bounded task and records commands.
4. Architecture owner reviews the semantic mutant before ordinary style review.
5. Merge only after dependency tasks are integrated.
6. Run focused tests on PR, full workspace/nightly matrix after merge.
7. Update support registry and generated docs in the same PR that changes support.
8. Rebaseline performance only with environment metadata and explicit approval.
9. End day with `HANDOFF.md`: merged SHA, active branches, blockers, next exact prompts.

---

# 6. Program stop conditions

Pause and request architecture review if any task discovers:

- a required cycle between core/compiler/backend crates;
- need to make a sealed validation constructor public;
- need to put std/IREE dependencies in default core;
- inability to preserve dtype/layout/shape metadata;
- hidden host/device copy required for correctness;
- dynamic constraint beyond the approved bounded algebra;
- artifact ABI change;
- unsafe DLPack/CUDA lifetime not covered by a written ownership proof;
- unsupported IREE target semantics;
- performance result that conflicts with public claims;
- test suite that cannot kill the declared mutant.



# Appendix F — Current official technical references used for external integration choices

Repository findings in this document come from the supplied source archive. The following external design choices were checked against official documentation current at the time of writing. Pin actual versions in implementation and treat experimental APIs as unstable.

- PyTorch custom compiler backends: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_custom_backends.html
- PyTorch `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch `torch.export` API and dynamic shape constraints: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html
- IREE API binding status: https://iree.dev/reference/bindings/
- IREE deployment targets: https://iree.dev/guides/deployment-configurations/
- IREE compiler driver and StableHLO input examples: https://iree.dev/developers/general/developer-overview/
- ONNX IR graph/type/initializer/attribute requirements: https://onnx.ai/onnx/repo-docs/IR.html
- ONNX shape inference: https://onnx.ai/onnx/repo-docs/ShapeInference.html
- Hugging Face safetensors repository metadata/shard index: https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
- NumPy DLPack exchange: https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html

# Appendix G — Final coordinator checklist

Before assigning any implementation task, verify:

- [ ] the issue exists and contains the exact acceptance criteria;
- [ ] prerequisites are merged on green `develop`;
- [ ] one implementation agent is assigned;
- [ ] the agent has read this document, `PROPOSALS.md`, relevant task documents, and current source;
- [ ] baseline commands for the affected crate are recorded;
- [ ] public API impact is explicitly stated;
- [ ] no task relies on a placeholder graph or simulated timing;
- [ ] negative tests and semantic mutants are named before code changes;
- [ ] unsafe, allocations, cache, serialization, and backward-compatibility impacts are considered;
- [ ] documentation changes are part of the same issue when behavior is public;
- [ ] merge evidence is independent of agent self-report;
- [ ] the next sequential issue is not started until the current PR is merged or formally abandoned.
