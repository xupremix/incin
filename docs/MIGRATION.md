# Incin Core 0.1.0 Migration Guide

This document outlines the API changes, stabilization milestones, and migration pathways introduced across `incin-core` and `incin-backends`.

---

## Overview of Core Stabilization (`REL-001`)

The `0.1.0` core stabilization milestone establishes key architectural invariants:

1. **Proof-Carrying Shapes (`SHP-001`..`SHP-008`):** Shape transformations are verified at compile time via `s!` and `idx!` or checked at runtime with structured error returns. Backend panics on invalid shapes are eliminated.
2. **Decoupled Execution Architecture (`EXE-006`..`EXE-009`):** The monolithic 254-method backend supertrait and unsupported-operation fallback surface are removed. Execution targets explicitly implement `StorageBackend`, `Execute<Op>`, and `Capabilities`.
3. **Unified Autograd Engine (`GRD-001`..`GRD-006`):** Backend-local autograd tapes are replaced by the core graph engine (`incin_core::exec::tape`). Saved tensor lifetimes are owned by the graph, and workspace-wide `TensorId` guarantees global identification across devices.
4. **Checked Precision and Allocation (`EXE-008`):** F32-hardcoded byte widths are replaced with checked `DTypeId::size_bytes()`, supporting multi-byte and quantized formats safely.

## Release-readiness migration (0.1.0)

These are the changes a reader upgrading from a `0.0.0` development snapshot
has to act on.

### Tensor operator syntax is now the panic-on-error convenience boundary

`+`, `-`, `*`, `/`, tensor-scalar forms, and unary `-` now produce tensors
directly, so remove `?` around operator expressions. The same operations stay
recoverable through `try_add`, `try_sub`, `try_mul`, `try_div`, `try_neg`, and
the scalar named methods. Use those named methods at library and application
boundaries that must handle dynamic-shape, device, or backend failures. An
operator failure panics with a fixed, operator-only message and never includes
tensor contents or backend diagnostic text.

### `protoc` is no longer a build dependency

`incin-core` had a build script that ran `prost-build` on every build, which
made a system protobuf compiler mandatory for every crate that depended on the
facade, including the overwhelming majority that never call the ONNX
exporter. The generated module is checked in at
`crates/incin-core/src/generated/onnx.rs` instead. Nothing changes for callers;
the ONNX API is identical. Maintainers regenerate with `cargo xtask onnx` and
`cargo xtask onnx --check` verifies the checked-in file against
`proto/onnx.proto` in CI.

`incin-macros` no longer depends on `onnx-pb`, which was unreleased since 2020
and pinned a second `prost` major into the tree.

### `AutogradBackend::set_grad` is required (backend authors)

`AutogradBackend` gained a required method:

```rust
fn set_grad<K: DType>(
    storage: &Self::Storage<K>,
    grads: &mut Self::Grads,
    value: Self::Storage<K>,
) -> Result<()>;
```

It exists so that post-backward transforms which rescale a whole gradient set (`clip_grad_norm` and `clip_grad_value` are the two in tree) can be written once against the trait
rather than once per backend. It is a replacement, not an accumulation; the
reverse walk's own accumulation has finished before anything calls it.

It is required rather than defaulted on purpose. A default that silently
dropped the value would turn clipping into a no-op, and a caller cannot tell a
no-op rescale from a rescale by one. A backend with no gradient map should
return `Error::UnsupportedBackendOperation` rather than `Ok(())`.

### An optimizer step that reaches no parameter is an error

`SGD`, `Adam`, and `AdamW` previously skipped a parameter they had no gradient
for and returned `Ok(())` regardless. Skipping *some* parameters is still
legal (a parameter the forward pass did not use has nothing to apply), but
skipping *every* parameter in a non-empty group now returns
`Error::InvalidModuleState`.

That state means the backward pass did not reach the group: it was never run,
the graph was detached, or the forward pass was recorded on a different thread
from the `backward` call, since a tape is thread-local and the reverse walk on
another thread drains an empty graph. Each of those used to produce a training
loop that ran to completion with parameters that never moved.

One consequence is worth knowing even if you never hit the error: **a
`Gradients` value is spent by the step that consumes it.** A committed step
assigns fresh storage to every parameter it updates, and gradients are looked
up by the identity of the storage they were recorded against, so a second step
against the same `Gradients` matches nothing. Recompute gradients per step.

### `incin::hub` is behind the `data-hub` feature

The Hugging Face Hub client pulled an async runtime and a second TLS stack into
the dependency graph of every crate that depended on the facade. It is now
opt-in:

```toml
incin = { version = "0.1.0", features = ["data-hub"] }
```

Dataset downloading does not need it: that is `incin-data`'s `download`
feature, which is on by default.

### A non-default loss reduction is built with `with_reduction()`

```rust,ignore
use incin::nn::Sum; // Mean, Sum, and NoneReduction live here, not the prelude

// before
let loss = MSELoss::<Sum>::new().forward(&pred, &target)?;
// after
let loss = MSELoss::<Sum>::with_reduction().forward(&pred, &target)?;
```

`MSELoss`, `CrossEntropyLoss`, `L1Loss`, and `BCEWithLogitsLoss` each declare
`R: ReductionMode = Mean`, but a type-parameter default does not drive
inference for an associated function, so `MSELoss::new()` failed with `E0283`
and every call site had to write `MSELoss::<Mean>::new()`. `new()` is now
defined on the `Mean` instantiation alone, so it resolves by itself and reads
like `torch.nn.MSELoss()`. The explicit `MSELoss::<Mean>::new()` form is
unchanged; only a non-default reduction has to be renamed.

### `ModelExt::load` no longer takes a device

```rust,ignore
// before
model.load(Format::Safetensors, path, &DeviceId::cpu())?;
// after
model.load(Format::Safetensors, path)?;
```

The argument was ignored. `load` restores state in place and leaves every
parameter where it already lives, so a call that passed a different device was
reading as a relocation that never happened. Moving a model between devices is
`ToDevice`, which is explicit and returns the relocated model.

### State files carry a format version

Both state formats now record `STATE_FORMAT_VERSION` (currently `1`):
safetensors under the `incin.format.version` metadata key, postcard as the
first field of its envelope. Reading refuses a file whose version is newer than
the build, naming both numbers.

This is breaking for files written by a `0.0.0` snapshot, which carry no
version: loading one now fails with a message saying so. Re-save it with a
build of this version. Foreign safetensors files (a Hugging Face checkpoint,
for instance) were never loadable through `ModelExt::load` and are unaffected;
`import_model!` reads those and does not look for the key.

The sharded-checkpoint manifest has always carried a `version` field but never
checked it on load. It does now, against `CHECKPOINT_MANIFEST_VERSION`.

### Dependency upgrades

`rand` 0.8 → 0.10, `rand_distr` 0.4 → 0.6, `hashbrown` 0.14 → 0.17, `spin` 0.9
→ 0.12, `safetensors` 0.4 → 0.8, `pollster` 0.3 → 1.0, and `criterion` 0.5 →
0.8. These are internal; the facade API is unaffected.

### Minimum supported Rust version

`rust-version = "1.88"`, verified by a CI job pinned to exactly that toolchain
rather than asserted. 1.87 is refused by the dependency graph.

## R1 execution-policy migration

`ExecutionPolicy` now defaults to `FallbackPolicy::AllowComposition`: operations
advertised as `Composed` may execute on the same device by default, while
`Fallback` still requires the explicit `AllowTransfer` policy. Every canonical
dispatch route checks this policy before calling `Execute<O>` and reports a
typed `CanonicalError::Policy(PolicyViolation)` when it refuses an otherwise
supported invocation.

`AllocatorPolicy` and the general determinism setting have been removed from
`ExecutionPolicy` and `ExecutionContext`, including their builders and
accessors. Backend tuning continues to use its separate `Determinism` type;
it was not a general execution guarantee.
5. **Typed Distributed Topologies (`DST-001`..`DST-005`):** Logical device meshes (`ValidMesh`) and placement typestates (`Replicated`, `Sharded`, `Partial`, `PipelineStage`) enforce distributed execution contracts before launch.

### Custom logical dtypes and tensor elements

`DType` and `ConstDType` remain extensible for downstream logical dtypes that
provide their own `DTypeDescriptor`; no `DTypeId` variant is required. The
`TensorElement` marker is intentionally different: its implementation set is
sealed to Incin's built-in scalar element types. Downstream code must not
implement `TensorElement` for a custom Rust type. Block-quantized logical dtypes
such as `Q8_0` likewise do not implement `PlainDType` or `TensorElement`.

---

## Breaking Changes & Migration Pathways

### Public facade tiers

The stable `incin` facade keeps model-building and tensor ergonomics in its
root and prelude. Backend extension contracts are explicitly namespaced and
feature-gated. Compiler inspection and tuning are preview surfaces, and test
backends are never part of a default build.

| Old path | New path | Feature | Tier | Replacement | Breaking |
|---|---|---|---|---|---|
| `incin::Backend` | `incin::backend_authoring::Backend` | `backend-authoring` | backend authoring | `use incin::backend_authoring::Backend;` | yes |
| `incin::VariableBackend` | `incin::backend_authoring::VariableBackend` | `backend-authoring` | backend authoring | Import the trait beside the backend implementation | yes |
| `incin::prelude::Backend` | `incin::backend_authoring::Backend` | `backend-authoring` | backend authoring | Keep backend bounds out of ordinary model code | yes |
| `incin_core::prelude::Backend` | `incin_core::backend_authoring::Backend` | none | backend authoring | Import the trait beside the backend implementation | yes |
| `incin_core::prelude::StorageBackend` | `incin_core::backend_authoring::StorageBackend` | none | backend authoring | Import the storage contract explicitly | yes |
| `incin_core::prelude::TracingBackend` | `incin_core::tensor::tracing::TracingBackend` | none | graph inspection | Use the named tracing module only in capture code | yes |
| `incin_core::prelude::StorageEncoding` | `incin_core::tensor::dtype::StorageEncoding` | none | expert dtype API | Import storage layout metadata explicitly | yes |
| `incin_core::prelude::Graph` | `incin_core::graph::Graph` | `std` as applicable | graph inspection | Import `Graph` only in capture/compiler code | yes |
| `incin::advanced::ReshapeTargetSpec` | no public replacement | none | implementation detail | Use `ReshapeTarget` and the `idx!` macro | yes |
| `incin::advanced::SliceSpec` | no public replacement | none | implementation detail | Use `SliceTarget` and the `idx!` macro | yes |
| `incin_core::prelude::export_to_onnx` | `incin_core::onnx::export_to_onnx` | `std` | graph interchange | Import the named ONNX module | yes |
| `incin::experimental::compiled::*` | `incin::experimental::compiled::*` | `compiled` | preview | Enable `compiled` and use the documented inspection types | no, when already namespaced |
| tuning service internals | `incin::experimental::tuning::*` | `autotune` | preview | Use the documented policy and explanation types | yes |
| `incin::test_utils::DummyBackend` | removed | - | test-only | Use a real backend; `incin::test_utils` now holds only fault injection | yes |

The `Dyn` marker remains a normal public type. Its constructor and proof
invariants are covered by the later API-002 constructor audit.

### 1. Backend Trait & Dispatch Migration

#### Old Pattern
In early snapshots, backends implemented a monolithic trait containing all tensor operations with blanket default panics:
```rust
// Obsolete: Monolithic adapter with panic-on-unsupported defaults
let backend = CpuBackendImpl::new();
backend.add(&a, &b);
```

#### New Pattern
Operations are modularized around sealed operation descriptors (`Execute<OpSpec>`) and checked capabilities:
```rust
use incin_core::exec::{Capabilities, Execute};
use incin_backends::cpu::CpuBackendImpl;

let backend = CpuBackendImpl::default();
// Capabilities query before execution
if backend.capabilities().supports_dtype(DTypeId::F32) {
    // Operation execution routed through sealed descriptor validator
}
```

---

### 2. Autograd Tape & Gradient Queries

#### Old Pattern
Each backend maintained a separate thread-local tape with its own `TensorId` and panic-on-error reverse walk:
```rust
// Obsolete: Backend-local tape and un-checked backward pass
let grads = cpu::tape::backward(&loss)?;
let grad = grads.get(tensor_id);
```

#### New Pattern
A single workspace-wide `TensorId` counter and unified graph engine (`incin_core::exec::tape`) manage backward execution across CPU, WGPU, and CUDA:
```rust
use incin_core::exec::tape;
use incin_backends::cpu::backward;

// Unified backward pass using GradientMap
let grads = backward(&loss)?;
if let Some(grad_storage) = grads.get(tensor_id) {
    // Inspect accumulated gradient
}
```

---

### 3. Grad Mode & Fallible Policy Controls

#### Old Pattern
Disabling gradient tracking or checking non-finite values required ad-hoc backend calls or separate entry points.

#### New Pattern
Ambient execution policies govern gradient recording (`GradMode`) and non-finite value detection (`NanPolicy`):
```rust
use incin_core::exec::{check_gradients, GradMode};

// Temporarily disable gradient recording (records 0 tape nodes)
GradMode::Disabled.scope(|| {
    // Forward pass with zero autograd overhead
});

// Enable NaN / non-finite gradient checks without panicking
let result = check_gradients(|| backward(&loss));
```

---

### 4. Quantization & Checked Byte Lengths

#### Old Pattern
Buffer allocation used hardcoded `* 4` (F32 assumption) or `size_of::<f32>()`.

#### New Pattern
All allocations and storage length computations route through `DTypeId::size_bytes()`:
```rust
use incin_core::prelude::{DTypeId, OperationKind};

let byte_len = DTypeId::Q8_0.size_bytes(num_elements, OperationKind::Storage)?;
```

---

### 5. Environment & System Diagnostics

Use `cargo incin doctor` or `incin::doctor::probe()` to verify active toolchains, compiled backend features, CPU SIMD features, and device capabilities:

```bash
cargo incin doctor
cargo incin doctor --json
```

---

### 6. Feature Naming & Deprecations (`REL-002`)

#### Old Pattern
The third-party Candle adapter feature was aliased as `candle`.

#### New Pattern
The deprecated `candle` alias is removed. Use explicit `external-candle` in `Cargo.toml`:
```toml
incin = { version = "0.1.0", features = ["external-candle"] }
```

---

### 7. Compiled Graph Subsystem (`CMP-001`..`CMP-006`)

#### Overview

The `compiled` feature is a preview-only CPU reference-evaluation and plan
inspection surface. Facade users must access it through
`incin::experimental::compiled`; no compiled types are part of the root or
prelude compatibility baseline. It is not a stable compiler, deployment target,
or portable artifact ABI.

Lower-level `incin-core` exposes the implementation pipeline under
`incin_core::compiled`:

| Component | Module | Purpose |
|-----------|--------|---------|
| `CapturedGraph` / `CapturedNode` | `compiled::capture` | Captures an eager `Graph` IR for offline analysis |
| `CompiledPlan` / `CompileOptions` | `compiled::plan` | Immutable compiled plan with input guards |
| `ShapeGuard` / `DynamicShapePolicy` | `compiled::plan` | Runtime dtype/shape verification |
| `ConstantFolder` / `WeightPrepacker` / `ShapeBucket` | `compiled::fold` | Inspection/prototype pass types; folding and prepacking have no executable lowering and fail closed |
| `LivenessMap` / `AllocationPlanner` / `MemoryPlan` | `compiled::alloc` | Per-value liveness analysis and buffer slot assignment |
| `SavedTensorSet` | `compiled::alloc` | Extends liveness for autograd-saved tensors |
| `FusionPass` / `FusionCandidate` / `FusedKernel` | `compiled::fusion` | Inspection/prototype fusion types; executable fusion is unavailable and fails closed |
| `CompiledArtifact` / `ArtifactVersion` / `ArtifactHeader` | `compiled::artifact` | Preview plan snapshots with integrity checks; not a deployment format |

These preview types are available from `incin_core::experimental::compiled` to
lower-level users and `incin::experimental::compiled` through the facade. They
are intentionally absent from both stable preludes.

#### Graph Capture and Compilation

```rust
use incin::experimental::compiled::{CapturedGraph, CompileOptions, CompiledPlan, DTypeId, Graph};

fn inspect_input_only_plan() -> incin::Result<CompiledPlan> {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    graph.mark_input(x);
    graph.mark_output(x);
    // A non-empty CPU plan also needs the canonical descriptor payload. The
    // facade-only Relu invocation is exercised in
    // `crates/incin/tests/consumer-fixtures/experimental-compiled-pass`.

    let captured = CapturedGraph::capture(&graph)?;
    CompiledPlan::compile(captured, CompileOptions::new())
}
```

#### Liveness and Allocation Planning

```rust
let liveness = LivenessMap::compute(&plan.graph);
let planner = AllocationPlanner;
let memory_plan = planner.plan(&liveness, &plan.graph)?;
println!("Peak live slots: {}", memory_plan.peak_live_slots);
```

#### Autograd-Aware Liveness (GRD-007)

```rust
let mut liveness = LivenessMap::compute(&plan.graph);
let mut saved = SavedTensorSet::new();
saved.save(activation_id);
liveness.extend_for_saved_tensors(&saved, backward_end_node);
```

#### Fusion and prepacking

`FusionPass`, `ConstantFolder`, and `WeightPrepacker` remain inspection and
prototype types. The CPU reference evaluator rejects fusion, folding, and
prepacking requests without an executable lowering rather than silently
claiming an optimization.

#### Artifact Serialization

```rust
let version = ArtifactVersion::new(0, 1, 0);
let artifact = CompiledArtifact::new(plan, version.clone(), "my_model".into())?;
let bytes = artifact.serialize()?;

// Reload with integrity + compatibility verification
let loaded = CompiledArtifact::load(&bytes, &version)?;
```

Snapshots are accepted only when their artifact format and the
caller-supplied compatibility major/minor values match the requested version.
Patch values may differ. This is a local preview compatibility policy; it does
not verify the running framework version or promise a portable artifact ABI.

---
