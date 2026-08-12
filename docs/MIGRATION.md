# Incin Core 0.1.0 Migration Guide

This document outlines the API changes, stabilization milestones, and migration pathways introduced across `incin-core` and `incin-backends`.

---

## Overview of Core Stabilization (`REL-001`)

The `0.1.0` core stabilization milestone establishes key architectural invariants:

1. **Proof-Carrying Shapes (`SHP-001`..`SHP-008`):** Shape transformations are verified at compile time via `s!` and `idx!` or checked at runtime with structured error returns. Backend panics on invalid shapes are eliminated.
2. **Decoupled Execution Architecture (`EXE-006`..`EXE-009`):** The monolithic 254-method backend supertrait and unsupported-operation fallback surface are removed. Execution targets explicitly implement `StorageBackend`, `Execute<Op>`, and `Capabilities`.
3. **Unified Autograd Engine (`GRD-001`..`GRD-006`):** Backend-local autograd tapes are replaced by the core graph engine (`incin_core::exec::tape`). Saved tensor lifetimes are owned by the graph, and workspace-wide `TensorId` guarantees global identification across devices.
4. **Checked Precision and Allocation (`EXE-008`):** F32-hardcoded byte widths are replaced with checked `DTypeId::size_bytes()`, supporting multi-byte and quantized formats safely.
5. **Typed Distributed Topologies (`DST-001`..`DST-005`):** Logical device meshes (`ValidMesh`) and placement typestates (`Replicated`, `Sharded`, `Partial`, `PipelineStage`) enforce distributed execution contracts before launch.

---

## Breaking Changes & Migration Pathways

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
incin = { version = "0.0.0", features = ["external-candle"] }
```

---

### 7. Compiled Graph Subsystem (`CMP-001`..`CMP-006`)

#### Overview

`incin-core` now exposes a compiled-graph pipeline under `incin_core::compiled`:

| Component | Module | Purpose |
|-----------|--------|---------|
| `CapturedGraph` / `CapturedNode` | `compiled::capture` | Captures an eager `Graph` IR for offline analysis |
| `CompiledPlan` / `CompileOptions` | `compiled::plan` | Immutable compiled plan with input guards |
| `ShapeGuard` / `DynamicShapePolicy` | `compiled::plan` | Runtime dtype/shape verification |
| `ConstantFolder` / `WeightPrepacker` / `ShapeBucket` | `compiled::fold` | Constant folding and shape bucketing passes |
| `LivenessMap` / `AllocationPlanner` / `MemoryPlan` | `compiled::alloc` | Per-value liveness analysis and buffer slot assignment |
| `SavedTensorSet` | `compiled::alloc` | Extends liveness for autograd-saved tensors |
| `FusionPass` / `FusionCandidate` / `FusedKernel` | `compiled::fusion` | Safe pointwise kernel fusion pass |
| `CompiledArtifact` / `ArtifactVersion` / `ArtifactHeader` | `compiled::artifact` | Versioned, integrity-checked artifact serialization |

All types are re-exported through `incin_core::prelude`.

#### Graph Capture and Compilation

```rust
use incin_core::prelude::*;
use incin_core::graph::Graph;
use incin_core::prelude::OperationKind;
use std::collections::BTreeMap;

let mut graph = Graph::new();
let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
graph.mark_input(x);
graph.mark_output(y);
graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());

let captured = CapturedGraph::capture(&graph)?;
let plan = CompiledPlan::compile(captured, CompileOptions::new());
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

#### Kernel Fusion

```rust
let pass = FusionPass;
let candidates = pass.find_candidates(&plan.graph);
let (fused_graph, kernels) = pass.apply(&plan.graph, &candidates)?;
println!("Fused {} kernel chains", kernels.len());
```

#### Artifact Serialization

```rust
let version = ArtifactVersion::new(0, 1, 0);
let artifact = CompiledArtifact::new(plan, version.clone(), "my_model".into())?;
let bytes = artifact.serialize()?;

// Reload with integrity + compatibility verification
let loaded = CompiledArtifact::load(&bytes, &version)?;
```

---
