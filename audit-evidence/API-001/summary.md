# Audit Evidence Summary: API-001 — Replace wildcard facade exports

**Task ID:** API-001  
**Task Name:** Replace wildcard facade exports  
**Priority:** P0  
**Audit Spec Reference:** Section 5.1, Section 6, Section 7 (API-001)

---

## 1. Current Source Behavior

### File and Symbol References

1. **`crates/incin/src/lib.rs` (line 87)**:
   - `pub use incin_backends::*;`
   - Re-exports the entire `incin_backends` crate at the root of `incin`.
2. **`crates/incin/src/lib.rs` (lines 290-292)**:
   - `pub use incin_backends::prelude::*;`
   - `pub use incin_core::prelude::*;`
   - Glob imports both backend and core preludes into `incin::prelude`.
3. **`crates/incin-backends/src/lib.rs` (line 6)**:
   - `pub use incin_core::prelude::*;`
   - Re-exports the entire core prelude at the root of `incin-backends`.
4. **`crates/incin-core/src/lib.rs` (lines 45-50, 78-90)**:
   - Re-exports compiler internal types (`AllocationPlanner`, `CapturedGraph`, `CapturedNode`, `ConstantFolder`, `FusedKernel`, `FusionPass`, `LivenessMap`, `MemoryPlan`, etc.) in `incin_core::prelude`.
   - Re-exports autoref fallback traits (`AutorefNamedLayers`, `AutorefNamedLayersFallback`, `AutorefParameters`, `AutorefParametersFallback`, etc.) and `ComputeStats` fallback traits in `incin_core::prelude`.
5. **`crates/incin-core/src/tensor/mod.rs` (lines 27-39)**:
   - `pub use super::arg::*;`, `pub use super::arg_into::*;`, `pub use super::auto_device::*;`, `pub use super::backend::*;`, `pub use super::base::*;`, `pub use super::conv2d::*;`, `pub use super::device::*;`, `pub use super::dtype::*;`, `pub use super::grad::*;`, `pub use super::matmul::*;`, `pub use super::tracing::*;`
   - Globs internal tensor argument conversion and execution details into `tensor::prelude`.

---

## 2. Inventory of Transitive Wildcard Exports

Names entering `incin` through wildcard/transitive re-exports:
- All items in `incin_backends`: `backend_kind`, `capability`, `capability_docs`, `codegen`, `dispatch`, `dist`, `detect`, `iteration`, `simd`, `tuning`, `cpu`, `cuda`, `wgpu`, `metal`, `external`, `telemetry`, `BackendFor`, `DispatchBackend`, `IncinBackend`, `simd_lanes`, `detect_device`, `detect_device_in`, `set_emitter`.
- Transitive core prelude items from `incin_backends` root.
- Internal compiler types: `AllocationPlanner`, `ArtifactHeader`, `ArtifactVersion`, `BufferSlot`, `CapturedGraph`, `CapturedNode`, `CompileOptions`, `CompiledArtifact`, `CompiledPlan`, `ConstantFolder`, `DynamicShapePolicy`, `FusedKernel`, `FusionBlocker`, `FusionCandidate`, `FusionPass`, `FusionPolicy`, `LivenessInterval`, `LivenessMap`, `MemoryPlan`, `SavedTensorSet`, `ShapeBucket`, `ShapeGuard`, `WeightPrepacker`.
- Autoref fallback helper traits: `AutorefNamedLayers`, `AutorefNamedLayersFallback`, `AutorefParameters`, `AutorefParametersFallback`, `AutorefShapeInfo`, `AutorefShapeInfoFallback`, `AutorefStateDict`, `AutorefStateDictFallback`, `AutorefTrainMode`, `AutorefTrainModeFallback`, `AutorefComputeStats`, `AutorefComputeStatsFallback`.
- Graph IR internals: `Graph`, `OpType`.

---

## 3. Proposed Export Map

### Stable Facade Root (`incin::*`)
- Explicit core re-exports: `Error`, `Result`, `typenum`.
- Explicit tensor & shape re-exports: `Tensor`, `Shape`, `ConstShape`, `PartialDynShape`, `Dyn`, `DTypeId`, `DeviceId`, `Grad`, `NoGrad`.
- Explicit backend re-exports: `IncinBackend`, `Cpu`, `DefaultBackend`, `DefaultDevice` (feature-gated).
- Feature-gated backend markers: `Cuda`, `CudaN` (cuda feature), `Wgpu`, `WgpuN` (wgpu feature), `Metal`, `MetalN` (metal feature).
- Explicit submodules: `nn`, `optim`, `metrics`, `data`, `transforms`, `hub`, `macros`.
- Curated tier submodules:
  - `incin::compile` (feature = "compiled"): curated compiled preview types (`CompileOptions`, `CompiledProgram`, `DynamicShapePolicy`, `ArtifactHeader`, etc.).
  - `incin::backend_authoring` (feature = "backend-authoring"): extension traits (`BackendFor`, `StorageBackend`, `Execute`, `OperationDescriptor`, `CapabilityRegistry`).
  - `incin::distributed` (feature = "distributed"): distributed preview types.

### Stable Prelude (`incin::prelude::*`)
- High-frequency user types only: `Tensor`, `Result`, `Error`, `DTypeId`, `DeviceId`, `Grad`, `NoGrad`, `Dyn`, `Cpu`, `DefaultBackend`, `DefaultDevice`.
- High-frequency NN modules & parameters: `Linear`, `Conv1d`, `Conv2d`, `BatchNorm2d`, `LayerNorm`, `AvgPool2d`, `MaxPool2d`, `Sequential`, `Param`, `Embedding`, `RNN`, `RNNCell`, `LSTM`, `LSTMCell`, `ReLU`, `GELU`, `Sigmoid`, `Softmax`, `Swish`, `Tanh`, `Dropout`, `Flatten`, `Init`, `RMSNorm`, `BCEWithLogitsLoss`, `CrossEntropyLoss`, `L1Loss`, `MSELoss`.
- High-frequency Optimizers: `SGD`, `Adam`, `AdamW`, `Optimizer`, `LRScheduler`, `Gradients`.
- High-frequency Macros: `s`, `idx`, `module`, `seq`, `SeqTy`, `import_model`, `model`, `mesh`, `axes`, `einsum`, `parallel`, `placement`.
- NO compiler pass types, IR graph types, autoref fallback traits, storage handles, or test backends in default prelude.

---

## 4. Acceptance Criteria

- [ ] No wildcard `pub use` from another Incin crate in a public facade/prelude. => `audit-evidence/API-001/summary.md`
- [ ] Public API snapshot reviewed and checked in (`api-after.txt`). => `audit-evidence/API-001/api-after.txt`
- [ ] All workspace doctests compile (`cargo test --doc --workspace --all-features`). => `audit-evidence/API-001/commands.log`
- [ ] Compile-pass and compile-fail API fixtures pass (`cargo test -p incin-core --test compile_tests`). => `audit-evidence/API-001/commands.log`
- [ ] `cargo semver-checks` report is archived. => `audit-evidence/API-001/semver-checks.log`
