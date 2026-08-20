# Incin Codebase Truth Audit and Implementation Specification

**Audit date:** 2026-08-02  
**Repository audited:** uploaded `incin(2).zip`, extracted at `/mnt/data/incin_audit/incin`  
**Source revision:** `1264f7b` (`develop`)  
**Purpose:** replace completion claims with source-grounded work instructions that a coding agent can execute without inventing architecture or acceptance criteria.

---

## 1. Executive verdict

The repository is **not complete in the sense implied by its checked task ledger**. It contains a large amount of real implementation, tests, documentation, and useful architecture, but several public-facing subsystems are still scaffolding, partially wired, misleadingly exposed, or represented as complete by tests that only validate structure rather than behavior.

The most important conclusions are:

1. **The public API leaks implementation details.** The facade re-exports the entire backend crate, the backend crate re-exports the entire core prelude, and the core prelude exports compiler internals, tuning internals, autoref helpers, graph internals, and test backend types. Marker and proof types such as `Dyn`, `Cuda`, `Wgpu`, `Metal`, `CheckedNumel`, `CheckedByteLen`, and `BufferSlot` expose public tuple constructors.
2. **“Compiled execution” is not an executable compiler.** Capture discards graph metadata and attributes; input guards are fabricated as empty `F32` shapes; constant folding and prepacking are clones/no-ops; fusion loses operation semantics; memory planning lacks byte sizes, alignment, dtype, device, aliasing, and storage class; tuning uses synthetic scores; and no validated public `compile -> executable -> run` path exists.
3. **The backend abstraction migration is incomplete.** `StorageBackend` and `Execute<O>` were added, but the old monolithic `Backend` supertrait remains and still requires all large operation traits. The ledger’s claims that the 254-method surface and default unsupported-operation adapter were removed are contradicted by source.
4. **Accelerator and external backends have large latent operation gaps.** CUDA and Metal explicitly route 38 of 49 tensor operations to unsupported stubs; WGPU and Candle route 33. Each leaves 26 of 42 float operations unsupported, plus creation and reduction gaps.
5. **Capability reporting is too coarse to describe reality.** The generated capability document tracks broad classes, while individual operations inside a class remain unsupported. A user can therefore see “native pointwise support” and still receive a runtime unsupported error for `sin`, `sign`, `atan2`, or `clamp`.
6. **There are production panic and silent-failure paths.** The static live-code scan found 39 panic-class sites, 283 `unwrap()` calls, 163 `expect()` calls, and 49 default/unimplemented-error sites after roughly masking tests. These counts are triage signals, not proof that every occurrence is wrong. The report identifies the high-risk occurrences that cross public, I/O, backend, macro-generated, or failure-recovery boundaries.
7. **The ONNX importer generates invalid product behavior.** It creates zero-filled parameters with `unwrap()`, exposes all generated parameter fields publicly, has a `load_default_weights()` method that does nothing, falls back to invented rank-4 dynamic shapes, suppresses parse failures, and generates runtime panics for malformed control-flow nodes.
8. **Data loading and transform validation are incomplete.** Zero batch size can panic; worker errors are not modeled in the iterator item; deterministic order is not guaranteed; worker lifecycle is weak; MNIST data consistency is not validated; downloads have incomplete integrity/resource controls; and transforms can index malformed buffers or panic on invalid probability.
9. **Release evidence is not reproducible from this environment.** The repository version is `0.0.0` while artifact tests and release documents discuss later versions. Several checked rows depend on hardware workflows whose logs are not source evidence. They must be considered unverified until rerun and archived.

### Recommended release classification

| Area | Current truth | Required public status now |
|---|---|---|
| CPU eager tensor core | Substantial, but dtype/quantized and panic/error contracts need work | Alpha/core preview |
| CUDA/WGPU/Metal | Useful vertical slices with broad operation gaps | Experimental, exact capability-gated |
| Candle adapter | Partial adapter with broad unsupported surface | Experimental |
| Autograd/training | Real implementation, requires failure-path and parity validation | Preview |
| Compiled execution | Structural prototype, not executable compilation | Private experimental or feature-gated prototype |
| Distributed | Prototype with explicit unsupported ragged/padding cases and hardware-dependent claims | Experimental preview |
| ONNX import | Unsafe to present as transparent model import | Experimental, fail-closed |
| Serialization/artifacts | Useful pieces, but compiled artifact contract not deployment-ready | Preview |
| Data loader/hub/transforms | Functional baseline, reliability and validation incomplete | Preview |
| LSP/viz/telemetry | Tooling prototypes with lifecycle/error hardening needed | Experimental |

Do **not** publish a `1.0` claim until all P0 and P1 acceptance gates in this document are satisfied and evidence is archived.

---

## 2. Audit boundary and confidence

### 2.1 What was inspected

The audit performed an exhaustive static scan over every production Rust source file under `crates/*/src/**/*.rs`, plus workspace manifests, CI/configuration, documentation, the checked proposal ledger, and the two existing implementation plans. High-risk paths were then manually reviewed, including:

- facade and prelude exports;
- shape/device/proof wrappers;
- backend traits, descriptor dispatch, capabilities, and unsupported macros;
- CPU, CUDA, WGPU, Metal, and Candle operation surfaces;
- compiled capture, guards, folding, fusion, memory planning, tuning, and artifacts;
- ONNX and safetensors procedural macros;
- optimizer failure paths and tensor manipulation/indexing;
- data loader, MNIST, transforms, and downloader behavior;
- distributed planning/checkpoint/reproducibility surfaces;
- LSP process plumbing, visualization panic test surfaces, and telemetry drop behavior;
- task-ledger rows whose checked status makes strong implementation claims.

### 2.2 What could not be executed

This environment did not contain `cargo` or `rustc`, and `~/.cargo/bin` was absent. Therefore:

- no compile, test, clippy, rustdoc, miri, sanitizer, GPU, distributed, or benchmark command was run;
- all semantic findings are source-based;
- any “passes tests” claim in the repository remains **unverified in this audit**;
- tasks below require exact dynamic validation commands before they may be marked complete.

This limitation does not weaken direct source contradictions such as a method returning `Ok(())` without loading weights, a no-op optimization pass, synthetic benchmark values, or a broad public re-export. It does mean that additional compiler errors and runtime bugs may still exist.

### 2.3 Scale

- Workspace package version: `0.0.0`
- Rust edition: 2024
- Production Rust files scanned: **272**
- Production Rust lines scanned: **approximately 106,000**
- Total Rust files including tests/examples generated in the earlier scan: **523**
- Raw lexical scan: 69 `panic!`, 1,167 `unwrap`, 309 `expect`, 60 explicit unsupported markers.
- Live-code-biased scan: 39 panic-class sites, 283 `unwrap`, 163 `expect`, 49 default/unimplemented-error sites.

Counts are discovery aids. Every acceptance decision must be based on the semantics of the occurrence, not on making the count zero blindly.

---

## 3. Source-of-truth rules for implementation agents

These rules override the existing checked boxes in `PROPOSALS.md` and the prior implementation plans.

1. **Source and executable evidence outrank task ledgers.** A checked row is not evidence.
2. **A compile-only test is insufficient for semantic work.** Every operation needs numerical, shape, dtype, device, error, and gradient tests where applicable.
3. **An unsupported operation is acceptable only when it is accurately declared before execution.** It must not be implied as supported by trait bounds, broad capability classes, docs, or a prelude.
4. **A no-op pass is not an implementation.** It must either transform/evaluate real data, return an explicit `NotImplemented`/`Unsupported` result, or remain private and clearly named as a prototype.
5. **Never fabricate metadata.** Do not invent rank 4, empty shapes, `F32`, device 0, or default attributes when source metadata is absent.
6. **Do not recover from invalid model files by generating runtime panics.** Procedural macros must emit a precise compile error at the source invocation.
7. **Do not mark hardware work complete without hardware logs.** Missing CUDA/Metal/multi-node access is `BLOCKED`, not `DONE`.
8. **Public constructors must not forge validated invariants.** Use private fields and checked constructors.
9. **Generated docs and runtime dispatch must consume the same capability registry.** Two manually maintained truth tables are forbidden.
10. **Every task below has a definition of done.** Do not reinterpret it downward.

---

## 4. Verified contradictions to the completion ledger

| Ledger row | Ledger claim | Source truth | Status |
|---|---|---|---|
| `EXE-006` | Split storage, `Execute<O>`, and capabilities out of the 254-method supertrait | `Backend` still inherits `TensorOps + NumericOps + FloatOps + CreationOps + ReductionOps + QuantizedOps + OptimizerOps + ModuleOps + LossOps` in `crates/incin-core/src/tensor/backend.rs`; the new traits coexist with the old surface | **Partial / contradicted** |
| `EXE-009` | Remove monolithic adapter and default unsupported-operation surface | Large unsupported macro blocks remain in CUDA/WGPU/Metal/Candle, and the monolithic trait remains the public requirement | **False as stated** |
| `CMP-001` | Capture eager graph into validated IR with descriptor parity | `CapturedNode` retains IDs/op/input/output IDs but drops operation attributes, value metadata, initializers, layouts, devices, and constants | **False as stated** |
| `CMP-002` | Immutable compiled plans and dynamic guards | Guards are created with `shape: []` and `dtype: F32`; an out-of-range input index is silently accepted | **Structural scaffold only** |
| `CMP-003` | Constant folding | `ConstantFolder::fold` returns a graph clone and empty folded set | **Not implemented** |
| `CMP-004` | Fusion | Fusion rewrites the producer into a node that retains the producer op while taking the consumer outputs, losing consumer semantics | **Semantically invalid prototype** |
| `CMP-005` | Weight prepacking | Prepacker returns a clone; no target layout or packed bytes are produced | **Not implemented** |
| `CMP-006` | Artifacts | Artifact serializes a plan, but there is no executable lowered program; validation is shallow and invariant-bearing fields are public | **Partial container, not executable artifact** |
| `DST-013` | Bounded plan tuning measured against one-GPU baseline | Baseline is `node_count * 10 µs`, and each iteration synthesizes a 5% improvement | **Simulated, not measured** |
| `SEC-010` | Hardened artifact framing/integrity/semantic verification | Magic and Adler32 were added, but checksum is non-authenticating, decoding remains only coarsely bounded, and semantic validation cannot validate semantics absent from the captured IR | **Partial** |
| `SEC-011` | Checked arithmetic enforced once and everywhere | Validated wrappers are publicly forgeable; Q8 and other backend paths still panic or use unchecked assumptions | **Partial** |
| `REL-001`–`REL-004` | Release readiness | Workspace remains `0.0.0`; core implementation claims above are incomplete; hardware workflow evidence was not bundled and cannot be inferred from source | **Unverified / blocked** |

The prior master plan itself acknowledges that the checked `CMP-001` through `CMP-006` rows are not an executable graph compiler. That acknowledgement is accurate; the task ledger should be corrected rather than retaining contradictory `[x]` states.

---

# PART I - PUBLIC API RECONSTRUCTION

## 5. Current public API defects

### 5.1 Transitive wildcard re-export chain

The facade uses `pub use incin_backends::*` in `crates/incin/src/lib.rs:87`. The backend root uses `pub use incin_core::prelude::*` in `crates/incin-backends/src/lib.rs:6`. The facade prelude then globs both backend and core preludes at `crates/incin/src/lib.rs:290-292`.

This produces four problems:

- internal backend modules and types become facade API accidentally;
- core names can enter through more than one path;
- adding any `pub` item to a lower crate can become a semver change in `incin` without review;
- explicit user-friendly aliases coexist with raw lower-level types, creating ambiguity and undocumented alternate construction paths.

### 5.2 Core prelude exports compiler and helper internals

`crates/incin-core/src/lib.rs:45-50` exports allocation plans, artifact headers, captured graph internals, folding/fusion internals, liveness structures, saved-tensor internals, and prepackers. Lines 78-90 export autoref fallback traits used for implementation dispatch. `Graph` and `OpType` are also prelude names.

These are not “common imports.” They are implementation or advanced compiler-authoring surfaces. Keeping them in the main prelude effectively freezes unstable representations.

### 5.3 Tensor prelude exports implementation modules

`crates/incin-core/src/tensor/mod.rs` globs argument conversion, automatic device selection, backend contracts, and tracing. This makes internal extension and dispatch details indistinguishable from stable user API.

### 5.4 Publicly forgeable marker/proof types

| Type | Current constructor | Problem | Required representation |
|---|---|---|---|
| `Dyn` | `pub struct Dyn(pub ())` | User can instantiate a shape/dtype marker value that should only appear at type level | `pub struct Dyn(PrivateZst);` with private field, or a unit struct only if value construction is intentionally supported and documented |
| `Cuda` / `Wgpu` / `Metal` | `pub struct Cuda(pub usize)` etc. | Marker value duplicates the runtime ordinal already represented by `Device::Arg/Field` | Opaque ZST marker: `pub struct Cuda(PrivateZst);`; ordinal enters only through backend/tensor construction argument |
| `CheckedNumel` | `pub struct CheckedNumel(pub usize)` | Bypasses resource-limit validation | Private field plus `get()`; constructor only from checked functions |
| `CheckedByteLen` | `pub struct CheckedByteLen(pub usize)` | Bypasses checked multiplication and allocation limits | Private field plus checked constructor chain |
| `BufferSlot` | `pub struct BufferSlot(pub usize)` | Lets callers forge planner-owned identities | Private field with `index()`; created only by planner/validated deserializer |
| `LaunchCandidate` fields | public mutable fields | Allows invalid tuning configurations to bypass device limits | Validating constructor; immutable accessors; candidate generated internally |
| backend variables/storage | CUDA exposes storage publicly | Lets users violate shape/storage/device invariants | Private or `pub(crate)` storage; safe read-only debug/introspection API |

`Sequential(pub L1, pub L2)` and `Gradients(pub G)` may be intentionally transparent ergonomic wrappers. They should be reviewed, not automatically privatized. The criterion is whether the tuple fields carry a validated invariant or permit mutation that invalidates object semantics.

### 5.5 Test backend escapes into production API

`DummyBackend` is documented as test-only but is in a public module and enters the main preludes. A fake shape-only backend is dangerous in production examples because it can make unsupported numerical behavior look valid.

Required action: move it behind `#[cfg(any(test, feature = "test-utils"))]`, re-export it only from `incin_core::test_utils`, and ensure the `test-utils` feature is excluded from default/release features.

### 5.6 Global lint suppression hides unfinished code

`crates/incin-core/src/lib.rs:2-3` allows `dead_code` and `unused_imports` for the entire crate. This prevents the compiler from exposing abandoned branches and incomplete migrations.

Remove the crate-wide allowances. Apply narrow `#[allow(...)]` only with a reason comment at the smallest scope, and make CI run `-D warnings` on every supported feature combination.

---

## 6. Target API architecture

### 6.1 Stable facade

The `incin` crate root should explicitly export only deliberate, documented names. No `pub use some_crate::*` is allowed at crate root or in the default prelude.

Recommended root structure:

```rust
// crates/incin/src/lib.rs
pub use incin_core::{Error, Result};
pub use incin_core::shapes::{Dyn, Shape, ConstShape, PartialDynShape};
pub use incin_core::tensor::{Tensor, DTypeId, DeviceId, Grad, NoGrad};
pub use incin_backends::{IncinBackend, Cpu};

#[cfg(feature = "cuda")]
pub use incin_backends::{Cuda, CudaN};
#[cfg(feature = "wgpu")]
pub use incin_backends::{Wgpu, WgpuN};
#[cfg(feature = "metal")]
pub use incin_backends::{Metal, MetalN};

pub mod nn { /* explicit stable list */ }
pub mod optim { /* explicit stable list */ }
pub mod data { /* explicit stable list */ }

#[cfg(feature = "compiled")]
pub mod compile;             // intentionally preview, curated API
#[cfg(feature = "backend-authoring")]
pub mod backend_authoring;   // explicit extension contracts
#[cfg(feature = "distributed")]
pub mod distributed;         // preview namespace
```

### 6.2 Default prelude

The prelude should contain only high-frequency user names:

- `Tensor`, `Result`, `Error`;
- `s!`, `idx!`, `module`, `seq!`, `SeqTy!`;
- `Dyn`, common dtype/grad/device markers;
- `Module`, `Param`, `Linear`, convolution/pooling/norm/activation basics;
- `Optimizer`, `SGD`, `Adam`, `AdamW`;
- `Cpu` and enabled device markers;
- no graph IR, compiler pass, artifact header, tuning candidate, autoref fallback, storage handle, tracing internals, serializer internals, or test backend.

### 6.3 Experimental namespaces

Use explicit feature and namespace boundaries:

- `incin::compile::{CompileOptions, CompiledProgram, DynamicShapePolicy, Artifact}`;
- `incin::backend_authoring::{StorageBackend, Execute, OperationDescriptor, CapabilityRegistry}`;
- `incin::distributed::{Mesh, Placement, Sharding, CheckpointManifest}`;
- `incin::diagnostics`, `incin::telemetry`, and `incin::viz` remain separate crates unless a deliberate facade wrapper is needed.

Mark preview types `#[non_exhaustive]` where downstream struct literals should not be stable. Prefer builders for configuration objects. Keep validated plan and artifact representations private behind read-only inspection methods.

### 6.4 API conformance tests

Add:

1. `trybuild` compile-fail cases that external code cannot construct `Dyn(())`, `CheckedNumel(1)`, `CheckedByteLen(1)`, `BufferSlot(0)`, or backend variables with arbitrary storage.
2. A compile-pass “prelude contract” fixture importing the intended names and nothing else.
3. A compile-fail fixture proving `DummyBackend` is absent without `test-utils`.
4. `cargo public-api` snapshots for `incin`, `incin-core`, `incin-backends`, and `incin-data` per feature tier.
5. `cargo semver-checks` in release CI.
6. Rustdoc link and doctest validation for every facade example.

---

## 7. API implementation tasks

### API-001 - Replace wildcard facade exports

**Priority:** P0  
**Files:** `crates/incin/src/lib.rs`, `crates/incin-backends/src/lib.rs`, `crates/incin-core/src/lib.rs`, `crates/incin-core/src/tensor/mod.rs`  
**Depends on:** none

**Implementation:**

1. Generate a baseline public API snapshot before editing.
2. Delete root wildcard exports.
3. Add explicit exports following section 6.
4. Split `incin_core::prelude` into:
  - `prelude`: stable common names;
  - `compile`: curated compiled preview;
  - `backend_authoring`: descriptor and executor extension API;
  - `test_utils`: feature-gated fake backends and test helpers.
5. Remove autoref fallback traits and compiler representation types from the prelude.
6. Update examples/docs to import from the correct namespaces.
7. Add a migration table for every removed path.

**Acceptance:**

- no wildcard `pub use` from another Incin crate in a public facade/prelude;
- public API snapshot reviewed and checked in;
- all workspace doctests compile;
- compile-pass and compile-fail API fixtures pass;
- `cargo semver-checks` report is archived.

### API-002 - Make marker and proof constructors opaque

**Priority:** P0  
**Files:** `tensor/base.rs`, `tensor/device.rs`, `shapes/shape.rs`, `compiled/alloc.rs`, backend tuning/storage/var files  
**Depends on:** API-001

**Implementation:**

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Dyn(private::Marker);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckedNumel(usize);
impl CheckedNumel {
    pub const fn get(self) -> usize { self.0 }
    pub(crate) const fn new_unchecked_after_validation(value: usize) -> Self { Self(value) }
}
```

- use a private marker type to prevent external construction while retaining derives;
- change partial device markers to opaque ZSTs; keep ordinal in `Device::Arg/Field` only;
- validate deserialized proof wrappers through custom `Deserialize` or deserialize raw values into a validator, never derive direct `Deserialize` on the proof type;
- make planner IDs and launch candidates private and expose inspection accessors;
- make backend storage fields private and expose safe data-transfer/debug methods.

**Acceptance:** external tuple construction fails; all valid constructors remain ergonomic; serialization cannot bypass validation; no unsafe transmute is introduced to regain construction.

### API-003 - Remove production exposure of test/prototype internals

**Priority:** P0  
**Files:** `tensor/backend.rs`, compiled module exports, viz panel exports  
**Depends on:** API-001

Move `DummyBackend` and `PanicTestPanel` behind test/test-utils gates. Keep compiled pass representations private until executable semantics exist. Rustdoc must not advertise fake numerical behavior as a backend.

### API-004 - Establish API tier and semver policy

**Priority:** P1  
**Files:** `README.md`, `CHANGELOG.md`, new `docs/API_STABILITY.md`, CI  
**Depends on:** API-001..003

Document stable/preview/experimental tiers, feature promises, minimum supported Rust version policy, and deprecation process. Version the crate consistently; do not use `1.0` language while workspace version is `0.0.0` or while P0 gates are open.

---

# PART II - EXECUTION AND BACKEND ARCHITECTURE

## 8. Monolithic backend trait: current state and replacement

`Backend` currently requires nine broad operation traits. The approximate method counts are:

| Trait | Declared operation methods |
|---|---:|
| `TensorOps` | 49 |
| `NumericOps` | 4 |
| `FloatOps` | 42 |
| `CreationOps` | 11 |
| `ReductionOps` | 19 |
| `QuantizedOps` | 3 |
| `OptimizerOps` | 1 |
| `ModuleOps` | 9 |
| `LossOps` | 4 |

Adding `StorageBackend` and `Execute<O>` without removing the old requirements does not complete the migration. It creates two parallel abstractions that must remain synchronized.

### Required design

Use sealed operation descriptors and per-operation execution:

```rust
pub trait StorageBackend: Clone + Send + Sync + 'static {
    type Storage: BackendStorage;
    type Device: Device;
    fn capabilities(&self) -> &CapabilityRegistry;
}

pub trait OperationDescriptor: sealed::Sealed + Send + Sync + 'static {
    type Output;
    const ID: OperationId;
    fn validate(&self, ctx: &ValidationContext) -> Result<Validated<Self>>
    where Self: Sized;
}

pub trait Execute<O: OperationDescriptor>: StorageBackend {
    fn execute(&self, op: &Validated<O>, ctx: &mut ExecutionContext<Self>)
        -> Result<O::Output>;
}
```

Tensor methods build typed descriptors, validate shape/dtype/device/layout, query exact capability, and call `Execute<O>`. A backend only implements operations it supports. Generic/fallback implementations must be explicit adapter types, not default methods that return runtime `UnsupportedBackendOperation` for a trait the backend claims to implement.

### Capability contract

Each capability key must include at least:

```rust
pub struct CapabilityKey {
    pub operation: OperationId,
    pub input_dtypes: SmallVec<[DTypeId; 4]>,
    pub output_dtype: DTypeId,
    pub rank_range: RangeInclusive<u8>,
    pub layouts: LayoutSet,
    pub mode: ExecutionMode,       // inference/training/backward
    pub determinism: Determinism,
}

pub enum SupportLevel {
    Native,
    Decomposed { recipe: RecipeId },
    HostFallback { warning: FallbackWarning },
    Unsupported { reason: &'static str },
}
```

Dispatch must query this exact key before allocating output storage or mutating optimizer state.

---

## 9. Backend operation gap matrix

The following operations are explicitly routed to unsupported macro implementations in source. This is not inferred from missing tests.

### CUDA

- **TensorOps - 38:** `where_cond`, `gather`, `scatter`, `index_select`, `masked_fill`, `unsqueeze`, `repeat`, `pad`, `triu`, `tril`, `diag`, `cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `logical_and`, `logical_or`, `logical_not`, `sub_scalar`, `div_scalar`, `maximum`, `minimum`, `abs_diff`, `lerp`, `addmm`, `bmm`, `scaled_dot_product_attention`, `unfold`, `pixel_shuffle`, `group_norm`, `instance_norm`, `float_to_scalar`, `float_to_vec1`, `int_to_scalar`, `int_to_vec1`, `tensor_to_dtype`
- **FloatOps - 26:** `sign`, `floor`, `ceil`, `round`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `erf`, `rsqrt`, `trunc`, `frac`, `powf`, `clamp`, `atan2`, `fmod`, `remainder`
- **CreationOps - 3:** `full`, `arange`, `linspace`
- **ReductionOps - 3:** `prod_all`, `prod_dim`, `cumsum`

### WGPU

- **TensorOps - 33:** `where_cond`, `gather`, `scatter`, `index_select`, `masked_fill`, `unsqueeze`, `repeat`, `pad`, `triu`, `tril`, `diag`, `cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `logical_and`, `logical_or`, `logical_not`, `sub_scalar`, `div_scalar`, `maximum`, `minimum`, `abs_diff`, `lerp`, `addmm`, `bmm`, `scaled_dot_product_attention`, `unfold`, `pixel_shuffle`, `group_norm`, `instance_norm`
- **FloatOps - 26:** `sign`, `floor`, `ceil`, `round`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `erf`, `rsqrt`, `trunc`, `frac`, `powf`, `clamp`, `atan2`, `fmod`, `remainder`
- **CreationOps - 3:** `full`, `arange`, `linspace`
- **ReductionOps - 3:** `prod_all`, `prod_dim`, `cumsum`

### Metal

- **TensorOps - 38:** `where_cond`, `gather`, `scatter`, `index_select`, `masked_fill`, `unsqueeze`, `repeat`, `pad`, `triu`, `tril`, `diag`, `cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `logical_and`, `logical_or`, `logical_not`, `sub_scalar`, `div_scalar`, `maximum`, `minimum`, `abs_diff`, `lerp`, `addmm`, `bmm`, `scaled_dot_product_attention`, `unfold`, `pixel_shuffle`, `group_norm`, `instance_norm`, `float_to_scalar`, `float_to_vec1`, `int_to_scalar`, `int_to_vec1`, `tensor_to_dtype`
- **FloatOps - 26:** `sign`, `floor`, `ceil`, `round`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `erf`, `rsqrt`, `trunc`, `frac`, `powf`, `clamp`, `atan2`, `fmod`, `remainder`
- **CreationOps - 3:** `full`, `arange`, `linspace`
- **ReductionOps - 9:** `max_all`, `min_all`, `max_dim`, `max_keepdim`, `min_dim`, `min_keepdim`, `prod_all`, `prod_dim`, `cumsum`

### Candle adapter

- **TensorOps - 33:** `where_cond`, `gather`, `scatter`, `index_select`, `masked_fill`, `unsqueeze`, `repeat`, `pad`, `triu`, `tril`, `diag`, `cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `logical_and`, `logical_or`, `logical_not`, `sub_scalar`, `div_scalar`, `maximum`, `minimum`, `abs_diff`, `lerp`, `addmm`, `bmm`, `scaled_dot_product_attention`, `unfold`, `pixel_shuffle`, `group_norm`, `instance_norm`
- **FloatOps - 26:** `sign`, `floor`, `ceil`, `round`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `erf`, `rsqrt`, `trunc`, `frac`, `powf`, `clamp`, `atan2`, `fmod`, `remainder`
- **CreationOps - 3:** `full`, `arange`, `linspace`
- **ReductionOps - 3:** `prod_all`, `prod_dim`, `cumsum`
- **QuantizedOps - 1:** `all quantized operations return UnsupportedBackendOperation`


### CPU caveats

CPU has broader method coverage, but support is not uniform:

- several Q8 storage/reduction/tape/shape paths panic instead of returning a typed unsupported/error result;
- fused optimizer logic is effectively F32-only;
- `arange`/`linspace` accept only selected float domains;
- Q8 conversion is not generally available;
- method presence in the monolithic trait must not be treated as proof of dtype support.

### Policy for closing gaps

For each operation/backend pair, choose exactly one outcome:

1. **Native:** implement and test on the device.
2. **Decomposed:** provide a documented composition of already-supported operations, with shape/dtype/gradient parity tests and a capability entry `Decomposed`.
3. **Host fallback:** only when explicitly enabled by policy, with visible telemetry/warning and transfer-cost semantics. Never silently copy to CPU.
4. **Unsupported:** remove the compile-time claim where possible and report exact unsupported capability before execution.

Do not mechanically implement every operation merely to make an unsupported count zero. Correct capability truth is more important than nominal breadth.

---

## 10. Backend/execution tasks

### EXE-001 - Finish the descriptor executor migration

**Priority:** P0  
**Files:** `incin-core/src/tensor/backend.rs`, `incin-core/src/exec/*`, `incin-backends/src/descriptor_bind.rs`, `dispatch.rs`, `dispatch_executor.rs`, every backend  
**Depends on:** API-001

**Procedure:**

1. Inventory every public tensor operation and assign one stable `OperationId` and one descriptor type.
2. Move validation currently repeated in tensor methods/backends into descriptor validation.
3. Implement `Execute<Op>` for the CPU vertical slice first.
4. Change tensor methods to require `B: Execute<Op>`, not `B: Backend` plus a large operation trait.
5. Introduce explicit decomposition adapters where useful.
6. Remove default unsupported method bodies and unsupported macro implementations from public trait conformance.
7. Reduce `Backend` to storage/device/element associated types and capabilities, or remove it in favor of `StorageBackend` plus a compatibility alias.
8. Migrate CUDA/WGPU/Metal/Candle one operation family at a time.
9. Delete the compatibility supertrait after all public call sites have moved.

**Acceptance:** no public backend type can satisfy an operation bound without implementing or deliberately adapting that operation; unsupported operations are rejected by capability/trait selection before allocation; old broad operation traits are private compatibility code or deleted.

### CAP-001 - Replace broad capability classes with operation-granular truth

**Priority:** P0  
**Files:** `incin-backends/src/capability.rs`, `capability_docs.rs`, backend registries, `docs/capabilities.md`  
**Depends on:** EXE-001

Create one typed registry consumed by dispatch, docs, doctor, and tests. Generate a row per operation, backend, dtype family, mode, and support level. Add a meta-test that enumerates every public descriptor and fails if any enabled backend lacks an explicit registry decision.

**Acceptance:** the docs cannot claim pointwise support while a member operation returns unsupported; registry and actual `Execute<Op>` implementations are cross-checked at compile/test time.

### BE-CPU-001 - Make CPU the semantic reference backend

**Priority:** P0  
**Files:** `incin-backends/src/cpu/**`  
**Depends on:** EXE-001

- replace Q8 panics with validated errors or real kernels;
- define dtype domains for every operation;
- fix optimizer state transactionality: state must not be removed permanently when a backend step errors;
- use checked shape/stride/byte arithmetic everywhere;
- define NaN, signed-zero, integer overflow, empty-reduction, and division semantics;
- add property tests against a simple scalar reference implementation;
- add finite-difference gradient tests for differentiable operations.

### BE-GPU-001 - Shared pointwise expression layer

**Priority:** P1  
**Files:** CUDA/WGPU/Metal codegen and operation modules  
**Depends on:** EXE-001, CAP-001, BE-CPU-001

Implement the 26 missing float operations through a shared validated expression IR where device languages permit it. Each operator defines mathematical semantics once; each code generator maps to CUDA C/PTX, WGSL, or MSL intrinsics and handles edge cases. Do not generate raw source from unchecked user strings.

Required tests per operation: scalar and broadcast shapes, empty dimensions where legal, F32/F64/F16/BF16 domain as supported, NaN/inf, signed zero where relevant, CPU parity tolerance, and backward parity.

### BE-GPU-002 - Indexing, masking, and shape operations

**Priority:** P1  
**Operations:** `where_cond`, gather/scatter/index_select/masked_fill, unsqueeze/repeat/pad, triangle/diag, comparisons, logical ops  
**Depends on:** BE-GPU-001

Use a common index linearization library with checked rank/stride arithmetic. Scatter must specify duplicate-index semantics and determinism. Pad must validate negative/positive padding and mode. Comparisons return a canonical boolean representation. Add randomized parity tests over non-contiguous inputs.

### BE-GPU-003 - Matrix/attention and normalization families

**Priority:** P1/P2  
**Operations:** `addmm`, `bmm`, scaled dot-product attention, unfold, pixel shuffle, group norm, instance norm  
**Depends on:** BE-GPU-002

Prefer vendor primitives only behind a capability check and deterministic policy. Provide mathematically simple reference decompositions first, then optimized paths. Attention must specify mask layout, causal behavior, scaling, dropout/training, accumulation dtype, and memory bounds. Every optimized path is compared with the decomposition.

### BE-IO-001 - Host transfer and dtype conversion

**Priority:** P0 for CUDA/Metal  
**Operations:** scalar/vector readback and `tensor_to_dtype`  
**Depends on:** EXE-001

Create asynchronous transfer primitives with synchronization semantics. Validate element count before host allocation. Conversion must use a checked conversion table; disallow silent float-to-int truncation and overflow unless an explicitly named casting mode requests it.

### BE-REDUCE-001 - Complete reduction semantics

**Priority:** P1  
**Operations:** product/cumsum everywhere; Metal min/max gaps  
**Depends on:** BE-CPU-001

Define identity for empty reductions, accumulation dtype, overflow behavior, NaN propagation, keepdim behavior, deterministic tree order, and gradient behavior. Use segmented scan for cumsum. Add large/non-power-of-two and non-contiguous tests.

### BE-CREATE-001 - Full/sequence creation

**Priority:** P1  
**Operations:** `full`, `arange`, `linspace`  
**Depends on:** EXE-001

Validate shape/numel through proof wrappers, reject zero step, define inclusivity and rounding, prevent sequence length overflow, and generate directly on device. Match CPU semantics exactly.

### BE-QNT-001 - Quantized scope decision

**Priority:** P1  
**Files:** CPU quant/storage/tape, Candle quant adapter, GGUF  
**Depends on:** CAP-001

Choose a supported quantization matrix and make public enums reflect it. Either implement each advertised scheme end-to-end - load, validate, execute, serialize, autograd policy - or remove it from stable API and return a parse-time capability error. Q8 must never enter a branch that panics.

---

# PART III - COMPILED EXECUTION

## 11. Why the current compiled subsystem is not executable

### 11.1 Capture is lossy

`CapturedNode` stores node ID, op, input IDs, and output IDs. It does not retain the full operation descriptor or attributes. Value metadata and initializer payloads are not represented. Therefore the compiler cannot reproduce operations whose semantics depend on axes, strides, padding, dilation, groups, epsilon, transpose flags, reduction mode, constants, or dtype/device/layout.

### 11.2 Input guards are fabricated

`CompiledPlan::compile` creates guards using an empty shape and `DTypeId::F32` for every input. `verify_input` silently succeeds when the requested index is outside the guards vector. A guard system that does not derive from input schema cannot protect execution.

### 11.3 Optimization passes are placeholders or invalid

- constant folding returns a graph clone and no folded values;
- prepacking returns a graph clone and no packed representation;
- fusion keeps the producer operation while replacing outputs with consumer outputs, so the fused node does not encode both operations;
- adjacency is treated as sufficient fusion eligibility even when the producer value may have other consumers.

### 11.4 Memory planning is not a memory plan

The planner tracks abstract slots and basic liveness, but not:

- byte length, dtype, alignment, device, memory space, storage class;
- alias/view relationships;
- in-place safety and saved tensors;
- constants, parameters, inputs, outputs, or persistent buffers;
- stream/event lifetime;
- compatibility constraints for slot reuse.

It also assumes node ID corresponds to vector position in a filter, which must be validated rather than assumed.

### 11.5 Tuning is synthetic

The “baseline” is derived from node count, and improvements are arithmetic percentages. No executable candidate is benchmarked. This must be named analytical scoring if retained; it cannot satisfy a measured tuning claim.

### 11.6 Artifact is a plan envelope, not a program

JSON plus magic and Adler32 can detect some corruption, but:

- no backend executable/kernel/module is represented;
- public fields permit inconsistent object construction;
- checksum is not authentication;
- semantic verification is shallow because semantic data was discarded;
- format limits do not comprehensively bound every nested length/depth/string;
- compatibility checks do not include operation schema/capability/backend ABI hashes.

---

## 12. Target compiled architecture

### 12.1 Validated portable IR

```rust
pub struct ProgramIr {
    schema: IrSchemaVersion,
    values: IndexVec<ValueId, ValueDesc>,
    nodes: IndexVec<NodeId, NodeDesc>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    constants: ConstantPool,
}

pub struct ValueDesc {
    shape: ShapeExpr,
    dtype: DTypeId,
    device: DeviceConstraint,
    layout: LayoutConstraint,
    storage: StorageClass,
    alias: AliasInfo,
}

pub struct NodeDesc {
    op: OperationDescriptorOwned,
    inputs: SmallVec<[ValueId; 4]>,
    outputs: SmallVec<[ValueId; 2]>,
    effects: EffectSet,
}
```

Every captured eager descriptor must round-trip into `OperationDescriptorOwned` without losing attributes. `ProgramIr::validate()` checks IDs, topological order, arity, schema, shape equations, dtype rules, effects, constant bounds, and resource limits.

### 12.2 Lowered executable

```rust
pub struct CompiledProgram<B: StorageBackend> {
    input_schema: Vec<InputGuard>,
    output_schema: Vec<OutputDesc>,
    schedule: Vec<ExecutableStep<B>>,
    memory: MemoryPlan,
    constants: PreparedConstants<B>,
    provenance: CompilationProvenance,
}

pub trait Lower<B: StorageBackend> {
    fn lower(&self, ir: &ValidatedProgramIr, options: &CompileOptions)
        -> Result<CompiledProgram<B>>;
}
```

An executable step contains a validated backend operation/kernel handle, buffer bindings, synchronization requirements, and debug origin. Runtime execution validates inputs, binds resources, executes schedule order, and returns typed outputs.

### 12.3 Shape guards

Guards derive from `ShapeExpr` and support:

- exact static dimensions;
- equal symbolic dimensions across inputs;
- bounded/range dimensions;
- divisibility/alignment constraints;
- dtype/device/layout requirements;
- bucket selection based on real input metadata.

Out-of-range input index is an error. Missing/extra inputs are errors. Error messages identify input, axis, expected constraint, and actual value.

### 12.4 Artifacts

Portable artifacts contain validated IR, constants, options, capability requirements, schema hashes, and optional target sections. Backend runtime handles are never serialized. Use a bounded binary format with explicit lengths, per-section hashes, total resource limits, and optional signature/authentication outside the core format. Deserialization produces an untrusted DTO that must pass validation before becoming `ValidatedArtifact`.

---

## 13. Compiled implementation tasks

### CMP-001 - Replace lossy capture with descriptor-parity IR

**Priority:** P0  
**Files:** `graph.rs`, `compiled/capture.rs`, operation descriptor definitions  
**Depends on:** EXE-001

Implement the IR in section 12. Capture every operation attribute and every value’s shape/dtype/device/layout. Preserve constant payloads with strict limits. Add one round-trip test per descriptor family comparing eager descriptor to captured owned descriptor. Delete any default metadata fabrication.

### CMP-002 - Implement validated input/output schema and guards

**Priority:** P0  
**Files:** `compiled/plan.rs`, shape-expression support  
**Depends on:** CMP-001

Derive guards from IR inputs. Verify exact input count. Implement symbolic equality/range/divisibility constraints. Add negative tests for wrong dtype, rank, axis, device, layout, missing/extra input, and index out of range.

### CMP-003 - Implement executable lowering and `run`

**Priority:** P0  
**Files:** new `compiled/lower.rs`, `compiled/program.rs`, backend lowerers  
**Depends on:** CMP-001, CMP-002, EXE-001

Start with a CPU vertical slice: creation, elementwise, matmul, reshape/view, reduction, one neural layer. `CompiledProgram::run` must numerically match eager execution and execute a stored schedule rather than replaying a graph through ad hoc dynamic dispatch.

### CMP-004 - Implement semantics-preserving constant evaluation

**Priority:** P1  
**Files:** `compiled/fold.rs`, constant pool  
**Depends on:** CMP-001

Maintain an evaluator registry keyed by operation ID. Fold only pure operations whose complete inputs are constants and whose resource cost is within limits. Record replacement value metadata and remove dead nodes after use analysis. Test actual constant values, not only node counts.

### CMP-005 - Replace fake fusion with fusion groups

**Priority:** P1  
**Files:** `compiled/fusion.rs`, backend codegen  
**Depends on:** CMP-001, CMP-003

A fused unit must retain the ordered operation sequence/expressions. Require single-use producer outputs unless duplicated computation is explicitly costed. Respect side effects, aliases, saved tensors, training/backward needs, dtype/layout/device boundaries, and numerical policy. Compare fused output and gradients with unfused execution.

### CMP-006 - Implement real memory planning

**Priority:** P1  
**Files:** `compiled/alloc.rs`, storage descriptors  
**Depends on:** CMP-001, CMP-003

Compute byte ranges with checked arithmetic; track alignment, device/space, dtype, storage class, alias views, persistence, stream/events, output escape, and saved-for-backward values. Reuse only compatible slots whose lifetimes do not overlap. Validate node IDs through maps, not vector-index assumptions. Add a debug plan report with peak bytes and reuse decisions.

### CMP-007 - Implement weight prepacking

**Priority:** P1  
**Files:** prepack module, backend lowerers  
**Depends on:** CMP-003

Define backend-specific `PackedConstant` with source hash, target capability, layout, dtype, and bytes/handle. Prepack only immutable constants. Invalidate on backend/driver/schema mismatch. Verify numerical parity and measure load/execution benefit.

### CMP-008 - Replace synthetic tuning with measured candidates

**Priority:** P1  
**Files:** `compiled/tuning.rs`, runtime benchmark hooks  
**Depends on:** CMP-003, CMP-006

Separate `AnalyticalScore` from `MeasuredDuration`. Warm up, synchronize device, measure repeated samples, reject outliers using a documented method, enforce a time/iteration budget, compare against the actual baseline executable, and only select a candidate when confidence and regression thresholds pass. Persist hardware/software fingerprint and shape signature. No arithmetic “5% improvement” is allowed.

### CMP-009 - Harden and version artifacts

**Priority:** P1  
**Files:** compiled artifact modules, `ResourceLimits`  
**Depends on:** CMP-001..008

Use private fields and validated constructors. Implement bounded section decoding, operation schema hash, capability requirements, target fingerprint, constant hashes, and migration policy. Separate corruption detection from authenticity. Fuzz parser and semantic validator. Invalid artifacts must never allocate based on unvalidated lengths or execute anything.

### CMP-010 - Rename or hide prototype until CMP-001..003 land

**Priority:** Immediate safety action  
**Files:** exports/docs/features  
**Depends on:** API-001

Until a real executable vertical slice exists, remove current compiled types from the default prelude and label the feature `compiled-prototype` or keep it private. Documentation must not state or imply that it compiles executable graphs.

---

# PART IV - TENSOR, AUTOGRAD, MODULE, AND OPTIMIZER CORRECTNESS

## 14. High-risk failure paths

### 14.1 Operator trait panics

Rust `Add`, `Mul`, and similar operator trait implementations cannot return `Result`, so they panic when fallible backend work fails. This may be an intentional ergonomic tradeoff, but it must not be the only API.

Required contract:

- fallible named methods (`try_add`, `try_mul`, etc.) are canonical and used internally;
- operator overloads are documented as panic-on-error convenience only;
- optionally gate panic operators behind an `operators` feature;
- never use panic operators in library internals where `Result` can propagate.

### 14.2 Shape conversion unwraps

Concat/stack and index resolution use `unwrap()` after internal assumptions. Replace with `field_from_dims`/typed errors. If an invariant truly cannot fail, prove it through a validated type rather than a comment.

### 14.3 Optional bias unwraps

Linear/conv code unwraps bias after checking a type/value condition. Use pattern matching and return a structured internal-invariant error if representations diverge. This avoids public panics after deserialization or future refactors.

### 14.4 Optimizer transaction bug

Adam removes moment tensors from maps and then invokes fallible backend logic. If the step fails, state may not be reinserted. Implement transactional update:

- validate all parameters/gradients/capabilities first;
- borrow or replace state through a guard that restores on drop;
- compute new parameter and state values;
- commit all three only after success;
- add an injected-backend-failure test proving state and parameters are unchanged.

### 14.5 Scalar conversion semantics

Float-to-integer conversion must not silently truncate NaN, infinity, or out-of-range values. Replace permissive helpers with `TryFrom<ScalarValue>` or explicit modes (`Exact`, `Truncate`, `Saturate`) and typed errors.

---

## 15. Core correctness tasks

### CORE-001 - Panic-free internal fallible paths

**Priority:** P0  
**Files:** live panic/unwrap inventory, especially tensor manipulation/indexing, module forward, optimizer, exec proof  
**Depends on:** none

Classify every live panic/unwrap/expect as:

- impossible invariant represented by type and debug assertion;
- recoverable user/backend/I/O error to propagate;
- process boundary where a fatal message is appropriate;
- test-only.

Convert all public-path recoverable cases. `paranoid-validation` must return an error or fail only under an explicitly documented debug/testing mode.

### CORE-002 - Operation semantic specification

**Priority:** P0  
**Files:** new `docs/OPERATION_SEMANTICS.md`, descriptor validation  
**Depends on:** EXE-001

For every public operation specify input ranks, broadcasting, dtype promotion, output shape/dtype, empty tensor behavior, NaN/inf, overflow, layout, gradient, determinism, and unsupported cases. Tests and backends consume this specification.

### CORE-003 - Autograd parity and mutation safety

**Priority:** P1  
**Files:** tape/autograd across core/backends  
**Depends on:** CORE-002, BE-CPU-001

Add finite-difference and cross-backend gradient tests, repeated-backward policy, detach/no-grad tests, saved-tensor lifetime tests, and failure rollback. Backend tape closures must return `Result`; CUDA backward must not `unwrap()` launch failures.

### CORE-004 - Module/state invariants

**Priority:** P1  
**Files:** `nn/*`, state dict, serialization  
**Depends on:** API-002

Validate parameter shapes/dtypes/devices during construction/load. Keep optional fields internally consistent. Make generated/manual module state load atomic and report every missing/unexpected/mismatched key.

### CORE-005 - Remove broad lint allowances

**Priority:** P1  
**Files:** core crate root and CI  
**Depends on:** API migration

Delete crate-wide `dead_code` and `unused_imports` allowances, eliminate dead paths, and enforce warnings across feature combinations.

---

# PART V - ONNX, MACROS, SERIALIZATION, AND MODEL I/O

## 16. ONNX importer defects

The generated model currently:

- initializes all parameters with zero tensors and `unwrap()`;
- exposes each parameter field publicly;
- implements `load_default_weights()` as `Ok(())` without loading anything;
- converts internal token strings back into tokens using `parse().unwrap()`;
- invents `[Dyn; 4]` when shape metadata is missing;
- emits runtime `panic!` for malformed `If`/`Loop` attributes;
- inadequately models loop-carried condition/state and scan outputs;
- can collide sanitized identifiers (`a.b`, `a/b`, and `a-b` all normalize similarly);
- writes a metadata cache but does not reliably read/use it;
- does not make initializer ownership/embedding/loading policy explicit.

This is a P0 truthfulness problem because the public docs claim a strongly typed generated model with weights and a typed forward method.

---

## 17. Macro/model I/O tasks

### ONNX-001 - Fail closed during macro expansion

**Priority:** P0  
**Files:** `incin-macros/src/onnx.rs`  
**Depends on:** none

Change parser/lowering functions to `syn::Result<TokenStream>` or a domain error converted to `compile_error!`. Never generate runtime panic for malformed static model metadata. Never use `parse().unwrap_or(empty)` or `parse().unwrap()` for generated fragments; retain token streams structurally.

### ONNX-002 - Correct shape and identifier handling

**Priority:** P0  
**Depends on:** ONNX-001

- missing shape is `UnknownRank`, not fabricated rank 4;
- validate negative/overflowing dimensions;
- create a deterministic identifier allocator with collision suffix and a reverse map to original tensor names;
- verify all node input/output references and graph outputs;
- reject unsupported ops at macro expansion with node name/opset/domain context.

### ONNX-003 - Implement real initializer loading

**Priority:** P0  
**Depends on:** ONNX-001, CORE-004

Choose one explicit API:

```rust
impl<B> Model<B> {
    pub fn from_onnx_initializers(device: B::DeviceArg) -> Result<Self>;
    pub fn from_state_dict(state: &StateDict, device: B::DeviceArg) -> Result<Self>;
}
```

Embed initializer bytes only within documented size/resource limits, or generate a sidecar loader keyed by original ONNX names. Remove zero-filled `new()` unless it is named `new_zeroed_for_testing` and test-gated. `load_default_weights` must either load verified bytes or not exist.

### ONNX-004 - Implement or reject control flow correctly

**Priority:** P1/P2  
**Depends on:** ONNX-001, CMP-001 if compiled execution is used

Implement ONNX `If` and `Loop` according to opset semantics, including captures, condition, loop-carried dependencies, trip count, scan outputs, and shape joins. Until then, emit a compile error naming the unsupported node. Partial lowering is forbidden.

### ONNX-005 - Opset/domain conformance suite

**Priority:** P1  
**Depends on:** ONNX-002..004

Use small generated ONNX fixtures and official backend-test style vectors. Test attributes, optional inputs, dynamic ranks, external data, malformed protobufs, resource limits, identifier collisions, and parity with a reference runtime.

### MACRO-001 - Audit all procedural macro panic/error boundaries

**Priority:** P1  
**Files:** `incin-macros/src/*`  
**Depends on:** ONNX-001

Convert path-extension unwraps and unsafe token assumptions into compile diagnostics. Add `trybuild` pass/fail fixtures with stable error codes/messages. Validate safetensors names and dimensions with the same identifier allocator and resource limits.

### IO-001 - Quantized export/API truth

**Priority:** P1  
**Files:** GGUF/MLX/export/import modules  
**Depends on:** BE-QNT-001

Do not publicly advertise quantization schemes that exporter/runtime cannot support. Validate tensor count, dimensions, byte lengths, offsets, string lengths, and duplicate keys before allocation. Add round-trip and malformed-file fuzz tests.

---

# PART VI - DATA PIPELINE AND HUB

## 18. Data loader and dataset defects

- `batch_size == 0` reaches division/chunk logic that can panic.
- “single-threaded” mode still spawns a worker thread.
- iterator items do not carry worker errors; a failed worker can look like early end-of-data.
- multiworker collection can reorder batches even when shuffle is false.
- thread join handles are not retained for deterministic cancellation/join.
- poisoned mutexes are unwrapped.
- no complete epoch/seed/drop-last/prefetch/timeout contract exists.
- dataset `get()` returning `None` conflates out-of-range, corrupt sample, and decode failure.
- MNIST stores raw buffers publicly, does not robustly verify image/label counts and dimensions, and can slice malformed data.
- transforms do not uniformly verify `data.len() == product(shape)`; normalize can truncate channel groups; invalid flip probability can panic; crop/flip can index malformed buffers.
- downloads have incomplete checksum/content-length/decompression limits, cache validation, and concurrent writer locking.

---

## 19. Data implementation tasks

### DATA-001 - Validating loader builder

**Priority:** P0  
**Files:** data loader  
**Depends on:** none

```rust
pub struct DataLoaderBuilder<D> {
    dataset: D,
    batch_size: NonZeroUsize,
    workers: usize,
    prefetch: NonZeroUsize,
    drop_last: bool,
    ordering: OrderingPolicy,
    seed: u64,
    timeout: Option<Duration>,
}

pub type BatchResult<T> = Result<T, DataError>;
```

Return `Result<DataLoader<_>>`. A zero-worker loader executes synchronously without a thread. Retain worker handles and a cancellation token. Iterator yields `Result<Batch, DataError>`. Ordered mode uses sequence numbers and a reorder buffer. Drop joins/cancels without deadlock.

### DATA-002 - Dataset error contract and MNIST validation

**Priority:** P0  
**Depends on:** DATA-001

Change dataset access to distinguish index absence from sample decode/corruption. MNIST constructor validates magic, dimensions, checked count/byte length, exact or explicitly allowed trailing bytes, image-label equality, and limits. Keep buffers/count private. `get` uses checked offsets.

### DATA-003 - Transform invariants

**Priority:** P0  
**Depends on:** DATA-002

Every transform validates rank, dimensions, buffer length, channels, parameters, and output allocation before indexing. Probability constructors require `0.0..=1.0` and reject NaN. Normalize requires exact channel divisibility. Add property tests over malformed shapes and arbitrary byte/vector lengths.

### DATA-004 - Downloader integrity and concurrency

**Priority:** P1  
**Files:** hub/downloader  
**Depends on:** resource-limit API

Add maximum compressed/uncompressed bytes, streaming length checks, checksum/hash verification, ETag/Last-Modified metadata, atomic unique temp files, per-target lock, fsync/rename policy, cache revalidation, redirect/domain policy, decompression bomb defense, cancellation, and explicit offline mode. Never trust an existing cache file only because its path exists.

### DATA-005 - Determinism and reproducibility tests

**Priority:** P1  
**Depends on:** DATA-001

For the same dataset/seed/epoch/config, batch order and augmentations must match across runs. Define whether worker count may change deterministic output. Archive seed and epoch in training checkpoints.

---

# PART VII - DISTRIBUTED, CHECKPOINTS, AND REPRODUCIBILITY

## 20. Distributed scope truth

The distributed codebase contains significant planning and checkpoint work, but source still explicitly states that padding/ragged sharding is not implemented. Hardware behavior, collective correctness, fail-stop behavior, topology performance, and multi-node recovery cannot be validated statically.

The public API must reject unsupported placements during planning, before execution. It must not advertise general sharding when only divisible/even partitions are valid.

### DIST-001 - Encode shard divisibility constraints

**Priority:** P0 for public distributed preview  
**Files:** distributed plan/placement/mesh  
**Implementation:** represent even, padded, and ragged shard policies distinctly. Only construct a validated plan when backend/collective supports the selected policy. Report axis, dimension, mesh size, and alternatives on failure.

### DIST-002 - Reproducibility manifest completeness

**Priority:** P1  
**Files:** reproducibility/checkpoint modules  
**Implementation:** use fixed-size typed digests rather than unvalidated strings. Include and compare environment/runtime/backend/driver/capability/schema versions. `replay_diff` must report environment differences rather than omit them.

### DIST-003 - Checkpoint transaction and reshard validation

**Priority:** P1  
**Implementation:** validate all shard hashes, tensor metadata, global shape, placement, dtype, world/mesh mapping, and resource limits before committing load. Reshard through a planned data movement graph with failure cleanup. Test interrupted writes and partial/mismatched shard sets.

### DIST-004 - Hardware evidence gate

**Priority:** P0 release gate  
**Implementation:** run real two-rank, multi-GPU, and multi-node suites. Archive commands, commit, hardware inventory, driver/runtime versions, logs, numerical parity, timeout/failure injection, and performance distributions. A workflow dispatch command without resulting logs is not evidence.

---

# PART VIII - LSP, VIZ, TELEMETRY, AND DIAGNOSTICS

## 21. Tooling hardening tasks

### TOOL-001 - LSP child lifecycle and pipe errors

**Priority:** P1  
**Files:** `incin-lsp`, `cargo-incin` proxy code  
**Implementation:** replace `expect`/mutex unwrap with typed errors; retain child lifecycle ownership; support shutdown/cancellation; avoid joining a thread blocked forever on editor stdin after server stdout closes; kill/wait child on pipe failure; propagate both pump errors. Add fake-child integration tests for EOF, crash, malformed JSON, broken pipe, and shutdown.

### TOOL-002 - Remove public panic test UI

**Priority:** P0 API cleanup  
**Files:** viz panels/exports  
**Implementation:** put `PanicTestPanel` behind `cfg(test)` or a non-release dev feature. Keep panic containment tests internal.

### TOOL-003 - Telemetry drop/backpressure safety

**Priority:** P1  
**Files:** telemetry emitter  
**Implementation:** avoid `expect` in `Drop`; make shutdown idempotent; define bounded queues and priority behavior; report dropped events; join workers with timeout/failure reporting outside `Drop`; ensure telemetry failure cannot panic model execution.

### TOOL-004 - Diagnostics parser stability

**Priority:** P1  
**Files:** diagnostics  
**Implementation:** use structured error codes/data where available; fixture-test supported rustc versions; provide fallback for unknown wording; never claim a rewrite/fix when span parsing is uncertain.

---

# PART IX - SECURITY, RESOURCE LIMITS, AND ERROR CONTRACT

## 22. Required error model

Create structured error domains with source chaining and stable machine-readable codes:

- shape/dtype/device/layout validation;
- unsupported capability with exact operation/backend/dtype;
- allocation/resource limit;
- backend launch/synchronization;
- model parse/opset/attribute;
- serialization/artifact validation;
- data worker/download/decode;
- distributed collective/checkpoint;
- tooling process/protocol.

Error messages must include operation and relevant dimensions but must not include unbounded attacker-controlled strings or secrets. Internal invariant failures should be distinguishable from user input errors.

### SEC-001 - Enforce validated arithmetic at allocation boundaries

**Priority:** P0  
**Implementation:** all numel, byte length, stride, offset, sequence length, tensor slice, constant pool, artifact section, download, decompression, and sharding calculations use checked arithmetic and configured limits. Proof wrappers have private fields. Add boundary/property/fuzz tests.

### SEC-002 - Remove backend `transmute(...).unwrap()` paths

**Priority:** P0  
**Files:** CUDA and other backend storage/cast sites  
**Implementation:** provide typed allocation/copy APIs returning `Result`, validate exact element/byte counts, alignment, and dtype before conversion. Use `bytemuck` only with proven `Pod` types and checked lengths. No unsafe conversion may be justified only by “backend created it.”

### SEC-003 - Artifact parser fuzzing and authentication boundary

**Priority:** P1  
**Depends on:** CMP-009

Fuzz framing, section lengths, nested descriptors, constants, schema versions, and semantic validator. Document that checksums detect accidental corruption, while authenticity requires a signature/trusted transport policy.

### SEC-004 - Poison and panic policy

**Priority:** P1  
**Implementation:** decide per mutex whether poison propagates, recovers by extracting inner state, or aborts subsystem; never blindly unwrap. Catching a panic is only valid at explicit plugin/task isolation boundaries and must not continue with potentially corrupted mutable state.

---

# PART X - TEST AND RELEASE SYSTEM

## 23. Mandatory validation commands

Run from repository root on a machine with the declared stable toolchain. Preserve complete stdout/stderr and exit code under `audit-evidence/<task-id>/`.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace --no-default-features
cargo doc --workspace --all-features --no-deps
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps
```

Feature combinations must not be approximated by only `--all-features`. At minimum test:

```bash
cargo test -p incin --no-default-features --features std,cpu
cargo test -p incin --no-default-features --features std,cpu,train
cargo test -p incin --no-default-features --features std,wgpu
cargo test -p incin --no-default-features --features std,cuda
cargo test -p incin --no-default-features --features std,metal
cargo test -p incin --no-default-features --features std,external-candle
cargo check -p incin-core --no-default-features
```

Use a feature-powerset tool with an allowlist for mutually exclusive/platform features. Also run:

```bash
cargo test -p incin-core --test compile_tests
cargo semver-checks check-release -p incin
cargo public-api --simplified --manifest-path crates/incin/Cargo.toml
cargo deny check
cargo audit
cargo miri test -p incin-core --tests              # supported subsets
cargo test --doc --workspace --all-features
```

Recommended additional gates:

- `cargo nextest run` for suite reliability;
- proptest for shapes/indexing/reductions/transforms;
- `cargo fuzz` for artifacts, ONNX, safetensors/GGUF, downloader metadata, and shape parsing;
- loom for loader/telemetry/cache concurrency units;
- sanitizers for CPU/data/tooling tests;
- criterion or the repository benchmark harness with stored baselines;
- real CUDA/WGPU/Metal parity jobs, each with device/driver fingerprint;
- real distributed failure-injection jobs.

### Numerical acceptance pattern

Every implemented mathematical operation needs:

1. exact shape and dtype tests;
2. small hand-computed values;
3. randomized CPU reference parity;
4. non-contiguous/broadcast/empty/edge dimensions;
5. NaN/inf/signed-zero policy;
6. gradient check where differentiable;
7. device error propagation;
8. capability declaration consistency;
9. deterministic-mode behavior;
10. benchmark/regression only after correctness passes.

---

## 24. Evidence standard

Each task completion directory must contain:

```text
audit-evidence/<TASK-ID>/
  summary.md
  commands.log
  environment.json
  changed-files.txt
  api-before.txt          # API tasks
  api-after.txt
  test-results/
  benchmark-results/      # performance/tuning tasks
  hardware.json           # accelerator/distributed tasks
  known-limitations.md
```

`summary.md` must list each acceptance criterion and link it to a test/log. “Implemented,” “looks correct,” a diff, or a successful compile alone is not evidence.

---

## 25. Dependency-ordered execution plan

### Phase 0 - Truth and containment

- correct false ledger states;
- hide compiled prototype/test internals;
- API-001/002/003;
- CORE-001 immediate public panic fixes;
- DATA-001 zero-batch/lifecycle minimum fix;
- ONNX-001 remove generated runtime panics/no-op weight claim.

### Phase 1 - Semantic foundation

- EXE-001 descriptor executor;
- CAP-001 exact registry;
- CORE-002 operation semantics;
- BE-CPU-001 reference backend;
- SEC-001 checked arithmetic;
- API-004 and API test gates.

### Phase 2 - Compiled CPU vertical slice

- CMP-001/002/003;
- real eager-vs-compiled numerical tests;
- CMP-010 can be reversed only when this passes.

### Phase 3 - Model/data reliability

- ONNX-002/003;
- CORE-003/004;
- DATA-002/003/004/005;
- IO-001 and quantized scope.

### Phase 4 - Accelerator breadth

- BE-IO-001;
- BE-GPU-001/002;
- BE-REDUCE-001/BE-CREATE-001;
- BE-GPU-003;
- exact capability docs and hardware evidence.

### Phase 5 - Compiler optimization and artifacts

- CMP-004 through CMP-009;
- measured, not synthetic, tuning;
- fuzz/security gates.

### Phase 6 - Distributed/tooling/release

- DIST-001..004;
- TOOL-001..004;
- feature powerset, public API, semver, docs, benchmarks;
- only then assign a non-zero release candidate version and evaluate 1.0 criteria.

---

## 26. Global definition of done

The project is not “done” until all of the following are true:

- no default-prelude exposure of compiler/tuning/test/storage internals;
- all proof/marker invariants are unforgeable through safe public API;
- every public operation has one descriptor and exact capability entry;
- a backend cannot compile as supporting an operation it only rejects through a default runtime stub;
- CPU semantics are specified and tested as reference;
- accelerator gaps are either implemented or accurately unavailable before execution;
- compiled execution retains full semantics and has a real executable `run` path;
- folding/fusion/prepacking/tuning produce demonstrated semantic/measurement effects;
- ONNX never invents shapes, creates fake loaded weights, or generates runtime panics for static invalidity;
- data worker errors, lifecycle, ordering, validation, and downloads are explicit and tested;
- public library paths do not panic on recoverable user/backend/I/O errors;
- resource limits protect every untrusted size/allocation boundary;
- artifact/model/data parsers are fuzzed;
- public API snapshots and semver checks are clean;
- all supported feature/platform/hardware jobs have archived logs tied to the commit;
- documentation statements are generated from or tested against implementation truth;
- every remaining limitation is named as unsupported/experimental, not hidden behind a checked ledger row.

---

# APPENDIX A - High-priority source locations

| Area | Location | Finding |
|---|---|---|
| Facade wildcard | `crates/incin/src/lib.rs:87` | `pub use incin_backends::*` |
| Prelude globs | `crates/incin/src/lib.rs:290-292` | imports entire backend and core preludes |
| Backend root leak | `crates/incin-backends/src/lib.rs:6` | exports entire core prelude |
| Core compiler leak | `crates/incin-core/src/lib.rs:45-50` | compiled internal representations in prelude |
| Autoref leak | `crates/incin-core/src/lib.rs:78-90` | internal fallback traits in prelude |
| Lint masking | `crates/incin-core/src/lib.rs:2-3` | crate-wide dead/unused allowances |
| `Dyn` constructor | `crates/incin-core/src/tensor/base.rs:13` | public tuple field |
| Device marker constructors | `crates/incin-core/src/tensor/device.rs:115,149,276` | public ordinal tuple fields |
| Checked wrappers | `crates/incin-core/src/shapes/shape.rs:105,116` | public tuple fields bypass validation |
| Planner slot | `crates/incin-core/src/compiled/alloc.rs:157` | public tuple field |
| Monolithic backend | `crates/incin-core/src/tensor/backend.rs:114-129` | old broad supertrait remains |
| Capture loss | `crates/incin-core/src/compiled/capture.rs` | IDs/op only, no descriptor/value parity |
| Fabricated guards | `crates/incin-core/src/compiled/plan.rs:112-131` | empty F32 guards; out-of-range accepted |
| No-op folding | `crates/incin-core/src/compiled/fold.rs:51-55` | clone plus empty folded set |
| Invalid fusion | `crates/incin-core/src/compiled/fusion.rs:138-148` | producer op retained, consumer semantics lost |
| Abstract memory plan | `crates/incin-core/src/compiled/alloc.rs` | no bytes/alignment/dtype/device/alias model |
| Synthetic tuning | `crates/incin-core/src/compiled/tuning.rs` | node-count baseline and synthetic improvement |
| ONNX fake init | `crates/incin-macros/src/onnx.rs:455` | zero parameter with unwrap |
| ONNX no-op loader | `crates/incin-macros/src/onnx.rs:529-532` | returns `Ok(())` |
| ONNX invented rank | `crates/incin-macros/src/onnx.rs:372-376,404-407` | fallback `[Dyn; 4]` |
| ONNX runtime panics | `crates/incin-macros/src/onnx.rs:187-258` | malformed control flow becomes runtime panic |
| Optimizer state | `crates/incin-core/src/optim/mod.rs:239-240` | removes state then unwraps before fallible work |
| Shape unwraps | `crates/incin-core/src/tensor/ops/manipulation.rs:628,702`; `shapes/idx.rs:143` | public-path assumptions panic |
| CUDA init | `crates/incin-backends/src/cuda/gpu.rs:149` | context initialization `expect` |
| Distributed limitation | `crates/incin-core/src/dist/plan.rs` near explicit “padding/ragged sharding not implemented” | public preview scope incomplete |
| Viz panic panel | `crates/incin-viz` panel exports | intentional panic utility visible outside tests |

---

# APPENDIX B - Public tuple-field review

- `crates/incin-core/src/shapes/shape.rs:105` - `pub struct CheckedNumel(pub usize);`
- `crates/incin-core/src/shapes/shape.rs:116` - `pub struct CheckedByteLen(pub usize);`
- `crates/incin-core/src/shapes/dim.rs:174` - `pub struct ProdDim<A, B>(pub usize, core::marker::PhantomData<(A, B)>);`
- `crates/incin-core/src/nn/module.rs:506` - `pub struct Sequential<L1, L2>(pub L1, pub L2);`
- `crates/incin-core/src/compiled/alloc.rs:157` - `pub struct BufferSlot(pub usize);`
- `crates/incin-core/src/tensor/base.rs:13` - `pub struct Dyn(pub ());`
- `crates/incin-core/src/tensor/device.rs:115` - `pub struct Cuda(pub usize);`
- `crates/incin-core/src/tensor/device.rs:149` - `pub struct Wgpu(pub usize);`
- `crates/incin-core/src/tensor/device.rs:276` - `pub struct Metal(pub usize);`
- `crates/incin-core/src/optim/mod.rs:10` - `pub struct Gradients<G>(pub G);`
- `crates/incin-backends/src/cpu/var.rs:25` - `pub struct CpuVar(pub(crate) Rc<RefCell<CpuStorage>>);`
- `crates/incin-viz-plugin-api/src/render_ctx.rs:11` - `pub struct HitId(pub u32);`


Not every public tuple field is wrong. The implementation agent must apply the invariant criterion in section 5.4 and document each keep/change decision.

---

# APPENDIX C - Live panic-class inventory

The following scan excludes much obvious test-only code but is still lexical. Each item must be classified, not blindly deleted.

- `crates/incin-macros/src/onnx.rs:187` - `quote::quote! { panic!("If node missing then_branch graph") }`
- `crates/incin-macros/src/onnx.rs:190` - `quote::quote! { panic!("If node missing then_branch") }`
- `crates/incin-macros/src/onnx.rs:214` - `quote::quote! { panic!("If node missing else_branch graph") }`
- `crates/incin-macros/src/onnx.rs:217` - `quote::quote! { panic!("If node missing else_branch") }`
- `crates/incin-macros/src/onnx.rs:255` - `quote::quote! { panic!("Loop node missing body graph") }`
- `crates/incin-macros/src/onnx.rs:258` - `quote::quote! { panic!("Loop node missing body attribute") }`
- `crates/incin-macros/src/module.rs:483` - `_ => unreachable!(),`
- `crates/incin-core/src/exec/proof.rs:263` - `panic!("paranoid-validation: descriptor failed its own invariants: {error:?}");`
- `crates/incin-core/src/tensor/ops/binary.rs:377` - `panic!(`
- `crates/incin-core/src/tensor/ops/binary.rs:410` - `panic!(`
- `crates/incin-core/src/tensor/ops/binary.rs:442` - `panic!(`
- `crates/incin-core/src/tensor/ops/binary.rs:474` - `panic!(`
- `crates/incin-core/src/tensor/ops/unary.rs:423` - `.unwrap_or_else(|e| panic!("Tensor `*` (scalar) operator panicked: {e:?}"))`
- `crates/incin-core/src/tensor/ops/unary.rs:436` - `.unwrap_or_else(|e| panic!("Tensor `*` (scalar) operator panicked: {e:?}"))`
- `crates/incin-core/src/tensor/ops/unary.rs:449` - `.unwrap_or_else(|e| panic!("Tensor `+` (scalar) operator panicked: {e:?}"))`
- `crates/incin-core/src/tensor/ops/unary.rs:462` - `.unwrap_or_else(|e| panic!("Tensor `+` (scalar) operator panicked: {e:?}"))`
- `crates/incin-viz/src/panels/panic_test.rs:44` - `panic!("Manual panic triggered from PanicTestPanel");`
- `crates/incin-backends/src/kernel.rs:472` - `_ => unreachable!("CudaScalarSpec already rejected non-float dtype"),`
- `crates/incin-backends/src/kernel.rs:1052` - `_ => unreachable!(),`
- `crates/incin-backends/src/kernel.rs:1106` - `_ => unreachable!(),`
- `crates/incin-backends/src/tuning/service.rs:799` - `_ => unreachable!("non-coordinated policies returned above"),`
- `crates/incin-backends/src/wgpu/backend.rs:1189` - `_ => panic!("Unknown reduce dim mode"),`
- `crates/incin-backends/src/dist/nccl.rs:1168` - `TensorParallelCollective::RowOutputSum => unreachable!(),`
- `crates/incin-backends/src/cpu/tape.rs:314` - `CpuBuffer::Q8_0(_) => panic!("sum_dim_keepdim not supported on Q8_0 buffer"),`
- `crates/incin-backends/src/cpu/stride.rs:24` - `panic!(`
- `crates/incin-backends/src/cpu/storage.rs:170` - `panic!("from_f64_values not supported on Q8_0 quantized buffer")`
- `crates/incin-backends/src/cpu/storage.rs:188` - `panic!("get_f64 not supported directly on Q8_0 quantized buffer")`
- `crates/incin-backends/src/cpu/storage.rs:268` - `CpuBuffer::Q8_0(_) => panic!("ones_like not supported on Q8_0 buffer"),`
- `crates/incin-backends/src/cpu/storage.rs:448` - `CpuBuffer::Q8_0(_) => panic!("materialize not supported on Q8_0 buffer"),`
- `crates/incin-backends/src/cpu/storage.rs:534` - `CpuBuffer::Q8_0(_) => panic!("scatter_into_zeros not supported on Q8_0 buffer"),`
- `crates/incin-backends/src/cpu/creation.rs:103` - `_ => unreachable!(),`
- `crates/incin-backends/src/cpu/creation.rs:140` - `_ => unreachable!(),`
- `crates/incin-backends/src/cpu/gradcheck.rs:49` - `_ => panic!("gradcheck: perturbation only supported for F32/F64 buffers"),`
- `crates/incin-backends/src/external/conformance.rs:382` - `_ => unreachable!("is_supported() is false only for Unsupported"),`
- `crates/incin-backends/src/external/conformance.rs:467` - `_ => unreachable!("is_supported() is false only for Unsupported"),`
- `crates/incin-backends/src/cpu/ops/shape_ops.rs:335` - `CpuBuffer::Q8_0(_) => panic!("concat not supported on Q8_0 buffer"),`
- `crates/incin-backends/src/cpu/ops/elementwise_kernel.rs:1513` - `_ => unreachable!("inner stride pattern was validated"),`
- `crates/incin-backends/src/cpu/ops/reduce.rs:117` - `CpuBuffer::Q8_0(_) => panic!("sum_axis_keepdim not supported on Q8_0 buffer"),`
- `crates/incin-backends/src/cpu/ops/reduce.rs:151` - `CpuBuffer::Q8_0(_) => panic!("fill_like not supported on Q8_0 buffer"),`


---

# APPENDIX D - Production source inventory

Every file below was included in the static scan. Flags are lexical triage counts (`panic`, `unwrap`, `expect`, explicit unsupported markers, public declarations, unsafe tokens), not final defect judgments.

| File | Lines | Scan flags |
|---|---:|---|
| `crates/incin/src/bin/cargo-incin.rs` | 531 | expect:2, public:3 |
| `crates/incin/src/doctor.rs` | 867 | public:26 |
| `crates/incin/src/lib.rs` | 332 | unwrap:4, public:60 |
| `crates/incin/src/plan_report.rs` | 98 | public:4 |
| `crates/incin/src/train.rs` | 619 | unwrap:2, public:23 |
| `crates/incin/src/tune_report.rs` | 95 | public:3 |
| `crates/incin-backends/src/backend_kind.rs` | 408 | unwrap:19, unsupported:2, public:2 |
| `crates/incin-backends/src/bytes.rs` | 80 | unwrap:7, public:1 |
| `crates/incin-backends/src/capability.rs` | 354 | public:6 |
| `crates/incin-backends/src/capability_docs.rs` | 293 | public:1 |
| `crates/incin-backends/src/codegen/mod.rs` | 12 | public:2 |
| `crates/incin-backends/src/codegen/pointwise.rs` | 500 | unwrap:40, public:14 |
| `crates/incin-backends/src/cpu/creation.rs` | 419 | panic:1, unwrap:15, unsupported:5 |
| `crates/incin-backends/src/cpu/executor.rs` | 612 | unwrap:1, expect:1 |
| `crates/incin-backends/src/cpu/gradcheck.rs` | 229 | panic:1, unwrap:6, expect:2, public:1 |
| `crates/incin-backends/src/cpu/mod.rs` | 273 | expect:1, public:15 |
| `crates/incin-backends/src/cpu/ops/conv.rs` | 1164 | panic:1, unwrap:24, expect:1, public:3 |
| `crates/incin-backends/src/cpu/ops/elementwise.rs` | 1752 | panic:3, unwrap:81, expect:2, public:5 |
| `crates/incin-backends/src/cpu/ops/elementwise_kernel.rs` | 2274 | panic:1, unwrap:27, public:9, unsafe:104 |
| `crates/incin-backends/src/cpu/ops/embedding.rs` | 255 | panic:1, unwrap:8, expect:2, public:1 |
| `crates/incin-backends/src/cpu/ops/loss.rs` | 521 | panic:1, unwrap:26, expect:3 |
| `crates/incin-backends/src/cpu/ops/matmul.rs` | 1357 | panic:1, unwrap:40, expect:6, public:3, unsafe:11 |
| `crates/incin-backends/src/cpu/ops/mod.rs` | 63 | unsupported:1, public:13 |
| `crates/incin-backends/src/cpu/ops/module.rs` | 133 | - |
| `crates/incin-backends/src/cpu/ops/norm.rs` | 632 | panic:1, unwrap:15, public:2 |
| `crates/incin-backends/src/cpu/ops/optimizer.rs` | 77 | unsupported:1 |
| `crates/incin-backends/src/cpu/ops/pool.rs` | 648 | panic:1, unwrap:21, expect:3, public:3 |
| `crates/incin-backends/src/cpu/ops/quant.rs` | 377 | panic:2, unwrap:5, unsupported:7, unsafe:3 |
| `crates/incin-backends/src/cpu/ops/reduce.rs` | 1469 | panic:4, unwrap:41, expect:12, public:2 |
| `crates/incin-backends/src/cpu/ops/shape_ops.rs` | 2016 | panic:3, unwrap:55, expect:12, unsupported:6 |
| `crates/incin-backends/src/cpu/storage.rs` | 768 | panic:6, unwrap:11, expect:1, public:23, unsafe:1 |
| `crates/incin-backends/src/cpu/stride.rs` | 170 | panic:1, unwrap:3, public:4 |
| `crates/incin-backends/src/cpu/tape.rs` | 572 | panic:2, unwrap:11, expect:2, public:9 |
| `crates/incin-backends/src/cpu/typed_kernel.rs` | 133 | public:4 |
| `crates/incin-backends/src/cpu/var.rs` | 127 | panic:1, unwrap:13, public:4 |
| `crates/incin-backends/src/cuda/backend.rs` | 2902 | unwrap:84, expect:70, unsupported:6, public:5 |
| `crates/incin-backends/src/cuda/executor.rs` | 552 | - |
| `crates/incin-backends/src/cuda/gpu.rs` | 170 | unwrap:2, expect:1, public:13 |
| `crates/incin-backends/src/cuda/mod.rs` | 18 | public:8 |
| `crates/incin-backends/src/cuda/ops/conv.rs` | 347 | unwrap:8, expect:4, public:6, unsafe:4 |
| `crates/incin-backends/src/cuda/ops/elementwise.rs` | 825 | unwrap:14, public:2, unsafe:2 |
| `crates/incin-backends/src/cuda/ops/embedding.rs` | 156 | unwrap:6, expect:2, public:2, unsafe:2 |
| `crates/incin-backends/src/cuda/ops/kernels.rs` | 14 | public:9 |
| `crates/incin-backends/src/cuda/ops/loss.rs` | 79 | unwrap:3, expect:1, public:1, unsafe:1 |
| `crates/incin-backends/src/cuda/ops/matmul.rs` | 93 | unwrap:3, expect:1, public:1, unsafe:1 |
| `crates/incin-backends/src/cuda/ops/mod.rs` | 43 | public:12 |
| `crates/incin-backends/src/cuda/ops/norm.rs` | 414 | unwrap:1, public:2, unsafe:2 |
| `crates/incin-backends/src/cuda/ops/pool.rs` | 418 | unwrap:14, expect:7, public:6, unsafe:6 |
| `crates/incin-backends/src/cuda/ops/quant.rs` | 168 | unwrap:2, expect:2, public:2, unsafe:2 |
| `crates/incin-backends/src/cuda/ops/reduce.rs` | 562 | unwrap:1, public:4, unsafe:4 |
| `crates/incin-backends/src/cuda/ops/shape.rs` | 312 | unwrap:5, expect:3, public:5, unsafe:2 |
| `crates/incin-backends/src/cuda/storage.rs` | 242 | expect:9, public:8 |
| `crates/incin-backends/src/cuda/tape.rs` | 144 | unwrap:1, public:9 |
| `crates/incin-backends/src/descriptor_bind.rs` | 552 | unwrap:4, expect:14, unsupported:2, public:11 |
| `crates/incin-backends/src/detect.rs` | 246 | expect:1, public:5 |
| `crates/incin-backends/src/dispatch.rs` | 1678 | unsupported:1, public:6 |
| `crates/incin-backends/src/dispatch_executor.rs` | 381 | - |
| `crates/incin-backends/src/dist/collective.rs` | 95 | public:7 |
| `crates/incin-backends/src/dist/mod.rs` | 36 | public:8 |
| `crates/incin-backends/src/dist/nccl.rs` | 2948 | panic:1, unwrap:64, public:40, unsafe:2 |
| `crates/incin-backends/src/dist/reference.rs` | 682 | public:15 |
| `crates/incin-backends/src/dist/tuning.rs` | 1378 | public:74 |
| `crates/incin-backends/src/dtype_policy.rs` | 188 | unwrap:1, unsupported:1, public:4 |
| `crates/incin-backends/src/external/candle/backend.rs` | 284 | unwrap:6 |
| `crates/incin-backends/src/external/candle/convert.rs` | 72 | unsupported:2, public:4 |
| `crates/incin-backends/src/external/candle/executor.rs` | 236 | public:5 |
| `crates/incin-backends/src/external/candle/mod.rs` | 57 | unwrap:3, public:5 |
| `crates/incin-backends/src/external/candle/ops/creation.rs` | 122 | unsupported:1 |
| `crates/incin-backends/src/external/candle/ops/float.rs` | 161 | unsupported:1 |
| `crates/incin-backends/src/external/candle/ops/loss.rs` | 107 | - |
| `crates/incin-backends/src/external/candle/ops/mod.rs` | 15 | - |
| `crates/incin-backends/src/external/candle/ops/module.rs` | 236 | unsupported:1 |
| `crates/incin-backends/src/external/candle/ops/numeric.rs` | 46 | - |
| `crates/incin-backends/src/external/candle/ops/optimizer.rs` | 9 | - |
| `crates/incin-backends/src/external/candle/ops/quant.rs` | 38 | unsupported:6 |
| `crates/incin-backends/src/external/candle/ops/reduce.rs` | 197 | unsupported:2 |
| `crates/incin-backends/src/external/candle/ops/tensor.rs` | 261 | unsupported:1 |
| `crates/incin-backends/src/external/conformance.rs` | 613 | public:14 |
| `crates/incin-backends/src/external/mod.rs` | 25 | public:3 |
| `crates/incin-backends/src/iteration.rs` | 532 | unwrap:11, expect:1, public:11 |
| `crates/incin-backends/src/kernel.rs` | 1712 | panic:5, unwrap:23, public:18 |
| `crates/incin-backends/src/lib.rs` | 116 | public:31 |
| `crates/incin-backends/src/metal/backend.rs` | 1892 | unsupported:7, public:3 |
| `crates/incin-backends/src/metal/executor.rs` | 374 | - |
| `crates/incin-backends/src/metal/mod.rs` | 24 | public:11 |
| `crates/incin-backends/src/metal/mps.rs` | 233 | public:14 |
| `crates/incin-backends/src/metal/shaders/mod.rs` | 87 | public:7 |
| `crates/incin-backends/src/metal/storage.rs` | 318 | unwrap:4, unsupported:2, public:15 |
| `crates/incin-backends/src/metal/tape.rs` | 72 | public:9 |
| `crates/incin-backends/src/metal/tuning.rs` | 425 | panic:1, unwrap:9, public:13 |
| `crates/incin-backends/src/simd.rs` | 51 | public:2 |
| `crates/incin-backends/src/telemetry.rs` | 72 | public:5 |
| `crates/incin-backends/src/tuning/cache.rs` | 1017 | public:42 |
| `crates/incin-backends/src/tuning/identity.rs` | 1352 | expect:1, public:65, unsafe:2 |
| `crates/incin-backends/src/tuning/service.rs` | 1255 | public:61 |
| `crates/incin-backends/src/tuning/signature.rs` | 416 | public:27 |
| `crates/incin-backends/src/tuning/telemetry.rs` | 148 | public:7 |
| `crates/incin-backends/src/tuning.rs` | 672 | panic:1, unwrap:16, public:31 |
| `crates/incin-backends/src/unsupported.rs` | 366 | unsupported:36, public:5 |
| `crates/incin-backends/src/wgpu/backend.rs` | 3209 | panic:1, unwrap:2, expect:32, unsupported:8, public:6 |
| `crates/incin-backends/src/wgpu/device.rs` | 51 | expect:2, public:2 |
| `crates/incin-backends/src/wgpu/dispatch.rs` | 774 | public:18 |
| `crates/incin-backends/src/wgpu/executor.rs` | 554 | - |
| `crates/incin-backends/src/wgpu/mod.rs` | 28 | public:9 |
| `crates/incin-backends/src/wgpu/pipeline.rs` | 47 | public:1 |
| `crates/incin-backends/src/wgpu/storage.rs` | 216 | unwrap:3, expect:2, public:12 |
| `crates/incin-backends/src/wgpu/tape.rs` | 178 | public:9 |
| `crates/incin-backends/src/wgpu/tests.rs` | 1419 | unwrap:154, expect:16 |
| `crates/incin-core/src/compiled/alloc.rs` | 234 | public:16 |
| `crates/incin-core/src/compiled/artifact.rs` | 205 | public:15 |
| `crates/incin-core/src/compiled/capture.rs` | 104 | public:5 |
| `crates/incin-core/src/compiled/fold.rs` | 68 | public:7 |
| `crates/incin-core/src/compiled/fusion.rs` | 167 | public:6 |
| `crates/incin-core/src/compiled/manifest.rs` | 87 | public:5 |
| `crates/incin-core/src/compiled/mod.rs` | 24 | public:16 |
| `crates/incin-core/src/compiled/plan.rs` | 134 | public:10 |
| `crates/incin-core/src/compiled/tuning.rs` | 65 | public:4 |
| `crates/incin-core/src/dist/collective.rs` | 258 | public:14 |
| `crates/incin-core/src/dist/context.rs` | 1358 | public:62 |
| `crates/incin-core/src/dist/data_parallel.rs` | 276 | public:21 |
| `crates/incin-core/src/dist/fsdp.rs` | 312 | public:23 |
| `crates/incin-core/src/dist/mesh.rs` | 1046 | expect:1, public:48 |
| `crates/incin-core/src/dist/mod.rs` | 95 | public:21 |
| `crates/incin-core/src/dist/pipeline.rs` | 912 | public:49 |
| `crates/incin-core/src/dist/placement.rs` | 532 | public:21 |
| `crates/incin-core/src/dist/plan.rs` | 2306 | unsupported:1, public:117 |
| `crates/incin-core/src/dist/rule.rs` | 712 | public:24 |
| `crates/incin-core/src/dist/tensor_parallel.rs` | 639 | public:32 |
| `crates/incin-core/src/distributions/mod.rs` | 342 | unwrap:7, public:11 |
| `crates/incin-core/src/err.rs` | 273 | unsupported:2, public:5 |
| `crates/incin-core/src/exec/capability.rs` | 383 | public:13 |
| `crates/incin-core/src/exec/context.rs` | 142 | public:18 |
| `crates/incin-core/src/exec/meta.rs` | 409 | unwrap:6, public:20 |
| `crates/incin-core/src/exec/mod.rs` | 103 | public:23 |
| `crates/incin-core/src/exec/policy.rs` | 488 | public:36 |
| `crates/incin-core/src/exec/precision.rs` | 206 | public:12 |
| `crates/incin-core/src/exec/proof.rs` | 390 | panic:1, unwrap:3, public:13 |
| `crates/incin-core/src/exec/request.rs` | 54 | public:4 |
| `crates/incin-core/src/exec/rule.rs` | 514 | public:11 |
| `crates/incin-core/src/exec/spec.rs` | 1403 | public:41 |
| `crates/incin-core/src/exec/tape.rs` | 325 | unwrap:1, expect:1, public:19 |
| `crates/incin-core/src/graph.rs` | 479 | public:15 |
| `crates/incin-core/src/io/gguf.rs` | 302 | unsupported:1, public:12 |
| `crates/incin-core/src/io/inspect.rs` | 339 | public:4 |
| `crates/incin-core/src/io/limits.rs` | 150 | public:7 |
| `crates/incin-core/src/io/mlx.rs` | 35 | public:2 |
| `crates/incin-core/src/io/mod.rs` | 16 | public:8 |
| `crates/incin-core/src/lib.rs` | 123 | public:49 |
| `crates/incin-core/src/metrics/mod.rs` | 314 | public:21 |
| `crates/incin-core/src/nn/activation.rs` | 454 | public:9 |
| `crates/incin-core/src/nn/adaptive_avg_pool2d.rs` | 71 | public:2 |
| `crates/incin-core/src/nn/avg_pool2d.rs` | 74 | public:2 |
| `crates/incin-core/src/nn/batch_norm.rs` | 179 | public:3 |
| `crates/incin-core/src/nn/conv1d.rs` | 369 | unwrap:1, public:3 |
| `crates/incin-core/src/nn/conv2d.rs` | 383 | unwrap:1, public:3 |
| `crates/incin-core/src/nn/dropout.rs` | 92 | public:2 |
| `crates/incin-core/src/nn/embedding.rs` | 128 | public:3 |
| `crates/incin-core/src/nn/flatten.rs` | 44 | public:2 |
| `crates/incin-core/src/nn/init.rs` | 49 | public:1 |
| `crates/incin-core/src/nn/layer_norm.rs` | 121 | public:3 |
| `crates/incin-core/src/nn/linear.rs` | 449 | unwrap:2, public:6 |
| `crates/incin-core/src/nn/loss.rs` | 317 | public:25 |
| `crates/incin-core/src/nn/lstm.rs` | 479 | public:5 |
| `crates/incin-core/src/nn/max_pool2d.rs` | 75 | public:2 |
| `crates/incin-core/src/nn/mod.rs` | 117 | public:44 |
| `crates/incin-core/src/nn/module.rs` | 1078 | public:23 |
| `crates/incin-core/src/nn/module_optional.rs` | 44 | - |
| `crates/incin-core/src/nn/optional.rs` | 47 | public:3 |
| `crates/incin-core/src/nn/param.rs` | 670 | public:22 |
| `crates/incin-core/src/nn/rms_norm.rs` | 128 | public:3 |
| `crates/incin-core/src/nn/rnn.rs` | 384 | unwrap:3, public:5 |
| `crates/incin-core/src/nn/save.rs` | 441 | public:12 |
| `crates/incin-core/src/nn/stats.rs` | 313 | unwrap:3, public:7 |
| `crates/incin-core/src/onnx_exporter.rs` | 187 | public:5 |
| `crates/incin-core/src/onnx_pb.rs` | 6 | public:1 |
| `crates/incin-core/src/optim/mod.rs` | 447 | unwrap:2, public:18 |
| `crates/incin-core/src/optim/scheduler.rs` | 135 | public:9 |
| `crates/incin-core/src/serialize.rs` | 384 | public:12 |
| `crates/incin-core/src/shapes/arithmetic.rs` | 10 | public:2 |
| `crates/incin-core/src/shapes/broadcast.rs` | 477 | public:3 |
| `crates/incin-core/src/shapes/buf.rs` | 554 | public:38 |
| `crates/incin-core/src/shapes/concat.rs` | 57 | public:2 |
| `crates/incin-core/src/shapes/dim.rs` | 305 | public:4 |
| `crates/incin-core/src/shapes/error.rs` | 405 | public:10 |
| `crates/incin-core/src/shapes/idx.rs` | 298 | unwrap:1, public:7 |
| `crates/incin-core/src/shapes/mod.rs` | 73 | public:40 |
| `crates/incin-core/src/shapes/named.rs` | 53 | public:2 |
| `crates/incin-core/src/shapes/reshape.rs` | 205 | public:4 |
| `crates/incin-core/src/shapes/shape.rs` | 630 | unwrap:1, public:18 |
| `crates/incin-core/src/shapes/shape_ops.rs` | 71 | public:4 |
| `crates/incin-core/src/shapes/spatial.rs` | 570 | public:7 |
| `crates/incin-core/src/shapes/stack.rs` | 38 | public:1 |
| `crates/incin-core/src/shapes/tail_shape.rs` | 397 | unwrap:3, public:6 |
| `crates/incin-core/src/tensor/arg.rs` | 36 | public:1 |
| `crates/incin-core/src/tensor/arg_into.rs` | 618 | public:4 |
| `crates/incin-core/src/tensor/auto_device.rs` | 132 | unwrap:1, public:8 |
| `crates/incin-core/src/tensor/backend.rs` | 2194 | unsupported:10, public:20 |
| `crates/incin-core/src/tensor/base.rs` | 909 | unwrap:7, public:40, unsafe:1 |
| `crates/incin-core/src/tensor/conv2d.rs` | 148 | public:2 |
| `crates/incin-core/src/tensor/device.rs` | 850 | unwrap:6, expect:2, public:47 |
| `crates/incin-core/src/tensor/dtype.rs` | 282 | public:18, unsafe:2 |
| `crates/incin-core/src/tensor/grad.rs` | 109 | public:5 |
| `crates/incin-core/src/tensor/matmul.rs` | 456 | public:10 |
| `crates/incin-core/src/tensor/mod.rs` | 40 | public:24 |
| `crates/incin-core/src/tensor/ops/binary.rs` | 492 | panic:4, unwrap:24, public:11 |
| `crates/incin-core/src/tensor/ops/index.rs` | 192 | public:4 |
| `crates/incin-core/src/tensor/ops/loss.rs` | 170 | unwrap:6, public:8 |
| `crates/incin-core/src/tensor/ops/manipulation.rs` | 1080 | unwrap:16, public:41, unsafe:4 |
| `crates/incin-core/src/tensor/ops/mod.rs` | 17 | public:8 |
| `crates/incin-core/src/tensor/ops/module.rs` | 62 | public:2 |
| `crates/incin-core/src/tensor/ops/reduce.rs` | 532 | unwrap:24, public:14 |
| `crates/incin-core/src/tensor/ops/unary.rs` | 472 | panic:4, unwrap:32, public:6 |
| `crates/incin-core/src/tensor/tracing.rs` | 1923 | panic:1, unwrap:1, unsupported:1, public:8 |
| `crates/incin-data/src/dataset.rs` | 35 | public:1 |
| `crates/incin-data/src/downloader.rs` | 85 | public:3 |
| `crates/incin-data/src/hub.rs` | 77 | public:8 |
| `crates/incin-data/src/lib.rs` | 73 | public:14 |
| `crates/incin-data/src/loader.rs` | 331 | unwrap:1, expect:1, public:6 |
| `crates/incin-data/src/transforms/mod.rs` | 287 | unwrap:3, public:13 |
| `crates/incin-data/src/vision/mnist.rs` | 177 | unwrap:16, public:2 |
| `crates/incin-data/src/vision/mod.rs` | 3 | public:1 |
| `crates/incin-diagnostics/src/lib.rs` | 1740 | unwrap:20, public:66 |
| `crates/incin-lsp/src/bin/mock_rust_analyzer.rs` | 58 | expect:2 |
| `crates/incin-lsp/src/config.rs` | 79 | public:5, unsafe:3 |
| `crates/incin-lsp/src/frame.rs` | 117 | unwrap:14, public:2 |
| `crates/incin-lsp/src/lib.rs` | 26 | public:3 |
| `crates/incin-lsp/src/main.rs` | 80 | unwrap:2, expect:3 |
| `crates/incin-lsp/src/rewrite.rs` | 571 | unwrap:10, public:4 |
| `crates/incin-macros/src/arg_into.rs` | 175 | public:2 |
| `crates/incin-macros/src/axes.rs` | 76 | public:1 |
| `crates/incin-macros/src/distributed_main.rs` | 46 | public:1 |
| `crates/incin-macros/src/einsum.rs` | 129 | public:1 |
| `crates/incin-macros/src/idx.rs` | 208 | public:1 |
| `crates/incin-macros/src/lib.rs` | 342 | unwrap:3, public:16 |
| `crates/incin-macros/src/mesh.rs` | 151 | public:1 |
| `crates/incin-macros/src/module.rs` | 603 | public:2 |
| `crates/incin-macros/src/onnx.rs` | 552 | panic:6, unwrap:5, public:6 |
| `crates/incin-macros/src/parallel_block.rs` | 54 | public:1 |
| `crates/incin-macros/src/placement.rs` | 124 | public:1 |
| `crates/incin-macros/src/rank.rs` | 329 | expect:3, public:3 |
| `crates/incin-macros/src/safetensors.rs` | 249 | unwrap:3, public:3 |
| `crates/incin-macros/src/shape.rs` | 213 | public:2 |
| `crates/incin-macros/src/shape_ops.rs` | 116 | public:1 |
| `crates/incin-telemetry/src/emitter.rs` | 714 | panic:2, unwrap:8, expect:7, public:8, unsafe:4 |
| `crates/incin-telemetry/src/err.rs` | 52 | public:2 |
| `crates/incin-telemetry/src/events.rs` | 173 | expect:3, public:9 |
| `crates/incin-telemetry/src/lib.rs` | 29 | public:10 |
| `crates/incin-telemetry/src/reporter.rs` | 245 | unwrap:12, expect:6, public:1 |
| `crates/incin-telemetry/src/run_dir.rs` | 236 | unwrap:8, expect:6, public:6, unsafe:4 |
| `crates/incin-telemetry/src/transport/file.rs` | 291 | panic:3, unwrap:8, expect:21, public:2 |
| `crates/incin-telemetry/src/transport/mod.rs` | 20 | public:3 |
| `crates/incin-telemetry/src/transport/socket.rs` | 354 | panic:5, expect:17, public:2 |
| `crates/incin-viz/src/app.rs` | 713 | panic:1, public:10 |
| `crates/incin-viz/src/dispatch.rs` | 150 | panic:1, expect:2, public:5 |
| `crates/incin-viz/src/err.rs` | 53 | public:2 |
| `crates/incin-viz/src/lib.rs` | 18 | public:5 |
| `crates/incin-viz/src/main.rs` | 158 | - |
| `crates/incin-viz/src/panels/graph.rs` | 454 | public:3 |
| `crates/incin-viz/src/panels/loss.rs` | 104 | public:2 |
| `crates/incin-viz/src/panels/mod.rs` | 19 | public:6 |
| `crates/incin-viz/src/panels/norms.rs` | 145 | public:3 |
| `crates/incin-viz/src/panels/panic_test.rs` | 54 | panic:1, public:1 |
| `crates/incin-viz/src/panels/scalar.rs` | 98 | public:2 |
| `crates/incin-viz/src/panels/system.rs` | 110 | public:2 |
| `crates/incin-viz/src/transport_reader.rs` | 198 | expect:15, public:3 |
| `crates/incin-viz-plugin-api/src/err.rs` | 28 | public:2 |
| `crates/incin-viz-plugin-api/src/event.rs` | 96 | public:5 |
| `crates/incin-viz-plugin-api/src/keymap.rs` | 40 | public:2 |
| `crates/incin-viz-plugin-api/src/lib.rs` | 76 | public:13 |
| `crates/incin-viz-plugin-api/src/panel.rs` | 45 | public:1 |
| `crates/incin-viz-plugin-api/src/plugin.rs` | 13 | public:1 |
| `crates/incin-viz-plugin-api/src/render_ctx.rs` | 67 | public:8 |


---

# APPENDIX E - Crate scale and scan summary

| Crate | Production files | Approx. lines | Lexical flags |
|---|---:|---:|---|
| `incin` | 6 | 2542 | expect:2, public:119, unwrap:6 |
| `incin-backends` | 102 | 53420 | expect:217, panic:40, public:857, unsafe:149, unsupported:99, unwrap:913 |
| `incin-core` | 105 | 38846 | expect:4, panic:10, public:1524, unsafe:7, unsupported:15, unwrap:151 |
| `incin-data` | 8 | 1068 | expect:1, public:48, unwrap:20 |
| `incin-diagnostics` | 1 | 1740 | public:66, unwrap:20 |
| `incin-lsp` | 6 | 931 | expect:5, public:14, unsafe:3, unwrap:26 |
| `incin-macros` | 15 | 3367 | expect:3, panic:6, public:42, unwrap:11 |
| `incin-telemetry` | 9 | 2114 | expect:60, panic:10, public:43, unsafe:8, unwrap:36 |
| `incin-viz` | 13 | 2274 | expect:17, panic:3, public:44 |
| `incin-viz-plugin-api` | 7 | 365 | public:32 |


---

# APPENDIX F - Instructions for maintaining this report

When implementation changes land:

1. Never replace a finding with a checked box without attaching evidence.
2. Update source locations when code moves.
3. Split a task if one acceptance criterion cannot be proven by the same change.
4. Mark hardware-dependent work `BLOCKED` with missing hardware named.
5. Add newly discovered defects rather than narrowing task scope silently.
6. Keep generated capability tables and public API snapshots in version control.
7. Preserve this audit as a historical baseline; create dated addenda rather than rewriting history to imply the old claims were true.
