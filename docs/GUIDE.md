# The Incin Guide

A tour of the whole system: every subsystem, how the pieces fit, and the
idiomatic way to use each one. Where `docs/README.md`'s other documents are
contracts, status reports, or generated inventories, this one is prose — read
it first if you are new to the tree, then use the others as reference.

Every claim below was checked against the source it describes while this was
written (2026-08-06). Where a feature is a prototype, partial, or has a
documented gap, that is stated plainly rather than smoothed over — the
generated documents this guide points to (`docs/capabilities.md`,
`docs/OPERATION_SEMANTICS.md`, `audit-evidence/FND-005/cpu-migration-status.md`)
are the ones that stay current automatically; if this guide and one of them
disagree later, believe the generated one and fix this file.

## Table of contents

1. [The crate map](#1-the-crate-map)
2. [The mental model](#2-the-mental-model)
3. [The type-level shape system](#3-the-type-level-shape-system)
4. [Tensors: creation, dtype, device](#4-tensors-creation-dtype-device)
5. [Operations: the stable surface today](#5-operations-the-stable-surface-today)
6. [The canonical execution architecture](#6-the-canonical-execution-architecture)
7. [The target API: allocation targets](#7-the-target-api-allocation-targets)
8. [Backends and how to author one](#8-backends-and-how-to-author-one)
9. [Autograd](#9-autograd)
10. [Modules and the `#[module]` macro](#10-modules-and-the-module-macro)
11. [Errors and the panic policy](#11-errors-and-the-panic-policy)
12. [Feature flags](#12-feature-flags)
13. [Idioms: how this codebase wants to be written](#13-idioms-how-this-codebase-wants-to-be-written)
14. [What's next](#14-whats-next)

---

## 1. The crate map

Incin is a workspace, not one crate. What you depend on is `incin`; it
re-exports the rest.

| Crate | Role |
|---|---|
| `incin-core` | The `Tensor` type, the shape/dtype/device type-level system, autograd, `nn`/`optim`/`metrics`, the operation catalog and descriptor contract, distributed primitives. `no_std` unless a feature says otherwise |
| `incin-backends` | Concrete backends: `cpu`, `cuda`, `wgpu`, `metal`, and the third-party `external::candle` adapter. Also `incin_backends::target`, the allocation-target prototype (§7) |
| `incin-macros` | Procedural macros: `s!`, `shape!`, `idx!`, `tensor!`, `#[module]`, `dim!`, `mesh!`/`parallel!`/`placement!`, `model!`/`import_model!` |
| `incin-data` | `Dataset`, `DataLoader`, vision datasets, transforms, HuggingFace Hub downloading |
| `incin-viz` / `incin-viz-plugin-api` | Graph visualization tooling, plugin surface for it |
| `incin-telemetry` | Structured run emission (`Emitter`) |
| `incin` | The facade. Re-exports the above under `incin::prelude`, plus `incin::nn`, `incin::optim`, `incin::data`, `incin::backend_authoring`, `incin::experimental` |

You will almost always write `use incin::prelude::*;` and nothing else. The
`prelude` module (`crates/incin/src/lib.rs:471`) is the curated, high-frequency
surface; `incin::backend_authoring` and `incin::experimental` are opt-in and
named that way on purpose — reaching for them is a signal, not an accident.

## 2. The mental model

Three ideas run through everything else in this guide:

**Shapes are usually types, not values.** `Tensor<s![2, 3, 224, 224], B>` says
what shape a value has in a way `rustc` checks at every call site — a `matmul`
between operands whose inner dimensions disagree does not compile. `Dyn` opts
a shape out of that when it is not known until runtime; you can mix the two in
one program, and the two rows in the Quick Start example in
`crates/incin/src/lib.rs` show both.

**The backend is a type parameter, not a runtime switch.** `Tensor<S, B, ...>`
carries its backend in `B`. `CpuBackendImpl<T, D>`, `CudaBackendImpl<T, D>`,
and so on each implement the traits `Tensor` needs; which one a given tensor
uses is fixed at compile time by which `B` you wrote, not decided by a global
"current device". There is deliberately no such global — see
[`incin-auto-device-selection`](../docs/README.md) territory: `best_device!()`
picks `B` at compile time from enabled features, `detect_device()` probes
hardware at runtime and returns a value, and the two answer different
questions.

**Ordinary tensor methods use one canonical execution architecture.** Their
backend bounds name an exact operation descriptor (`op::X`) and execution is
validated before it reaches a backend. The former operation-family traits are
hidden compatibility adapters for backend implementations and tests; they are
not required by `Backend`, are not part of the normal facade prelude, and are
not a second application-facing execution model.

## 3. The type-level shape system

### `Shape`, `Dim`, and what "static" means

`Shape` (`crates/incin-core/src/shapes/shape.rs`) is the trait every shape
type implements. Three families implement it:

- **A recursive `DimCons` shape**, e.g. `DimCons<U2, DimCons<U3, Nil>>` —
  every axis known at compile time. `s![2, 3]` expands to exactly this.
- **`Dyn`** — rank itself is unknown until a value exists. `Tensor<Dyn, B>`.
- **A recursive shape mixing `usize` and `Dim` types** — rank is known, some
  axes are not. `s![usize, 3]`, or a named axis via `dim!(Batch)` used as
  `s![Batch, 128]`.

Two facts about a `Shape` are readable from the type alone, and both exist
because a backend executor cannot ask for an optional stronger static-shape
implementation on stable Rust without specialization. Instead both are restated
as an `Option` on `Shape` itself, defaulting to "unknown", so any `S: Shape`
can be asked without an extra bound:

- **`Shape::PROOF: ProofLevel`** — `Static` (every axis fixed by the type),
  `Mixed` (rank known, at least one axis is not), or `Dynamic` (rank itself
  is runtime). This is what a `Validated<O>` carries as its
  [`proof_level()`](../crates/incin-core/src/exec/proof.rs) — see §6.
- **`Shape::STATIC_NUMEL: Option<usize>`** — the element count, when every
  axis is static; `None` otherwise. `if let Some(n) = S::STATIC_NUMEL`
  collapses to one arm at monomorphization, not a runtime branch, so a
  generic executor can specialize on it for free. `Dim::STATIC_EXTENT` is the
  per-axis version this folds over.

Fully static shapes expose their proof through `Shape::STATIC_NUMEL`; runtime
dimensions are always resolved into `ShapeBuf`. Layer constructors use that
validated shape contract when sizing their weights.

### The `s!` macro

`s![2, 3, 224, 224]` expands to a recursive `DimCons` type whose extents are
raw typenum binary integers. It is a **type**, used as
`Tensor<s![2, 3, 224, 224], B>`.
Mixed and named forms:

```rust
use incin::prelude::*;
dim!(BatchSize);

type Image = s![3, 224, 224];
type Batched = s![BatchSize, 128];   // named axis, size decided at runtime
type NamedStatic = s![BatchSize: 25, 128]; // named axis with static extent
type Loose = s![usize, 128];         // unnamed runtime axis
```

`idx![0..5, .., 15..30]` is the slicing analogue — it builds the type
`.slice_idx::<...>()` expects, translating `a..b` to a bounded `Slice`, `..`
to "take the whole axis", and `-1` to `InferDim` for reshape.

### Why two type-level systems (`s!` vs `shape!`)

`s!` names a *type*. Allocation targets (§7) need a *value* to pass to a
constructor, and that value has to encode the same static/runtime split —
`shape!` is that value-level counterpart:

```rust
let w = gpu.zeros(shape![128, 784])?;   // Tensor<s![128, 784], ..>
let x = gpu.zeros(shape![batch, 784])?; // Tensor<s![usize, 784], ..>
```

`shape!`'s inference is syntactic: a literal is static, anything else
(including a named `const`) is a runtime axis — read as a weaker shape, never
a wrong one. Where the stronger, fully-static form matters, write it directly:
`s![32, 784]`.

## 4. Tensors: creation, dtype, device

`Tensor<S, B, K, G, P>` — shape, backend, element type (defaults to the
backend's float element), gradient-tracking marker (`Grad` or `NoGrad`,
defaults to `Grad`), and placement (defaults to `Local`). Most code only ever
writes the first two.

**Two ways to create one**, and they are not interchangeable conveniences —
each is right for a different thing:

```rust
// 1. The classic constructor: shape is the type parameter, `Arg` is the
//    per-shape argument (`()` for a static shape, a `Vec<usize>` for `Dyn`).
let a = Tensor::<s![2, 3], Backend>::zeros(())?;
let b = Tensor::<Dyn, Backend>::ones(vec![2, 3])?;

// 2. `tensor!`, the array-literal convenience — shape and dtype inferred
//    from the literal's own nesting and suffixes, always on the default CPU
//    backend. No `device:`/`backend:` clause; use a target for that.
let c = tensor![[1.0, 2.0], [3.0, 4.0]]?;         // shape [2,2], f32
let d = tensor![1, 2, 3]?;                        // i64, matching torch.tensor's default

// 3. (feature = "target-api") An allocation target — see §7 — for anything
//    that needs to say *where*.
let e = Cpu.zeros(s![2, 3])?;
```

Dtype comes from `K`. `Dyn` as the dtype parameter (`Tensor<S, B, Dyn>`) keeps
the element type itself as a runtime tag rather than a compile-time one —
different question from `S = Dyn`, and the two compose independently.

Device comes from `B`: `Tensor<S, CpuBackendImpl<f32, Cpu>>` vs `Tensor<S,
CudaBackendImpl<f32, Cuda>>`, or the `incin::Tensor` alias's default,
`DefaultBackend = IncinBackend<f32, Cpu>` — present only when the `cpu`
feature is on, and *not* substituted by anything else when it is off: every
alias in `crates/incin/src/lib.rs` that defaults `B` to it is declared twice,
once with the default and once (behind `#[cfg(not(feature = "cpu"))]`)
without one, so disabling `cpu` without picking another backend is a clear
"expected 2 generic arguments" at your own call site rather than a trait-bound
error three layers down.

## 5. Operations: the stable surface today

`Tensor` methods such as `add`, `matmul`, `reshape`, reductions, losses, and
creation all bind an exact catalog descriptor and execute through the
validated dispatch contract. Read `docs/OPERATION_SEMANTICS.md` for the exact
contract of every catalog operation (broadcasting rule, dtype rule, gradient
rule, output shape) and `docs/capabilities.md` for backend support levels
(`Native`/`Composed`/`Fallback`/`Unsupported`). Both are generated from source
and re-checked by tests.

Backend authors may encounter the compatibility adapters under the explicitly
named legacy authoring tier while migrating an implementation. Application
code should use the ordinary tensor methods and does not need those traits.

## 6. The canonical execution architecture

This section explains the internal architecture behind the ordinary tensor
methods. Application code normally does not call the dispatcher directly —
skip to §7 if you just want to allocate tensors on a specific device.

### Why it exists

The old family adapters let *any* implementation answer for an operation:
`Backend: TensorOps<Self>` says "this backend has some `matmul`", not "this
backend has *this exact, validated* `matmul`, refusing anything it cannot
prove". Two consequences: a backend cannot be asked "do you support `matmul`
on `f16` at rank 3?" without running it and seeing what happens, and there is
no single point where "here is the operation" and "here is the metadata
proving it is well-formed" travel together, sealed, into the kernel. The
canonical path is now the ordinary tensor execution architecture. The family
traits remain only as hidden compatibility adapters for backend
implementations and tests while their remaining methods are extracted.

### The pieces, in the order data flows through them

1. **`OPERATION_CATALOG`** (`crates/incin-core/src/operation_catalog.rs`) —
   one macro-generated declaration of all 174 operations: an identity (`op::X`
   marker type), a `SemanticProfile` (broadcasting/dtype/gradient/output
   rules), an `Attributes` type, an operand arity, and an `ExecutionSite`
   (whether `Execute` can carry it at all — more on this below). Every other
   piece in this list is generated *from* this declaration, which is what
   makes "advertised" and "implemented" impossible to drift apart.
2. **`Descriptor<O>`** — attributes plus inferred output metadata for one
   `op::X`. Building one runs `O`'s `infer_outputs`, so the shape a backend
   receives was derived from the real operand metadata, never accepted as a
   caller's claim.
3. **`Validated<O>`** — a `Descriptor<O>` plus a `ProofLevel`
   (`Static`/`Mixed`/`Dynamic`, read off the frontend's `Shape::PROOF`, §3).
   The only public constructor is validation itself — see
   `docs/INVARIANT_TYPES.md`'s "proof token" row. `proof_level()` is the one
   public accessor.
4. **`Execute<O>`** (`crates/incin-core/src/tensor/backend.rs`) — what a
   backend implements per exact operation. The required method is
   `execute_shaped<S: Shape>(&self, request: ExecutionRequest<'_, O, Self>)`;
   `execute(...)` is a **provided** default that calls `execute_shaped::<Dyn>`.
   That direction is deliberate: a required `execute` with a defaulted
   `execute_shaped` would let a backend implement only the erased form and
   never specialize on `S` at all, silently. `S` is not decoration — the CPU
   creation family reads `S::STATIC_NUMEL` to skip a runtime element-count
   computation entirely when the caller's shape is fully static (see
   `crates/incin-backends/src/cpu/stride.rs`'s `numel_for`); that is a real,
   measured effect (an order of magnitude at the point it applies, several
   percent end-to-end for the cheapest allocation), not a type-system
   flourish with no consumer.
5. **`Capabilities` / `CapabilityQuery` / `SupportLevel`**
   (`crates/incin-core/src/exec/capability.rs`) — before a backend is asked to
   execute, `dispatch::execute[_shaped]` queries its exact capability row for
   this `(operation, dtype, layout, rank, training, math_mode)`. `Unsupported`
   carries a typed `UnsupportedReason`, never a string a caller has to parse.
6. **`dispatch::execute_shaped::<O, B, S>`** /
   **`dispatch::execute::<O, B>`** (= `execute_shaped::<O, B, Dyn>`)
   (`crates/incin-core/src/exec/dispatch.rs`) — the single route: validate,
   query capability, call `Execute::execute_shaped`. `S` here is a type
   argument the *caller* supplies, not derived from the descriptor — it has to
   travel beside the attributes rather than be read off them, or a caller
   could claim `ShapeEvidence::of::<s![2, 3]>()` next to metadata describing
   something else and be believed.

### What has an executor today, and what does not

`Execute` cannot carry every operation. Sixteen sit at a non-backend
`ExecutionSite` (`Mutation` — writes through an operand, e.g.
`Tensor::add_`; `DeviceTransfer` — produces storage on another backend;
`GraphState` — acts on autograd state, e.g. `backward`) and are excluded from
"migrated" counts rather than counted as unwritten. Of the 161 operations
`Execute` *can* carry, the CPU backend implements 156; the remaining 5 are
each blocked by a stated, specific gap in the descriptor contract (not
laziness) — `audit-evidence/FND-005/cpu-migration-status.md` is the
machine-checked, regenerated-on-drift account of exactly which and why.
`embedding` and `cross_entropy_loss` were the last two of that kind: their
operands admit different dtypes by construction, and the fix was not a
`CapabilityRule` struct change but a row stating the honest *union* of both
operands' dtypes (`INDEX_AND_F32_DTYPES` in
`crates/incin-backends/src/capability.rs`) — the same technique
`descriptor_min_rank` already used for rank — relying on the descriptor's own
per-operand contract, which runs first, to reject the wrong combination
before any capability query does, and on an executor-side `f32_only` check
for whichever operand the union cannot pin down alone. That blocker is now
closed; none of the remaining 5 is waiting on a dtype set.

Backend-executable operations in the stable tensor surface now use this path.
The remaining family-trait references are backend-local adapters for fused
kernels, host readback, tracing, and special execution sites. They do not form
an alternate stable tensor execution path.

## 7. The target API: allocation targets

Feature `target-api`. A **target** is a value that knows where and how to
allocate — a device (`Cpu`, `Wgpu::new(0)`, ...) or a backend value rebound to
a specific dtype (`.with_float::<f64>()`). It has no construction step and
owns no resources; it is a value you pass around, not a runtime handle you
initialize and hold.

```rust
use incin::prelude::*;

let x = Cpu.zeros(s![2, 3])?;                       // static shape, static proof
let y = Cpu.zeros(shape![2, 3])?;                    // same, via the value macro
let batch = 4;
let z = Cpu.zeros(shape![batch, 3])?;                // dynamic batch axis
let w = Cpu.zeros([batch, 3])?;                      // fully dynamic — Shape = Dyn
```

`ShapeSpec` is the trait that makes all four forms above accept the same
method: `Static<S>` (fully compile-time), `Bound<S>` (mixed — carries the
runtime axes it needs), and `[usize; N]` (fully dynamic, `Shape = Dyn`) all
implement it, each producing the `Shape::PROOF` its own staticness earns —
never more.

`.zeros(...)`/`.ones(...)`/`.rand(...)`/`.randn(...)` are the ordinary public
path and use exact descriptor execution bounds. The `_canonical` constructors
are experimental lower-level entry points for descriptor-oriented work and
are available only where a `CanonicalOperation` bound is satisfied (today:
CPU). Prefer the ordinary forms for application code.

`TensorTarget`/`DtypeTarget` extend the same idea to data-carrying and
dtype-rebinding constructors — `gpu.tensor([[1.0, 2.0]])`,
`gpu.with_float::<f64>().zeros(...)`. This whole surface is marked
**experimental** in the prelude comment (`crates/incin/src/lib.rs:496`) — it
is real and tested, not a stub, but its API is not yet frozen the way §5's is.

## 8. Backends and how to author one

`incin::backend_authoring` (feature `backend-authoring`) is the contract a new
backend implements: `StorageBackend` (associated `Storage<K>`, `Device`,
`metadata()`), `Capabilities`, named optional capability views, and — per
operation — `Execute<Descriptor<op::X>>`. The old family traits are available
only under the hidden `backend_authoring::legacy` compatibility namespace.

**`StorageBackend::Storage<K>`** is a physical allocation plus
`TensorMeta` (shape, strides, offset, dtype, device, alignment, capacity — a
proof token per `docs/INVARIANT_TYPES.md`, constructed only through
`TensorMeta::try_new`/`contiguous`). A foreign tensor type that carries no
such metadata can still join the canonical contract by wrapping itself —
`incin_backends::external::candle::CandleStorage` is the worked example:
`CandleBackend`'s own `Storage<K>` stays the raw `candle_core::Tensor` (so its
existing family-trait operations are untouched), while a *separate*
`Execute<MatMulSpec>`/`Execute<ReshapeSpec>` impl operates on the wrapper.
Joining the descriptor contract does not require rewriting the adapter that
already exists.

**Capability declarations** (`crates/incin-backends/src/capability.rs`) are
grouped by *rule shape*, not by operation family — migrating an operation onto
the canonical path is one more name in an existing list, not a new match arm
in every consumer (`docs/FROZEN_FOUNDATIONS.md`'s "the completeness proof"
row). `cpu::canonical`'s `assert_every_advertised_row_executes!` makes an
advertised-but-unimplemented row a compile error, not a runtime surprise.

**Feature isolation.** Every backend feature (`cpu`, `cuda`, `wgpu`, `metal`)
implies `std`; the crate itself is `#![cfg_attr(not(feature = "std"),
no_std)]`. A module that is not feature-gated (like `layout.rs`, shared
between all backends) still has to build in the bare, no-feature
configuration — where nothing implies `std`, and `Vec` must come from
`alloc` explicitly rather than the prelude. The CI job `backend-isolation`
checks each backend feature standalone (`--no-default-features --features
std,<backend>`) precisely because a backend accidentally depending on another
backend being enabled is invisible until someone tries the combination that
was never tested.

## 9. Autograd

`Grad`/`NoGrad` are the two static markers for `Tensor`'s gradient parameter;
`Dyn` is a third, runtime-toggled option for when the choice is not known
until a value exists. A `Var`/`RawVar` (backend-associated) is the tape-linked
form a `Param` in a module holds. `GradMode::Disabled.scope(|| { ... })`
scopes gradient recording off for a closure; `ExecutionContext::with_grad_mode` is the
descriptor-path equivalent §6's `dispatch::execute` reads to build a
`CapabilityQuery`'s `training` flag — the same policy, read from the same
place, whichever path a given operation takes, so the two paths cannot
disagree about whether a call is a training call.

`Tensor::backward()` walks the tape and returns `Gradients`;
`Backend::get_grad::<K>(&tensor, &grads)` reads one tensor's gradient back
out. Both are `GraphState`-sited operations (§6) — outside what `Execute` can
carry, by design, since they act on tape state rather than producing an
allocation.

## 10. Modules and the `#[module]` macro

```rust
use incin::prelude::*;

type Backend = IncinBackend<f32, Cpu>;

#[module]
pub struct MLP {
    net: SeqTy!(
        Linear<s![768, 256], Backend>,
        ReLU,
        Linear<s![256, 10], Backend>
    ),
}

impl MLP {
    pub fn new() -> Result<Self> {
        Ok(Self {
            net: seq!(
                Linear::<s![768, 256], Backend>::build(())?,
                ReLU,
                Linear::<s![256, 10], Backend>::build(())?
            ),
        })
    }
}
```

`#[module]` derives `StateDict` and `Parameters` by walking every field: one
implementing either trait (a layer, or a nested `Sequential`) is aggregated
recursively; a plain field is skipped. `SeqTy!` names the same nested
`Sequential<...>` type `seq!` builds a value of, so a layer list is written
once instead of the field type and the constructor drifting independently.

Layer constructors like `Linear<S, B>` require the shape's static element
count where sizing must be known without a fallible runtime step.
`ShapeValue<S>` and `ShapeBuf` remain the runtime metadata boundary; a layer's
shape parameter may otherwise carry mixed or dynamic extents when its
constructor accepts runtime shape arguments.

## 11. Errors and the panic policy

`incin_core::prelude::Error` is the top-level fallible-path type; every
constructor in this guide returns `Result<T, Error>` (aliased `Result<T>`).
`BackendError` is the narrower type an `Execute` executor returns — typed
variants (`InvalidInput`, `Execution`, `unsupported(name, reason)`, …), never
a bare string a caller has to pattern-match against text. See
`docs/ERROR_CONTRACT.md` for the full category list and which categories are
allowed to panic (essentially none, outside an established internal
invariant already checked at a boundary — `docs/INVARIANT_TYPES.md`'s
"Checked sizes and arithmetic" section states the rule precisely: an `expect`
is permitted only *after* a value has crossed a checked construction
boundary, and represents an internal invariant violation rather than a
public input error).

## 12. Feature flags

| Feature | Enables |
|---|---|
| `cpu` | `CpuBackendImpl`, `DefaultBackend`, `DefaultDevice` |
| `cuda` / `wgpu` / `metal` | The respective accelerator backend |
| `external-candle` | The third-party Candle adapter under `incin::backend_authoring`... `external::candle` |
| `target-api` | §7 — `TargetExt`, `Static`, `Bound`, `ShapeSpec`, `shape!` |
| `backend-authoring` | §8 — the contract for writing a new backend |
| `distributed` / `distributed-nccl` | `incin::experimental::distributed` — mesh, placement, collective planning |
| `train` | The preview automatic `Trainer` under `incin::experimental::training` |
| `autotune` | Preview kernel tuning cache and inspection types |
| `compiled` | Structural compiled-execution prototype — does not execute graphs yet |
| `std` | Lifts `no_std` restrictions; several backend/target/test-utils features imply it |
| `test-utils` | `DummyBackend` and other test-only scaffolding, exported for downstream test code |

`cargo incin doctor` (via `cargo run --bin cargo-incin -- doctor`, or the
library's `doctor` module directly) reports which of these are active,
detected devices, and cache state for the running build.

## 13. Idioms: how this codebase wants to be written

These are policies stated elsewhere in `docs/` and enforced by tests or CI;
collected here because they shape almost every code path this guide
describes.

- **`pub(crate)` by default.** A `pub` item is a long-term contract
  (`docs/API_DESIGN.md`). Internal state (`WgpuDeviceState`), dispatch
  helpers, raw buffers stay `pub(crate)`; a `pub struct` that satisfies a
  public trait (`CpuStorage`) still keeps its *fields* private.
- **Checked arithmetic at every boundary, not just the obvious ones.**
  Element counts, byte lengths, and stride products all use
  `checked_mul`/`ShapeBuf::checked_numel`/`CheckedNumel`/`CheckedByteLen`
  rather than a bare `.iter().product()` — an oversized or crafted shape
  overflowing `usize` silently under release-mode wraparound is exactly the
  class of bug this exists to make impossible (`docs/INVARIANT_TYPES.md`).
- **Invariant-bearing values have exactly one door in.** `TensorMeta`,
  `Validated<O>`, `ShapeBuf`, every ID type — constructed only through the
  function that checks the invariant, never assembled field-by-field by a
  caller. If you find yourself wanting to build one directly, the invariant
  you are about to skip is the reason not to.
- **"Don't Hand-Roll."** `rand`/`rand_distr` for sampling (never a
  hand-written Box-Muller for `randn`), established crates for anything with
  a well-known correct implementation, over a bespoke one that has to be
  independently verified.
- **Generated documents are load-bearing, not decorative.** `docs/README.md`
  lists which files are generated from source and how to regenerate them; a
  test fails if the committed copy and a fresh regeneration disagree. Never
  hand-edit one — edit the source and regenerate.
- **A capability claim and an executor are the same edit.** The pattern in
  §6 and §8: one declaration feeds the capability row, the legacy executor,
  and the canonical executor, so a row that claims support the tree does not
  provide is a compile error, not a fact someone has to notice and file.
- **Report exactly what was measured, including when it's small.** Where this
  guide or the codebase's own doc comments cite a number (the 12.6ns/call
  pointwise-descriptor saving, the order-of-magnitude static-numel saving in
  §6), it was measured on this tree, in release mode, and reported as
  measured — not rounded up into a bigger claim than the data supports.

## 14. What's next

The authoritative, current version of this section is
`docs/FROZEN_FOUNDATIONS.md`'s "Next steps, in dependency order" — read that
directly; it is regenerated in spirit every time a step completes; below is a
snapshot as of this writing plus the smaller, additive work this session left
queued.

**FND-005's remaining path** (each step blocked by the one above it):

1. Let a descriptor carry a payload and a weight set — unblocks
   `tensor_from_data`/`tensor_from_bytes` (need a data payload the current
   `CreationAttributes` has no field for) and `rnn`/`lstm` (need weight
   matrices the current `RecurrentAttributes` cannot name).
2. Add a distribution registry mapping a name and parameter buffer to a
   sampler — unblocks `sample`.
3. Widen `Execute` to reach the sixteen non-backend-sited operations, or
   split them into a contract that can carry them.
4. Finish extracting the remaining optional methods and associated types from
   `Backend`, bounding each stable tensor method by only the capability it
   uses. This is source-breaking for backend implementations and remains the
   principal handoff item.
5. Delete the broad family capability rows, the grouped
   `Execute<MatMulSpec>` adapters, the `cpu::canonical` compatibility adapter,
   and the `the_migration_is_recorded_as_incomplete` test — each exists only
   to keep the dual architecture honest while it is dual.

**Smaller, additive threads also open**, none of them source-breaking:

- Route Metal through `DispatchBackend` — `DispatchStorage`/`DispatchVar`/
  `DispatchGrads` currently have no Metal variant, so a `Dyn`-device
  operation on Apple Silicon returns `BackendUnavailable` even where Metal
  itself implements the operation.
- canonical graph operation metadata and the
  `AxisContract` step referenced in `docs/plan/UX-ARCHITECTURE-HANDOFF.md`.
- Extend the `S::STATIC_NUMEL` specialization in §6 beyond the CPU creation
  family to other shape-sensitive kernels, now that one real, measured
  instance of it exists to model the next one on.
