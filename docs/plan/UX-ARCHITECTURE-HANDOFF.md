# Historical audit: user-facing architecture findings

> **Non-normative snapshot.** This document records an earlier UX/dispatch
> audit and its proposed sequencing. It is retained for migration history, not
> as current architecture guidance. For current truth, use
> [`docs/GUIDE.md`](../GUIDE.md), [`docs/FROZEN_FOUNDATIONS.md`](../FROZEN_FOUNDATIONS.md),
> and [`docs/HANDOFF.md`](../HANDOFF.md).

The dispatch and production-caller counts below describe the checkout audited
when this file was written; they must not override current source or generated
evidence.

**Status:** landed and uncommitted. Nothing here is on the critical path of an
existing ledger task; none of it is a `PROPOSALS.md` row.

**Test counts, as of writing:** 22 (`target_api`, CPU) + 3 (`target_api_wgpu`,
real WGPU execution) + 20 (`tensor_macro`, incl. 7 trybuild snapshots) + 4 new
unit tests + 4 integration tests in `incin-core`. `cargo fmt --check` clean, `clippy` clean on
`incin-core`, `incin-macros`, and `incin-backends --features target-api,cpu`.

**Audience:** whoever picks this up next. This document is meant to be
followed without re-deriving anything. Every claim below was verified by
running the compiler or the test suite, not by reading a doc comment - several
doc comments in this repo are wrong, and they are named where relevant.

---

## 0. Orientation in one page

The review that produced this work asked one question: *which object should be
the user-facing allocation target?* Answering it required auditing how
construction, lowering, and dispatch actually work. That audit found problems
considerably more serious than the ergonomics question, and they are the
reason the work is sequenced the way it is.

**The single most important finding:** `exec::dispatch::execute` - documented
in `docs/FROZEN_FOUNDATIONS.md` as "the single production route from an
operation to a kernel" - **had zero production callers.** All 216 of its call
sites were test code (56 + 128 + 32, tabulated in §1.1); the handful of other
textual matches are comments. The real production path is the nine
operation-family supertraits on `Backend` (`B::add(..)`, `B::zeros(..)`).

This work gives it production callers for the zero-operand fill family
(`zeros_canonical`, `ones_canonical`, `rand_canonical`, `randn_canonical`,
§2.5), sharing one body, as proof that the pipeline is reachable from a typed
frontend. **Everything else still goes through the family traits.** Do not
build on the assumption that canonical dispatch is live for any other
operation; four of the catalog's rows are reached and the rest are not.

`FROZEN_FOUNDATIONS.md` has been corrected and now carries a note saying so.

### If you read only one thing

| Question | Answer | Where |
|---|---|---|
| What is the allocation target? | a device value bound to a float dtype | §2.2 |
| Should `Runtime` exist? | no - nothing to own, all state is process-global | §1.4, §6 |
| Does canonical dispatch work? | yes, on the CPU only, and the fill family uses it | §1.1, §2.5 |
| Is the proof defect fixed? | for the fill family via the target API only | §2.5, §3 Step 1 |
| What happened to `tensor!`? | demoted; `backend:`/`device:` deleted | §2.3 |
| Do boolean tensors exist? | no, and `BoolDType` is a dead trait | §1.9, §4.1 |
| What is a runtime axis spelled? | `usize`, **not** `Dyn` | §1.11 |
| What should I do next? | §3, in order | §3 |

---

## 1. Verified findings

Each of these was established by compiling or running something. Where a
finding contradicts a doc comment in the tree, that is called out.

### 1.1 Canonical dispatch is test-only

```
crates/incin-backends/tests/canonical_dispatch_smoke.rs   current canonical dispatch test
crates/incin-backends/src/cpu/canonical.rs         56   all after #[cfg(test)] @2596
crates/incin-backends/src/capability.rs             current capability registrations
                                                  ---
                              real call sites     216   every one a test
crates/incin-backends/src/capability.rs             2   comments, not calls
crates/incin-core/src/exec/catalog.rs               1   doc comment
```

Production callers: **0** (before this work; see §2.5). Reproduce with - note
the counts now also include the call sites this work added, so subtract
`incin-backends/src/target.rs` and `incin-core/tests/proof_reaches_the_backend.rs`
to recover the numbers above:

```bash
for f in $(grep -rl "dispatch::execute" crates/ --include=*.rs); do
  echo "$f  calls=$(grep -c 'dispatch::execute' "$f")  cfg(test)@=$(grep -n '#\[cfg(test)\]' "$f" | head -1 | cut -d: -f1)"
done
```

The repo's own generated evidence agrees - 
`audit-evidence/FND-005/cpu-migration-status.md` says the family traits
"remain the path the stable tensor surface uses". `FROZEN_FOUNDATIONS.md`
does not, and is the document that is wrong.

### 1.2 There are three operation vocabularies, not one

| # | Vocabulary | Size | Where | Used by |
|---|---|---|---|---|
| 1 | `OperationKind` / `op::X` / `Descriptor<O>` | 174 rows | generated from `operation_catalog.rs` | tests only |
| 2 | Nine `Backend` family supertraits | - | `tensor/backend.rs` | **all production tensor ops** |
| 3 | canonical `OperationIdentity` | catalog-derived | `graph.rs` and `compiled/capture.rs` | canonical catalog |

Vocabulary 3 has no reference to the catalog. So "capture records the same
operation identity used for execution" is **false**: capture speaks 3,
execution speaks 2, the frozen contract is 1.

### 1.3 Proof provenance was being discarded (fixed for `op::Zeros` - §2.1, §2.5)

`Shape::PROOF` is correctly implemented (Static / Mixed / Dynamic, folded per
axis). `dispatch::execute` hardcoded `ProofLevel::Dynamic` because it has no
`S` type parameter to read from. Verified empirically:
`ProofLevel::of::<s![2,3]>()` is `Static`; the value reaching the descriptor
was `Dynamic`.

`ProofLevel` is also a single aggregate: it cannot distinguish
`s![usize, 784]` from `s![32, usize]` - both are `Mixed`. Survivable for eager
execution, **not** survivable for capture, plan caching, or shape guards.

### 1.4 Backends own nothing; devices own nothing; resources are global

- `WgpuBackendImpl<T, D>(PhantomData)` - "the stateless WGPU executor",
  `const fn new()`. A ZST.
- `IncinBackend<T, D>` is a **type alias** for `<D as BackendFor<T>>::Backend`,
  not a struct.
- `Cuda::new(0)` / `Wgpu::new(0)` are `const fn -> Self`, **infallible**, and
  touch no hardware. `Cpu` is a unit struct with **no `new()`**.
- `ExecutionContext<B>` = `{ backend: B (a ZST), policy: ExecutionPolicy (Copy) }`.
  No queue, allocator, RNG, cache, or capture state.
- `Tensor` holds storage + five metadata fields. No backend value, no context,
  no borrow of anything.
- Real resources are process-global and deliberately never released:
  `WGPU_STATE: OnceLock` (`wgpu/device.rs:15`),
  `CUDA_DEVICES: OnceLock<Mutex<BTreeMap<usize, Arc<CudaContext>>>>`
  (`cuda/gpu.rs:110`, kept forever because re-retaining costs 131 ms vs 1 µs).

**Consequence: a `Runtime` type has nothing to own.** Do not add one until
something genuinely needs per-instance allocator / queue / RNG / plan-cache
state. See §6 for the decision rule.

### 1.5 Multi-device works only on CUDA

- **CUDA:** real. Contexts keyed by ordinal; ordinal threaded into allocation.
- **WGPU:** one process-global adapter obtained with `request_adapter(...)` - 
  **the ordinal is never passed**. `wgpu/backend.rs:95` explicitly rejects any
  ordinal ≠ 0 with `InvalidDeviceOrdinal`.
- **CPU:** ordinal must be 0.

Any claim of multi-device support must name the backend it is true for.

### 1.6 `ArgInto`'s positional tuple is the root ergonomic problem

16 lifting impls (C(4,k), k=0..4). The trap: a **fully static shape's `Arg` is
a tuple of units**, and `impl<T1> NotUnit for (T1,)` makes that count as "the
caller supplied something". So the shape slot is occupied even when nothing
was written, and the device selector shifts to second position.

`crates/incin-core/src/tensor/device.rs`'s own module doc claimed
`zeros(Cuda::new(2))` compiles. It does not. **Fixed in this session** - the
doc now shows `((), Cuda::new(2))` and explains why.

### 1.7 Layer init is worse than tensor construction

`Linear::build<A>(args)` takes `A: LayerArgInto<(InArg, OutArg, DTypeArg, DeviceArg, BiasArg)>`
 -  a **5-position** compressed tuple. Real usage:
`Linear::<Dyn, Backend>::build((784, 128))`. There is no
`Linear::new(784, 128, &target)`.

### 1.8 The flagship example bypasses the whole typed surface

`crates/incin/examples/mnist_training.rs` uses `unsafe` pointer casts, calls
`<Backend as Backend>::from_bytes` directly, builds `DeviceId::cpu()` by hand,
and does `labels.push(label as f32)` with the comment "F32 target tensor for
CrossEntropyLoss". When the framework's own showcase routes around the
constructors, the constructors have failed. **Not yet fixed** - see §4.3.

### 1.9 Boolean tensors

- `DTypeId::Bool` and `bool` implement `DType`, `ConstDType`, `BuiltinDType`,
  `PlainDType`, and `BoolDType`.
- Comparisons return `Tensor<..., bool, NoGrad>` and logical operations consume
  boolean tensors. Boolean readback validates the logical `0`/`1` values.
- The current boolean contract is covered by the target-api and tensor
  operation tests; this section is descriptive rather than a pending design.

### 1.10 Transfer structurally cannot be canonical, and the repo says so

`ExecutionSite::DeviceTransfer.blocking_reason()` → *"produces storage on
another backend, which the executor cannot name"*. `is_backend_executable()`
returns false. `Tensor::to_device` calls `B::transfer_storage` directly and
**drops the placement parameter `P`**.

Never claim transfer participates in canonical lowering without first changing
the `Execute` trait.

### 1.11 Shape spellings that surprise people

- A runtime axis is spelled **`usize`**, not `Dyn`. `s![Dyn, 784]` **does not
  compile** (`Dyn: Dim` unsatisfied). `Dyn` means whole-shape dynamism.
- Verified working today:
  `Tensor::<s![usize, 784], DefaultBackend>::zeros((4usize,))` → `[4, 784]`.
- Refinement (`into_shape::<S2>()`) and erasure (`into_dyn()`) already exist
  and are zero-copy. Refinement's diagnostics are poor: `Error::Msg` plus
  `core::any::type_name`, no offending axis.

### 1.12 Creation ops: which are migrated

From `audit-evidence/FND-005/cpu-migration-status.md`:

| Operation | Migrated |
|---|---|
| `zeros`, `ones`, `rand`, `randn`, `full`, `arange`, `linspace` | **yes** |
| `tensor_from_data`, `tensor_from_bytes` | **yes** |
| `sample` | **no: distribution registry is still required** |

The data constructors now use `DataAttributes` and exact canonical dispatch.
The remaining creation gap is `sample`, whose arbitrary distribution value
still needs a registry that can be represented by the descriptor contract.

---

## 2. What landed in this session

All changes are additive. Nothing existing changed behaviour.

### 2.1 `ShapeEvidence` + `execute_with_evidence` (foundation)

**Files:** `crates/incin-core/src/exec/proof.rs`,
`crates/incin-core/src/exec/dispatch.rs`, `crates/incin-core/src/exec/mod.rs`

`ShapeEvidence` is a `ProofLevel` that can only be obtained from a shape
*type* (`ShapeEvidence::of::<S>()`) or as the no-claim value
(`ShapeEvidence::dynamic()`). A bare `ProofLevel` parameter would have let any
caller assert `Static` beside arbitrary metadata - the forgery `Validated`
prevents one layer down.

`execute_with_evidence` is a new `pub fn` beside `execute`; `execute`
delegates to it with `ShapeEvidence::dynamic()`. **Deliberately additive**:
`dispatch.rs` is a frozen foundation and `execute` has ~218 call sites, so
changing its signature would have been a large mechanical churn for no
behavioural gain. Nothing changes for existing callers.

Tests: `exec::proof::tests::{shape_evidence_reports_exactly_what_the_shape_type_proves,
dynamic_evidence_claims_nothing, evidence_meets_at_the_weaker_operand}` and
`exec::catalog::tests::frontend_shape_evidence_reaches_the_validated_descriptor`.

This is now *used*, not just plumbing - see §2.5, which added the first
non-test caller of canonical dispatch in the repository. The defect is closed
for `zeros` via the target API and open everywhere else.

### 2.2 The `target-api` prototype

**Files:** `crates/incin-backends/src/target.rs` (new, feature-gated),
`crates/incin-backends/src/lib.rs`, `crates/incin-backends/Cargo.toml`

Feature `target-api`, **off by default**. Enable with
`--features target-api`.

```rust
use incin::prelude::*;   // or incin_backends::prelude::*; the traits must be in scope

let cpu = Cpu;                    // unit struct - no new(), no ?
let gpu = Wgpu::new(0);           // const fn, infallible

let x      = cpu.tensor([[1.0_f32, 2.0], [3.0, 4.0]])?;  // s![2,2], f32, NoGrad
let labels = cpu.tensor([0_i64, 1])?;                     // s![2],   i64, NoGrad - not cast

let a = gpu.zeros(Static::<s![32, 784]>::new())?;         // Tensor<s![32,784], ..>
let b = gpu.zeros([batch, 128])?;                         // Tensor<Dyn, ..>
let c = gpu.zeros(Bound::<s![usize, 784]>::new((batch,))?)?;  // Tensor<s![usize,784], ..>

let w = cpu.parameter(Static::<s![128, 784]>::new(), GeneratedFill::Normal)?;  // Grad

let fp64 = cpu.with_dtype::<f64>()?;   // fallible: the device must be able to store f64

// Canonical lowering (see 2.5) - same result, real proof carried:
let z = cpu.zeros_canonical(shape![2, 3])?;

// Layers begin at the layer type (see 2.6):
let layer = incin_core::nn::Linear::<Dyn, _>::new(784, 128, &cpu)?;
```

Key design decisions and *why*, so they are not relitigated:

- **The target is a device value carrying a float dtype.** `BackendFor<F>` is a
  *device × float → backend* mapping, so a bare device does not determine a
  backend and every call site would leave `F` ambiguous (this reproduces as
  `E0283`). The float is an **associated type**, not a generic parameter, which
  keeps inference total.
- **Not a `Runtime`** - see §1.4; there is nothing to own.
- **Not a backend value** - backends are ZSTs and type aliases (§1.4).
- **The shape spec decides the result type**, so there is no
  `zeros_static`/`zeros_partial`/`zeros_dynamic` family. `ShapeSpec::resolve`
  returns the field *and* the dims together, because both are needed and
  deriving one from the other twice is how they drift.
- **`Bound::new` takes a tuple, not `[usize; N]`.** It delegates to `ArgInto`,
  which only has an impl for the exact number of runtime axes, so wrong arity
  is a **compile** error rather than a runtime `Err`. Strictly stronger.
- **Everything is `NoGrad` except `parameter`.** The grad rule follows the
  object being created, not a setting on the target.
- **Data dtype is never the target's dtype.** `cpu.tensor([0_i64, 1])` on an
  `f32` target is `i64`.
- **It must be an extension trait.** Devices are defined in `incin-core`;
  `BackendFor` is defined in `incin-backends`. Rust's orphan rule forbids an
  inherent `impl Cpu` in `incin-backends`, and core cannot see `BackendFor`.
  Cost: methods only autocomplete when the trait is in scope.

Tests: `crates/incin-backends/tests/target_api.rs` (17, CPU) and
`target_api_wgpu.rs` (3, **real WGPU execution** - asserts `device=wgpu:0`
actually appears in `Display` output, so the ordinal genuinely lands).

### 2.3 `tensor!` demoted to a CPU-literal convenience

**Removed:** the `backend:` and `device:` clauses.

The `device:` clause inferred a backend type by *sniffing token spelling*
(`Wgpu::new(0)` → `Wgpu`). It could not see through a binding - `let d =
Wgpu::new(0); tensor![1.0; device: d]` failed - and inferring types from how an
expression is written is not something a macro should do. The target API
removes the need. `backend:` went with it because without `device:` it could
not reach a Tier 2 device anyway, and the target API covers the rest.

Remaining surface: `tensor![data...; dtype: T, grad: G]`, either order, default
CPU backend only. Unknown clauses produce an error that points at the target
API.

Deleted: `crates/incin/tests/tensor_macro_device_tier2{,_wgpu}.rs`,
`tests/tensor_compile_fail/device_without_backend_needs_hint.*`. Added
`tests/tensor_compile_fail/backend_clause_is_gone.rs`.

### 2.4 Documentation defects fixed

- `crates/incin-core/src/tensor/device.rs` module doc: the Tier 2 example
  `zeros(Cuda::new(2))` does not compile. Corrected to `((), Cuda::new(2))`
  with an explanation of the `NotUnit` mechanism that forces it. **Pre-existing
  bug, not mine.**
- Tier numbering: three places in the `tensor!` macro and its tests called
  `DeviceId::wgpu(0)` "Tier 3". `device.rs` defines **Tier 1 = `Dyn`**,
  Tier 3 = fully static. **My error from earlier in the session**, now corrected.

### 2.5 Production callers of canonical dispatch

**`TargetExt::generated_canonical`** in `target.rs`, reached through
`zeros_canonical`, `ones_canonical`, `rand_canonical` and `randn_canonical`.
These are the only non-test calls to `exec::dispatch` in the repository.

One body serves all four because `op::Zeros`, `op::Ones`, `op::UniformRandom`
and `op::NormalRandom` are exactly the catalog rows whose attributes are
`CreationAttributes` and whose operand arity is zero. The helper is bounded on
`O: CanonicalOperation<Attributes = CreationAttributes>`, so it cannot be
pointed at `op::Full` or `op::Arange`, which need a value or a range. Those
need their own bodies, not a wider bound here.

How this sidesteps the blocker described in §3 Step 1: `Tensor::zeros` is
generic over `B: Backend`, and `Backend` does not require
`Execute<Descriptor<op::Zeros>>` - adding that is FND-005's completion
condition and breaks every backend. A *target*, unlike `Tensor`, knows its
concrete backend, so each method asks for the bound at its own signature and
leaves `Backend` untouched. Backends that have not migrated an operation
simply do not offer the method, which is a compile-time fact.

**The evidence chain is now closed and tested end to end**, which is what the
optimization phase needs:
`ShapeSpec::Shape` → `Shape::PROOF` → `ShapeEvidence` →
`execute_with_evidence` → `Validated::proof_level()` at the executor. The first
link is asserted by `each_shape_specification_carries_its_own_proof_level` in
`crates/incin-backends/tests/target_api.rs`, the middle by
`crates/incin-core/tests/proof_reaches_the_backend.rs`. A `Static<S>`
allocation reaches the backend stamped `ProofLevel::Static`, a
`Bound<S>` one as `Mixed`, and a runtime array as `Dynamic`.

**What the executors now consume is the descriptor, not the proof level.**
The CPU's four pointwise binary executors (`Add`, `Sub`, `Mul`, `Div`) take
their output shape from `Descriptor::outputs()` instead of calling
`broadcast_shape` on the raw operands. `dispatch::execute_with_evidence` has
already run `infer_outputs` and sealed the result in a `Validated`, so the
re-derivation was a second fallible right-aligned loop and a second heap
allocation for an answer the request was carrying. Measured at ~12.6 ns per
call on an 8x8 broadcast add, about 1.7% of that call - small, and the reason
to do it is structural rather than the nanoseconds: an executor that trusts the
validated descriptor is the precondition for one that trusts anything else in
it. Each site cross-checks the descriptor against a re-derivation under
`debug_assert`, so every test run verifies that `infer_outputs` and
`broadcast_shape` agree and release builds pay nothing.

**The proof level itself is still unconsumed.** `Static` says the extents were
compile-time constants; acting on that means a kernel that can be specialized
on constant extents - const-generic tile sizes, unrolled inner loops - and no
such kernel exists on any backend. Adding a use of `proof_level` that does not
change what runs would be decoration. The value arrives, is proven correct, and
is readable via `Validated::proof_level()`; the kernels that could exploit it
are the next piece of work, and they are also the reason Step 1b's accelerator
gap matters - matmul and elementwise are the operations worth specializing and
they have no canonical executor outside the CPU.

**Which today means: the CPU and nothing else.** Every
`Execute<Descriptor<op::X>>` impl in the repository lives in
`crates/incin-backends/src/cpu/canonical.rs`; there is no `canonical.rs` under
`wgpu/`, `cuda/` or `metal/`, and those backends implement `Execute` only for
the five grouped specs (`Conv2dSpec`, `MatMulSpec`, `Pool2dSpec`,
`ReductionSpec`, `ReshapeSpec`). So `gpu.zeros_canonical(..)` does not fail at
runtime, it does not compile, and the same will be true of every `*_canonical`
method added next. `crates/incin/tests/target_api_compile_fail_wgpu/` pins that
with the diagnostic a caller actually sees, which names the five specs wgpu
does implement. This was missed for a while because the wgpu suite exercised
`zeros` and never `zeros_canonical`.

It passes `ShapeEvidence::of::<Sp::Shape>()`, so a `Static<S>` request reaches
the descriptor stamped `ProofLevel::Static` rather than `Dynamic`.

**Two real findings came out of making this work, both worth knowing:**

1. **Grad mode is part of the capability query.** `dispatch::execute` turns
   the context's `GradMode` into the query's `training` flag, and the CPU
   registry advertises `zeros` for **inference only** - correctly, a fill has
   no backward. Building the context with the default (grad enabled) asks
   "can `zeros` participate in training", gets a truthful *no*, and surfaces
   it as `training is unsupported for zeros`. `zeros_canonical` therefore
   builds its context with `GradMode::Disabled`, which is also semantically
   right: it allocates a `NoGrad` tensor. **Anyone migrating another creation
   operation will hit this.**
2. **`Q8_0` is refused before allocation**, proving the capability gate is
   live on this path rather than decorative
   (`canonical_zeros_refuses_an_unsupported_dtype`).

**The proof is observed, not inferred.**
`crates/incin-core/tests/proof_reaches_the_backend.rs` implements a minimal
recording backend (`StorageBackend` is 3 items, `Capabilities` is 1, `Execute`
is 1 - the whole thing is ~40 lines) that stores
`request.operation.proof_level()` and asserts what actually arrives:

| Shape type | Observed at the backend |
|---|---|
| `s![2, 3]` | `ProofLevel::Static` |
| `s![usize, 784]` | `ProofLevel::Mixed` |
| `Dyn` | `ProofLevel::Dynamic` |

plus a regression that plain `execute` still reports `Dynamic`, so adding the
evidence-carrying overload did not quietly upgrade callers that know nothing.

### 2.6b `shape!` - the shape argument's surface

`Static::<s![32,784]>::new()` and `Bound::<s![usize,784]>::new((batch,))?` were
too much ceremony for the thing you write most often. `shape!`
(`incin-macros/src/shape_value.rs`) is the value-level counterpart of `s!` and
expands to exactly those types:

| Written | Expands to | Result shape |
|---|---|---|
| `shape![32, 784]` | `Static::<s![32, 784]>::new()` | `s![32, 784]` |
| `shape![batch, 784]` | `Bound::<s![usize, 784]>::new((batch,))` | `s![usize, 784]` |
| `shape![rows, cols]` | `Bound::<s![usize, usize]>::new((rows, cols))` | `s![usize, usize]` |
| `[rows, cols]` | *(unchanged)* | `Dyn` |

**The property worth protecting:** the shortest thing to write now *keeps* the
rank. Before, the shortest form was `[batch, 128]`, which erases to `Dyn`. Now
you have to deliberately reach for the array to throw rank away.

Two supporting changes:

- **`Bound::new` is now infallible.** Arity was already a compile error via
  `ArgInto`, so there was nothing left to reject; element-count overflow moved
  to `ShapeSpec::resolve`, which is where the dims are actually about to reach
  an allocator. This is what lets `shape!` expand to a plain expression rather
  than generating a `?` or an `unwrap` in the caller's code.
- **Staticness is inferred syntactically.** An integer literal is a static
  axis, anything else is runtime - the same split `s!` already makes, with the
  `usize` inferred rather than spelled. A `const N: usize` therefore reads as
  an expression and yields a runtime axis. That is a *weaker* shape than was
  available, never a wrong one, and `Static`/`Bound` stay public for when it
  matters.

This is **not** the token-sniffing I removed from `tensor!` (§2.3). That
inferred a *backend type* from an expression's spelling and broke outright on
`let d = Wgpu::new(0)`. This infers *shape staticness*, degrades safely, and
mirrors what `s!` already does.

`shape!` expands to `::incin::prelude::…`, so like `s!` and `tensor!` it is
only usable from the `incin` façade - `incin-backends`' own tests use
`Static`/`Bound` directly.

Tests: `crates/incin/tests/shape_macro.rs` - 11, each pinning the resulting
`Tensor` *type* rather than just its dims, plus 3 trybuild cases (negative,
fractional, wrong literal suffix).

### 2.6c How dtype is decided

Three sources, none of which can be confused for another:

| Constructor | dtype from | Example |
|---|---|---|
| `target.tensor(data)` | the **data's** element type | `cpu.tensor([0_i64, 1])` → `i64` |
| `target.tensor_from_vec(v, ..)` | the `Vec`'s element type | `Vec<f64>` → `f64` |
| `zeros`/`ones`/`rand`/`randn` | the **target's** dtype | `cpu.zeros(..)` → `f32` |
| parameters / layer weights | the target's dtype | `f32` |

The target's dtype is rebound with `with_dtype::<K>()`, not chosen per call.
There is deliberately no dtype argument on the constructors: a program almost
always has one working precision, and a shape argument that could also change
dtype would be the same mistake as a device argument that silently chose a
backend.

**Two corrections made here after probing the first version:**

1. `TensorTarget::Float` / `with_float` were **misnamed**. `BackendFor<T>` is
   generic over any `DType`, so `with_dtype::<i64>()` already worked and
   produced perfectly good `i64` zeros. Renamed to `Dtype` / `with_dtype`,
   because a target rebound to `i64` for index buffers or masks is a
   legitimate thing to want and the old name said otherwise.
2. `rand`/`randn` on an integer target **failed at runtime**
   (`Dtype I64 is unsupported by backend 'Cpu' for 'randn'`). Correct, but
   late. They now carry `where Self::Dtype: FloatDType`, so it is a compile
   error, and the diagnostic names the cause directly:

   ```text
   error[E0277]: the trait bound `i64: FloatDType` is not satisfied
     note: `TensorTarget::Dtype` is `i64` here
   ```

   `zeros`/`ones`/`full` keep no such bound: a fill is meaningful for any
   dtype.
3. `with_dtype` itself was **infallible and unchecked**. The
   `where Self::Device: BackendFor<K>` bound reads like a filter and is not
   one: `BackendFor` is blanket-implemented for every dtype on every device,
   so `wgpu.with_dtype::<f64>()` compiled, succeeded, and left every later
   allocation to fail on its own, each one reported far from the line that
   picked `f64`. It now returns `Result` and asks the shared capability table
   under `OperationKind::Storage`:

   ```text
   Dtype F64 is unsupported by backend 'Wgpu' for 'with_dtype'
   ```

   Storage is the only question a rebinding can answer, so per-operation
   support stays with the operation. `Cpu.with_dtype::<Q8_0>()` succeeds
   because CPU storage holds packed blocks, and `zeros` on that view still
   refuses because there is no fill kernel for them.

> **Do not write `compile_fail` doctests in `tests/` files.** rustdoc only
> collects doctests from a crate's *library* target, so a `compile_fail` block
> in an integration test is never executed and asserts nothing. I wrote two of
> them before checking; both are now trybuild cases under
> `crates/incin/tests/target_api_compile_fail/`, which do run.

### 2.6 `Linear::new(in, out, &target)`

**`LinearInit`** in `target.rs`. Construction begins at the *layer type*, per
the consistency rule: the target decides where memory goes, the layer type
decides parameter structure. `target.linear(..)` was deliberately not added.

Parameters are allocated through the same path as everything else, so there is
no second allocator.

`S` cannot be inferred (no argument mentions it), so the call needs a
turbofish: `Linear::<Dyn, _>::new(784, 128, &cpu)?`. That is a genuine wart,
not an oversight - it follows from `Linear` being generic over its shape
family.

**Required a small core change:** `Linear::build_full` was `pub(crate)` and is
now `pub`. `build` takes a *compressed* tuple with the unit positions omitted,
and its arity therefore depends on which positions are `()` for the concrete
device - `(in, out)` for `Cpu`, `(in, out, Wgpu)` for `Wgpu`. Generic code
cannot name that arity, and `LayerArgInto` only accepts `NotUnit` positions,
so the uncompressed form is the only one callable from a device-generic
context. This is the same defect as §1.6, one layer up.

---

## 3. What to do next, in dependency order

Each step is blocked by the one above it.

### Step 1 - Proof provenance  ✅ done

`TargetExt::zeros_canonical` (§2.5) is the first production caller of
`dispatch::execute`, it carries real evidence, and
`crates/incin-core/tests/proof_reaches_the_backend.rs` observes the level
arriving at a backend for all three shape kinds.

**Scope of the claim, precisely:** the defect is closed *for the zero-operand
fill family reached through the target API* - `op::Zeros`, `op::Ones`,
`op::UniformRandom`, `op::NormalRandom`. Every other operation still goes
through the family traits and carries no proof at all, because it never reaches
`dispatch::execute`. Do not describe the defect as closed generally.

Note that `RecordingBackend` in that test cannot be plugged in as a
`TargetBackend`, because `BackendFor` is sealed (`backend_kind.rs:7`) and a
`tests/` file is a separate crate. It exercises `execute_with_evidence`
directly instead, which is the link that needed observing.

> **If you add a Cargo feature to the `incin` façade**, register it in
> `crates/incin/src/doctor.rs::compiled_features()` as well. That list is
> hand-written (nothing can enumerate its own features at runtime) and
> `incin/tests/doctor.rs::the_reported_features_are_exactly_the_manifests`
> fails when it drifts from the manifest. Adding `target-api` tripped exactly
> this.

### Step 1b - Migrate the remaining creation operations  ✅ done

All seven zero-operand creation operations are now reachable from a target and
routed through canonical dispatch:

| Direct | Canonical | Attributes |
|---|---|---|
| `zeros` `ones` `rand` `randn` | `*_canonical` | `CreationAttributes` |
| `full` | `full_canonical` | `FullAttributes` |
| `arange` | `arange_canonical` | `ArangeAttributes` |
| `linspace` | `linspace_canonical` | `LinspaceAttributes` |

`full`, `arange` and `linspace` had **no target-API form at all** before this,
canonical or otherwise - a target could produce zeros, ones and two
distributions, and a constant needed dropping back to
`Tensor::<S, B>::full((..))`.

The four fills share `generated_canonical`; the other three cannot, because a
different attribute type is a different type rather than a wider one. What they
all share is `canonical_creation`, which takes a closure building the
attributes from the resolved shape, dtype and device - so the evidence, the
grad-mode handling and the `Validated` plumbing exist once, and only the
attribute construction varies.

This step widened the CPU-only surface rather than closing it: all seven
canonical methods compile for `Cpu` targets and for no other backend, because
the `Execute<Descriptor<op::X>>` impls exist only under `cpu/`. That was a
knowing call, not momentum - the direct forms work on every backend, so nothing
is unreachable off the CPU, it is only unproven.

`tensor_from_data` and `tensor_from_bytes` now use `DataAttributes`, which carry
the runtime shape, dtype, device, and payload required by their exact
descriptors. The remaining creation gap is `sample`; migrating it requires a
distribution registry or an equivalent stable descriptor payload.

### Step 2 - `Linear::new(in, out, &target)`  ✅ done

Landed as `LinearInit` (§2.6). Extend the same pattern to `Conv2d`,
`LayerNorm`, `Embedding` - each needs its own `build_full` made `pub` for the
same reason `Linear`'s did.

A builder (`Linear::builder(784, 128).without_bias().build(&target)`) remains
possible and was not attempted; the default constructor was the priority.

### Step 3 - Unify capture's vocabulary with the catalog

Graph capture records `OperationIdentity` directly. Built-in identities come
from the canonical operation catalog, while custom operations retain their
namespaced `OperationKey`. Descriptor payloads and execution-site metadata are
captured with each node.

### Step 4 - Axis-level shape contract

`ProofLevel` cannot distinguish `s![usize, 784]` from `s![32, usize]`. For
plan caching and shape guards you need something like:

```rust
enum AxisContract { Const(usize), Runtime { slot: usize }, Symbol { id: ShapeSymbol } }
```

`ShapeSpec` is the natural place to produce it - it already knows which axes
the caller bound. Add `fn contract(&self) -> ShapeContract` there.

### Step 5 - Decide the prototype's fate

See §5 for criteria.

---

## 4. Known gaps deliberately not addressed

### 4.1 Boolean tensors

Resolved in the current contract: `DTypeId::Bool`, `bool`, `BoolDType`, and
boolean comparison/logical outputs are implemented and covered by the current
target-api and tensor operation tests. This is no longer a pending foundation
change.

### 4.2 Transfer

`tensor.to(&target)` was **not** added. Transfer cannot participate in
canonical execution (§1.10), and adding a target-flavoured wrapper over
`to_device` would create the impression that it does. `to_device` also drops
the placement parameter `P`, which should be fixed independently.

If you add it, delegate to `B::transfer_storage` and **document that it
bypasses canonical lowering**.

### 4.3 The MNIST example

Still uses `unsafe`, direct backend calls, and `label as f32`. It should be
rewritten against whatever survives Step 1/Step 2 - that rewrite is the real
acceptance test for the whole design. Do not do it before Step 1, or it will
be rewritten twice.

### 4.4 WGPU's ignored ordinal

`Wgpu::new(3)` type-checks and fails at runtime. Consider making non-zero
ordinals unrepresentable for WGPU, or implementing real adapter selection.

---

## 5. Prototype deletion criteria

Delete `crates/incin-backends/src/target.rs`, its feature, and its two test
files if any of these hold:

1. Step 1 cannot be completed - i.e. `TargetExt::zeros` cannot be made to
   route through `dispatch::execute` with real evidence. Then the prototype is
   a fourth construction path, which is worse than the three that exist.
2. The extension-trait import requirement proves unacceptable in practice
   (methods not discoverable without `use incin_backends::prelude::*`).
3. A decision is taken to put allocation on `ExecutionContext` or a `Runtime`
   after all - in which case the shape-spec types (`ShapeSpec`, `Static`,
   `Bound`) should be salvaged, since they are independent of what the target
   is.

Nothing outside the feature gate depends on it, so deletion is a clean revert.

---

## 6. Decision rule

> Use device values as public allocation targets, bound to a float dtype, for
> as long as backends remain stateless ZSTs and device resources remain
> process-global. Introduce a `Runtime` only when something must own an
> allocator, queue, RNG, capture, or compilation cache **per instance** - and
> then introduce it by wrapping the target, not replacing it. Keep shape
> contracts and semantic lowering independent of both. Do not add a
> user-facing construction surface that does not terminate in canonical
> dispatch.

---

## 7. Reproduce the verification

```bash
# Foundation
cargo test -p incin-core --lib exec::proof
cargo test -p incin-core --lib frontend_shape_evidence
cargo test -p incin-core --test proof_reaches_the_backend

# Prototype, CPU (includes canonical dispatch + Linear::new)
cargo test -p incin-backends --features target-api,cpu --test target_api

# Prototype, real accelerator
cargo test -p incin-backends --features target-api,wgpu --test target_api_wgpu

# Reachable from the public façade
cargo test -p incin --features cpu,target-api --test target_api_facade

# The shape! macro (includes trybuild snapshots)
cargo test -p incin --features cpu,target-api --test shape_macro

# Macro surface after demotion (includes trybuild snapshots)
cargo test -p incin-macros --test tensor_macro

# Hygiene
cargo fmt --all -- --check
cargo clippy -p incin-core -p incin-macros --all-targets -- -D warnings
cargo clippy -p incin-backends --features target-api,cpu --all-targets
```

## 8. Files touched

| File | Change |
|---|---|
| `incin-core/src/exec/proof.rs` | + `ShapeEvidence`, + 3 tests |
| `incin-core/src/exec/dispatch.rs` | + `execute_with_evidence`; `execute` delegates |
| `incin-core/src/exec/mod.rs` | export `ShapeEvidence` |
| `incin-core/src/exec/catalog.rs` | + evidence-threading test |
| `incin-core/tests/proof_reaches_the_backend.rs` | **new**, 4 tests, recording backend |
| `incin-core/src/tensor/device.rs` | fixed non-compiling Tier 2 doc example |
| `incin-backends/src/target.rs` | **new**, feature `target-api` |
| `incin-backends/src/lib.rs` | register module + prelude exports |
| `incin-backends/Cargo.toml` | + `target-api` feature, + `typenum` (const-generics) |
| `incin-core/src/nn/linear.rs` | `build_full` `pub(crate)` → `pub` (+ rationale) |
| `incin-backends/tests/target_api.rs` | **new**, 22 tests |
| `incin-backends/tests/target_api_wgpu.rs` | **new**, 3 tests, real WGPU |
| `incin/Cargo.toml`, `incin/src/lib.rs` | forward `target-api`; prelude re-export |
| `incin/tests/target_api_facade.rs` | **new**, reachability through the façade |
| `docs/FROZEN_FOUNDATIONS.md` | corrected the "single production route" claim |
| `docs/README.md` | index entry for this document |
| `incin/src/doctor.rs` | registered `target-api` in `compiled_features()` |
| `incin-macros/src/tensor.rs` | removed `backend:`/`device:` + token sniffing |
| `incin-macros/src/lib.rs` | doc rewrite for reduced scope; `shape!` export |
| `incin-macros/src/shape_value.rs` | **new**, the `shape!` macro |
| `incin/tests/shape_macro.rs` | **new**, 11 tests |
| `incin/tests/shape_compile_fail/` | **new**, 3 trybuild cases |
| `incin/tests/target_api_compile_fail/` | **new**, 2 trybuild cases |
| `incin-macros/tests/tensor_macro.rs` | dropped removed-clause tests |
| `incin-macros/tests/tensor_compile_fail/` | −1 case, +1 case |
| `incin/tests/tensor_macro_device_tier2*.rs` | **deleted** |

Also present from earlier in the same session, unrelated to this review:
`Tensor`'s real `Display`/`Debug` (`incin-core/src/tensor/display.rs` and the
per-backend stub removals), and `crates/incin/tests/tensor_display.rs`.
