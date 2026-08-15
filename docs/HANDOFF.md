# Incin maintainer handoff

This is the shortest map for changing Incin safely. It describes the current
repository and the intended direction; generated capability and operation
documents remain outputs of the source and tests.

## Repository map

- `crates/incin-core`: foundational types, operation descriptors, tensors, and
  neural-network contracts.
- `crates/incin-backends`: CPU and optional accelerator implementations plus
  target/backend adapters.
- `crates/incin-macros`: procedural macros such as `#[module]`, `shape!`, and
  `tensor!`.
- `crates/incin`: the user-facing facade and public prelude.
- `crates/incin-data`: data and file-format support.
- `docs`: binding contracts, status notes, guides, and this handoff.
- `tools`: reproducibility and repository checks.

## Canonical handoff snapshot

Create the handoff artifact only with:

```text
tools/export-snapshot.sh <output.zip>
```

The command requires a clean tracked checkout, builds the ZIP from `HEAD`,
compares its file set with tracked source, unpacks that exact ZIP, verifies the
required workspace and check files, runs the architecture, large-file, and
public-API gates inside the unpacked copy, and runs the smallest no-default
feature core check when Cargo is available. The ZIP, not `cargo package`, is
the handoff artifact and its validation result.

## Layer diagram

```text
foundation: error / shape / dtype / device / layout / identity
       ↓
operation semantics: catalog / descriptors / metadata / validation / effects
       ↓
tensor runtime: Tensor / storage / backend / dispatch / autograd / tracing
       ↓
NN and state: Module / Param / Buffer / state / layers / init / optimizers
       ↓
higher or experimental: graph / compiled / distributed / import / tooling
```

Dependencies should point downward in this diagram. Tensor-owned reduction and
device-transfer contracts live below `nn`; NN re-exports are only facade-level
ergonomics.

## Ownership and tiers

Foundation owns invariants and small value types. Operation semantics owns the
canonical descriptor execution contract. Tensor runtime owns storage, backend
capabilities, transfers, dispatch, gradients, and tracing. NN owns module trees,
parameters, buffers, state, and layers. Higher-level features must not become
required dependencies of the lower tiers.

The normal public tier is the facade prelude. Checkpoint and transactional
state staging types are under the named `incin::state` surface rather than the
normal prelude. Backend authors should use the named `backend_authoring` and
`types` surfaces. The backend contract is split
by responsibility across `backend/execute.rs`, `backend/transfer.rs`,
`backend/variable.rs`, `backend/autograd.rs`, and `backend/capability.rs`.
Named `HostInterop`, `VariableBackend`, `AutogradBackend`, `StorageTransfer`,
and `TransferBackend` views are the capability contract. `StorageTransfer` is
the inference-safe storage movement contract; variable-capable backends add
`TransferTo` for typed variable handles. `VariableBackend::Var<K>` carries the
variable dtype at the Rust type level; there is no erased `RawVar` escape hatch.
The core `backend.rs` identity contract is intentionally small and the
operation path is the descriptor `Execute<O>` contract. The shape-only test
backend lives in `backend/dummy.rs`.
Host byte serialization and tensor formatting belong to `HostInterop`; neither
is required by the base `Backend` contract. `AutogradBackend` is likewise an
independent capability, so an inference-only backend need not implement it.
Experimental graph, compiled,
distributed, import, and tooling APIs are explicitly unstable.

The ordinary facade prelude allowlist is intentionally user-shaped: tensor
and shape construction, dtype/device selection, gradients, module/layer
building, state snapshots, optimizers, and the stable macros. Visitor traits,
backend capability traits, and variable handles remain available through named
expert modules. It does
not export graph capture, proof-construction helpers, storage/backend-authoring
traits, physical storage encodings, legacy state staging traits, or backend variable handles; those
names require a named expert surface or are reserved for macro expansion.

## Adding an operation

Add the descriptor and typed attributes to the operation catalog, define its
validation and metadata, implement the canonical `Execute<Op>` path in the
backend, and add focused shape/error/CPU tests. Update generated operation
documents through their generator; do not hand-edit generated tables.

For a backend implementation, add the capability declaration and matching
`Execute<Op>` implementation in `incin-backends`, then run the focused backend
test and generated capability-document test. Backend authors should depend on
the named `backend_authoring` surface; they should not import the facade
prelude or private tensor-backend modules.

## Debugging an operation

Start with the descriptor and its validation evidence, then inspect dispatch,
the backend executor, and the smallest failing test. Check shape, dtype,
device, layout, aliasing, and gradient behavior separately. A backend-local
family trait is not a second public execution model.

## CPU and optional backends

CPU code is the reference implementation and the smallest validation target.
CUDA, WGPU, and Metal are optional feature surfaces: keep their device,
storage, transfer, and executor implementations behind their feature gates.
An inference-only backend should implement only the capabilities it provides;
training, autograd, and state are optional owners, not requirements of the
base backend contract.

## Handwritten modules

The stable conceptual path is a direct `impl Module<Input>` with explicit
`VisitState`/`VisitParameters` implementations for the fields the module
exposes. A custom module does not need `#[module]` to implement forward. State
snapshots are owned and loaded through the visitor-backed staging adapter, so a
failed load must not partially mutate the module. Model authors do not handle
backend variable types or `StateLoadPlan`.

The macro is convenience syntax for the same explicit visitor traversal. It is
not a requirement for custom networks and must not hide an unsupported field
behind compiler method-resolution tricks. Use `#[module(ignore)]` for a field
that is intentionally outside module traversal.

Capability generation is explicit. `#[module]` accepts `no_stats`,
`no_parameters`, `no_state`, `no_named_layers`, `no_shape_info`,
`no_train_mode`, and `no_to_device`; a forward-only or specialized module can
opt out of unrelated contracts while retaining an ordinary `Module` impl.
Unknown arguments are rejected. The handoff fixture includes both the
manual/macro equivalence path and a forward-only macro path.

## Macro difference

Use the macro when a struct is a conventional field-wise module and its fields
are explicitly traversable. Use handwritten implementations when traversal,
state names, or forward behavior is non-standard. Keep equivalent manual and
macro fixtures in the handoff tests.

## State and checkpoints

`Param` owns trainable state and `Buffer` owns non-trainable state. State paths
are stable semantic names, not backend handles. Save snapshots, validate the
complete key set, prepare replacements, then commit. Never make a backend
storage object part of the public checkpoint format.

## Dtype and device

Choose dtype and device at the target/allocation boundary. Preserve the typed
`Tensor` shape and dtype invariants through execution. `ToDevice` is the
tensor-layer ownership-transfer contract; module fields delegate to it.

## Smallest useful tests

```text
cargo check -p incin-core --lib --no-default-features
cargo test -p incin-core --lib <focused_test>
cargo test -p incin --test foundation_handoff_contract --features target-api,cpu
tools/check-architecture.sh
tools/export-snapshot.sh /tmp/incin-handoff.zip
```

Begin with the smallest crate and feature set that exercises the changed
contract. Expand to backend features only when the touched surface requires it.

## Feature matrix

The default facade is the normal user path. `cpu` enables the reference backend;
`cuda`, `wgpu`, and `metal` add optional accelerators; `target-api` enables
target-first allocation; `compiled`, `distributed`, and import/tooling features
are higher or experimental surfaces. Check each crate's `Cargo.toml` before
assuming a feature is available in a fixture.

## Generated files and commands

`docs/capabilities.md` and `docs/OPERATION_SEMANTICS.md` are generated. Use the
repository's documented generation commands and inspect the resulting diff.
After source changes, run `graphify update .` when the graph is available; the
graph is navigation aid, not a source-of-truth substitute.

## Historical and generated directories

Treat `graphify-out/`, build output, and generated documentation as derived
artifacts. Historical design notes belong under clearly labelled history or
status paths and must not be presented as current API guidance.

The long-form remediation and growth plans under `docs/plan/` and
`docs/growth/` are historical or active planning records as indicated by their
own status headers. They do not override the source, tests, or current
contracts in `docs/`.

## Unresolved architecture

The legacy operation-family adapters have been removed. Remaining backend
decomposition is a maintainability follow-up: large files are tracked by the
large-file inventory and must be split by storage, capability, and executor
responsibility when each extraction can be validated independently.
Experimental graph, compiled, distributed, import, and tooling APIs are
explicitly unstable. Future architecture remains open for placement algebra,
ragged and sparse tensors, true physical mutation/aliasing, richer training
contexts, semantic dtype identity, backend resource sessions, custom-op
VJP/JVP/batching, higher-order AD, compiled execution, and distributed runtime
maturity. These are future work, not missing requirements of the stable core.

## Large-file inventory

The remaining files above 1,200 lines are inventoried exactly by
`tools/check-large-files.sh`; the architecture gate fails when a new file
crosses the threshold without a named reason here. They are not all one kind
of problem:

| Area | Current reason for size | Maintainer action |
| --- | --- | --- |
| `crates/incin-core/src/exec/catalog.rs` | canonical operation schema and generated-like catalog table | keep catalog ownership centralized; extract attribute families only with generated-doc updates |
| `crates/incin-core/src/tensor/ops/manipulation.rs` | descriptor adapters for many shape operations | split by descriptor family once catalog ownership is stable |
| `crates/incin-core/src/tensor/backend/dummy.rs` | shape-only test backend and test operation coverage | keep test-only behavior isolated from production backend identity |
| `crates/incin-core/src/dist/{plan,context}.rs` | distributed placement/planning prototypes | remain feature-gated and split only with a concrete ownership seam |
| `crates/incin-core/src/tensor/base.rs` | central Tensor invariant and constructor implementation | keep invariant-preserving constructors together; extract only neutral value helpers |
| `crates/incin-core/src/tensor/dtype.rs` | logical dtype descriptors, built-in dtype implementations, and storage encodings share one registry boundary | keep identity/validation/encoding contracts together; split only along a stable registry seam |
| `crates/incin-backends/src/{cuda,wgpu,metal}/backend.rs` | feature-gated backend identity, storage, capability, and executor implementations | split remaining responsibility clusters when validated independently |
| `crates/incin-backends/src/{cpu/canonical.rs,dispatch.rs,capability.rs}` | canonical registrations, dispatch routing, and capability declarations | keep generated/completeness coupling intact; extract operation families only with focused tests |
| `crates/incin-backends/src/cpu/ops/{elementwise_kernel,elementwise,shape_ops,reduce,matmul,conv}.rs` | cohesive CPU operation families and kernel helpers | preserve family-local tests; split only where execution ownership becomes clearer |
| `crates/incin-backends/src/{dist/nccl.rs,dist/tuning.rs,tuning/identity.rs,tuning/service.rs}` | feature-gated distributed/tuning services | keep experimental ownership local; split resource protocols when they stabilize |
| `crates/incin-backends/src/wgpu/tests.rs` | feature-gated backend integration tests | split by operation family when test fixtures stop sharing setup |
| `crates/incin-backends/src/kernel.rs` | kernel template/rendering and specialization test vocabulary | retain as a mechanical kernel source boundary |
| `crates/incin-diagnostics/src/lib.rs` | diagnostic command and report surface | split command families when the diagnostic API stabilizes |

These are explicit staged extraction targets, not permission to add more
responsibilities. The current checkpoint documents the boundary and adds the
first capability seam; it does not claim the final split is complete.

## What not to change casually

Do not add a second public execution architecture, speculative runtime/session
handles, compatibility adapters for superseded operation families, backend
requirements that force autograd or state, or hand-edits to generated docs.
Do not weaken shape/dtype/device/error invariants to make a fixture compile.

## Validation checkpoint

The latest reproducible artifact validation is regenerated at the HND-004b
handoff checkpoint:

```text
zip-proj /home/xupremix/Projects/incin /tmp/hnd004b-final-v10.zip # passed
```

That command validated the generated ZIP itself: it matched the tracked file
set, contained both distributed source trees, unpacked successfully, ran the
architecture, large-file, and public-API gates inside the unpacked copy, and
ran `cargo check -p incin-core --no-default-features` there. The ZIP is the
artifact result; `tools/check-package.sh` remains an internal component check,
not an alternate snapshot workflow.

The export gate validates the tracked source set, both distributed source
trees, the structural gates inside the unpacked snapshot, and the minimal core
build. The current Book proof path is also:

```text
mdbook build docs/book
cargo test -p incin --features 'target-api backend-authoring' --doc
```

The Cargo command is intentional: standalone `mdbook test` does not receive
Cargo's dependency metadata and cannot resolve the facade's workspace crates.
The HND-004b executable proof includes the CPU Transformer composition in
`crates/incin/tests/transformer_block.rs`, the six-fixture clean/incremental
compile benchmark in `tools/bench-compile.sh`, lazy zero-worker data loading,
and dependency checks
inside exported snapshots. The normal workspace suite, mdBook build, docs
drift gate, budget gate, and export gates pass. Remaining caveats are
explicitly recorded in `docs/PROTOTYPING.md` and `docs/PROJECT_STATUS.md`,
including unavailable accelerator hardware, Miri's numerical-test exclusions,
and the repository's pre-existing formatting drift. Representative backend
feature combinations pass; the literal cargo-hack backend powerset expands to
16,420 combinations in this manifest and was stopped as impractical after
focused combinations passed. Clippy remains blocked by the pre-existing
`incin-macros` complex-type warning under `-D warnings`.

## First 30 minutes

1. Read this file and `docs/README.md`.
2. Run `git status --short` and preserve unrelated work.
3. Query the graph or read `graphify-out/wiki/index.md` for navigation.
4. Read the relevant frozen/API/error/invariant contract.
5. Run the smallest focused test before editing.
6. Trace the descriptor → dispatch → backend path for operation work.
7. Trace `Module` → typed visitors → `Param`/`Buffer` for NN work.
8. Make one coherent checkpoint, run its focused validation, update the graph,
   and commit it.
