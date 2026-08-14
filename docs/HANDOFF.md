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

The normal public tier is the facade prelude. Backend authors should use the
named `backend_authoring` and `types` surfaces. The backend contract is split
by responsibility across `backend/execute.rs`, `backend/transfer.rs`,
`backend/variable.rs`, `backend/autograd.rs`, and `backend/capability.rs`.
Named `HostInterop`, `VariableBackend`, `AutogradBackend`, and
`TransferBackend` views are the migration seam. The core `backend.rs` identity
contract is intentionally small; legacy operation-family declarations live in
`backend/legacy.rs`, and the shape-only test backend lives in `backend/dummy.rs`.
Host byte serialization now belongs to `HostInterop`; only the tensor formatting
compatibility methods remain on `Backend` while the descriptor migration removes
their operation-family dependencies.
Experimental graph, compiled,
distributed, import, and tooling APIs are explicitly unstable.

## Adding an operation

Add the descriptor and typed attributes to the operation catalog, define its
validation and metadata, implement the canonical `Execute<Op>` path in the
backend, and add focused shape/error/CPU tests. Update generated operation
documents through their generator; do not hand-edit generated tables.

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
implementations for the capabilities the module exposes: `Parameters`,
`StateDict`, `TrainMode`, `NamedLayers`, or transfer. A custom module does not
need `#[module]` to implement forward. State snapshots are owned and loaded
through the prepare/commit contract, so a failed load must not partially mutate
the module.

The macro is convenience syntax for the same explicit traversal. It is not a
requirement for custom networks and must not hide an unsupported field behind
compiler method-resolution tricks. Use `#[module(ignore)]` for a field that is
intentionally outside module traversal.

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
tools/check-package.sh
tools/check-architecture.sh
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

## Unresolved architecture

The complete removal of the remaining legacy formatting methods, the final public
API allowlist, and the complete manual-first module traversal contract remain
active consolidation work. Resolve these against
`docs/FROZEN_FOUNDATIONS.md`, `docs/API_DESIGN.md`, and source tests before
expanding the public surface.

## Large-file inventory

The remaining files above 1,200 lines are inventoried exactly by
`tools/check-large-files.sh`; the architecture gate fails when a new file
crosses the threshold without a named reason here. They are not all one kind
of problem:

| Area | Current reason for size | Maintainer action |
| --- | --- | --- |
| `crates/incin-core/src/exec/catalog.rs` | canonical operation schema and generated-like catalog table | keep catalog ownership centralized; extract attribute families only with generated-doc updates |
| `crates/incin-core/src/tensor/tracing.rs` | tracing backend, graph recording, and legacy adapters | split graph recording from backend adapters after capability migration |
| `crates/incin-core/src/tensor/ops/manipulation.rs` | descriptor adapters for many shape operations | split by descriptor family once catalog ownership is stable |
| `crates/incin-core/src/tensor/backend/dummy.rs` | shape-only test backend and test operation coverage | keep test-only behavior isolated from production backend identity |
| `crates/incin-core/src/dist/{plan,context}.rs` | distributed placement/planning prototypes | remain feature-gated and split only with a concrete ownership seam |
| `crates/incin-core/src/tensor/base.rs` | central Tensor invariant and constructor implementation | keep invariant-preserving constructors together; extract only neutral value helpers |
| `crates/incin-core/src/shapes/shape.rs` | core shape traits and implementations | keep the shape proof vocabulary together while preserving the foundation boundary |
| `crates/incin-backends/src/{cuda,wgpu,metal}/backend.rs` | feature-gated backend identity, storage, and compatibility implementations | split storage, capability, and legacy adapters per backend |
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

The current consolidation checkpoint has passed:

```text
./tools/check-package.sh                                      # passed
./tools/check-public-api.sh                                   # passed
./tools/check-architecture.sh                                 # passed
cargo check -p incin-core --no-default-features                # passed
cargo check -p incin-backends --no-default-features --features std,cuda,target-api   # passed
cargo check -p incin-backends --no-default-features --features std,metal,target-api  # passed
cargo check -p incin-backends --no-default-features --features std,wgpu,target-api   # passed
cargo test -p incin --features cpu,target-api --test handoff_manual_module --no-default-features # passed
```

These are checkpoint results, not the final HND-001 gate: the legacy backend
operation-family removal, full capability extraction, and final public API
allowlist remain open in the unresolved architecture section.

## First 30 minutes

1. Read this file and `docs/README.md`.
2. Run `git status --short` and preserve unrelated work.
3. Query the graph or read `graphify-out/wiki/index.md` for navigation.
4. Read the relevant frozen/API/error/invariant contract.
5. Run the smallest focused test before editing.
6. Trace the descriptor → dispatch → backend path for operation work.
7. Trace `Module` → `Parameters`/`StateDict` → `Param`/`Buffer` for NN work.
8. Make one coherent checkpoint, run its focused validation, update the graph,
   and commit it.
