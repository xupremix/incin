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
operation path is the descriptor `Execute<O>` contract. There is no shape-only
test backend: every test that needs a backend uses a real one, so a passing
test implies the operation both exists and computes.
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
cargo test -p incin --test foundation_handoff_contract --features cpu
tools/check-architecture.sh
tools/export-snapshot.sh /tmp/incin-handoff.zip
```

Begin with the smallest crate and feature set that exercises the changed
contract. Expand to backend features only when the touched surface requires it.

## Feature matrix

The default facade is the normal user path. `cpu` enables the reference backend;
`cuda`, `wgpu`, and `metal` add optional accelerators; target-first allocation
is part of the normal API; `compiled`, `distributed`, and import/tooling features
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
| `crates/incin-core/src/exec/catalog/tests.rs` | the catalog module's test suite; `exec/catalog.rs` itself was split into `exec/catalog/{classification,coverage,table,meta,descriptor,error,shape_transform,attributes,inference,validated,lookup}.rs` (each under the threshold) per `docs/CONVENTIONS.md`'s file-organization convention | further split by test theme only if the file grows past this size again |
| `crates/incin-core/src/tensor/ops/manipulation.rs` | descriptor adapters for many shape operations | split by descriptor family once catalog ownership is stable |
| `crates/incin-core/src/shapes/shape.rs` | structural shape algebra, validation, and proof-preserving transformations | keep cursor-independent shape operations together; extract arithmetic and validation families only with independent proof tests |
| `crates/incin-core/src/generated/onnx.rs`, `crates/incin-macros/src/generated/onnx.rs` | checked-in `prost-build` output for `proto/onnx.proto` | never hand-edit; regenerate with `cargo xtask onnx` |
| `crates/incin-core/src/dist/plan.rs` | distributed placement/planning prototype; split into `plan/{collective,preflight,error,digest,topology,strategy,workload,evidence,candidate,planner,hybrid_error}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-core/src/dist/context.rs` | two-rank distributed process identity, TCP rendezvous, and fail-stop lifecycle; split into `context/{rank,identity,state,error,lifecycle,rendezvous,bootstrap,wire}.rs` per `docs/CONVENTIONS.md`'s file-organization convention, with `rendezvous`/`bootstrap`/`wire` gated at the module declaration since every item in them is `std`-only | further split only if a file grows past the threshold again |
| `crates/incin-core/src/tensor/base.rs` | central Tensor definition, invariant-preserving constructors, and conversions; split into `base/{types,error,accessors,placed,distributed,local,creation,convert}.rs` per `docs/CONVENTIONS.md`'s file-organization convention, keeping the `Local`-placement constructor family in one file (`local.rs`) since `docs/HANDOFF.md` previously flagged that path as a single invariant boundary not to scatter; every `Tensor` field stays `pub(crate)` as before, so the only privacy this split actually enforces (`ConstructionWitness` staying file-private to `local.rs`) is unchanged | further split only if a file grows past the threshold again |
| `crates/incin-core/src/tensor/dtype.rs` | logical dtype descriptors, built-in dtype implementations, and storage encodings; split into `dtype/{registry,traits,builtin,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention, keeping the mutually referential registry types (`DTypeKey`/`DTypeKind`/`StorageEncoding`/`DTypeDescriptor`/`DTypeId`) in one file rather than crossing that seam | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/cuda/backend.rs` | feature-gated backend identity, storage, capability, and executor implementations; split into `backend/{types,shape_ops,elementwise,creation,reduce,nn,contract,autograd,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/wgpu/backend.rs` | feature-gated backend identity, storage, capability, and executor implementations; split into `backend/{types,util,contract,creation,elementwise,shape_ops,reduce,nn,autograd}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/cpu/canonical.rs` | canonical registrations shared across the CPU backend; split into `canonical/{common,creation,elementwise,linalg,nn,reduce,shape_ops,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/capability.rs` | authoritative native-backend capability registrations; split into `capability/{constants,declarations,rules,tables,query,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention. The four `pub static ..._CAPABILITIES` tables carry no `#[cfg]` and stay that way after the split: a capability claim is data, reported by `registry`/`coverage_report` regardless of which backends are compiled in, so `tables.rs` reaches every backend's `*_descriptor_operations!` macro through an ungated `pub(crate)` path in `declarations.rs`; only the *executor*-facing re-export of that same macro (consumed by each backend's own coverage-assertion callback) keeps the original feature gate | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/cpu/ops/elementwise_kernel.rs` | architecture-specialized elementwise kernel bodies | split into `elementwise_kernel/{mod,types,dispatch,scalar,avx2,neon,wasm,strided,util,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/cpu/ops/elementwise.rs` | elementwise op dispatch and canonical unary/binary registrations | split into `elementwise/{mod,index,dispatch,unary,binary,softmax,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/cpu/ops/shape_ops.rs` | shape-manipulating op registrations | split into `shape_ops/{mod,view,combine,convert,select,linalg,cmp,triangular,norm,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/cpu/ops/reduce.rs` | CPU reduction kernels; split into `reduce/{helpers,all,dim,select,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/cpu/ops/matmul.rs` | stride-aware CPU matmul; split into `matmul/{types,transpose,batched,unbatched,gemm,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/cpu/ops/conv.rs` | `conv1d`/`conv2d`/`conv_transpose2d` via im2col + `batched_matmul_impl`; split into `conv/{helpers,unfold1d,window,conv1d,conv2d,conv_transpose2d,combine,tests}.rs` per `docs/CONVENTIONS.md`'s file-organization convention | further split only if a file grows past the threshold again |
| `crates/incin-backends/src/{dist/nccl.rs,dist/tuning.rs,tuning/identity.rs,tuning/service.rs}` | feature-gated distributed/tuning services | keep experimental ownership local; split resource protocols when they stabilize |
| `crates/incin-backends/src/kernel.rs` | kernel template/rendering and specialization test vocabulary | retain as a mechanical kernel source boundary |
| `crates/incin-diagnostics/src/lib.rs` | diagnostic command and report surface | split command families when the diagnostic API stabilizes |
| `crates/incin-core/src/optim/mod.rs` | `Optimizer`/`OptimizerBackend`/`ValueClippingBackend` traits, `SGD`/`AdamW`/`Adam`, and the `clip_grad_norm`/`clip_grad_value` free functions share one contract | keep the per-optimizer-backend blanket impls together; extract individual optimizers only once a second consumer needs them independently |

These are explicit staged extraction targets, not permission to add more
responsibilities. The current checkpoint documents the boundary and adds the
first capability seam; it does not claim the final split is complete.

## What not to change casually

Do not add a second public execution architecture, speculative runtime/session
handles, compatibility adapters for superseded operation families, backend
requirements that force autograd or state, or hand-edits to generated docs.
Do not weaken shape/dtype/device/error invariants to make a fixture compile.

## Current baseline

Broad stabilization is complete. The current baseline includes rectangular,
static, mixed, named-axis, and dynamic `Shape` forms; canonical
`Descriptor<O>` and `Execute<O>` operation execution; typed `Tensor` and
`Var<K>` values; `VisitParameters` and `ParameterGroup`; transactional state
loading; exact optimizer-state errors; lazy and error-reporting `DataLoader`
behavior; current CPU autograd; the explicit backend capability model; the
Book and generated capability/operation documents; and the executable
Transformer-style differentiable composition proof. The canonical exporter
also verifies the tracked source set and required distributed source trees.

The current feature contract is the 32-row general matrix in
`docs/FEATURE_MATRIX.md`, plus the dedicated platform and hardware jobs named
there. It does not promise every Cartesian combination of optional Cargo
features.

## Experimental and not permanently frozen

Target API, compiled execution, distributed execution, telemetry/viz plugin
boundaries, and accelerator/runtime maturity remain experimental or
platform-dependent according to their feature and module documentation. These
surfaces are validated at their documented compile or hardware boundary and
are not presented as stable CPU-only guarantees.

## Future human-directed architecture

Deliberate future topics are mutation and alias semantics, training-state
semantics, dtype and storage identity, backend resource ownership, placement,
ragged tensors, sparse tensors, custom-operation VJP/JVP and batching,
higher-order autodiff, compiler evolution, and distributed runtime evolution.
These are isolated future subsystem questions, not reasons for another
repository-wide stabilization pass.

## Final stabilization checkpoint

The final reproducible validation and export evidence is tracked under
`audit-evidence/HND-final/summary.md`. Its validation is run from the final
source commit and includes formatting, workspace checks, tests and doctests,
supported clippy, the complete feature contract, documentation, structural
gates, soundness status, package checks, mdBook, and canonical export
metadata. Hardware-only runtime checks are explicitly marked not run when no
matching runner is available.

The canonical artifact is created only with:

```text
tools/export-snapshot.sh <output.zip>
```

The broad HND stabilization sequence is complete. No further repository-wide
migration is required before normal human-owned development. Future work
should proceed through focused subsystem tasks.

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
