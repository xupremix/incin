# Incin Foundation-First Remediation Prompt

You are continuing development of the Incin Rust workspace from the current `develop` branch.

The inspected starting revision is:

```text
fa8d2030141b04bc7c0dfccb382bfa60647223cf
fix(api-001): harden facade; all 1233 workspace tests pass
```

That commit subject and the existing completion summaries are **claims, not evidence**. Revalidate everything from source and fresh command output.

This run is not about maximizing the number of checked tasks. Its purpose is to build the smallest set of durable foundations that later tensor, backend, compiler, ONNX, data, and distributed work can depend on without being rewritten.

---

## 1. Read before modifying source

Read these files in this order:

1. `AGENTS.md`
2. `docs/plan/remediation/codebase-truth-audit.md`
3. `audit-evidence/API-001/summary.md`
4. `audit-evidence/API-001/api-before.txt`
5. `crates/incin/src/lib.rs`
6. `crates/incin-core/src/lib.rs`
7. `crates/incin-core/src/tensor/backend.rs`
8. every file under `crates/incin-core/src/exec/`
9. the CPU executor and capability implementation under `crates/incin-backends/src/`

When `graphify-out/graph.json` and a usable `graphify` executable are available, use them to inspect dependencies before broad manual browsing. Archive the exact command and output. If the executable is unavailable, record that fact and continue from source; do not claim that graph analysis ran.

Before production changes, record:

```bash
git rev-parse HEAD
git status --short
git log -5 --oneline --decorate
rustc --version --verbose
cargo --version
```

Never rewrite existing commits. Create focused new commits after each completed foundation task.

---

# 2. Governing objective

Implement foundations in dependency order:

```text
truth and containment
    -> public boundary and stability tiers
        -> invariant-bearing types and checked arithmetic
            -> error and failure contract
                -> operation semantics and validated descriptors
                    -> CPU reference execution contract
```

Later systems must build on those foundations:

```text
CPU reference contract
    -> autograd/module reliability
    -> ONNX/model loading
    -> data/model training workflows
    -> compiled CPU execution
    -> accelerator breadth
    -> compiler optimization/tuning
    -> distributed/release work
```

Do not reverse this order merely because a later feature is more visible.

---

# 3. Architectural rules that must remain true

These are design invariants, not suggestions.

## 3.1 Truthful behavior

- A feature that is not semantically implemented must be private, explicitly experimental, or fail with a typed unsupported error.
- A no-op pass must not be described as optimization.
- Missing metadata must never be replaced with invented shape, dtype, device, rank, attributes, or initializer values.
- Static model errors must fail during macro expansion, not generate runtime panics.
- Unsupported operations must be knowable before kernel execution.

## 3.2 One source of truth per operation

Every public tensor operation must eventually have exactly one canonical semantic descriptor containing or deriving:

- operation identity;
- attributes;
- input requirements;
- broadcasting/rank rules;
- dtype rules;
- device and placement rules;
- output metadata inference;
- gradient support status;
- deterministic/nondeterministic status;
- backend capability identity.

Tensor methods, backend dispatch, capability reporting, compiler capture, generated documentation, and conformance tests must consume this same descriptor contract. Do not create parallel hand-maintained operation tables.

## 3.3 Validation before execution

Validate shapes, dtypes, devices, placement, attributes, element counts, byte lengths, and capabilities before:

- allocating storage;
- launching a backend operation;
- mutating a parameter or optimizer state;
- serializing an artifact;
- accepting untrusted model/data dimensions.

## 3.4 CPU is the semantic reference

CPU eager execution is the first complete reference implementation. Other backends may be narrower, but they must match CPU semantics for every operation they claim to support.

Do not implement broad CUDA/WGPU/Metal/Candle coverage before CPU semantics and exact capability reporting are stable.

## 3.5 Public API stability tiers

Use only these tiers:

- **stable facade:** ordinary end-user tensor, module, optimizer, shape, dtype, device, error, and documented macro APIs;
- **backend authoring:** extension contracts required to implement a backend;
- **experimental:** unstable compiler, tuning, distributed, training automation, and model-import surfaces;
- **test utilities:** mocks and fixtures available only behind an explicit test feature.

Do not expose an unstable subsystem at the stable root merely because its internal types are public.

## 3.6 Invariants cannot be forged

A type named `Checked*`, `Validated*`, `Guard`, `Proof`, `Slot`, `Id`, or representing a backend/device identity must not expose tuple fields that bypass validation.

A pure marker that carries no invariant may remain trivially constructible. In particular, preserve ergonomic value-level use of `Dyn` by making it a safe zero-sized marker such as:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Dyn;
```

Do not hide the `Dyn` type and do not require an unsafe or internal constructor for it.

## 3.7 Recoverable errors do not panic

Public library paths must return typed errors for invalid user input, malformed files, backend failure, unsupported operations, allocation overflow, and inconsistent serialized state.

Operator traits may have an explicitly documented panic contract if Rust's trait signature cannot return `Result`, but library internals must use fallible methods rather than those operators.

---

# 4. Confirmed starting-state contradictions

Treat these as defects to verify and repair, not as optional discoveries.

## 4.1 `API-001` is not complete

The current source still contains cross-crate wildcard facade exports, including in `crates/incin/src/lib.rs`:

- `incin_core::compile::*`;
- `incin_core::backend_authoring::*`;
- `incin_core::test_utils::*`;
- `incin_core::nn::*`;
- `incin_core::metrics::*`;
- `incin_core::dist::*`;
- `incin_data::*`;
- `incin_data::transforms::*`;
- `incin_data::hub::*`.

The core prelude still exposes graph and implementation surfaces through items such as `Graph`, `OpType`, `shapes::prelude::*`, and `tensor::prelude::*`.

The facade prelude still includes backend/helper contracts such as `SupportsDType`, `TransferTo`, and autoref fallback traits that are not ordinary end-user imports.

`audit-evidence/API-001/` contains only the before snapshot and a summary. It lacks the command logs, after snapshots, compile-contract fixtures, feature evidence, rustdoc evidence, and semver/API review that the summary claims or acknowledges as pending.

## 4.2 Invariant constructors remain public

Examples include:

- `Dyn(pub ())`;
- `CheckedNumel(pub usize)`;
- `CheckedByteLen(pub usize)`;
- `BufferSlot(pub usize)`;
- runtime device markers with public tuple fields.

Audit every similar public tuple struct rather than fixing only these examples.

## 4.3 The execution architecture is duplicated

`StorageBackend` and `Execute<O>` exist, but `Backend` still inherits the old broad operation-trait surface. Both architectures coexist, which forces backend work, capability reporting, and compiler work to depend on two contracts.

## 4.4 Compiled execution is a prototype

The current implementation still contains behavior such as:

- constant folding returning a graph clone and an empty folded set;
- weight prepacking returning a clone;
- lossy capture and fabricated guards;
- fusion that does not preserve complete consumer semantics;
- no validated public `compile -> executable -> run` path.

Do not implement compiler optimization during this foundation run. Contain the public claims instead.

## 4.5 ONNX import is not product-safe

The macro still contains behavior such as:

- fallback to four dynamic dimensions when rank is unknown;
- token parsing with `unwrap` or silent empty fallback;
- generated runtime `panic!` for malformed `If` and `Loop` nodes;
- zero-filled generated parameters with `unwrap`;
- a `load_default_weights()` path that does not load real weights.

Do not build convenience features on this behavior. Make it fail closed during the containment stage.

## 4.6 Existing test-count claims are unproven

Do not repeat “1,233 tests pass” unless a fresh archived run on the current revision reproduces it. If the environment cannot run Rust, mark validation `BLOCKED`; never copy the old number.

---

# 5. Scope and return-on-time policy

Complete the tasks below in strict order. Do not begin a later task until the current task's acceptance gate is satisfied.

This run stops after **FND-005**. Do not implement compiled execution, broad accelerator coverage, real ONNX initializer loading, distributed features, or performance tuning in this run.

| Task | Purpose | Expected investment | Return | Rework risk |
|---|---|---:|---|---|
| FND-000 | Truth reset and feature containment | 0.5–1.5 days | Immediate | Very low |
| FND-001 | Stable facade and stability tiers | 1–3 days | Very high | Low |
| FND-002 | Opaque invariants and checked arithmetic | 2–5 days | Very high | Very low |
| FND-003 | Typed error/failure contract | 3–7 days | Very high | Low |
| FND-004 | Canonical operation semantics/descriptors | 5–10 days | Highest architectural return | Low if done before backend work |
| FND-005 | CPU reference executor migration | 1–3 weeks | Highest implementation return | Moderate, but required |

These are planning ranges, not deadlines. Report actual work and blockers rather than claiming the estimate was met.

Do not spend foundation time on:

- exhaustive GPU kernels;
- benchmark tuning;
- compiler fusion/folding/prepacking;
- multi-node distributed execution;
- polished visualization;
- broad semver archaeology across internal `0.0.0` crates.

For this run, API tooling should protect the external `incin` facade. Internal crate API snapshots are useful diagnostics but are not a blocker unless those crates are explicitly promised as public extension surfaces.

---

# 6. FND-000 - Reset truth and contain false product claims

## 6.1 Reopen false task states

Before production code changes:

- change `audit-evidence/API-001/summary.md` from `COMPLETE` to `INCOMPLETE`;
- list the exact surviving wildcard exports and leaked internal names;
- state that the archived evidence does not reproduce the claimed test run;
- preserve historical text in a clearly labeled `Previous claim` section rather than deleting it.

Create `docs/PROJECT_STATUS.md` with these columns:

| Subsystem | Implemented behavior | Known gaps | Public tier | Evidence | Next dependency |
|---|---|---|---|---|---|

The status document must distinguish:

- complete and dynamically verified;
- implemented but not verified in this environment;
- partial;
- structural prototype;
- intentionally unsupported;
- blocked by hardware.

## 6.2 Contain compiled claims

Until a real executable path exists:

- remove compiled internals from the stable root and default prelude;
- place any necessary inspection-only surface behind an explicit experimental feature and namespace;
- document that capture/artifact types are non-executable prototypes;
- rename methods or docs that imply execution when they only serialize or inspect plans;
- make no-op folding/prepacking APIs private or return an explicit `Unsupported`/`NotImplemented` result instead of success.

Do not implement the compiler in this task.

## 6.3 Make ONNX fail closed

Implement only the durable safety containment now:

- malformed static ONNX metadata produces a precise compile error;
- unknown rank remains unknown and is never fabricated as rank four;
- remove generated runtime panics for malformed control-flow attributes;
- remove or rename the no-op `load_default_weights()` method;
- do not generate zero-filled parameters as if they were imported weights;
- unsupported operators/control flow fail at macro expansion with node, op, domain, and opset context.

Do not implement full initializer loading or ONNX control flow in this task.

## 6.4 Acceptance gate

FND-000 passes only when:

- status documents no longer call prototypes complete;
- compiled and ONNX false-success behavior is removed or explicitly rejected;
- tests cover the fail-closed paths;
- evidence is archived under `audit-evidence/FND-000/`.

---

# 7. FND-001 - Lock the public boundary and stability tiers

## 7.1 Stable root

The default `incin` root may expose only deliberate end-user contracts. Preserve ergonomic access to:

- `Tensor`;
- `IncinBackend` and the supported default backend/device aliases;
- `Result` and `Error`;
- `Shape`, `ConstShape`, `DynShape`, `PartialDynShape`, and `Dyn`;
- public dtype/device/gradient markers;
- common documented modules and optimizers;
- stable user macros.

Do not expose backend operation traits, graph IR, compiler passes, test backends, raw storage, validation proofs, autoref fallback machinery, tuning candidates, or service internals at the root.

## 7.2 Default prelude

Use an explicit allow-list. Include only high-frequency end-user names.

The default prelude must not contain:

- `Graph` or `OpType`;
- `Execute`, `ExecutionRequest`, `OperationSpec`, `Validated`, or capability registry contracts;
- `SupportsDType` or backend transfer/extension traits unless a normal tensor user must name them;
- autoref fallback traits;
- compiler, tracing, raw storage, tuning-service, or test types.

Do not fix a facade wildcard by hiding another wildcard inside a public prelude.

## 7.3 Backend-authoring tier

Expose only contracts required by an external backend implementation through an explicit `incin::backend_authoring` allow-list. Feature-gate it if that remains the intended product policy.

Separate:

- stable storage/device/backend identity contracts;
- validated operation execution contracts;
- optional backend-author tuning hooks.

Do not mirror all of `incin-core` or `incin-backends`.

## 7.4 Experimental and test tiers

- Keep compiled/tuning/distributed/trainer/model-import surfaces out of the stable prelude unless they have stable semantics.
- Put genuinely unstable surfaces under an explicit experimental namespace or clearly documented feature-specific module.
- Gate test utilities only with `test-utils` or `cfg(test)`.
- `DummyBackend` must never appear in a normal build.

## 7.5 Replace facade globs

Eliminate cross-crate wildcard public re-exports in the public `incin` facade, including `nn`, `metrics`, `dist`, `data`, `transforms`, `hub`, `compile`, `backend_authoring`, and `test_utils`.

Internal module-local re-exports inside an owning crate may remain when they are an intentional internal organization tool, but they must not create an uncontrolled external facade.

## 7.6 Contract tests

Add isolated consumer fixtures proving:

1. default prelude tensor/model use;
2. `Dyn` type and value usability;
3. backend authoring imports only from its tier;
4. internal names are absent from root and prelude;
5. compiled/experimental names are absent without their feature;
6. test utilities are absent without `test-utils`;
7. feature-disabled builds do not accidentally expose aliases whose bounds cannot be satisfied.

Use compile-pass and compile-fail tests. Review every expected diagnostic.

## 7.7 Evidence proportional to value

Archive:

- before/after `cargo public-api` for the `incin` facade;
- compile-contract logs;
- default/no-default/CPU feature checks;
- rustdoc warnings;
- a reviewed migration table for removed facade paths.

Do not block this task on semver reports for every private internal crate. Run `cargo semver-checks` for `incin` when available and archive the result; if unavailable, record the tool failure without pretending it ran.

---

# 8. FND-002 - Make invariants explicit and unforgeable

## 8.1 Classify public data types

Create `docs/INVARIANT_TYPES.md` with a table:

| Type | Category | Invariant | Public construction | Validation | Accessor |
|---|---|---|---|---|---|

Classify each as:

- marker with no invariant;
- validated value;
- opaque identifier;
- proof token;
- configuration;
- runtime handle.

Audit public tuple fields across core, backends, compiled prototypes, tuning, distributed, and data APIs.

## 8.2 Required treatment

- Pure markers such as `Dyn` become ordinary zero-sized structs with `Default`/`new` where useful.
- Validated numeric wrappers such as element counts and byte lengths get private fields and checked constructors.
- IDs/slots get private fields, explicit accessors, and constructors limited to the subsystem that allocates them.
- Proof types such as `Validated<T>` cannot be built without running validation.
- Runtime device indices use explicit constructors and do not imply device availability before probing.
- Deserialization must run the same invariant checks as normal construction.

Do not add `unsafe fn new_unchecked` to the public API unless a documented performance-critical need is proven. Keep unchecked constructors crate-private.

## 8.3 Central checked arithmetic

Create or consolidate checked helpers for:

- shape element count;
- byte length;
- strides;
- offsets and slices;
- concatenation/stack output sizes;
- allocation sizes;
- model/data dimension conversion.

The helpers must return typed errors that include the operation and offending dimensions. Remove ad hoc `product`, cast, and `unwrap` paths at allocation boundaries.

## 8.4 Tests

For each invariant-bearing type test:

- valid boundary values;
- zero behavior where legal;
- overflow;
- malformed deserialization;
- round trip;
- inability to construct through the public tuple syntax.

Use property tests where arithmetic combinations are large.

---

# 9. FND-003 - Establish one typed error and failure contract

## 9.1 Preserve compatibility where sensible

Do not gratuitously replace every existing error type. First inventory the current `Error`, `BackendError`, shape, dtype, serialization, data, and macro errors. Consolidate them behind a coherent public error contract while preserving useful existing variants.

## 9.2 Required categories

The public contract must represent, directly or through sourced variants:

- invalid shape/rank/axis/broadcast;
- dtype mismatch or invalid conversion;
- device/placement mismatch;
- unsupported operation/capability;
- allocation and arithmetic overflow;
- backend execution failure;
- autograd failure/non-finite policy;
- invalid module/state dictionary;
- malformed model/data/artifact;
- I/O/resource limit;
- internal invariant violation.

Errors must include operation identity and relevant metadata without dumping unbounded tensor contents.

## 9.3 Public panic remediation

Audit production `panic!`, `unwrap`, and `expect` paths. Do not mechanically remove every occurrence. Classify each occurrence as:

- statically impossible and represented by types;
- debug/test assertion;
- process boundary;
- recoverable public/backend/I/O failure.

Convert recoverable cases. Prioritize:

- tensor operator internals;
- shape/index conversion;
- optional module bias/state;
- optimizer state mutation;
- CUDA/WGPU/Metal failure propagation;
- macro expansion and model I/O;
- data loader and download boundaries.

## 9.4 Transactional optimizer behavior

Fix optimizer updates so a backend failure cannot remove or partially mutate moment/state tensors or parameters.

Required sequence:

1. validate all parameters, gradients, dtype/device/capability, and arithmetic;
2. compute candidate parameter/state updates without committing;
3. commit all related values atomically;
4. on any error, preserve the exact pre-step parameter and optimizer state.

Add an injected failure backend/test proving rollback.

## 9.5 Scalar conversion

Replace silent float-to-integer truncation of NaN, infinity, and out-of-range values with checked conversion. If truncation/saturation is needed, make the mode explicit in the API.

---

# 10. FND-004 - Freeze canonical operation semantics and descriptors

This is the most important design task in the run. Do not rush it or patch only the visible operations.

## 10.1 Inventory every public operation

Generate a machine-reviewed inventory of every public tensor, creation, reduction, module, loss, optimizer, transfer, and quantized operation.

For each operation record:

- canonical operation ID;
- descriptor type;
- attributes;
- input arity;
- accepted ranks;
- broadcasting;
- dtype constraints/promotion;
- output shape and dtype inference;
- device/placement constraints;
- empty tensor behavior;
- NaN/infinity/overflow behavior;
- gradient definition/support;
- determinism;
- aliasing/layout behavior;
- backend support status.

Write the human-readable form to `docs/OPERATION_SEMANTICS.md`, but make code - not prose - the source consumed by validation and capability reporting.

## 10.2 Descriptor contract

Each operation descriptor must:

- retain all semantic attributes;
- validate input metadata without storage access where possible;
- infer exact output metadata;
- produce an opaque validated invocation/proof;
- expose a stable operation identity for errors, capability lookup, tracing, and compiler capture;
- not invent defaults when metadata is absent.

Use concrete descriptor structs rather than a single untyped bag of strings/maps. Shared helper traits and enums are encouraged when they preserve typed attributes.

## 10.3 Single registry

Build one registry or generated inventory consumed by:

- capability documentation;
- runtime/dynamic capability checks;
- descriptor conformance tests;
- backend coverage reports;
- compiler capture eligibility.

Do not create separate manually maintained CPU/GPU/docs lists.

## 10.4 Semantic conformance vectors

For each operation family, add reusable CPU reference vectors covering:

- normal values;
- scalar and rank edge cases;
- broadcasting;
- zero-length dimensions where legal;
- invalid axes/shapes;
- dtype boundaries;
- NaN/infinity;
- gradient checks where applicable.

Backends later reuse these vectors.

## 10.5 Design gate

Before migrating execution, prove:

- all currently public operations appear exactly once in the inventory;
- every descriptor preserves all attributes currently required by eager execution;
- capability docs are generated from the same source;
- capture can serialize/retain descriptor semantics without depending on backend storage;
- no output shape/dtype is fabricated.

Archive a reviewed mapping from old trait method to descriptor type.

---

# 11. FND-005 - Migrate CPU eager execution to the durable contract

Do not implement compiler execution or GPU breadth. The goal is to make ordinary CPU eager execution consume the new stable contracts.

## 11.1 End the dual architecture

Refactor `Backend` toward storage/backend identity and essential lifecycle only. Remove operation-family supertraits from the central backend identity contract.

Operations execute through typed descriptor contracts such as `Execute<O>` or an equivalent architecture that satisfies these properties:

- support for an operation is explicit;
- descriptor validation occurs before execution;
- no default method silently returns unsupported;
- ordinary tensor methods depend only on the operation capability they use;
- dynamic dispatch uses the exact capability registry;
- compiler capture can retain the same descriptor.

A private temporary compatibility adapter is allowed only during the migration. It must not be public, must be clearly marked for deletion, and must not become the source for new implementation.

## 11.2 Complete CPU first

Migrate the entire stable CPU eager operation surface, not just a demonstration method. CPU becomes the semantic oracle for the operation inventory.

For every operation:

- descriptor validation matches `OPERATION_SEMANTICS.md`;
- execution matches reference vectors;
- metadata matches actual storage;
- errors are typed;
- gradients match finite differences where defined;
- unsupported dtype/quantized cases are declared exactly.

Do not claim CPU completeness merely because trait methods exist.

## 11.3 Exact capability truth

Generate CPU capability output from actual descriptor implementations and dtype/device constraints.

For non-CPU backends during this run:

- preserve compilation where feasible;
- declare only the operations actually implemented;
- do not broaden support claims;
- do not add placeholder implementations to satisfy a trait;
- a backend feature may remain experimental or temporarily blocked with an explicit migration issue if the old architecture cannot be safely retained.

## 11.4 Eager regression and parity

Required CPU tests include:

- tensor creation and byte round trips;
- reshape/transpose/broadcast/slice/concat/stack;
- arithmetic and pointwise functions;
- reductions including empty-axis behavior;
- matmul and batched matmul;
- convolution/pooling/normalization;
- modules and losses;
- autograd for the supported differentiable subset;
- optimizer success and rollback;
- serialization/state consistency.

Use tolerances appropriate to dtype and operation. Record tolerance rationale.

## 11.5 Completion condition

FND-005 passes only when:

- stable CPU tensor methods no longer rely on the old monolithic operation supertrait architecture;
- the canonical descriptors drive validation, execution identity, capability reporting, and tests;
- CPU support is exact and source-generated;
- all software-only validation commands pass;
- remaining accelerator/compiler/model/data work has explicit dependencies on this foundation.

Stop after this task. Do not proceed into compiled execution or accelerator breadth in the same run.

---

# 12. Deferred work and why it is deferred

Record these in `docs/PROJECT_STATUS.md`; do not implement them during this run.

## 12.1 Real ONNX loading

Deferred until module/state invariants and CPU semantics are stable. Immediate safety containment is required in FND-000, but real initializers, control flow, and opset breadth come later.

## 12.2 Full data-pipeline reliability

Deferred until the core error contract is stable. Later work must address zero batch size, worker errors/lifecycle/order, dataset validation, transform invariants, and download integrity.

## 12.3 Compiled execution

Deferred until descriptors and CPU execution are canonical. The first future compiler milestone is a CPU-only `capture -> validated IR -> executable -> run` vertical slice with eager parity. Fusion, folding, prepacking, tuning, and artifact hardening come only after that.

## 12.4 Accelerator breadth

Deferred until exact semantics and capability reporting exist. Implementing dozens of kernels before then creates duplicated validation and repeated rewrites.

## 12.5 Distributed and performance tuning

Deferred until local semantics, storage identity, error behavior, and artifacts are stable. Hardware claims require archived hardware logs.

---

# 13. Evidence and command policy

Create one directory per task:

```text
audit-evidence/FND-000/
audit-evidence/FND-001/
audit-evidence/FND-002/
audit-evidence/FND-003/
audit-evidence/FND-004/
audit-evidence/FND-005/
```

Each directory must contain:

```text
summary.md
commands.log
environment.txt
changed-files.txt
test-results/
known-limitations.md
```

API tasks additionally include before/after public API snapshots and compile-contract results. Semantic tasks include the operation inventory and conformance summary.

Every command log records:

- exact command;
- working directory;
- relevant environment/features;
- start/end timestamp;
- exit code;
- full output or path to full output;
- commit hash.

Do not paste a result from a prior commit.

## 13.1 Software-only validation baseline

Resolve exact feature combinations from manifests, then run at least:

```bash
cargo fmt --all -- --check
cargo check -p incin-core --no-default-features
cargo check -p incin-core --no-default-features --features std
cargo check -p incin --no-default-features
cargo check -p incin --no-default-features --features std
cargo check -p incin --features cpu
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p incin-core
cargo test -p incin-backends --features cpu
cargo test -p incin
cargo test --doc --workspace
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
```

Do not use `--all-features` as the only gate when it requires unavailable CUDA, Metal, NCCL, or platform libraries. Instead define and archive a software-only feature matrix plus separate hardware/platform jobs.

Run `cargo public-api` for the `incin` facade before and after FND-001 when the tool is available.

If a tool is missing and installation is permitted, install it and record the version. If installation or hardware is unavailable, archive the exact failure, mark only the affected criterion `BLOCKED`, and continue only when the remaining task can be truthfully completed without it.

## 13.2 No fabricated completion

A task is not complete because:

- code was written;
- the workspace compiled once;
- a test count was copied;
- a checklist was edited;
- an unsupported branch was hidden by a feature;
- a no-op returned `Ok`;
- only structural tests passed.

A task is complete only when every acceptance criterion links to current evidence.

---

# 14. Commit policy

Create focused commits in this order:

1. `audit(fnd-000): reset project truth and contain false claims`
2. `refactor(fnd-001): establish stable facade and API tiers`
3. `refactor(fnd-002): make invariant types opaque and arithmetic checked`
4. `fix(fnd-003): establish typed failure and rollback contracts`
5. `refactor(fnd-004): canonicalize operation semantics and descriptors`
6. `refactor(fnd-005): migrate CPU eager execution to descriptor contract`

Do not combine unrelated formatting or feature work. Do not amend or rewrite the inspected commits.

---

# 15. Final report format

At the end, report:

1. starting and ending commit hashes;
2. status of FND-000 through FND-005: `DONE`, `PARTIAL`, or `BLOCKED`;
3. exact architectural contracts now frozen;
4. public API paths added, moved, or removed;
5. remaining old execution paths and why they remain;
6. commands run and exit codes;
7. test counts reproduced from the current commit only;
8. blockers and missing hardware/tooling;
9. the next recommended task, which must be one of:
  - autograd/module conformance on the CPU contract;
  - real ONNX initializer/state loading;
  - data-pipeline reliability;
  - compiled CPU vertical slice.

Do not recommend GPU breadth, compiler optimization, or distributed expansion unless FND-005 is complete.
