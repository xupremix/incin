# Incin Next-Steps Execution Prompt — Finish the Public API Remediation

You are continuing remediation of the Incin Rust workspace from the current repository state.

## Current repository state

- Work from the current checked-out `develop` branch.
- The inspected archive was at commit:
  - `8f90364a0226655d36a43bb391446fdf0f490964`
  - commit subject: `refactor(core): finalize backend_authoring facade migration and trait visibility`
- Read these files before editing:
  1. `AGENTS.md`
  2. `docs/plan/remediation/codebase-truth-audit.md`
  3. `audit-evidence/API-001/summary.md`
  4. `audit-evidence/API-001/api-before.txt`
- Follow `AGENTS.md`. When `graphify-out/graph.json` exists, use `graphify query`, `graphify path`, or `graphify explain` before broad source browsing. If the `graphify` executable is unavailable, record that fact in evidence and continue with source inspection; do not pretend it ran.

The current commit is **not accepted as completion of API-001**. Its commit message and any prior ledger checkboxes are claims, not evidence.

## Confirmed problems in the current state

Treat the following as known defects that must be resolved, not rediscovered and ignored:

1. `audit-evidence/API-001/summary.md` still has every acceptance criterion unchecked.
2. `audit-evidence/API-001/` contains no `api-after.txt`, `commands.log`, or archived semver report.
3. `crates/incin/src/lib.rs` still contains public wildcard re-exports, including:
   - `incin_core::compile::*`
   - `incin_core::backend_authoring::*`
   - `incin_core::test_utils::*`
   - `incin_core::nn::*`
   - `incin_core::optim::*`
   - `incin_core::metrics::*`
   - `incin_core::dist::*`
   - `incin_data::*`
   - `incin_data::transforms::*`
   - `incin_data::hub::*`
   - `incin_backends::prelude::*`
   - `incin_core::prelude::*`
4. Backend-authoring traits are still re-exported at the stable `incin` root, including `CreationOps`, `FloatOps`, `ModuleOps`, `NumericOps`, `ReductionOps`, `SupportsDType`, and `TensorOps`. These belong only in the backend-authoring tier unless a concrete user-facing API proves otherwise.
5. `incin::compile` is exposed unconditionally even though it is intended to be a preview/experimental feature tier.
6. `incin_core::test_utils` is public unconditionally, `incin_core::prelude` exposes it as `dummy`, and `incin-backends` re-exports that alias at its root. `DummyBackend` therefore remains visible in production configurations.
7. `incin_core::prelude` still exposes graph/compiler-oriented names such as `Graph` and `OpType`; these must not remain in the default end-user prelude without an explicit, documented user requirement.
8. The current API change touched many executor/backend files but did not provide the required task evidence. Do not broaden the next change further.
9. `Dyn` is publicly nameable, but remains `pub struct Dyn(pub ())`. The final API must preserve user ergonomics while preventing tuple-field forgery.
10. Removing the old root wildcard also made tuning controls difficult or impossible to reach through the `incin` facade. A deliberate, feature-gated tuning namespace is required; do not restore the wildcard.

## Mission

Complete the following tasks in strict order:

1. `API-001` — finish the public facade and prelude reconstruction.
2. `API-002` — make marker, proof, planner, tuning, and backend-storage constructors safe and opaque while preserving legitimate user construction.
3. `API-003` — remove production exposure of test and prototype internals.

Do not begin `API-002` until every `API-001` acceptance criterion is proven. Do not begin `API-003` until every `API-002` acceptance criterion is proven. Stop after `API-003`; do not begin backend breadth, compiled execution, ONNX, data, or distributed remediation in this run.

If a required tool or platform is unavailable, mark the current task `BLOCKED`, archive the exact failure, and stop. Never mark a task done from source inspection alone.

---

# Task 1: API-001 — Finish the public facade and prelude

## 1. Re-open the task honestly

Update `audit-evidence/API-001/summary.md` before editing:

- add the current commit hash;
- add a section named `Post-8f90364 verification`;
- list each remaining violation above with exact file and line/symbol references;
- leave all acceptance boxes unchecked until corresponding evidence exists;
- record that the prior commit changed files outside the narrow facade boundary and that no archived command evidence was present.

Do not delete the existing before-state record.

## 2. Define the API tiers

Implement these tiers explicitly.

### Stable `incin` root

The stable root may expose only deliberate, user-facing names. At minimum, preserve ergonomic access to:

- `Tensor`
- `IncinBackend`
- `DefaultBackend` and `DefaultDevice` when applicable
- `Result` and `Error`
- `Shape`, `ConstShape`, `DynShape`, `PartialDynShape`, and the public `Dyn` marker type
- `DType`, `DTypeId`, `DeviceId`, `Grad`, `NoGrad`
- common modules and aliases intentionally documented in the README
- public macros intentionally documented for users

Do not export backend operation traits, compiler passes, planner internals, test backends, autoref fallback traits, raw backend storage, launch candidates, or graph IR through the stable root.

### Stable `incin::prelude`

Replace both wildcard imports with an explicit allow-list. The prelude should contain only high-frequency end-user names required by examples and normal model code.

It must not contain:

- `Graph` or `OpType`;
- `Execute`, `ExecutionRequest`, `OperationSpec`, `Validated`, or capability/executor traits;
- compiler pass or artifact representation types;
- autoref fallback traits;
- `DummyBackend` or a `dummy` alias;
- tuning service internals;
- backend storage types.

Add a compile-pass contract fixture that imports `incin::prelude::*` and exercises the intended public surface without importing internal crates directly.

### `incin::backend_authoring`

This is the only normal facade location for backend extension contracts. Use explicit re-exports, not `*`.

Curate the list from actual external-backend author requirements. It may include contracts such as:

- `Backend`
- `StorageBackend`
- `Execute`
- `ExecutionRequest`
- `OperationSpec`
- `Validated`
- `CapabilityRegistry`
- operation-family traits required to implement a backend
- execution context and precision-policy types required by those traits

Do not mirror the whole core or backend crate.

Remove backend-authoring traits from the stable root and default prelude. Update all workspace call sites to import them from `incin_core::backend_authoring` or `incin::backend_authoring`, according to whether the call site is internal or an external-facing example/test.

### `incin::compile`

Expose this only through an explicit `compiled` feature on the `incin` facade. Add the feature if it does not exist. The feature may initially gate only facade exposure if the core implementation is still always compiled internally.

Use an explicit allow-list. Do not export every pass and representation merely because it is public in `incin-core`.

Until executable compiled semantics exist, expose only types that are honestly useful for preview inspection/configuration and clearly document the tier as preview or experimental. Do not call a plan or artifact executable if it is not executable.

### `incin::tuning`

Add a deliberate preview namespace gated by the relevant tuning feature, normally `autotune`. Do not restore any root wildcard.

Expose only safe user-facing tuning configuration, inspection, identity, cache, and explanation types. Start from this candidate list and remove any item that cannot yet be safely constructed or meaningfully used by an end user:

- `AutotunePolicy`
- `TuningScope`
- `TuningContext`
- `TuningSelection`
- `SelectionSource`
- `TuningExplain`
- `TuningProvenance`
- `CacheLimits`
- `CacheRecovery`
- `PersistentTuningCache`
- `DeviceFingerprint`
- `CompilerFingerprint`
- `TuningEnvironmentFingerprint`
- `KernelSignature`
- `RankClass`
- `AlignmentClass`
- `DTypePolicyId`

Keep raw kernel handles, internal measurement records, mutable launch-candidate fields, service coordination internals, and implementation-only pruning machinery out of the stable root and prelude. Put backend-author-only tuning hooks under `incin::backend_authoring::tuning` if they are genuinely required.

Document every exposed tuning type with:

- required feature;
- whether it is stable, preview, or experimental;
- safe constructor or builder;
- invariants;
- whether it represents measured data, heuristic data, or configuration.

### `incin::test_utils`

The module must exist only under `#[cfg(any(test, feature = "test-utils"))]`. The same gate must exist in `incin-core`; no unconditional public `test_utils` module or prelude alias is allowed.

### Other public facade modules

Replace cross-crate wildcard re-exports in `nn`, `optim`, `metrics`, `data`, `transforms`, `hub`, and distributed facade modules with explicit documented allow-lists. A public facade module is still a facade; moving `*` one level down does not satisfy the task.

Where an exhaustive explicit list would be large, create a dedicated curated prelude/export module in the owning crate and explicitly re-export that named module's public contract. Do not use glob syntax in the external facade.

## 3. Preserve `Dyn` usability correctly

For `API-001`, users must be able to name `Dyn` in public generic positions, for example:

```rust
use incin::prelude::*;

type B = IncinBackend<f32, Cpu>;
let tensor = Tensor::<Dyn, B>::zeros(vec![2, 3])?;
```

Do not hide the `Dyn` type. Constructor opacity belongs to `API-002`.

## 4. Add migration documentation

Create or update a migration document with a table containing, for every removed or moved public path:

- old path;
- new path;
- required feature;
- stability tier;
- replacement example;
- whether the change is breaking.

At minimum, include paths moved from:

- `incin::*` to `incin::backend_authoring::*`;
- `incin::prelude::*` to `incin::backend_authoring::*`;
- compiler types to `incin::compile::*`;
- tuning types to `incin::tuning::*`;
- dummy/test types to `incin::test_utils::*`.

## 5. Add API fixtures

Add all of these:

1. Compile-pass: intended default prelude use.
2. Compile-pass: `Dyn` can be named and used for a dynamic tensor.
3. Compile-pass: backend author imports required contracts only from `incin::backend_authoring`.
4. Compile-pass under `autotune`: supported public tuning configuration/inspection types are reachable from `incin::tuning`.
5. Compile-fail: backend-authoring traits are absent from the stable root and default prelude.
6. Compile-fail without `compiled`: `incin::compile` is absent.
7. Compile-pass with `compiled`: curated compile API is present.
8. Compile-fail without `test-utils`: `DummyBackend` and `incin::test_utils` are absent.
9. Compile-pass with `test-utils`: the intended test utility is reachable only through `incin::test_utils`.
10. Compile-fail: graph/compiler internals are absent from the default prelude.

Do not update `.stderr` files blindly. Read each diagnostic and confirm it fails for the intended reason.

## 6. Generate evidence

Create these files under `audit-evidence/API-001/`:

- `api-after-incin.txt`
- `api-after-incin-core.txt`
- `api-after-incin-backends.txt`
- `api-after-incin-data.txt`
- `api-diff-reviewed.md`
- `commands.log`
- `doctests.log`
- `compile-tests.log`
- `semver-checks.log`
- `feature-matrix.log`
- `toolchain.txt`
- `git-status.txt`

Record exact commands, exit codes, tool versions, feature sets, and commit hash.

Run at least:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace --no-default-features
cargo test --doc --workspace --all-features
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps
cargo test -p incin-core --test compile_tests
cargo public-api --simplified --manifest-path crates/incin/Cargo.toml
cargo public-api --simplified --manifest-path crates/incin-core/Cargo.toml
cargo public-api --simplified --manifest-path crates/incin-backends/Cargo.toml
cargo public-api --simplified --manifest-path crates/incin-data/Cargo.toml
cargo semver-checks check-release -p incin
```

Also test the relevant feature combinations, including:

```bash
cargo check -p incin --no-default-features
cargo check -p incin --no-default-features --features std
cargo check -p incin --features cpu
cargo check -p incin --features compiled
cargo check -p incin --features test-utils
cargo check -p incin --features autotune
```

Adjust combinations only when Cargo feature dependencies require it, and record the exact resolved command.

If `cargo-public-api`, `cargo-semver-checks`, a Rust target, or any other required tool is missing, install it only if the environment policy permits. Otherwise archive the error, mark `API-001` blocked, and stop.

## 7. API-001 completion gate

`API-001` is done only when:

- no public facade or prelude uses a cross-crate wildcard re-export;
- the stable root and prelude contain no backend-authoring, compiler, tuning-internal, graph-IR, storage, or test-only names;
- the compiled, tuning, backend-authoring, and test tiers are correctly gated and documented;
- all required fixtures pass;
- all command evidence exists;
- the public API diff has been manually classified;
- every acceptance item in `summary.md` points to a specific passing log or snapshot.

Commit only this task with a subject beginning `API-001:`. Do not squash it into the previous commit.

---

# Task 2: API-002 — Safe public construction and opaque invariants

Begin only after `API-001` is proven done.

Create `audit-evidence/API-002/summary.md` before editing.

## Required semantic rule

Publicly usable does not mean publicly forgeable.

A user must be able to use public marker/configuration types through safe constructors, builders, `Default`, or validated parsing. They must not be able to bypass invariants through public tuple fields, public mutable fields, derived deserialization of proof types, or arbitrary backend-storage construction.

## `Dyn`

Replace:

```rust
pub struct Dyn(pub ());
```

with an opaque representation. Preserve these capabilities:

- `Dyn` remains publicly nameable in generic positions;
- `Dyn: Default` when a value-level marker is useful and carries no unchecked data;
- users can write `Dyn::default()` or another explicit safe constructor if a value is required;
- all existing dynamic tensor examples remain ergonomic.

Add:

- compile-pass: `Tensor::<Dyn, ...>` works;
- compile-pass: `let _: Dyn = Dyn::default();` works if value construction is part of the public contract;
- compile-fail: `Dyn(())` is impossible externally.

## Checked arithmetic/proof wrappers

Make fields private for at least:

- `CheckedNumel`
- `CheckedByteLen`
- `BufferSlot`

Provide only:

- checked public constructors returning `Result`/`Option` where users legitimately need construction;
- `get()`/inspection accessors;
- crate-private constructors that are named to show validation has already happened;
- custom deserialization that revalidates raw values.

No derived `Deserialize` may bypass validation.

## Tuning and planner types

Review at least:

- `LaunchCandidate`
- `TuningCandidate`
- both `ShapeBucket` types
- cache keys/records and environment fingerprints
- compiled planner identifiers and slots

For each type, classify it in `summary.md` as:

- public end-user configuration;
- public read-only inspection;
- backend-authoring API;
- internal implementation detail.

Then enforce that classification with visibility, private fields, constructors/builders, and accessors.

A user-facing tuning type must be constructible without touching private backend/kernel internals. An internal candidate must not be exposed merely to make another public signature compile.

## Backend storage and variables

Review CPU, CUDA, WGPU, Metal, dispatch, and external backend variable/storage types. Ensure external code cannot create a tensor variable with arbitrary raw storage or inconsistent metadata.

Keep fields private. Expose only safe transfer, readback, debug, or backend-authoring constructors that validate dtype, shape, device, byte length, and ownership.

## API-002 tests

Add compile-fail cases for external tuple/field construction and runtime/serialization tests for validation. Include malformed, overflow, zero/empty, and mismatched metadata cases where applicable.

Run and archive the API-002-specific tests plus the global API-001 gates. Commit only when every criterion is proven.

---

# Task 3: API-003 — Remove test and prototype internals from production

Begin only after `API-002` is proven done.

Create `audit-evidence/API-003/summary.md` before editing.

Required work:

1. Gate `DummyBackend` and all dummy aliases behind `test` or `test-utils` in every crate and facade.
2. Prove with a consumer fixture that normal dependencies cannot import or name it.
3. Move `PanicTestPanel` and equivalent intentionally panicking diagnostic/test UI behind test-only gates.
4. Review compiled pass representations. Until `CompiledProgram::run` with numerical parity exists, do not advertise prototype representations as stable execution APIs. Keep only the explicitly approved preview inspection/configuration surface from API-001.
5. Remove rustdoc claims that fake numerical behavior constitutes a production backend.
6. Generate default-feature and all-feature rustdoc snapshots proving that test-only items are absent from production docs.

Run and archive all task-specific and global gates. Commit with a subject beginning `API-003:`.

Stop after reporting API-003. Do not proceed to `API-004` or any later phase in this run.

---

# Non-negotiable operating rules

- Do not trust checked boxes, commit messages, changelog claims, or previous generated reports without source and command evidence.
- Do not mark a task complete because `cargo check` passes.
- Do not modify unrelated backend numerical behavior while fixing exports.
- Do not add wildcard re-exports elsewhere to hide a removed wildcard.
- Do not expose an internal type solely because an existing public signature leaks it; redesign the public signature.
- Do not weaken a compile-fail test or regenerate `.stderr` snapshots without verifying the intended failure.
- Do not use `todo!`, `unimplemented!`, fake defaults, empty success values, fabricated timing, or silent CPU fallback.
- Do not use public fields as a shortcut around constructor design.
- Do not use `unwrap`, `expect`, or `panic!` on user-controlled construction, parsing, deserialization, storage, artifact, or tuning inputs.
- Do not claim a command ran unless its complete output and exit code are archived.
- Keep the working tree clean at each task boundary.
- Update `graphify-out` after source changes when the tool is available, as required by `AGENTS.md`.

# Required response after each task

Return exactly this structure:

```text
TASK: <API-001 | API-002 | API-003> — <name>
STATUS: DONE | PARTIAL | BLOCKED
BASE COMMIT: <hash>
RESULT COMMIT: <hash or NONE>
SOURCE BEHAVIOR BEFORE:
- <file:symbol and observed behavior>
CHANGES:
- <production/API change>
PUBLIC PATHS ADDED:
- <path, feature, tier>
PUBLIC PATHS REMOVED OR MOVED:
- <old path> -> <new path>
TESTS ADDED:
- <test and what it proves>
COMMANDS:
- <command> => PASS/FAIL/BLOCKED; <evidence path>
ACCEPTANCE:
- [x]/[ ] <criterion> => <specific evidence path>
UNVERIFIED OR BLOCKED:
- <specific item, or NONE>
REMAINING RISKS:
- <specific risk, or NONE>
NEXT ELIGIBLE TASK:
- <task ID or NONE>
```

A `PARTIAL` or `BLOCKED` task must keep all unproven criteria unchecked and must not begin the next task.

# First action

Do not edit source immediately.

First:

1. verify the current commit and clean working tree;
2. run `graphify query` for the public export paths if available;
3. inspect the current `API-001` evidence directory;
4. update `audit-evidence/API-001/summary.md` with the post-commit truth;
5. generate a fresh source-level export inventory;
6. state whether API-001 is `INCOMPLETE`, with exact reasons;
7. only then begin the narrow facade corrections described above.
