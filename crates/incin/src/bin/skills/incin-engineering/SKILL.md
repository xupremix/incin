---
name: incin-engineering
description: Use for substantive Incin Rust implementation, refactors, debugging, public API work, core/backend/macro changes, invariant-sensitive changes, performance work, or validation planning. Do not trigger for tiny comments, prose-only questions, or unrelated generic tasks.
---

# Incin engineering workflow

## 1. Orient with the smallest context

If `graphify-out/graph.json` exists, query Graphify before broad source browsing. Prefer `graphify query`, `path`, and `explain` over recursive file reading. Search for a concrete symbol before opening large implementation files.

Several central files are thousands of lines long (catalogs and backend implementations). Never read one end-to-end merely to understand one operation. Read the declaration, relevant expansion/implementation, and its tests.

## 2. Establish authority before editing

Use source and contracts in this order:

1. The relevant Rust declarations/implementations and tests.
2. Binding contract docs when the surface is governed by one: `docs/FROZEN_FOUNDATIONS.md`, `docs/API_DESIGN.md`, `docs/ERROR_CONTRACT.md`, `docs/INVARIANT_TYPES.md`.
3. `docs/README.md` to identify generated docs and their regeneration commands.
4. Status/plan documents as snapshots; verify important claims against current source/tests before relying on them.

Never hand-edit generated evidence (`docs/capabilities.md`, `docs/OPERATION_SEMANTICS.md`, generated audit evidence). Change its authoritative Rust input and regenerate with the documented command.

Public API changes must obey `docs/API_DESIGN.md`; do not restate or invent a parallel visibility policy.

## 3. Choose model tier by risk, not size

Use Luna/`quick_worker` for deterministic local changes whose implementation and verification are obvious.

Use Terra/`worker` for normal substantive Rust work, including multi-file implementation, backend/core changes, refactors, integration, and ordinary debugging.

Use Sol/`architect` only when the correct design is genuinely unresolved and affects a frozen foundation, public API, type-level shape/dtype/device/gradient invariant, execution contract, unsafe/soundness boundary, or other high-blast-radius behavior. If a binding doc already specifies the answer, skip Sol and implement the contract with Terra.

If Terra finds a previously hidden architectural contradiction, stop and escalate the **decision**, not the entire implementation task. After the decision, return to Terra.

## 4. Keep implementation scoped

Prefer the smallest coherent diff. Do not perform unrelated cleanup. Preserve existing layering and avoid introducing new public surface for internal convenience.

For operation/capability work, identify the authoritative catalog/descriptor/capability path first instead of adding parallel one-off mechanisms.

For shape/axis work, preserve the rank-independent stable-Rust architecture and do not reintroduce a fixed framework rank ceiling for convenience.

## 5. Validate progressively

Start with the exact affected test or crate. Expand only if the change crosses a boundary.

Useful patterns (choose only what is relevant):

- Core-only/no_std-sensitive: `cargo check -p incin-core --no-default-features`
- Core tests: `cargo test -p incin-core <filter-or-test>`
- Macro work: `cargo test -p incin-macros <filter-or-test>`
- CPU backend work: `cargo test -p incin-backends --no-default-features --features std,cpu <filter-or-test>`
- Facade work: `cargo test -p incin <filter-or-test>` with only the features the change needs
- Generated operation semantics: `INCIN_DOCS=overwrite cargo test -p incin-core --test generated_operation_semantics`
- Generated capability docs: `INCIN_DOCS=overwrite cargo test -p incin-backends --test generated_docs`
- Shape audit when shape-proof sites change: `tools/audit-shapes.sh --check`
- Ledger/budget/docs/feature-matrix xtasks only when the change touches the governed surface

Do not run `tools/ci-local.sh` as a routine first check. It fans out across formatting, governance, CPU, examples, preview features, BLAS, no_std, WGPU, CUDA checks/runtime when available, docs/doctests, and optionally feature powersets. Reserve it for cross-cutting/final validation or when explicitly requested.

After code changes, run `graphify update .` when Graphify is available.

## 6. Performance changes

Do not infer a performance improvement from code shape alone. Capture the relevant before baseline, make the change, capture after, compare, and update the repository's baseline/budget artifacts only when the measured change is deliberate and the governing docs require it.

## 7. Return a compact handoff

Report:
- files/symbols changed
- contract/decision followed
- targeted checks run and results
- broader checks intentionally not run
- unresolved risk or next action

Do not paste full compiler logs or generated files into the parent context.
