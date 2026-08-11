# Documentation index

Four kinds of document live here, and the difference matters when they
disagree.

**Guide.** Narrative, not binding. Explains how the pieces fit and the
idiomatic way to use each one; every code example in it is checked to compile
and run against the current tree. Read it first if you are new to the
codebase.

| Document | Covers |
|---|---|
| [GUIDE.md](GUIDE.md) | the crate map, the type-level shape system, tensor creation, the operation surface, the canonical execution architecture, the target API, backend authoring, autograd, modules, errors, feature flags, and the idioms the rest of this tree assumes. Concept-oriented: "how does the shape-proof system work" |
| [book/src/](book/src/SUMMARY.md) | the full user-facing book, 23 chapters. Task-oriented for the core path (tensors, autograd, building models, training, data loading, saving/loading, backends) — "how do I train a model" — plus an exhaustive reference section: every macro, every feature flag, the invariant/proof types, backend authoring, and the experimental surfaces. Every code block is checked to compile (and where practical, to run and produce the stated result) against the current tree. An mdBook source tree; read the chapters directly, build with `mdbook build docs/book`, or generate a single self-contained HTML file with `python3 docs/book/make_single_page.py` |

**Generated.** Written by a test from the Rust source and re-checked on every
run. If one of these is wrong, the source is wrong. Never edit them by hand.

| Document | Generated from | Regenerate with |
|---|---|---|
| [capabilities.md](capabilities.md) | the backend capability registrations | `INCIN_DOCS=overwrite cargo test -p incin-backends --test generated_docs` |
| [OPERATION_SEMANTICS.md](OPERATION_SEMANTICS.md) | `incin_core::exec::OPERATION_CATALOG` | `INCIN_DOCS=overwrite cargo test -p incin-core --test generated_operation_semantics` |
| [audit/shape-proof-inventory.md](audit/shape-proof-inventory.md) | the shape-proof sites in the tree | `tools/audit-shapes.sh --check` |

`audit-evidence/FND-005/cpu-migration-status.md` is generated the same way, by
`INCIN_DOCS=overwrite cargo test -p incin-backends --test cpu_migration_status`.

**Contracts.** Hand-written, reviewed, and binding on new code.

| Document | Covers |
|---|---|
| [FROZEN_FOUNDATIONS.md](FROZEN_FOUNDATIONS.md) | the finished, load-bearing parts that should not be rewritten, what is still moving, and the next steps in dependency order. Read this first |
| [API_DESIGN.md](API_DESIGN.md) | the stable facade, the tiers, and what may appear in each |
| [ERROR_CONTRACT.md](ERROR_CONTRACT.md) | the typed failure categories and the panic policy |
| [INVARIANT_TYPES.md](INVARIANT_TYPES.md) | every public type that carries an invariant, and how it is constructed |
| [MIGRATION.md](MIGRATION.md) | the public paths that moved, and where they moved to |

**Status and plans.** True as of the commit that last touched them.

| Document | Covers |
|---|---|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | what is implemented, what is verified, and what is only structural |
| [plan/roadmap.md](plan/roadmap.md) | the task ledger's shape |
| [plan/UX-ARCHITECTURE-HANDOFF.md](plan/UX-ARCHITECTURE-HANDOFF.md) | the user-facing allocation and initialization architecture: what the audit found — including that `exec::dispatch` had no production callers at all — the `target-api` prototype that gave it one, and the remaining steps in dependency order |
| [plan/remediation/](plan/remediation/README.md) | the long-form audits and the active FND-000..005 brief |
| [growth/](growth/README.md) | subsystem plans deferred until the foundation sequence completes |

Per-task command logs, before/after API snapshots and test output are under
`audit-evidence/`, one directory per task. A claim in any document above that
is not reproducible from a log there should be treated as unproven.
