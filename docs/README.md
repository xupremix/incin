# Documentation index

Three kinds of document live here, and the difference matters when they
disagree.

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
| [API_DESIGN.md](API_DESIGN.md) | the stable facade, the tiers, and what may appear in each |
| [ERROR_CONTRACT.md](ERROR_CONTRACT.md) | the typed failure categories and the panic policy |
| [INVARIANT_TYPES.md](INVARIANT_TYPES.md) | every public type that carries an invariant, and how it is constructed |
| [MIGRATION.md](MIGRATION.md) | the public paths that moved, and where they moved to |

**Status and plans.** True as of the commit that last touched them.

| Document | Covers |
|---|---|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | what is implemented, what is verified, and what is only structural |
| [plan/roadmap.md](plan/roadmap.md) | the task ledger's shape |
| [plan/remediation/](plan/remediation/README.md) | the long-form audits and the active FND-000..005 brief |
| [growth/](growth/README.md) | subsystem plans deferred until the foundation sequence completes |

Per-task command logs, before/after API snapshots and test output are under
`audit-evidence/`, one directory per task. A claim in any document above that
is not reproducible from a log there should be treated as unproven.
