# Documentation index

Four kinds of document live here, and the difference matters when they
disagree.

**Guide.** Narrative, not binding. Explains how the pieces fit and the
idiomatic way to use each one. Rust examples are checked where they are
included in doctests or executable fixtures; prose snippets are not promised
to be automatically executed. Read it first if you are new to the codebase.

| Document | Covers |
|---|---|
| [GUIDE.md](GUIDE.md) | the crate map, the type-level shape system, tensor creation, the operation surface, the canonical execution architecture, the target API, backend authoring, autograd, modules, errors, feature flags, and the idioms the rest of this tree assumes. Concept-oriented: "how does the shape-proof system work" |
| [book/src/](book/src/SUMMARY.md) | the full user-facing book. The chapter count and hierarchy are checked from `SUMMARY.md`; `mdbook build docs/book` renders the source, `python3 docs/book/build_site.py` builds the chaptered Pages site, `python3 tools/check-book-site.py` validates its static shell, `python3 tools/test-book-site.py` exercises routing and theme behavior in Chromium, and `python3 docs/book/make_single_page.py` builds the separate self-contained offline artifact. |
| [security/unsafe-ledger.md](security/unsafe-ledger.md) | production unsafe-code inventory and the checker that keeps new unsafe-bearing files visible during review. |
| [../SECURITY.md](../SECURITY.md) | vulnerability reporting guidance and the security review boundary. |
| [public-api/hidden-items.md](public-api/hidden-items.md) | reviewed inventory of hidden exports used for macro, proof, backend, and compatibility plumbing. |
| [architecture/](architecture/) | validated repository architecture map covering runtime layering, extension boundaries, and documentation delivery. |

Release packaging is also versioned: the release workflow pins mdBook,
Node.js, and the VS Code packaging tool, and uploads the book, VS Code,
Neovim, IntelliJ-platform, `incin-lsp`, and `cargo-incin` artifacts together.

Rust documentation is a CI contract as well as a generated output: `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps` must pass before changes are considered complete. This catches broken intra-doc links and rustdoc warnings across the workspace.

**Generated.** Written by a test from the Rust source and re-checked on every
run. If one of these is wrong, the source is wrong. Never edit them by hand.

| Document | Generated from | Regenerate with |
|---|---|---|
| [capabilities.md](capabilities.md) | the backend capability registrations | `INCIN_DOCS=overwrite cargo test -p incin-backends --test generated_docs` |
| [OPERATION_SEMANTICS.md](OPERATION_SEMANTICS.md) | `incin_core::exec::OPERATION_CATALOG` | `INCIN_DOCS=overwrite cargo test -p incin-core --test generated_operation_semantics` |
| [operation-coverage.md](operation-coverage.md) | `incin_core::exec::OPERATION_CATALOG` execution-site counts | `INCIN_DOCS=overwrite cargo test -p incin-core --test generated_operation_coverage` |
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
| [plan/UX-ARCHITECTURE-HANDOFF.md](plan/UX-ARCHITECTURE-HANDOFF.md) | **Historical/non-normative** UX and dispatch audit; current allocation and execution guidance lives in `GUIDE.md`, `FROZEN_FOUNDATIONS.md`, and `HANDOFF.md` |
| [plan/remediation/](plan/remediation/README.md) | the long-form audits and the active FND-000..005 brief |
| [growth/](growth/README.md) | subsystem plans deferred until the foundation sequence completes |

Per-task command logs, before/after API snapshots and test output are under
`audit-evidence/`, one directory per task. A claim in any document above that
is not reproducible from a log there should be treated as unproven.
