# Workspace conventions: a house style for human editing

## Status

Design approved. First spec in a multi-spec restructuring effort tracked on
the `design` branch. This spec produces one deliverable — a conventions
document — and touches no crate's actual source. Per-crate application is
separate, follow-up work.

## Problem

incin is 164,736 lines of Rust across 10 crates, all 101 tasks in its
execution ledger marked complete. The code works and is heavily tested, but
it is dense to read and edit: individual files run long (`exec/catalog.rs` at
6097 lines, `cpu/canonical.rs` at 4235, `tensor/ops/manipulation.rs` at 3464,
`dist/nccl.rs` at 2955), and doc comments frequently read as audit-trail
paragraphs — bench deltas, "a second Miri run tripped X instead of Y,"
deviation justifications — rather than short API summaries. Nothing here is
a correctness problem; it is a navigation and editing-speed problem.

## Goal

A codebase that reads the way a well-known, idiomatic Rust library reads,
with the ergonomic, discoverable feel of PyTorch's API surface, built from
what incin already has: 101 completed, tested capabilities, presented with
short docs, real runnable examples, and a book that doesn't restate what the
rustdoc already shows.

## Non-goals

- **Crate boundaries.** The 10 crates stay exactly as they are. Nothing here
  splits, merges, or moves a crate.
- **Behavior.** No crate is required to preserve its current public API —
  nothing beyond the `0.0.0` placeholder versions is published, so breaking
  changes are free — but no crate is required to change behavior either.
  This spec is about organization and presentation, not semantics.
- **Enforcement tooling.** No new CI gate, lint, or `xtask` budget check.
  incin already gates aggressively (audit-shapes, soundness.sh,
  feature-matrix, budgets); adding another gate before a single file has
  been split under this convention would be enforcing a rule nobody has
  used yet. Revisit once the convention has been applied somewhere real.
- **Mass rewrite.** This spec does not touch any crate's files. Applying the
  convention is scoped per crate (or per related group of crates) in
  follow-up specs.

## Conventions

### File organization

Not a hard line-count ceiling. A generated-feeling file — a genuine
declarative table, or the output shape of a macro like
`impl_typed_kernel!(f32, f64, f16, bf16, u8, i8, u32, i32, i64)` — can be
long and still be one clear thing. The split trigger is **responsibility
count**, not length: a file is split when it visibly mixes more than one
concern.

Piloted against `exec/catalog.rs` (6097 lines) to check the heuristic isn't
just theoretical: it is not one declarative table. In its first 1100 lines
alone it carries classification enums (`SemanticProfile`,
`BroadcastingRule`, `ExecutionSite`, ...), a coverage-reporting pair
(`operation_coverage`/`operation_coverage_document`), the
`OperationCatalogEntry` row type and its classification logic, tensor
metadata types (`LogicalTensorMeta`, `CreationPayload`), an open-operation
identity (`OperationKey`), and a `Descriptor<O>` typed-wrapper module — at
least six distinct concerns sharing one file. This is exactly the case the
heuristic is meant to catch, and the split follow-up for `incin-core` should
treat it as a primary target rather than an exception.

Where a split happens, it favors a directory with `mod.rs` (or the crate's
established equivalent) as the public surface, with concerns broken into
named siblings (`kernels.rs`, `tests.rs`, and so on) rather than one file's
contents spread flat into many same-level files with no grouping.

### Doc comments

Every public item carries:

1. A one-line summary.
2. An optional 2-4 sentence "why," only when the decision is genuinely
   non-obvious from the signature and name alone.
3. A `# Examples` block with a real, runnable doctest.

Evidence-log material — bench deltas, Miri-flakiness investigation notes,
"we tried X, it failed because Y" narratives, deviation justifications —
moves to where incin already has a proper home for it: `CHANGELOG.md` for
anything user-facing, `docs/plan/tasks/<ID>.md` for the historical
task-completion narrative (the ledger's existing pattern for exactly this
kind of record). It does not disappear; it moves to where a reader looking
for "what does this do" doesn't have to wade through it, and a reader
looking for "why was this built this way" still knows where to find it.

`SAFETY:` comments on `unsafe` blocks are unaffected. That is a separate,
load-bearing convention tied to `docs/security/unsafe-ledger.md` and
`tools/check-unsafe-ledger.py`, not the audit-trail problem this spec
addresses.

### Examples and tests

"Document and test everything" is a presentation change, not new test
authorship: every public function and type gets a doctest under
`# Examples` that exercises real, already-proven behavior — the 101
completed ledger tasks already built and verified it; this makes that
behavior visible where a reader looks for it. `cargo test --doc` is already
a CI gate (the Documentation Build job), so a doctest that goes stale fails
the same way any other test failure does.

Where a doctest doesn't fit the crate's shape — `incin-lsp`'s binary,
`incin-viz`'s TUI — the example is a file under `crates/<crate>/examples/`
or a `docs/book/` walkthrough instead.

### Book alignment

`docs/book/` chapters link to the real rustdoc examples rather than
restating them inline. One example, one source of truth; a book chapter and
a rustdoc example describing the same behavior in independently-maintained
prose is exactly the kind of drift this effort is meant to reduce, not add.

## Deliverable

`docs/CONVENTIONS.md`, alongside the existing `docs/API_DESIGN.md` and
`docs/GUIDE.md`, containing the four conventions above in a form a
contributor can act on directly: what triggers a split, the doc-comment
shape with a worked before/after example, the doctest expectation, and the
book-linking rule.

## Sequencing after this spec

This spec's deliverable is the convention only. Follow-up specs apply it,
one per crate or small related group (for example: `incin-core` +
`incin-macros` together, given how tightly coupled they are; the four
backend targets in `incin-backends` as their own spec; the smaller support
crates — `incin-data`, `incin-diagnostics`, `incin-telemetry`, `incin-lsp`,
`incin-viz`, `incin-viz-plugin-api` — grouped by size). Each follow-up spec
gets its own design-doc → plan → implementation cycle rather than being
folded into this one.
