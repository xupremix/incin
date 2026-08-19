# incin conventions

A house style for how this codebase should read and how new work should be
organized. Nothing here is a correctness rule — incin's 101-task execution
ledger (see `PROPOSALS.md`) already covers that. This is about how fast a
human can find, read, and change something.

## Problem

incin is 164,736 lines of Rust across 10 crates. It works and is heavily
tested, but it is dense to navigate: individual files run long
(`crates/incin-core/src/exec/catalog.rs` at 6097 lines,
`crates/incin-backends/src/cpu/canonical.rs` at 4235,
`crates/incin-core/src/tensor/ops/manipulation.rs` at 3464,
`crates/incin-backends/src/dist/nccl.rs` at 2955), and doc comments
frequently read as audit-trail paragraphs — bench deltas, "a second Miri run
tripped X instead of Y," deviation justifications — rather than short API
summaries.

## Goal

A codebase that reads the way a well-known, idiomatic Rust library reads,
with the ergonomic, discoverable feel of PyTorch's API surface, built from
what incin already has: 101 completed, tested capabilities, presented with
short docs, real runnable examples, and a book that doesn't restate what the
rustdoc already shows.

## Non-goals

- **Crate boundaries.** The 10 crates stay exactly as they are.
- **Behavior.** No crate is required to preserve its current public API, but
  none is required to change behavior either. This is about organization
  and presentation, not semantics.
- **Enforcement tooling.** No new CI gate, lint, or `xtask` budget check
  accompanies this document. incin already gates aggressively; add
  enforcement once the convention has been applied somewhere real, not
  before.

## File organization

Not a hard line-count ceiling. A generated-feeling file — a genuine
declarative table, or the output shape of a macro like
`impl_typed_kernel!(f32, f64, f16, bf16, u8, i8, u32, i32, i64)` — can be
long and still be one clear thing. The split trigger is **responsibility
count**, not length: split a file when it visibly mixes more than one
concern, not when it crosses a number.

This is not just theoretical: `exec/catalog.rs` (6097 lines) looks at a
glance like it could be "one big catalog," but it is not one declarative
table. In its first 1100 lines alone it carries classification enums
(`SemanticProfile`, `BroadcastingRule`, `ExecutionSite`, ...), a
coverage-reporting pair (`operation_coverage`/`operation_coverage_document`),
the `OperationCatalogEntry` row type and its classification logic, tensor
metadata types (`LogicalTensorMeta`, `CreationPayload`), an open-operation
identity (`OperationKey`), and a `Descriptor<O>` typed-wrapper module — at
least six distinct concerns sharing one file. When this file's turn comes up
in a future crate-specific restructuring pass, treat it as a primary split
target, not an exception the length rule was designed to protect.

Where a split happens, it favors a directory with `mod.rs` as the public
surface and concerns broken into named siblings (`kernels.rs`, `tests.rs`,
and so on), not one file's contents spread flat into many same-level files
with no grouping.
