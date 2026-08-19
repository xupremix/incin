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
