# Workspace Conventions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `docs/CONVENTIONS.md`, the house-style document defining how incin's file organization, doc comments, examples/tests, and book docs should read going forward.

**Architecture:** One markdown file, built section by section across four tasks, each committed independently. Two sections (file organization, doc comments) ground their guidance in real, currently-true facts about the codebase (exact line counts, a real doc comment rewritten in full) rather than abstract advice; each such fact gets a verification step that checks it against the live source before that task's commit, so the document can't ship claiming something about the codebase that isn't actually true. Every "append a section" step is a literal `cat >>` heredoc rather than prose to copy by hand, so there is no ambiguity about exact content or markdown-fence nesting.

**Tech Stack:** Markdown; `rustc`/`cargo test --doc` for verifying the doc-comment example actually compiles against `incin-core`.

**Spec:** `docs/superpowers/specs/2026-08-19-workspace-conventions-design.md`

## Global Constraints

- No file outside `docs/CONVENTIONS.md` is modified permanently. Task 3 makes a temporary edit to `crates/incin-core/src/exec/catalog.rs` to verify a doctest compiles, then reverts it with `git checkout --` before committing - that revert is itself a required step, not optional cleanup.
- Crate boundaries are not discussed as changeable anywhere in the document (per spec non-goals).
- No new CI gate, lint, or `xtask` check is added by this plan (per spec non-goals - approach B).
- Every concrete number or code example the document states must be independently verified against the live source in the task that adds it, not copied from the spec without re-checking (the spec was written 2026-08-19; verify these are still current at implementation time).

---

### Task 1: Scaffold the document - title, problem, goal, non-goals

**Files:**
- Create: `docs/CONVENTIONS.md`

**Interfaces:**
- Produces: the file itself, with top-level headers `## Problem`, `## Goal`, `## Non-goals` that Tasks 2-4 append sibling `## ` sections after.

- [ ] **Step 1: Verify the cited line counts are current before writing them down**

Run:
```bash
wc -l crates/incin-core/src/exec/catalog.rs \
      crates/incin-backends/src/cpu/canonical.rs \
      crates/incin-core/src/tensor/ops/manipulation.rs \
      crates/incin-backends/src/dist/nccl.rs
git ls-files 'crates/**/*.rs' | xargs cat | wc -l
```
Expected (as of 2026-08-19): `6097`, `4235`, `3464`, `2955`, and `164736`
respectively. If any number has drifted, use the current number in Step 2
below instead of the one shown there - do not write down a stale claim.

- [ ] **Step 2: Write the scaffold**

```bash
cat > docs/CONVENTIONS.md << 'MDEOF'
# incin conventions

A house style for how this codebase should read and how new work should be
organized. Nothing here is a correctness rule - incin's 101-task execution
ledger (see `PROPOSALS.md`) already covers that. This is about how fast a
human can find, read, and change something.

## Problem

incin is 164,736 lines of Rust across 10 crates. It works and is heavily
tested, but it is dense to navigate: individual files run long
(`crates/incin-core/src/exec/catalog.rs` at 6097 lines,
`crates/incin-backends/src/cpu/canonical.rs` at 4235,
`crates/incin-core/src/tensor/ops/manipulation.rs` at 3464,
`crates/incin-backends/src/dist/nccl.rs` at 2955), and doc comments
frequently read as audit-trail paragraphs - bench deltas, "a second Miri run
tripped X instead of Y," deviation justifications - rather than short API
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
MDEOF
```

If Step 1 found a drifted number, edit the file just written so the
`## Problem` section states the current numbers before moving on.

- [ ] **Step 3: Verify structure**

Run: `grep -c '^## ' docs/CONVENTIONS.md`
Expected: `3` (Problem, Goal, Non-goals).

- [ ] **Step 4: Commit**

```bash
git add docs/CONVENTIONS.md
git commit -m "docs: scaffold conventions doc with problem and non-goals"
```

---

### Task 2: File organization section

**Files:**
- Modify: `docs/CONVENTIONS.md` (append after `## Non-goals`)

**Interfaces:**
- Consumes: nothing from Task 1 beyond the file existing.
- Produces: a `## File organization` section that Task 5's coverage check
  looks for by name.

- [ ] **Step 1: Verify the catalog.rs pilot claim before writing it down**

This section cites `exec/catalog.rs` as proof the split heuristic (below)
isn't just theoretical. Confirm the claim is still true:

```bash
grep -n "^pub enum SemanticProfile\|^pub enum BroadcastingRule\|^pub enum ExecutionSite\|^pub fn operation_coverage\|^pub struct OperationCatalogEntry\|^pub struct LogicalTensorMeta\|^pub struct CreationPayload\|^pub struct OperationKey\|^mod private" crates/incin-core/src/exec/catalog.rs
```
Expected: all of these still appear (line numbers may have shifted since
2026-08-19; that's fine, only presence matters). If one is gone, drop the
corresponding clause from Step 2's heredoc text rather than asserting
something no longer true.

- [ ] **Step 2: Append the section**

```bash
cat >> docs/CONVENTIONS.md << 'MDEOF'

## File organization

Not a hard line-count ceiling. A generated-feeling file - a genuine
declarative table, or the output shape of a macro like
`impl_typed_kernel!(f32, f64, f16, bf16, u8, i8, u32, i32, i64)` - can be
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
identity (`OperationKey`), and a `Descriptor<O>` typed-wrapper module - at
least six distinct concerns sharing one file. When this file's turn comes up
in a future crate-specific restructuring pass, treat it as a primary split
target, not an exception the length rule was designed to protect.

Where a split happens, it favors a directory with `mod.rs` as the public
surface and concerns broken into named siblings (`kernels.rs`, `tests.rs`,
and so on), not one file's contents spread flat into many same-level files
with no grouping.
MDEOF
```

- [ ] **Step 3: Verify structure**

Run: `grep -c '^## ' docs/CONVENTIONS.md`
Expected: `4`.

- [ ] **Step 4: Commit**

```bash
git add docs/CONVENTIONS.md
git commit -m "docs: add file organization convention"
```

---

### Task 3: Doc-comment convention, with a real before/after

**Files:**
- Modify: `docs/CONVENTIONS.md` (append after `## File organization`)
- Temporarily modify, then revert: `crates/incin-core/src/exec/catalog.rs`

**Interfaces:**
- Consumes: nothing from prior tasks beyond the file existing.
- Produces: a `## Doc comments` section containing a real "before" (the
  current `ExecutionSite` doc comment, quoted verbatim) and a real "after"
  (a rewritten version, with its example doctest proven to compile).

- [ ] **Step 1: Capture the current "before" verbatim**

```bash
sed -n '137,187p' crates/incin-core/src/exec/catalog.rs
```
Expected: the `ExecutionSite` enum and its doc comment, matching the text
used in Step 2 below. If it differs (it may have been edited since
2026-08-19), use what this command actually prints as the "before" in
Step 5's heredoc instead - never quote a comment that no longer exists in
the source.

- [ ] **Step 2: Apply the "after" temporarily to prove it compiles**

The rewrite moves the migration-counting history (the "used to be one
number... sixteen of them cannot be an `Execute<O>` implementation" story)
out of the doc comment - it's implementation history, not something a
caller needs to use `ExecutionSite`. It keeps the one fact a caller actually
needs (`is_backend_executable` is the predicate) and adds a real doctest.

```bash
python3 - << 'PYEOF'
path = "crates/incin-core/src/exec/catalog.rs"
old = '''/// Where an operation's result is produced, and therefore what shape of
/// contract can carry it.
///
/// This says nothing about whether an operation is implemented. It says what
/// kind of implementation is even possible, and it exists because the CPU
/// migration's remainder used to be one number. One number implies every
/// unmigrated operation is the same kind of missing work. It is not: most are a
/// kernel nobody has routed yet, but sixteen of them cannot be an
/// `Execute<O>` implementation as that trait is currently written,
/// so counting them beside a missing kernel describes a task that does not
/// exist and hides one that does.
///
/// [`ExecutionSite::is_backend_executable`] is the predicate that separates the
/// two. Every variant states its own reason rather than deferring to prose.'''
new = '''/// Where an operation's result is produced, and therefore what shape of
/// execution contract can carry it.
///
/// This is about what kind of implementation is possible, not whether one
/// exists yet -- [`ExecutionSite::is_backend_executable`] is the predicate
/// that tells them apart.
///
/// ```
/// use incin_core::exec::catalog::ExecutionSite;
///
/// assert!(ExecutionSite::Kernel.is_backend_executable());
/// assert!(!ExecutionSite::Mutation.is_backend_executable());
/// ```'''
src = open(path).read()
if old not in src:
    raise SystemExit("before-text not found verbatim -- use Step 1's actual output instead")
open(path, "w").write(src.replace(old, new, 1))
print("patched")
PYEOF
```
Expected output: `patched`. If it instead raises `before-text not found
verbatim`, the comment changed since this plan was written - copy Step 1's
actual current text into `old` above (adjusting `new` to match the same
opening/closing sentences) and re-run.

- [ ] **Step 3: Run the doctest to prove the "after" example actually compiles**

Run: `cargo test -p incin-core --doc exec::catalog::ExecutionSite`
Expected: `test result: ok. 1 passed`. This exact command was verified
working during plan review (2026-08-19). If it fails, fix the example in
Step 2's `new` text (not the prose in Step 5 below) and re-run Steps 2-3
until it passes - the document must not ship a doctest that doesn't
compile.

- [ ] **Step 4: Revert the temporary source edit**

```bash
git checkout -- crates/incin-core/src/exec/catalog.rs
git status --porcelain crates/incin-core/src/exec/catalog.rs
```
Expected: the second command prints nothing (file is clean - the temporary
edit is gone; only `docs/CONVENTIONS.md` will be committed from this task).

- [ ] **Step 5: Append the section**

```bash
cat >> docs/CONVENTIONS.md << 'MDEOF'

## Doc comments

Every public item carries:

1. A one-line summary.
2. An optional 2-4 sentence "why," only when the decision is genuinely
   non-obvious from the signature and name alone.
3. A runnable doctest, wherever the item's usage can be shown in a handful
   of lines.

Evidence-log material - bench deltas, Miri-flakiness investigation notes,
"we tried X, it failed because Y" narratives, deviation justifications - 
moves to where incin already has a proper home for it: `CHANGELOG.md` for
anything user-facing, `docs/plan/tasks/<ID>.md` for the historical
task-completion narrative (the ledger's existing pattern for exactly this
kind of record). It does not disappear; it moves to where a reader looking
for "what does this do" doesn't have to wade through it, while a reader
looking for "why was this built this way" still knows where to find it.

`SAFETY:` comments on `unsafe` blocks are unaffected - that is a separate,
load-bearing convention tied to `docs/security/unsafe-ledger.md`, not the
audit-trail problem this section addresses.

**Before**, from `crates/incin-core/src/exec/catalog.rs` - six sentences of
migration history before the enum a reader came to look up:

```rust
/// Where an operation's result is produced, and therefore what shape of
/// contract can carry it.
///
/// This says nothing about whether an operation is implemented. It says what
/// kind of implementation is even possible, and it exists because the CPU
/// migration's remainder used to be one number. One number implies every
/// unmigrated operation is the same kind of missing work. It is not: most are a
/// kernel nobody has routed yet, but sixteen of them cannot be an
/// `Execute<O>` implementation as that trait is currently written,
/// so counting them beside a missing kernel describes a task that does not
/// exist and hides one that does.
///
/// [`ExecutionSite::is_backend_executable`] is the predicate that separates the
/// two. Every variant states its own reason rather than deferring to prose.
```

**After** - the fact a caller needs, plus a doctest proving the predicate
does what it says (verified to compile with `cargo test -p incin-core --doc
exec::catalog::ExecutionSite` before this section was written):

```rust
/// Where an operation's result is produced, and therefore what shape of
/// execution contract can carry it.
///
/// This is about what kind of implementation is possible, not whether one
/// exists yet -- [`ExecutionSite::is_backend_executable`] is the predicate
/// that tells them apart.
///
/// ```
/// use incin_core::exec::catalog::ExecutionSite;
///
/// assert!(ExecutionSite::Kernel.is_backend_executable());
/// assert!(!ExecutionSite::Mutation.is_backend_executable());
/// ```
```

The migration-history sentences from the "before" aren't lost - they belong
in `docs/plan/tasks/EXE-008.md` (or wherever the migration task that
motivated them lives), as a record of why the type exists, not as the first
thing a caller reads on the way to calling it.
MDEOF
```

- [ ] **Step 6: Verify structure**

Run: `grep -c '^## ' docs/CONVENTIONS.md`
Expected: `5`.

- [ ] **Step 7: Confirm the source revert held, then commit**

```bash
git status --porcelain crates/incin-core/src/exec/catalog.rs
```
Expected: no output. Then:

```bash
git add docs/CONVENTIONS.md
git commit -m "docs: add doc-comment convention with a worked example"
```

---

### Task 4: Examples/tests convention and book alignment

**Files:**
- Modify: `docs/CONVENTIONS.md` (append after `## Doc comments`)

**Interfaces:**
- Consumes: nothing from prior tasks beyond the file existing.
- Produces: `## Examples and tests` and `## Book alignment` sections.

- [ ] **Step 1: Verify the cited paths are real**

```bash
test -d crates/incin-lsp && echo "incin-lsp: ok"
test -d crates/incin-viz && echo "incin-viz: ok"
test -d docs/book/src && echo "docs/book/src: ok"
```
Expected: all three print `ok`.

- [ ] **Step 2: Append the sections**

```bash
cat >> docs/CONVENTIONS.md << 'MDEOF'

## Examples and tests

"Document and test everything" is a presentation change, not new test
authorship: every public function and type gets a doctest that exercises
real, already-proven behavior - the 101 completed ledger tasks already
built and verified it; this makes that behavior visible where a reader
looks for it. `cargo test --doc` is already a CI gate, so a doctest that
goes stale fails the same way any other test failure does.

Where a doctest doesn't fit the crate's shape - `incin-lsp`'s binary,
`incin-viz`'s TUI - the example is a file under `crates/<crate>/examples/`
or a `docs/book/` walkthrough instead.

## Book alignment

`docs/book/src/` chapters link to the real rustdoc examples rather than
restating them inline. One example, one source of truth: a book chapter and
a rustdoc example describing the same behavior in independently-maintained
prose is exactly the drift this document exists to reduce, not add.
MDEOF
```

- [ ] **Step 3: Verify structure**

Run: `grep -c '^## ' docs/CONVENTIONS.md`
Expected: `7`.

- [ ] **Step 4: Commit**

```bash
git add docs/CONVENTIONS.md
git commit -m "docs: add examples/tests and book alignment conventions"
```

---

### Task 5: Final coverage check against the spec

**Files:**
- Modify: `docs/CONVENTIONS.md` (only if a gap is found)

**Interfaces:**
- Consumes: the complete document from Tasks 1-4.
- Produces: nothing new if Step 1 finds no gaps; otherwise a fix commit.

- [ ] **Step 1: Check every spec deliverable requirement has a home**

The spec's Deliverable section requires: "what triggers a split, the
doc-comment shape with a worked before/after example, the doctest
expectation, and the book-linking rule."

```bash
grep -n "responsibility count" docs/CONVENTIONS.md
grep -n "^\*\*Before\*\*" docs/CONVENTIONS.md
grep -n "^\*\*After\*\*" docs/CONVENTIONS.md
grep -n "doctest" docs/CONVENTIONS.md
grep -n "docs/book/src/ chapters link" docs/CONVENTIONS.md
```

Expected: every grep returns at least one match. If any returns nothing,
add the missing content to the relevant section before continuing.

- [ ] **Step 2: Confirm the doc has no placeholder language**

Run: `grep -niE "TBD|TODO|FIXME|XXX|placeholder" docs/CONVENTIONS.md`
Expected: no output.

- [ ] **Step 3: Confirm everything from Tasks 1-4 is committed**

Run: `git status --porcelain`
Expected: no output.

- [ ] **Step 4: If Step 1 found a gap, commit the fix**

```bash
git add docs/CONVENTIONS.md
git commit -m "docs: fill conventions coverage gap"
```

(Skip this step entirely if Step 1 found no gaps - there is nothing to
commit.)
