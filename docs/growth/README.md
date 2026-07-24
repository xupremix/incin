# Kindle — Growth & Adoption Plan (Executable Edition)

> **Audience: an implementing agent (human or AI) who has *not* read the whole
> repo.** This directory is a task-by-task execution plan for the features that
> turn Kindle from "a good Rust DL framework" into "the one people switch to and
> make videos about." Each document is self-contained and cites exact file
> paths and line ranges so you can act without holding the whole codebase in
> your head.

This plan is the operational sibling of the root-level `IMPLEMENTATION_PLAN.md`
(backend/kernel work) and `IDEAS.md` (the un-scoped idea pool these documents
graduate from). It follows the **same conventions** as `IMPLEMENTATION_PLAN.md`
§0 — read those below before touching anything.

---

## 0. How to use this plan — read first

1. **Do exactly one workstream at a time, in the sequence in §3.** Each numbered
   doc (`01`…`08`) is an independent, shippable unit with its own acceptance
   criteria. Finish and verify one before starting the next. Do **not**
   interleave — that is how an agent loses the thread.
2. **Every claim here is cited to a file/line.** Code moves. Before acting on a
   claim like "`translate_typenum_text` lives at `cargo-kindle.rs:32`", open the
   cited location and confirm it still says that. **If the code contradicts this
   doc, trust the code** and append a dated correction to the relevant doc (do
   not silently rewrite — mirror the `ROADMAP.md` "follow-up" convention).
3. **Run the verification loop (§2) after every change, before every commit.**
4. **Read the DO-NOT list (§4) before writing any code.**
5. **Do not invent scope.** If something seems missing, add it as a new dated
   task in the relevant doc; don't silently build it.
6. When a task says **"compiles clean, not run"** (CUDA / no-hardware paths),
   never upgrade that to "works"/"verified" — that exact conflation is a
   documented past failure mode (see `IMPLEMENTATION_PLAN.md` §0.3).

---

## 1. The one-sentence strategy (why these features and not others)

Kindle's single unfair advantage is that **tensor shapes are types, not runtime
values.** PyTorch, Candle, Burn, and tinygrad resolve shapes at runtime and
*cannot* copy this; `dfdx` shares the idea but is single-backend and dormant.
Every feature in this plan compounds that advantage or removes a reason a
PyTorch developer bounces in their first five minutes. We are **not** trying to
out-flexibility PyTorch for research — the pitch is **safety, observability, and
deployment**. Keep that framing; it is what makes the marketing credible.

---

## 2. Verification loop (run after every change)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --no-default-features --features kindle-backends/cpu,kindle/cpu -- -D warnings
cargo test  --workspace --all-targets --no-default-features --features kindle-backends/cpu,kindle/cpu
cargo build --examples --workspace --no-default-features --features kindle-backends/cpu,kindle/cpu
```

Touching `wgpu`: additionally `cargo test -p kindle-backends --no-default-features --features wgpu,std --all-targets`.
Touching `cuda` (no hardware here): only `cargo check -p kindle-backends --no-default-features --features cuda,std` — report as **"compiles clean, not run."**

For docs that ship a **non-Rust** component (the LSP server is Rust; the VS Code
extension is TypeScript; the book is mdBook), each doc lists its own extra
build/test commands in its "Verification" section.

---

## 3. Sequencing (dependency graph — build in this order)

```
        ┌─────────────────────────────────────────────┐
        │  00  SHARED: extract `kindle-diagnostics`     │  ← the lynchpin.
        │      crate (typenum <-> decimal formatter)    │     Do this FIRST.
        └───────────────┬───────────────┬───────────────┘
                        │               │
      ┌─────────────────▼──┐     ┌──────▼──────────────────┐
      │ 01 Shape diagnostics│     │ 02 IDE extensions + LSP │
      │  (compile-time msgs)│     │  (VS Code/Neovim/RustR.)│
      └─────────┬───────────┘     └───────────┬─────────────┘
                │                              │
   ┌────────────▼───────────┐                 │
   │ 03 Named dimensions     │                 │  (02 consumes 00's
   │ 04 Compile-time stats   │                 │   formatter; can run
   └────────────┬───────────┘                 │   in parallel with 01)
                │                              │
   ┌────────────▼──────────────────────────────▼────────────┐
   │ 05 Observability + `cargo kindle new`/`watch` scaffolder │
   │ 06 Deployment (single-binary / GGUF loop / WASM)         │
   └────────────┬────────────────────────────────────────────┘
                │
   ┌────────────▼───────────┐   ┌──────────────────────────┐
   │ 07 The Book (mdBook)   │   │ 08 Agent skills          │
   │  (teaches 01–06)       │   │  (encode 01–06 for AIs)  │
   └────────────────────────┘   └──────────────────────────┘
```

**Rule:** `07` (book) and `08` (agent skills) are written *after* the feature
they document lands, or in lockstep with it — never speculatively ahead, or they
document an API that then changes. They may be built incrementally (one chapter /
one skill per shipped feature).

---

## 4. Shared infrastructure — task 00 (the lynchpin, do this first)

**Problem:** the typenum→decimal translation logic
(`translate_typenum_text`, `parse_single_typenum`) currently lives **inside a
binary**, `crates/kindle/src/bin/cargo-kindle.rs:7-95`. A binary's functions
cannot be imported by another crate, so the LSP server (doc `02`) and any future
tooling would have to *copy-paste* it — guaranteeing drift.

**Fix:** extract it into a new tiny library crate `crates/kindle-diagnostics`
that has **no heavy dependencies** (no backends, no `wgpu`, no `prost`) so it
compiles in milliseconds and can be a dependency of the CLI, the LSP, and build
scripts alike.

### Task 00.1 — create `crates/kindle-diagnostics`
- `crates/kindle-diagnostics/Cargo.toml`: `name = "kindle-diagnostics"`,
  `version = "0.2.0"`, edition/license from `[workspace.package]`. Dependencies:
  none required for the core formatter (optionally `serde` behind a `std`
  feature if you later parse cargo JSON here — not needed for v1).
- Add `"crates/kindle-diagnostics"` to root `Cargo.toml` `[workspace] members`.

### Task 00.2 — move the translator, verbatim, into `src/lib.rs`
- Move `parse_single_typenum` and `translate_typenum_text`
  (`cargo-kindle.rs:6-95`) into `crates/kindle-diagnostics/src/lib.rs`, keeping
  them `pub`. Move the two existing unit tests
  (`cargo-kindle.rs:262-288`, incl. the nested-expression regression test) with
  them.
- Add one new public entry point the LSP and CLI both call:
  ```rust
  /// Rewrites a compiler diagnostic string in place, replacing every typenum
  /// expression with its decimal value and returning the rewritten text plus
  /// the (decimal, original) hint pairs discovered. This is the single source
  /// of truth for typenum humanization across the CLI and the LSP.
  pub fn humanize_diagnostic(text: &str) -> Translated { /* wraps translate_typenum_text */ }
  ```

### Task 00.3 — make `cargo-kindle` depend on it
- Add `kindle-diagnostics = { path = "../kindle-diagnostics" }` to
  `crates/kindle/Cargo.toml`.
- In `cargo-kindle.rs`, delete the moved functions and
  `use kindle_diagnostics::{translate_typenum_text, ...}` instead. The binary's
  behavior must not change — the existing two tests (now in the new crate) plus
  a manual `cargo kindle translate "…"` smoke test are the proof.

**Acceptance:** `cargo test -p kindle-diagnostics` passes; `cargo kindle
translate` still humanizes a pasted error identically to before; verification
loop green. Commit as `refactor(diagnostics): extract typenum formatter into
kindle-diagnostics crate`.

**DO-NOT:** do not pull `kindle-core`/`kindle-backends` into
`kindle-diagnostics` — it must stay dependency-light so the LSP starts instantly.

---

## 5. Status ledger

| # | Workstream | Doc | Status | Depends on |
|---|------------|-----|--------|-----------|
| 00 | Shared `kindle-diagnostics` crate | this file §4 | ✅ DONE (2026-07-23) | — |
| 01 | Compile-time shape diagnostics | [`01-shape-diagnostics.md`](01-shape-diagnostics.md) | ✅ DONE (2026-07-23): `SameCount`/`ReshapeShape` fix + full audit found & fixed 3 more message-less traits (`EndsWith`, `HasChannels1D`/`2D`, `KernelConv2dShape` — the last two hit Linear/Conv1d/Conv2d/BatchNorm directly) + 4 broken compile-fail fixtures repaired (stale API, bad imports) + new snapshot. Permanent gap: typenum arithmetic-underflow errors (foreign trait, can't decorate) | 00 |
| 02 | IDE extensions + LSP | [`02-ide-extensions.md`](02-ide-extensions.md) | PARTIAL (2026-07-23): `kindle-lsp` proxy + VS Code + Neovim clients done; VS Code activation/config-rewrite now verified by an automated test against a real VS Code (not just compiled). RustRover fallback verified; LSP mode still unverified (needs a Gradle/IntelliJ-SDK plugin project) | 00 |
| 03 | Named dimensions (headline) | [`03-named-dimensions.md`](03-named-dimensions.md) | ✅ DONE (2026-07-23): full audit + 6 real bugs found & fixed (`concat`/`stack`/`matmul`/`broadcast`/`SpatialConv1d`+`2d`/`KernelConv2dShape`) + matmul/broadcast/conv support added codebase-wide on request + promoted example + README. Only permanent gap: `.reshape()` and conv/pool spatial dims (need real type-level arithmetic, mathematically impossible for a runtime-valued dim) | 01 |
| 04 | Compile-time model stats | [`04-compile-time-stats.md`](04-compile-time-stats.md) | PARTIAL (2026-07-23): v1 (runtime) done — `ComputeStats` auto-derived for every `#[module]` struct via a new `#[module(no_stats)]` opt-out (needed for `Linear`'s real formula), `model.stats()`/`summary_with_stats()`, tested against this doc's own worked example numbers. v2 (true `const`) not attempted; `activation_bytes` and a true per-layer stats column also deferred (see doc) | — |
| 05 | Observability + scaffolder | [`05-observability-and-scaffolding.md`](05-observability-and-scaffolding.md) | PARTIAL (2026-07-23): reporter ergonomics (`.scalar()`/`.gradient_norm()`/... + `Emitter::to_run_dir()`) done, `cargo kindle watch` done (file-transport only — kindle-viz has no socket reader), `cargo kindle new mnist` done (synthetic data, path deps — see doc). Anomaly panel (05.4) and static-shape graph panel (05.5) not attempted | — |
| 06 | Deployment (binary/GGUF/WASM) | [`06-deployment.md`](06-deployment.md) | PARTIAL (GGUF Q8_0 done) | — |
| 07 | The Book (mdBook) | [`07-the-book.md`](07-the-book.md) | NOT STARTED | 01–06 land first |
| 08 | Agent skills | [`08-agent-skills.md`](08-agent-skills.md) | NOT STARTED | 01–06 land first |

Update this table (and the target doc's own header) as work lands, with a date
and commit hash, exactly like `IMPLEMENTATION_PLAN.md` does.

---

## 6. Global DO-NOT list (in addition to `IMPLEMENTATION_PLAN.md` §0.3)

- **Do NOT change any `Backend`-family trait signature** for any feature here.
  None of these features require it. If you think one does, stop and ask — it is
  a semver-break needing sign-off.
- **Do NOT add AI-attribution trailers** (`Co-Authored-By: Claude`, etc.) to
  commits — repo-wide user preference (`IMPLEMENTATION_PLAN.md` §0.2).
- **Do NOT write template doc comments** ("Auto-generated documentation for X").
  Every new `pub` item gets one real sentence of real behavior.
- **Do NOT let `kindle-diagnostics` or the LSP depend on a GPU backend.** They
  must build on any machine with only a Rust toolchain.
- **Do NOT commit anything under `.claude/`, `.planning/`, `.agents/`** — they
  are gitignored (`IMPLEMENTATION_PLAN.md` §0.2). The **book** (`docs/book/`) and
  these growth docs (`docs/growth/`) *are* committed.
- **Do NOT hand-roll a second typenum parser** anywhere. Everything routes
  through `kindle-diagnostics` (task 00). If you need a new capability, add it
  there.
