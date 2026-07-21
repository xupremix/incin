# Kindle — Release Roadmap

> **Goal:** Ship stable `0.1.0` crates to crates.io with a public API surface
> that can evolve without breaking changes in any `0.x` patch release, and
> with a clearly documented semver contract for the `1.0` boundary.
>
> **Guiding principle:** Every `pub` item is a promise. Anything that is not
> deliberately part of the public API must be `pub(crate)` or `pub(super)`.
> Adding a function later is semver-compatible; removing or changing one is not.

> **2026-07-21 full-codebase audit.** This document was rewritten after a
> line-by-line review of every crate (kindle-core, kindle-backends, kindle-macros,
> kindle, kindle-data, kindle-telemetry, kindle-viz, kindle-viz-plugin-api). The
> previous version of this roadmap had drifted significantly from reality — it
> described crates (`kindle-native`, `kindle-wgpu`) that no longer exist (both were
> consolidated into `kindle-backends`), marked several already-fixed bugs as open,
> and — more importantly — missed real correctness bugs that are now the top
> priority below. Treat the "Critical" section as the actual release blocker list;
> the old B-1..B-6 numbering is kept only for traceability and mapped to its
> current status in the "Legacy blocker status" section.

---

## Current State

| Crate | Tests | Status |
|-------|-------|--------|
| `kindle-core` | Passing | ⚠ Type-safety promise has real gaps (see C-2, C-3) |
| `kindle-backends` (cpu) | Passing on CI-testable path | ⚠ Silent f32 downcast bug (C-1) |
| `kindle-backends` (cuda) | Untested on CI (needs GPU) | ❌ Every op panics on first call (C-1... wait, see C-0) |
| `kindle-backends` (wgpu) | Passing | ❌ Autograd silently produces no gradients (C-4) |
| `kindle-backends` (legacy: candle/ndarray/burn) | Partial | ⚠ Ndarray ~61% stubbed, Burn permanently dead code |
| `kindle-macros` | Passing | ✅ Solid — hygiene good, doc examples present |
| `kindle-data` | **0 tests** | ❌ Untested despite concurrent DataLoader logic |
| `kindle` (facade) | Passing | ⚠ API surface leaks internals via wildcard re-exports |
| `kindle-viz` / `kindle-telemetry` / `kindle-viz-plugin-api` | Passing | 🔲 Prototype, correctly marked `publish = false` |

`kindle-native` and `kindle-wgpu` **no longer exist as separate crates** — CPU,
CUDA, and WGPU execution all live under `crates/kindle-backends/src/{cpu,cuda,wgpu}`
now. Anything below referencing those old crate names has been corrected.

---

## Critical — release blockers, found in this audit

These are silent-wrong-answer or guaranteed-panic bugs in the numeric core, not
style issues. None of them were on the previous roadmap.

### C-1 — Every CUDA op panics on its first invocation
`crates/kindle-backends/src/cuda/ops/{elementwise,norm,embedding,quant,reduce,shape,loss}.rs`
(e.g. `elementwise.rs:132-134`). The pattern
`let mut x = out_b.data.clone(); Arc::get_mut(&mut x).unwrap();` requires the
`Arc` refcount to be exactly 1, but `out_b` (holding the other strong reference)
is still alive at that point — refcount is always 2, so `Arc::get_mut` always
returns `None` and the `.unwrap()` always panics. This is every add/sub/mul/div,
layer/batch norm, embedding forward+backward, quantize/dequantize, reduce/argmax,
concat, and cross-entropy on the CUDA backend. Almost certainly never exercised
in CI (no GPU runner), which is why it's shipped silently.
**Fix:** stop cloning into a shared `Arc` before mutating; allocate a fresh
buffer for the output or use `Arc::make_mut` correctly (which copies-on-write
instead of panicking on refcount > 1).

### C-2 — CPU elementwise ops silently downcast every dtype to f32
`crates/kindle-backends/src/cpu/ops/elementwise.rs:85-88,108-111`. Regardless of
the operands' actual `KindleDType` (F64, F16, BF16, ...), the result is always
constructed as `CpuBuffer::F32(out)`. Every add/sub/mul/div/relu/exp/log/tanh/
sigmoid/gelu/softmax on a non-f32 tensor silently loses precision with no error.
This is the exact anti-pattern the old roadmap accused the legacy Candle/Ndarray
backends of (see B-4 below, which *was* fixed there) — except it's real and
unflagged in the backend meant to replace that legacy code.
**Fix:** dispatch on the actual `CpuBuffer` variant and construct the matching
output variant; add a cross-backend dtype-preservation test as a regression
guard (none currently exists).

### C-3 — WGPU autograd silently produces no gradients
`crates/kindle-backends/src/wgpu/tape.rs` + `wgpu/backend.rs:91,96`. `backward()`
is wired to `tape::backward`, but `tape::push` (`tape.rs:26`) has **zero call
sites** anywhere under `wgpu/*.rs`. The tape is always empty, so `backward()`
only ever returns a gradient for the loss node itself — no parameter tensor ever
receives a gradient on the WGPU backend. `unbroadcast`/`sum_dim_squeeze`/
`sum_dim_keepdim` (`tape.rs:137,160,167`) are correctly implemented and are
*needed* — they're just never reached because no forward op records a
`TapeEntry`. Training on this backend currently runs, converges toward nothing,
and reports no error.
**Fix:** wire every WGPU forward op to call `tape::push` with its backward
closure, the same way the CPU backend does; add a parity test asserting WGPU
gradients match CPU gradients (numerically, not just shape).

### C-4 — CUDA autograd is fully disconnected
`crates/kindle-backends/src/cuda/backend.rs`: `type Grads = ()`, `backward()`
unconditionally returns `Ok(())`, `get_grad()` always returns `Ok(None)`.
`cuda/tape.rs`'s `push`/`backward`/`unbroadcast`/`sum_dim_*` have zero callers
anywhere in `cuda/`. Even more disconnected than the WGPU case (C-3) — there
isn't even a partial wiring attempt.
**Fix:** same shape as C-3's fix, applied to the CUDA backend. Should probably
be done as one shared piece of work since the tape logic is duplicated three
times (cpu/cuda/wgpu) with only the CPU copy actually wired up — worth asking
whether tape/autograd logic can be lifted into `kindle-core` and made backend-
generic instead of re-implemented per backend.

### C-5 — Unchecked shape-multiplication overflow feeds allocation and stride math
`crates/kindle-backends/src/cpu/stride.rs:17-19` (`contiguous_strides`, plain
`*`) and `cpu/creation.rs:70,81,98,127` (`shape.iter().product()` for `Vec`
length). No overflow guard on user-supplied shapes. In release builds (integer
overflow checks are off by default) a crafted or accidentally-huge shape can
wrap the element count to a small number, undersizing the backing `Vec`, while
strides computed from the same unchecked shape are later used to index into it
— a path to out-of-bounds read/write, not just a panic.
**Fix:** use `checked_mul`/`try_fold` when computing `numel` and strides from a
shape, and return `Err(Error::ShapeOverflow)` (or similar) instead of trusting
the multiplication.

### C-6 — kindle-core's dynamic-shape broadcast doesn't actually verify anything
`crates/kindle-core/src/shapes/broadcast.rs:329-420` (all `(usize,)` /
`(usize, B)` / `(usize, B, C)` dynamic-dim impls). `output_shape` computes
`lhs.0.max(rhs.0)` with **no check** that the two dims are equal or that one is
1. This is neither a compile-time check (they're runtime `usize`s by design,
that's fine) nor a runtime check (nothing rejects incompatible dims) — it
silently accepts a shape mismatch and fabricates a plausible-looking but wrong
output shape. This directly undermines the crate's headline "compile-time shape
verification" claim for any tensor with a `Dyn` dimension, which is most real
models (batch size is almost always dynamic).
**Fix:** add the standard NumPy/PyTorch broadcast-compatibility check
(`lhs == rhs || lhs == 1 || rhs == 1`) to every dynamic-dim `output_shape` impl,
returning `Err` on mismatch instead of `.max()`.

### C-7 — Arithmetic operators panic instead of propagating errors
`crates/kindle-core/src/tensor/ops/binary.rs:196,220,243,266` and the scalar
variants in `unary.rs`. `impl core::ops::Add/Sub/Mul/Div for Tensor` call
`self.backend_method(&rhs).unwrap()`. Combined with C-6, a shape mismatch that
slips past the type system on a `Dyn` dimension turns into an uncatchable panic
the first time a user writes ordinary `a + b`, in a language whose entire pitch
is "verified at compile time, not discovered at runtime."
**Fix:** either don't implement `std::ops::Add` et al. for fallible tensor
addition (require `.add()?` everywhere), or keep the operator sugar but make it
panic with a clear, actionable message pointing at the shape mismatch rather
than an opaque `unwrap()` on an internal `Result`. Fixing C-6 removes most real
occurrences of this panic; this item is about not having *any* silent
`.unwrap()` between user code and a backend `Result`.

---

## High priority — real gaps, not yet release-blocking on their own

- **`kindle-data` has zero tests** despite `loader.rs` implementing nontrivial
  concurrent logic (bounded channel, worker pool, shared `Mutex<Iterator>`).
  Highest-risk untested code in the workspace by a wide margin.
- **`legacy::burn_backend` is permanently dead code**:
  `crates/kindle-backends/src/legacy/mod.rs:2138` gates it with `#[cfg(any())]`,
  which is always false — it can never compile in under any feature
  combination. Even if unlocked, it implements a `Backend<(D,)>`/`RawTensor`
  shape from an old API that doesn't match the current `Backend` trait in
  `kindle-core` (no generic parameter, uses `Storage<K>` not `RawTensor`) — it
  cannot compile against present-day `kindle-core` regardless. Meanwhile the
  `burn` dependency is still pulled in transitively by the `legacy` feature for
  code that can never run. **Decide:** delete it outright, or file it as a
  genuine future rewrite and stop compiling `burn` as a dependency until then.
- **`NdarrayBackend` is ~61% stubbed** (54 of 89 ops return
  `UnsupportedBackendOperation`, `legacy/mod.rs:1181-2131`) — confirmed accurate
  from the old roadmap, still true. Fine as a documented `0.1.0-alpha` limitation
  if `legacy` ships with a clear "not all ops implemented" notice; not fine as
  an undocumented surprise.
- **`pub` API leakage is inconsistently fixed across backends.** The old
  roadmap's B-3 (below) was closed for `wgpu`, which now correctly scopes
  `dispatch`/`pipeline`/`device`/`storage` as `pub(crate)`. It was **never
  applied to `cuda`**: `cuda/mod.rs` still marks `ops`, `storage`, `gpu`, `tape`
  as `pub`; `CudaBuffer` exposes raw `Arc<CudaSlice<u8>>`/`Arc<CudaContext>`
  fields as `pub`; `TapeEntry`/`CudaGrads` and dispatch functions
  (`launch_quantize`/`launch_dequantize`/`launch_concat`) are all `pub`. The
  `cpu` backend has the same problem: `CpuBuffer` is a `pub enum`, `CpuGrads`
  has a `pub grads` field. And a **new** leak has appeared in `kindle-core`
  itself: `tensor::tracing::TRACING_GRAPH` is a `pub static Lazy<Mutex<Graph>>`
  — a process-wide mutable singleton that downstream crates can `.lock()` and
  mutate via the trait methods on `Graph` even though `Graph` itself is
  `pub(crate)`. This also means every test in the same process shares one
  tracing graph — a source of cross-test pollution independent of the API
  leak. **B-3 should not be marked done anywhere until cuda + cpu + this new
  kindle-core leak are all closed.**
- **Facade (`kindle`) blanket-exports internals via wildcard globs**:
  `crates/kindle/src/lib.rs:83` (`pub use kindle_backends::*;`) and `:170`
  (`pub use kindle_macros::*;`) re-export everything, including internal
  proc-macro helpers like `generate_shape_ops` and `impl_arg_into` that have no
  business in the public prelude. The old roadmap already asked for this audit
  (line "Audit re-exports") — this confirms it's still open, with the exact
  offending lines.
- **`DefaultBackend` silently degrades to `()` when `cpu` is disabled**:
  `crates/kindle/src/lib.rs:107-109`. Since `Tensor<S, B = DefaultBackend, ...>`
  requires `B: Backend`, a user who disables the `cpu` feature without
  explicitly naming a backend gets a default that can't satisfy its own trait
  bound — a confusing error far from the actual misconfiguration, not a clear
  "you forgot to pick a backend" message.
- **Dead `#[cfg(feature = "candle")]` block in the facade**:
  `crates/kindle/src/lib.rs:189` references a feature that doesn't exist in
  `crates/kindle/Cargo.toml` (`std`, `cpu`, `cuda`, `wgpu`, `legacy`, `nightly`
  only) — confirmed via a `cargo` "unexpected cfg" warning. This code path can
  never activate. The crate's own top-doc comment still advertises Candle
  support, which is now misleading in two ways (README's version of this claim
  was already fixed in this pass, this one wasn't).
- **The 3-tier CUDA autotuning engine described in `PROJECT_MEMORY.md` and
  `docs/AUTOTUNING_AND_ARCHITECTURE.md` does not exist.** No `autotune`,
  occupancy-query, `LRUCache`, or cached-`LaunchConfig` code exists anywhere
  under `cuda/*.rs` — only hardcoded per-call `LaunchConfig` literals (fixed
  block size 256). This was already an open checkbox in `PROJECT_MEMORY.md`'s
  status section, but the docs describe it in the present tense as if partially
  built; it is fully aspirational today. Either scope it as real upcoming work
  or reframe the docs as a design doc for unbuilt work, not architecture-in-place.

---

## Medium priority

- **Unsound raw-byte reinterpretation**: `to_scalar<E: Copy>`/`to_vec1<E: Copy>`
  in `kindle-core/src/tensor/ops/manipulation.rs:362,370,387-393` only check
  byte-length equality against `size_of::<E>()`, never cross-check the tensor's
  actual `KindleDType` against `E` before `read_unaligned`/`copy_nonoverlapping`.
  Undefined behavior if `E = bool` (or similar) and the underlying byte isn't
  0/1. Add a dtype/`E` compatibility check before the unsafe reinterpret.
- **`panic!`/`unimplemented!` inside `Result`-returning library functions**:
  `kindle-core/src/serialize.rs:82` (Q8_0), `onnx_exporter.rs:110`
  (`dtype_to_onnx`, reachable from `export_to_onnx`), `shapes/idx.rs:137`
  (multi-inferred-dim reshape, reachable from user-facing `reshape`). All should
  return `Err` instead of panicking — they're reachable from ordinary public API
  calls, not internal invariants.
- **ONNX model loading is a non-functional stub wired up as if real**:
  `OnnxImporter::deserialize` (`onnx_exporter.rs:168-184`) always returns
  `Err("ONNX loading is currently unsupported...")`, but `nn/save.rs:340-356`
  (`ModelExt::load`) calls it as a normal code path. (Note: the actual
  `import_model!` protobuf parsing that runs at compile time lives in
  `kindle-macros`, is separate from this, and works — so there's no untrusted-
  file attack surface here, just a misleading dead-end runtime API.)
- **Dead `if true {...} else {...}` branches across CUDA ops** (`elementwise.rs`,
  `embedding.rs`, `quant.rs`, `reduce.rs`, `norm.rs`): unreachable
  `Err("Not a CUDA buffer")` arms are leftover refactor cruft that hides the
  fact no real type-check happens at those call sites.
- **Dynamic CUDA kernel source is built by string templating**
  (`cuda/ops/elementwise.rs`, `ELEMENTWISE_*_TEMPLATE.replace("{OP}", op_expr)`)
  and compiled at runtime via `nvrtc::compile_ptx`. Currently only fed trusted
  string literals, so not exploitable today, but this is a compile-injection-
  shaped pattern — if it's ever extended to accept user-defined ops, it needs
  sanitization first.
- **`PROJECT_MEMORY.md` describes `KindleBackend<T, D>` as an already-unified
  backend struct** ("Instead of separate backend structs, kindle uses a single
  unified `KindleBackend<T, D>`..."), but its own status checklist lists this as
  unchecked, and the audit confirms the actual code still has separate
  `CpuBackend`/`CudaBackend`/`WgpuBackend`/`CandleBackend`/`NdarrayBackend`
  types (these are the graph's top god-nodes by edge count). Reframe that
  section as a design target, not current architecture, until the refactor lands.
- **`kindle-telemetry`'s file transport has no explicit permission hardening**
  (`transport/file.rs:35`, relies on default umask). Not currently exploitable
  (resolves under `~/.local/share`), but no defense-in-depth if that assumption
  ever breaks (shared multi-user data dir, misconfigured `XDG_DATA_HOME`).
  Consider explicit `0600` via `PermissionsExt`.

---

## Low priority / cleanup

- Vestigial unused `op_name`/`op_expr` params in `cpu/ops/elementwise.rs:67-68,94-95`
  — dead scaffolding for a debug/introspection feature that was never wired up
  (`TapeEntry` has no field to receive them). Delete or actually wire them into
  error messages.
- `DummyBackend` conv/pool shape math (`kindle-core/tensor/backend.rs:1357,1380-1381,1405-1406,1423-1424`)
  risks unsigned underflow/panic for small input + large kernel/dilation/padding.
  B-5 (below) is functionally fixed but undefended against pathological inputs.
- `kindle-viz/src/panels/graph.rs:163` has a fragile `unwrap()` that's only
  safe because of an early-return three lines above — refactor to `if let`/`else`.
- Doc comments across most of `kindle-core`/`kindle-native`/`kindle-wgpu`-successor
  code are content-free templates ("Auto-generated documentation for X") —
  they satisfy a doc-coverage lint while conveying nothing. The real
  documentation debt is hidden behind an apparently-met bar, not resolved.
- ~1,200 `unwrap()`/`expect()` call sites project-wide. Not all are wrong (many
  are in tests or genuinely-infallible contexts), but this is a systemic pattern
  worth auditing incrementally, starting with anything reachable from public
  API entry points (see C-7, the manipulation.rs and serialize.rs items above
  for the ones already confirmed reachable from user code).
- `s![]`/`idx![]` doc examples exist (`kindle-macros/src/lib.rs:42-57,84-92`)
  contrary to the old roadmap's claim they were missing — but they're marked
  `rust,ignore`, so `cargo test --doc` never actually compiles/verifies them.
  Worth making at least one of each a real, compiled doctest.
- `crates/kindle/Cargo.toml`'s `anyhow` dev-dependency is used by exactly two
  examples (`mnist_training.rs`, `rnn_sequence_prediction.rs`), not by tests.
  Switching those examples to `kindle_core::Result` would let the dependency
  be dropped entirely.
- `kindle-telemetry`'s `HyperparamEvent` is a free-form value bag with no
  automatic capture of env vars or paths — but if a caller logs a secret-bearing
  string as a "hyperparameter," it persists verbatim in shareable JSONL output.
  Worth a doc caveat.
- graphify's own knowledge graph (`graphify-out/`) contains stale nodes
  referencing a `kindle-core/src/dashboard/` module that no longer exists in the
  tree — re-run `graphify update .` (per `.agents/rules/graphify.md`) after
  landing any of the fixes above.

---

## New Implementation Plan (ordered)

This supersedes the old "Summary Checklist" — it's re-ordered around what's
actually load-bearing for correctness, not just release paperwork.

**Phase 0 — stop the bleeding (do first, small and isolated)**
1. Fix CI (`kindle-backends/native` → a real feature — **done in this pass**,
   now runs `--no-default-features --features kindle-backends/cpu,kindle/cpu`).
   GPU-gated jobs (cuda/wgpu) still need a runner with actual hardware or a
   software fallback before they can be added — currently there is no CI
   coverage at all for the two backends with the worst bugs (C-1, C-3, C-4),
   which is exactly how those bugs shipped unnoticed.
2. Fix C-1 (CUDA `Arc::get_mut` panic pattern) — mechanical, same fix repeats
   across ~7 files.
3. Fix C-2 (CPU f32 downcast) — dispatch on actual `CpuBuffer` variant.
4. Fix C-6 (dynamic broadcast shape check) then C-7 (remove/guard the
   now-mostly-unreachable operator-overload panics).
5. Fix C-5 (checked shape multiplication).

**Phase 1 — autograd correctness (the framework's actual value proposition)**
6. Fix C-3 (wire WGPU tape) and C-4 (wire CUDA tape). Consider lifting the
   duplicated tape/autograd logic into `kindle-core` as a backend-generic
   implementation instead of three parallel copies, only one of which works.
7. Add a cross-backend gradient-parity test (CPU vs WGPU vs CUDA, numeric
   tolerance) as a permanent regression guard — this class of bug is invisible
   without one.

**Phase 2 — API surface & encapsulation audit**
8. Close B-3 for real: `cuda` and `cpu` modules brought to the same
   `pub(crate)` discipline `wgpu` already has; remove the `kindle-core`
   `TRACING_GRAPH` leak (make it `pub(crate)`, and consider whether a
   process-wide singleton is the right design at all vs. a graph handle threaded
   through the API).
9. Audit `kindle`'s wildcard re-exports (`pub use kindle_backends::*` /
   `kindle_macros::*`) down to an explicit allowlist.
10. Fix or remove `DefaultBackend = ()` fallback; fix or delete the dead
    `#[cfg(feature = "candle")]` block.
11. Add `#[non_exhaustive]` / doc-coverage items already tracked below.

**Phase 3 — test debt**
12. Write tests for `kindle-data`'s `DataLoader` (highest-risk untested code).
13. Decide `legacy::burn_backend`'s fate (delete vs. real rewrite ticket) and
    stop paying its dependency cost either way.
14. Make `s![]`/`idx![]` doctests real (`rust,ignore` → compiled).

**Phase 4 — docs & release paperwork** (mostly unchanged from before, see below)

---

## Legacy blocker status (old B-1..B-6, for traceability)

| ID | Was | Now |
|----|-----|-----|
| B-1 | wgpu `test_adamw_step` GPU/CPU race | ✅ Fixed — `device.poll` present at `wgpu/dispatch.rs:214` |
| B-2 | `kindle` facade linker bus-error | ✅ Fixed — `cargo test -p kindle` passes clean, full suite + doctests |
| B-3 | Accidental `pub` leakage across crates | ⚠ **Partially fixed** — `wgpu` done correctly; `cuda`, `cpu`, and a new `kindle-core` leak (`TRACING_GRAPH`) are open. Do not mark this done — see High priority section above. |
| B-4 | Legacy backends ignore `FloatElem` generic | ✅ Fixed — both Candle and Ndarray legacy wrappers do `type FloatElem = T` correctly now |
| B-5 | `DummyBackend` conv/pool shape math wrong | ✅ Fixed, but see Low-priority underflow-hardening note above |
| B-6 | Compile-fail tests exercising wrong failure | ✅ Fixed — macro imports present, `.stderr` snapshots updated |

---

## API Stability Contract (must define before release)

### What is stable in `0.1.0`

| Symbol | Crate | Stable? |
|--------|-------|---------|
| `Backend` trait + all sub-traits (`FloatOps`, `NumericOps`, ...) | `kindle-core` | ✅ Yes — this is the extension point |
| `Tensor<S, B, K, D, G>` type and its inherent methods | `kindle-core` | ✅ Yes |
| `s![]`, `idx![]` macros | `kindle-macros` | ✅ Yes |
| `CpuBackend<T>`, `CudaBackend<T>`, `WgpuBackend<T>` | `kindle-backends` | ✅ Yes (the structs only) |
| `Error` enum variants | `kindle-core` | ✅ Yes (`#[non_exhaustive]` — already applied) |
| `nn::Linear`, `Conv2d`, `LayerNorm`, etc. | `kindle-core` | ✅ Yes |
| `dispatch_*` / `launch_*` functions | `kindle-backends` (cuda, wgpu) | ❌ No — must be `pub(crate)` everywhere (cuda still leaks these) |
| `CpuBuffer`, `CudaBuffer`, `WgpuBuffer`/`WgpuStorage` fields | `kindle-backends` | ❌ No — fields/enums must be private or `pub(crate)` |
| `TRACING_GRAPH` static | `kindle-core` | ❌ No — must be `pub(crate)`, see High priority section |
| Internal modules (`tape`, `stride`, `creation`, `ops`, `gpu`) | `kindle-backends` | ❌ No — `pub(crate)` |

### `#[non_exhaustive]` status
- `Error` enum — ✅ already applied (`kindle-core/src/err.rs:8`)
- `KindleDType` enum — ✅ already applied (`kindle-core/src/tensor/dtype.rs:43`)
- `KindleDevice` struct — check and apply if missing

### Semver implications
Adding a new associated type or method to the `Backend` trait is a **breaking
change** (implementors must add it). Default impls returning
`Err(Error::UnsupportedBackendOperation {...})` are already in place for
non-critical `Backend` sub-trait methods (`kindle-core/src/tensor/backend.rs:495-563`)
— keep this pattern for any new methods.

---

## Documentation Requirements

Doc-coverage is nominally met but hollow: most `pub` items across
`kindle-core`/`kindle-backends` have a `///` comment, but it's frequently an
"Auto-generated documentation for X" template with no real content. Treat doc
coverage as **not done** until comments describe actual behavior, not just
satisfy a lint.

**Minimum bar for `0.1.0`:** every `pub` item has a real one-line description
(not a template); every crate has a `//!` module doc; every non-trivial public
type has a *compiled* usage example (not `rust,ignore`).

---

## Testing Requirements

| Type | Current | Target for `0.1.0` |
|------|---------|---------------------|
| Unit tests (per-op) | Present in core/cpu, absent in kindle-data | All crates covered, 0 known-failing |
| Cross-backend numeric parity | None | CPU vs WGPU vs CUDA parity to 1e-4, **including gradients** (would have caught C-1/C-3/C-4) |
| Compile-fail shape tests | Fixed (B-6) | Extend to cover the C-6 dynamic-broadcast gap once fixed |
| Doc tests | Present but `rust,ignore` | At least `s![]`/`idx![]` compiled as real doctests |
| Concurrency tests | None for `kindle-data` | `DataLoader` worker pool / channel / mutex coverage |

---

## Repository Hygiene

- [x] Scratch files at root (`diagnostic_test.rs`, `scratch*.py`, etc.) — already gone
- [x] `publish = false` on `kindle-viz` / `kindle-telemetry` / `kindle-viz-plugin-api` — already set
- [x] README no longer claims to "wrap candle and burn" as the primary story — fixed in this pass
- [ ] Move planning docs (`FUTURE_ROADMAP.md`, `GPU_ROADMAP.md`, etc.) to `docs/` if any still exist at root
- [ ] `anyhow` dev-dependency in `kindle` facade — narrow to the 2 examples that use it, or remove
- [ ] Add `[workspace.metadata.release]` or similar to control which crates get published
- [x] CI feature flag fixed (`kindle-backends/native` → `kindle-backends/cpu,kindle/cpu`) in this pass
- [ ] Add GPU-hardware-gated CI jobs for `cuda`/`wgpu` (or explicit software-fallback jobs) — currently zero CI coverage for the backends with C-1/C-3/C-4
- [ ] Add `CHANGELOG.md` following Keep-a-Changelog format
- [x] `CONTRIBUTING.md` exists

---

## Feature Flag Audit

```bash
cargo check -p kindle-core --no-default-features
cargo check -p kindle-core --all-features
cargo check -p kindle-backends --no-default-features
cargo check -p kindle-backends --features cuda    # needs CUDA env, CI-gated
cargo check -p kindle-backends --features wgpu
cargo check -p kindle --no-default-features
cargo check -p kindle --features legacy
```

Known gap: `--features legacy` pulls in `burn` for a permanently-dead
`burn_backend` module (`#[cfg(any())]`) — see High priority section.

---

## Publish Order (dependency graph)

```
kindle-macros   (no workspace deps)
    ↓
kindle-core     (dep: kindle-macros)
    ↓
kindle-backends (dep: kindle-core)
    ↓
kindle          (dep: kindle-core, kindle-macros, kindle-backends, kindle-data)
kindle-data     (dep: none from workspace, could publish independently)
```

Publish `kindle-viz`, `kindle-telemetry`, `kindle-viz-plugin-api` after `kindle`
is stable.

---

## Version Strategy

| Milestone | Version | Meaning |
|-----------|---------|---------|
| Internal only | `0.1.0-alpha.1` | Fixes C-1..C-7, API cleanup, no public announcement |
| Beta | `0.1.0-beta.1` | All tests pass, gradient parity tests green, full docs, CI green (including GPU-gated jobs) |
| Release | `0.1.0` | crates.io publish, README updated, announcement |
| First breaking change | `0.2.0` | Semver minor — breaking only in `0.x` land |
| Stable API | `1.0.0` | `Backend` trait frozen, no breaking changes without major bump |

---

## Summary Checklist (ordered by priority)

1. **C-1** Fix CUDA `Arc::get_mut` panic (every CUDA op)
2. **C-2** Fix CPU elementwise f32 downcast (dtype correctness)
3. **C-6 / C-7** Fix dynamic broadcast validation, then the operator-overload panics it was masking
4. **C-5** Checked shape multiplication (overflow → OOB path)
5. **C-3 / C-4** Wire up WGPU and CUDA autograd tapes (currently silent no-ops)
6. Add cross-backend gradient-parity tests
7. Close B-3 for real: `cuda`/`cpu` `pub(crate)` audit + `kindle-core` `TRACING_GRAPH` leak
8. Audit `kindle` facade wildcard re-exports; fix `DefaultBackend = ()` trap; remove dead `candle` cfg
9. Write `kindle-data` `DataLoader` tests
10. Decide `legacy::burn_backend`'s fate; stop paying its dependency cost
11. Add GPU-gated CI jobs (cuda/wgpu) — the current gap is why C-1/C-3/C-4 shipped unnoticed
12. Make doc comments real (not templates); make `s![]`/`idx![]` doctests compile for real
13. Write `CHANGELOG.md`, finish remaining repo hygiene items above
