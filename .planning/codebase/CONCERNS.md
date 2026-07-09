# Codebase Concerns

**Analysis Date:** 2026-07-09

## Context

The repo just finished a large refactor of `kindle-core`'s `Backend` trait (dtype moved from
a fixed per-backend associated type to a per-call generic `Storage<K: DType>`, `Tensor` gained
an explicit `Device` generic: `Tensor<S, B, K, D, G>`). `cargo build --workspace --examples`
and `cargo test --workspace` are reported clean, but several backends are stubs and a root
`TODO.md` (repo root, `/home/xupremix/kindle/TODO.md`) documents known gaps from that pass.
This document incorporates those plus independent findings.

## Tech Debt

**`BurnBackend` is fully defunct:**
- Issue: Built against a completely different, already-abandoned API shape
  (`Backend<(D0, D1, ...)>` with per-tensor `Dim` generics) that predates the `Storage<K>`
  refactor and doesn't exist anywhere in the current `Backend` trait.
- Files: `crates/kindle-backends/src/lib.rs` (search `burn` feature block, ~line 1486-1536),
  `crates/kindle-backends/Cargo.toml` (`burn = ["dep:burn"]`, `burn-ndarray` unconditional dep),
  `crates/kindle/Cargo.toml` (`burn = ["kindle-backends/burn"]`)
- Impact: Does not compile with `--features burn`. Feature is off by default so CI (which
  doesn't pass `--features burn`) never catches this; it will silently keep rotting.
- Fix approach: From-scratch redesign against the current `Backend`/`Storage<K>` trait shape,
  not a mechanical port. Consider removing the feature entirely until someone commits to it,
  since a permanently-broken opt-in feature is worse than no feature.

**`NdarrayBackend` is mostly stubbed with `UnsupportedBackendOperation`:**
- Issue: `matmul`, `stack`, `concat`, `broadcast_as`, most reductions, all conv/pool ops, and
  most activations return `Error::UnsupportedBackendOperation`. Only a handful of elementwise
  ops, `reshape`, and `slice` are real implementations.
- Files: `crates/kindle-backends/src/lib.rs` — 104 occurrences of `UnsupportedBackendOperation`
  in this file (spanning both the Ndarray and Burn backend impls); Ndarray-specific stub block
  starts around line 1172 (`matmul`), 1178 (`stack`), 1184 (`concat`), 1190 (`broadcast_as`),
  1272 (`adaptive_avg_pool2d`), 1386-1402 (`l1_loss`/`bce_with_logits_loss`/`mse_loss`).
- Impact: `NdarrayBackend` cannot run any real model that uses matmul, conv, or most losses —
  usable only as a minimal reference/compile-time backend, not for real workloads. Anyone
  reaching for it as a lightweight CPU backend (its likely intended use case) will hit a wall
  immediately.
- Fix approach: Prioritize `matmul` and `broadcast_as` first (they gate almost everything else
  via composition), then conv/pool, then remaining reductions/losses.

**`DummyBackend` conv/pool ops are shape-incorrect passthroughs:**
- Issue: `conv1d`/`conv2d`/`conv_transpose2d`/`max_pool2d`/`avg_pool2d` all `return t.clone()`
  (unchanged input shape) instead of computing the real output shape from
  kernel/stride/padding/dilation.
- Files: `crates/kindle-core/src/tensor/backend.rs` (the `dummy` module) — `t.clone()`
  occurrences at lines 348, 380, 452, 488-550 span the shape-changing op stubs; only
  `adaptive_avg_pool2d` and the `*_keepdim` reductions were fixed to compute real shapes.
- Impact: `DummyBackend` is used for compile-time shape-checking tests (`trybuild`
  compile-fail fixtures) and general unit tests. Any test that exercises conv/pool shape math
  through `DummyBackend` will silently pass with a wrong shape instead of catching a real bug.
- Fix approach: Implement the standard output-shape formula per spatial dim —
  `(in + 2*pad - dilation*(kernel-1) - 1) / stride + 1` — as described in `TODO.md`. Low
  priority unless/until an ONNX-import test starts relying on conv/pool shape correctness here.

**`trybuild` compile-fail fixtures missing macro imports:**
- Issue: `crates/kindle-core/tests/compile_fail/stack_static_mismatch.rs` and
  `concat_static_mismatch.rs` are missing `use kindle_macros::s;`, so they fail to compile
  for the wrong reason ("cannot find macro `s`") instead of exercising the intended
  static-shape-mismatch check.
- Files: `crates/kindle-core/tests/compile_fail/stack_static_mismatch.rs`,
  `crates/kindle-core/tests/compile_fail/concat_static_mismatch.rs`, and their paired
  `.stderr` snapshots.
- Impact: These two tests pass today, but only by accident — they don't test what their
  filenames claim. A real regression in static shape-mismatch detection for `stack!`/`concat!`
  would go undetected.
- Fix approach: Add the missing import and regenerate snapshots with `TRYBUILD=overwrite`.
  Pre-existing gap, not introduced by the current refactor.

**Documentation drift on `dummy` backend's purpose:**
- Issue: The module doc comment on `crates/kindle-core/src/tensor/backend.rs`'s `dummy` module
  says it's "strictly used for testing compile-time shape verification," but it now also does
  real runtime shape bookkeeping (`Storage<K>` carries a real shape vector, `*_keepdim`
  reductions are real).
- Files: `crates/kindle-core/src/tensor/backend.rs`
- Impact: Contributors reading the doc comment will assume shape math in `DummyBackend` is
  fake/unchecked, when parts of it (reductions, `adaptive_avg_pool2d`) are now real. Risk of
  someone "fixing" a doc-comment-driven assumption incorrectly, or not trusting output that is
  in fact correct.
- Fix approach: Either lean fully into real shape math (finish conv/pool per above) and update
  the doc comment, or explicitly scope `DummyBackend` back down to compile-time-only and revert
  the runtime shape logic. Currently in an inconsistent middle state.

**Working tree has large uncommitted, machine-generated refactor churn plus stray scratch files:**
- Issue: At time of analysis `git status` shows ~30 modified files across `kindle-core`,
  `kindle-macros`, and `kindle-backends` (the backend-trait refactor) uncommitted on
  `dev/refactor`, alongside a stray `expanded.rs` (0 bytes, macro-expansion dump — already
  covered by `.gitignore`) and `rewrite_backend.py` at the repo root (an ad hoc Python codegen
  helper used to mechanically rewrite backend impls during the refactor).
- Files: repo root `expanded.rs`, `rewrite_backend.py`; modified files across
  `crates/kindle-core/src/**`, `crates/kindle-macros/src/module.rs`,
  `crates/kindle-backends/src/lib.rs`
- Impact: `rewrite_backend.py` is not part of the build and risks being mistaken for a
  supported dev tool; if it encodes assumptions about the old trait shape it is now stale and
  misleading if run again.
- Fix approach: Commit the refactor changes with a clear message (in progress per `TODO.md`),
  then delete `rewrite_backend.py` and `expanded.rs` or move them to a `scripts/` or `scratch/`
  directory excluded from the crate (note `.gitignore` already excludes `scratch/`).

## Known Bugs

**`Tensor::from_slice` always interprets input bytes as `f32`, ignoring the target dtype `K`:**
- Symptoms: `from_slice` takes `data: &[f32]`, transmutes it to raw bytes via
  `core::slice::from_raw_parts`, then calls `B::from_bytes(bytes, dims, KindleDType::F32, ...)`
  — the `KindleDType::F32` tag is hardcoded regardless of the tensor's actual const-generic
  dtype `K`. If a caller constructs a non-f32 tensor (e.g. `Tensor<_, _, I64, _, _>`) via
  `from_slice`, the backend is told the bytes are `f32` while the `Tensor`'s type-level dtype
  says otherwise.
- Files: `crates/kindle-core/src/tensor/base.rs:163-177`
- Trigger: Call `Tensor::<Shape, Backend, K, Device, Grad>::from_slice(&[1.0f32, 2.0], args)`
  where `K` is not `F32` (e.g. an integer or f64 dtype).
  In this state, either the backend silently reinterprets bytes as the wrong width (data
  corruption) or later dtype-tagged ops panic/produce garbage since the `Storage<K>` metadata
  and raw bytes disagree.
- Workaround: None currently; only safe when `K = F32`. No compile-time constraint enforces
  this today (only `A: ArgInto<...>` and `K: ConstDType`, `D: ConstDevice` are required —
  `K::DTYPE == F32` is not asserted).

**Multiple raw pointer reinterpret-casts around f32 byte buffers, unchecked for platform
endianness or dtype width mismatches:**
- Symptoms: Same class of issue as above — several `unsafe` blocks reinterpret `&[f32]`/byte
  buffers via `core::ptr::read_unaligned` / `core::slice::from_raw_parts` casts between `f32`
  and raw bytes without asserting the buffer length is dtype-width-aligned or validating the
  requested `E`/`K` type actually matches 4-byte width.
- Files: `crates/kindle-core/src/tensor/ops/manipulation.rs:331,339,356`,
  `crates/kindle-core/src/tensor/base.rs:174`, `crates/kindle-backends/src/lib.rs:126,141,860,880`
- Trigger: Any codepath that stores/reads scalar values through these unsafe transmute helpers
  for a dtype whose width isn't 4 bytes (e.g. `f64`, `i8`, `bool`-as-byte) risks reading
  out-of-bounds or garbage data — `read_unaligned::<E>` reads `size_of::<E>()` bytes starting
  at a pointer sized for a different type in at least one call site.
- Workaround: None; needs an audit pass confirming every call site's `E`/dtype width matches
  the buffer's actual byte layout before trusting these paths for non-f32 dtypes.

## Security Considerations

**Multiple `unsafe` blocks performing raw pointer casts without bounds/alignment validation:**
- Risk: `unsafe { core::slice::from_raw_parts(...) }` and
  `unsafe { core::ptr::read_unaligned(...) }` calls construct slices/reads from raw pointers
  based on lengths computed from a *different* type's size (see Known Bugs above). If a
  mismatch ever occurs (wrong dtype tag, corrupted length), this is undefined behavior — not
  just a logic bug — with potential for out-of-bounds reads.
- Files: `crates/kindle-core/src/tensor/ops/manipulation.rs:331-356`,
  `crates/kindle-core/src/tensor/base.rs:174`, `crates/kindle-backends/src/lib.rs:126-141,860-880`
- Current mitigation: None beyond the type system's `ConstDType`/`DType` bounds, which do not
  enforce byte-width agreement between the raw buffer and the requested reinterpretation
  (see `from_slice` bug above, which is the most direct way to break this invariant).
  This is a numerical computing library, not code processing untrusted external input over a
  network boundary, so exploitability is low — but incorrect/garbage tensor values silently
  propagating through a training loop is a correctness and reproducibility hazard.
- Recommendations: Add `debug_assert!`/const-eval checks that `size_of::<E>()` (or the dtype's
  declared width) matches the buffer stride before every raw-pointer reinterpret; consider
  `bytemuck`'s checked cast functions (`try_cast_slice`) instead of hand-rolled
  `from_raw_parts`/`read_unaligned` to get panic-on-mismatch instead of UB.

**No `#![forbid(unsafe_code)]` or centralized unsafe audit boundary:**
- Risk: `unsafe` usage is scattered across `kindle-core` and `kindle-backends` (8 call sites
  found) rather than isolated behind a single reviewed module/newtype boundary, making it
  easy for future contributors to add more ad hoc unsafe code without review discipline.
- Files: `crates/kindle-core/src/tensor/ops/manipulation.rs`, `crates/kindle-core/src/tensor/base.rs`,
  `crates/kindle-backends/src/lib.rs`
- Current mitigation: None.
- Recommendations: Consolidate raw byte <-> typed value conversions into one small, well-tested
  internal module (or adopt `bytemuck`) so all reinterpret-casts go through one audited path.

## Performance Bottlenecks

**`NdarrayBackend`'s stubbed `matmul`/`broadcast_as` block any real numeric workload:**
- Problem: Since `matmul`, `stack`, `concat`, and `broadcast_as` are unimplemented for
  `NdarrayBackend`, this backend cannot currently be benchmarked or used for anything beyond
  toy elementwise-op tests — not a "slow path", but a "no path."
- Files: `crates/kindle-backends/src/lib.rs` (Ndarray backend block, ~line 1172-1402)
- Cause: Incomplete backend implementation (tracked above as tech debt; listed again here
  because it is also the reason no performance baseline exists for this backend).
- Improvement path: Once `matmul`/`broadcast_as` land, benchmark against `CandleBackend` to
  decide whether `NdarrayBackend` is worth maintaining as more than a reference implementation.

## Fragile Areas

**Backend trait surface (`Storage<K>` generic reshape) is mid-refactor across crates:**
- Files: `crates/kindle-core/src/tensor/backend.rs` (894 lines, defines `Backend`/`CreationOps`/
  `NumericOps`/`TensorOps`/`FloatOps`/`ReductionOps`/`ModuleOps`/`LossOps` traits and the
  `DummyBackend` reference impl), `crates/kindle-backends/src/lib.rs` (1585 lines, `CandleBackend`
  + `NdarrayBackend` + defunct `BurnBackend` in one file), `crates/kindle/src/lib.rs`
  (convenience `Tensor<S, B, G>` alias)
- Why fragile: `kindle-backends/src/lib.rs` implements three entire backends (Candle, Ndarray,
  Burn) in a single 1585-line file with heavy macro-driven boilerplate
  (`kindle_core::prelude::Backend`/associated-type-qualified signatures repeated per impl).
  A single misalignment between the trait definition in `kindle-core` and any one backend's
  impl (as already happened with `kindle::Tensor`'s positional-argument alias bug, see below)
  silently produces wrong behavior rather than a compile error, because generic parameters are
  positionally forwarded in several places.
- Safe modification: When changing the `Backend` trait shape in `kindle-core`, grep across all
  three backend impls in `kindle-backends/src/lib.rs` in one pass rather than trusting the
  compiler alone to catch every signature drift — the recently-fixed `kindle::Tensor` alias bug
  (3rd generic param silently going into the dtype slot instead of the grad slot) shows the
  compiler did NOT catch a positional-argument mismatch of this kind; it just silently
  miscompiled semantics until someone traced example failures.
- Test coverage: Backend impls have basic unit tests (`crates/kindle-backends/tests/ndarray.rs`,
  `crates/kindle-backends/tests/ops.rs`) but given the extent of the stubbed surface, coverage
  of the *implemented* subset is what's tested — the large stubbed portion has no failing-test
  signal, it just returns `Err(UnsupportedBackendOperation)` if ever exercised.

**Positional generic-parameter forwarding in `kindle`'s convenience `Tensor` alias:**
- Files: `crates/kindle/src/lib.rs`
- Why fragile: The crate's ergonomic `Tensor<S, B, G>` alias forwards its parameters
  positionally into the underlying 5-param `kindle_core::Tensor<S, B, K, D, G>`. This exact
  class of bug (3rd param going into the wrong slot) was just fixed per `TODO.md`, but the
  underlying pattern — hand-maintained positional forwarding between a short alias and a longer
  canonical generic type — remains and can silently regress again if either type gains/loses/
  reorders a generic parameter without the alias being updated in lockstep.
- Safe modification: Any change to `kindle_core::Tensor`'s generic parameter order or count
  must be paired with an update to `crates/kindle/src/lib.rs`'s alias in the same commit;
  consider adding a `trybuild`/doctest that would fail to compile if the mapping silently
  breaks (e.g. a test asserting a known-good tensor construction path type-checks with the
  expected concrete dtype).
- Test coverage: Not verified for a regression test guarding this exact class of bug going
  forward; the fix was discovered via example build failures, not a targeted test.

**`kindle-macros/src/onnx.rs` codegen relies on `panic!()`-emitting generated code for
malformed ONNX graphs:**
- Files: `crates/kindle-macros/src/onnx.rs:181-252` (`If`/`Loop` node handling)
- Why fragile: When the `import_model!` macro encounters an ONNX `If`/`Loop` node missing its
  `then_branch`/`else_branch`/`body` graph or attribute, it generates code that does
  `panic!("If node missing then_branch graph")` etc. at runtime inside the imported model's
  generated `forward()`, rather than surfacing a `Result::Err` or failing at macro-expansion
  (compile) time.
- Safe modification: Prefer emitting a compile_error!/proc-macro diagnostic at macro-expansion
  time when the ONNX graph is known malformed at codegen time (since `import_model!` runs at
  compile time over a fixed `.onnx` file, the malformed-graph case is knowable statically) —
  avoids a runtime panic surprise for users of generated models.
- Test coverage: Not verified whether any test exercises `If`/`Loop` ONNX import with a
  malformed graph to confirm the panic path is even reachable/tested.

## Dependencies at Risk

**`burn` / `burn-ndarray` (0.21.0) — dead weight given `BurnBackend` is non-functional:**
- Risk: `burn-ndarray = "0.21.0"` is an *unconditional* dependency in
  `crates/kindle-backends/Cargo.toml` (not gated behind the `burn` feature), even though the
  `burn`-feature-gated `BurnBackend` itself doesn't compile. This adds a real, always-built
  dependency for a backend nobody can currently use.
- Impact: Slower builds/larger dependency tree for every consumer of `kindle-backends`
  regardless of whether they want Burn support, in exchange for a codepath that isn't
  reachable end-to-end today.
- Migration plan: Either gate `burn-ndarray` behind the same `burn` feature flag as `burn`
  itself, or remove it until `BurnBackend` is redesigned and actually wired up.

## Test Coverage Gaps

**Stubbed backend operations have no "expected to fail" regression tests:**
- What's not tested: There is no test asserting that `NdarrayBackend::matmul` (and the ~100
  other `UnsupportedBackendOperation` call sites) currently *correctly* returns an error rather
  than silently becoming a correctness bug if someone half-implements it later.
- Files: `crates/kindle-backends/src/lib.rs`, `crates/kindle-backends/tests/ndarray.rs`,
  `crates/kindle-backends/tests/ops.rs`
- Risk: A partially-correct future implementation of e.g. `matmul` for `NdarrayBackend` could
  land with subtle bugs (wrong broadcasting, wrong dim order) and nothing would catch it beyond
  manual testing, since there's currently no baseline test fixture prepared for these ops.
- Priority: Medium — matters once someone starts implementing the stubbed ops, not urgent now.

**`DummyBackend` conv/pool shape math has no direct unit test:**
- What's not tested: No test directly asserts `DummyBackend::conv2d`/`max_pool2d`/etc. produce
  correct *output shapes* for given kernel/stride/padding/dilation combinations — the TODO.md
  notes these are currently pass-throughs and only escaped notice because no exercised test
  path relies on their shape correctness.
- Files: `crates/kindle-core/src/tensor/backend.rs` (dummy module), `crates/kindle-core/tests/*`
- Risk: Any future ONNX-import test or compile-time shape-check relying on conv/pool through
  `DummyBackend` will get silently wrong shapes rather than a clear failure.
- Priority: Medium — becomes high priority the moment ONNX conv/pool shape tests are added.

**`stack_static_mismatch.rs` / `concat_static_mismatch.rs` `trybuild` fixtures test the wrong
failure mode:**
- What's not tested: Static (compile-time) shape-mismatch detection for `stack!`/`concat!`
  macros is nominally covered by these two fixtures, but since they're missing
  `use kindle_macros::s;`, they currently fail to compile because the `s!` macro isn't in
  scope — not because the shape-mismatch check triggered. The regenerated `.stderr` snapshots
  encode this wrong-reason failure as "passing."
- Files: `crates/kindle-core/tests/compile_fail/stack_static_mismatch.rs`,
  `crates/kindle-core/tests/compile_fail/concat_static_mismatch.rs`
- Risk: If the actual static-shape-mismatch check for `stack!`/`concat!` regresses (e.g. starts
  compiling successfully when it shouldn't), these tests would not catch it — they already
  "pass" for an unrelated reason.
- Priority: High — cheap fix (add one import line, re-bless snapshots with
  `TRYBUILD=overwrite`), and it's the difference between having real coverage or none for this
  check.

**No test constrains `Tensor::from_slice`'s dtype/byte-width assumption:**
- What's not tested: Nothing asserts `from_slice` rejects or correctly handles a non-`F32`
  target dtype `K`, despite the function hardcoding `KindleDType::F32` for the reinterpreted
  bytes (see Known Bugs above).
- Files: `crates/kindle-core/src/tensor/base.rs:163-177`
- Risk: Silent data corruption for any non-f32 tensor built via `from_slice`, undetected by
  the existing test suite.
- Priority: High — this is a live correctness bug, not just a missing test; fixing the bug and
  adding a regression test should happen together.

---

*Concerns audit: 2026-07-09*
