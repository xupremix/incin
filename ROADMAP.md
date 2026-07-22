# Kindle — Release Roadmap

The physical plan for scalable dtype support, reusable kernel templates,
layout specialization, fusion, autotuning, and performance gates is
documented in [DType and Kernel Specialization Architecture](docs/DTYPE_KERNEL_ARCHITECTURE.md).
The first CPU slice now has typed kernels for every float layout, normalized
and coalesced unary/binary iteration, AVX2 dense arithmetic on x86-64, measured
serial/Rayon cutoffs, parallel AVX2 chunks, and a median-sampled
dense-broadcast/reference benchmark. The same writer family is projected over
vectorizable dense broadcasts rather than generating per-operation kernels.
CUDA storage/rendering/raw launch ABI are now float-family aware internally and
all launch metadata is checked before GPU work; public CUDA support stays F32
until the remaining operation families and real-hardware tests are dtype-safe.
One centralized policy now separates storage, fill, random, pointwise,
reduction, and normalization capabilities and resolves storage, compute,
accumulator, and output dtype for CPU, CUDA, and WGPU.
CUDA pointwise dispatch now selects metadata-free contiguous and whole-scalar-
broadcast templates from the normalized plan. Aligned dense views select true
packed `half2`/`bfloat162`/`float4`/`double2` loads and stores with masked scalar
tails; unaligned dense views retain separate scalar ILP candidates, and strided
views retain the generic fallback and distinct cache identity.
CUDA reductions are also generated from the dtype policy: half formats
accumulate in F32, F64 stays F64, contiguous last-axis work uses block/warp
parallel reduction, and arbitrary views retain a checked strided fallback.
CUDA layer normalization now uses typed Welford accumulation with warp-shuffle
combination and a fused affine write; batch-normalization inference shares the
same dtype renderer and checked launch boundary.
Generated CUDA families now carry typed, schema-versioned specialization keys.
Pointwise and reduction dispatch enumerate small legal launch candidate sets
and, with `autotune`, compile legal candidates before timing, run two warmups
and seven stream-ordered CUDA-event samples per candidate, and cache the median
winner by device compute capability, canonical problem identity, and workload
bucket. Without `autotune`, cold selection remains deterministic.
Non-x86 CPU vector paths, packed CPU half lanes, arbitrary-stride SIMD,
real-GPU validation, normalization backward kernels, and measured/autotuned GPU
policy remain.

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
>
> **2026-07-21 same-day follow-up: Phase 0 fixed.** C-1, C-2, C-5, C-6, and C-7
> below are now fixed, tested, and committed (see each section for what changed
> and where the tests live). A new hygiene finding also surfaced while fixing
> these: `cargo fmt --all -- --check` fails across most of the repository
> (pre-existing, not introduced by this pass) — see Repository Hygiene below.
>
> **2026-07-21 second follow-up: C-3 partially fixed.** WGPU's autograd tape is
> now wired for `NumericOps` (add/sub/mul/div), `matmul`, scalar ops, and 9
> unary activations (relu/step/neg/abs/sqrt/exp/log/tanh/sigmoid/swish), with
> 18 new gradient tests run end-to-end against a real (software) WGPU adapter
> in this environment. `layer_norm`/`batch_norm`/`softmax` are single
> monolithic GPU kernels with no decomposed primitives to compose a backward
> from (would need new WGSL kernels — deliberately not attempted blind); conv,
> pooling, embedding, reductions, cross-entropy, and quantization are also
> **still unwired on WGPU** — see C-3's section for the exact remaining list.
>
> **2026-07-21 third follow-up: C-4 partially fixed, and rescoped.** CUDA's
> `add`/`sub`/`mul`/`div` are now gradient-wired the same way, but fixing this
> surfaced that `CudaBackendImpl` implements almost nothing else — `CreationOps`,
> `FloatOps`, `ReductionOps`, `ModuleOps`, `LossOps`, `QuantizedOps`,
> `OptimizerOps` are all empty impl blocks falling through to `Err`. This is a
> bigger, more foundational gap than "autograd disconnected" — see C-4's
> section. Unlike the WGPU fix, this environment has no CUDA hardware, so the
> CUDA changes are compile-verified only, not runtime-verified.
>
> **2026-07-22 fourth follow-up: C-3's "still NOT wired" list was stale.**
> Auditing `wgpu/backend.rs` directly (grep for `tape::push`/
> `push_unary_tape_entry`, then tracing every function that instead delegates
> to an already-wired op) found that most of the list below this note is
> **already gradient-correct on `main`**, apparently from WGPU/conv-pool
> autograd work done in a later session that never updated this file (see
> `git log -- crates/kindle-backends/src/wgpu/backend.rs`, e.g. "fix: resolve
> clippy errors in WGPU conv/pool autograd backward"). Confirmed wired, either
> directly or by composition: `mish`/`elu`/`gelu` (direct), `softmax` (composed
> from `log_softmax`'s already-wired `sub`/`exp`/`sum_keepdim`/`log`/`broadcast_as`
> chain — `max_keepdim` deliberately stays untracked in that chain, which is
> mathematically correct: softmax is invariant to the per-row constant it
> subtracts for numerical stability, so that path *should* be a stop-gradient,
> not an oversight — but it means `max_keepdim` must not be wired generically
> without special-casing this call site, or this becomes silently wrong), all
> of `TensorOps` (`matmul`, `reshape`, `transpose`, `narrow`, `broadcast_as`,
> `concat` directly; `flatten`/`squeeze` via `reshape`; `broadcast_left` via
> `broadcast_as`; `slice` via `narrow`; `stack` via `reshape`+`concat`),
> `embedding`, `sum_all`/`mean_all`/`sum_dim`/`sum_keepdim`/`mean_dim`/
> `mean_keepdim`, and `conv1d`/`conv2d`/`conv_transpose2d`. Verified with
> existing tests in `wgpu/tests.rs` (e.g. `softmax_backward_is_tape_tracked`,
> `gelu_backward_matches_derivative`) already passing against the real
> software WGPU adapter in this environment — not just re-reading the code.
> **Still genuinely unwired**, confirmed by the same audit: `layer_norm`,
> `batch_norm`, `adaptive_avg_pool2d`/`avg_pool2d`/`max_pool2d`,
> `cross_entropy_loss`, `quantize`/`dequantize`/`quantized_matmul`, and the
> max/min-family reductions (`max_all`/`min_all`/`max_dim`/`max_keepdim`/
> `min_dim`/`min_keepdim`/`argmax`/`argmin`/`topk`/`argsort`) — this is the
> real remaining list, not the one below. `layer_norm`/`batch_norm` remain the
> next priority per the original note.
>
> **2026-07-22 fifth follow-up: `layer_norm`/`batch_norm` were already
> correct too, just untested.** Same story as `softmax` above: both forward
> implementations are fully composed from already-wired primitives
> (`mean_keepdim`/`broadcast_as`/`sub`/`mul`/`sqrt`/`div`/`add_scalar_float`/
> `reshape`/`add`), mirroring the CPU backend's own `layer_norm_impl` (also
> composed rather than directly wired, verified there by
> `cpu::gradcheck::gradcheck`'s `layer_norm_gradcheck`). Added the WGPU
> equivalent — a small central-difference gradcheck helper local to
> `wgpu/tests.rs` (`gradcheck_wgpu`/`numerical_grad_wgpu`), used by two new
> tests, `layer_norm_backward_matches_finite_difference` and
> `batch_norm_backward_matches_finite_difference` — run against the real
> software WGPU adapter in this environment, not compile-checked: max
> absolute difference against the analytic gradient was ~1.1e-4
> (layer_norm) and ~3.2e-4 (batch_norm) at finite-difference eps=1e-3, well
> inside f32 noise. Finite-difference checking was used instead of
> hand-derived closed-form values (the pattern used elsewhere in this file)
> specifically because hand-deriving the layer/batch-norm backward formula
> to compare against would duplicate the same derivation the composed graph
> already performs. Updated remaining list: `adaptive_avg_pool2d`/
> `avg_pool2d`/`max_pool2d`, `cross_entropy_loss`,
> `quantize`/`dequantize`/`quantized_matmul`, and the max/min-family
> reductions are what's actually left on WGPU now — pooling is the next
> priority (also real-hardware-verifiable here, same pattern).
>
> **2026-07-22 sixth follow-up: pooling wired.** Unlike softmax/layer_norm/
> batch_norm, `pool2d`'s forward (`shaders/pool2d.wgsl`) is a genuine
> monolithic kernel with nothing to compose a backward from, so this needed
> real new code, not just wiring. Followed the pattern `conv2d`'s backward
> already established on this backend (readback to a flat host `Vec`,
> compute with plain Rust loops, upload the result) and ported the exact
> algorithms the CPU backend already proves correct
> (`max_window_2d`/`scatter_pool_grad_2d`/`avg_pool2d_impl`/
> `adaptive_avg_pool2d_impl` in `cpu/ops/pool.rs`) after confirming the WGSL
> forward's algorithm matches CPU's exactly. 4 new gradcheck tests (disjoint
> and overlapping-window `avg_pool2d`, `max_pool2d`, uneven-window
> `adaptive_avg_pool2d`) pass against the real software WGPU adapter.
> Remaining: `cross_entropy_loss`, `quantize`/`dequantize`/`quantized_matmul`,
> and the max/min-family reductions. `cross_entropy_loss` is next — it's
> almost certainly composable from already-wired primitives (softmax's
> `log_softmax` helper is already `pub(crate)` and reusable), unlike pooling.

---

## Current State

| Crate | Tests | Status |
|-------|-------|--------|
| `kindle-core` | Passing | ✅ C-6/C-7 fixed; C-3 (autograd design gaps) still open |
| `kindle-backends` (cpu) | Passing, +2 new regression tests | ✅ C-2 (f32 downcast), C-5 (overflow), C-8 (mis-gated `elementwise` module — CPU couldn't build standalone) fixed |
| `kindle-backends` (cuda) | Compiles (no GPU in this env to run it) | ✅ C-1 fixed; ⚠ C-4: add/sub/mul/div now gradient-wired (unverified on real hardware); everything else (`CreationOps`/`FloatOps`/norm/embedding/quant/reduce/loss) is still an empty trait impl falling to `Err` |
| `kindle-backends` (wgpu) | Passing, +8 tests | ✅ C-9 (embedding/cross_entropy index bit-reinterpret) fixed; ⚠ C-3: elementwise/activation/`matmul`/`TensorOps`/`embedding`/`conv*`/sum-reductions/`layer_norm`/`batch_norm`/pooling autograd all wired and tested against a real adapter (see 2026-07-22 follow-ups); `cross_entropy_loss`/quantization/max-min-reductions still ungradiented |
| `kindle-backends` (legacy: candle only now) | Partial | ✅ `ndarray`/`burn` backends + deps deleted (2026-07-21, both were permanently dead code); only `CandleBackend` remains |
| `kindle-macros` | Passing | ✅ Solid — hygiene good, doc examples present |
| `kindle-data` | 9 tests | ✅ `DataLoader` tested (incl. multi-worker concurrency); `default-features = false` fixed on its `kindle-backends` dep (was leaking `cuda`/`wgpu`, see C-8) |
| `kindle` (facade) | Passing | ⚠ API surface leaks internals via wildcard re-exports |
| `kindle-viz` / `kindle-telemetry` / `kindle-viz-plugin-api` | Passing | 🔲 Prototype, correctly marked `publish = false` |

`kindle-native` and `kindle-wgpu` **no longer exist as separate crates** — CPU,
CUDA, and WGPU execution all live under `crates/kindle-backends/src/{cpu,cuda,wgpu}`
now. Anything below referencing those old crate names has been corrected.

---

## Critical — release blockers, found in this audit

These are silent-wrong-answer or guaranteed-panic bugs in the numeric core, not
style issues. None of them were on the previous roadmap.

### C-1 — Every CUDA op panics on its first invocation — ✅ FIXED (2026-07-21)
`crates/kindle-backends/src/cuda/ops/{elementwise,norm,embedding,quant,reduce,shape,loss}.rs`
(e.g. `elementwise.rs:132-134`). The pattern
`let mut x = out_b.data.clone(); Arc::get_mut(&mut x).unwrap();` required the
`Arc` refcount to be exactly 1, but `out_b` (holding the other strong reference)
was still alive at that point — refcount was always 2, so `Arc::get_mut` always
returned `None` and the `.unwrap()` always panicked. This was every add/sub/mul/div,
layer/batch norm, embedding forward+backward, quantize/dequantize, reduce/argmax,
concat, and cross-entropy on the CUDA backend. Almost certainly never exercised
in CI (no GPU runner), which is why it shipped silently.
**Fixed:** removed the unnecessary `.clone()` in all 13 call sites across 7
files — `out_b.data` (etc.) is freshly allocated immediately before each of
these blocks, so it's already uniquely owned (refcount 1); `Arc::get_mut`
now operates on it directly instead of on a doomed clone. Verified with
`cargo check -p kindle-backends --features cuda` (compiles clean) and the
existing CPU/WGPU test suite (no regressions); actual GPU execution couldn't
be runtime-tested in this environment (no CUDA hardware/toolkit available) —
**still needs a real-GPU CI job or manual verification before shipping.**

### C-2 — CPU elementwise ops silently downcast every dtype to f32 — ✅ FIXED (2026-07-21)
`crates/kindle-backends/src/cpu/ops/elementwise.rs:85-88,108-111`. Regardless of
the operands' actual `DTypeId` (F64, F16, BF16, ...), the result was always
constructed as `CpuBuffer::F32(out)`. Every add/sub/mul/div/relu/exp/log/tanh/
sigmoid/gelu/softmax on a non-f32 tensor silently lost precision with no error.
This was the exact anti-pattern the old roadmap accused the legacy Candle/Ndarray
backends of (see B-4 below, which *was* fixed there) — except it was real and
unflagged in the backend meant to replace that legacy code.
**Fixed:** added `CpuBuffer::from_f64_values` (`cpu/storage.rs`), which builds
a buffer matching the *input's* actual dtype variant instead of hardcoding F32;
`elementwise_binary`/`elementwise_unary` and three backward closures
(`mul_scalar_float`, `step`, `swish`) now use it. Added two regression tests
(`add_preserves_f64_dtype_and_precision`, `relu_preserves_f64_dtype` in
`cpu/ops/elementwise.rs`) using values exactly representable in f64 but not in
f32, so a silent f32 round-trip would fail them. Full workspace test suite
passes (285 tests in `kindle-backends`, up from 281).

### C-3 — WGPU autograd silently produces no gradients — ⚠ PARTIALLY FIXED (2026-07-21)
`crates/kindle-backends/src/wgpu/tape.rs` + `wgpu/backend.rs:91,96`. `backward()`
was wired to `tape::backward`, but `tape::push` (`tape.rs:26`) had **zero call
sites** anywhere under `wgpu/*.rs`. The tape was always empty, so `backward()`
only ever returned a gradient for the loss node itself — no parameter tensor
ever received a gradient on the WGPU backend. `unbroadcast`/`sum_dim_squeeze`/
`sum_dim_keepdim` (`tape.rs:137,160,167`) were correctly implemented and
*needed* — they were just never reached because no forward op recorded a
`TapeEntry`. Training on this backend used to run, converge toward nothing,
and report no error.

**Fixed for:** `NumericOps` (`add`/`sub`/`mul`/`div`, matching CPU's math
exactly, using `unbroadcast` at every input — currently a no-op since WGPU's
`binary_op` requires equal shapes and has no broadcasting yet, but wired for
when it does), `FloatOps` scalar ops (`add_scalar_float`, `mul_scalar_float`)
and unary activations (`relu`, `step`, `neg`, `abs`, `sqrt`, `exp`, `log`,
`tanh`, `sigmoid`, `swish`) — all composed from the existing
`binary_op`/`unary_op`/`scalar_op` GPU dispatch helpers already in
`wgpu/backend.rs`, no new WGSL kernels needed. 16 new tests in `wgpu/tests.rs`
verify each op's gradient against hand-derived expected values (e.g.
`div_backward_matches_quotient_rule`, `sigmoid_backward_matches_out_times_one_minus_out`),
plus `chained_ops_accumulate_gradient_through_multiple_hops`, which composes
`mul`+`add`+`relu` with a tensor reused by two ops and asserts the gradient
contributions are *summed*, not overwritten — the CPUBACK-05 correctness class
of bug this same tape design already guards against on the CPU backend. All
298 `kindle-backends` `--features wgpu` tests pass (up from 283, actually run
end-to-end against a software WGPU adapter in this environment, not just
compile-checked).

**Still NOT wired** (forward runs, but produces no gradient, same as before):
`mish`, `elu`, `gelu` (derivatives need a composition or kernel this pass
didn't build — `gelu` in particular needs `erf`, which has no GPU primitive
here), `softmax` (a monolithic kernel, not decomposed like CPU's
`exp(log_softmax(x))`, so needs its own dedicated backward), and everything in
`TensorOps`/`ReductionOps`/`ModuleOps`/`LossOps`/`QuantizedOps` — `matmul`,
`reshape`/`transpose`/`narrow`/`stack`/`concat`, all reductions
(`sum`/`mean`/`max`/`min`/`argmax`/`topk`/...), `embedding`, `layer_norm`,
`batch_norm`, pooling, `conv1d`/`conv2d`/`conv_transpose2d`,
`cross_entropy_loss`, and quantization. **A real model's forward pass
(matmul + conv + norm layers) still won't get gradients on WGPU** — this fix
covers elementwise arithmetic and common activations only. The remaining ops
need the same treatment; prioritize `matmul` and `layer_norm`/`batch_norm`
next since those are on the critical path for any real network.

### C-4 — CUDA autograd is fully disconnected — ⚠ PARTIALLY FIXED (2026-07-21), and the backend itself is far less complete than C-4 originally described
`crates/kindle-backends/src/cuda/backend.rs`: `type Grads` was `()`, `backward()`
unconditionally returned `Ok(())`, `get_grad()` always returned `Ok(None)`.
`cuda/tape.rs`'s `push`/`backward`/`unbroadcast`/`sum_dim_*` had zero callers
anywhere in `cuda/`.

**Correction, found while fixing this:** the original C-4 write-up undersold
the actual gap. `CudaBackendImpl`'s trait impls in `cuda/backend.rs` were, before
this fix, almost entirely **empty blocks relying on `Backend` trait defaults**
(which mostly return `Err(UnsupportedBackendOperation)`):
`impl FloatOps<Self> for CudaBackendImpl<T, D> {}`,
`impl CreationOps<Self> for CudaBackendImpl<T, D> {}`,
`impl ReductionOps<Self> for CudaBackendImpl<T, D> {}`,
`impl QuantizedOps<Self> for CudaBackendImpl<T, D> {}`,
`impl OptimizerOps<Self> for CudaBackendImpl<T, D> {}`,
`impl ModuleOps<Self> for CudaBackendImpl<T, D> {}`,
`impl LossOps<Self> for CudaBackendImpl<T, D> {}` — every one of these is still an
empty `{}` today. Only `TensorOps::concat` and `NumericOps` (`add`/`sub`/`mul`/`div`)
were actually implemented. This means **you cannot even create a CUDA tensor**
via the standard `zeros`/`ones`/`rand`/`randn` API (`CreationOps` has no
overrides, so every call falls through to the default `Err`) — despite the
extensive, mostly-working `cuda/ops/{norm,embedding,quant,reduce,loss}.rs`
dispatch functions existing and (since C-1) being individually correct: they
are **dead code**, never called from anywhere in `cuda/backend.rs`. The
"autograd disconnected" framing undersold this — the more accurate framing is
"the CUDA backend implements a small elementwise-arithmetic slice of the full
`Backend` trait surface, and everything outside that slice doesn't exist yet,"
not "everything exists but doesn't compute gradients."

**Fixed:** wired `tape::push` into the 4 `NumericOps` methods that actually
exist (`add`/`sub`/`mul`/`div`), mirroring C-3's WGPU fix exactly (same math,
same `unbroadcast` helper, already present unused in `cuda/tape.rs`). Wired
`Backend::backward`/`backward_with_nan_check`/`get_grad` to the real
`cuda::tape` functions instead of the `()`/`Ok(None)` placeholders, and
changed `type Grads` from `()` to `crate::cuda::tape::CudaGrads`. Verified via
`cargo check`/`cargo test --no-run -p kindle-backends --features cuda`
(compiles clean, test binaries build) — **not runtime-verified**, unlike C-3's
WGPU fix: this environment has no CUDA hardware/toolkit, so unlike WGPU (which
ran against a real software adapter), these 4 ops' gradients have not been
executed even once. Treat this as "should be correct by construction, mirrors
an already-verified pattern" rather than "verified" until someone runs it on
real hardware.

**Deliberately NOT attempted in this pass:** building out `CreationOps` (so
tensors can even be created), `FloatOps` (activations), or wiring up the
existing `norm`/`embedding`/`quant`/`reduce`/`loss` dispatch code into the
trait at all. Doing that safely requires either CUDA hardware to verify
against, or extreme care given this exact audit's whole finding is "silently
wrong numbers are worse than an honest `Err`." Do this as a dedicated,
hardware-verified follow-up, not blind.

### C-5 — Unchecked shape-multiplication overflow feeds allocation and stride math — ✅ FIXED (2026-07-21)
`crates/kindle-backends/src/cpu/stride.rs:17-19` (`contiguous_strides`, plain
`*`) and `cpu/creation.rs:70,81,98,127` (`shape.iter().product()` for `Vec`
length). No overflow guard on user-supplied shapes. In release builds (integer
overflow checks are off by default) a crafted or accidentally-huge shape could
wrap the element count to a small number, undersizing the backing `Vec`, while
strides computed from the same unchecked shape were later used to index into it
— a path to out-of-bounds read/write, not just a panic.
**Fixed:** `contiguous_strides` now uses `checked_mul` and panics with a clear
message on overflow (kept infallible rather than changing its signature — it's
called from dozens of sites across the codebase, all currently assuming
`Vec<usize>` back, not `Result`); added `cpu::stride::checked_numel` (via
`try_fold`/`checked_mul`, returns `Result<usize>`) and switched `zeros`/`ones`/
`rand`/`randn` in `cpu/creation.rs` to it, since those are already
`Result`-returning and can propagate a real `Err` instead of panicking.

### C-6 — kindle-core's dynamic-shape broadcast doesn't actually verify anything — ✅ FIXED (2026-07-21), impact was smaller than first assessed
`crates/kindle-core/src/shapes/broadcast.rs:329-420` (all `(usize,)` /
`(usize, B)` / `(usize, B, C)` dynamic-dim impls). `output_shape` computed
`lhs.0.max(rhs.0)` with **no check** that the two dims are equal or that one is
1. **Correction after tracing every call site:** this is *not* reachable as a
live "silently accepts and computes wrong data" bug via the public `Tensor`
API. The only caller (`impl_broadcast_binary_op!`/`impl_std_ops!` in
`tensor/ops/binary.rs`) always computes this value, then calls the backend's
`add`/`sub`/`mul`/`div`, which independently calls the already-correctly-validated
`kindle_backends::cpu::stride::broadcast_shape` (returns `Err` on real
mismatch, with its own passing tests) and propagates its error via `?` *before*
the fabricated shape is ever used to build a `Tensor`. So today, a genuine
shape mismatch already errors out correctly through that separate path. It's
still a real gap for defense-in-depth (a hypothetical future direct caller of
`BroadcastShape::output_shape` would get silently wrong data) and for
`Tensor::shape()` introspection semantics.
**Fixed:** added `checked_broadcast_dim(lhs, rhs)` (asserts `lhs == rhs || lhs
== 1 || rhs == 1`, same NumPy/PyTorch rule already enforced elsewhere) and
used it at all 4 call sites (the `(usize,)`/`(usize,B)`/`(usize,B,C)`/
`(usize,B,C,D)` impls). Kept as a panic rather than a `Result` — changing
`BroadcastShape::output_shape`'s signature would ripple through ~50 other
(fully static, always-compatible-by-construction) impls in the same file for
no behavior change on the only real call site.

### C-7 — Arithmetic operators panic instead of propagating errors — ✅ FIXED (2026-07-21)
`crates/kindle-core/src/tensor/ops/binary.rs:196,220,243,266` and the scalar
variants in `unary.rs`. `impl core::ops::Add/Sub/Mul/Div for Tensor` called
`self.backend_method(&rhs).unwrap()` with a bare `.unwrap()` and no context.
Since C-6's fix confirms shape mismatches already error out cleanly before
reaching this `.unwrap()`, this item was purely about panic-message quality:
whatever *does* fail here fails behind an opaque `unwrap()`.
**Fixed:** replaced every such `.unwrap()` (4 in the `impl_std_ops!` macro
body in `binary.rs`, 2 in the `impl_std_scalar_ops!` macro body in `unary.rs`)
with `.unwrap_or_else(|e| panic!(...))` messages naming the operator and
printing the underlying `Error`, so a real failure is immediately debuggable
instead of a bare "called `Result::unwrap()` on an `Err` value" with no context.

---

### C-8 — `cpu::ops::elementwise` was `#[cfg(feature = "cuda")]`-gated; CI's "cpu-only" test run silently never tested cpu-only — ✅ FIXED (2026-07-21)
Two compounding bugs, both found while wiring up CI (item 12):
1. `crates/kindle-backends/src/cpu/ops/mod.rs` had `pub mod elementwise;` — the
   module implementing `NumericOps`/`FloatOps` for `CpuBackendImpl`, i.e. the
   entire reason the CPU backend can add/mul/relu/etc. — preceded by *seven*
   duplicate `#[cfg(feature = "cuda")]` attributes. With `cuda` off,
   `cpu::ops::elementwise` didn't exist at all and `CpuBackendImpl` failed to
   implement `NumericOps`/`FloatOps`, i.e. **the CPU backend could not compile
   standalone**.
2. This was invisible because it never *was* standalone: `kindle-data/Cargo.toml`
   depended on `kindle-backends` without `default-features = false`, so every
   `cargo test --workspace ...` — including CI's own
   `--no-default-features --features kindle-backends/cpu,kindle/cpu` — silently
   re-enabled `kindle-backends`' full default feature set (`std, cpu, wgpu,
   cuda`) through that one crate, masking bug (1) for the whole session and,
   presumably, since whenever these two files were introduced. Two example
   crates (`tui_graph_demo`, `native_training_demo`) had the same leak pattern
   (needed `wgpu` legitimately, pulled unwanted `cuda` too).
**Fixed:** removed the 7 stray `#[cfg(feature = "cuda")]` attributes; added
`default-features = false` to all three leaking `Cargo.toml`s, keeping only
the features each crate actually needs. Re-verified: `cargo check -p
kindle-backends --no-default-features --features cpu,std` now compiles clean,
and the full workspace test suite (48 suites, 0 failed) still passes under
the *real* cpu-only feature set — this is the first time in the audit this
was actually verified rather than assumed. This is the same class of bug as
C-1/C-3/C-4 (a real defect with zero automated coverage), found by the same
root cause the roadmap already flagged: no CI job ever verified a feature
combination actually meant what its name said.

### C-9 — WGPU `embedding` backward and `cross_entropy_loss` bit-reinterpreted index storage instead of converting it — ✅ FIXED (2026-07-22)
`crates/kindle-backends/src/wgpu/backend.rs`: `embedding`'s backward closure
did `indices_capture.buffer.to_vec::<u32>()`, and `cross_entropy_loss`'s
one-hot construction did `target.buffer.to_vec::<u32>()`. WGPU has no genuine
integer storage — index/target tensors are physically F32 bytes, and the
embedding WGSL kernel (`shaders/embedding.wgsl`) proves the correct read:
it declares `indices: array<f32>` and does `u32(indices[i])`, a real value
conversion. `to_vec::<u32>()` instead bit-reinterprets those same bytes via
`bytemuck` — which only happens to produce the right answer for index/class
`0.0` (IEEE bit pattern `0x00000000`, coincidentally also integer `0`). Any
other value (`1.0f32` = bit pattern `1065353216`) reads back as a huge
garbage integer, fails every `idx < vocab_size`/`class_idx < classes` bounds
check, and silently drops that row's entire gradient/loss contribution
instead of erroring — the exact "silently wrong answer" class of bug this
roadmap treats as a release blocker, not a missing feature. Found while
auditing `cross_entropy_loss` for autograd wiring (see C-3's 2026-07-22
follow-ups above), not by suspicion of this specific bug.
**Existing tests never caught this:** `embedding_backward_accumulates_gradients`
only used index `0.0` for both positions; `test_cross_entropy_mean` only
asserted loose bounds (`loss > 0.0 && loss < 5.0`) despite already using
target class `1` in one row.
**Fixed:** both sites now read `to_vec::<f32>()` and convert the value
(`idx as usize`), matching the WGSL forward exactly. Verified the bug was
real, not theoretical, by temporarily reverting the fix and confirming two
new regression tests fail without it — `embedding_backward_handles_nonzero_indices`
(wrong gradient rows) and
`cross_entropy_loss_matches_hand_computed_value_for_nonzero_target` (0.157
instead of the hand-computed 0.196) — before restoring the fix and
confirming both pass. Full `--features wgpu,std` suite: 88/88 passing,
actually run against the real software WGPU adapter in this environment.

---

## High priority — real gaps, not yet release-blocking on their own

- **`kindle-data` has zero tests — ✅ FIXED (2026-07-21).** Added 9 tests in
  `loader.rs` covering single-threaded ordering, exact/short final batches,
  empty dataset, batch size larger than the dataset, shuffle (verifies the
  full item *set* is preserved, just reordered), and — the actual
  concurrency-risk cases — multi-worker runs (with and without shuffle) that
  assert every item is yielded **exactly once, no duplicates, no drops**
  across 8 worker threads pulling from the shared `Mutex<Iterator>`, plus a
  more-workers-than-batches edge case and a real (sum, not passthrough)
  `Collate` impl to prove the collate function actually runs. Every test goes
  through a 10s `recv_timeout` wrapper so a deadlock/hang fails loudly and
  fast instead of stalling the suite. Ran 5x in a row with no flakiness.
- **`legacy::burn_backend` AND `legacy::ndarray_backend` were both permanently dead code — ✅ DELETED (2026-07-21).**
  The original finding only caught `burn_backend` (`legacy/mod.rs:2138`,
  `#[cfg(any())]` — always false, unreachable under any feature combination).
  Re-checking while fixing it found `ndarray_backend` (`legacy/mod.rs:1180`,
  containing the actual `NdarrayBackend` struct) was gated the **exact same
  way** — meaning the "`NdarrayBackend` is ~61% stubbed" finding below was
  analyzing dead code too; it was never reachable under any feature flag,
  not just incomplete. `burn_backend` additionally implemented a
  `Backend<(D,)>`/`RawTensor` shape from an old API that doesn't match the
  current `kindle-core::Backend` trait at all, so it couldn't have compiled
  even if unlocked. **Fix:** deleted both modules outright (~1,200 lines) —
  dead code with no realistic path back to working, not worth a rewrite
  ticket. Removed the now-unused `ndarray`/`burn` optional dependencies from
  `Cargo.toml` and `legacy = [...]`'s feature list; confirmed via
  `cargo tree -p kindle-backends --features legacy` that neither appears in
  the dependency tree anymore (`Cargo.lock` shrank by ~4,400 lines). Only
  `legacy::candle` (the real, live `CandleBackend`) remains. Verified
  `cargo check`/`cargo test -p kindle-backends --features legacy` and the
  full workspace suite all still pass.
- **`pub` API leakage across backends — ✅ FIXED (2026-07-21).** `wgpu` was
  already correctly scoped. Brought `cuda` up to the same standard:
  `cuda/mod.rs`'s `ops`/`storage`/`gpu`/`tape` submodules are now `pub(crate)`
  (only `CudaBackendImpl`/`CudaGrads`/`CudaVar` are re-exported, mirroring wgpu's
  pattern exactly, including a new `pub type CudaGrads = crate::cuda::tape::CudaGrads;`
  alias in `backend.rs` so the type stays externally nameable through the
  `pub(crate)`-module `tape`); `CudaBuffer`'s raw `Arc<CudaSlice<u8>>`/
  `Arc<CudaContext>` fields, `CudaStorage`'s fields, `TapeEntry`'s fields, and
  all `launch_*` dispatch functions are now `pub(crate)`. `CpuGrads`'s
  `grads` field (previously `pub`) and `WgpuGrads`'s (same issue, not
  previously flagged) are now `pub(crate)`, each with a `.get(id)` accessor
  method instead (added one for `WgpuGrads`/`CudaGrads`; `CpuGrads` already
  had one). `kindle-core`'s `tensor::tracing::TRACING_GRAPH` static is now
  `pub(crate)` — it turned out to be a genuine (if leaky) cross-crate
  integration point, not just an oversight: `kindle-backends`' `cpu/tape.rs`
  and `wgpu/tape.rs` read it for telemetry snapshots, and the
  `tui_graph_demo`/`onnx_export` examples wrote to it directly via
  `.lock().mark_input(...)`. Added three narrow public functions instead —
  `extract_graph()` (already existed), `tracing_graph_snapshot()` (new,
  non-destructive clone for telemetry), `tracing_mark_input()`/
  `tracing_mark_output()` (new) — and updated all four call sites to use
  them instead of reaching into the raw `Mutex<Graph>`. `CpuBuffer` being
  `pub` was **not** changed — re-reading `cpu/mod.rs`'s own module doc, this
  one is explicitly documented as intentional ("`CpuBuffer` for
  pattern-matching in `to_bytes`/`from_bytes`"), and its variants are plain
  data containers with no invariant an external `CpuBuffer::F32(vec![...])`
  construction could violate — unlike the raw GPU-context/slice handles this
  fix targeted. Full workspace test suite + `cargo build --examples
  --workspace` both pass with zero regressions.
- **Facade (`kindle`) blanket-exports internals via wildcard globs — ✅ FIXED (2026-07-21) for `kindle_macros::*`.**
  `crates/kindle/src/lib.rs`'s `pub mod macros` and `prelude` module both used
  `kindle_macros::{idx, impl_arg_into, s}` / `kindle_macros::*`, which pulled in
  `generate_shape_ops` and `impl_arg_into` — internal codegen helpers invoked
  only by `kindle-core` itself (`kindle_macros::generate_shape_ops!()` in
  `shapes/shape_ops.rs`, `kindle_macros::impl_arg_into!(7)` in
  `tensor/arg_into.rs`; confirmed via grep that no end-user code calls either).
  Both are now explicit lists (`{idx, s}` and `{idx, import_model, module, s}`)
  with neither leaked symbol included. **`pub use kindle_backends::*;`
  (crate root) deliberately left as-is** — narrowing it risks breaking
  external consumers I can't fully enumerate in this pass, and unlike the
  macros case, everything it pulls in (`cpu`/`cuda`/`wgpu` modules) is already
  intentionally `pub` within `kindle-backends` itself; this is redundant
  multi-path exposure of already-public items, not the same class of leak as
  a genuinely-internal symbol escaping. Worth a dedicated follow-up.
- **`DefaultBackend` silently degrades to `()` when `cpu` is disabled — ✅ FIXED (2026-07-21).**
  Removed the `()` fallback entirely. `Tensor` already had the correct fix
  (no `B` default at all when `cpu` is off, forcing an explicit, immediate
  error at the actual call site) — applied the same split-by-`cfg` pattern to
  the 9 other `B`-taking aliases (`Linear`, `Conv1d`, `Conv2d`, `BatchNorm2d`,
  `LayerNorm`, `Param`, `RNNCell`, `RNN`, `Embedding`), which previously all
  still defaulted to the broken `()`. Verified both
  `cargo check -p kindle --features cpu` and `--no-default-features` (cpu off)
  compile clean, and the full workspace test suite still passes.
- **Dead `#[cfg(feature = "candle")]` block in the facade — ✅ FIXED (2026-07-21).**
  The crate's top-doc comment (`lib.rs:8`) claiming to wrap Candle "out of the
  box" was corrected (matches the README fix from earlier in this audit); the
  dead `#[cfg(feature = "candle")]` test block in `test_tensor_export` (which
  never ran, since `candle` isn't a real feature) was replaced with a real,
  `#[cfg(feature = "cpu")]`-gated assertion that actually exercises
  `Tensor`/`DefaultBackend` end-to-end — confirmed it now runs and passes
  (`cargo test -p kindle --features cpu`).
- **CUDA autotuning — 🟡 FOUNDATION IMPLEMENTED, coordinator wired (2026-07-22).**
  The `autotune` feature now provides typed canonical problem keys,
  pointwise/reduction launch candidates, synchronized-median winner selection,
  and a bounded device/shape cache consumed by dispatch. Cold-cache selection
  is deliberately deterministic and is never stored as benchmark evidence.
  Pointwise and reduction cold buckets now perform CUDA-event measurement
  after JIT and use compute-capability identity. Concurrent first-use
  (in-flight) suppression is now wired into both CUDA pointwise and reduction
  dispatch (`tuning::claim_tuning`/`TuningPermit`) — a coordinator that
  existed, fully unit-tested, but was never called from any dispatch path
  before this session; racing callers for the same key now block on the
  in-progress measurement instead of redundantly benchmarking it. Pointwise
  candidates also get Tier-2 occupancy pruning
  (`cuOccupancyMaxActiveBlocksPerMultiprocessor`), conservatively dropping
  only candidates the driver confirms have zero active blocks and never
  narrowing the legal set to zero. Reduction occupancy pruning, richer
  device/compiler identity, persistence, and telemetry remain upcoming work;
  reduction pruning specifically needs `reduce.rs`'s launch-selection flow
  restructured so the compiled `CudaFunction` is available at candidate-
  selection time (block size there is a launch parameter, not a
  compiled-kernel axis, unlike pointwise). All of this is compile/clippy-
  verified only — still no CUDA hardware in this environment to runtime-verify
  against.

---

## Medium priority

- **Unsound raw-byte reinterpretation**: `to_scalar<E: Copy>`/`to_vec1<E: Copy>`
  in `kindle-core/src/tensor/ops/manipulation.rs:362,370,387-393` only check
  byte-length equality against `size_of::<E>()`, never cross-check the tensor's
  actual `DTypeId` against `E` before `read_unaligned`/`copy_nonoverlapping`.
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
- **CUDA kernel templates accept trusted internal scalar expressions.** The
  templates and dtype policy now live in `kernel.rs`, validate operation
  identifiers, specialize float storage/compute conversions, and use distinct
  dtype cache keys. Expressions still come only from internal literals. Before
  supporting user-defined expressions, replace free-form source fragments with
  a closed scalar IR and renderer so source injection is structurally
  impossible.
- **`PROJECT_MEMORY.md` describes `KindleBackend<T, D>` as an already-unified
  backend struct** ("Instead of separate backend structs, kindle uses a single
  unified `KindleBackend<T, D>`..."), but its own status checklist lists this as
  unchecked, and the audit confirms the actual code still has separate
  `CpuBackendImpl`/`CudaBackendImpl`/`WgpuBackendImpl`/`CandleBackend`/`NdarrayBackend`
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
8. ✅ DONE (2026-07-21): closed B-3 for real — `cuda` and `cpu` modules brought
   to the same `pub(crate)` discipline `wgpu` already has; removed the
   `kindle-core` `TRACING_GRAPH` leak (now `pub(crate)`, with 3 narrow public
   functions replacing direct `Mutex<Graph>` access). Still worth a follow-up
   design question: is a process-wide singleton the right shape at all vs. a
   graph handle threaded through the API? Not changed in this pass — only the
   encapsulation, not the underlying design.
9. Audit `kindle`'s wildcard re-exports (`pub use kindle_backends::*` /
   `kindle_macros::*`) down to an explicit allowlist.
10. Fix or remove `DefaultBackend = ()` fallback; fix or delete the dead
    `#[cfg(feature = "candle")]` block.
11. Add `#[non_exhaustive]` / doc-coverage items already tracked below.

**Phase 3 — test debt**
12. ✅ DONE (2026-07-21): wrote tests for `kindle-data`'s `DataLoader`
    (highest-risk untested code) — 9 tests incl. multi-worker concurrency.
13. ✅ DONE (2026-07-21): deleted `legacy::burn_backend` AND
    `legacy::ndarray_backend` (both permanently dead code, `#[cfg(any())]`)
    and removed the now-unused `burn`/`ndarray` dependencies entirely.
14. ✅ DONE (2026-07-21): repo-wide `cargo fmt --all` pass (237 files) as its
    own change; found and fixed C-8 (see above) while getting CI's cpu-only
    test command to actually mean what it said; brought `cargo clippy
    --workspace --all-targets --features kindle-backends/cpu,kindle/cpu -- -D
    warnings` from ~120 real errors (after subtracting C-8's false-positive
    noise) to a clean pass — `cargo fmt --all -- --check` and this clippy
    invocation are now both genuinely green, not just untested.
15. Make `s![]`/`idx![]` doctests real (`rust,ignore` → compiled).

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
| `Tensor<S, B, K, G>` type and its inherent methods | `kindle-core` | ✅ Yes |
| `s![]`, `idx![]` macros | `kindle-macros` | ✅ Yes |
| `CpuBackendImpl<T>`, `CudaBackendImpl<T>`, `WgpuBackendImpl<T>` | `kindle-backends` | ✅ Yes (the structs only) |
| `Error` enum variants | `kindle-core` | ✅ Yes (`#[non_exhaustive]` — already applied) |
| `nn::Linear`, `Conv2d`, `LayerNorm`, etc. | `kindle-core` | ✅ Yes |
| `dispatch_*` / `launch_*` functions | `kindle-backends` (cuda, wgpu) | ❌ No — `pub(crate)` everywhere ✅ (2026-07-21) |
| `CudaBuffer`, `WgpuBuffer`/`WgpuStorage` fields | `kindle-backends` | ❌ No — `pub(crate)` ✅ (2026-07-21). `CpuBuffer` itself stays `pub` (intentional, see High priority section) |
| `TRACING_GRAPH` static | `kindle-core` | ❌ No — `pub(crate)` ✅ (2026-07-21) |
| Internal modules (`tape`, `stride`, `creation`, `ops`, `gpu`) | `kindle-backends` | ❌ No — `pub(crate)` |

### `#[non_exhaustive]` status
- `Error` enum — ✅ already applied (`kindle-core/src/err.rs:8`)
- `DTypeId` enum — ✅ already applied (`kindle-core/src/tensor/dtype.rs:43`)
- `DeviceId` struct — check and apply if missing

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
| Concurrency tests | ✅ `kindle-data` covered (2026-07-21) | Extend the same pattern to any future concurrent code |

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
- [ ] **New finding (2026-07-21):** `cargo fmt --all -- --check` fails across most of the repository (e.g. `tensor/backend.rs`, `wgpu/backend.rs`, `wgpu/dispatch.rs` have hundreds of formatting diffs) — pre-existing, not introduced by this session's fixes. Since CI (`.github/workflows/ci.yml`) runs `cargo fmt --all -- --check` as its first step, **CI would fail immediately on `main` today**, before ever reaching the test suite. Files touched in this session's fixes were reformatted with `rustfmt` and are clean; the rest of the repo needs a dedicated `cargo fmt --all` pass (deliberately NOT done in this session — it would touch nearly every file and bury the semantic fixes in formatting noise). Do this as its own isolated commit before relying on CI.
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

1. ~~**C-1** Fix CUDA `Arc::get_mut` panic (every CUDA op)~~ — ✅ fixed 2026-07-21, needs real-GPU verification
2. ~~**C-2** Fix CPU elementwise f32 downcast (dtype correctness)~~ — ✅ fixed 2026-07-21, regression-tested
3. ~~**C-6 / C-7** Fix dynamic broadcast validation, then the operator-overload panics it was masking~~ — ✅ fixed 2026-07-21
4. ~~**C-5** Checked shape multiplication (overflow → OOB path)~~ — ✅ fixed 2026-07-21
5. **C-3** (partial ✅: elementwise + matmul + activations wired on WGPU, 18 tests, hardware-verified) / **C-4** (partial ✅: same 4 ops wired on CUDA, compile-verified only — no hardware in this env) — remaining: WGPU conv/norm/pooling/embedding/reductions/loss/quant; CUDA needs `CreationOps`/`FloatOps` built from scratch (currently empty) before any of that can even run — **next priority**
6. Add cross-backend gradient-parity tests
7. ~~Close B-3 for real: `cuda`/`cpu` `pub(crate)` audit + `kindle-core` `TRACING_GRAPH` leak~~ — ✅ fixed 2026-07-21
8. ~~Audit `kindle` facade wildcard re-exports; fix `DefaultBackend = ()` trap; remove dead `candle` cfg~~ — ✅ fixed 2026-07-21 (macros wildcard narrowed, `kindle_backends::*` deliberately left, see High priority section for why)
9. ~~Write `kindle-data` `DataLoader` tests~~ — ✅ fixed 2026-07-21 (9 tests, incl. multi-worker concurrency)
10. ~~Decide `legacy::burn_backend`'s fate; stop paying its dependency cost~~ — ✅ fixed 2026-07-21 (deleted, along with equally-dead `ndarray_backend`)
11. ~~**C-8** `cpu::ops::elementwise` mis-gated behind `#[cfg(feature = "cuda")]`; `kindle-data`/example crates leaked `kindle-backends`' default features, masking it~~ — ✅ fixed 2026-07-21 (see C-8 above) — found *while* doing item 12
12. ~~Run a dedicated `cargo fmt --all` pass (repo-wide) as its own commit, and get `cargo clippy --workspace --all-targets -- -D warnings` genuinely green~~ — ✅ fixed 2026-07-21 (237 files reformatted; ~120 real clippy errors fixed; both gates are real now, not silently red)
13. Add GPU-gated CI jobs (cuda/wgpu), and update `.github/workflows/ci.yml`'s fmt/clippy steps to match what's now actually green — the CI gap (no GPU jobs, an unverified cpu-only feature set) is why C-1/C-3/C-4/C-8 all shipped unnoticed
14. Make doc comments real (not templates); make `s![]`/`idx![]` doctests compile for real
15. Write `CHANGELOG.md`, finish remaining repo hygiene items above
