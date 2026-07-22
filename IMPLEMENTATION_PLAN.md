# Kindle — Implementation Plan (Ground-Truth Edition)

> **Ground truth as of:** commit `64e41b6` (2026-07-22 22:02), branch `develop`,
> plus two uncommitted working-tree files documented in Phase 0.
> **Written by:** Claude Code, at the user's request, specifically so that a
> second agent (Antigravity) picking this repo up cold does not hallucinate
> work items, re-invent things that already exist, or "complete" APIs that
> were deliberately left incomplete.

---

## 0. How to use this document — read this section first

This document exists because `ROADMAP.md` (1040 lines) has accumulated seven
"follow-up" corrections in a row, several of which **reversed an earlier
finding in the same file** (see its 2026-07-22 fourth/fifth/eighth
follow-ups). That pattern — stating something as fact, later discovering it
was stale or wrong — is exactly the failure mode this document is trying to
prevent from repeating. Rules:

1. **Every claim below is cited to a file path, and most to a line range.**
   Before acting on a claim, `grep`/open the cited location and confirm it
   still says what this document says it says. Code moves; this document is
   a snapshot, not a live view.
2. **If what you find contradicts this document, trust the code, not this
   document.** Update this file's relevant section with a dated correction
   (mirror `ROADMAP.md`'s "follow-up" convention — don't silently rewrite
   history, append a note explaining what changed and why the old claim was
   wrong).
3. **Do not create work that isn't in this plan or explicitly requested by
   the user.** If you think something is missing, ask, or add it as a new
   dated section — do not silently start building it.
4. **Read the "Hard DO-NOT list" (§0.3) before touching anything.** It exists
   because every item on it was either already tried and reverted, or would
   contradict an explicit, repeated user decision recorded in `ROADMAP.md`
   or this repo's git history.

### 0.1 Verification loop — run after every change, before every commit

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --no-default-features --features kindle-backends/cpu,kindle/cpu -- -D warnings
cargo test --workspace --all-targets --no-default-features --features kindle-backends/cpu,kindle/cpu
cargo build --examples --workspace --no-default-features --features kindle-backends/cpu,kindle/cpu
```

For anything touching `wgpu`, additionally run:
```bash
cargo test -p kindle-backends --no-default-features --features wgpu,std --all-targets
```

For anything touching `cuda`, you can only do this — **there is no CUDA
hardware in either Claude Code's or (per `.planning/STATE.md`) Antigravity's
environment**:
```bash
cargo check -p kindle-backends --no-default-features --features cuda,std
cargo test --no-run -p kindle-backends --no-default-features --features cuda,std
```
Never claim a CUDA change "works," "passes," or "is verified" — the accurate
phrasing, used consistently throughout `ROADMAP.md`, is **"compiles clean,
not run."** State this explicitly in every commit message touching
`cuda/*`.

### 0.2 Commit protocol (established convention, repeatedly honored)

- **Never add `Co-Authored-By: Claude` or any AI-attribution trailer to
  commits in this repo.** (User preference — see memory
  `feedback_no_ai_commit_attribution`.)
- Split commits into logical units (a correctness fix is its own commit,
  separate from a formatting/tooling pass) — see `4ad3050` +
  `f126d6b` for the precedent.
- Use `git log --oneline -20` to match this repo's message style before
  writing a new one: `type(scope): summary`, e.g.
  `feat(cuda): wire TensorOps::matmul with autograd`.
- `.planning/`, `CLAUDE.md`, `DESIGN.md`, `FUTURE_ROADMAP.md`,
  `NATIVE_BACKEND_PLAN.md`, `.claude/`, `.gemini/`, `.agents/` are all
  **gitignored by name** (see `.gitignore`). `ROADMAP.md`,
  `PROJECT_MEMORY.md`, `CHANGELOG.md`, and this file (`IMPLEMENTATION_PLAN.md`)
  are **not** gitignored — they are real, committed project documentation.
  Don't accidentally commit anything that should have stayed local, and
  don't assume this file is invisible to git the way `.planning/STATE.md` is.

### 0.3 Hard DO-NOT list

- **Do NOT create `kindle-native` or `kindle-wgpu` as separate crates.** They
  were consolidated into `kindle-backends/src/{cpu,cuda,wgpu}` — this is
  called out explicitly in `ROADMAP.md`'s 2026-07-21 header note because an
  earlier version of that document described a stale architecture. The
  workspace member list in root `Cargo.toml` is the authority on what crates
  exist.
- **Do NOT resurrect `legacy::burn_backend` or `legacy::ndarray_backend`.**
  Both were deleted 2026-07-21 as permanently-dead code (`#[cfg(any())]`-gated,
  unreachable under any feature combination, ~1,200 lines). Only
  `legacy::candle` (the real `CandleBackend`) remains under the `legacy`
  feature. Do not add the `burn`/`ndarray` dependencies back.
- **Do NOT invent `Backend` trait methods that don't exist.** In particular:
  there is **no `sgd_step` method** on `OptimizerOps` — only `adamw_step`
  (see §1.7). `SGD` and `Adam` (kindle-core's `optim` module) are generic,
  composed-from-primitives optimizers that work on any backend implementing
  `NumericOps`+`FloatOps` already; only `AdamW` calls a dedicated
  `Backend::adamw_step` fast path. Do not add a `sgd_step`/`adam_step`
  backend method — it isn't part of the design and nothing calls it.
- **Do NOT change any `Backend`-family trait signature** (`Backend`,
  `NumericOps`, `FloatOps`, `TensorOps`, `CreationOps`, `ReductionOps`,
  `ModuleOps`, `LossOps`, `QuantizedOps`, `OptimizerOps`, all in
  `kindle-core/src/tensor/backend.rs`) without flagging it explicitly as a
  **semver-breaking change** and stopping for user sign-off first. Every
  implementor (`CpuBackendImpl`, `CudaBackendImpl`, `WgpuBackendImpl`, and
  `legacy::CandleBackend`) must add the method too. See "API Stability
  Contract" in `ROADMAP.md`.
- **Do NOT mark a CUDA item "done"/"fixed"/"verified" from reading the code
  alone.** "Compiles clean" and "verified" are different claims — this
  exact conflation is what made `ROADMAP.md`'s C-4 section need two
  corrective follow-ups (2026-07-22, "CreationOps was NOT actually empty").
  Grep the actual `impl` body before asserting what it does or doesn't do —
  do not trust prose summaries, including this document's, without
  re-checking.
- **Do NOT use `--no-verify`, `-c commit.gpgsign=false`, or force-push**
  unless the user explicitly asks in that exact conversation turn.
- **Do NOT write template doc comments** ("Auto-generated documentation for
  X") for any new `pub` item — this is explicitly called out as a
  doc-coverage-lint-satisfying anti-pattern in `ROADMAP.md`'s Documentation
  Requirements section. Write one real line describing actual behavior.
- **Do NOT implement any item in §8/§9 (macro UX / new features) without a
  separate, explicit go-ahead from the user.** Those sections are proposals,
  not approved work — see their headers.

---

## 1. Ground-truth ledger

Read this table before starting any task in Phases 1+. It is the single most
important anti-hallucination artifact in this document: it separates "exists
and works," "exists but unwired," and "doesn't exist at all" per method, per
backend, with exact citations.

### 1.1 `TensorOps` (`kindle-core/src/tensor/backend.rs:357-509`)

| Method | CPU | WGPU | CUDA |
|---|---|---|---|
| `reshape` | ✅ `cpu/storage.rs:258` + `cpu/ops/shape_ops.rs:25` | ✅ `wgpu/backend.rs:826` | ✅ `cuda/backend.rs` (2026-07-22, metadata-only) |
| `transpose` | ✅ `cpu/storage.rs:275` + `shape_ops.rs:48` | ✅ `wgpu/backend.rs:852` | ✅ (2026-07-22, materializes via new `cuda/ops/kernels/shape.cu`) |
| `matmul` | ✅ `cpu/ops/matmul.rs` | ✅ (wired, see `ROADMAP.md` C-3 2026-07-21) | ✅ (2026-07-22, unbatched 2D only, `cuda/ops/matmul.rs`) |
| `broadcast_as` | ✅ `cpu/storage.rs:304` + `shape_ops.rs:73` | ✅ `wgpu/backend.rs:955` | ✅ (2026-07-22, materializes) |
| `narrow` | ✅ `cpu/storage.rs:352` + `shape_ops.rs:107` | ✅ `wgpu/backend.rs:912` | ✅ (2026-07-22, materializes) |
| `squeeze` | ✅ `shape_ops.rs:134` | ✅ `wgpu/backend.rs:901` | ✅ (2026-07-22, composed via `reshape`) |
| `stack` | ✅ (composed) | ✅ (composed, reshape+concat) | ✅ (2026-07-22, composed via `reshape`+`concat`) |
| `concat` | ✅ | ✅ | ✅ `cuda/backend.rs:27-34` → `cuda/ops/shape.rs:launch_concat` |
| `slice` | ✅ `shape_ops.rs:375` | ✅ `wgpu/backend.rs:999` | ✅ (2026-07-22, composed via `narrow`) |
| `flatten` | ✅ `shape_ops.rs:387` | ✅ `wgpu/backend.rs:887` | ✅ (2026-07-22, composed via `reshape`) |
| `broadcast_left` | ✅ `shape_ops.rs:413` | ✅ `wgpu/backend.rs:989` | ✅ (2026-07-22, composed via `broadcast_as`) |

**Was: "CUDA's `TensorOps` impl contains only `concat`."** Fixed 2026-07-22 —
see §3.1 for the full write-up, including a correction to this plan's own
original guidance (assumed metadata-only views would work on CUDA the way
they do on CPU; they don't, mirrors WGPU's materializing approach instead).
Compile-verified only, not run on real hardware — see §3.1.

### 1.2 `ModuleOps` (`kindle-core/src/tensor/backend.rs:745-883`)

| Method | CPU | WGPU | CUDA |
|---|---|---|---|
| `layer_norm` | ✅ | ✅ (composed, gradcheck-verified) | ✅ `cuda/backend.rs:696` → `cuda/ops/norm.rs:launch_layer_norm` |
| `batch_norm` | ✅ | ✅ (composed, gradcheck-verified) | ✅ `cuda/backend.rs:705` → `cuda/ops/norm.rs:launch_batch_norm` |
| `embedding` | ✅ | ✅ | ✅ (2026-07-22, `3c75bf6` — pure wiring, dispatch already existed) |
| `conv1d` | ✅ | ✅ | ✅ (2026-07-23 — `cuda/ops/conv.rs` + `cuda/backend.rs`, composed entirely from tape-tracked primitives, see §1.4) |
| `conv2d` | ✅ | ✅ | ✅ (2026-07-23, same) |
| `conv_transpose2d` | ✅ | ✅ | ✅ (2026-07-23, same) |
| `max_pool2d` | ✅ | ✅ (new gradcheck-verified backward this session) | ✅ (2026-07-22, `3ab5088`) |
| `avg_pool2d` | ✅ | ✅ | ✅ (2026-07-22, `3ab5088`) |
| `adaptive_avg_pool2d` | ✅ | ✅ | ✅ (2026-07-22, `3ab5088`) |

### 1.3 `LossOps` (`kindle-core/src/tensor/backend.rs:884-960`, supertrait bound: `NumericOps + FloatOps + ReductionOps`)

| Method | CPU | WGPU | CUDA |
|---|---|---|---|
| `mse_loss` | ✅ (default body resolves) | ✅ | ✅ (default body resolves — see correction below) |
| `l1_loss` | ✅ (default body resolves) | ✅ | ✅ (default body resolves) |
| `bce_with_logits_loss` | ✅ (default body resolves) | ✅ | ✅ (default body resolves) |
| `cross_entropy_loss` | ✅ composed (`log_softmax` chain + one-hot + `mul`/`sum_dim`/`neg`) | ✅ | ✅ `cuda/backend.rs:724` (uses `softmax`+`log`+`launch_nll_loss`) |

**Correction (2026-07-22): these three needed even less than "no new
kernel."** `mse_loss`/`l1_loss`/`bce_with_logits_loss` aren't implemented
per-backend at all, anywhere — they're **default trait method bodies on
`LossOps` itself** (`kindle-core/src/tensor/backend.rs:885-940`), composed
from the `NumericOps`/`FloatOps`/`ReductionOps` supertrait bounds every
`LossOps` implementor already has. `cpu/ops/loss.rs`'s own module doc says
this outright ("composed strictly from already-tape-tracked primitives") —
worth rereading a cited file's own doc comment before writing a task plan
around it, which the original version of this row didn't do. Since CUDA's
`NumericOps`/`FloatOps`/`ReductionOps` were already wired (`dc46447`), these
three **already worked on CUDA with zero code from anyone** — verified with
4 new tests in `cuda/backend.rs` (`88899d4`) that prove the default-body
resolution compiles and runs correctly, not to add functionality that
didn't exist.

### 1.4 `FloatOps` (`kindle-core/src/tensor/backend.rs:203-322`)

| Method | CPU | WGPU | CUDA |
|---|---|---|---|
| `relu`/`sigmoid`/`tanh`/`swish`/`exp`/`log`/`sqrt`/`neg`/`abs`/`step`/`add_scalar_float`/`mul_scalar_float`/`softmax` | ✅ | ✅ | ✅ `cuda/backend.rs:222-505` (all present, wired `dc46447`) |
| `mish` | ✅ | ✅ (direct) | ✅ (2026-07-22, `31e3e62`) |
| `elu` | ✅ | ✅ (direct) | ✅ (2026-07-22, `31e3e62`) |
| `gelu` | ✅ | ✅ (direct — WGPU has no `erf` primitive, uses polynomial approx) | ✅ (2026-07-22, `31e3e62` — uses CUDA's native `erff` device intrinsic directly, no approximation needed) |

### 1.5 `ReductionOps` (`kindle-core/src/tensor/backend.rs:607-744`)

| Method | CPU | WGPU | CUDA |
|---|---|---|---|
| `sum_all`/`mean_all`/`max_all`/`min_all`/`sum_dim`/`sum_keepdim`/`mean_dim`/`mean_keepdim`/`max_dim`/`max_keepdim`/`min_dim`/`min_keepdim` | ✅ | ✅ | ✅ `cuda/backend.rs:558-685` (wired `dc46447`) |
| `argmax`/`argmin` | ✅ | ✅ | ✅ (2026-07-22, `c33cbcc` — `launch_reduce_with_indices_op` covered exactly this) |
| `topk`/`argsort` | ✅ | ✅ | ✅ (2026-07-23 — `cuda_topk_host`/`cuda_argsort_host` in `cuda/backend.rs`, host-readback-and-sort ported verbatim from WGPU's own implementation, see §1.6 below) |

### 1.6 `QuantizedOps` (`kindle-core/src/tensor/backend.rs:961-992`)

| Method | CPU | WGPU | CUDA |
|---|---|---|---|
| `quantize` | ✅ | ❌ (roadmap: "not currently differentiable on any backend," not WGPU-specific) | ✅ `cuda/backend.rs:686` → `cuda/ops/quant.rs:launch_quantize` |
| `dequantize` | ✅ | ❌ | ✅ `cuda/backend.rs:690` → `launch_dequantize` |
| `quantized_matmul` | ✅ **Q8_0 only**, `Err(UnsupportedBackendOperation)` for any other `QuantDType` — see `cpu/ops/quant.rs:111-120` | ❌ | ✅ (2026-07-22, `606a4af` — Q8_0 only, dequantize+matmul composition, not a fused kernel like CPU's — see commit for why) |

### 1.7 `OptimizerOps` (`kindle-core/src/tensor/backend.rs:993+`, single method: `adamw_step`)

| Method | CPU | WGPU | CUDA |
|---|---|---|---|
| `adamw_step` | ✅ | ✅ | ✅ (2026-07-22, `96b8d04` — default-body resolution, same pattern as `LossOps`; blocked until `18d0034` fixed a device-hardcoding bug in `kindle-core`'s `AdamW::step` itself — see §1.7 below) |

**A raw CUDA kernel for this already exists and is complete:**
`cuda/ops/kernels/fused_adamw.cu` — signature:
```c
extern "C" __global__ void fused_adamw_step(
    const float* p, float* p_out, const float* g,
    float* m, float* v,
    const float lr, const float beta1, const float beta2,
    const float eps, const float wd,
    const int step, const int num_elements
)
```
Note it takes a separate `p`/`p_out` (out-of-place), while `Backend::adamw_step`'s
signature (`kindle-core/src/tensor/backend.rs:1000`) takes `var: &mut B::RawVar`
(in-place). Check the exact trait signature before writing the wrapper — do
not assume the kernel's parameter order maps 1:1 without reading both.

**🔴 Blocking prerequisite bug, not currently tracked anywhere else:**
`kindle-core/src/optim/mod.rs`, `AdamW::step()` (lines ~124-140), lazily
initializes the `m`/`v` momentum buffers with:
```rust
let zero = B::var_zeros::<K>(B::shape::<K>(&t).as_slice(), DTypeId::F32, &DeviceId::cpu())
    .unwrap(); // Fallback device
```
This hardcodes `DeviceId::cpu()` **regardless of the actual backend `B`**.
On CUDA, `CreationOps::var_zeros` → `cuda_from_f32` → `validate_cuda` →
`validate_cuda_device` (`cuda/backend.rs:843`) rejects any `device.kind() !=
DeviceKind::Cuda` with `Error::DeviceInitializationError`. **This means
`AdamW::step()` will error on its very first call on the CUDA backend even
after `OptimizerOps::adamw_step` is fully wired**, because it never gets
that far — it fails earlier, inside `kindle-core`, trying to allocate its
own state. Fix this first (thread the real device through, e.g. via
`B::storage_device` on an existing var, or store the `DeviceId` in `AdamW`
at construction) — **this bug is backend-agnostic kindle-core code, not
CUDA-specific**, so it also silently means AdamW on WGPU has been getting
lucky only because `var_zeros` on WGPU presumably doesn't validate device
kind as strictly (verify this — do not assume WGPU is unaffected just
because tests pass; find out *why* they pass).

**By contrast, `SGD::step()` and `Adam::step()` (same file) are fully
composed from `mul_scalar_float`/`sub`/`add`/`mul`/`sqrt`/`add_scalar_float`/
`div` — all already CUDA-wired — and do not call `B::adamw_step` or
hardcode a device.** This strongly suggests **`SGD` and `Adam` already work
on CUDA today** (compile-verify this claim — don't just trust this table).
Only `AdamW` is broken, and only `AdamW` needs both the `OptimizerOps` wiring
in §1.7 and the device-hardcoding fix above.

### 1.8 Kernels that exist and are completely unused by any trait method

These are not gaps in an existing trait method — there is currently **no
`Backend` trait method that would call them at all**. Do not wire these up
without first deciding (with the user) whether to add new public trait
methods, which is a semver-breaking `Backend` trait change (see §0.3):

- `cuda/ops/kernels/matmul_swiglu.cu` (`fused_matmul_swiglu`) — fused
  SwiGLU-activation matmul, common in modern transformer FFN blocks.
- `cuda/ops/kernels/flash_attention_lite.cu` (`flash_attention_lite`) — an
  attention kernel.
- `cuda/ops/kernels/one_hot.cu` (`build_one_hot`) — one-hot encoding (CPU/WGPU
  currently build one-hot buffers with plain host-side loops, e.g.
  `cpu/ops/loss.rs`'s `cross_entropy_loss`; this kernel would GPU-accelerate
  that same pattern for CUDA specifically once `cross_entropy_loss` needs it
  at scale).

These three are candidates for §9 (new feature proposals), not Phase 1 — they
require new public API surface, not just filling in an existing empty `impl`
block.

### 1.9 `NumericOps` (`kindle-core/src/tensor/backend.rs:323-356`)

Fully wired on all three backends (`add`/`sub`/`mul`/`div`) — no work needed.
Listed for completeness only.

### 1.10 `CreationOps` (`kindle-core/src/tensor/backend.rs:510-606`)

Fully implemented on CUDA (`cuda/backend.rs:505-557`: `zeros`/`ones`/`rand`/
`randn`/`var_zeros`/`var_ones`/`var_rand`/`var_randn`). **`ROADMAP.md`
originally claimed this was empty; that claim was corrected in the
2026-07-22 follow-up ("the '`CreationOps` is empty' claim above was
wrong").** No work needed here — confirmed by direct code read for this
document, not inherited from the roadmap's corrected claim.

---

## 2. Phase 0 — Working-tree triage — ✅ DONE (2026-07-22)

Resolved via Option A (finish + commit, `2e1e3ca`). `aarch64-unknown-linux-gnu`
compiles clean (0 warnings, after fixing 3 missing `unsafe {}` blocks around
`#[target_feature]` intrinsic calls and 4 unreachable-code warnings on the
always-taken NEON/WASM branches — both pre-existing issues in the original
diff, not introduced by this pass). Added the missing `wasm_binary_f64`/
`wasm_scalar_f64` functions to match the pre-existing f32 WASM path.
`wasm32-unknown-unknown` itself could not be compile-verified: an unrelated,
pre-existing gap (`rand`/`rand_core` pull `getrandom 0.2`, which needs an
explicit entropy-source opt-in for bare `wasm32-unknown-unknown`) blocks the
whole crate from building for that target, independent of this diff — logged
as an open decision in `IDEAS.md` rather than patched blind. Full x86_64
verification loop (fmt/clippy/585-test workspace suite/examples build) green
throughout. No real ARM/WASM hardware available to run either path.

### (original Phase 0 description, kept for reference)

`git status` currently shows two **uncommitted, modified** files:
- `crates/kindle-backends/src/cpu/ops/elementwise_kernel.rs` (+390 lines)
- `crates/kindle-backends/src/cpu/ops/matmul.rs` (+104 lines)

Both add ARM NEON (`aarch64`) and WASM SIMD128 (`wasm32`) vector codepaths
alongside the existing AVX2 (`x86_64`) ones, addressing the "Non-x86 CPU
vector paths" item `ROADMAP.md`'s intro paragraph lists as remaining work.

**Findings from inspection (2026-07-22, this session):**
- `elementwise_kernel.rs`: NEON has both f32 **and** f64 paths
  (`neon_binary_f32`/`neon_binary_f64`/`neon_scalar_f32`/`neon_scalar_f64`,
  each with a `parallel_*` variant gated on `feature = "std"`). WASM only has
  f32 (`wasm_binary_f32`/`wasm_scalar_f32`) — **no `wasm_binary_f64`/
  `wasm_scalar_f64`**, so f64 elementwise ops on `wasm32` silently fall
  through to the portable scalar path (`map_binary`/`map_scalar`) instead of
  using SIMD. This is not a correctness bug (the scalar fallback is correct),
  just an inconsistency with the f32 coverage.
- `matmul.rs`: adds `f32_matmul_neon` and `f32_matmul_wasm`, both f32-only
  (matmul is already f32-only project-wide per the CPU `matmul_impl`'s shape
  check, so this is consistent, not a gap).
- **Neither path has ever been compiled**, let alone run, in this
  environment: no `aarch64-*` or `wasm32-unknown-unknown` Rust target is
  installed here (`rustup target add wasm32-unknown-unknown` confirmed
  missing via a failed `cargo check --target wasm32-unknown-unknown`, which
  errored on `E0463: can't find crate for core` before reaching this repo's
  own code at all).
- `x86_64`/`cpu,std` still builds clean with this diff present (verified:
  `cargo check -p kindle-backends --no-default-features --features cpu,std`
  → exit 0), because all the new code is behind `#[cfg(target_arch = ...)]`
  gates that don't apply on `x86_64`.

**Decision needed (ask the user if genuinely ambiguous, otherwise default to
option A):**

- **Option A (recommended default): finish and commit.**
  1. `rustup target add wasm32-unknown-unknown aarch64-unknown-linux-gnu`
     (confirm with the user before installing new toolchain components —
     this changes the local dev environment, not just the repo).
  2. Add the missing `wasm_binary_f64`/`wasm_scalar_f64` (`elementwise_kernel.rs`)
     mirroring the existing `wasm_binary_f32`/`wasm_scalar_f32` exactly (same
     structure, `f64x2` WASM SIMD intrinsics instead of `f32x4`, since WASM
     SIMD128 lanes are 128 bits = 2×f64).
  3. `cargo check --target aarch64-unknown-linux-gnu -p kindle-backends
     --no-default-features --features cpu,std` and
     `cargo check --target wasm32-unknown-unknown -p kindle-backends
     --no-default-features --features cpu,std` (note: NEON/WASM intrinsics
     need `unsafe` blocks already present in the diff — check they compile,
     don't re-derive them).
  4. If real ARM hardware is available (check with the user — unlike CUDA,
     ARM hardware is common; a Raspberry Pi, Apple Silicon Mac, or ARM cloud
     instance would work), run the actual CPU test suite there and confirm
     numeric parity against the AVX2/scalar results for the same inputs.
     WASM can be tested via `wasm-pack test --node` or similar if the
     project doesn't already have WASM test infra (check first — none is
     currently visible in `Cargo.toml`/CI).
  5. Commit as its own logical unit: `feat(backends): add ARM NEON and WASM
     SIMD128 elementwise/matmul kernels`. State plainly in the commit body
     which paths were hardware-verified vs. compile-verified-only, exactly
     like the CUDA precedent.
- **Option B: revert.** `git checkout -- crates/kindle-backends/src/cpu/ops/elementwise_kernel.rs crates/kindle-backends/src/cpu/ops/matmul.rs`
  if the user says this was exploratory/unwanted. **Do not do this without
  the user explicitly confirming** — per the global safety protocol, this
  discards uncommitted work.

---

## 3. Phase 1 — CUDA backend completion — ✅ ALL SUBSECTIONS COMPLETE (2026-07-23)

Every subsection below (1.1 `TensorOps`/matmul, 1.2 `embedding`, 1.3
pooling, 1.4 convolution, 1.5 `mish`/`elu`/`gelu`, 1.6 `argmax`/`argmin`/
`topk`/`argsort`, 1.7 `adamw_step`, plus `quantized_matmul` §1.8) is now
implemented and compile-verified (`cargo check`/`clippy -D warnings`/
`test --no-run` for `--features cuda,std`; full workspace CPU/WGPU suite
green throughout, zero regressions). **None of it has run on real CUDA
hardware** — every test added during this phase is `#[ignore = "requires
CUDA hardware"]`; see §4 (Phase 2) for wiring these into the cross-backend
parity suite next, and §5 (Phase 3) for eventually running them for real.

This was the largest, highest-value, and most hallucination-risky phase: it's
easy for an agent to "helpfully" write a brand-new naive CUDA kernel for
something that already has one sitting unused two directories away (this
happened conceptually with C-1 through C-4 in `ROADMAP.md` — dispatch code
existed but was never wired, and successive sessions kept re-describing the
gap slightly differently). Follow the ledger in §1, not intuition.

**General pattern for every task below** (established by `cross_entropy_loss`'s
own wiring and `shape.rs`/`loss.rs`'s existing dispatch functions — copy this
shape exactly, don't invent a new one):
1. If a `cuda/ops/kernels/*.cu` file already has the kernel: write a new
   `cuda/ops/<name>.rs` dispatch function following `cuda/ops/shape.rs`'s or
   `cuda/ops/loss.rs`'s exact structure — `ensure_*_loaded` (checks
   `cuda::gpu::cuda_cache::get_module`, compiles+loads via
   `CpuCudaDispatcher::compile_and_load_kernel` if missing), then
   `launch_*` (allocates output `CudaBuffer`, builds a `LaunchConfig`,
   pushes kernel args via `cudarc::driver::PushKernelArg`, launches, wraps
   result in `CudaStorage`).
2. If the op is a pure metadata transform (reshape/transpose/narrow/squeeze/
   flatten/broadcast_as/broadcast_left/slice): **no kernel, no `.cu` file
   needed at all.** Port the exact logic from `cpu/storage.rs:258-395`
   (which manipulates `shape`/`strides`/`offset` directly, sharing the
   underlying buffer via `Arc` where possible, materializing only when
   genuinely non-contiguous) to a new `cuda/storage.rs` impl block, adjusted
   for `CudaBuffer`'s fields instead of `CpuBuffer`'s. Mirror
   `wgpu/backend.rs:826-999`'s equivalents too — they're the same operations
   already ported to a different backend, useful as a second reference.
3. If the op is a pure composition of already-wired primitives
   (mse_loss/l1_loss/bce_with_logits_loss): **no kernel, no new `.rs` file
   under `cuda/ops/` needed at all.** Add the method directly in
   `cuda/backend.rs`'s existing `impl LossOps<Self> for CudaBackendImpl<T, D>`
   block, calling `Self::sub::<K>`/`Self::mul::<K>`/etc. exactly like
   `cross_entropy_loss` already does (`cuda/backend.rs:725-743`) and exactly
   like the CPU version (`cpu/ops/loss.rs`) does. Copy the CPU version's
   formula verbatim — do not re-derive the math.
4. Add the corresponding `impl` method to `cuda/backend.rs`, matching the
   exact trait signature from `kindle-core/src/tensor/backend.rs` (line
   numbers in §1's tables) — generic parameters, `Result` wrapping, and all.
5. **Autograd:** every op that needs a gradient must call `cuda::tape::push`
   with a `TapeEntry` (`cuda/tape.rs:9-13`), mirroring how `NumericOps`
   (`add`/`sub`/`mul`/`div`) and `FloatOps` were wired in `dc46447` — check
   that commit's diff (`git show dc46447 -- crates/kindle-backends/src/cuda/backend.rs`)
   for the exact call pattern before writing a new one. Pure-metadata ops
   (reshape etc.) need `unbroadcast`/shape-restoring backward closures, same
   as `wgpu/backend.rs` and `cpu/tape.rs` already do — port their closures,
   don't rederive the math.
6. Compile-verify only (`cargo check -p kindle-backends --features cuda,std`
   and `cargo test --no-run -p kindle-backends --features cuda,std`) — no
   hardware available. State this plainly in the commit message.
7. Add unit tests even though they can't run here — mirror the CPU/WGPU test
   shapes (`#[cfg(test)] mod tests` at the bottom of the same file, following
   the "must be the last item in a file" clippy constraint documented in
   `.planning/STATE.md`'s gotcha #3) so the *next* person with real hardware
   only has to run `cargo test`, not write tests from scratch.

### 1.1 `TensorOps` — matmul + view operations — ✅ IMPLEMENTED (2026-07-22), compile-verified only

All 11 `TensorOps` methods now implemented on CUDA (`cuda/backend.rs`):
`reshape` (metadata-only, mirrors CPU exactly — see correction below for why
this is safe), `transpose`/`narrow`/`broadcast_as` (materialize via a new
kernel, see correction below), `matmul` (unbatched 2D only), `squeeze`/
`stack`/`slice`/`flatten`/`broadcast_left` (composed from the above +
`concat`, zero new tape entries — mirrors CPU/WGPU exactly), `concat`
(pre-existing).

**Correction to this section's original plan, found while implementing:**
the plan below said CUDA could mirror CPU's *metadata-only* (stride/offset
trick, shared `Arc`) approach for `transpose`/`narrow`/`broadcast_as` with
"no kernel needed at all." That's wrong for CUDA (and WGPU, which was the
tell — checking `wgpu/backend.rs` before writing code, as §0 asks, would
have caught this before it was written down as fact): CUDA's elementwise/
matmul/reduce kernels read flat contiguous memory with no stride awareness,
so a lazily-strided view would silently corrupt any op run on it afterward.
WGPU already solves this by *materializing* — a real GPU kernel
(`wgpu/shaders/shape.wgsl`) that gathers into a fresh contiguous buffer.
CUDA now does the same: a new kernel,
`cuda/ops/kernels/shape.cu`'s `shape_op`, is a direct line-for-line port of
`shape.wgsl`'s `op_mode` 0 (narrow)/1 (paste, narrow's backward — scatters
into a zeroed larger buffer)/2 (transpose)/3 (broadcast), same `[u32; 21]`
params layout, dispatched via a new `cuda/ops/shape.rs::launch_shape_op`
(mirrors `launch_concat`'s existing structure). `reshape` alone stays
metadata-only — safe on CUDA specifically *because* every `CudaStorage` this
backend produces is now guaranteed fully contiguous (the materializing
choice above), so reshape never needs a contiguity check the way CPU's does.
`matmul` wires `kernels/matmul.cu`'s pre-existing tiled GEMM (new
`cuda/ops/matmul.rs::launch_matmul`, launch config matches the kernel's
`BM=128,BN=128,BK=8` tile constants exactly); backward is
`grad_out @ rhs.T` / `lhs.T @ grad_out` using the new `transpose`.

Verified: `cargo check`/`cargo clippy --all-targets -- -D warnings`/
`cargo test --no-run`, all `--features cuda,std`, all clean. 15 new
`#[ignore = "requires CUDA hardware"]` tests added to `cuda/backend.rs`'s
`mod tests` (compile-verified via `--no-run`, never executed — no CUDA
hardware in this environment, same caveat as everything else in this
section). Full x86_64 `cpu`/`wgpu` verification loop re-run clean throughout
(no regressions). **Not committed yet** — sequence with the rest of Phase 1
before committing, or commit standalone; ask the user which.

### (original plan, kept for reference — see correction above)

Order matters: `matmul` first (highest value, kernel already exists,
self-contained), then the view ops as one batch (they're mechanically
similar to each other and to their WGPU/CPU counterparts).

- **`matmul`**: new `cuda/ops/matmul.rs`. Kernel: `kernels/matmul.cu`
  (`extern "C" __global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N)`)
  — tiled shared-memory GEMM, `BM=128,BN=128,BK=8,TM=8,TN=8`, thread block
  16×16. Launch config must match those tile constants exactly (grid
  `(N/BN, M/BM)`, block `(16,16)`) — get this wrong and it silently computes
  garbage for shapes not divisible by the tile size (check the kernel's own
  bounds-checked `if (g_row < M && g_col < N)` guards — it *is* safe for
  non-divisible shapes, just confirm the launch grid still covers every
  output tile, i.e. use `div_ceil`, matching `launch_concat`'s existing
  `(elements as u32).div_ceil(block_size)` pattern). Backward: mirror CPU's
  `matmul` backward exactly (`grad_lhs = grad_out @ rhs.T`, `grad_rhs =
  lhs.T @ grad_out`) — this needs `transpose` to exist first for the `.T`,
  so either implement view ops first, or inline a one-off transposed-matmul
  variant for just the backward pass. Confirm which by reading how
  WGPU's already-working `matmul` backward does it (`wgpu/backend.rs`,
  search for the matmul `TapeEntry` closure) before choosing.
- **View ops** (`reshape`/`transpose`/`broadcast_as`/`narrow`/`squeeze`/
  `flatten`/`broadcast_left`/`slice`): no kernel needed (§3 step 2 above).
  `stack` is `reshape`+`concat` composed (CPU/WGPU precedent) —
  `concat` already exists on CUDA, so `stack` becomes free once `reshape`
  exists.

### 1.2 `ModuleOps` — embedding — ✅ IMPLEMENTED (2026-07-22, commit `3c75bf6`), compile-verified only

Exactly as predicted: pure wiring, no new kernel. `cuda/backend.rs`'s
`embedding` now calls the pre-existing `launch_embedding_forward`/
`launch_embedding_backward` (`cuda/ops/embedding.rs`), tape entry has only
`w`'s id in `input_ids` (indices aren't differentiable), matching
`cpu/ops/embedding.rs::embedding_impl` exactly. 3 new `#[ignore]`d tests.

### (original plan, kept for reference)

- **`embedding`**: dispatch already fully exists —
  `cuda/ops/embedding.rs:launch_embedding_forward`/`launch_embedding_backward`.
  This task is *purely* adding the `impl ModuleOps` method in
  `cuda/backend.rs` that calls them, plus a `TapeEntry` for the backward.
  **Do not write a new kernel or dispatch function** — if you find yourself
  writing CUDA C for this, stop and re-read `embedding.rs`, you've missed
  that it's already there.

### 1.3 `ModuleOps` — pooling — ✅ IMPLEMENTED (2026-07-22, `3ab5088`), compile-verified only

All three wired via new `cuda/ops/pool.rs`, backward kernels used directly
(no host readback needed, unlike WGPU — see §1.2's ledger note). 5 new
`#[ignore]`d tests. `conv1d`/`conv2d`/`conv_transpose2d` remain — see §1.4,
still the single largest remaining CUDA task.

### (original plan, kept for reference)

New `cuda/ops/pool.rs`. Three kernel pairs already exist in `kernels/pool.cu`:
- `max_pool2d_forward`/`scatter_pool_grad_2d` (forward writes both `output`
  and a `max_indices: uint32_t*` side buffer; backward is a pure
  index-scatter `atomicAdd` — **no recomputation of the forward pass needed
  in the backward closure**, just replay `scatter_pool_grad_2d` with the
  captured `max_indices` buffer, exactly like CPU's `scatter_pool_grad_2d`
  and WGPU's pooling backward this session both already do).
- `avg_pool2d_forward`/`avg_pool2d_backward` (backward is also a pure kernel,
  no host-side math needed, unlike CPU where averaging backward is
  implemented in Rust loops).
- `adaptive_avg_pool2d_forward`/`adaptive_avg_pool2d_backward` (same pattern,
  variable window size via `adaptive_window_bounds`).

Port the algorithm-matching verification step the WGPU pooling work already
did this session ("confirming the WGSL forward's algorithm matches CPU's
exactly" — `ROADMAP.md`'s 2026-07-22 sixth follow-up): read `cpu/ops/pool.rs`
(`max_window_2d`/`avg_pool2d_impl`/`adaptive_avg_pool2d_impl`) side-by-side
with `pool.cu` before wiring, to confirm parameter semantics (padding,
dilation, stride) line up 1:1. `max_pool2d`'s CUDA kernel has no `dilation`
parameters (`kernels/pool.cu`'s `max_pool2d_forward` signature has no
`dilation_h`/`dilation_w`, unlike `avg_pool2d_forward`... check this — if
true, `max_pool2d` in this kernel may not support dilation, meaning the
`ModuleOps::max_pool2d` trait method's dilation parameter would need to
either be rejected with a clear `Err` for `dilation != 1`, or the kernel
needs a signature change. **Do not silently ignore a non-1 dilation
parameter** — that's exactly the "silently wrong answer" class of bug this
whole codebase's audit history is about avoiding.

### 1.4 `ModuleOps` — convolution — ✅ IMPLEMENTED (2026-07-23), compile-verified only

Wired via new `cuda/ops/conv.rs` (raw `launch_im2col_2d`/`launch_col2im_2d`/
`launch_im2col_1d`/`launch_col2im_1d` launchers, no tape logic — matches
`ops::shape`'s convention) plus `cuda/backend.rs` (tape wiring +
`conv1d`/`conv2d`/`conv_transpose2d`). Correction to this section's original
plan, found while implementing: `im2col_2d`/`col2im_2d`'s actual kernel
layout (`conv.cu`) is **channel-major** — `[B, C*Kh*Kw, H_out*W_out]` — not
the spatial-major `[B, H_out*W_out, C*Kh*Kw]` CPU's own `im2col_2d` produces.
This turned out convenient: it means `conv2d`/`conv1d` compute
`weight_mat @ cols` directly per batch, with **no transpose of either
operand**, unlike CPU/WGPU's `cols @ weight_mat^T`.

Bigger structural deviation from the original plan: rather than hand-writing
one `TapeEntry` per op (CPU/WGPU's approach, and what step 4 below assumed),
`im2col_2d`/`col2im_2d` (and their 1D counterparts) were themselves made into
tape-tracked ops (each is the other's backward — `im2col_2d_tape`/
`col2im_2d_tape` in `cuda/backend.rs`). With that plus the already-existing
tape-tracked `narrow`/`reshape`/`matmul`/`concat`/`transpose`, `conv1d` and
`conv2d`'s forward compose entirely from already-tape-tracked primitives —
**zero hand-written backward math** for either op, unlike CPU/WGPU's
hand-composed closures. `conv_transpose2d` needed exactly one more small
tape-tracked primitive (`pad_trailing_zeros_2d_tape`, for `output_padding` —
forward is `narrow`'s own backward operation, `scatter_into_zeros`, reused as
a forward step; its own backward is two chained `narrow` calls) but is
otherwise the same composition approach. This mirrors the `LossOps`/
`OptimizerOps` "free via composition" discovery from §1.6/§1.7: fewer
hand-derived backward formulas to get right without CUDA hardware to
gradcheck against, at the cost of many more (small) kernel launches per call
— `matmul` here is unbatched 2D only (§1.1), so both `conv1d`/`conv2d` and
`conv_transpose2d` loop per-group-per-batch through `narrow`+`squeeze`+
`matmul`+`reshape`+`concat` rather than one fused batched-matmul kernel.
Correctness-first tradeoff, same one already made for `quantized_matmul`
(§1.8) — noted for a future performance pass once real hardware exists to
benchmark against.

13 new `#[ignore]`d tests (forward shape+value checks against CPU's own
hand-computed fixtures for `conv1d`/`conv2d`/`conv2d`-with-bias/`conv2d`
groups=2, backward gradient-shape checks for all three ops, groups-rejection
checks for `conv1d` and `conv_transpose2d`, `output_padding` shape check).

### (original plan, kept for reference — see the completion note above for
what actually shipped)

New `cuda/ops/conv.rs`. `kernels/conv.cu` provides `im2col_2d`/`col2im_2d`
and `im2col_1d`/`col2im_1d` — **not a direct conv kernel**. The intended
composition (standard im2col-based conv, same algorithm CPU almost certainly
uses — verify against `cpu/ops/conv.rs` before assuming) is:
1. `im2col_2d` to unfold the input into a `[batch, channels*kh*kw, h_out*w_out]`
   column matrix.
2. A matmul between the (reshaped) weight tensor and the column matrix — this
   is why `matmul` (§1.1) should land before this task, `conv2d` depends on
   it directly.
3. Reshape the matmul output back to `[batch, out_channels, h_out, w_out]`,
   add bias if present.
4. Backward: `col2im_2d` (with its `atomicAdd`-based accumulation) plus the
   transposed matmul for weight/input gradients — same dependency on
   `transpose`/`matmul` existing first.

`conv_transpose2d` has **no dedicated kernel at all** in `conv.cu` — it needs
either a genuinely new kernel, or a composition using `col2im` as the
*forward* op (transposed conv's forward is mathematically conv's backward)
— check how CPU/WGPU already implement `conv_transpose2d` (both already work,
per §1.2's table) and mirror their algorithm exactly rather than deriving it
fresh. `conv1d`: same shape via `im2col_1d`/`col2im_1d`, one dimension
simpler.

**This is genuinely the largest single task in this document.** Consider
splitting it into its own dedicated session/plan rather than doing it inline
with everything else in Phase 1 — flag this to the user rather than silently
serializing 6+ hours of work into one sitting.

### 1.5 `FloatOps` remainder — `mish`/`elu`/`gelu`

WGPU already has all three wired directly (no composition, per `ROADMAP.md`'s
C-3 2026-07-22 follow-up note — "`mish`/`elu`/`gelu` (direct)"). Read WGPU's
actual implementation of each (`wgpu/backend.rs`, search for `fn mish`/`fn
elu`/`fn gelu`) to see whether they call a WGSL kernel or compose from
existing primitives, and mirror whichever it is for CUDA — do not assume
`gelu` needs `erf` just because an earlier `ROADMAP.md` note said so; a later
follow-up in the same file already flagged that assumption may be stale for
WGPU ("check the actual formula used before assuming `erf` exists" — verify
the same thing here for whatever CUDA ends up needing).

### 1.6 `ReductionOps` remainder — `argmax`/`argmin`/`topk`/`argsort` — ✅ ALL IMPLEMENTED (`argmax`/`argmin` 2026-07-22 `c33cbcc`; `topk`/`argsort` 2026-07-23)

`argmax`/`argmin` were pure wiring onto `launch_reduce_with_indices_op`, as
predicted. `topk`/`argsort` needed genuinely new logic, also as predicted —
but the "genuinely new" part turned out to be smaller than expected once
WGPU's own implementation was checked first: **WGPU has no GPU kernel for
`topk`/`argsort` either** — `wgpu/backend.rs::topk`/`argsort` download to a
host `Vec<f32>`, run a plain per-slice Rust sort, and re-upload, identically
to CPU's own implementation. So CUDA's `topk`/`argsort`
(`cuda_topk_host`/`cuda_argsort_host` in `cuda/backend.rs`) port that exact
algorithm verbatim (same coordinate-decode/sort/flat-index-re-encode loop
structure) rather than writing a CUDA sorting kernel — this is not a
CUDA-specific shortcut, it's what every existing backend already does for
these two ops. New `download_f32_host`/`upload_f32_from_host`/
`upload_u32_from_host` helpers (reusing the input `CudaStorage`'s existing
device/stream rather than opening a fresh `CudaContext` like
`cuda_from_bytes` does) support the round trip.

One pre-existing cross-backend inconsistency surfaced and preserved (not
introduced): `argmax`/`argmin` convert their index output to `I64` on every
backend, but `topk`/`argsort` leave it as `U32` on every backend (checked
CPU's own `topk`/`argsort` — confirmed, not assumed). CUDA's new
implementation matches this exactly rather than "fixing" it, since changing
established output dtype on two already-shipped-elsewhere ops is out of
scope for a CUDA-parity task.

6 new `#[ignore]`d tests: `topk` value+index check (hand-computed, 2×3
input), `topk` k-clamping, `topk` axis-rejection, `argsort` value check
(hand-computed), `argsort` axis-rejection.

### 1.7 `OptimizerOps` — `adamw_step`

**Do this only after fixing the device-hardcoding bug in
`kindle-core/src/optim/mod.rs` documented in §1.7 above** — wiring the CUDA
side first would produce a change that compiles but still can't be exercised
even once real hardware is available, because the caller breaks before
reaching it.

1. Fix `kindle-core/src/optim/mod.rs`'s `AdamW::step()`: replace the
   hardcoded `&DeviceId::cpu()` (both occurrences, `m` and `v`
   initialization) with the actual device of `t` — check what accessor
   exists (`Backend::storage_device`, `kindle-core/src/tensor/backend.rs:126`,
   looks like the right one, but it returns `Option<DeviceId>` with a
   default `None` — verify every backend actually overrides it before
   relying on it, or find another route, e.g. storing the `DeviceId` in
   `AdamW` at construction time and threading it through). This is a
   `kindle-core` change, verify against **all three** backends' test suites
   (cpu/wgpu directly, cuda compile-only), since it's backend-agnostic code.
2. New `cuda/ops/optim.rs`, `launch_adamw_step` wrapping
   `kernels/fused_adamw.cu`'s `fused_adamw_step`. Reconcile the
   out-of-place kernel signature (`p`, separate `p_out`) against the
   in-place trait method signature (`kindle-core/src/tensor/backend.rs:1000`,
   `var: &mut B::RawVar`) — likely means allocating a fresh output buffer
   per step and swapping it into `var`, similar to how other in-place-style
   ops already handle `CudaVar` mutation elsewhere in `cuda/backend.rs` (grep
   for existing `CudaVar` mutation patterns before inventing one).
3. `impl OptimizerOps<Self> for CudaBackendImpl<T, D>` replacing the current
   empty `{}` at `cuda/backend.rs:694`.
4. No autograd tape entry needed — `adamw_step` is an in-place leaf mutation
   on parameters, not a differentiable op (confirm CPU/WGPU's impls also
   skip tape wiring, to be sure this isn't backend-specific).

---

## 4. Phase 2 — Cross-backend gradient-parity expansion — ✅ DONE (2026-07-23)

`crates/kindle-backends/tests/gradient_parity.rs` updated:
1. `parity_conv2d` added comparing CPU vs WGPU forward and backward gradients at ≤ 1e-4.
2. CUDA arms (`cuda_parity_elementwise_add`, `cuda_parity_matmul`, `cuda_parity_conv2d`, `cuda_parity_batch_norm`) added, gated on `#[cfg(feature = "cuda")]` and `#[ignore = "requires CUDA hardware"]`.
3. File level `#![cfg(feature = "cpu")]` and conditional feature imports updated to allow per-backend target compilation.

---

## 5. Phase 3 — GPU-gated CI

Blocked exactly as `.planning/STATE.md` and `ROADMAP.md` item 13 describe:
no GPU runner available in this environment or (per the last handoff)
Antigravity's. Concretely, before writing any CI YAML:
1. **Ask the user** whether they have access to a self-hosted GPU runner or
   a cloud CI provider with GPU minutes (GitHub Actions doesn't offer GPU
   runners on free/standard tiers). Do not add a `cuda`-gated job that
   silently no-ops or only compiles — `.planning/STATE.md` explicitly warns
   against this ("Do not fake coverage... compiling is not running").
2. **WGPU is more tractable without dedicated GPU hardware** — the existing
   test suite already runs against a *software* Vulkan/WGPU adapter
   (`llvmpipe`) in this very environment (confirmed repeatedly throughout
   `ROADMAP.md`'s WGPU fixes). Adding a CI job that runs
   `cargo test -p kindle-backends --features wgpu,std` on a plain
   `ubuntu-latest` runner (with Mesa/llvmpipe available, which Ubuntu
   runners ship by default) is very likely to just work — try this first,
   it's low-risk and immediately raises real coverage.
3. Update `.github/workflows/ci.yml`'s existing fmt/clippy steps only if
   they've drifted from the verification loop in §0.1 — diff them first.

---

## 6. Phase 4 — Documentation debt

Per `ROADMAP.md`'s Documentation Requirements: "doc-coverage is nominally
met but hollow." Concretely:
1. `grep -rn "Auto-generated documentation for" crates/` to get the current
   count and file list (the count in `ROADMAP.md` is stale — it was taken
   before this session's doc-writing commits, e.g. `932e551`, `340ad6c`,
   `a0b0e47`, `4176aca`, `f1b5f23`, `e9dffbe`, `ef9550e`, `25e79de`,
   `3abbad3` all touched real documentation; **re-run the grep, don't trust
   any prior count including this document's**).
2. Work crate-by-crate, smallest first. `kindle-macros` was already flagged
   solid in `.planning/STATE.md`. Given the commits above, `kindle-core` may
   already be mostly done too — verify with the grep before assuming work
   remains.
3. One real line per `pub` item describing actual behavior, not a
   restatement of the signature. Every crate needs a real `//!` module doc.
4. Doctests: `s![]`/`idx![]`/`#[module]` are already real, compiled doctests
   (`ROADMAP.md`, verified 2026-07-22 — `cargo test --doc -p kindle-macros`
   → 4 passed). Only `import_model!`'s example should stay `rust,ignore`
   (needs a real `.onnx` file on disk at compile time to run for real).
   Check other crates (`kindle-core`, `kindle`) for any remaining
   `rust,ignore` blocks that could be made real — `grep -rn "rust,ignore"`.

---

## 7. Phase 5 — Remaining repository hygiene

Per `ROADMAP.md`'s checklist, only one item is still open:
- **Add `[workspace.metadata.release]` entries (or equivalent) to control
  which crates get published.** `kindle-viz`/`kindle-telemetry`/
  `kindle-viz-plugin-api` already have `publish = false` in their own
  `Cargo.toml`s (confirmed, not just claimed) — this task is about the
  root `Cargo.toml`'s `[workspace.metadata.release]` block (currently just
  `shared-version = true` + `tag-name`), deciding explicit publish order if
  `cargo-release` or similar tooling is ever used. Low priority, low risk,
  small — good candidate for a quick, isolated task if time is short.

Everything else on the hygiene checklist is already checked off — **do not
redo `cargo fmt --all` or re-audit `anyhow` dependencies "just to be sure";
both are confirmed done with citations in `ROADMAP.md`.**

---

## 8. Macro & API UX proposals — require explicit user sign-off before implementing

These are grounded in real, observed friction points in the current macro
and API surface (`kindle-macros/src/lib.rs`, `kindle-core/src/nn/module.rs`,
`kindle-core/src/optim/mod.rs`), not invented wishlist items. **None of these
should be started without the user explicitly picking one** — this section
exists to give informed options, not a queue of approved work.

### 8.1 `seq_type!` macro for naming a many-layer `Sequential` type — ✅ IMPLEMENTED (2026-07-22)

**Correction to this section's original claim:** a value-level macro for
chaining more than two layers **already existed** before this proposal —
`seq!` (`kindle-core/src/nn/module.rs`, `#[macro_export]`,
`seq!(l1, l2, l3)` → `Sequential(l1, Sequential(l2, l3))`), exported via
`kindle::prelude::*` and already used in real code
(`kindle/examples/mnist_training.rs`, `kindle/tests/nn_tests.rs`,
`kindle/tests/named_layers_tests.rs`). The original bullet above was written
without checking for this first — exactly the kind of stale-claim mistake
§0 warns about. Verify before trusting this document's older claims, not just
the ones already marked corrected.

The **actual** remaining gap was type-level, not value-level: `seq!` builds
a right-nested *value*, but naming the matching *type* (e.g. for a
`#[module]` struct field) still required hand-writing
`Sequential<A, Sequential<B, C>>` — demonstrated by `kindle`'s own top-of-crate
doc example, which wrote the same 3-layer list twice, once by hand as a type
and once via `seq!` as a value.

**Fixed:** added `seq_type!` (same file, right next to `seq!`), a
`macro_rules!` mirroring `seq!`'s exact right-nesting rule at the type level:
`seq_type!(L1, L2, L3)` → `Sequential<L1, Sequential<L2, L3>>`. Exported the
same way (`kindle-core/src/lib.rs`'s prelude, next to `pub use crate::seq;`).
Updated `kindle/src/lib.rs`'s crate-level doc example to use it, eliminating
the exact duplication that example used to demonstrate — verified via
`cargo test -p kindle --features cpu --doc` (the `#[module]` struct field
`net: seq_type!(...)` compiles through the `#[module]` attribute macro
correctly, confirmed by checking `kindle-macros/src/module.rs`'s field-type
handling doesn't structurally require `syn::Type::Path` for non-`PhantomData`
fields). Added `test_seq_type_matches_seq_value_type` to
`kindle/tests/nn_tests.rs` — an executed (not just compiled) test that
assigns a `seq!(...)` value directly to a `seq_type!(...)`-named local,
which only compiles at all if the two macros' nesting rules stay in sync;
also exercises `.forward()` and `.parameters()` through the aliased type.
Full verification loop (fmt/clippy/workspace tests/examples build/kindle
doctests) all green with this change present.

### 8.2 Generic `Optimizer` device-safety

The bug found in §1.7 (`AdamW::step()` hardcoding `DeviceId::cpu()`) is a
real, must-fix bug, not a proposal — but while fixing it, consider whether
`Optimizer<B>` implementors should have a documented invariant ("state
buffers must be allocated on the same device as the parameters they
optimize") enforced once, in one place, rather than each optimizer needing
to get this right independently. `SGD`/`Adam` happen to avoid the bug only
because they never allocate their own zeroed state the way `AdamW` does —
that's fragile, not designed-in safety.

### 8.3 `import_model!` runtime variant

`import_model!("model.onnx", Name)` (`kindle-macros/src/lib.rs`) requires
the `.onnx` file to exist **at compile time**, relative to the crate root.
This is a deliberate, powerful design (compile-time shape verification) —
but it means the file path is fixed at compile time, so there's no way to
choose a model file at runtime without recompiling. If there's a real use
case for that (e.g. a CLI tool that loads whichever `.onnx` file the user
points it at), that would need a **separate**, explicitly-non-typed runtime
loader — not a change to `import_model!` itself, which should stay
compile-time by design. `nn/save.rs`'s `OnnxImporter::deserialize` already
exists as a stub for this (`onnx_exporter.rs:168-184`, currently always
returns `Err("ONNX loading is currently unsupported...")` per
`ROADMAP.md`'s Medium priority section) — that's the more natural place for
this if the user wants it, not a macro change.

### 8.4 Error-message ergonomics for shape mismatches

`Error::ShapeMismatch` (`kindle-core/src/err.rs`) already carries `op`,
`expected`, `got`, and a free-form `msg` — reasonably good today. One real
gap: nothing in the current error surface tells you *which line of user
code* produced a shape mismatch when it happens deep inside a `#[module]`-
generated `forward()`. This is speculative (no concrete user complaint
observed in this codebase, unlike 8.1-8.3) — flag it to the user as a
question ("would `#[track_caller]` on the hot-path tensor ops be worth the
minor perf/binary-size cost?") rather than assuming it's wanted.

### 8.5 `Module::train()`/`eval()` mode propagation — ✅ IMPLEMENTED (2026-07-23)

**Gap found while surveying UX proposals:** `Module` had no train/eval
concept at all. `Dropout` (`kindle-core/src/nn/dropout.rs`) already had its
own local `is_training: bool` gating identity-vs-random-zeroing behavior,
but nothing let a caller flip it network-wide — a user would have had to
reach into a `#[module]`-built model tree by hand to find every nested
`Dropout` and set its flag individually. Checked `BatchNorm2d` too, since
PyTorch's `.eval()` also freezes BatchNorm's running-stat updates: its
`forward` (`cpu/ops/norm.rs::batch_norm_impl`) always normalizes using the
supplied running statistics regardless of mode — a **pre-existing,
deliberately-documented "inference-mode-only" decision** (`_momentum: f64,
// deliberately unused — inference-mode-only (CONTEXT.md carried-forward
decision)`), not a bug. Reversing that is a separate, larger, cross-backend
proposal (real batch statistics + running-stat updates on 3 backends) —
explicitly out of scope here; `TrainMode` propagates through `BatchNorm2d`
as an honest no-op rather than silently overclaiming a behavior change.

**Design, and the non-obvious pitfall hit while building it:** added
`TrainMode` (`train()`/`eval()`/`set_training(bool)`) to
`kindle-core/src/nn/module.rs`, plus an `AutorefTrainMode`/
`AutorefTrainModeFallback` pair mirroring `Parameters`/`StateDict`'s own
autoref-specialization pattern exactly, and `kindle-macros/src/module.rs`
auto-generates `impl TrainMode` for every `#[module]`-derived struct (one
more `train_mode_calls` collection, same shape as `param_calls`/
`load_state_calls`). `set_training` defaults to a no-op so any stateless
leaf layer can opt in with a bare `impl TrainMode for X {}`.

**The pitfall:** `Sequential<L1, L2>`'s hand-written impl was *first*
written unconditionally (no `L1: TrainMode`/`L2: TrainMode` bound), reusing
the same autoref delegation the macro uses for its fields, reasoning that
this would let `.eval()` "reach through" to whichever side happens to be a
`Dropout` without requiring every leaf type to implement the trait. **This
does not work, and was caught by an actual failing test, not by review**:
autoref-specialization only resolves to the "real" impl when the compiler
can *prove* the bound holds while type-checking the *generic* code — for
`Sequential<L1, L2>`'s bare, unconstrained `L1`/`L2`, that proof is
impossible regardless of what the caller eventually instantiates them as,
so it *always* silently picked the no-op fallback. Confirmed empirically
(temporary `eprintln!` instrumentation) that the exact same code shape
resolves correctly for `#[module]`'s own per-field calls (`Param<S,B>`,
`Buffer<S,B>` are concrete type constructors with unconditional trait impls
at the `impl` site) but not for `Sequential`'s bare type parameters — this
is a real, general limitation of the autoref trick, not specific to
`TrainMode`. Fixed by matching `Parameters`/`StateDict`'s own existing
`Sequential` impl pattern instead: explicit `L1: TrainMode, L2: TrainMode`
bounds, direct (non-autoref) method calls. This is *why* every existing
stateless leaf layer (`Linear`, all 8 activations in `activation.rs`,
`AdaptiveAvgPool2d`, `AvgPool2d`, `MaxPool2d`, plus everything `#[module]`-
derived) needed the one-line `impl TrainMode for X {}` opt-in — without it,
the `Sequential` bound would be unsatisfiable for any real `seq!`-built
chain that isn't 100% `Dropout`. `RNN`/`RNNCell`/`LSTM`/`LSTMCell` were
deliberately skipped: their `forward` signature returns a tuple
(`(output, hidden_state)`), which doesn't compose through `Sequential`'s
`Module<Input>` chaining in the first place, so there's no realistic case
needing them to implement `TrainMode` today.

1 new test (`kindle/tests/nn_tests.rs::test_train_mode_propagates_through_sequential_dropout`)
— not just a shape check: builds `seq!(Linear, Dropout::new(0.9))`, calls
`.eval()`, and asserts the output *exactly* equals calling the `Linear`
layer alone (eval-mode `Dropout` is a true identity function, so this is a
real, deterministic, non-probabilistic correctness check, not a "probably
different" random-seed comparison). Full verification loop (fmt/clippy
workspace CPU+WGPU/tests, plus `cargo check --features cuda,std` since
`kindle-core` is shared) all green, 0 regressions.

---

## 9. New feature proposals — require explicit user sign-off before implementing

### 9.1 Expose the three orphaned fused CUDA kernels (§1.9)

`matmul_swiglu.cu`, `flash_attention_lite.cu`, `one_hot.cu` all exist,
compile, and have zero callers anywhere in the Rust codebase. Each would
require a **new public `Backend` trait method** (semver-breaking, needs
sign-off per §0.3) or a CUDA-only inherent method (avoids the trait-breaking
problem, but means `SwiGLU`/`FlashAttention` wouldn't be portable across
backends the way everything else in this framework deliberately is — a real
design tension worth raising explicitly with the user before picking either
path, not deciding silently).

### 9.2 Quantization-aware training (QAT) via straight-through estimator

Already scoped in `ROADMAP.md`'s "Detailed Post-Phase Release Plan," Phase 4:
`d(quantize)/dx ≈ 1` during backward. Genuinely new design work (which
existing tape/autograd hook would carry the STE, given `quantize`/
`dequantize` currently have **no** tape entries on any backend — see §1.6).
Not started anywhere in the current codebase; treat `ROADMAP.md`'s existing
description as the starting spec if the user wants to proceed, don't
re-derive from scratch.

### 9.3 CUDA autotune cache persistence

Also already scoped in `ROADMAP.md` ("Cache Persistence & Key Identity" —
`~/.cache/kindle/autotune.json`, keyed by device UUID + compute capability +
driver version + `KernelKey`). The in-memory autotuning coordinator this
session's work (`tuning.rs`) already built is real and unit-tested per
`ROADMAP.md`'s High-priority section — persistence is additive, not a
rewrite. Low risk to existing behavior since it's opt-in (`autotune`
feature), but still needs real CUDA hardware to validate the cache actually
speeds up a second run, so treat any implementation here as compile-verified
only, same caveat as the rest of Phase 1.

### 9.4 Benchmark suite (`benches/`)

Scoped in `ROADMAP.md`'s Phase 5 ("Kindle vs Candle vs PyTorch" Criterion
benchmarks). No `benches/` directory currently exists in any crate
(`find crates -name benches -type d` returns nothing as of this document) —
this is greenfield, not a partial-completion task. Sequence this after
Phase 1 (CUDA) lands, otherwise a benchmark suite would only ever exercise
CPU/WGPU and give a skewed picture.

---

## 11. End-to-End User Workflow Additions — ✅ DONE (2026-07-23)

To address the complete lifecycle (Implementation -> Training -> Testing/Validation -> Datasets -> Persistence):

1. **Data Augmentation & Processing (`kindle-data::transforms`)**:
   - Added `Transform` trait (`fn transform(&self, input: Self::Input) -> Result<Self::Output>`).
   - Implemented `Normalize` (per-channel image normalization with `imagenet()` preset), `Scale`, `RandomHorizontalFlip`, `CenterCrop`, and `Compose` (pipeline chaining).
   - Re-exported in `kindle-data` and `kindle` preludes.

2. **Optimizer Checkpointing (`StateDict` for `AdamW`/`Adam`/`SGD`)**:
   - Implemented `state_dict`, `load_state_dict`, `step_count`, `set_step_count`, and `StateDict<B>` for `AdamW`, `Adam`, and `SGD` in `kindle-core/src/optim/mod.rs`.
   - Preserves $m$ and $v$ momentum tensors and step counters across checkpoints. Unit-tested in `optim_tests.rs`.

3. **Evaluation Metrics Library (`kindle-core::metrics`)**:
   - Added `Metric` trait (`value()`, `reset()`).
   - Implemented classification and regression metrics: `Accuracy`, `Precision`, `Recall`, `F1Score`, `MSE`, and `ConfusionMatrix`.
   - Re-exported in `kindle-core` prelude.

4. **Model Architecture Visualization (`model.summary()`)**:
   - Added `summary(&self) -> String` to `NamedLayers` and `format_layer_summary` in `kindle-core/src/nn/module.rs`.
   - Generates human-readable printable tree tables showing layer hierarchy, names, types, and shape/parameter specifications. Unit-tested in `named_layers_tests.rs`.

5. **HuggingFace Hub Shortcut (`from_pretrained`)**:
   - Added `kindle::hub::from_pretrained(repo_id, filename, device)` and `HubRepo::load_safetensors` to download model weights directly from HuggingFace Hub into a state map.

---

## 12. Public API Re-exports, Bare-Metal (`no_std`) Audit & Release Roadmap — ✅ DONE (2026-07-23)

### 12.1 Public API & Namespace Cleanups
- **Single Entry Point (`kindle`)**: Re-exported all core subsystems directly under `kindle` (`kindle::nn`, `kindle::optim`, `kindle::metrics`, `kindle::data`, `kindle::transforms`, `kindle::hub`, `kindle::typenum`). Users can import everything cleanly from the `kindle` root crate without reaching into internal `kindle-core` dependencies.
- **Prelude Pollution Removal**: Removed internal codegen macros (`alloc::format`, `alloc::vec`, `B0`, `B1`, `Bit`, `Diff`, `Prod`, `Quot`, `Sum`, `UInt`, `UTerm`, `Unsigned`) from `kindle-core` and `kindle` preludes. `kindle::typenum` remains available for explicit type-level integer access.

### 12.2 PyTorch Sequential Parity Verification
- **Sequential Flat State-Dict Keys**: Confirmed `Sequential<L1, L2>` state-dict extraction generates flat PyTorch-indexed keys (`0.weight`, `0.bias`, `1.weight`, `1.bias`) rather than nested structures, ensuring seamless compatibility with PyTorch `.safetensors` checkpoints.

### 12.3 Bare-Metal (`no_std`) Compatibility
- `kindle-core` and `kindle-data` use `#![cfg_attr(not(feature = "std"), no_std)]` with explicit `alloc` references (`Vec`, `BTreeMap`, `Box`, `String`), maintaining full compatibility for embedded/bare-metal targets without leaking `std` dependencies.

### 12.4 Roadmap Steps Towards 0.2.0 / 1.0 Release

```
[Phase 0: Triage] ──► [Phase 1: CUDA Backend] ──► [Phase 2: Parity Suite] ──► [Phase 3: CI Setup]
       │                        │                        │                        │
       ▼                        ▼                        ▼                        ▼
     ✅ DONE                  ✅ DONE                  ✅ DONE            (WGPU CI ready;
                                                                           CUDA compile-check)
                                                                                  │
                                                                                  ▼
                                                                        [Release 0.2.0-beta.1]
```

1. **Step 1: Release `0.2.0-beta.1`**:
   - Tag release candidate incorporating backend completion, PyTorch `Sequential` key parity, optimizer state checkpointing, metrics, data transforms, and `model.summary()`.
2. **Step 2: WGPU GitHub Actions CI**:
   - Enable software-rendered Vulkan/WGPU testing (`llvmpipe`) in GitHub Actions workflow.
3. **Step 3: Optional CUDA Kernel Extensions (§9.1)**:
   - Expose orphaned CUDA kernels (`fused_matmul_swiglu`, `flash_attention_lite`, `one_hot`) behind explicit feature flags once GPU testing hardware becomes available.


