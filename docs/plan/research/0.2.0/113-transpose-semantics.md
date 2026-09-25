# #113 — transpose semantics: SOTA survey + local CPU measurement

Research lane S1 for [issue #113](https://github.com/xupremix/incin/issues/113).
The maintainer's directive was to measure locally *and* research what SOTA
frameworks do, with the decision following the evidence. This memo is the
evidence. It implements nothing.

Related: [typed-layout-decisions.md](typed-layout-decisions.md) (the section on
this, including the prior CUDA measurement) and
[proof-directed-codegen.md](proof-directed-codegen.md) (the CUDA-scoped
strided-path finding this issue corrects).

## 1. SOTA survey: who promises views, who promises copies

| Framework | Transpose semantics | Materialization API | Contract (quote / link) |
|---|---|---|---|
| PyTorch (`torch.transpose`, `permute`) | **View, everywhere.** Result shares storage with the input; may be non-contiguous. Same on CPU and CUDA. | `.contiguous()` — returns self if already contiguous, else copies. | "If `input` is a strided tensor then the resulting `out` tensor shares its underlying storage with the `input` tensor" — [torch.transpose](https://docs.pytorch.org/docs/main/generated/torch.transpose.html); view list including `transpose`/`permute` — [Tensor Views](https://docs.pytorch.org/docs/main/tensor_view.html) |
| `torch.compile` (Inductor) | Views survive as `ReinterpretView` IR nodes; pointwise consumers fuse over them. A copy is inserted only where a layout constraint forces it (e.g. explicit `.contiguous()`, `view()`→`reshape` rewrite for layout optimization, matmul tuning workarounds). | Same user-level `.contiguous()`; compiler inserts the rest. | Evidence: `compile_fx.py` `view_to_reshape` ("an contiguous tensor in eager mode may becomes a channels last tensor"); `ReinterpretView` handling in `lowering.py`/`ir.py` — [pytorch source](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/compile_fx.py) |
| JAX (`jnp.transpose`, via XLA) | **Copy at the language level, lazy op underneath.** Eager `jnp.transpose` returns a copy (JAX arrays are immutable; there are no views at all). Under `jit`, transpose is an HLO op in the graph and XLA's fusion decides whether any copy survives. | No user-level materialization API needed — the compiler "optimize[s]-away such copies when possible". | "Unlike `numpy.transpose()`, `jax.numpy.transpose()` will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize-away such copies when possible" — [jnp.transpose](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.transpose.html); same note in [jax.numpy.rst](https://github.com/google/jax/blob/master/docs/jax.numpy.rst) |
| TensorFlow (`tf.transpose`) | **Always a new tensor.** TF has no strided-tensor concept, so a transpose cannot be a metadata relabelling; elements are permuted into fresh storage on every backend. | N/A — there is nothing else it could be. | "In `numpy` transposes are memory-efficient constant time operations as they simply return a new view of the same data with adjusted `strides`. TensorFlow does not support strides, so `transpose` returns a new tensor with the items permuted." — [tf.transpose](https://www.tensorflow.org/api_docs/python/tf/transpose) |
| NumPy | **View whenever possible.** | `.copy()` / `np.ascontiguousarray()` when the caller wants density. | "`transpose(a).shape == a.shape[::-1]` … A view is returned whenever possible." — [numpy.transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html) |
| MLX | **Lazy graph op, no copy at construction.** Transpose enqueues into the lazy graph; nothing moves until evaluation/scheduling. (One-line entry; verified against the op reference, not a from-source audit.) | `mx.eval` forces evaluation; `.contiguous()`-style materialization where needed. | Op reference — [MLX array conversion / lazy semantics](https://ml-explore.github.io/mlx/build/html/usage/numpy.html) |
| tinygrad | **Lazy view.** `permute`/`transpose` lower to `Ops.PERMUTE` movement-op metadata on the `UOp` graph — "views are not copies"; reshape/permute/expand fuse into the consuming kernel at realize. | `.contiguous()` / `.realize()` materializes. | "Returns a tensor that is a transposed version of the original tensor" via `permute` — [tinygrad movement ops](https://docs.tinygrad.org/tensor/movement/); lazy-fusion account — [everything is a UOp](https://ssenthilnathan3.github.io/blog/tinygrad/) |
| Candle (Rust; in-tree adapter exists) | **View.** `Layout::transpose` swaps strides over the same `Arc`-shared storage; `Tensor::transpose` keeps `self.storage.clone()`. The in-tree adapter (`crates/incin-backends/src/external/candle/ops/tensor.rs`, `CandleTransposeOps::transpose`) forwards directly to it, so it inherits view semantics. | `.contiguous()` materializes. | `candle-core/src/layout.rs` (`transpose` swaps `stride`/`dims`, keeps `start_offset`); `candle-core/src/tensor.rs` ("Storages are also refcounted independently so that its possible to avoid copying the storage for operations that only modify the shape or stride") — [candle-core](https://github.com/huggingface/candle/blob/main/candle-core/src/tensor.rs) |

### The pattern to extract

Eager frameworks with mutable, strided storage (PyTorch, NumPy, Candle, and
tinygrad/MLX in lazy form) all converge on the same contract: **views
everywhere, with an explicit, caller-named materialization** (`.contiguous()`).
The promise is uniform across backends — PyTorch's transpose is a view on CUDA
too — and the escape hatch has a name, so a caller who needs density says so.

Functional/graph frameworks (JAX, TensorFlow/XLA) promise **values, not
storage**: eager JAX copies because immutability forbids aliasing, TF copies
because strides do not exist, and under `jit`/graph capture the *compiler*
decides what survives. There is no user-visible view-vs-copy question because
there is no user-visible storage.

What breaks for users when backends differ (i.e. the status quo this issue
reports) is threefold, and each has a SOTA mirror:

1. **Mutation visibility.** PyTorch documents shared storage precisely so users
   know a write through one alias appears in the other. Per-backend semantics
   make this unknowable — the portability hazard in the issue.
2. **Silent perf cliffs.** Code that relies on transpose being cheap (an
   attention block transposes constantly) silently pays allocation + kernel
   launch + `numel` of bandwidth on the copying backend. SOTA avoids this by
   making the cheap thing the *default* (view) and the expensive thing
   *explicit* (`.contiguous()`), never the reverse.
3. **Compile-time layout claims.** This is incin-specific and the sharpest of
   the three: a `RowMajor` result type is true on CUDA and false on CPU, so
   `transpose` can only return `Dyn` and every caller pays `into_row_major` to
   re-derive a proof the framework already knew. No surveyed framework lets two
   backends disagree about this — the contract is fixed once, at the operation
   level.

## 2. Local measurement (CPU only)

### Method

No criterion benches exist in `incin-core` (`crates/incin/benches/baselines.rs`
is a separate facade crate), so per the task brief this used a scratch
integration test (`crates/incin-core/tests/scratch_transpose_bench.rs`) that
printed timings and was **deleted before finishing** — `git status` is clean of
it. Each case times end-to-end `transpose + consume`, because that is the
actual decision (a transpose is never consumed zero times): arm "copy" is
`transpose_structural` (the materializing `TransposeExact` path), arm "view" is
`transpose_view`, both on `CpuBackendImpl`, `f32`. Sinks are `std::hint::black_box`
on the output tensor so no pass is eliminated. Warmup then timed reps;
reported as min/median/max in microseconds, plus the median view/copy ratio
(< 1 favors the view).

Premises verified at the measured shapes before timing: the copy arm returns
strides `[1024, 1]` for shape `[1024, 1024]` (dense row-major), the view arm
returns `[1, 1024]` (shared buffer, permuted strides).

- Machine: AMD Ryzen 7 7735HS (8C/16T), 16 MiB L3, single NUMA node.
  `available_parallelism` = 16; the CPU backend links rayon, so kernels may use
  the shared thread pool. No thread pinning, stock frequency scaling.
- Build: `--release` numbers below. A debug run was also taken and discarded
  for magnitudes (everything ~10x slower) but kept for one qualitative check —
  see notes.
- Data: `ones` fills (CPU kernels take no value-dependent branches).
- Honest error bars: min–max spread is typically ±5–15% of the median; the
  `2 adds` vs `4 adds` copy medians overlap (15.8 ms vs 14.9 ms) and should be
  read as noise, not as "4 reads are cheaper than 2".

### Results (`--release`, medians; full min/med/max in the table)

| Case | Copy med (us) | View med (us) | view/copy (med) | min–max spread |
|---|---|---|---|---|
| transpose-only, [1024, 1024] (1M el) | 8362.8 | 2.3 | **0.000** | copy 8054–9344, view 1.9–2.4 |
| single-read `sum_all`, [1024, 1024] | 8341.8 | 8295.7 | 0.994 | copy 7935–9503, view 7955–9541 |
| 1 chained add, [1024, 1024] | 7564.4 | 2769.4 | **0.366** | copy 7358–9663, view 2445–3511 |
| 2 chained adds, [1024, 1024] | 15847.3 | 5725.6 | **0.361** | copy 14538–16721, view 4924–7456 |
| 4 chained adds, [1024, 1024] | 14894.9 | 12308.4 | 0.826 | copy 14549–16202, view 11205–13567 |
| 8 chained adds, [1024, 1024] | 18622.1 | 24090.0 | **1.294** | copy 17935–19222, view 23167–26814 |
| one matmul [256,512]×[512,128] | 2689.7 | 1193.3 | **0.444** | copy 2358–3216, view 1168–1354 |
| two matmuls [256,512]×[512,*] | 3020.2 | 1946.9 | **0.645** | copy 2944–3154, view 1913–2126 |
| attention single-read `sum_all`, [2,4,128,128] (131k el) | 1797.5 | 1529.5 | 0.851 | copy 1754–1876, view 1490–1654 |
| attention 4 chained adds, [2,4,128,128] | 1976.2 | 1082.7 | **0.548** | copy 1939–2103, view 1048–1153 |

### Reading

- The single-read `[1024, 1024]` case is a **tie** (0.994): one strided pass
  costs about the same as strided-copy-plus-dense-read. The attention-shaped
  single read instead favors the view by 15% (0.851) — stride pattern and cache
  behavior matter, so there is no single constant here.
- The multi-read cases cross over **between 4 and 8 chained consumers**
  (0.826 → 1.294), replicating the shape of the prior CUDA finding (view +45%
  at one read, −23% at eight, crossover ≈ 4). Two independent backends, same
  crossover neighborhood.
- Matmul consumers favor the view at one *and* two reads (0.444, 0.645) in
  this size regime, where the copy overhead dominates the GEMM itself. Larger
  GEMMs amortize the copy and would move this; do not extrapolate.
- Note the copy itself is not free of strided reads: materializing a transpose
  reads the source in transposed order, which is why transpose-only copy costs
  ~8.3 ms for 4 MB. A "copy" is a strided-read + dense-write, not a memcpy.

## 3. What this measurement cannot answer (CUDA side)

This machine has no CUDA device (AMD CPU + Radeon iGPU), so nothing above says
anything about device memory behavior, where the calculus differs structurally:
a transposed read on GPU is the worst case for coalescing, the copy pays a
kernel launch plus `2 × numel` of device bandwidth, and fusion (of the kind
Inductor and XLA do) can dissolve either arm into its consumer. The prior CUDA
numbers in [typed-layout-decisions.md](typed-layout-decisions.md) (GTX 1650,
single pointwise consumer, stream-synchronized barrier) stand as the
device-side evidence; they agree with §2 in shape but are one GPU, one dtype,
square shapes, and a pointwise consumer.

A CUDA measurement that would close the gap needs the GPU CI runner tracked in
#82, running the same harness shape: transpose + consume-`k`-times for
`k ∈ {1, 2, 4, 8}`, plus a matmul consumer (a consumer that cannot take
arbitrary strides moves the crossover — the memo's own caveat), with a
stream-synchronize barrier rather than a host readback (the readback barrier
lesson from the prior round: a barrier costlier than the work biases every
ratio toward unity, which reads as "no difference").

## 4. Recommendation

**Materialize in `transpose` on every backend, and keep the no-copy path as a
separately named `transpose_view` — i.e. views-everywhere is wrong as the
default, materialize-everywhere is wrong as the only option, so the operation
splits in two and the caller picks.** The reasoning is the crossover, now
measured on two backends: the view wins by up to ~2.7x for few consumers and
loses by ~1.3x at eight, and the producing operation cannot know which regime
its caller is in — unifying on either behavior makes the framework reliably
wrong for half its callers, which is exactly what the SOTA survey says mature
frameworks refuse to do (PyTorch/TF/JAX each fix *one* contract per operation
name and never let backends disagree). The default named `transpose` should be
the materializing one: it matches TensorFlow/JAX value semantics, it matches
what CUDA already does, it keeps every downstream kernel on the dense fast
path, and — the type-level consequence from the issue — it makes the result
layout statable as `RowMajor` instead of `Unknown`/`Dyn`, deleting the
`into_row_major` recovery tax at every transpose call site. The view half keeps
the PyTorch/NumPy/Candle pattern (cheap by default, explicit by name) for the
single-consumer hot paths like attention scores, returning `Dyn` so a
`Contiguous` bound correctly refuses it until re-proven. Note this is also what
the working tree already converges on (`TransposeExact` copies on CPU,
`TransposeView` exists, public `transpose` states `RowMajor`); this memo
records that the evidence supports keeping that settlement, not revisiting it.

### What changes if recommended

Nothing in this branch — the settlement described above is already implemented
here. For the record, the change surface if it were *not* (or if the decision
went the other way) is:

- **Contract text location:** the `transpose` row in
  `crates/incin-core/src/operation_catalog.rs` (layout rule), regenerated
  `docs/OPERATION_SEMANTICS.md` (`INCIN_DOCS=overwrite`), the doc comments on
  `Tensor::transpose` / `transpose_view` in
  `crates/incin-core/src/tensor/ops/manipulation/transpose.rs`, and the
  capability rows in `incin-backends/src/capability/`.
- **Views-everywhere alternative (rejected):** CUDA stops copying —
  `launch_transpose` drops the permutation kernel and builds
  `CudaStorage::try_from_parts` with permuted strides; CPU keeps
  `transpose_storage`. `transpose` returns `Dyn` and `into_row_major` stays
  mandatory.
- **Materialize-everywhere (recommended, present):** CPU's `TransposeExact`
  copies via `transpose_exact_storage`
  (`crates/incin-backends/src/cpu/canonical/shape_ops.rs`); CUDA already did.
- **The pinned test flips accordingly:**
  `shape_changing_operations_produce_dense_results` in
  `crates/incin-core/tests/typed_layout.rs` currently asserts CPU transpose
  strides `[3, 1]` (dense); under views-everywhere it would assert `[1, 4]`
  (shared buffer) instead — the test is written to fail rather than drift,
  which is the property that caught this issue in the first place.
