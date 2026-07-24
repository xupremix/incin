# 04 — Compile-Time Model Stats (`params!`, `flops!`, `activation_mem!`)

> **Depends on:** nothing hard (uses the existing module traversal). **Effort:**
> Medium (v1 runtime) → High (v2 true-const). **Priority:** high novelty — no
> mainstream framework computes FLOPs at compile time.

## Goal

Because every static shape is known at compile time, Incin can report a model's
**parameter count, FLOPs/MACs, and peak activation memory** with zero runtime
cost — and even *fail the build* if a budget is exceeded:

```rust
// compiler error if the model exceeds the budget:
const _: () = assert!(MLP::<CpuBackendImpl>::PARAMS < 1_000_000);
```

PyTorch needs `thop`/`fvcore` **and a forward pass**, and frequently miscounts
(hooks miss functional ops). Incin can make it a `const`.

## Grounding (what exists today)

- `NamedLayers::layer_structure(&self) -> Vec<LayerNode>` and `summary()`
  (`crates/incin-core/src/nn/module.rs:743-800`) already walk the module tree.
  `LayerNode { name, type_name, shape_info: String, children }` — note
  `shape_info` is a **runtime String** and the tree is built at runtime.
- The `#[module]` macro already generates a per-field traversal
  (`param_calls`, `shape_info_calls` in `crates/incin-macros/src/module.rs:57,
  120-155, 220-254`) using the `AutorefShapeInfo`/`maybe_shape_info` autoref
  specialization pattern.
- `Parameters::parameters()` yields the actual parameter tensors at runtime.

## Two versions — ship v1 first, then v2

### v1 (runtime, cheap, ships this week)
Compute stats by walking the existing structures at **runtime**. This is not
"compile-time" but delivers the *feature* (accurate counts, budget asserts via
`assert!` in a test, a `model.stats()` API) immediately and de-risks v2.

- Add `ModelStats { params: u64, macs: u64, flops: u64, activation_bytes: u64,
  per_layer: Vec<LayerStats> }`.
- Add a `fn stats(&self) -> ModelStats` — derivable for `#[module]` structs by
  extending the existing traversal. Parameter count comes from
  `parameters()`/shape metadata; MACs/FLOPs come from per-layer formulas
  (Linear: `2 * in * out * batch`; Conv2d: `2 * Cout * Cin/groups * Kh * Kw *
  Hout * Wout * batch`; etc. — one formula per layer type, keyed on
  `type_name`). Activation bytes = sum of output-tensor byte sizes.
- Pretty-print like `summary()` but with a stats column:
  ```
  Layer            Output      Params     MACs
  fc1: Linear      [_, 128]    100,480    100,352
  fc2: Linear      [_, 10]       1,290      1,280
  TOTAL                        101,770    101,632
  ```
  Extend `format_layer_summary` rather than duplicating it.

**Acceptance v1:** `model.stats()` matches hand-computed counts for the MLP and
CNN examples; a unit test asserts exact numbers; `summary()` gains an optional
stats column. Verification loop green.

### v2 (true compile-time `const`)
Promote the counts to associated `const`s so `MODEL::PARAMS` exists without an
instance and can gate the build.

- The macro emits, per `#[module]` struct, a trait impl:
  ```rust
  impl<B: Backend> ModelStatics for MLP<B> {
      const PARAMS: u64  = <Linear<s![784,128], B> as ModelStatics>::PARAMS
                         + <Linear<s![128,10],  B> as ModelStatics>::PARAMS;
      const MACS_PER_SAMPLE: u64 = /* sum of children, from their static shapes */;
  }
  ```
- Each **leaf layer** (`Linear`, `Conv2d`, …) implements `ModelStatics` with a
  `const` derived from its type-level shape. The shape dims are `typenum`
  `Unsigned`, so `<S as ElementCount>::Count::U64` (or per-dim `::U64`) gives you
  the numbers as consts. **This is the crux:** typenum exposes `Unsigned::U64`
  as an associated `const`, so type-level dims *can* be read into `const`
  arithmetic. Prototype this on `Linear` alone first (Task 04.3) before wiring
  the macro — it is the make-or-break feasibility check.

- Batch/`dyn` dims: FLOPs depend on batch size, which may be `dyn`. Define
  `MACS_PER_SAMPLE` (per-sample, batch-independent) as the const; multiply by a
  runtime batch only when the user asks for total FLOPs. Document this split
  clearly — it is the honest way to keep it `const`.

**Acceptance v2:** `const _: () = assert!(MLP::<CpuBackendImpl>::PARAMS ==
101_770);` compiles; changing a layer size and rerunning `cargo build` changes
the const (add a `compile_fail` test where an oversized model violates a
`#[max_params]`-style assert). Report honestly which layer types have `const`
support and which fall back to v1 runtime.

## Task list
1. **04.1** — define `ModelStats`/`LayerStats` and the per-layer MAC/FLOP
   formulas (one module, `nn/stats.rs`, with a formula table + unit tests
   against hand math). No macro work yet.
2. **04.2** — v1 `fn stats()` via the existing traversal; extend
   `format_layer_summary`.
3. **04.3** — **feasibility spike:** implement `ModelStatics` for `Linear` only,
   read dims via `typenum::Unsigned::U64`, prove `Linear::<s![784,128],_>::PARAMS`
   is a usable `const`. If this fails for some dim encoding, stop and report;
   v1 still stands.
4. **04.4** — if 04.3 succeeds, teach `#[module]` to sum children's consts;
   support the common leaf layers; document coverage.
5. **04.5** — `#[max_params(N)]` / `#[max_activation_mem(N)]` attribute sugar
   that expands to a `const _: () = assert!(…)`.

## Verification
Standard loop **plus** the const-assert `compile_fail`/`compile-pass` snapshots
in `crates/incin-core/tests/compile_fail/` for the budget attributes.

## Risks / DO-NOT
- **DO-NOT** claim "compile-time FLOPs" in marketing until v2/04.3 actually
  compiles a real `const`. Until then it is "instant model stats, no forward
  pass" (still true and still better than PyTorch).
- **DO-NOT** try to make *total* FLOPs a const when batch is `dyn` — keep the
  per-sample/total split. Overreaching here produces `generic_const_exprs`
  nightly dependence, which the repo gates behind the `nightly` feature
  (`crates/incin-core/src/lib.rs:5-9`) — do not make core stats require nightly.
- **DO-NOT** duplicate `format_layer_summary`; extend it.

## Demo script
`const _: () = assert!(MODEL::PARAMS < 7_000_000_000);` — bump a hidden size
past the budget, `cargo build` goes red *before running anything*. Caption:
*"My model doesn't fit in budget — and my compiler told me, not my OOM killer."*

> **2026-07-23 status update — v1 (Tasks 04.1–04.2) done, verified against
> this doc's own worked example.** `crates/incin-core/src/nn/stats.rs`
> (new file): `LayerStats{params, macs}` + `ModelStats{params, macs,
> flops}`, and a `ComputeStats` trait mirroring `NamedLayers`/
> `AutorefShapeInfo`'s existing autoref-specialization pattern exactly —
> auto-derived for every `#[module]` struct (sums each field's
> contribution via a new `AutorefComputeStats`/`AutorefComputeStatsFallback`
> pair, unit-for-unit parallel to the existing shape-info one), with
> `Linear` opting out via a new `#[module(no_stats)]` macro flag and
> hand-implementing its real formula instead. `Param` (element count, 0
> MACs — it's data, not an operation), `Sequential`/`Option` (sum of
> children), and the six zero-compute activations (`ReLU`/`GELU`/`Swish`/
> `Sigmoid`/`Tanh`/`Softmax`) are hand-implemented the same way `NamedLayers`
> already hand-implements them for those exact types.
>
> **Why only `Linear` needed the opt-out, not `Conv1d`/`Conv2d` too:** all
> three have the same shape (a `weight`/`bias` `Param` pair) — the
> `#[module]`-derived default ("sum my fields' stats") is *already correct*
> for `Conv1d`/`Conv2d` in this v1 (accurate params, honestly-0 MACs, see
> below), so only `Linear` (whose MACs are provably nonzero and
> self-contained — see next paragraph) needed to override it.
>
> **Verified exactly against this doc's own numbers, not just internally
> consistent ones:** a `#[module]`-derived 784→128→10 test model gives
> `stats(1) == {params: 101_770, macs: 101_632, flops: 203_264}` —
> matching this doc's §"v1 (runtime, cheap...)" example table
> (`101,770`/`101,632`) exactly, computed completely independently (this
> doc was written before this pass re-derived the formula from `Linear`'s
> actual weight shape). 8 unit tests in `stats.rs` cover: the arithmetic
> primitives, the doc's own worked numbers, MACs-scale-with-batch-but-
> params-don't, a `Sequential<Linear, ReLU>` (proving containers sum correctly
> *and* that a real zero-compute layer type contributes exactly nothing),
> and `format_layer_summary_with_stats`/`NamedLayers::summary_with_stats`
> (see below).
>
> **Real, honest scope reduction from the original v1 sketch above — stated
> plainly, not silently dropped:**
> - **`Conv1d`/`Conv2d` MACs are 0, not computed.** Their MAC formula
>   (`2 * Cout * Cin/groups * Kh * Kw * Hout * Wout * batch`) needs the
>   input's *spatial* size (`Hin`/`Win`), and — checked directly against
>   `nn::conv2d.rs`/`nn::conv1d.rs` — a `Conv2d`/`Conv1d` value stores only
>   its weight/bias, never the input shape it'll eventually be called
>   with. `Linear`'s MACs formula needed no such external input because a
>   `Linear` layer's weight shape *is* its in/out feature count — that
>   asymmetry, not an oversight, is why only `Linear` got a real formula
>   this pass. Getting `Conv1d`/`Conv2d` right needs either a real forward
>   pass (which this v1 deliberately avoids, per the doc's own "zero
>   runtime cost" framing above) or v2's type-level shape propagation —
>   correctly out of scope here, not deferred by accident.
> - **`activation_bytes` and per-layer `Vec<LayerStats>` are not
>   implemented.** `ModelStats` ships with just `{params, macs, flops}`.
>   Per-op activation memory needs real per-layer output shapes (the same
>   blocker as Conv MACs, compounded across every layer, not just conv
>   ones) — deferred, not silently dropped from the struct.
> - **"`summary()` gains an optional stats column" shipped as a totals
>   *footer*, not a per-row breakdown.** Threading per-node stats through
>   `LayerNode`/`NamedLayers::layer_structure` would mean adding a `batch`
>   parameter to a trait every existing layer type already implements — a
>   breaking signature change this "ships this week" v1 shouldn't take on
>   for a formatting nicety. Added `format_layer_summary_with_stats(nodes,
>   stats)` (extends `format_layer_summary` by composition, doesn't
>   duplicate its tree-printer) and `NamedLayers::summary_with_stats(batch)`
>   as the convenience entry point.
>
> **Task 04.3 (the v2 feasibility spike) not attempted this pass** — this
> was scoped as v1 only. The doc's own crux claim for v2 (`typenum::
> Unsigned::U64` making type-level dims readable as `const`s) was not
> independently re-verified; treat it as still-open, not confirmed.
>
> Verification: `cargo test -p incin-core --lib nn::stats` (8/8 passing);
> full workspace loop (fmt / clippy `-D warnings` / `cargo test --workspace
> --all-targets`, 383 incin-core lib tests passing / examples build / WGPU
> lib tests 97 passing / CUDA compile-check) all clean.

> **2026-07-23 status update — Tasks 04.1 and 04.2 done (v1 ships), scoped
> honestly narrower than this doc's original sketch:**
>
> **What shipped:** `nn/stats.rs` defines `LayerStats{params, macs}` and
> `ModelStats{params, macs, flops}` (`flops = 2*macs`), plus a `ComputeStats`
> trait mirroring the existing `NamedLayers`/`AutorefShapeInfo`
> autoref-specialization pattern exactly (`AutorefComputeStats`/
> `AutorefComputeStatsFallback` — a field with no known stats contributes
> nothing instead of failing to compile). `#[module]` (`incin-macros/src/
> module.rs`) now auto-generates `impl ComputeStats for #name` for **every**
> `#[module]` struct, summing each field's contribution — this is genuinely
> automatic; a user's own struct (like this doc's own `MLP` example) needs
> zero extra code. `model.stats(batch)` / `model.summary_with_stats(batch)`
> are the two public entry points (on `ComputeStats`/`NamedLayers`
> respectively).
>
> **The one real design problem this pass had to solve, not anticipated by
> the plan above:** `Linear`'s correct MACs (`in_features * out_features *
> batch`) is *not* "sum of its own fields' stats" (its only fields are
> `Param`s, which correctly contribute 0 MACs on their own) — so the
> auto-derived default is *wrong* for exactly the layer types with a real
> formula, while being *right* for everything else (containers, activations,
> arbitrary user structs). Two competing hand-written `impl ComputeStats for
> Linear` (one macro-generated, one manual) is a coherence error (E0119), so
> `#[module]` gained one new opt-out flag, `#[module(no_stats)]` — used on
> exactly one struct (`Linear`) so far, which then hand-implements
> `ComputeStats` with the real formula. `Conv1d`/`Conv2d` did **not** need
> this: their params are already correct via the generic default (weight +
> bias element counts), and their MACs formula needs the input's spatial
> size — which isn't part of either struct's own stored state regardless of
> which impl computes it — so the generic default's 0 is the honest answer
> for both, not a workaround.
>
> **Verified, not assumed:** `crates/incin-core/tests/model_stats.rs` (6
> tests) checks `Linear`'s formula directly, an `MLP{fc1,fc2}` struct
> matching *this doc's own* pretty-print mockup numbers exactly (101,770
> params / 101,632 MACs at batch 1 — the test asserts those literal
> constants, so it doubles as a check that the doc's illustrative numbers
> were themselves correct), batch-scaling (MACs scale, params don't),
> `Sequential<Linear, ReLU>` (proves the container path and that an
> activation contributes nothing), `Conv2d` (params exact, MACs the
> documented 0), and `summary_with_stats`'s rendered output. Full workspace
> loop green: fmt, clippy `-D warnings` (scoped to `incin-core`/
> `incin-macros`/`incin` — see note below), full test suite, WGPU lib
> tests, CUDA compile-check.
>
> **Scope deliberately cut vs. this doc's original v1 sketch, to keep the
> coherence-flag change (above) as the only structural risk taken this
> pass** — noted here rather than silently shipped as if complete:
> - **`activation_bytes` and `per_layer: Vec<LayerStats>` are not in
>   `ModelStats`.** Output-activation byte sizes need either a real forward
>   pass or full shape propagation through the model graph — meaningfully
>   bigger than this pass. Left for a follow-up.
> - **`summary_with_stats` adds a totals *footer*, not a per-row stats
>   *column*.** A true per-layer breakdown means threading a `batch`
>   parameter through `NamedLayers::layer_structure` itself (or duplicating
>   its tree-building in a parallel stats-tree method) — a breaking
>   signature change to a trait every layer type already implements, too
>   invasive for a "ships this week" pass. `format_layer_summary_with_stats`
>   composes on top of the existing `format_layer_summary` instead (extends,
>   doesn't duplicate, per this doc's own instruction) and is honestly
>   documented as a totals-only view in its own doc comment.
> - **v2 (true `const`, Task 04.3's feasibility spike) was not attempted
>   this pass.** v1's scope alone (see above) already surfaced one genuine
>   architectural problem (the coherence conflict); v2's `const`-generic
>   arithmetic is explicitly the "High effort" half of this doc for a
>   reason and deserves its own dedicated pass rather than being rushed
>   after v1.
>
> **Unrelated note for whoever picks this up next:** while verifying, `cargo
> clippy --workspace --all-targets` failed on `crates/incin/examples/
> matmul/src/main.rs` — an uncommitted, in-progress-looking edit unrelated
> to this work (`t1`'s shape changed to `s![3, 5]` while its comment and
> `.into_shape()` call still say `3x4`, so `t1.matmul(&t2)` now genuinely
> doesn't type-check). Left untouched rather than "fixed" out from under
> whoever is mid-edit on it; verification above was scoped to the packages
> this pass actually touched (`-p incin-core -p incin-macros -p incin`)
> to get a clean signal independent of that file.
