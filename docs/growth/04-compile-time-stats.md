# 04 — Compile-Time Model Stats (`params!`, `flops!`, `activation_mem!`)

> **Depends on:** nothing hard (uses the existing module traversal). **Effort:**
> Medium (v1 runtime) → High (v2 true-const). **Priority:** high novelty — no
> mainstream framework computes FLOPs at compile time.

## Goal

Because every static shape is known at compile time, Kindle can report a model's
**parameter count, FLOPs/MACs, and peak activation memory** with zero runtime
cost — and even *fail the build* if a budget is exceeded:

```rust
// compiler error if the model exceeds the budget:
const _: () = assert!(MLP::<CpuBackendImpl>::PARAMS < 1_000_000);
```

PyTorch needs `thop`/`fvcore` **and a forward pass**, and frequently miscounts
(hooks miss functional ops). Kindle can make it a `const`.

## Grounding (what exists today)

- `NamedLayers::layer_structure(&self) -> Vec<LayerNode>` and `summary()`
  (`crates/kindle-core/src/nn/module.rs:743-800`) already walk the module tree.
  `LayerNode { name, type_name, shape_info: String, children }` — note
  `shape_info` is a **runtime String** and the tree is built at runtime.
- The `#[module]` macro already generates a per-field traversal
  (`param_calls`, `shape_info_calls` in `crates/kindle-macros/src/module.rs:57,
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
in `crates/kindle-core/tests/compile_fail/` for the budget attributes.

## Risks / DO-NOT
- **DO-NOT** claim "compile-time FLOPs" in marketing until v2/04.3 actually
  compiles a real `const`. Until then it is "instant model stats, no forward
  pass" (still true and still better than PyTorch).
- **DO-NOT** try to make *total* FLOPs a const when batch is `dyn` — keep the
  per-sample/total split. Overreaching here produces `generic_const_exprs`
  nightly dependence, which the repo gates behind the `nightly` feature
  (`crates/kindle-core/src/lib.rs:5-9`) — do not make core stats require nightly.
- **DO-NOT** duplicate `format_layer_summary`; extend it.

## Demo script
`const _: () = assert!(MODEL::PARAMS < 7_000_000_000);` — bump a hidden size
past the budget, `cargo build` goes red *before running anything*. Caption:
*"My model doesn't fit in budget — and my compiler told me, not my OOM killer."*
