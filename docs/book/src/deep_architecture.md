# The layered architecture

Incin is a workspace, not one crate. This chapter is the map of those layers
and the route data takes through them - the "how it is put together" companion
to [the target API](./target_api.md) and
[from proofs to execution](./proofs_to_execution.md) chapters. The narrative
version of the same material lives in `docs/GUIDE.md` §1-§2 and §6; the
binding version lives in `docs/FROZEN_FOUNDATIONS.md`. This chapter distills
both and adds the file-level index neither has room for.

## The layers

The layering mirrors the c10/aten/torch split of a classic PyTorch stack:
small core types at the bottom, an operation vocabulary above them,
concrete device implementations beside that, and one ergonomic facade on top.
Where PyTorch's layers meet at runtime through C++ dispatch, Incin's meet at
compile time through traits and type parameters. Arrows point at what a layer
depends on; everything above `incin-core` exists to make one core contract
ergonomic.

<svg class="incin-diagram" viewBox="0 0 780 300" role="img" aria-label="Layer stack: the incin facade depends on incin-macros and incin-backends, which depend on incin-core." xmlns="http://www.w3.org/2000/svg">
  <style>
    .dg1-box { fill: currentColor; fill-opacity: 0.05; stroke: currentColor; stroke-opacity: 0.4; stroke-width: 1; rx: 7; }
    .dg1-box-accent { fill: currentColor; fill-opacity: 0.03; stroke: var(--links, #2b79a2); stroke-width: 1.4; }
    .dg1-name { font: 600 14px ui-monospace, "Source Code Pro", Menlo, monospace; fill: currentColor; }
    .dg1-role { font: 12.5px system-ui, sans-serif; fill: currentColor; opacity: 0.75; }
    .dg1-note { font: italic 11.5px system-ui, sans-serif; fill: var(--links, #2b79a2); }
    .dg1-edge { stroke: var(--links, #2b79a2); stroke-width: 1.4; fill: none; marker-end: url(#dg1-arrow); }
  </style>
  <defs>
    <marker id="dg1-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--links, #2b79a2)"/>
    </marker>
  </defs>

  <!-- Facade -->
  <rect class="dg1-box-accent" x="150" y="16" width="480" height="52" rx="7"/>
  <text class="dg1-name" x="390" y="38" text-anchor="middle">incin</text>
  <text class="dg1-role" x="390" y="57" text-anchor="middle">facade: prelude, nn, optim, data, backend_authoring</text>

  <!-- Macros -->
  <rect class="dg1-box" x="40" y="112" width="330" height="52" rx="7"/>
  <text class="dg1-name" x="205" y="134" text-anchor="middle">incin-macros</text>
  <text class="dg1-role" x="205" y="153" text-anchor="middle">s! / shape! / tensor! / #[module] / model!</text>

  <!-- Backends -->
  <rect class="dg1-box" x="410" y="112" width="330" height="52" rx="7"/>
  <text class="dg1-name" x="575" y="134" text-anchor="middle">incin-backends</text>
  <text class="dg1-role" x="575" y="153" text-anchor="middle">cpu / cuda / wgpu / metal / candle + capability rows</text>

  <!-- Core -->
  <rect class="dg1-box-accent" x="150" y="208" width="480" height="52" rx="7"/>
  <text class="dg1-name" x="390" y="230" text-anchor="middle">incin-core</text>
  <text class="dg1-role" x="390" y="249" text-anchor="middle">Tensor, shapes, op catalog, descriptors, dispatch, autograd</text>

  <!-- Edges -->
  <path class="dg1-edge" d="M240,68 L205,108"/>
  <path class="dg1-edge" d="M540,68 L575,108"/>
  <path class="dg1-edge" d="M205,164 L240,204"/>
  <path class="dg1-edge" d="M575,164 L540,204"/>

  <!-- Annotations -->
  <text class="dg1-note" x="668" y="45">what you depend on</text>
  <text class="dg1-note" x="30" y="238" text-anchor="start">one execution</text>
  <text class="dg1-note" x="30" y="253" text-anchor="start">contract</text>
</svg>

| Crate | Role |
|---|---|
| `incin-core` | The `Tensor` type, the shape/dtype/device type system, autograd, `nn`/`optim`/`metrics`, the operation catalog and descriptor contract, distributed primitives. `no_std` unless a feature says otherwise |
| `incin-backends` | Concrete backends (`cpu`, `cuda`, `wgpu`, `metal`, `external::candle`) plus `incin_backends::target`, the allocation-target API |
| `incin-macros` | The procedural macros: `s!`, `shape!`, `idx!`, `tensor!`, `#[module]`, `dim!` support, `mesh!`/`placement!`, `model!`/`import_model!` |
| `incin-data` | `Dataset`, `DataLoader`, vision datasets, transforms, Hub downloading |
| `incin-viz` / `incin-telemetry` | Graph visualization and structured run emission |
| `incin` | The facade. Re-exports the rest under `incin::prelude`; opt-in surfaces are named `incin::state`, `incin::backend_authoring`, `incin::experimental` |

Almost every program writes `use incin::prelude::*;` and nothing else. The
facade is deliberately thin: it names things, it does not wrap or re-validate
them, so there is exactly one execution contract and the facade is a spelling
of it.

## One declaration, many consumers

The single most load-bearing file in the tree is
[`crates/incin-core/src/operation_catalog.rs`](../../../crates/incin-core/src/operation_catalog.rs).
It contains one macro-generated declaration of every stable operation - 174
rows, each naming an identity marker (`op::X`), a semantic profile, an
attributes type, and operand arity:

```rust,ignore
// crates/incin-core/src/operation_catalog.rs (excerpt; generated consumers
// expand these rows, they do not restate them)
(Zeros, "zeros", Fill, Creation, CreationAttributes, 0, 0, "Descriptor<op::Zeros>"),
(Relu, "relu", Pointwise, UnaryFloat, NoAttributes, 1, 1, "::relu"),
(MatMulExact, "matmul", Reduction, MatMul, NoAttributes, 2, 2, "::matmul"),
```

Everything else in the execution stack is expanded *from* that declaration,
which is what makes "advertised" and "implemented" impossible to drift apart:

| Consumer | Where | What it generates from the rows |
|---|---|---|
| Descriptor vocabulary | `crates/incin-core/src/exec/catalog/` | `op::X` markers, `Descriptor<O>`, `OperationCatalogEntry`, per-operation attribute contracts |
| Shape diagnostics | `crates/incin-core/src/shapes/error.rs` | exact `OperationKind` identities for errors |
| Capability rows | `crates/incin-backends/src/capability/declarations.rs` | grouped-by-rule-shape declarations feeding both the registry and the executor completeness proof |
| Generated docs | `docs/capabilities.md`, `docs/OPERATION_SEMANTICS.md` | per-operation semantics and backend support, regenerated by tests |

Adding an operation is an edit to the catalog and nowhere else. A row cannot
be added to one consumer only, because there are no other places to add rows.

## The route data takes

An ordinary tensor method runs this route (distilled from
[the lowering chapter](./deep_lowering.md), which walks it line by line):

<svg class="incin-diagram" viewBox="0 0 780 560" role="img" aria-label="Execution route: a typed tensor method lowers through TensorHandle and ExecutionContext into dispatch::execute_shaped, which infers outputs, validates payloads, admits capabilities, then calls the backend executor." xmlns="http://www.w3.org/2000/svg">
  <style>
    .dg2-node { fill: currentColor; fill-opacity: 0.05; stroke: currentColor; stroke-opacity: 0.4; stroke-width: 1; }
    .dg2-code { font: 600 13.5px ui-monospace, "Source Code Pro", Menlo, monospace; fill: currentColor; }
    .dg2-sub { font: 12px system-ui, sans-serif; fill: currentColor; opacity: 0.75; }
    .dg2-stage { font: italic 12px system-ui, sans-serif; fill: var(--links, #2b79a2); }
    .dg2-edge { stroke: var(--links, #2b79a2); stroke-width: 1.4; fill: none; marker-end: url(#dg2-arrow); }
    .dg2-inner { fill: none; stroke: currentColor; stroke-opacity: 0.25; stroke-dasharray: 4 3; }
  </style>
  <defs>
    <marker id="dg2-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--links, #2b79a2)"/>
    </marker>
  </defs>

  <!-- Frontend -->
  <rect class="dg2-node" x="190" y="14" width="400" height="46" rx="7"/>
  <text class="dg2-code" x="390" y="35" text-anchor="middle">Tensor&lt;S, B, K, G&gt; method</text>
  <text class="dg2-sub" x="390" y="52" text-anchor="middle">compile time: shapes resolve Output; B: Execute&lt;O&gt;</text>

  <path class="dg2-edge" d="M390,60 L390,84"/>

  <rect class="dg2-node" x="190" y="88" width="400" height="40" rx="7"/>
  <text class="dg2-code" x="390" y="108" text-anchor="middle">TensorHandle::from_storage</text>
  <text class="dg2-sub" x="390" y="122" text-anchor="middle">checked metadata read off real storage</text>

  <path class="dg2-edge" d="M390,128 L390,152"/>

  <rect class="dg2-node" x="190" y="156" width="400" height="40" rx="7"/>
  <text class="dg2-code" x="390" y="176" text-anchor="middle">ExecutionContext</text>
  <text class="dg2-sub" x="390" y="190" text-anchor="middle">grad mode = scope ceiling AND output marker</text>

  <path class="dg2-edge" d="M390,196 L390,220"/>

  <!-- Dispatch group -->
  <rect class="dg2-node" x="120" y="224" width="540" height="150" rx="7"/>
  <text class="dg2-code" x="390" y="248" text-anchor="middle">dispatch::execute_shaped::&lt;O, B, S&gt;</text>
  <rect class="dg2-inner" x="140" y="262" width="500" height="100" rx="5"/>
  <text class="dg2-stage" x="155" y="284">1</text>
  <text class="dg2-sub" x="172" y="284">infer_invocation_typed - outputs derived, typed shape cross-checked</text>
  <text class="dg2-stage" x="155" y="310">2</text>
  <text class="dg2-sub" x="172" y="310">payload validated against DataAttributes byte length</text>
  <text class="dg2-stage" x="155" y="336">3</text>
  <text class="dg2-sub" x="172" y="336">admit_invocation - exact capability row per operand,</text>
  <text class="dg2-sub" x="172" y="352">filtered by the fallback policy</text>

  <path class="dg2-edge" d="M390,374 L390,398"/>

  <rect class="dg2-node" x="190" y="402" width="400" height="40" rx="7"/>
  <text class="dg2-code" x="390" y="422" text-anchor="middle">Execute&lt;O&gt;::execute(ExecutionRequest)</text>
  <text class="dg2-sub" x="390" y="436" text-anchor="middle">the backend kernel</text>

  <path class="dg2-edge" d="M390,442 L390,466"/>

  <rect class="dg2-node" x="190" y="470" width="400" height="40" rx="7"/>
  <text class="dg2-code" x="390" y="490" text-anchor="middle">output storage</text>
  <text class="dg2-sub" x="390" y="504" text-anchor="middle">rebuilt under the derived shape</text>
</svg>

Three properties of that route are structural rather than conventional
(`docs/FROZEN_FOUNDATIONS.md`'s wording):

- **Support is explicit.** `B: Execute<O>` is a compile-time fact; the
  capability row is queried before launch; there is no default method to fall
  through.
- **Validation precedes execution.** The descriptor is validated against real
  storage metadata, never against what a caller claims.
- **Outputs are derived, not accepted.** The caller never states output
  metadata, so it cannot fabricate any.

## The files that carry each responsibility

When a stack trace or a design question lands you inside the tree, these are
the files to open first:

| Question | File |
|---|---|
| What operations exist? | [`operation_catalog.rs`](../../../crates/incin-core/src/operation_catalog.rs) |
| What does a descriptor look like? | [`exec/catalog/descriptor.rs`](../../../crates/incin-core/src/exec/catalog/descriptor.rs), [`exec/spec.rs`](../../../crates/incin-core/src/exec/spec.rs) |
| How are outputs inferred? | [`exec/catalog/inference.rs`](../../../crates/incin-core/src/exec/catalog/inference.rs) |
| Where is validation sealed? | [`exec/proof.rs`](../../../crates/incin-core/src/exec/proof.rs) |
| What runs before a kernel? | [`exec/dispatch.rs`](../../../crates/incin-core/src/exec/dispatch.rs) |
| What does an executor implement? | [`tensor/backend/execute.rs`](../../../crates/incin-core/src/tensor/backend/execute.rs) |
| What can a backend refuse, and why? | [`exec/capability.rs`](../../../crates/incin-core/src/exec/capability.rs) |
| How do backends declare support? | [`incin-backends capability declarations`](../../../crates/incin-backends/src/capability/declarations.rs) |
| What metadata reaches a kernel? | [`exec/meta.rs`](../../../crates/incin-core/src/exec/meta.rs) (`TensorMeta`) |

Each of those paths appears in `docs/FROZEN_FOUNDATIONS.md` with the
mechanism that keeps its contract true, and
[`frozen_foundations.rs`](../../../crates/incin-core/tests/frozen_foundations.rs)
fails if any of them moves.

## What does not exist

Two things readers often look for are absent by design:

- **There is no global "current device".** The backend is the type parameter
  `B`; which backend a tensor uses is fixed by the type, not by thread-local
  state. `best_device!()` picks `B` at compile time from features;
  `detect_device()` probes hardware at runtime and returns a value. They
  answer different questions ([the backends chapter](./backends.md)).
- **There is no second application-facing execution path.** Backend-local
  helpers and fused special sites are ordinary functions inside a backend.
  The former operation-family traits were removed from production source; a
  method on `Tensor` and a hand-built `dispatch::execute` call resolve to the
  same catalog entry, the same validation, and the same executor.

## Where to read next

- [`deep_type_semantics.md`](./deep_type_semantics.md) - the shape/proof types the
  frontend carries into that route.
- [`deep_lowering.md`](./deep_lowering.md) - descriptors, schema versioning,
  capture, and the dispatcher itself.
- [`deep_proofs.md`](./deep_proofs.md) - what keeps each claim above true:
  inventories, completeness proofs, and generated evidence.
- `docs/GUIDE.md` remains the prose tour; when it and a generated document
  disagree, believe the generated one.
