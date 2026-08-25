# The layered architecture

Most frameworks decide things at runtime that Incin decides at compile time,
and that single difference shapes the whole organization. This chapter is the
map: what the layers are, the route a tensor operation takes through them,
and why the route looks the way it does. It is the longer companion to
[the target API](./target_api.md),
[from proofs to execution](./proofs_to_execution.md), and
[backend authoring](./backend_authoring.md), which covers the hands-on parts.

## Four layers, one contract

The layering will look familiar if you know how a classic PyTorch stack is
organized: small core types at the bottom, an operation vocabulary above
them, concrete device implementations beside that, and one ergonomic facade
on top. The difference is where the layers meet. PyTorch's layers meet at
runtime through a dispatcher; Incin's meet at compile time through traits
and type parameters.

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

| Layer | Role |
|---|---|
| Core | The `Tensor` type, the shape/dtype/device type system, autograd, `nn`/`optim`/`metrics`, the operation catalog and descriptor contract |
| Backends | Concrete implementations (`cpu`, `cuda`, `wgpu`, `metal`, a Candle adapter) plus the allocation-target API |
| Macros | The compile-time surface: `s!`, `shape!`, `idx!`, `tensor!`, `#[module]`, `model!`/`import_model!` |
| Facade | Re-exports the rest under `incin::prelude`; opt-in surfaces live at `incin::state`, `incin::backend_authoring`, `incin::experimental` |

Almost every program writes `use incin::prelude::*;` and nothing else. The
facade is deliberately thin. It names things; it does not wrap or re-validate
them. That restraint is what keeps exactly one execution contract in the
system instead of one per entry point.

## Every operation is declared once

The heart of the design is a single catalog of operations. Each stable
operation gets exactly one declaration naming its identity, its semantic
profile (pointwise, reduction, creation, and so on), its attributes type, and
its operand arity. Everything else in the execution stack is derived from
that declaration rather than restated beside it:

- the typed descriptors your methods build,
- the output-shape inference rules,
- the per-operation error identities,
- the capability rows each backend registers against,
- the generated reference pages ([capabilities](https://github.com/xupremix/incin/blob/master/docs/capabilities.md),
  [operation semantics](https://github.com/xupremix/incin/blob/master/docs/OPERATION_SEMANTICS.md)).

Because there is one place where an operation exists, "what the docs
advertise" and "what a backend implements" cannot quietly drift apart.
Adding an operation is one declaration plus the executor work behind it;
there is no second list to update and no third one to forget.

## The route an operation takes

When you call a tensor method, control passes through five stops. The
compile-time part happens first, before the program even runs; the runtime
part is a fixed sequence of checks followed by the kernel.

<svg class="incin-diagram" viewBox="0 0 780 560" role="img" aria-label="Execution route: a typed tensor method resolves support and output types at compile time, builds a validated descriptor, passes capability admission, then reaches the backend kernel." xmlns="http://www.w3.org/2000/svg">
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
  <text class="dg2-sub" x="390" y="52" text-anchor="middle">compile time: output shape resolved; B must execute this operation</text>

  <path class="dg2-edge" d="M390,60 L390,84"/>

  <rect class="dg2-node" x="190" y="88" width="400" height="40" rx="7"/>
  <text class="dg2-code" x="390" y="108" text-anchor="middle">handles over real storage</text>
  <text class="dg2-sub" x="390" y="122" text-anchor="middle">metadata read off the allocations that will run</text>

  <path class="dg2-edge" d="M390,128 L390,152"/>

  <rect class="dg2-node" x="190" y="156" width="400" height="40" rx="7"/>
  <text class="dg2-code" x="390" y="176" text-anchor="middle">execution context</text>
  <text class="dg2-sub" x="390" y="190" text-anchor="middle">grad mode = scope ceiling combined with the output's marker</text>

  <path class="dg2-edge" d="M390,196 L390,220"/>

  <!-- Dispatch group -->
  <rect class="dg2-node" x="120" y="224" width="540" height="150" rx="7"/>
  <text class="dg2-code" x="390" y="248" text-anchor="middle">canonical dispatch</text>
  <rect class="dg2-inner" x="140" y="262" width="500" height="100" rx="5"/>
  <text class="dg2-stage" x="155" y="284">1</text>
  <text class="dg2-sub" x="172" y="284">outputs derived from inputs, cross-checked against the caller's type</text>
  <text class="dg2-stage" x="155" y="310">2</text>
  <text class="dg2-sub" x="172" y="310">attribute payloads checked against their declared byte length</text>
  <text class="dg2-stage" x="155" y="336">3</text>
  <text class="dg2-sub" x="172" y="336">capability admission per operand, filtered by the</text>
  <text class="dg2-sub" x="172" y="352">fallback policy</text>

  <path class="dg2-edge" d="M390,374 L390,398"/>

  <rect class="dg2-node" x="190" y="402" width="400" height="40" rx="7"/>
  <text class="dg2-code" x="390" y="422" text-anchor="middle">Execute&lt;O&gt;::execute(request)</text>
  <text class="dg2-sub" x="390" y="436" text-anchor="middle">the backend kernel</text>

  <path class="dg2-edge" d="M390,442 L390,466"/>

  <rect class="dg2-node" x="190" y="470" width="400" height="40" rx="7"/>
  <text class="dg2-code" x="390" y="490" text-anchor="middle">output storage</text>
  <text class="dg2-sub" x="390" y="504" text-anchor="middle">allocated under the derived shape</text>
</svg>

Three properties of that route are structural, not conventional:

- **Support is explicit.** Whether a backend runs an operation is a
  compile-time fact (`B: Execute<O>`) backed by a runtime capability row
  checked before launch. There is no default method that falls through to
  some other device.
- **Validation precedes execution.** The descriptor is checked against the
  metadata of the storage that will actually run, never against what a
  caller claims about it.
- **Outputs are derived, not accepted.** A caller never states the output
  shape; it is computed from the inputs and compared against the type the
  caller predicted. Agreement is required, so fabricated metadata has
  nothing to attach to.

If you write your own backend, you sit at the last two stops: you implement
`Capabilities` (the admission answers) and one `Execute<O>` per operation you
advertise. [Backend authoring](./backend_authoring.md) walks that contract;
[the lowering chapter](./deep_lowering.md) explains what happens in the
checks your executor inherits.

## What does not exist

Two things readers often look for are absent on purpose.

**There is no global "current device".** In many frameworks you set a device
somewhere and subsequent allocations land there. In Incin the backend is the
type parameter `B`: which backend a tensor uses is fixed by its type, visible
at the call site, and checked by the compiler. `best_device!()` picks `B` at
compile time from enabled features; `detect_device()` probes hardware at
runtime and returns a value. They answer different questions, and
[the backends chapter](./backends.md) covers both.

**There is no second application-facing execution path.** Tensor methods and
a hand-built dispatch call resolve to the same catalog entry, run the same
validation, and reach the same executor. Backend-local helpers and fused
special sites are ordinary functions inside a backend, behind the same
boundary as everything else. One path means one set of guarantees; a fast
lane with weaker checks would be a different framework wearing the same
name.

## Where to go next

- [Type semantics](./deep_type_semantics.md): what the `S`, `K`, and `G`
  parameters encode, and how to define your own shapes, dtypes, and devices.
- [Lowering: from descriptor to kernel](./deep_lowering.md): the checks
  between a method call and your executor, stage by stage.
- [Proofs: how claims are checked](./deep_proofs.md): which guarantees hold
  at compile time, which hold at lowering, and which are the backend's job.
- [Backend authoring](./backend_authoring.md): the implementation contract,
  start to finish.
