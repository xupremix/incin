# Proofs: how claims are checked

Incin's pitch is that shape, dtype, and device rules are *proven*, not
promised. The interesting question behind that pitch is: proven by whom,
at which stage, and what happens where no check exists? This chapter gives
you the map so you know what to rely on when you call, and what you owe
when you implement.

## Five stages of enforcement

Every rule in the system lives at one of five stages, ordered by how early
it can catch a mistake:

<svg class="incin-diagram" viewBox="0 0 780 150" role="img" aria-label="The five proof stages, earliest first: Type, Lowering, Binding, Native, Unproven. Type means an illegal program does not compile; Unproven means checked ad hoc at runtime." xmlns="http://www.w3.org/2000/svg">
  <style>
    .dg5-node { fill: currentColor; fill-opacity: 0.05; stroke: currentColor; stroke-opacity: 0.4; stroke-width: 1; }
    .dg5-strong { stroke: var(--links, #2b79a2); stroke-width: 1.4; }
    .dg5-letter { font: 700 17px ui-monospace, Menlo, monospace; fill: currentColor; }
    .dg5-name { font: 600 12px system-ui, sans-serif; fill: currentColor; }
    .dg5-sub { font: 10.5px system-ui, sans-serif; opacity: 0.7; fill: currentColor; }
    .dg5-axis { stroke: var(--links, #2b79a2); stroke-width: 1.4; fill: none; marker-end: url(#dg5-arrow); }
    .dg5-label { font: italic 11.5px system-ui, sans-serif; fill: var(--links, #2b79a2); }
  </style>
  <defs>
    <marker id="dg5-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--links, #2b79a2)"/>
    </marker>
  </defs>

  <rect class="dg5-node dg5-strong" x="16" y="24" width="130" height="64" rx="7"/>
  <text class="dg5-letter" x="81" y="52" text-anchor="middle">T</text>
  <text class="dg5-name" x="81" y="70" text-anchor="middle">Type</text>

  <rect class="dg5-node" x="170" y="24" width="130" height="64" rx="7"/>
  <text class="dg5-letter" x="235" y="52" text-anchor="middle">L</text>
  <text class="dg5-name" x="235" y="70" text-anchor="middle">Lowering</text>

  <rect class="dg5-node" x="324" y="24" width="130" height="64" rx="7"/>
  <text class="dg5-letter" x="389" y="52" text-anchor="middle">B</text>
  <text class="dg5-name" x="389" y="70" text-anchor="middle">Binding</text>

  <rect class="dg5-node" x="478" y="24" width="130" height="64" rx="7"/>
  <text class="dg5-letter" x="543" y="52" text-anchor="middle">N</text>
  <text class="dg5-name" x="543" y="70" text-anchor="middle">Native</text>

  <rect class="dg5-node" x="632" y="24" width="130" height="64" rx="7"/>
  <text class="dg5-letter" x="697" y="52" text-anchor="middle">U</text>
  <text class="dg5-name" x="697" y="70" text-anchor="middle">Unproven</text>

  <path class="dg5-axis" d="M20,116 L740,116"/>
  <text class="dg5-label" x="20" y="138">earliest enforcement: trait resolution</text>
  <text class="dg5-label" x="744" y="138" text-anchor="end">ad hoc runtime checks &#8596;</text>
</svg>

| Stage | Name | Meaning |
|---|---|---|
| **T** | Type | Proven by trait resolution. An illegal program does not compile. |
| **L** | Lowering | Proven once, when an operation resolves into a descriptor ([the lowering chapter](./deep_lowering.md)). |
| **B** | Binding | Proven when a resource is constructed, imported, or bound to a plan. |
| **N** | Native | Not re-proven. The executor trusts a sealed descriptor and runs it. |
| **U** | Unproven | Re-checked ad hoc at runtime, or rejected only when something fails. |

Examples put flesh on the table. Static matmul compatibility is stage T:
incompatible shapes are a compile error, full stop. Broadcast output
arithmetic is stage L: computed once per invocation, fallibly, never
silently. "The CUDA device you named actually exists" is inherently stage B
or later. And a backend's own internal kernel correctness sits wherever that
backend puts it, which is why the capability contract matters (below).

The single most important footnote to the whole table:

> **Stage T is not a hardware claim.** Naming `Cuda` in a type proves which
> backend was *selected*, not that the machine running the binary has CUDA.
> Physical-resource facts cannot move earlier than binding; a build that
> probed the compile machine would be wrong for cross-compilation,
> containers, and CI boxes without accelerators.

## What you can rely on as a caller

Reading the table from the application side:

- **If it compiles, static shape agreements held.** There is no code path
  where a static-shape violation reaches execution, because trait
  resolution has no fallback.
- **Runtime geometry fails loudly and early**, at lowering, with a typed
  error from [the taxonomy](./errors.md), before any allocation for the
  operation's output.
- **Support refusals name their reason.** An unsupported dtype, layout, or
  rank arrives as a typed refusal, not a generic error and not a silent
  transfer to another device.
- **What a backend claims is what runs.** The next section is why that
  sentence holds beyond goodwill.

## What you owe as an implementer

The design keeps "advertised" and "implemented" from drifting apart through
one mechanism worth copying even outside Incin: **an advertised operation
must execute, and the language enforces it.** A backend's capability
declarations and its executor set come from one place, so claiming support
for an operation without providing the executor is a compile error, not a
runtime surprise waiting for a user to find it.

That closes the loop on stage N above. Executors trust sealed descriptors,
and do not re-derive shapes, precisely because everything upstream of
them is enforced. The trust chain is only as strong as its weakest entry
point, so there are no side entrances: custom operations go through the
same admission and validation as built-ins, and a handle cannot be
manufactured around the checks.

For your own backends, the practical checklist is short: claim exactly what
you execute, refuse everything else with a typed reason, and test each
advertised row by *running* it. [Backend authoring](./backend_authoring.md)
walks the details; the reasoning is here because the reasoning is what
generalizes.

## Paranoid mode

When you suspect the trust chain, while debugging a new backend or a new
custom operation, the `paranoid-validation` feature adds recomputation:
descriptors get their invariants re-derived and executors can assert them
on hot paths, paying only in builds that asked to pay. If paranoid mode
ever fires, a descriptor reached a backend without passing through a
checked constructor, which is a bug in lowering, not bad input; that is why
the assertion panics instead of returning an error. It is a testing aid,
deliberately outside the release contract: production pays for validation
once, at lowering.

## Reading the evidence yourself

The guarantees above are backed by generated artifacts rather than prose:

- [`docs/capabilities.md`](https://github.com/xupremix/incin/blob/master/docs/capabilities.md)
  and
  [`docs/OPERATION_SEMANTICS.md`](https://github.com/xupremix/incin/blob/master/docs/OPERATION_SEMANTICS.md)
  are regenerated from executor registrations and the operation catalog on
  every relevant test run. If the committed copy and a fresh regeneration
  disagree, CI fails.
- Differentiable operations carry finite-difference gradcheck suites over
  their kernels.
- Conformance vectors cover every CPU-executable catalog operation.

A claim about behavior that cannot be reproduced from those artifacts is,
by house rule, treated as unproven, a habit worth importing into any
project.
