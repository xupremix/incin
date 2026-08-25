# What the macros guarantee

The macros are the part of Incin you touch first and think about least. This
chapter is about the promises they make (what an expansion can and cannot
do to your program) rather than how they are implemented. For usage and
syntax, see [the macro reference](./macros.md); for what the generated
shape types mean, see [type semantics](./deep_type_semantics.md).

| Macro | What it produces |
|---|---|
| `s![...]` | shape *types* for `Tensor<S, ...>` |
| `shape![...]` | shape *values* for targets and checks |
| `tensor![...]` | tensors from literals |
| `idx![...]` / `i![...]` | type-level vs runtime indexing |
| `axis![...]` | axis selectors |
| `#[module]` | module plumbing for your structs |
| `model!` / `import_model!` | typed modules from `.safetensors` / `.onnx` at compile time |

## Expansions cannot be hijacked

Every expansion names what it needs through absolute paths into the real
crate. That is a stronger promise than it sounds. Your crate may define its
own modules called `incin` or `typenum`; glob-imports may drag conflicting
names into scope; none of it matters. An expansion resolves against the
published crate or fails loudly at the macro's own invocation, never
against whatever happens to be visible where you used it.

<svg class="incin-diagram" viewBox="0 0 780 210" role="img" aria-label="Macro hygiene: an expansion parses its input against the macro grammar and emits absolute ::incin paths, so decoy modules named incin or typenum in the caller's scope are never captured." xmlns="http://www.w3.org/2000/svg">
  <style>
    .dg6-node { fill: currentColor; fill-opacity: 0.05; stroke: currentColor; stroke-opacity: 0.4; stroke-width: 1; }
    .dg6-code { font: 600 12.5px ui-monospace, "Source Code Pro", Menlo, monospace; fill: currentColor; }
    .dg6-sub { font: 11px system-ui, sans-serif; fill: currentColor; opacity: 0.75; }
    .dg6-decoy { stroke-dasharray: 4 3; fill: none; stroke: currentColor; stroke-opacity: 0.35; }
    .dg6-reject { stroke-dasharray: 4 3; fill: none; stroke: currentColor; stroke-opacity: 0.35; }
    .dg6-edge { stroke: var(--links, #2b79a2); stroke-width: 1.4; fill: none; marker-end: url(#dg6-arrow); }
    .dg6-note { font: italic 11.5px system-ui, sans-serif; fill: var(--links, #2b79a2); }
  </style>
  <defs>
    <marker id="dg6-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--links, #2b79a2)"/>
    </marker>
  </defs>

  <!-- Caller -->
  <rect class="dg6-node" x="20" y="20" width="220" height="76" rx="7"/>
  <text class="dg6-code" x="130" y="42" text-anchor="middle">caller crate</text>
  <text class="dg6-sub" x="130" y="60" text-anchor="middle">s![2, 3]</text>
  <text class="dg6-sub" x="130" y="76" text-anchor="middle">+ whatever is in scope</text>

  <!-- Decoys -->
  <rect class="dg6-decoy" x="20" y="128" width="220" height="56" rx="7"/>
  <text class="dg6-code" x="130" y="150" text-anchor="middle" opacity="0.7">mod incin { struct Decoy }</text>
  <text class="dg6-code" x="130" y="170" text-anchor="middle" opacity="0.7">mod typenum { struct UTerm }</text>

  <!-- Macro -->
  <path class="dg6-edge" d="M240,58 L300,58"/>
  <rect class="dg6-node" x="304" y="20" width="200" height="76" rx="7"/>
  <text class="dg6-code" x="404" y="46" text-anchor="middle">proc macro</text>
  <text class="dg6-sub" x="404" y="64" text-anchor="middle">parses against its own</text>
  <text class="dg6-sub" x="404" y="80" text-anchor="middle">closed grammar</text>
  <path class="dg6-reject" d="M130,100 L130,124"/>

  <!-- Expansion -->
  <path class="dg6-edge" d="M504,58 L560,58"/>
  <rect class="dg6-node" x="564" y="20" width="196" height="76" rx="7"/>
  <text class="dg6-code" x="662" y="46" text-anchor="middle">expansion</text>
  <text class="dg6-sub" x="662" y="64" text-anchor="middle">absolute paths only:</text>
  <text class="dg6-code" x="662" y="82" text-anchor="middle">::incin::prelude::...</text>

  <!-- Resolution note -->
  <text class="dg6-note" x="404" y="126" text-anchor="middle">decoys are never captured,</text>
  <text class="dg6-note" x="404" y="142" text-anchor="middle">never resolved against</text>
  <text class="dg6-note" x="662" y="126" text-anchor="middle">resolves to the real crate,</text>
  <text class="dg6-note" x="662" y="142" text-anchor="middle">or the build fails loudly</text>
</svg>

One documented limit: renaming the *package* in your manifest
(`incin_x = { package = "incin" }`) breaks the absolute paths, since `::incin`
then names a dependency you do not have. Depending on the crate under its
own name is the supported configuration; depending on it through an alias
with normal `use` renames works fine.

## Rejections are named, not inferred

When a macro rejects input, the diagnostic says which rule failed. A
negative dimension, an unknown `#[module]` argument, a ragged `tensor!`
literal: each fails expansion with a message naming the problem, because a
macro rejection carries no error code; the message is the contract you
read. Nothing expands "as if you hadn't written it": there is no input that
silently changes behavior by accepting a typo.

That last point has teeth in `#[module]`. Its struct arguments form a fixed
vocabulary (`no_stats`, `no_parameters`, ...), and unknown keys fail by
name. A misspelled flag is an error pointing at the misspelling, not a
quietly different module than the one you meant to configure.

## `s!` versus `shape!`: types and values

`s!` builds shape *types*: five dimension forms (static literal, runtime,
named tag, named-with-extent, const path) folded into one chain whose
proof level the compiler computes (see [type semantics](./deep_type_semantics.md)).
The ellipsis forms currently render as fully dynamic on purpose: partial
rank knowledge the parser has not verified would promise more proof than it
checked.

`shape!` builds shape *values*. Its static/runtime split is syntactic: an
integer literal is static, everything else runs. That is deliberately a
weaker answer, never a wrong one. If you need the strongest possible proof
at a boundary, write the type with `s!` and let conversion checking do the
rest.

`tensors from literals`: `tensor!` infers shape from nesting depth exactly
like a Rust array literal and dtype in a fixed order: explicit clause,
consistent numeric suffixes, integer-literals-mean-`i64` (matching
`torch.tensor`), else `f32`. Ragged literals are expansion errors naming
the offending dimension; there is no best-effort reshape of what you
clearly did not mean.

## Two vocabularies that stay apart

`axis!` selects axes; `i!` indexes. They look similar and share nothing:
axis selection accepts expressions, negative positions, and named tags, and
carries compile-time position proofs that still get checked against the
real rank at runtime; indexing has its own grammar of ranges, negatives, and
inference. Keeping them separate means learning one never corrupts the
other; an `i![..]` range will not quietly mean something in a position
where only an axis selector belongs.

## Compile-time import fails closed

`model!` and `import_model!` read `.onnx` graphs and `.safetensors` headers
during compilation and emit typed module structs. Where they cannot help
(unknown rank, control flow, custom domains, unsupported nodes), they stop
with an expansion diagnostic instead of emitting code that would misbehave
later. Partial support plus fail-closed beats broad support plus surprises;
[experimental surfaces](./experimental.md) tracks exactly which graph shapes
are covered today.
