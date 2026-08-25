# Lowering: from descriptor to kernel

Between `x.matmul(&w)?` and a running kernel sits a short, fixed pipeline.
This chapter walks it in order, because each stage answers a different
question and produces a different class of failure when it is unhappy. The
pipeline is the same whether the call came from a typed tensor method or
from [a custom operation](./custom_operations.md) you defined yourself, and
it is the part of Incin your executor inherits rather than reimplements.

## The descriptor: computed, never dictated

The unit that crosses from frontend to backend is a descriptor: one
operation's attributes plus the input and output metadata inferred for it.

The rule that makes descriptors trustworthy is in the name of this section:
they are *derived*. You construct one from operand shapes, and everything
else (output shape, broadcast masks, reduction geometry) is computed.
There is no constructor that accepts an output shape as an argument, so a
descriptor whose outputs disagree with its inputs cannot be written, by you
or by the framework.

Two consequences fall out of that construction rule:

- **A descriptor describes logical geometry only**: shapes and how axes
  relate. Per-tensor facts like storage offset, dtype, device, alignment,
  and strides live beside it in the tensor metadata. Keeping them out is
  what lets one descriptor be reused across operands, cached, and used as a
  kernel specialization key without those caches lying about layout.
- **There is no rank ceiling built into lowering.** Axis bookkeeping uses a
  structure that handles rank 4 and rank 40 identically; if some backend
  caps rank at 6, that restriction belongs to that backend's capability
  rows, where it can be reported honestly per operation.

## Why schema versions exist

Anything derived from a descriptor's contents and kept across runs (a
kernel cache, autotune records, a serialized plan) is only valid for the
layout of fields it was produced under. Descriptors therefore carry a
schema version, and compatibility checking is exact equality, on purpose:
if a field's meaning changed, a range check would not notice, while
re-deriving a descriptor is cheap next to executing against a stale cache
entry. Adding a brand-new descriptor type does not bump the version,
because nothing cached could refer to it yet.

If you build tooling that persists anything keyed on descriptors, treat the
schema version the same way: equality or nothing.

## The four checks before a kernel runs

Canonical dispatch runs four stages in order. Each has its own failure
class, which is worth memorizing because the class tells you who is at
fault.

<svg class="incin-diagram" viewBox="0 0 780 420" role="img" aria-label="Dispatch's four ordered stages: metadata read from real handles, output inference cross-checked against the caller's type, payload validation, capability admission filtered by policy, then the backend launch. Each stage has its own error class." xmlns="http://www.w3.org/2000/svg">
  <style>
    .dg4-node { fill: currentColor; fill-opacity: 0.05; stroke: currentColor; stroke-opacity: 0.4; stroke-width: 1; }
    .dg4-code { font: 600 13px ui-monospace, "Source Code Pro", Menlo, monospace; fill: currentColor; }
    .dg4-sub { font: 11.5px system-ui, sans-serif; fill: currentColor; opacity: 0.75; }
    .dg4-stage { font: italic 12px system-ui, sans-serif; fill: var(--links, #2b79a2); }
    .dg4-err { font: 11.5px ui-monospace, Menlo, monospace; fill: var(--links, #2b79a2); }
    .dg4-edge { stroke: var(--links, #2b79a2); stroke-width: 1.4; fill: none; marker-end: url(#dg4-arrow); }
    .dg4-reject { stroke: currentColor; stroke-opacity: 0.35; stroke-width: 1; stroke-dasharray: 4 3; fill: none; marker-end: url(#dg4-arrow-faint); }
  </style>
  <defs>
    <marker id="dg4-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--links, #2b79a2)"/>
    </marker>
    <marker id="dg4-arrow-faint" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor" fill-opacity="0.35"/>
    </marker>
  </defs>

  <!-- Step 1 -->
  <rect class="dg4-node" x="40" y="16" width="480" height="52" rx="7"/>
  <text class="dg4-stage" x="56" y="34">1</text>
  <text class="dg4-code" x="280" y="36" text-anchor="middle">logical metadata</text>
  <text class="dg4-sub" x="280" y="55" text-anchor="middle">shape/dtype/device read off the storage that will run</text>

  <path class="dg4-edge" d="M280,68 L280,86"/>

  <!-- Step 2 -->
  <rect class="dg4-node" x="40" y="90" width="480" height="52" rx="7"/>
  <text class="dg4-stage" x="56" y="108">2</text>
  <text class="dg4-code" x="280" y="110" text-anchor="middle">output inference + cross-check</text>
  <text class="dg4-sub" x="280" y="129" text-anchor="middle">outputs derived; caller's predicted shape must agree</text>

  <path class="dg4-edge" d="M280,142 L280,160"/>

  <!-- Step 3 -->
  <rect class="dg4-node" x="40" y="164" width="480" height="52" rx="7"/>
  <text class="dg4-stage" x="56" y="182">3</text>
  <text class="dg4-code" x="280" y="184" text-anchor="middle">payload validation</text>
  <text class="dg4-sub" x="280" y="203" text-anchor="middle">data-carrying attributes must carry exactly their declared bytes</text>

  <path class="dg4-edge" d="M280,216 L280,234"/>

  <!-- Step 4 -->
  <rect class="dg4-node" x="40" y="238" width="480" height="60" rx="7"/>
  <text class="dg4-stage" x="56" y="256">4</text>
  <text class="dg4-code" x="280" y="260" text-anchor="middle">capability admission</text>
  <text class="dg4-sub" x="280" y="279" text-anchor="middle">exact support row per operand, filtered by fallback policy:</text>
  <text class="dg4-sub" x="280" y="293" text-anchor="middle">native always; composed and fallback only if allowed</text>

  <path class="dg4-edge" d="M280,298 L280,316"/>

  <!-- Launch -->
  <rect class="dg4-node" x="40" y="320" width="480" height="52" rx="7"/>
  <text class="dg4-code" x="280" y="340" text-anchor="middle">backend.execute(ExecutionRequest)</text>
  <text class="dg4-sub" x="280" y="359" text-anchor="middle">the value handed over is the value a capture would record</text>

  <!-- Error taxonomy -->
  <text class="dg4-stage" x="560" y="34">error taxonomy</text>
  <text class="dg4-err" x="560" y="112">DescriptorError</text>
  <text class="dg4-sub" x="560" y="128">request was never legal</text>
  <text class="dg4-err" x="560" y="262">Error::Policy</text>
  <text class="dg4-sub" x="560" y="278">support disallowed before launch</text>
  <text class="dg4-err" x="560" y="340">Error::Backend</text>
  <text class="dg4-sub" x="560" y="356">legal request failed at or after launch</text>
  <path class="dg4-reject" d="M520,116 C545,116 545,116 556,116"/>
  <path class="dg4-reject" d="M520,264 C545,264 545,264 556,264"/>
  <path class="dg4-reject" d="M520,342 C545,342 545,342 556,342"/>
</svg>

**1. Metadata comes from the handles, not the caller.** Shape, dtype, and
device are read off the allocations the backend will actually run on. A
caller cannot satisfy validation with metadata describing some other tensor,
because the caller is never asked.

**2. Outputs are inferred, then cross-checked.** The catalog's inference
rule computes the outputs from the inputs, and the result is compared
against what the typed frontend predicted. For a custom operation, this is
the check that catches an inference rule that disagrees with itself across
erased and typed paths. Proof provenance is stamped here too: evidence of
how much was compile-time-known travels with the invocation, and the erased
entry point stamps "dynamic", claiming nothing it cannot see.

**3. Payloads are checked against declarations.** Operations whose
attributes embed data (weights handed to a creation op, say) must arrive
with exactly the declared byte length; payloads on operations that cannot
carry them are rejected the same way. Nothing with a size mismatch reaches
a kernel.

**4. Capability admission, filtered by policy.** Per operand, dispatch asks
the backend's registry how it would serve this exact request: operation,
dtype, layout, rank, training flag, math mode. The answer comes back as a
support level:

| Level | Meaning | Passes when |
|---|---|---|
| `Native` | your kernel runs it directly | always |
| `Composed` | you rewrite it into other operations | composition allowed |
| `Fallback` | a transfer/materialization would do it | transfers allowed |
| `Unsupported(reason)` | you refuse | never |

Composition is allowed by default and transfer is not, and both are
per-context policies. When the policy refuses, the caller gets a typed
violation naming the operation, the level, and the policy, never a silent
copy to another device. If you are writing a backend, this table is also
your honesty budget: claim `Native` only where a kernel exists, because
[the proof machinery](./deep_proofs.md) holds advertised claims to account.

## Reading the errors

The taxonomy mirrors the stages, and the distinction decides where to look:

- **Descriptor**: the request was never legal (`DescriptorError`, or a
  shape error for the geometry cases). Look at your shapes, attributes, and
  payload sizes; no change of device or features helps.
- **Policy**: the backend may support it, but the context forbade the kind
  of support on offer (`Error::Policy`). Loosen the fallback policy or pick
  a native path.
- **Backend**: a legal request failed at or after launch
  (`Error::Backend`). This is the device-level bucket: out of memory,
  driver fault, a genuine kernel bug.

Collapsing these three into one error type would discard exactly the
information that tells you whether to fix your program, your policy, or
your hardware budget.

## Capture: recording what was validated

Because every stage above runs before launch and produces one validated
invocation, capturing a plan adds nothing new: recording is writing down
the already-validated value. That property (*the value handed to the
backend is the value a compiler would record*) is what lets the preview
`compiled` feature snapshot plans for inspection and CPU reference
evaluation. Those snapshots are a development aid, not a deployment format;
[experimental surfaces](./experimental.md) draws that line explicitly.

## Where to go from here

- [Type semantics](./deep_type_semantics.md), for what the frontend proves
  before any of these stages run.
- [Proofs: how claims are checked](./deep_proofs.md), for how each guarantee
  on this page stays true instead of merely documented.
- [Backend authoring](./backend_authoring.md), to write executors against
  this boundary yourself.
