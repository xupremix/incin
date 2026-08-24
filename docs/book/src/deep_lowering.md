# Lowering: from descriptor to kernel

The route a validated operation takes from the typed frontend to a backend
executor - descriptor construction, schema versioning, capability admission,
and what "capture" means here. This expands
[from proofs to execution](./proofs_to_execution.md), which is the short
version, and `docs/GUIDE.md` §6. Sources:
[`crates/incin-core/src/exec/`](../../../crates/incin-core/src/exec/dispatch.rs).

## Descriptors are derived, never asserted

A descriptor (`crates/incin-core/src/exec/catalog/descriptor.rs`) holds one
operation's attributes plus inferred input/output metadata. Its module doc in
`crates/incin-core/src/exec/spec.rs` states the contract this whole chapter
rests on:

> **A descriptor is derived, never asserted.** Every constructor here takes
> operand *shapes* and computes the rest. There is no way to hand a
> descriptor an output shape that disagrees with its inputs, or a broadcast
> mask that disagrees with its strides, because neither is an argument.

Two structural rules follow:

- **Logical geometry only.** Storage offsets, dtype, device, and alignment
  are per-tensor facts and live in `TensorMeta`, not in the descriptor.
  Keeping them out is what lets one descriptor be reused across operands,
  cached, and used as a specialization key.
- **No rank ceiling.** Axis collections use `AxisSet` - an inline `u64` mask
  for ranks up to 64 that spills to owned storage beyond that - so descriptor
  construction has no frontend rank limit; backend capability and resource
  policies are the places rank may be restricted.

## Schema versioning

Anything derived from descriptor *contents* and kept across runs - kernel
caches, autotune records, serialized plans - is only valid for the schema it
was produced under. `DescriptorSchemaVersion` polices that:

```rust,ignore
// crates/incin-core/src/exec/spec.rs (abridged)
pub struct DescriptorSchemaVersion(u32);

impl DescriptorSchemaVersion {
    /// v2 added operator identity to reduction/pooling geometry;
    /// v3 added optional operator identity to broadcast geometry.
    pub const CURRENT: Self = Self(3);

    /// Exact equality, deliberately. A descriptor schema has no compatible
    /// subset: a field whose meaning changed is not detectable by a range
    /// check, and re-deriving a descriptor is cheap next to executing one
    /// against a stale cache entry.
    pub const fn is_compatible_with(self, other: Self) -> bool {
        self.0 == other.0
    }
}
```

Bump `CURRENT` whenever any descriptor gains, loses, or reinterprets a field.
Adding a whole new descriptor does not require a bump - nothing cached can
refer to it yet. A pinning test makes accidental field-set drift fail CI
instead of silently invalidating caches.

## The dispatcher

`dispatch::execute` / `dispatch::execute_shaped`
([`crates/incin-core/src/exec/dispatch.rs`](../../../crates/incin-core/src/exec/dispatch.rs))
is the single production route. The shaped form, which typed tensor methods
call, is:

```rust,ignore
// crates/incin-core/src/tensor/ops/binary.rs (the call site)
let h_lhs = TensorHandle::from_storage::<B, KIn, Local>(&lhs.inner);
let h_rhs = TensorHandle::from_storage::<B, KIn, Local>(&rhs.inner);
let shape_val = lhs._shape.clone();
let context = crate::tensor::grad::execution_context::<B, GOut>(&grad_out);
let storage =
    dispatch::execute_shaped::<O, B, S>(&context, NoAttributes, &[h_lhs, h_rhs], &shape_val)
        .map_err(crate::err::Error::from)?;
```

Inside, four steps run in order.

<svg class="incin-diagram" viewBox="0 0 780 420" role="img" aria-label="The dispatcher's four ordered checks: logical metadata from handles, output inference cross-checked against the caller's type, payload validation, capability admission filtered by policy - then the backend launch. Each stage has its own error class." xmlns="http://www.w3.org/2000/svg">
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
  <text class="dg4-code" x="280" y="36" text-anchor="middle">logical_meta(handles)</text>
  <text class="dg4-sub" x="280" y="55" text-anchor="middle">shape/dtype/device read off the storage that will run</text>

  <path class="dg4-edge" d="M280,68 L280,86"/>

  <!-- Step 2 -->
  <rect class="dg4-node" x="40" y="90" width="480" height="52" rx="7"/>
  <text class="dg4-stage" x="56" y="108">2</text>
  <text class="dg4-code" x="280" y="110" text-anchor="middle">infer_invocation_typed</text>
  <text class="dg4-sub" x="280" y="129" text-anchor="middle">outputs derived; caller's ShapeValue&lt;S&gt; cross-checked</text>

  <path class="dg4-edge" d="M280,142 L280,160"/>

  <!-- Step 3 -->
  <rect class="dg4-node" x="40" y="164" width="480" height="52" rx="7"/>
  <text class="dg4-stage" x="56" y="182">3</text>
  <text class="dg4-code" x="280" y="184" text-anchor="middle">payload &#8960; DataAttributes byte length</text>
  <text class="dg4-sub" x="280" y="203" text-anchor="middle">missing payload or stray payload is a descriptor error</text>

  <path class="dg4-edge" d="M280,216 L280,234"/>

  <!-- Step 4 -->
  <rect class="dg4-node" x="40" y="238" width="480" height="60" rx="7"/>
  <text class="dg4-stage" x="56" y="256">4</text>
  <text class="dg4-code" x="280" y="260" text-anchor="middle">admit(backend, operation, meta, training, math_mode)</text>
  <text class="dg4-sub" x="280" y="279" text-anchor="middle">exact capability row per operand, filtered by fallback policy:</text>
  <text class="dg4-sub" x="280" y="293" text-anchor="middle">Native always; Composed and Fallback only if allowed</text>

  <path class="dg4-edge" d="M280,298 L280,316"/>

  <!-- Launch -->
  <rect class="dg4-node" x="40" y="320" width="480" height="52" rx="7"/>
  <text class="dg4-code" x="280" y="340" text-anchor="middle">backend.execute(ExecutionRequest)</text>
  <text class="dg4-sub" x="280" y="359" text-anchor="middle">the value handed over is the value a capture would record</text>

  <!-- Error taxonomy -->
  <text class="dg4-stage" x="560" y="34">error taxonomy</text>
  <text class="dg4-err" x="560" y="112">CanonicalError::Descriptor</text>
  <text class="dg4-sub" x="560" y="128">request was never legal</text>
  <text class="dg4-err" x="560" y="262">CanonicalError::Policy</text>
  <text class="dg4-sub" x="560" y="278">support disallowed before launch</text>
  <text class="dg4-err" x="560" y="340">CanonicalError::Backend</text>
  <text class="dg4-sub" x="560" y="356">legal request failed at or after launch</text>
  <path class="dg4-reject" d="M520,116 C545,116 545,116 556,116"/>
  <path class="dg4-reject" d="M520,264 C545,264 545,264 556,264"/>
  <path class="dg4-reject" d="M520,342 C545,342 545,342 556,342"/>
</svg>

**1. Logical metadata comes from the handles, not the caller.**

```rust,ignore
// crates/incin-core/src/exec/dispatch.rs
pub fn logical_meta(metadata: &TensorMeta) -> LogicalTensorMeta {
    LogicalTensorMeta {
        shape: Some(metadata.shape.clone()),
        dtype: Some(metadata.dtype),
        device: Some(metadata.device),
    }
}
```

Every field is read off the allocation the backend will actually run on, so
validation cannot be satisfied by metadata describing some other tensor.

**2. Outputs are inferred, and the caller's type is cross-checked.**
`O::infer_invocation_typed(attributes, logical, expected)` runs the catalog's
`infer_outputs` for the operation and compares the result against the
`ShapeValue<S>` the typed frontend was holding. The two must agree; the
custom-operation path shows the check explicitly:

```rust,ignore
// crates/incin-core/src/exec/catalog/validated.rs (custom path)
if actual != expected.shape_buf() {
    return Err(DescriptorError::Shape(
        crate::shapes::ShapeError::TargetShapeRejected { .. },
    ));
}
Ok(Self {
    validated: crate::exec::Validated::new_with_evidence(descriptor,
        crate::exec::ShapeEvidence::of::<S>()),
})
```

This is why `S` travels *beside* the attributes instead of being read off
them (`docs/GUIDE.md` §6): a caller could claim `ShapeEvidence::of::<s![2, 3]>()`
next to metadata describing something else, but it cannot make that claim
agree with derived outputs. The proof level is stamped from the real shape
type by `ShapeEvidence::of::<S>()`; the erased `execute` entry point stamps
`Dynamic`, honestly claiming nothing.

**3. Payloads are checked against the attributes.** Operations whose
attributes carry data (`DataAttributes` for `tensor_from_data` and friends)
require a borrowed payload of exactly the declared byte length; payload-bearing
attributes without a payload, or payloads on operations that cannot carry
one, are `DescriptorError`s before anything launches.

**4. Capability admission, then launch.** Per operand, the exact row is
queried:

```rust,ignore
// crates/incin-core/src/exec/dispatch.rs (abridged)
fn admit<B: Capabilities>(backend: &B, operation: &OperationIdentity,
                          metadata: &TensorMeta, training: bool, math_mode: MathMode)
    -> Result<SupportLevel, UnsupportedReason>
{
    let query = CapabilityQuery {
        operation: operation.clone(),
        dtype: metadata.dtype,
        layout: metadata.layout,
        rank: metadata.shape.dims().len(),
        training,
        math_mode,
    };
    match backend.support(&query) {
        SupportLevel::Unsupported(reason) => Err(reason),
        level => Ok(level),
    }
}
```

The answer is then filtered through the context's fallback policy:
`Native` always passes; `Composed` passes only when composition is allowed;
`Fallback` (a transfer/materialization) only when transfers are allowed -
and a denial is a typed `PolicyViolation` naming the operation, the reported
level, and the policy, never a silent downgrade. Only then does dispatch call
`backend.execute(ExecutionRequest { operation: invocation.validated(), ... })`.

The error taxonomy mirrors the stages: `CanonicalError::Descriptor` means the
request was never legal, `CanonicalError::Policy` means its requested support
is disallowed before launch, and `CanonicalError::Backend` means a legal
request failed at or after launch. Collapsing them would lose the distinction
that decides whether the caller, the policy, or the device is at fault.

## Choosing backends uses the same registry

```rust,ignore
// crates/incin-core/src/exec/dispatch.rs
/// The support level `operation` would resolve to for one operand,
/// without running it.
pub fn support_for<O, B>(context: &ExecutionContext<B>, metadata: &TensorMeta)
    -> Result<SupportLevel, UnsupportedReason>
where O: CanonicalOperation, B: Capabilities + StorageBackend,
{ ... }
```

A caller choosing between backends gets its answer from the same registry the
execution path uses. There is deliberately no second source of truth a probe
could disagree with; `docs/capabilities.md` is generated from these rows and
re-checked by test.

## Capture

Dispatch's module doc names the property capture depends on: *"Capture keeps
the same descriptor - the value handed to the backend is the value a compiler
would record."* Nothing new is computed at capture time; recording is taking
the already-validated invocation and writing it down.

The current recording format lives under the preview `compiled` feature:
[`crates/incin-core/src/compiled/artifact.rs`](../../../crates/incin-core/src/compiled/artifact.rs).
A `CompiledArtifact` snapshots a plan with a magic header (`INCIN\x00\x01\x00`),
an `ArtifactVersion` whose `format` field must equal the current
`ARTIFACT_FORMAT_VERSION`, and an Adler-32 checksum over the payload.
Deserialization re-verifies integrity and re-runs semantic checks - the same
"deserialization is a constructor" rule as [Invariants](./invariants.md).

It is worth stating plainly what this is not: per its own module doc, these
snapshots are an inspection and CPU-reference-evaluator aid, *not* a
deployment format or portable ABI. See [Experimental surfaces](./experimental.md)
for the supported subset.

## Where to go from here

- [deep_type_semantics.md](./deep_type_semantics.md) - where `S`,
  `ProofLevel`, and `ShapeEvidence` come from.
- [deep_proofs.md](./deep_proofs.md) - how each guarantee on this page is kept
  true rather than merely documented.
- Writing executors against this boundary:
  [Backend authoring](./backend_authoring.md).
