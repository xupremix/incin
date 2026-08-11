# Invariant-bearing public types

This document records which public Incin values may be freely constructed and
which values certify an invariant. The rule is simple: marker and configuration
values may be assembled directly; validated values, identifiers, proof tokens,
and runtime handles may only be obtained through the constructor or subsystem
that establishes their contract.

## Classification

| Type | Category | Invariant | Public construction | Validation | Accessor |
| --- | --- | --- | --- | --- | --- |
| `Dyn`, `Cpu`, static ordinal/dtype/policy markers | marker with no invariant | No runtime validity claim | Unit/default construction | None | Marker identity/type |
| `ShapeBuf`, `StrideBuf` | validated value | Derived counts, strides, and spans are representable | `from_slice`; checked derived methods | Checked multiplication/addition per operation | `dims`, `strides` |
| `Alignment` | validated value | Nonzero power-of-two byte alignment | `Alignment::new`, `Alignment::of` | Constructor | `bytes` |
| `CheckedNumel`, `CheckedByteLen` | validated value | Resource-bounded element/allocation count | `from_dims`; tuple fields private | Shape/resource limits and checked arithmetic | `get` |
| `AxisSet`, `DeviceSet`, cache keys/records | validated value | Semantic axis collections, nonempty homogeneous devices, and bounded/provenanced cache data | Named checked constructors | Constructor and custom deserialization | Named read-only accessors |
| `TensorId`, `GradientId`, `FsdpParameterId`, `PipelineBoundaryId`, `TensorParallelId` | opaque identifier | Identity allocated or range-checked by its subsystem | Subsystem factory or checked `new`; tuple fields private | Allocation/range checks | `get`/named ID accessor |
| `HitId`, `BufferSlot`, distributed run/stream/mesh/sequence identifiers | opaque identifier | Stable identity within the owning subsystem | Renderer/planner/distributed factory; tuple fields private | Owner-specific bounds and uniqueness | `get`, `index`, or named accessor |
| `Validated<O>`, `ConstructionWitness` | proof token | Descriptor and metadata rules have run | No public unchecked constructor | Descriptor/front-end validation | Read-only descriptor/proof accessors |
| `TensorMeta` | proof token | Shape, strides, offset, dtype, device, layout, alignment, and capacity agree | `try_new`/`contiguous`; wrapper field private | Full metadata/span validation | Read-only fields and named accessors |
| `LivenessInterval`, `MemoryPlan`, committed tuning decisions | proof token | Ordered lifetime; slots fit plan; provisional choice was committed | Analyzer/planner/commit path | Construction and custom deserialization | Interval, assignment, and report accessors |
| `ResourceLimits`, optimizer/distribution/module/compiler/import options, `Sequential` | configuration | Caller intent only; no storage/hardware claim | Public fields/builders where documented | Checked before use; constrained builders validate eagerly | Public fields or configuration getters |
| `Tensor`, `Param`, `Gradients`, backend variables/storage, datasets | runtime handle | Bound to validated storage, backend state, gradients, or loaded data | Owning backend/allocation/load path | At creation/load and before execution | Handle-specific methods |
| `Cuda`, `Wgpu`, `Metal`; `CudaDevice`, `WgpuDevice`, `MetalDevice` | runtime handle | Requested logical ordinal only, not availability | Explicit `new`; ordinal fields private | Hardware probing is a later fallible step | `ordinal` |

## Device selectors

`Cuda::new(ordinal)`, `Wgpu::new(ordinal)`, and `Metal::new(ordinal)` describe a
requested runtime ordinal. Their private ordinal and `ordinal()` accessor prevent
the tuple syntax from becoming a compatibility commitment. Constructing a
selector does **not** prove that the requested accelerator, driver, adapter, or
feature is available. Availability remains a fallible backend initialization
step. Static selectors remain pure type/value markers.

## Checked sizes and arithmetic

`CheckedNumel::from_dims` validates rank and per-axis resource limits and uses
checked multiplication. Rank zero contains one scalar element. A shape with a
zero dimension contains zero elements even when irrelevant nonzero dimensions
would otherwise overflow. `CheckedByteLen::from_dims` additionally checks dtype
storage arithmetic and the tensor byte limit. Both expose only `get()` and have
private fields.

`ShapeBuf` is the canonical dimension sequence. `StrideBuf::contiguous_for`
uses checked multiplication and preserves conventional row-major strides for
empty tensors when those strides are representable. An unrepresentable layout
is rejected even when the element count is zero; this keeps layout metadata
canonical instead of assigning arbitrary strides. View spans, offsets, slicing,
reshape inference, concat/stack dimensions, allocation lengths, and model/data
dimension conversions must use the same checked primitives or an
operation-specific checked equivalent.

Internal helpers that use `expect` are permitted only after a value has crossed
one of these checked construction boundaries. They represent an internal
invariant violation, not a public input error. Type-level static dimension
products likewise check their factors; an unrepresentable compile-time product
cannot become runtime metadata.

## Serialization contract

Deserialization is a constructor, not a field-copy shortcut. Types whose
ordinary constructors validate data implement custom deserialization through a
wire representation and rerun the same checks. This currently includes tuning
cache keys and records, liveness intervals, and memory plans. Malformed values
such as an interval defined after its last use, a buffer slot outside the plan's
peak slot count, or oversized cache text are rejected.

Round-trip serialization preserves the validated value. Unknown fields may be
tolerated only where the format explicitly permits forward-compatible metadata;
they never suppress checksum, size, ordering, or provenance validation.

## Public tuple-field audit

The workspace-wide audit treats a `pub struct Name(pub T)` as suspicious unless
the value is a transparent marker or configuration with no invariant. All
validated sizes, proof tokens, IDs, runtime selectors, and planner slots now use
private fields. Remaining tuple forms are either private implementation details,
crate-visible backend storage, or deliberately transparent configuration such as
`Sequential`; none is an externally forgeable invariant token.

Unchecked constructors remain crate-private. A future public unchecked
constructor requires a documented caller proof, a safety boundary commensurate
with its consequences, and compile-contract coverage.
