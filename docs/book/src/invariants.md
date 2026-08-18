# Invariants: what you may construct, and what must be earned

Some values in Incin are plain data you can assemble however you like. Others
*certify* something - that a shape's element count doesn't overflow, that a
descriptor has been checked against real storage, that a device ordinal was
range-checked. The second kind has no public unchecked constructor, on
purpose: being able to hand-build one would make the certification worthless.

This chapter is the map of which is which. It is the most "you are holding a
sharp thing" part of the library, and the closest thing here to a nomicon
chapter - though note that unlike Rust's own, almost none of this is `unsafe`.
The enforcement is the type system and private fields, not undefined behaviour
waiting to happen.

## The five categories

| Category | Example | May you build one directly? |
|---|---|---|
| **Marker** - no runtime claim | `Dyn`, `Cpu`, dtype/ordinal markers | Yes. Unit construction. |
| **Configuration** - caller intent only | `ResourceLimits`, optimizer options, `Sequential` | Yes, via public fields/builders. |
| **Validated value** - a checked invariant | `ShapeBuf`, `Alignment`, `CheckedNumel` | Only through a checking constructor. |
| **Opaque identifier** - allocated identity | `TensorId`, `GradientId` | Only through the owning subsystem. |
| **Proof token** - "a rule has run" | `Validated<O>`, `TensorMeta`, `ConstructionWitness` | **No public constructor at all.** |

The last row is the load-bearing one. A `Validated<O>` exists only because
descriptor validation produced it; there is no way for user code to mint one
and hand it to a backend claiming checks happened that didn't.

## Checked arithmetic is not optional

A shape's element count is computed with `checked_mul`, never
`.iter().product()`. This is not fastidiousness - release builds have overflow
checks off, so a crafted or accidentally-huge shape wraps to a small number,
undersizes the allocation, and then stride-based indexing computed from the
same (differently-wrapped) shape reads and writes past the end of it.

```rust,no_run
// `ShapeBuf` and `OperationKind` live in `incin_core::prelude` and are not
// re-exported through the `incin` facade, so they need the direct path.
use incin_core::prelude::{OperationKind, ShapeBuf};

// The constructor that can fail, does. Rank-0 holds exactly one element;
// any zero dimension makes the count zero even if other axes would overflow.
let dims = ShapeBuf::from_slice(&[2, 3, 4]);
let count = dims.checked_numel(OperationKind::Storage)?;
assert_eq!(count, 24);
# Ok::<(), incin::Error>(())
```

`CheckedNumel::from_dims` additionally enforces rank and per-axis resource
limits; `CheckedByteLen::from_dims` also checks the dtype's storage arithmetic
and the tensor byte limit. Both expose only `get()` and have private fields.

The rule for the codebase, stated in `docs/INVARIANT_TYPES.md`: an internal
helper may use `expect` **only after** a value has crossed one of these checked
boundaries. At that point a failure is an internal invariant violation, not a
user input error - which is exactly the distinction the panic policy draws.

## Device selectors prove nothing about hardware

```rust,no_run
# #[cfg(feature = "cuda")]
# fn demo() {
use incin::prelude::*;

// This says "I want CUDA ordinal 0". It does NOT say a GPU exists,
// a driver is loaded, or the ordinal is valid.
let device = Cuda::new(0);
# }
```

Constructing a selector is infallible and free. Availability is a *later,
fallible* step - backend initialization. The ordinal field is private and
readable only through `ordinal()`, specifically so the tuple syntax never
becomes a compatibility commitment.

## Sealed traits

`CanonicalOperation` is sealed (`private::Sealed`), so the set of canonical
operation identities is closed to `incin-core`. A downstream crate cannot
declare a new `op::X` and route it through `dispatch::execute` - the catalog is
the single authoritative declaration of what operations exist, and sealing is
what keeps that true rather than merely conventional.

This is why adding an operation is a change to
`crates/incin-core/src/operation_catalog.rs` and nowhere else: every consumer
is generated from it.

## Deserialization is a constructor

A type whose ordinary constructor validates does **not** get to skip that on
the deserialization path. Types with checked constructors implement custom
deserialization through a wire representation and re-run the same checks  -
tuning cache keys and records, liveness intervals, and memory plans currently.
Malformed values (an interval ending before it starts, a buffer slot outside
the plan's peak count, oversized cache text) are rejected rather than trusted
because they arrived as bytes.

Unknown fields may be tolerated only where the format explicitly allows
forward-compatible metadata, and never in a way that suppresses checksum,
size, ordering, or provenance validation.

## What this buys you

Concretely: if you hold a `Validated<Descriptor<op::Add>>`, the operation's
attributes have been checked against the operands' real metadata, the output
shape has been *derived* rather than accepted from a caller, and the backend's
capability registry has admitted it. A backend author writing an executor can
read those fields without re-deriving them, and the
[canonical execution path](./target_api.md) is built on exactly that
assumption.

The flip side is the discipline it demands of contributors: a new public
unchecked constructor requires a documented caller proof, a safety boundary
proportional to its consequences, and compile-contract coverage. "It's more
convenient" is not one of those.
