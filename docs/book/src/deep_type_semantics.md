# Type semantics

Every `Tensor` in Incin carries six type parameters:

```rust,ignore
Tensor<S, B, K, G, P, L>
//     |  |  |  |  |  |
//     |  |  |  |  |  layout: what is known about where the elements sit
//     |  |  |  |  placement: Local, or sharded across a mesh
//     |  |  |  gradient state: Grad, NoGrad, or Dyn
//     |  |  dtype: f32, i64, Q8_0, your own...
//     |  backend: which executor runs the kernels
//     shape: what the compiler knows about the axes
```

The last four default, and each default is the one that claims the least:
`f32`, `NoGrad`, `Local`, `Dyn`. That is the same rule `ProofLevel` follows
-- silence is never credited -- applied to the type parameters themselves.

[Shapes](./shapes.md) and [Advanced shapes](./advanced_shapes.md) cover
day-to-day use. This chapter is about what those parameters *mean* to the
compiler, how much each one settles before the program runs, and how to
define new ones of your own.

## Three families of shape

A shape type answers one question: which parts of the geometry are known at
compile time? Three families cover the spectrum.

1. **Fully static.** Every axis is a compile-time number. Rank and extents
   are type-level data; two static shapes are equal or not by construction.
2. **`Dyn`.** Even the rank is unknown until a value exists. This is the
   honest type for "whatever came out of this ONNX graph".
3. **Mixed / runtime-carrying.** The rank is known but some axes resolve at
   runtime: a plain `usize` axis, a named dimension with a runtime extent,
   or a const expression evaluated later.

The macros map onto these directly:

| You write | What it means |
|---|---|
| `s![2, 3]` | fully static |
| `s![dyn]` / `s![_]` / `s![usize, 3]` | runtime axis, rank preserved |
| `s![Batch, 128]` after `dim!(Batch)` | typed identity, runtime extent |
| `s![Batch = 25, 128]` | typed identity *and* static extent |
| `s![const FEATURES, 128]` | unevaluated const path |

Two design points worth understanding, because they explain behavior you
will otherwise meet as surprises.

**Axis sizes have no ceiling.** A literal becomes a binary type-level
number built up bit by bit, so an axis of size 4,096 costs O(log N) of
type-level structure rather than requiring a pre-allocated alias. There is
no largest representable dimension lurking in the macro.

**A named axis separates identity from extent.** `dim!(Batch, Seq)`
declares distinct tag types, and every tensor whose shape mentions `Batch`
must agree on *which* axis is Batch. Nothing about that agreement says how
large Batch is; that settles when values exist, and it is checked exactly
once, when the operation lowers (see
[the proofs chapter](./deep_proofs.md)). Names give you mis-wiring errors
(`s![Batch, Seq]` passed where `s![Seq, Batch]` is expected) without paying
for premature equality checks.

## How much a shape proves

Each shape representation carries a proof level:

```text
Static   every axis known at compile time
Mixed    rank proven, extents partly runtime
Dynamic  only Dyn: runtime rank
```

Combining operands takes the weaker of the two proofs, and that rule is
exactly right rather than conservative: add one runtime axis anywhere to a
static shape and it becomes `Mixed`; only `Dyn` drags an operation down to
`Dynamic`. A backend can therefore rely on "if my kernel got a `Static`
descriptor, every extent was decided before the program ran"; the label
cannot overstate what was proven.

Where two operands must have identical shapes, the requirement is enforced
by trait resolution, not a runtime branch. `x.matmul(&y)` between
incompatible static shapes fails to compile because no trait impl exists
for that pair; there is nothing to catch at runtime because control never
reaches it. Runtime-carrying axes get their equality check once, at
lowering, instead of on every use.

The practical consequence for library-style code: generic functions over
`S: Shape` inherit exactly the guarantees their callers' shapes carry, and a
caller with dynamic data weakens only what is genuinely unknown. Nothing
erasures-shaped happens behind your back.

## Gradient state is a type, too

The `G` parameter works the same way as the shape. `Grad` and `NoGrad` are
compile-time markers: a tensor produced from parameters that require
gradients *is* a `Grad` tensor, and training a frozen parameter is a type
error rather than a silent empty backward pass. `Dyn` is the runtime-
toggled option for code that branches on a flag.

When an eager operation runs, the scope's grad mode and the output marker
combine by taking the weaker one, so calling `.backward()`-producing ops
inside a `no_grad` scope still records nothing even if the operand types say
`Grad`. That combined mode is what the dispatcher reports to capability
queries, which is how a backend can serve inference and training differently
without a separate API.

## Defining your own dtypes

`K` accepts any type implementing the public `DType` trait, and the trait is
deliberately small:

```rust,ignore
use incin::types::dtype::StorageEncoding;
use incin::{ConstDType, DType, DTypeDescriptor, DTypeKey, DTypeKind};

/// Three logical values packed into five physical bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Packed3x5;

impl DType for Packed3x5 {
    type Arg = ();
    type Field = core::marker::PhantomData<Self>;
    fn init(_: ()) -> Self::Field {
        core::marker::PhantomData
    }
    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for Packed3x5 {
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
        DTypeKey::new("mycrate", "packed3x5", 1),
        DTypeKind::Opaque,
        // 3 elements per block, 5 bytes per block, 1-byte aligned
        StorageEncoding::block(3, 5, 1),
    );
}
```

Three refinements sit above the base trait, and choosing among them is the
whole design decision:

- **`ConstDType`**: the identity is compile-time-fixed (`Arg = ()`). Add a
  `const DESCRIPTOR` so backends and capability rows can name you without a
  value. Most custom dtypes stop here.
- **`BuiltinDType`**: additionally binds to one of the closed built-in IDs.
  Subsystems that dispatch on a fixed vocabulary (capability registry,
  serialization, kernel tables) require this bound. If your dtype has no
  built-in ID, those subsystems will ask you to go through descriptors
  instead; the bound makes the limitation explicit rather than surprising.
- **`PlainDType`**: the dtype has one ordinary Rust scalar per logical
  element, which unlocks element-wise iteration and scalar conversion.
  Block-quantized formats do not qualify: the built-in `Q8_0` implements
  `ConstDType` but pointedly not `PlainDType`, and your block format should
  follow that example unless elements really do map to scalars.

Whether a backend accepts your dtype is a separate, explicit question:
backends declare per-operation dtype support through
[`SupportsDType`](./backend_authoring.md), and an unsupported pairing is a
compile error naming both sides.

## Defining your own devices

Devices come in three tiers of compile-time knowledge:

| Tier | Example | Constructor argument |
|---|---|---|
| Fully runtime | `Dyn` | a full `DeviceId` |
| Backend known, ordinal runtime | `Cuda`, `Wgpu` | an ordinal selector |
| Fully static | `Cpu`, `CudaN<N>`, `WgpuN<N>` | none |

Implementing `Device` is four small members: the constructor argument type,
the runtime-stored field, `init` to convert one to the other, and
`to_incin` to resolve the runtime `DeviceId`. A fully-static device
implements `ConstDevice` on top, which is what lets it be named in types
with no value present.

One boundary to internalize early, because no amount of typing changes it:
a device type proves *selection*, not *existence*. `CudaN<U0>` in a type
says "this tensor targets CUDA ordinal 0"; whether that hardware exists on
the machine running the binary is a runtime fact discovered when storage is
allocated. Frameworks that pretend otherwise break cross-compilation and
containers; Incin puts the check where the truth is.

Pair the device with a backend that executes on it (the
[authoring chapter](./backend_authoring.md) shows the whole contract), and
transfers between devices stay ordinary typed operations via `.to_device()`
rather than implicit copies.

## Where this crosses into execution

Everything on this page lives on the caller's side of a deliberate border:
the executor is *not* generic over your shape type. What crosses is a
validated descriptor plus evidence of how much was proven. The next chapter
follows that handoff stage by stage.
