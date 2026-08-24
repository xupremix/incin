# Type semantics

How Incin encodes "what shape, what proof, what gradient state" in types -
and what each encoding lets the compiler settle before the program runs. This
is the deeper companion to [Shapes](./shapes.md) and
[Advanced shapes](./advanced_shapes.md), which cover usage; here it is the
machinery. `docs/GUIDE.md` §3 is the prose version of the first half.

## The three shape families

Every shape type implements `Shape`
([`crates/incin-core/src/shapes/shape.rs`](../../../crates/incin-core/src/shapes/shape.rs)).
Three families do:

1. **A recursive `DimCons` chain**, e.g. `DimCons<U2, DimCons<U3, Nil>>` -
   every axis known at compile time. Rank and extents are type-level data.
2. **`Dyn`** - rank itself is unknown until a value exists.
3. **A known-rank runtime shape** - `Ranked<R>`, or a mixed `DimCons` chain
   whose axes are `usize`, named dimensions, or `ConstDim`.

`s![2, 3]` expands to exactly the first form. The macro's grammar maps onto
the dimension types like this:

| You write | It becomes | Meaning |
|---|---|---|
| `s![2, 3]` | `DimCons<UInt<..>, DimCons<UInt<..>, Nil>>` | fully static |
| `s![dyn]` / `s![_]` / `s![usize, 3]` | an extent of `usize` | runtime axis, rank preserved |
| `s![Batch, 128]` after `dim!(Batch)` | `NamedDim<Batch, usize>` | typed identity, runtime extent |
| `s![Batch = 25, 128]` | `NamedDim<Batch, UInt<..>>` | typed identity *and* static extent |
| `s![const FEATURES, 128]` | `ConstDim<{ FEATURES }, ..>` | unevaluated const path |

Two details of that table are worth pausing on.

**The typenum encoding is binary and unbounded.** A literal is rendered by
recursing over its bits rather than by looking up an alias catalogue:

```rust,ignore
// crates/incin-macros/src/shape.rs
pub(crate) fn lit_to_typenum(
    n: usize,
    path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if n == 0 {
        return quote! { #path typenum::UTerm };
    }
    let bit = if n.is_multiple_of(2) {
        quote! { #path typenum::B0 }
    } else {
        quote! { #path typenum::B1 }
    };
    let rest = lit_to_typenum(n / 2, path);
    quote! { #path typenum::UInt<#rest, #bit> }
}
```

That is O(log N) tokens for any literal, so there is no `U4096`-style ceiling
on an axis size.

**A named axis separates identity from extent.** `dim!(Batch, Seq)` declares
zero-sized tag types implementing `AxisTag`, plus a schema that gives each
tag a stable position (`AxisIdentity::Id`) within one `dim!` group
(`crates/incin-core/src/shapes/dim.rs`). Two tensors sharing `s![Batch, ...]`
must agree on which axis is `Batch`; nothing about that agreement says how
*large* Batch is until both values exist. That is why the runtime equality
check happens exactly once at lowering (see
[the proofs chapter](./deep_proofs.md)) instead of being implied by the name.

## What the `Shape` trait settles

Beyond construction, `Shape` exposes three associated constants. All default
to "nothing", so a `Shape` implemented outside the crate is credited with no
proof it has not shown:

```rust,ignore
// crates/incin-core/src/shapes/shape.rs (abridged)
pub trait Shape: sealed::Shape + 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    /// Compile-time validity gate for exact structural expressions.
    const STATIC_VALID: () = ();
    /// Compile-time rank when this representation preserves it.
    const RANK: Option<usize> = None;
    /// How much of this shape the compiler settled.
    const PROOF: ProofLevel = ProofLevel::Dynamic;
    /// Element count, when the type alone settles it.
    const STATIC_NUMEL: Option<usize> = None;
    ...
}
```

The recursive fold over `DimCons` computes all of them as constants:

```rust,ignore
// crates/incin-core/src/shapes/shape.rs (DimCons impl)
const PROOF: ProofLevel = match (H::STATIC_SIZE, T::PROOF) {
    (true, ProofLevel::Static) => ProofLevel::Static,
    _ => ProofLevel::Mixed,
};

const STATIC_NUMEL: Option<usize> = match (H::STATIC, T::STATIC_NUMEL) {
    (StaticExtent::Value(h), Some(t)) => h.checked_mul(t),
    _ => None,
};
```

Note `checked_mul` inside the const context: an unrepresentable compile-time
product becomes `None`/invalid rather than wrapping, per
`docs/INVARIANT_TYPES.md`'s checked-arithmetic rule.

`STATIC_NUMEL` is not decoration. Because `S` is a type parameter, a backend
asking `S::STATIC_NUMEL` gets a constant folded at monomorphization; the CPU
creation family uses this to skip computing element counts at runtime
entirely (`docs/GUIDE.md` §6 records the measured effect).

## The proof lattice

`ProofLevel` ([`crates/incin-core/src/shapes/proof.rs`](../../../crates/incin-core/src/shapes/proof.rs))
has three levels - `Static`, `Mixed`, `Dynamic` - ordered strongest first.
Binary operations combine operands with `meet`, the weaker-of-two operation:

```rust,ignore
// crates/incin-core/src/shapes/proof.rs
/// The weaker of two proofs.
pub const fn meet(self, other: Self) -> Self {
    if (self as u8) >= (other as u8) {
        self
    } else {
        other
    }
}
```

<svg class="incin-diagram" viewBox="0 0 780 190" role="img" aria-label="The ProofLevel semilattice: Static is the top and meet returns the weaker of two proofs. meet(Static, Mixed) is Mixed; meet(Mixed, Dynamic) is Dynamic." xmlns="http://www.w3.org/2000/svg">
  <style>
    .dg3-node { fill: currentColor; fill-opacity: 0.05; stroke: currentColor; stroke-opacity: 0.4; stroke-width: 1; }
    .dg3-top { stroke: var(--links, #2b79a2); stroke-width: 1.4; }
    .dg3-code { font: 600 13px ui-monospace, "Source Code Pro", Menlo, monospace; fill: currentColor; }
    .dg3-sub { font: 11.5px system-ui, sans-serif; fill: currentColor; opacity: 0.7; }
    .dg3-order { font: italic 12px system-ui, sans-serif; fill: var(--links, #2b79a2); }
    .dg3-edge { stroke: currentColor; stroke-opacity: 0.4; stroke-width: 1.2; }
  </style>

  <!-- Lattice chain -->
  <rect class="dg3-node dg3-top" x="300" y="16" width="180" height="40" rx="7"/>
  <text class="dg3-code" x="390" y="41" text-anchor="middle">Static</text>
  <line class="dg3-edge" x1="390" y1="56" x2="390" y2="76"/>
  <rect class="dg3-node" x="300" y="80" width="180" height="40" rx="7"/>
  <text class="dg3-code" x="390" y="105" text-anchor="middle">Mixed</text>
  <line class="dg3-edge" x1="390" y1="120" x2="390" y2="140"/>
  <rect class="dg3-node" x="300" y="144" width="180" height="40" rx="7"/>
  <text class="dg3-code" x="390" y="169" text-anchor="middle">Dynamic</text>

  <!-- Order annotations -->
  <text class="dg3-sub" x="500" y="41">every axis known at compile time</text>
  <text class="dg3-sub" x="500" y="105">rank proven, extents partly runtime</text>
  <text class="dg3-sub" x="500" y="169">only Dyn - runtime rank</text>
  <text class="dg3-order" x="270" y="130" text-anchor="end">weaker &#8595;</text>
  <text class="dg3-order" x="270" y="66" text-anchor="end">stronger &#8593;</text>
  <text class="dg3-order" x="390" y="14" text-anchor="middle">top = identity for meet</text>
</svg>

`meet` forms a bounded semilattice - commutative, associative, idempotent,
topped by `Static` - and the test suite proves that property exhaustively
over all three levels rather than asserting it:

```rust,ignore
// crates/incin-core/src/exec/proof.rs (test)
fn meet_is_commutative_idempotent_and_topped_by_static() {
    let levels = [ProofLevel::Static, ProofLevel::Mixed, ProofLevel::Dynamic];
    for a in levels {
        assert_eq!(a.meet(a), a, "idempotent");
        assert_eq!(ProofLevel::Static.meet(a), a, "Static is the identity");
        for b in levels {
            assert_eq!(a.meet(b), b.meet(a), "commutative");
            ...
        }
    }
}
```

The practical consequence: adding a runtime operand to any operation weakens
its proof to exactly the right level, never more, never less. One runtime
axis anywhere in a chain makes the whole shape `Mixed`; only `Dyn` (runtime
rank) is `Dynamic`.

## Compile-time equality assertions

Where two operands must have identical shapes, the bound is not a doc
comment but a trait with a proof obligation:

```rust,ignore
// crates/incin-core/src/tensor/ops/binary.rs (bound of the shared helper)
S: ShapeEq<S2>,
...
<S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
```

`ASSERT_SHAPES_MATCH` is an associated constant defined only for the
reflexive impl (`impl<S> ShapeEq<S> for S`), so a mismatch between
statically-known shapes is a trait-resolution failure at the call site - the
same mechanism [Invariants](./invariants.md) describes for sealed operations.
Runtime-carrying axes are checked once, when the descriptor is inferred.

## Witnessed construction

Everything above produces claims. Crossing into execution, those claims are
carried by values whose constructors do not let you forge them
(`crates/incin-core/src/exec/proof.rs`):

```rust,ignore
// crates/incin-core/src/exec/proof.rs (abridged)
pub struct Validated<O> {
    descriptor: O,
    proof: ProofLevel,
    evidence: ShapeEvidence,
}

impl<O> Validated<O> {
    /// Crate-private on purpose. Only lowering rules call this,
    /// because only they hold the shape types the proof derives from.
    pub(crate) const fn new(descriptor: O, proof: ProofLevel) -> Self { ... }

    pub const fn descriptor(&self) -> &O { ... }
    pub const fn proof_level(&self) -> ProofLevel { ... }
    pub const fn shape_evidence(&self) -> ShapeEvidence { ... }
}
```

`ShapeEvidence` deserves its own look because it solves a provenance problem:
a plain `ProofLevel` argument would let any caller write `ProofLevel::Static`
beside whatever metadata it liked. The only constructors are
`ShapeEvidence::of::<S>()`, which reads `S::PROOF` off a real shape type and
cannot be told a different answer, and `ShapeEvidence::dynamic()`, which
claims nothing. [Lowering](./deep_lowering.md) shows where each enters dispatch.

This is the "proof token" category from [Invariants](./invariants.md):
no public constructor at all, because the guarantee has to live somewhere a
caller cannot reach.

## Gradient-state typing

`Tensor<S, B, K, G>`'s fourth parameter is typed the same way
(`crates/incin-core/src/tensor/grad.rs`):

- `Grad` / `NoGrad` - compile-time-fixed markers, `Arg = ()`,
  `REQUIRES_GRAD` a constant.
- `Dyn` - the runtime-toggled option, `Arg = bool`, stored as a bool field.

The `RequiresGrad` trait ties the marker to the execution layer:

```rust,ignore
// crates/incin-core/src/tensor/grad.rs (abridged)
fn grad_mode(grad: &Self::Field) -> GradMode {
    if Self::requires_grad(grad) {
        GradMode::Enabled
    } else {
        GradMode::Disabled
    }
}
```

That method is *derived*, not supplied per impl: a marker cannot claim it
tracks gradients and then decline to record, because there is only one
answer and this reads it. When an eager operation builds its context, the
scoped policy and the output marker combine as ceilings:

```rust,ignore
// crates/incin-core/src/tensor/grad.rs
let context = ExecutionContext::from_scope(B::default());
let mode = context.grad_mode().and(G::grad_mode(grad));
context.with_grad_mode(mode)
```

so `NoGrad` output means the dispatcher sees `training: false` even inside a
grad-enabled scope - the same policy field
[lowering](./deep_lowering.md) hands to capability queries.

## Where this crosses into execution

These types end at a deliberate border: the executor is *not* generic over
`S`. What crosses is the validated descriptor plus `ShapeEvidence`. The next
chapter follows that handoff.
