# Layout: proving where elements live

A [shape](./shapes.md) says how many elements a tensor has and how they are
indexed logically. It says nothing about where those elements sit in memory.
That second question — the *layout* — decides whether a kernel can address
linearly, whether a vectorised load is legal, and whether reinterpreting a
tensor under a new shape is even meaningful.

Incin tracks layout in the type as well, through a sixth `Tensor` parameter.

```rust,ignore
Tensor<S, B, K, G, P, L>
//                   ^ layout
```

## Nothing is credited that has not been shown

The default is `Dyn` — the same marker the shape, dtype, device and placement
slots use for "decided at runtime" — and in the layout slot it means exactly
that: nothing has been established. Such a tensor works exactly as it always
did; every operation is available and every check happens at runtime.

```rust,ignore
// `Dyn` is the default, so this is unchanged from before layouts existed.
let t: Tensor<s![3, 4], B> = Tensor::zeros(())?;
```

What `Dyn` cannot do is satisfy a bound that needs a proof. That is the whole
mechanism.

One marker does duty in every slot, so a fully written-out tensor can name it
twice, once for a dynamic shape and once for an unproven layout:

```rust,ignore
Tensor<Dyn, B, f32, NoGrad, Local, Dyn>
//     ^ shape decided at runtime      ^ layout not proven
```

Editor hover and the `expected .. found ..` notes drop the layout one, because
it carries no information, and keep the shape one, because it carries a great
deal. They are told apart by position — the layout is the sixth argument — not
by name.

## Earning a proof

There are three ways, and the cheapest is usually the right one.

### Let a pointwise operation give you one

A pointwise operation allocates a fresh packed buffer whatever its operand's
strides were, so its result says so:

```rust,ignore
let t: Tensor<s![3, 4], B> = Tensor::zeros(())?;  // proves nothing
let r = t.relu()?;                                 // proves RowMajor
let flat = r.reshape_view::<s![12]>()?;            // opens, with no check
```

This is why the claim is *stated* rather than carried from the operand. Carrying
it would propagate only what the caller already had, and would be wrong the
moment a layout that is not row-major exists — a `ChannelsLast` operand would
hand its claim to a row-major result.

`Linear` and `matmul` deliberately do not do this. They allocate too, but the
claim here rests on conformance tests that feed a genuinely strided operand and
check the result, and those cover the pointwise surface only. A layout is
claimed by a test that fails without it.

### Ask for it at construction

A constructor allocates a packed row-major buffer, so it already knows the
answer. Name the layout you want and it hands one back — no check, because
there is nothing to check:

```rust,ignore
// A real `RowMajor` proof, straight from the allocation.
let dense: Dense<s![3, 4], B> = Tensor::zeros(())?;
```

`Dense<S, B>` is the alias for `Tensor<S, B, .., RowMajor<S>>`; it avoids naming
the shape twice, since a row-major layout is always congruent with the shape it
describes.

This works because the constructors are generic over the layout parameter, and
that generality is bounded on the sealed `FreshDense<S>`. The bound is what
stops the same mechanism from being a way to *forge* a proof: an unbounded
constructor would hand you a tensor claiming whatever layout you named. A fresh
allocation genuinely is `Dyn` and genuinely is `RowMajor`, so those two
implement it and nothing else can — not even a layout defined downstream.

### Check the strides of one you already have

When the tensor came from somewhere else, its strides are a runtime fact and the
only honest route is to look:

```rust,ignore
let proven = t.into_row_major()?;   // Tensor<s![3, 4], B, .., RowMajor<s![3, 4]>>
```

`into_row_major` compares the tensor's actual strides against the dense
row-major pattern and only succeeds if they match. There is deliberately **no**
`assume_row_major`. An unchecked promotion would make every downstream bound
meaningless, and the check it saves is a stride comparison over the rank.

## What a proof buys

Operations that are only meaningful on a contiguous buffer can require one.
`reshape_view` reinterprets a tensor's elements under a new shape without
copying, which is correct exactly when those elements form one unbroken run:

```rust,ignore
let flat = proven.reshape_view::<s![12]>()?;   // fine
```

On a tensor that has not proven contiguity, that call does not compile. In most
frameworks the equivalent is a runtime error discovered on a transposed input;
here it is rejected at the call site.

Element-count equality is settled statically too, from the shape types rather
than their runtime dimensions, so a reshape to an incompatible size is also a
compile error rather than an `Err`.

## Facts are traits, not parameters

Strides, offset, alignment and contiguity are four facts but one parameter.
They are exposed as traits you bound where you need them:

| Trait | Means |
|---|---|
| `Contiguous` | the elements form one unbroken ascending run |
| `LayoutOf<S>` | this layout describes a tensor of shape `S` |

There is no alignment trait yet. Alignment is a property of the allocation
rather than of the shape, so no layout derived from a shape can claim it; it
would need its own checked promotion and a backend that consumes the bound.

This is the same pattern [shapes](./advanced_shapes.md) use: a single parameter
carrying the bundle, with the individual facts as bounds. `L: Contiguous` reads
well; a separate type parameter per fact would not.

`LayoutOf<S>` is the congruence rule — one stride per extent. Stating it once
is why layout is a bundle rather than several independent parameters that
would have to be kept consistent by hand.

## Writing generic code without naming six parameters

`Tensor<S, B, K, G, P, L>` earns each of its parameters, but a helper that wants
to be generic over tensors would otherwise pay for all of them at once — six
type parameters and six bounds before it can say anything.

`AnyTensor` collapses that to one:

```rust,ignore
fn numel_of<T: AnyTensor>(t: &T) -> usize
where
    T::Shape: DynShape,
{
    t.as_tensor().numel()
}
```

The parameters are still reachable as associated types, so a bound that
genuinely needs one writes `T::Backend: Execute<op::Add>` or
`T::Layout: Contiguous`. What changes is that a helper names only the parts it
constrains.

For the concrete case there is `Dense<S, B>`, which also avoids naming the
shape twice — `RowMajor` is congruent with the shape it describes, so
`Tensor<S, .., RowMajor<S>>` repeats it for nothing.

## What the backend does with it

A layout, like a shape, travels to the backend as evidence rather than as a
type. `ShapeEvidence` already carries what the shape type settled — its proof
level, rank, element count and per-axis extents — and a backend reads it
through `Validated::shape_evidence()`.

The CUDA pointwise path uses that today: a statically known element count that
divides the vector width proves a packed kernel's ragged-tail branch
unreachable, so it is not emitted. Per-axis extents let a strided kernel's
index arithmetic use literal divisors, which the compiler lowers to
multiply-and-shift rather than leaving as integer division.

The distinction that makes this worth doing is not that the kernel learns the
element count — a launcher always knows that at runtime. It is that a
*statically* known count is a constant of the program, so specialising on it
produces a bounded number of kernels rather than one per observed shape.

## Current status

Layout is newer than the rest of the type system, but every operation now
accepts a tensor that carries one: accessors, constructors, `Clone`, pointwise
unary and binary operations, reductions, shape manipulation, matmul, and the
`nn` layers. A tensor that has earned a proof can be passed anywhere a tensor
without one can.

Two rules govern what an operation's *result* claims, and the difference is not
a style choice:

- **An operation that is known to allocate a fresh packed buffer states
  `RowMajor` of its own result shape.** Pointwise unary and binary operations
  (broadcasting ones included), the comparison and logical families,
  `masked_fill`, `where_cond`, `lerp`, `cumsum` and every reduction do this, so
  a proof appears out of the middle of a chain and `reshape_view` is reachable
  at the end of one. The claim is *stated*, never carried: carrying the operand's
  layout would propagate only what the caller already had, and would be false
  the moment a non-row-major layout exists.
- **An operation whose result's memory order is not settled states `Dyn`.**
  A layout describes one geometry and cannot be carried to another, and
  shape-changing operations do not agree across backends: CPU `transpose`
  returns a view while CUDA's returns a copy. Until that is settled
  ([#113](https://github.com/xupremix/incin/issues/113)) the honest answer is
  to claim nothing, and `into_row_major` recovers a proof where one is wanted.

Nothing carries `L` from an operand into a result. The layout parameter is an
input — it is what `reshape_view` reads and what `forget_layout` discards — and
every result's claim is made by the operation itself, backed by a conformance
test that feeds a genuinely strided operand and fails if the claim is wrong.

If you want the no-copy behaviour explicitly, `transpose_view` is a separate
operation that permutes shape and strides over the same buffer. Which of the two
is faster depends on how often you read the result — measured on a GTX 1650, the
view is about 45% faster for a single pointwise consumer and about 23% slower by
eight, crossing over at roughly four. That is a property of your consumer, so the
choice is yours to make rather than the framework's.
