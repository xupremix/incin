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

The default is `Unknown`, and it means exactly what it says: nothing has been
established. An `Unknown` tensor works exactly as it always did — every
operation is available and every check happens at runtime.

```rust,ignore
// `Unknown` is the default, so this is unchanged from before layouts existed.
let t: Tensor<s![3, 4], B> = Tensor::zeros(())?;
```

What `Unknown` cannot do is satisfy a bound that needs a proof. That is the
whole mechanism.

## Earning a proof

A tensor's strides are a runtime fact, so the way to acquire a layout is to
check them:

```rust,ignore
let proven = t.into_row_major()?;   // Tensor<s![3, 4], B, .., RowMajor<s![3, 4]>>
```

`into_row_major` compares the tensor's actual strides against the dense
row-major pattern and only succeeds if they match. There is deliberately **no**
`assume_row_major`. An unchecked promotion would make every downstream bound
meaningless, and the check it saves is a stride comparison over the rank.

For the common case there is an alias, which avoids naming the shape twice:

```rust,ignore
fn takes_dense(t: &Dense<s![3, 4], B>) { /* .. */ }
```

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

Layout is newer than the rest of the type system and is being adopted module by
module. Operations that have been converted carry their operand's layout
through; operations that have not still bind `L` to `Unknown`, which means a
tensor carrying a proof cannot call them yet.

The conversion is complete: accessors, constructors, `Clone`, pointwise unary
and binary operations, reductions, shape manipulation, matmul, and the `nn`
layers all accept a tensor that carries a layout.

Two rules govern what an operation's *result* carries, and the difference is
not a style choice:

- **Shape-preserving operations carry the operand's layout through.** A proof
  survives a chain of them, so `reshape_view` is still reachable at the end of
  one.
- **Shape-changing operations state their result's layout, and state it as
  `Unknown`.** A layout describes one geometry and cannot be carried to
  another. They *could* claim `RowMajor`, since the result is a fresh
  allocation — except that is not true on every backend: CPU `transpose`
  returns a view while CUDA's returns a copy. Until that is settled
  ([#113](https://github.com/xupremix/incin/issues/113)) the honest answer is
  to claim nothing, and `into_row_major` recovers a proof where one is wanted.

If you want the no-copy behaviour explicitly, `transpose_view` is a separate
operation that permutes shape and strides over the same buffer. Which of the two
is faster depends on how often you read the result — measured on a GTX 1650, the
view is about 45% faster for a single pointwise consumer and about 23% slower by
eight, crossing over at roughly four. That is a property of your consumer, so the
choice is yours to make rather than the framework's.
