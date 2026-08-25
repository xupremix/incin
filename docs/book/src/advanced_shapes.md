# Advanced shapes

[Shapes](./shapes.md) covers the three kinds of shape. This chapter is the
rest of the type-level machinery: reshaping vs re-asserting, slicing, the
broadcast rules, and the traits that decide what compiles.

## `reshape` vs `to_shape`: the distinction that bites

They sound interchangeable and are not:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let x = Tensor::<s![2, 6], B>::zeros(())?;

// reshape: a DIFFERENT geometry, same element count.
let r = x.reshape(shape![3, 4])?;
assert_eq!(r.dims().as_ref(), &[3, 4]);

// to_shape: the SAME dims, re-asserted as a static type.
let d = x.clone().into_dyn();
let same = d.to_shape::<s![2, 6]>()?;
assert_eq!(same.dims().as_ref(), &[2, 6]);

// to_shape to different dims fails at run time - it is not a reshape.
assert!(x.into_dyn().to_shape::<s![3, 4]>().is_err());
# Ok::<(), incin::Error>(())
```

`to_shape` calls `S2::from_dyn(current_dims)`, which returns `None` when the
target shape type does not accept the dims the tensor actually has. So it is
the tool for recovering a static type after `into_dyn` erased one, and never
the tool for changing a layout.

`reshape`'s argument is the target shape's `Arg`: `((), ())` for a fully
static rank-2 target, because each static axis contributes a unit. A target
with runtime axes takes their sizes there instead.

## Slicing with `i!`

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let t = Tensor::<s![10, 20, 30], B>::zeros(())?;

// One entry per axis. The result shape is computed in the type system.
let view = t.get(i![0..5, .., 15..30])?;
assert_eq!(view.dims().as_ref(), &[5, 20, 15]);
# Ok::<(), incin::Error>(())
```

`view` is dynamically shaped because indexing is a runtime operation. For
shape inference, use `shape![..., infer]` with `reshape_infer`; `-1` is not
used as an indexing sentinel.

## Broadcasting

Two different rules are in play, and which one you get depends on how you
wrote the operation:

| Form | Rule | Fails how |
|---|---|---|
| `a.try_add(&b)` | broadcasting | recoverable error |
| `a + b` | `BroadcastShape`, numpy-style | panic on dynamic/backend error |
| `a.broadcast_add(&b)` | `BroadcastShape` | compile error if unbroadcastable |

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 3], B>::ones(())?;
let b = Tensor::<s![3], B>::ones(())?;

let via_operator = a.clone() + b.clone();      // broadcasts; panic convenience boundary
let via_method = a.broadcast_add(&b)?;          // same thing
assert_eq!(via_operator.dims().as_ref(), &[2, 3]);
# Ok::<(), incin::Error>(())
```

The type-level `BroadcastShape` proof determines the output type. Runtime
shape or backend validation remains recoverable through `try_add` and the
other named methods. Operator syntax panics instead, with no error or tensor
contents in its fixed message.

`broadcast_left` exists for the left-aligned case that right-aligned
broadcasting cannot express.

## The shape traits

| Trait | Says |
|---|---|
| `Shape` | The base: rank, per-axis extents, ShapeBuf resolution, and proof constants. |
| `DynShape` | Rank and element count are recoverable at run time (`rank`, `checked_numel`). |
| `PartialDynShape` | Known rank, some runtime axes. |
| `ShapeEq<S2>` | These two shapes are the same. Used by the exact-match operations. |
| `BroadcastShape<S2>` | These two broadcast, and here is the `Output` shape. |

Two constants on `Shape` carry the proof information the execution path uses:

- **`PROOF`**: `Static` (rank and every extent known), `Mixed` (known rank,
  some runtime axis), or `Dynamic` (rank itself unknown). Both default to the
  weakest honest answer, so a `Shape` implemented outside the crate is
  credited with nothing it hasn't demonstrated.
- **`STATIC_NUMEL`**: `Some(n)` when the type alone settles the element
  count, `None` otherwise.

`STATIC_NUMEL` exists in that `Option` form for a specific reason: stable Rust
cannot branch on whether a generic `S` exposes a stronger static-shape trait
without specialization. Restating the count on the base trait means any `S` can
be asked, and because `S` is a type parameter, `if let Some(n) =
S::STATIC_NUMEL` collapses to one arm at monomorphization. That is what lets a
backend specialize on a static shape; see
[Backend authoring](./backend_authoring.md).

## Named dimensions and the middle ground

`dim!(Batch)` gives an axis a compile-time *identity* without a compile-time
*size*. `dim!(Channels)` can also be paired with a static extent as
`s![Channels = 64]`; the name and extent are independent. See [the macro
reference](./macros.md#dim--named-dimensions).

This matters for real models: batch size and sequence length genuinely are
runtime values, but confusing them with each other is still a bug the
compiler can catch.
