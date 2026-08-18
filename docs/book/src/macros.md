# The macro reference

Every macro Incin exports, what it produces, and when to reach for it. They
divide cleanly by what they *make*: a type, a value, or an item.

| Macro | Produces | Covered in |
|---|---|---|
| `s![...]` | a shape **type** | [Shapes](./shapes.md) |
| `shape![...]` | a shape **value** for a target | [Shapes](./shapes.md), [Target API](./target_api.md) |
| `axis!(...)` | an operation-axis **selector** | [Target API](./target_api.md) |
| `i![...]` | indexing and slicing **arguments** | below |
| `dim!(...)` | named dimension **types** | below |
| `tensor![...]` | a `Result<Tensor>` **value** | [Tensors](./tensors.md) |
| `idx![...]` | a slicing **type** | below |
| `#[module]` | trait **impls** on your struct | [Building models](./building_models.md) |
| `seq!` / `SeqTy!` | a `Sequential` value / its type | [Sequential](./sequential.md) |
| `best_device!()` | a device **type** | [Backends](./backends.md), below |
| `mesh!`, `placement!`, `parallel!` | distributed **types** | [Experimental](./experimental.md) |
| `model!`, `import_model!` | a **module** from an ONNX file | [Saving and loading](./saving_loading.md) |

## `dim!` - named dimensions

Declares a dimension type whose *size is a runtime value* but whose *identity
is compile-time*. Two tensors both carrying `Batch` are known to agree on that
axis; a `Batch` used where a `Seq` is expected does not compile - even though
neither size is known until run time.

```rust,no_run
use incin::prelude::*;

dim!(Batch, Seq);

let x = Tensor::<s![Batch, 128], DefaultBackend>::zeros((8usize, ()))?;
assert_eq!(x.dims().as_ref(), &[8, 128]);
# Ok::<(), incin::Error>(())
```

This is the middle ground the shape system exists to make available:
`s![8, 128]` is fully static and `s![usize, 128]` is anonymously dynamic, but
`s![Batch, 128]` is dynamic *and* named, so the compiler still catches an axis
mix-up. `Dim::STATIC_SIZE` is `false` for a named dimension, which is why it
weakens a shape's proof to `Mixed` - naming an axis makes it checkable, not
statically sized.

Names and extents are independent. A named static axis uses the same canonical
structural representation as an anonymous static axis:

```rust,no_run
use incin::prelude::*;

dim!(Batch);
type StaticBatch = s![Batch = 25, 128];
```

For a compile-time value held in a const item, use the explicit `const`
marker. This is distinct from a named runtime dimension:

```rust,no_run
use incin::prelude::*;

const BATCH: usize = 25;
dim!(Batch, Features);
type Fixed = s![Batch = const BATCH, Features = 128];
let _: Tensor<Fixed, DefaultBackend> =
    Cpu.zeros(shape![Batch = const BATCH, Features = 128])?;
# Ok::<(), incin::Error>(())
```

Bare paths such as `s![Batch, Features]` continue to mean named dimensions.
That distinction is required so existing named-shape code keeps its axis
identity checks.

## `idx!` - advanced type-level targets

`shape![..., infer]` is the normal reshape API. Known literal extents remain in
the output type, while the inferred extent is carried as `usize`. The older `idx!` macro builds
the heterogeneous type-level target used by the advanced `reshape_idx` API:

`idx!` is available from `incin::prelude::*`. The explicit
`incin::macros::advanced::idx` path remains available for code that keeps
advanced macros separate from the ordinary prelude.

| Syntax | Meaning |
|---|---|
| `0..5` | a statically bounded slice, `Slice<U0, U5>` |
| `..` | take the whole axis |
| `...` | ellipsis - fill the axes not otherwise named |
| `-1` | `InferDim`, an inferred extent in a type-level target |

```rust,no_run
use incin::prelude::*;
use incin::macros::advanced::idx;

let t = Tensor::<s![10, 20, 30], DefaultBackend>::zeros(())?;
let reshaped = t.reshape_infer(shape![6, infer])?;
// The result keeps the partial shape s![6, usize].
# Ok::<(), incin::Error>(())
```

Use `i![...]` for ordinary runtime indexing and slicing. It supports negative
indices and signed range bounds:

```rust,no_run
use incin::prelude::*;

let t = Tensor::<s![10, 20], DefaultBackend>::zeros(())?;
let last_row = t.get(i![-1, ..])?;
let middle = t.get(i![.., 2..5])?;
# Ok::<(), incin::Error>(())
```

`i![]` expands to a vector of index specifications, not a bounded tuple. It
therefore accepts any number of entries, including one entry for every axis of
a high-rank tensor. Indexing and reshape inference remain separate APIs.

The `reshape_idx` result shape is computed in the type system. Runtime indexing
returns a dynamic shape because the selected range can depend on runtime
values. For value-level runtime inference, use `shape![6, infer]` with
`reshape_infer`.

## `best_device!` - compile-time device selection

Expands to a device *type* chosen from the enabled Cargo features, optionally
at a given ordinal:

```rust,no_run
type Dev = incin_core::best_device!();
type Second = incin_core::best_device!(incin_core::typenum::U1);
# let _: core::marker::PhantomData<(Dev, Second)> = core::marker::PhantomData;
```

It performs no discovery: no filesystem, no network, no hardware probe. It is
a naming convenience over the feature-gated aliases and nothing more. For
*runtime* hardware detection use `incin_backends::detect_device()`, which is a
different question with a different answer - a build may have `cuda` compiled
in on a machine with no CUDA device.

The `cfg` resolution happens inside `incin-core` rather than in the macro body
on purpose: a `#[cfg(feature = "cuda")]` written inside a `macro_rules!` body
is evaluated against the *calling* crate's features, so it would read as
disabled in every downstream crate and silently select CPU.

## Path resolution, and the one thing that breaks it

Every macro here expands to absolute `::incin::...` (or `::incin_core::...`)
paths, so it resolves against the crate rather than whatever the caller has in
scope - including a module of the caller's own named `incin`.

The one form none of them survives is a *package rename* in the caller's
`Cargo.toml`:

```toml
incin_x = { package = "incin" }   # macros will not resolve
```

`::incin` then names a crate that isn't there. Resolving the real name would
mean reading the caller's manifest at expansion time, which the repository's
macro policy forbids.
