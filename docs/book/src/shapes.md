# Shapes: static, dynamic, and mixed

A `Shape` is a type, and three kinds of type implement it:

- **A tuple of `typenum` dims**: every axis known at compile time.
  `s![2, 3, 224, 224]` expands to exactly this.
- **`Dyn`**: rank itself is unknown until a value exists.
- **A tuple mixing `usize` and `typenum` dims**: rank known, some axes are
  not. `s![usize, 128]`, or a named axis via `dim!(Batch)` used as
  `s![Batch, 128]`.

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

dim!(BatchSize);

type Image = s![3, 224, 224];       // fully static
type Batched = s![BatchSize, 128];  // named runtime axis
type Loose = s![usize, 128];        // unnamed runtime axis

let img = Tensor::<Image, B>::zeros(())?;
let batch = Tensor::<Batched, B>::zeros((8usize, ()))?;
let loose = Tensor::<Loose, B>::zeros((4usize, ()))?;
# Ok::<(), incin::Error>(())
```

A fully dynamic tensor uses `Dyn` and takes a `Vec<usize>`:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let x = Tensor::<Dyn, B>::zeros(vec![2, 3, 4])?;
assert_eq!(x.dims(), vec![2, 3, 4]);
# Ok::<(), incin::Error>(())
```

## Why bother with the static form

Two tensors with mismatched static shapes fail to compile, not to run:

```rust,compile_fail
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 3], B>::ones(())?;
let b = Tensor::<s![3, 2], B>::ones(())?;
let c = &a + &b; // does not compile: `s![2, 3]` does not equal `s![3, 2]`
# Ok::<(), incin::Error>(())
```

A layer's weight shape is part of its type too. `Linear<s![768, 256],
Backend>` only accepts a `[.., 768]` input and only produces a `[.., 256]`
output, checked the same way.

## Converting between static and dynamic

```rust,ignore
use incin::prelude::*;
type B = DefaultBackend;

let x = Tensor::<s![2, 3], B>::ones(())?;

// Erase the static shape.
let dynamic: Tensor<Dyn, B> = x.clone().into_dyn();

// Re-assert a static shape, checked at runtime this time.
let reasserted: Tensor<s![2, 3], B> = dynamic.to_shape::<s![2, 3]>()?;
# Ok::<(), incin::Error>(())
```

`into_dyn` always succeeds (a static shape is always a valid dynamic one).
`to_shape` is fallible: it checks the runtime dims against the target shape
and returns a typed error if they disagree, rather than panicking.

## `s!` vs `shape!`

`s!` names a **type**, used as a generic parameter. When you need a **value**
to pass to an allocation target (see [The target API](./target_api.md)),
`shape!` is the value-level counterpart, inferring which axes are static from
how they're written:

```rust,ignore
use incin::prelude::*;

let batch = 8;
let w = Cpu.zeros(shape![128, 784])?;      // Tensor<s![128, 784], ..>
let x = Cpu.zeros(shape![batch, 784])?;    // Tensor<s![usize, 784], ..>
# Ok::<(), incin::Error>(())
```

An integer literal in `shape![...]` is a static axis. A named `const` is also
available when written explicitly as `shape![const N, 784]`; this preserves
the constant extent instead of silently turning it into `Dyn`.
