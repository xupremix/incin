# The target API and canonical dispatch

Feature `target-api`. This chapter covers an experimental, opt-in surface —
real and tested, but not yet the stable default the rest of this book uses.

## Allocation targets

A **target** is a value that knows where and how to allocate: a device
(`Cpu`, `Wgpu::new(0)`) or a backend rebound to a specific dtype. It has no
construction step and owns no resources.

```rust,no_run
use incin::prelude::*;

let x = Cpu.zeros(s![2, 3])?;                     // fully static shape and proof
let y = Cpu.zeros(shape![2, 3])?;                 // same, via the value macro
let batch = 4;
let z = Cpu.zeros(shape![batch, 3])?;             // dynamic batch axis
let w = Cpu.zeros([batch, 3])?;                   // fully dynamic
# Ok::<(), incin::Error>(())
```

`Static`, `Bound`, and a plain `[usize; N]` array all implement `ShapeSpec`,
each producing exactly the amount of compile-time proof its own staticness
earns — never more than what's actually known.

## Why this exists: the canonical execution path

Most of this book's `Tensor` methods reach their kernel through one of nine
broad "operation family" traits `Backend` requires. A newer, narrower path —
one exact identity per operation, validated before it reaches a backend,
output metadata derived rather than trusted — exists alongside it. The
allocation methods above (`zeros`, `ones`, `rand`, `randn`, and their
`_canonical`-suffixed siblings) are where it's wired in today.

The type-level shape (`s![2, 3]` vs `Dyn`) isn't just documentation on this
path — a backend can specialize on it. The CPU creation family reads
`S::STATIC_NUMEL` when it's known, skipping a runtime element-count
computation entirely for a `Static` shape spec. It's a small, measured win
(single-digit-percent end to end for the cheapest allocation), not a
type-system flourish with nothing behind it.

## Canonical arithmetic

`add` extends the same path to arithmetic — proof of the pattern,
not yet the whole operator set:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 2], B>::ones(())?;
let b = Tensor::<s![2, 2], B>::ones(())?;
let c = a.add(&b)?;
assert_eq!(c.dims().as_ref(), &[2, 2]);
# Ok::<(), incin::Error>(())
```

It behaves identically to `add` — same result, same `no_grad` handling, same
gradients — and today exists only where a backend has a real canonical
executor (CPU). It's a separate method rather than a change to `add`'s own
body: `add` is generic over every `Backend` implementation, including
lightweight test stand-ins with no descriptor metadata to build a request
from, and widening `add` itself would have meant either breaking those or
inventing a new cross-backend conversion this method didn't need.

## Should you reach for this?

Not by default. Everything in this chapter is marked experimental for a
reason — the ordinary `Tensor` methods used throughout the rest of this book
are the stable, documented surface. Reach for the target API when you
specifically want an allocation-target-style call site (`Cpu.zeros(...)`
instead of `Tensor::<S, B>::zeros(...)`), or when you're working on the
framework itself and want to route through the validated, capability-checked
path on purpose.
