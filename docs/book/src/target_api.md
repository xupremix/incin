# The target API and canonical dispatch

Feature `target-api` is experimental and not enabled by default. When enabled,
it is the preferred application-facing allocation surface. Explicit
`Tensor::<S, B>::...` constructors
remain the backend-authoring form for code that intentionally fixes `B`.

## Allocation targets

A **target** is the preferred user-facing allocation entry point: a value that
knows where and how to allocate: a device
(`Cpu`, `Wgpu::new(0)`) or a backend rebound to a specific dtype. It has no
construction step and owns no resources.

```rust,no_run
use incin::prelude::*;

let x = Cpu.zeros(shape![2, 3])?;                 // fully static shape and proof
let y = Cpu.zeros(shape![2, 3])?;                 // same, via the value macro
let batch = 4;
let z = Cpu.zeros(shape![batch, 3])?;             // dynamic batch axis
let w = Cpu.zeros([batch, 3])?;                   // fully dynamic
# Ok::<(), incin::Error>(())
```

`Static`, `Bound`, and a plain `[usize; N]` array all implement `ShapeSpec`,
each producing exactly the amount of compile-time proof its own staticness
earns — never more than what's actually known.

## Why this exists: the allocation-target UX

The target methods are an opt-in ergonomic surface around the validated
descriptor execution architecture. Stable tensor operations already use the
single descriptor path.
The allocation methods above (`zeros`, `ones`, `rand`, `randn`, and their
`_canonical`-suffixed siblings) expose the target-shaped spelling while that
surface is feature-gated and available where the target backend is enabled.

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

For application code, prefer the allocation-target spelling (`Cpu.zeros(...)`)
when `target-api` is enabled. Use `Tensor::<S, B>::zeros(...)` when writing
backend-generic or backend-authoring code that intentionally fixes `B`.
