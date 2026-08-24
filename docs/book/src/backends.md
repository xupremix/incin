# CPU, and what actually runs on GPU today

The backend is a type parameter (`B` in `Tensor<S, B, ...>`), not a runtime
switch - `CpuBackendImpl<Cpu>` and `CudaBackendImpl<Cuda>` are
different types, and which one you use is fixed at compile time by which
type you wrote.

```rust,no_run
use incin::prelude::*;

type OnCpu = IncinBackend<Cpu>;
#[cfg(feature = "cuda")]
type OnCuda = IncinBackend<Cuda>;

let x = Tensor::<s![2, 3], OnCpu>::zeros(())?;
# Ok::<(), incin::Error>(())
```

## The honest coverage picture

Every layer, loss, optimizer, and operation in this book runs on **CPU**.
Counted directly from `docs/capabilities.md`, which is generated from the
backend registrations rather than written by hand:

| Backend | Operations advertised | Tier |
|---|---:|---|
| CPU | 158 | complete, and the only one verified by execution |
| CUDA | 70 | preview |
| WGPU | 46 | preview |
| Metal | 25 | preview |

The previews all cover basic arithmetic (`add`/`sub`/`mul`/`div`), reductions,
`matmul`, and `conv2d`/pooling. WGPU additionally advertises thirteen unary
activations - `relu`, `step`, `mish`, `elu`, `gelu`, `abs`, `exp`, `neg`,
`sqrt`, `log`, `tanh`, `sigmoid`, `swish` - which Metal does not. CUDA
additionally advertises `layer_norm`, `batch_norm`, `rms_norm`, and
`softmax`, which neither of the other two has.

What **no** accelerator backend has: any loss function, `embedding`,
`dropout`, or `group_norm`.

Concretely: you can allocate tensors, run matrix arithmetic, and apply an
activation on a GPU today, but you cannot train the models in this book's
[Building models](./building_models.md) chapter on one, because the loss is
always the missing link even where the layers are present. CPU is where real
training happens right now.

This is not a documentation gap to work around by trying harder; it's
missing kernels. A backend that doesn't support an operation refuses it with
a typed `UnsupportedReason` rather than doing something wrong silently, so
you'll find out immediately rather than discover it three epochs in - but
the fix is writing the kernel, not finding the right incantation.

## Picking a backend at compile time

```rust,no_run
use incin::prelude::*;

// DefaultDevice is selected independently: CPU wins when enabled, otherwise
// WGPU, then CUDA. DefaultBackend exists only when CPU is enabled and is
// always IncinBackend<Cpu>; it has no accelerator fallback.
let x = Tensor::<s![2, 3], DefaultBackend>::zeros(())?;
# Ok::<(), incin::Error>(())
```

For an explicit choice regardless of what's enabled, name the backend and
device directly: `IncinBackend<Cpu>`, `IncinBackend<Cuda>`, and so
on, gated behind the matching Cargo feature (`cpu`, `cuda`, `wgpu`, `metal`).

## `cargo incin doctor`

The `doctor` report (via the library's `doctor` module, `std` feature) lists
which backend features are compiled in, what devices were detected, and
cache state for the running build - the fastest way to check what a given
build can actually reach before writing code against it.
