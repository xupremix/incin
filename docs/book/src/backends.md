# CPU, and what actually runs on GPU today

The backend is a type parameter (`B` in `Tensor<S, B, ...>`), not a runtime
switch — `CpuBackendImpl<Cpu>` and `CudaBackendImpl<Cuda>` are
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
Measured directly against `docs/capabilities.md` (generated from the actual
backend registrations, not aspirational): CUDA and WGPU each implement
roughly twenty to thirty operations — basic arithmetic (`add`/`sub`/`mul`/
`div`), reductions, `matmul`, and `conv2d`/pooling. Neither has **any**
activation function (`relu`, `gelu`, `sigmoid`, `tanh`, ...), normalization
(`layer_norm`, `batch_norm`, `group_norm`), loss function, `embedding`, or
`dropout`. Metal's coverage is narrower still.

Concretely: you can allocate tensors and run basic matrix arithmetic on an
accelerator today, but you cannot train the models in this book's [Building
models](./building_models.md) chapter — or a CNN, an RNN, or anything with a
normalization layer — on GPU. CPU is where real training happens right now.

This is not a documentation gap to work around by trying harder; it's
missing kernels. A backend that doesn't support an operation refuses it with
a typed `UnsupportedReason` rather than doing something wrong silently, so
you'll find out immediately rather than discover it three epochs in — but
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
cache state for the running build — the fastest way to check what a given
build can actually reach before writing code against it.
