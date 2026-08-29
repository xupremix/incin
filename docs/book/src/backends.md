# CPU, and what actually runs on GPU today

The backend is a type parameter (`B` in `Tensor<S, B, ...>`), not a runtime
switch: `CpuBackendImpl<Cpu>` and `CudaBackendImpl<Cuda>` are
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
| CUDA | 79 | preview |
| WGPU | 64 | preview |
| Metal | 34 | preview |

Those are counts from the `Element types by operation and backend` matrix, so
they say what each backend *advertises*, which is not the same as what it has
been observed to compute. Two caveats belong next to the numbers:

- **Advertised is not trainable.** The per-backend tables carry a `Training`
  column. CUDA marks 66 of its 93 rows trainable; `layer_norm` and
  `batch_norm` are `Training: no`, so they are forward-only and cannot appear
  in a backward pass. Metal marks 23 of 48 trainable, WGPU 53 of 76.
- **Metal is the least proven of the three.** Its own feature description
  calls its executors stubs pending MTL-002/003, and every Metal row reports
  `Implementation: native`, so that column does not distinguish a finished
  kernel from a placeholder. Read Metal's 25 as a registry claim, not a
  capability.

The previews all cover basic arithmetic (`add`/`sub`/`mul`/`div`), reductions,
`matmul`, and `conv2d`/pooling. WGPU additionally advertises thirteen unary
activations (`relu`, `step`, `mish`, `elu`, `gelu`, `abs`, `exp`, `neg`,
`sqrt`, `log`, `tanh`, `sigmoid`, `swish`), which Metal does not, plus
`maximum`/`minimum`/`abs_diff` and the shape views `transpose`, `flatten`,
`squeeze` and `unsqueeze`. CUDA and WGPU both advertise `softmax` and
`rms_norm`, trainable and composed from primitives rather than fused; CUDA
adds the two forward-only normalizations above.

What **no** accelerator backend has: any loss function, `embedding`,
`dropout`, or `group_norm`.

Concretely: you can allocate tensors, run matrix arithmetic, and apply an
activation on a GPU today, but you cannot train the models in this book's
[Building models](./building_models.md) chapter on one. Two independent things
are missing, not one: every loss function is CPU-only, and where a
normalization layer does exist on CUDA it has no backward. CPU is where real
training happens right now.

This is not a documentation gap to work around by trying harder; it's
missing kernels. A backend that doesn't support an operation refuses it with
a typed `UnsupportedReason` rather than doing something wrong silently, so
you'll find out immediately rather than discover it three epochs in, but
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

## Picking a backend when you don't know the machine

Two more rungs exist above the explicit choice, and they answer different
questions. Confusing them is the most common mistake on this axis.

`best_device!()` names the most capable device **this build can target**. It
expands to a type alias resolved from the Cargo features incin was compiled
with, probes no hardware, and touches neither the filesystem nor the network,
which is why it can appear in a type position at all:

```rust,no_run
use incin::prelude::*;

type Dev = IncinBackend<incin_core::best_device!()>;
let x = Tensor::<s![2, 3], Dev>::zeros(())?;
# Ok::<(), incin::Error>(())
```

`detect_device()` probes the **machine**, trying CUDA, then Metal, then WGPU,
then CPU, and returns the first family with usable hardware. Its answer is a
run-time `DeviceId` rather than a type.

You can allocate directly via the target-first API:

```rust,no_run
use incin::prelude::*;
use incin_backends::detect::detect_device;
use incin_backends::target::{Native, Target};
use incin_core::tensor::device::DeviceId;

let device = detect_device().unwrap_or_else(DeviceId::cpu);
let target: Target<Native, Dyn> = Target::new((), device, ());

let x = target.zeros([2, 3])?;
# Ok::<(), incin::Error>(())
```

Or via explicit type-level construction:

```rust,no_run
use incin::prelude::*;

let device = incin_backends::detect_device().expect("a usable backend");
let x = Tensor::<Dyn, IncinBackend<Dyn>>::zeros((vec![2, 3], device))?;
# Ok::<(), incin::Error>(())
```

The trade is the one this whole book is about. A compile-time backend lets
`B: Execute<op::MatMul>` be decided before the program runs, so an operation
the device cannot do is a compile error. A run-time backend accepts whatever
hardware is present, and in exchange the compiler can no longer tell you the
device is wrong for the operation, only the `Result` can.

The two disagree exactly when the most capable compiled-in backend has no
working hardware: a `--features cuda` build on a machine with no NVIDIA card
still resolves `best_device!()` to CUDA, while `detect_device()` falls through
to whatever is actually there. `detect_device_in(&[..])` pins a preference
order when the default one is wrong for you.

`cargo run -p incin --example device_selection --features cpu` prints all four
rungs side by side, and `cargo run -p incin --example target_api_dynamic --features cpu`
shows dynamic device and dtype creation end to end.

## `cargo incin doctor`

The `doctor` report (via the library's `doctor` module, `std` feature) lists
which backend features are compiled in, what devices were detected, and
cache state for the running build, the fastest way to check what a given
build can actually reach before writing code against it.

It also answers the dtype question from
[Tensors](./tensors.md#declaring-a-dtype-is-not-computing-in-one) directly: the
`[probes]` section asks each detected device about representative operations
and prints the capability verdict, so you can see what a build can compute
before writing code against it.

### Checking for a newer incin

`cargo incin doctor --check-updates` compares the running binary's version
against the newest non-yanked release on crates.io and prints one line:

```text
[update]
incin 0.2.0 is available (running 0.1.0); update with `cargo install cargo-incin --force`
```

Two deliberate constraints. It never runs on its own: no delegated command
(`build`, `check`, `test`, or anything forwarded to cargo) touches the network,
and the flag has to be typed. And it is compiled out unless the `update-check`
feature is on, so a build that does not want an HTTP stack does not link one:

```bash
cargo install cargo-incin --features update-check
```

Without the feature the flag reports that the build lacks it rather than
failing. `CARGO_NET_OFFLINE=true` skips the check, and any network or parse
failure is reported as inconclusive rather than turning into an error, so
`doctor` stays useful with no connectivity.
