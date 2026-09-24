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
| CPU | 169 | complete, and the only one verified across the full catalog |
| CUDA | 167 | preview, compile-gated (no automated device run) |
| WGPU | 137 | preview, software-adapter execution evidence |
| Metal | 107 | preview, host-verified + macOS Apple-Silicon job |

Those are counts from the `Element types by operation and backend` matrix, so
they say what each backend *advertises*, which is not the same as what it has
been observed to compute. Two caveats belong next to the numbers:

- **Advertised is not trainable.** The per-backend tables carry a `Training`
  column. CUDA marks 147 of its 204 rows trainable; CPU 137 of 187; WGPU
  123 of 155; Metal 89 of 119.
- **Metal is the least proven of the three.** The Metal shader and MPS
  infrastructure from MTL-001/002/003 is complete and the gaps are operation
  coverage on top of it (`conv2d`/pooling still empty; `instance_norm`,
  `masked_fill`/`where_cond`, and `cmp_*`/`logical_*` unadvertised), and
  every Metal row reports `Implementation: native`, so that column does not
  distinguish a finished kernel from a placeholder. Host-side suites
  (`metal_gap_ops`, `metal_attention_ops`) run the math on any OS; a real
  Metal device run is the macOS Apple-Silicon hardware job only. Read
  Metal's 107 as a registry claim, not a capability proof.

The previews all cover basic arithmetic (`add`/`sub`/`mul`/`div`), reductions,
`matmul`, and thirteen unary activations (`relu`, `step`, `mish`, `elu`,
`gelu`, `abs`, `exp`, `neg`, `sqrt`, `log`, `tanh`, `sigmoid`, `swish`) —
including Metal, which gained the Batch-A pointwise rows in `cbe7bac5`/`bf042c80`.
WGPU and Metal both advertise `softmax`/`log_softmax`, `rms_norm`, and the
composed losses; WGPU additionally covers `conv2d`/pooling, the normalization
family through `group_norm`/`instance_norm`, `embedding`, `dropout`, and
attention (`scaled_dot_product_attention`). CUDA adds the full training path
(batch-norm training, losses, dropout, embedding) with native and composed
rows.

What the accelerator previews still lack relative to CPU:

- **WGPU/Metal:** `cmp_*` and `logical_*` (bool rows need a multi-dtype
  advertising policy; WGPU storage admits ints/bools but `native_precision`
  only names `f32` compute). Metal also lacks spatial ops and bool-mask
  select ops.
- **WGPU:** compute (including `matmul`) is `f32`-only even though storage
  accepts `u8`/`u32`/`i64`/`bool`.
- **CUDA:** still no automated device execution in this repository — value
  tests are `#[ignore]`d pending `HARDWARE_CUDA_RUNNER`.

Concretely: you can allocate tensors, run matrix arithmetic, apply
activations, losses, and normalization on a GPU today, and train attention
on WGPU. Advertised is not the same as hardware-proven: every-PR CI runs a
WGPU **software** adapter (lavapipe) suite, but the CUDA training path
remains declared capability awaiting NVIDIA-device evidence (issues #82 and
#83). CPU remains where verified training happens for the complete catalog.

This is not a documentation gap to work around by trying harder; it's
missing kernels. A backend that doesn't support an operation refuses it with
a typed `UnsupportedReason` rather than doing something wrong silently, so
you'll find out immediately rather than discover it three epochs in, but
the fix is writing the kernel, not finding the right incantation.

### The cuBLASLt matmul path (CUDA, #85)

CUDA's `MatMulExact` dispatch tries cuBLASLt first for plain, epilogue-free
requests and falls back to the NVRTC tiled kernel on any non-fit. The path
forces `CUBLAS_COMPUTE_32F` rather than cudarc's safe wrapper, which would
silently pick TF32 against the `MathMode::Precise` capability claim. A
request fits only when both operands are `f32`, on the same device, rank 2,
contiguous at offset zero (`4023ceed`, `cuda/ops/cublaslt.rs`). The coarse
`MatMul` capability row moved `F32_ONLY` → `FLOAT_DTYPES` to match the exact
row #90 widened, so a coarse row no longer under-advertises work the exact
row already claims.

Epilogue requests (bias plus optional ReLU/GELU) fail closed: there is no
configuration in which an epilogue is requested and a kernel that omits it
is allowed to run. The epilogue surface is therefore strictly narrower than
the plain-product surface.

**Verification status:** compile-gated. Six host-side dispatch-policy unit
tests pass without hardware; six value tests are `#[ignore = "requires CUDA
hardware"]` in `cuda/backend/tests.rs`. This box has no CUDA device, so the
cuBLASLt numerics are compile-verified only — treat them as declared
capability awaiting the #82 runner, not as runtime-proven results.

### WGPU runtime evidence: Batches A–C, attention, cross-entropy (#91)

WGPU's first #91 gap-closure batch landed 24 operations, every one executed
on the local Vulkan adapter against a CPU-twin reference: `sign`/`floor`/
`ceil`/`round`, the full trig and inverse-trig families including
hyperbolics, `erf`/`rsqrt`/`log2`/`log10`/`trunc`/`frac`, scalar forms of
`add`/`sub`/`mul`/`div`, `powf`/`clamp`, and `atan2`/`fmod`/`remainder`,
plus backward kernels for `sin` (cosine gradient) and `clamp` (masked
gradient) (`c9c0d035`, `crates/incin-backends/tests/wgpu_pointwise_gap.rs`).
Batch B (`631e5a1d`, `wgpu_batch_b.rs`) added structural, normalization
(`layer_norm`/`group_norm`/`batch_norm`), loss (`mse`/`l1`), and matmul rows;
Batch C (`wgpu_batch_c.rs`) added `instance_norm`, dropout, BCE, SDPA, and
structural repeat/pad/chunk/split. Attention runs end-to-end against a CPU
twin — non-causal, causal, grouped-query, and rotary — plus a training smoke
that `backward`s to finite, non-zero projection gradients
(`0623e762`, `wgpu_attention.rs`). Cross-entropy records and backpropagates
against the hand-computed `(softmax − onehot) / batch` reference
(`wgpu_cross_entropy.rs`). The whole `wgpu_*.rs` suite is 95+ test functions
and runs green under every-PR CI's lavapipe software-adapter job.

This is the accelerator surface with real execution evidence in the tree:
CPU (complete catalog) and the WGPU software adapter (lavapipe/Vulkan) both
compute and are compared against host references. Skipped rows (the
comparison and logical families) are blocked on a multi-dtype advertising
policy, not only on kernels — WGPU storage admits `u8`/`u32`/`i64`/`bool`,
but `native_precision` only claims `f32` compute, so a comparison the
catalog types as `bool` cannot be honestly registered until the backend
grows a bool compute path.

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
