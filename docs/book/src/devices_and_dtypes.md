# Devices, dtypes, and checking what a backend supports

Everything in this chapter runs on the default CPU build. Device movement and
support queries compile everywhere; actually *executing* on an accelerator
requires its preview feature, and the [backend chapter](./backends.md) records
what each one covers today.

## Devices

A tensor's device is part of its type through the backend parameter. On the
default build every tensor lives on `Cpu`, and `DefaultBackend` is an alias for
the CPU backend with the `cpu` feature enabled.

Moving a tensor between two compiled devices is explicit:

```rust,ignore
let on_cpu: Tensor<s![2, 2], CpuBackendImpl> = Tensor::zeros(())?;
let on_gpu = on_cpu.to_device::<CudaBackendImpl>()?;
```

Two rules make this safe:

- **No silent transfers.** If the destination backend is not compiled in, the
  move is a type error, not a hidden copy onto CPU.
- **Round trips are honest.** Moving back produces storage on the original
  device again; nothing is cached behind your back.

The trainer applies the same rule to whole models: ask it to train on a device
family whose Cargo feature is off and it refuses with `NotCompiledIn`; ask for
a device that exists but is not present (say `cuda:1` on a single-GPU box) and
it refuses with `DeviceUnavailable`. A training run that quietly fell back to
CPU would be worse than an error.

## Switching dtypes

Creation infers dtype from how you build the tensor (`f32` slices make f32
tensors). Converting an existing tensor uses `to_dtype`:

```rust,ignore
use incin::prelude::*;

let f: Tensor<s![3], DefaultBackend> =
    Tensor::from_slice(&[1.0, 2.0, 3.0], ())?;
let i = f.to_dtype::<i64>()?;          // truncates toward zero, like Rust's `as`
let back = i.to_dtype::<f32>()?;
```

Integer conversions are deterministic Rust `as` casts: fractional parts
truncate and out-of-range values saturate. The *checked* conversion paths -
scalar readback, embedding indices, integer fills - are different functions
with exact policy semantics (NaN/infinity/fraction rejection); see
[`docs/ERROR_CONTRACT.md`](https://github.com/xupremix/incin/blob/master/docs/ERROR_CONTRACT.md)
for where each policy applies.

## Asking a backend what it supports

Capabilities are data, not folklore. Every backend ships a static rule table,
and the registry answers structured queries against it:

```rust,ignore
use incin_backends::capability::{registry, support};

fn registry_report() -> Vec<incin_backends::capability::BackendCoverageRow> {
    incin_backends::capability::coverage_report()
}
use incin_core::exec::capability::CapabilityQuery;
use incin_core::shapes::error::OperationKind;
use incin_core::prelude::{DTypeDescriptor, DTypeId};

// Does CUDA support f32 matmul at rank 4?
let level = support(
    incin_core::tensor::device::DeviceKind::Cuda,
    &CapabilityQuery::DType {
        operation: OperationKind::MatMulExact,
        dtype: DTypeDescriptor::builtin(DTypeId::F32),
    },
);
println!("{level:?}"); // Native | Composed | Fallback | Unsupported(..)

// Or walk the whole table (one row per operation, per backend):
for row in registry_report() {
    println!("{row:?}");
}
```

`SupportLevel::Native` means a real kernel, `Composed` means the backend builds
it from supported pieces, and `Unsupported(reason)` names exactly why not -
including which compile-time feature is missing. The same registrations drive:

- dispatch-time refusals (an executor that does not exist is a type error);
- [`cargo incin doctor`](#checking-from-the-cli), which probes your machine;
- the generated
  [capability matrix](https://github.com/xupremix/incin/blob/master/docs/capabilities.md).

## Checking from the CLI

```bash
cargo incin doctor              # human-readable report
cargo incin doctor --json       # machine-readable, schema-versioned
```

The report lists toolchain, active features, per-backend availability, caches,
and per-operation capability rows. CI runs the same probe, so a support claim
in documentation can be checked rather than believed.
