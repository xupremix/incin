# Facade and workspace feature flags

The facade feature inventory is checked against `crates/incin/Cargo.toml`.
The default is `["std", "cpu"]`  -  a
standard-library CPU build.

## Core

| Feature | Default | What it enables |
|---|:--:|---|
| `std` | ✅ | Standard library, serialization, filesystem APIs. Turning it off gives a `no_std` build  -  see below. |
| `cpu` | ✅ | The CPU backend, `DefaultBackend`, `DefaultDevice`. The only backend enabled by default. |
| `nightly` | | Nightly-only APIs in the core and macro crates. |

## Backends

| Feature | What it enables |
|---|---|
| `cuda` | The native CUDA backend. Never enabled implicitly. |
| `wgpu` | The cross-platform WGPU backend. Never enabled implicitly. |
| `metal` | The native Metal backend for Apple Silicon. Never enabled implicitly. |
| `metal-mps` | MPS and MPSGraph structured primitives for Apple Silicon. |
| `cpu-blas` | Routes large f32 CPU matmuls through a blocked GEMM. The CPU backend is complete without it. |
| `external-candle` | The third-party Candle adapter at `incin::external::candle`. |

Enabling an accelerator never silently changes behaviour: CPU stays the
default whenever it is available, and an accelerator-only build gets the one
enabled device family. See [Backends](./backends.md) for what each backend can
actually *run* today  -  the answer is much narrower than this table implies.

**Every backend feature implies `std`.** A bare `incin-backends` with no
features is the one configuration that is genuinely `no_std`.

## Opt-in surfaces

| Feature | What it enables |
|---|---|
| `backend-authoring` | Extension contracts for backends and custom operations. See [Backend authoring](./backend_authoring.md). |
| `train` | The preview `Trainer` at `incin::experimental::training`. The interface may change without a migration path. |
| `telemetry` | Backend telemetry hooks; `cargo incin doctor` also reports the run directory under this feature. |
| `autotune` | CUDA launch autotuning. Implies `cuda`. |
| `compiled` | Curated preview types for compiled plans and guards. Add `cpu` for the executable CPU lowering. |
| `test-utils` | Test-only backends (`DummyBackend`) and test utilities. |
| `hardware-tests` | Explicit hardware integration tests. These are never enabled by default. |

## Distributed

| Feature | What it enables |
|---|---|
| `distributed` | Typed meshes, static/runtime placements, distributed lowering proofs. A *planning* surface  -  see [Experimental](./experimental.md). |
| `distributed-reference` | The deterministic in-process collective transport used by conformance tests and local plan development. |
| `distributed-nccl` | Two-host process-per-rank CUDA transport and its TCP bootstrap. Implies `cuda`. |

Transports are deliberately separate opt-ins from `distributed` itself:
declaring a mesh and actually moving bytes between hosts are different
commitments.

## Internal feature gates

These gates belong to lower-level crates rather than the normal `incin`
facade. They are listed here so the documentation covers the complete
workspace feature vocabulary while making their intended audience explicit.

| Feature | Crate | What it enables |
|---|---|---|
| `paranoid-validation` | `incin-core` | Rechecks sealed execution proofs in validation-oriented builds; it is not required by release execution. |
| `cuda-vendor` | `incin-backends` | Enables the CUDA vendor-library selection layer on top of `cuda`; it does not claim vendor-kernel runtime parity by itself. |

## `no_std`

Turning off `std` gives a `no_std` build of `incin-core` (it needs `alloc`).
What changes:

- **Scoped policies become their defaults.** There is no thread-local, so
  `no_grad`'s scope cannot be installed and `GradMode::current()` answers
  `Enabled`. That is the true state rather than a weakened guarantee: nothing
  in a `no_std` build can express a disabled scope, and every tape in the
  workspace lives in a backend that requires `std`.
- **No filesystem or serialization surfaces**  -  checkpointing, the doctor
  report, and the ONNX/safetensors paths are all `std`.

The workspace CI builds `incin-core --no-default-features` and
`incin-backends --no-default-features` on every push precisely because this
configuration is easy to break without noticing: it is the only one where a
`Vec` needs an explicit `alloc` import, and the only one where a
`#[cfg(feature = "std")]`-gated helper going missing is a compile error rather
than an invisible non-event.

## Checking what a build actually has

```rust,no_run
// The doctor report lists compiled-in backend features, detected devices,
// and cache state for the running build. `run` takes the CLI's argument
// list and returns the rendered report plus an exit code.
let (report, exit_code) = incin::doctor::run(&[]);
println!("{report}");
```

Or from the command line, `cargo incin doctor`. The report lives in the
library rather than only in the binary so an integration test can link it.
