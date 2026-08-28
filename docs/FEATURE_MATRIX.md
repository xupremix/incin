# Supported feature contract

Feature declarations and their release-support metadata live together in each
crate manifest. `cargo xtask docs --check` rejects missing metadata and keeps
this inventory, the book, and the README synchronized.

The executable matrix is [`tools/feature-matrix.sh`](../tools/feature-matrix.sh).
Its stable tier is compiled with the workspace MSRV before preview and hardware
rows are considered. Hardware rows prove compilation only unless their named
hardware workflow runs them.

<!-- BEGIN GENERATED: feature-inventory -->
| Crate | Feature | Tier | Owner | Prerequisites | Default | Enables | Incompatibilities | Hardware | Purpose |
|---|---|---|---|---|:--:|---|---|---|---|
| `incin` | `std` | stable | facade | none | yes | `incin-core/std`, `incin-macros/std`, `incin-backends/std` | none | none | Enables standard-library functionality, serialization, and filesystem APIs. |
| `incin` | `nightly` | preview | facade | none | no | `incin-core/nightly`, `incin-macros/nightly` | none | nightly | Enables nightly-only APIs in the core and macro crates. |
| `incin` | `cpu` | stable | facade | `std` | yes | `std`, `incin-backends/cpu` | none | none | Enables the built-in CPU backend. This is the only default backend. |
| `incin` | `cpu-blas` | stable | facade | `std`, `cpu` | no | `std`, `cpu`, `incin-backends/cpu-blas` | none | none | Hands large f32 CPU matmuls to a blocked GEMM. The CPU backend is complete without it; see incin-backends for what it does and does not change. |
| `incin` | `cuda` | preview | facade | `std` | no | `std`, `incin-core/cuda`, `incin-backends/cuda` | none | cuda | Preview: the native CUDA backend, covering the subset in docs/capabilities.md. Never enabled implicitly. |
| `incin` | `wgpu` | preview | facade | `std` | no | `std`, `incin-backends/wgpu` | none | wgpu | Preview: the cross-platform WGPU backend, covering the subset in docs/capabilities.md. Never enabled implicitly. |
| `incin` | `metal` | preview | facade | `std` | no | `std`, `incin-backends/metal` | none | apple-silicon | Preview: the native Metal backend for Apple Silicon. Its executors are stubs pending MTL-002/003; see docs/capabilities.md. Never enabled implicitly. |
| `incin` | `metal-mps` | preview | facade | `metal` | no | `metal`, `incin-backends/metal-mps` | none | apple-silicon | Enables MPS and MPSGraph structured primitives for Apple Silicon. |
| `incin` | `external-candle` | stable | facade | `std` | no | `std`, `incin-backends/external-candle` | none | none | Enables the external Candle backend at `incin::external::candle`. |
| `incin` | `autotune` | preview | facade | `cuda` | no | `cuda`, `incin-backends/autotune` | none | cuda | Enables CUDA launch autotuning. |
| `incin` | `train` | preview | facade | `std` | no | `std` | none | none | Enables the preview trainer at `incin::experimental::training`. The interface may change without a migration path. |
| `incin` | `distributed` | preview | facade | none | no | `incin-core/distributed`, `incin-macros/distributed` | none | none | Preview: typed meshes, static/runtime tensor placements, and distributed lowering proofs. This is a planning and validation layer; there is no distributed execution path. Transports remain separate opt-in backend features. |
| `incin` | `distributed-reference` | preview | facade | `distributed` | no | `distributed`, `incin-backends/distributed-reference` | none | none | Enables the deterministic in-process collective transport used by conformance tests and local distributed-plan development. |
| `incin` | `distributed-nccl` | preview | facade | `distributed`, `cuda` | no | `distributed`, `cuda`, `incin-backends/distributed-nccl` | none | multi-host-cuda | Two-host process-per-rank CUDA transport and its TCP bootstrap. |
| `incin` | `telemetry` | stable | facade | `std` | no | `std`, `incin-backends/telemetry`, `dep:incin-telemetry` | none | none | Enables backend telemetry hooks. `cargo incin doctor` also reports the run directory under this feature, which is why the dependency is direct here and not only through incin-backends. |
| `incin` | `test-utils` | test-only | facade | `std`, `cpu` | no | `std`, `cpu`, `incin-backends/test-utils` | none | none | Deterministic fault-injection hooks for tests. No stand-in backend: a test that needs a backend uses a real one. |
| `incin` | `backend-authoring` | stable | facade | none | no | none | none | none | Extension contracts for backend authors. |
| `incin` | `data-hub` | stable | facade | `std` | no | `std`, `incin-data/hub` | none | network | The Hugging Face Hub client at `incin::hub`. Off by default because it brings an async runtime and a second TLS stack into the dependency graph for an API most training code never calls; dataset downloading does not need it. |
| `incin` | `compiled` | preview | facade | `std`, `cpu` | no | `std`, `cpu`, `incin-core/compiled`, `incin-backends/compiled` | none | none | Preview-only CPU reference evaluator and plan-inspection types under `incin::experimental::compiled`. No stable API, deployment format, or portable artifact ABI is promised. |
| `incin` | `hardware-tests` | hardware-only | facade | `distributed-nccl` | no | `distributed-nccl` | none | multi-host-cuda | Opt-in only: ignored multi-host CUDA runtime fixtures require actual hardware and are not part of compile-only feature coverage. |
| `incin` | `update-check` | stable | facade | `std` | no | `std`, `dep:ureq` | none | none | Lets `cargo incin doctor --check-updates` ask crates.io whether a newer incin exists. Off by default: it is the only feature that can reach the network. |
| `incin-backends` | `std` | stable | backends | none | yes | `incin-core/std`, `anyhow/std`, `rand/std`, `rand/std_rng`, `rand/thread_rng`, `rand_distr/std`, `half/std` | none | none | Enables APIs that require the Rust standard library. |
| `incin-backends` | `cpu` | stable | backends | `std` | yes | `std` | none | none | Enables Incin's built-in pure-Rust CPU backend. Declares `std` for the same reason `cuda` and `wgpu` do: the kernels reach for `Vec` and `Box` through the std prelude and the autograd tape is a `thread_local!`, so `cpu` without `std` was a combination the manifest offered and the crate never compiled under. |
| `incin-backends` | `compiled` | preview | backends | `std`, `cpu` | no | `std`, `cpu`, `incin-core/compiled` | none | none | Enables execution of compiled graphs through the canonical CPU descriptors. |
| `incin-backends` | `cpu-blas` | stable | backends | `std`, `cpu` | no | `std`, `cpu`, `dep:matrixmultiply` | none | none | Hands large f32 CPU matmuls to a blocked, register-tiled GEMM. Off by default: the pure-Rust kernels in cpu/ops/matmul.rs stay complete without it, and enabling it changes only floating-point accumulation order. |
| `incin-backends` | `cuda` | preview | backends | `std` | no | `std`, `dep:cudarc`, `incin-core/cuda` | none | cuda | Enables Incin's native CUDA backend through cudarc. |
| `incin-backends` | `cuda-vendor` | preview | backends | `cuda` | no | `cuda` | none | cuda | Gates cuBLAS / cuDNN vendor library call sites. |
| `incin-backends` | `wgpu` | preview | backends | `std` | no | `std`, `dep:wgpu`, `dep:pollster`, `incin-core/wgpu` | none | wgpu | Enables Incin's cross-platform GPU backend through wgpu. |
| `incin-backends` | `metal` | preview | backends | `std` | no | `std`, `incin-core/metal` | none | apple-silicon | Enables Incin's native Metal backend for Apple Silicon. |
| `incin-backends` | `metal-mps` | preview | backends | `metal` | no | `metal` | none | apple-silicon | Enables MPS and MPSGraph structured primitives for Apple Silicon. |
| `incin-backends` | `autotune` | preview | backends | `cuda` | no | `cuda` | none | cuda | Benchmarks legal CUDA launch configurations and caches the selected candidate. |
| `incin-backends` | `external-candle` | stable | backends | `std` | no | `std`, `dep:candle-core`, `dep:candle-nn` | none | none | Enables the external Candle backend at `incin_backends::external::candle`. |
| `incin-backends` | `telemetry` | stable | backends | `std` | no | `std`, `dep:incin-telemetry` | none | none | Connects backend execution and autograd events to `incin-telemetry`. |
| `incin-backends` | `distributed` | preview | backends | `std` | no | `std`, `incin-core/distributed` | none | none | Shared distributed contracts and topology types. Transport implementations remain opt-in so the normal CPU build does not gain communication machinery. |
| `incin-backends` | `distributed-reference` | preview | backends | `distributed` | no | `distributed` | none | none | Deterministic in-process CPU reference collectives used for conformance, adjoint, and planner tests. |
| `incin-backends` | `distributed-nccl` | preview | backends | `distributed`, `cuda` | no | `distributed`, `cuda`, `autotune`, `cudarc/nccl` | none | multi-host-cuda | Real process-per-rank CUDA collectives. cudarc loads NCCL at runtime. |
| `incin-backends` | `test-utils` | test-only | backends | none | no | none | none | none | Deterministic fault-injection hooks used only by integration tests. This is deliberately opt-in and is never part of a product build. |
| `incin-core` | `std` | stable | core | none | yes | `thiserror/std`, `half/std`, `anyhow/std`, `serde/std`, `prost/std`, `bytes/std`, `dep:safetensors`, `dep:serde_json`, `dep:postcard` | none | none | Enables standard-library error, filesystem, and model serialization support. |
| `incin-core` | `nightly` | preview | core | none | no | none | none | nightly | Enables nightly-only experiments. Empty on stable builds by design. |
| `incin-core` | `paranoid-validation` | test-only | core | none | no | none | none | none | Recomputes a descriptor's own invariants inside executors (EXE-002). A debug and test aid only: the contract is that a `Validated<O>` was already checked by the lowering rule that minted it, so a release build must never need this. |
| `incin-core` | `distributed` | preview | core | none | no | `incin-macros/distributed` | none | none | Typed logical meshes, physical binding, placement proofs, and placement-aware tensors (DST-001 through DST-004). Preview-tier and therefore not default; it adds no dependency, and reads no hardware outside `DeviceMesh::bind` through a `TopologyProbe` this crate never implements. |
| `incin-core` | `cuda` | preview | core | none | no | none | none | cuda | Enables CUDA device marker metadata used by CUDA-capable backend crates. |
| `incin-core` | `wgpu` | preview | core | none | no | none | none | wgpu | Enables WGPU device marker metadata used by WGPU-capable backend crates. |
| `incin-core` | `metal` | preview | core | none | no | none | none | apple-silicon | Enables Metal device marker metadata used by Metal-capable backend crates. |
| `incin-core` | `compiled` | preview | core | `serde_json` | no | `serde_json` | none | none | Enables preview artifact and typed execution descriptors for the CPU compiled evaluator. |
| `incin-core` | `postcard` | internal | core | `std` | no | `std`, `dep:postcard` | none | none | Serialization format switches are intentionally lower-crate internal. They remain explicit so their support classification and prerequisites are checked. Enables the compact postcard checkpoint codec for internal serialization paths. |
| `incin-core` | `safetensors` | internal | core | `std` | no | `std`, `dep:safetensors` | none | none | Enables SafeTensors checkpoint input and output in the internal I/O layer. |
| `incin-core` | `serde_json` | internal | core | `std` | no | `std`, `dep:serde_json` | none | none | Enables JSON metadata and diagnostic serialization in the internal I/O layer. |
| `incin-macros` | `std` | stable | macros | none | yes | none | none | none | Standard-library support for compile-time model and file processing. |
| `incin-macros` | `nightly` | preview | macros | none | no | none | none | nightly | Enables nightly-only macro experiments. Empty on stable by design. |
| `incin-macros` | `distributed` | preview | macros | none | no | none | none | none | Enables distributed macro tests and integrations. |
| `incin-diagnostics` | `std` | stable | diagnostics | none | yes | none | none | none | Standard-library support for filesystem-backed diagnostic expansion. |
| `incin-data` | `download` | stable | data | none | yes | `dep:ureq`, `dep:flate2` | none | network | HTTP download and gzip extraction, used by the bundled dataset fetchers. |
| `incin-data` | `hub` | stable | data | none | no | `dep:hf-hub` | none | network | The Hugging Face Hub client. Not a default: it brings an async runtime and a second TLS stack into the dependency graph of every crate that depends on the facade, for an API most training code never calls. |
<!-- END GENERATED: feature-inventory -->

`stable` is the 0.1 compatibility promise. `preview` may change without a
migration path. `test-only` and `hardware-only` are not product configurations.
`internal` names lower-crate controls that are intentionally absent from the
facade. The hardware column states a requirement, not evidence that it ran.

There are no rejected stable feature combinations in 0.1; every effective
stable combination in every contracted package, including bare no-default
builds, is compiled at the MSRV with metadata-derived `cargo-hack` commands.
A metadata-derived additive accelerator row covers the combined preview
accelerator surface. A future
rejected pair must be declared symmetrically in the manifest metadata, covered
by a focused compile-fail fixture, and excluded from the powerset with
`--mutually-exclusive-features`. It must not be skipped informally.
