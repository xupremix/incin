# Supported feature contract

The general compile contract is [`tools/feature-matrix.sh`](../tools/feature-matrix.sh).
It contains 32 explicit rows: 4 core rows, 13 backend rows, and 15 facade
rows. `cargo xtask feature-matrix`, CI, and `tools/ci-local.sh` all execute
that same script. The row count is checked from the executable `run_row`
declarations, not maintained as an independent claim.

The general matrix is only one part of the contract. Platform and hardware
jobs below are dedicated contract rows when compilation or execution depends
on a platform library, accelerator, or multiple processes.

## General matrix rows

| Area | Rows | Contract |
| --- | ---: | --- |
| Core | 4 | no-default, default, compiled plus distributed, and serialization opt-ins |
| Backend | 13 | CPU baseline/default, CPU BLAS, target API, WGPU, CUDA, CUDA vendor compatibility, Metal, external Candle, telemetry, reference distributed, NCCL compile, and CPU/WGPU telemetry |
| Facade | 15 | default/no-backend, target API, training, compiled, telemetry, backend authoring, WGPU, CUDA, Metal, external Candle, reference distributed, NCCL compile, and two orthogonal combinations |

All general rows are compile checks. The CUDA, CUDA vendor, WGPU, Metal, and
NCCL rows do not claim local hardware execution.

## Public feature inventory

| Crate | Feature | Validation and status |
| --- | --- | --- |
| `incin-core` | `std` | General core rows and workspace tests. |
| `incin-core` | `nightly` | Intentionally excluded from stable CI powersets because stable Rust rejects the feature gate. |
| `incin-core` | `paranoid-validation` | Internal validation aid, covered by the core powerset and not a product configuration. |
| `incin-core` | `distributed` | General compiled row plus dedicated distributed compile and tests. |
| `incin-core` | `cuda`, `wgpu`, `metal` | Marker/configuration features compiled through backend rows; runtime requires the corresponding backend job. |
| `incin-core` | `test-utils` | Test-only support, covered by core tests and powerset validation. |
| `incin-core` | `compiled` | General `core-compiled-distributed` row and compiled integration tests. |
| `incin-core` | `postcard`, `safetensors`, `serde_json` | `core-serialization` row and serialization tests. |
| `incin-backends` | `std`, `cpu` | General CPU rows and the focused CPU backend test suite. |
| `incin-backends` | `compiled`, `target-api`, `telemetry`, `external-candle` | General backend rows and focused crate tests where applicable. |
| `incin-backends` | `cpu-blas` | General compile row and dedicated CPU-BLAS test job. |
| `incin-backends` | `cuda` | General compile row and CUDA hardware job when a CUDA runner is selected. |
| `incin-backends` | `cuda-vendor` | General `backend-cuda-vendor-compile` row. This is currently a supported compatibility feature layered on `cuda`; no separate vendor-kernel runtime behavior is claimed. |
| `incin-backends` | `wgpu` | General compile row and WGPU hardware/software-adapter job. |
| `incin-backends` | `metal` | General compile row and Apple Silicon job. |
| `incin-backends` | `metal-mps` | Dedicated Apple Silicon compile/runtime job with `metal`; not a standalone general row because it requires the Metal platform. |
| `incin-backends` | `autotune` | Dedicated tuning tests, including CUDA and Metal jobs where the backend is available. |
| `incin-backends` | `distributed`, `distributed-reference` | General reference/distributed rows and dedicated reference conformance tests. |
| `incin-backends` | `distributed-nccl` | General compile row, NCCL contract tests, and multi-node hardware jobs. |
| `incin-backends` | `test-utils` | Test-only support, covered by focused tests rather than a product contract. |
| `incin` | `std`, `cpu` | General facade rows, workspace tests, and the focused facade suite. |
| `incin` | `nightly` | Stable CI exclusion because it forwards the core/macro feature gate. |
| `incin` | `target-api`, `compiled`, `telemetry`, `external-candle`, `backend-authoring` | General facade rows and focused API/compile tests. |
| `incin` | `cpu-blas`, `cuda`, `wgpu`, `metal` | Forwarding features covered by backend rows and dedicated platform jobs. |
| `incin` | `metal-mps`, `autotune` | Forwarding features covered by the Apple Silicon/CUDA tuning jobs. |
| `incin` | `train` | General `facade-training` row and trainer integration tests. |
| `incin` | `distributed`, `distributed-reference` | General rows and dedicated distributed preview tests. |
| `incin` | `distributed-nccl` | General compile row plus networked hardware jobs. |
| `incin` | `test-utils`, `hardware-tests` | Test-only/runtime-fixture controls, not product configurations. `hardware-tests` is enabled only by hardware jobs. |

The dedicated jobs are part of the feature contract, not hidden exceptions:
`.github/workflows/hardware.yml` owns WGPU, Metal/MPS, CUDA tuning, and
multi-process NCCL execution. `.github/workflows/ci.yml` owns CPU BLAS,
autotune/reference/NCCL contract tests, and the general matrix. Hardware tests
that cannot run on this host are recorded as not run in final evidence rather
than treated as compile or runtime passes.

## Powerset boundary

`cargo-hack --feature-powerset` remains enabled for the small core, macros, and
diagnostics spaces. Backend and facade Cartesian products are not supported
product contracts: many combinations are individually legal Cargo flags but
are not promised configurations, and the spaces are impractical to execute.
The explicit rows above are the maintained contract.
