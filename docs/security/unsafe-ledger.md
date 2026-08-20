# Unsafe code ledger

This ledger records the production Rust files that currently contain `unsafe`.
The enclosing function or block remains the authoritative place for the exact
preconditions and has a nearby `SAFETY` comment where the operation needs one.
The checker below prevents new unsafe-bearing files from bypassing review.

The architecture has three intentional unsafe families:

1. CPU kernels use architecture intrinsics and spare-capacity slice writes.
2. CUDA and NCCL adapters cross the cudarc FFI and validated device-buffer
   boundaries.
3. Core tensor construction and process integrations use narrowly scoped raw
   slices or platform APIs.

| Area | Reviewed files |
| --- | --- |
| CPU storage and kernels | `crates/incin-backends/src/cpu/storage.rs`, `crates/incin-backends/src/cpu/ops/elementwise_kernel/avx2.rs`, `crates/incin-backends/src/cpu/ops/elementwise_kernel/neon.rs`, `crates/incin-backends/src/cpu/ops/elementwise_kernel/scalar.rs`, `crates/incin-backends/src/cpu/ops/elementwise_kernel/wasm.rs`, `crates/incin-backends/src/cpu/ops/matmul.rs`, `crates/incin-backends/src/cpu/ops/quant.rs`, `crates/incin-backends/src/simd.rs` |
| CUDA kernels | `crates/incin-backends/src/cuda/ops/compare.rs`, `crates/incin-backends/src/cuda/ops/conv.rs`, `crates/incin-backends/src/cuda/ops/elementwise.rs`, `crates/incin-backends/src/cuda/ops/logical.rs`, `crates/incin-backends/src/cuda/ops/matmul.rs`, `crates/incin-backends/src/cuda/ops/norm.rs`, `crates/incin-backends/src/cuda/ops/pool.rs`, `crates/incin-backends/src/cuda/ops/reduce.rs`, `crates/incin-backends/src/cuda/ops/select.rs`, `crates/incin-backends/src/cuda/ops/shape.rs` |
| Distributed and tuning FFI | `crates/incin-backends/src/dist/nccl.rs`, `crates/incin-backends/src/tuning/identity.rs` |
| Core tensor representation | `crates/incin-core/src/tensor/ops/manipulation.rs` |
| Process integrations | `crates/incin-data/src/hub.rs`, `crates/incin-lsp/src/config.rs`, `crates/incin-telemetry/src/emitter.rs`, `crates/incin-telemetry/src/run_dir.rs` |

`tools/check-unsafe-ledger.py` compares this inventory with the source tree.
It is intentionally file-level so line numbers do not become stale when code
is reformatted; individual safety invariants belong beside each unsafe use.
