# Unsafe code ledger and threat model

This is the release audit for production unsafe code. `tools/check-unsafe-ledger.py`
does not count the word `unsafe`: it excludes `#[cfg(test)]` and `#[test]`
items, locates each production `unsafe { ... }` block, and rejects a source
file not listed below. A block carries an adjacent `SAFETY:` explanation when
its proof is site-specific; repeated target-intrinsic blocks inherit the
precise family proof below. The CPU, CUDA, and NCCL Clippy gates enforce
adjacent comments for their compiled blocks. The source checker makes every
remaining target-specific block visible and assigned to a tested invariant
family rather than pretending hardware-disabled code was executed locally.

The Rust lints `unsafe_op_in_unsafe_fn`, `clippy::undocumented_unsafe_blocks`,
and `clippy::missing_safety_doc` are deny-level workspace policy. An unsafe
function cannot silently acquire operations through its ambient contract; a
public unsafe function must document its caller contract.

| Family | Local invariant and native/FFI boundary | Source sites | Evidence |
| --- | --- | --- | --- |
| CPU byte views | The returned byte slice borrows the live allocation and uses its exact element count and byte width. | `crates/incin-backends/src/cpu/storage.rs` | CPU storage tests; Miri and ASan CPU gates. |
| CPU SIMD kernels | Runtime/compile-time target-feature guards hold; every vector load is in bounds; spare-capacity output ranges are disjoint and fully initialized before `set_len`. | `crates/incin-backends/src/cpu/ops/elementwise_kernel/avx2.rs`, `crates/incin-backends/src/cpu/ops/elementwise_kernel/neon.rs`, `crates/incin-backends/src/cpu/ops/elementwise_kernel/scalar.rs`, `crates/incin-backends/src/cpu/ops/elementwise_kernel/wasm.rs`, `crates/incin-backends/src/simd.rs` | CPU kernel tests; AVX2 ASan; focused TSan initialization stress. |
| CPU matrix and quantized kernels | Shape validation proves row offsets, block counts, and output capacity before intrinsic loads/stores. | `crates/incin-backends/src/cpu/ops/matmul/gemm.rs`, `crates/incin-backends/src/cpu/ops/quant.rs` | CPU matmul/quant tests; AVX2 ASan. |
| CUDA launch adaptation | Checked shape arithmetic, dtype, allocation length, and fresh-output ownership precede every CudaSlice reinterpretation and kernel launch. FFI errors are converted to typed backend errors. | `crates/incin-backends/src/codegen/jit.rs`, `crates/incin-backends/src/cuda/ops/compare.rs`, `crates/incin-backends/src/cuda/ops/conv.rs`, `crates/incin-backends/src/cuda/ops/elementwise.rs`, `crates/incin-backends/src/cuda/ops/logical.rs`, `crates/incin-backends/src/cuda/ops/matmul.rs`, `crates/incin-backends/src/cuda/ops/norm.rs`, `crates/incin-backends/src/cuda/ops/pool.rs`, `crates/incin-backends/src/cuda/ops/quant.rs`, `crates/incin-backends/src/cuda/ops/reduce.rs`, `crates/incin-backends/src/cuda/ops/select.rs`, `crates/incin-backends/src/cuda/ops/shape.rs` | CUDA compile and runtime matrix where a runner is available; checked-shape tests. |
| NCCL transport | Dtype selection and element counts are validated before the typed CudaSlice views; NCCL failures become `CollectiveError`. | `crates/incin-backends/src/dist/nccl/transport.rs` | Distributed transport tests and CUDA/NCCL compile gate. |
| CUDA version queries | Pointers passed to C ABI version calls reference initialized local integer storage for the duration of the call; error return codes and negative values are rejected. | `crates/incin-backends/src/tuning/identity/cuda.rs` | CUDA identity tests and compile gate. |
| Host extraction | Requested Rust type, dtype encoding, byte length, alignment-safe unaligned read, and boolean bit pattern are validated before reinterpretation. | `crates/incin-core/src/tensor/ops/manipulation/interop.rs` | Core interop negative tests; Miri. |

## Non-production unsafe

`crates/incin-data/src/hub.rs`, `crates/incin-lsp/src/config.rs`,
`crates/incin-telemetry/src/emitter.rs`, and
`crates/incin-telemetry/src/run_dir.rs` contain only test-scoped environment
mutation. They remain intentionally excluded from the production ledger; their
adjacent comments and per-crate tests document process-global serialization.

## Threat model and residual risk

The audit boundary is safe public Incin code plus malformed model/data inputs,
backend responses, and ordinary process failures. User-controlled shape,
dtype, index, conversion, allocation, and artifact inputs must take typed
error paths before reaching any unsafe family. Native driver, CUDA/NCCL, SIMD,
and raw-byte boundaries are trusted only after the local invariant above is
established.

Miri covers aliasing and invalid host memory accesses in core and CPU tests;
ASan covers CPU allocation bounds and leaks (including AVX2); TSan covers the
parallel AVX2 initialization path. Hardware-specific CUDA, NCCL, NEON, and
WASM execution still requires the corresponding CI runner or maintainer
hardware; compile checks and the local invariant audit do not substitute for
that runtime coverage. See `tools/soundness.sh` for the exact dynamic gates.

Production panic, unwrap, and expect sites are classified separately in the
cfg-aware, checked per-site inventory
`audit-evidence/FND-003/production-panic-sites.json`; the checker fails on
any source or classification drift. Indexing, arithmetic, allocation,
conversion, and user-input paths are bound by the typed failure contract in
`docs/ERROR_CONTRACT.md` and the accompanying FND-003 narrative.
