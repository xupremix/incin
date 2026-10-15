# CUDA: toward hardware-proven training

CUDA advertises 169 of the catalog's 170 backend-executable operations —
the widest preview surface in the tree — but its value tests are
`#[ignore]`d in CI pending `HARDWARE_CUDA_RUNNER`, and the weekly hardware
matrix still skips its CUDA job for the same reason. This chapter covers
both halves of that sentence: the kernel work that makes the surface real
(#85, #90), and the evidence story that keeps it honest (#82, #83).

## Fused GEMM epilogues instead of two launches

A linear layer followed by a bias-add used to be two dispatches. The
cuBLASLt route (`crates/incin-backends/src/cuda/ops/cublaslt.rs`,
`try_launch_matmul` and `try_launch_batched_matmul`) folds the epilogue
into the GEMM: one tensor-core launch computes `X @ W + b` in `fp16`/`bf16`
instead of a GEMM plus a broadcast add. The capability table records the
preference explicitly
(`crates/incin-backends/src/capability/declarations.rs`): the fused row is
chosen where legal, the native fallback stays where it is not.

Vendor versus native is measured, not assumed. The GEMM selector
(`gemm_select.rs`) times both paths and caches the winner per shape, and
the fused-SDPA dispatch covers unmasked batches the same way — one
descriptor, one launch, one tape entry. If you have an NVIDIA device, the
proof is one command:

```sh
cargo test -p incin-backends --features cuda -- --ignored cuda_matmul
```

Without a device those suites compile-check only, which is exactly what CI
does with them.

## Dtype-parametric kernels

Comparisons, `where_cond`/`masked_fill`, and the matmul family are written
once against the float dtypes (`FLOAT_DTYPES`) rather than once per dtype
(#90). The conformance fixtures pin the tolerance cost of that choice:
`f16`/`bf16` probe coverage plus matmul tolerance exceptions live beside
the kernels, and the semantics document records reduced-precision
accumulation as a contract, not a footnote.

The matmul tolerance story has its own oracle: an independent `f64` value
oracle for CPU conformance (#83) means a dtype-parametric CUDA kernel is
checked against a higher-precision CPU twin, not against itself.

## The evidence boundary

Three levels, and every CUDA claim in this book names one:

1. **Runtime-verified on real hardware, outside CI.** Matmul parity, fused
   attention including `f16`/`bf16`, quantize STE, and the two-rank NCCL
   harness (including a heterogeneous pair) all run green on NVIDIA
   devices. These are branch evidence with dated logs, not matrix
   artifacts.
2. **Compile-gated in CI.** Every-PR CI compiles the CUDA suites without a
   device. A green pipeline means the kernels type-check and the
   capability rows are consistent — nothing more.
3. **Scheduled hardware matrix, currently skipped.** The weekly
   `hardware.yml` CUDA job resolves its runner from `HARDWARE_CUDA_RUNNER`,
   which is unset, so it reports *skipped* rather than passing silently
   (#82). Registering a runner and setting the variable is
   repository-settings work described in `CONTRIBUTING.md`; until then,
   "CUDA works" means levels 1 and 2.

The remaining kernel work is fine-grained block-sparse causal skipping and
tensor-core tiling for attention (#104) — held up by a real decision, not
by effort: CC 7.5 tensor cores have no `f64` path, and the current parity
guarantee rests on the `f64` intermediate. That tradeoff is documented in
[What's not finished yet](../#/whats_not_finished) and stays open here.
