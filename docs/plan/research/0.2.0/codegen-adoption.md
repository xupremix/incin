codegen: two kernel emitters, one of them has no callers

Finding: `crates/incin-backends/src/` contains two independent kernel source
emitters, and only one is reachable from an executor.

The live one is `src/kernel/` (seven files). It renders CUDA C from string
templates through `render_cuda_unary`, `render_cuda_binary`,
`render_cuda_unary_packed`, `render_cuda_reduction` and
`render_cuda_normalization`, and it is called from `cuda/ops/elementwise.rs`,
`cuda/ops/reduce.rs` and `cuda/ops/norm.rs`. Those sources go to
`CpuCudaDispatcher::compile_and_load_kernel` in `cuda/gpu.rs`, which runs
`compile_ptx_with_cuda_includes` (NVRTC) and caches the loaded module.

The other is `src/codegen/` (24 files, roughly 280 KB). Nothing under
`crates/*/src/` imports it. Its only consumers are `tests/codegen_pointwise.rs`
and `tests/codegen_ir_pipeline.rs`, which assert on the text it emits. It holds
an expression IR with symbolic differentiation (`ir.rs`, 38 KB,
`KernelDefinition::render_forward_cuda` and `render_backward_cuda`), a public
custom-operation DSL (`dsl.rs`, `define_unary_custom_op` and friends), a working
NVRTC-backed `CudaJitKernel` that compiles and launches, tensor-core MMA layouts
(`mma.rs`), a CUTLASS-shaped GEMM emitter with epilogue activations
(`sota_gemm.rs`, `gemm.rs`, `quant_gemm.rs`), a Triton-shaped scheduler
(`scheduler.rs`: `BlockTensorPtr`, `MemorySpace`, `LoopScheduleKind`), an
autotune space keyed on `GpuArchProfile`, and emitters for attention, RoPE, MoE
gating, normalization, scan, reductions, strided indexing, vectorization, fused
epilogues and fused optimizers.

Two claims in the tree are false as written. `PRF-007` step 5 is checked and
reads "Connect WGPU and Metal backend dispatchers to consume generated shader
sources instead of hand-written redundant `.wgsl`/`.metal` pointwise files"; no
`render_wgsl` or `render_msl` call exists anywhere under `src/`, there are 12
`include_str!` shader loads in the WGPU and Metal backends, and 30 hand-written
`.wgsl`, `.metal` and `.cu` files remain. Separately, `jit.rs`'s header claims
"zero-overhead execution", while `CpuJitKernel::eval_f32` is a per-element
tree-walking interpreter over `f64`.

The important structural point is that adoption is not an infrastructure
project. Both emitters end at the same NVRTC dispatcher, so routing an operation
through `codegen` instead of `kernel` is a swap of the source producer. The
blocker is that nobody has done it once, so the IR has never been held to a
numerical result.

Recommendation: adopt exactly one path end to end before deciding anything else,
and pick it for verifiability rather than for value. CUDA pointwise is the
natural target on a machine with a GPU: replace the `render_cuda_unary` call in
`cuda/ops/elementwise.rs` with an `ir.rs` expression lowered through
`KernelDefinition::render_forward_cuda`, keep the existing tests as the oracle,
and delete the duplicated template on success. If that lands, `codegen`'s
symbolic backward becomes the argument for the rest of it. If it does not, the
honest move is to delete `codegen/` outright: it is 280 KB of `pub` surface in a
published crate with no execution behind it, and pre-1.0 removal is free.

Correct `PRF-007` step 5 either way. A checked step that the tree contradicts is
the same defect class as the stale MTL status claim in #92.

Risk: `dsl.rs` is public API and may be the intended user-facing custom-operation
story; check `docs/book/src/custom_operations.md` before removing anything, since
that chapter does not currently mention `codegen` at all.

Unblocks: nothing today, which is the finding.
