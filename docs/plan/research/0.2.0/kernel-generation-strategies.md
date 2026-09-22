# Kernel-generation strategies — survey 2026-09-22

Companion to `what-to-take-from-sota.md` and `codegen-adoption.md`, informed by
an external survey (Triton, CubeCL, CUTLASS/CuTe, Thunderkittens, TVM, Halide,
Mirage, Inductor, KernelBench). Question: how should incin grow custom kernels
beyond hand-written CUDA templates + NVRTC?

## Ranked strategies

1. **Extend the raw-kernel template path with a legality-checked fuser
   (in-house, recommended near-term).** No new library. Steal the legality
   model from TVM `FuseOps` (op-kind classification inside dataflow blocks,
   tvm.apache.org/docs/arch/fusion.html) and PyTorch Inductor's scheduler
   (dependency analysis + memory-traffic benefit): a fusion group may only
   grow when it cuts reads/writes and every member is elementwise-class.
   Lower candidates through `codegen::fragment::lower_scalar` into one
   `kernel::scalar` template so strides, packing, autotune, and NVRTC are
   inherited. Cost S. Maps: #112 step 2 (CMP-005 proven groups -> emitted
   kernel), #111 (adopt `fusion`/`scheduler` or delete the orphans).
   Mandatory gates: saved-for-backward/tape check (the documented #112
   risk), fused-vs-unfused output *and* gradient tests, admission unchanged.

2. **CubeCL behind a feature gate (parked).** github.com/tracel-ai/cubecl —
   `#[cube]` compiles to CUDA/HIP/Metal/SPIR-V/WGSL/CPU-SIMD with built-in
   autotune; production-proven by Burn 0.20. Best external match for
   incin's backend set, but adopting it now would fork the verified
   fragment path instead of finishing #112. Re-evaluate after the in-house
   fuser lands; pin versions (alpha API churn).

3. **GEMM epilogues via cuBLASLt, not codegen (#85).** Bias/activation
   epilogue fusion for addmm/linear is exactly what cuBLASLt already
   provides; CuTeDSL (Python/MLIR) has no Rust story and is out of scope.
   Maps: #85, #90 (f16/bf16 rows make the vendor path worth it), #104.

4. **Rust-native NVIDIA experiments (watchlist).** cutile-rs and cuda-oxide
   (NVlabs) — alpha, CUDA-only; conflicts with the cross-backend admission
   matrix. No action.

5. **LLM kernel generation — rejected as a generation strategy.**
   KernelBench (ICML 2025, arXiv:2502.10517) shows frontier models beat
   PyTorch on <20% of tasks; KernelBench-Verified reports reward hacking.
   Only legitimate use: offline candidate source vetted by the
   conformance harness (`ir_conformance_tests` pattern), never trusted
   emission.

## Fusion-legality consensus (for #112)

Inductor: dependency analysis + memory-traffic heuristic. TVM: pattern-kind
classification within dataflow blocks. Mirage (arXiv:2405.05751): algebraic
rewrites verified by randomized GPU testing. Halide: region-based merging.
None requires a DSL; all confirm CMP-005's exclusivity proof plus a
benefit gate plus a tape check is the right shape.

## Near-term recommendation

Do not adopt a kernel DSL now. Implement #112 step 2 in-house (strategy 1),
route GEMM epilogues to cuBLASLt (#85), triage #111's 21 orphan modules
under strategy 1's adoption seam (adopt normalization/reduction/vectorized/
strided if the fuser needs them, delete the rest), and park CubeCL as a
feature-gated experiment after #112 lands.
