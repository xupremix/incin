# Inference-engine substrate — how incin becomes the best option to build one on

Research lane R-Infer. No implementation, no decisions — roadmap with evidence.
Grounded in branch `feat/custom-autograd-dtype` as of 2026-09-25.

## 0. What exists today (ground truth)

- `KvCache<S, B, K>` — typed, preallocated, caller-owned, static rank-4
  `[batch, kv_heads, capacity, head_dim]`; overflow is a typed
  `Error::CacheCapacityExceeded`, `len` is runtime bookkeeping never
  serialized, `reset()` reuses the allocation.
  (`crates/incin-core/src/nn/kv_cache.rs`)
- `MultiHeadAttention::forward_with_cache` / `CrossAttention::forward_with_cache`
  + `CrossAttention::prefill_memory` — per-step projection of new tokens only,
  rotary applied at absolute offsets, forced `GradMode::Disabled`, returns
  `NoGrad`. (`crates/incin-core/src/nn/attention.rs`)
- `ScaledDotProductAttention` catalog row — CPU reference composes the same math
  as the manual path; CUDA/WGPU/Metal executors dispatch the row; selection is
  still a stub, no FlashAttention tiling / online-softmax kernel exists.
  (`crates/incin-backends/src/cpu/canonical/nn.rs`,
  `crates/incin-backends/src/cuda/executor.rs`,
  `crates/incin-backends/src/wgpu/executor.rs`,
  `crates/incin-backends/src/metal/executor.rs`,
  `crates/incin-core/src/exec/catalog/inference.rs`)
- `dist/` — collective contracts (`CollectiveBackend`, `CollectiveKind`),
  deterministic `ReferenceTransport`/`ReferenceTopology` for values/counts/dtype
  behavior/adjoins, optional NCCL transport, `DeviceMesh`/`MeshSpec`, DP/PP
  planners (`DataParallelPlanBuilder`, `PipelinePlanBuilder`), collective tuning
  (`select_collective_candidate`, `commit_collective_tuning`).
  (`crates/incin-backends/src/dist/`, `crates/incin-core/src/dist/`)
- `DTypeKey`/`DTypeRegistry` + checkpoint state traversal + safetensors/GGUF
  export (`GgufExporter`, `QuantScheme::{F32, Q8_0, W4A16_Q4_0, W4A16_Q4_K_M}`,
  GGUF v3, fail-closed on unimplemented schemes).
  (`crates/incin-core/src/tensor/dtype/registry.rs`,
  `crates/incin-core/src/io/gguf.rs`, `crates/incin-macros/src/safetensors.rs`)
- `Q8_0` quantize + `quantized_matmul` (CPU `quantized_matmul_storage`, CUDA
  `quantized_matmul_q8_0` kernel; K must be a multiple of 32; no backward rule —
  inference-only by construction).
  (`crates/incin-backends/src/cpu/ops/quant.rs`,
  `crates/incin-backends/src/cuda/ops/quant.rs`,
  `crates/incin-backends/src/cuda/ops/kernels/quant.cu`)
- Capability rows + `cargo incin doctor` — per-backend capability tables,
  fail-closed dispatch (`quantized_matmul` refuses training contexts), machine
  report (devices, features, caches, probes). (`crates/incin-backends/src/capability/`,
  `crates/incin/src/doctor.rs`)
- `Tensor` facade + `Module` system — typed shapes/dtypes, `Module::forward`,
  `VisitState`/`collect_state`/`load_state`, frozen vs trainable, `Buffer`
  (non-trainable state, e.g. rotary tables).
- `crates/incin/tests/gpt_decoder_model.rs` (TinyGpt) — trains on CPU and
  round-trips through state. **Inference path (`forward_with_cache`) exists at
  the attention layer but is NOT wired to a model-level generate loop anywhere:
  no sampler, no per-model `generate_step`, no end-to-end decode test.**

## 1. Mechanism inventory

| # | Mechanism | Who proves it | incin status + file refs |
|---|-----------|---------------|--------------------------|
| M1 | Paged KV: fixed-size blocks, non-contiguous physical storage, block table per sequence, near-zero fragmentation | [vLLM / PagedAttention](https://arxiv.org/html/2309.06180v1) (SOSP 2023); [design doc](https://github.com/vllm-project/vllm/blob/main/docs/design/paged_attention.md) | **Missing.** `KvCache` is one contiguous preallocated `[batch, kv_heads, capacity, head_dim]` buffer per sequence (`crates/incin-core/src/nn/kv_cache.rs`); no block table, no sharing, no eviction. |
| M2 | Block manager: allocation / preemption / swap, copy-on-write fork for parallel sampling, sliding-window support | [vLLM block manager](https://docs.vllm.ai/en/v0.9.0/api/vllm/core/block_manager.html) | **Missing.** Closest: `Error::CacheCapacityExceeded` is fail-closed instead of preempt/swap; `Clone` shares the variable slot but has no CoW accounting. |
| M3 | Continuous (in-flight) batching: per-iteration scheduling, prefill/decode co-batching | vLLM ([docs](https://docs.vllm.ai/en/latest/)); Orca baseline comparison in the PagedAttention paper | **Missing.** No scheduler, no request queue, no iteration driver. |
| M4 | Chunked prefill: split long prompts into token-budget chunks co-scheduled with decodes | vLLM ([docs](https://docs.vllm.ai/en/latest/)) | **Missing.** `forward_with_cache` accepts multi-token chunks (`[batch, seq, d_model]`) so the *kernel-side* primitive exists, but nothing chunks or budgets. |
| M5 | Prefix caching (block match) → RadixAttention (radix-tree LRU over all requests, multi-level sharing, cache-aware scheduling) | vLLM automatic prefix caching ([docs](https://docs.vllm.ai/en/v0.6.1/index.html)); [SGLang paper](https://arxiv.org/html/2312.07104v2); [radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py); HiCache extends to host/distributed storage ([design](https://docs.sglang.io/docs/advanced_features/hicache_design)) | **Missing.** No cross-request cache identity (no token-hash → block map at all). Build-on: `DTypeKey`-style stable keys + `Buffer` storage + reference-transport test pattern from `dist/`. |
| M6 | Compressed-FSM constrained decoding (multi-token fast path for structured output) | [SGLang paper §4](https://arxiv.org/html/2312.07104v2) | **Missing.** No sampler/grammar layer of any kind. |
| M7 | Overlapped scheduling (compute/swap overlap, prefetch nodes) | [SGLang paper](https://arxiv.org/html/2312.07104v2) | **Missing.** No async executor; nothing to overlap yet. |
| M8 | Engine build: AOT graph capture, fusion, plugin autotune per GPU | [TensorRT-LLM](https://developer.nvidia.com/tensorrt-llm) (trtllm-build / LLM API); speculative-decoding engine modes ([docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md)) | **Partial.** `CompiledPlan`/`CpuCompiledPlan` compile path exists (`crates/incin-backends/tests/compiled_cpu.rs`); codegen IR/fuser/catalog exist (`crates/incin-backends/src/codegen/`); tuning service exists (`crates/incin-backends/src/tuning/`). Missing: attention-aware fusion set, GPU plugin autotune call sites, CUDA-graph capture. |
| M9 | KV-cache quantization (FP8 E4M3/E5M2, NVFP4) | TensorRT-LLM ([KV cache docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md), `--kv_cache_dtype fp8`); vLLM FP8-KV docs | **Missing.** `Q8_0` weight quant + `QuantScheme` exist but no KV-dtype path; `KvCache<K: DType>` is generic over `K`, which is the seam to build on. |
| M10 | Speculative decoding (draft-target, Medusa heads, EAGLE-1/2/3, NGram/suffix-automaton) | TensorRT-LLM ([spec-decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md), [EAGLE on Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Feature_Guide/Speculative_Decoding/TRT-LLM/README.html)); vLLM EAGLE-3; SGLang hooks | **Missing.** Requires M3 (scheduler with lookahead slots) + accept/reject + `KvCache` rollback. Nothing exists. |
| M11 | Disaggregated prefill/decode (KV transfer overlapped with compute, NIXL/UCX backends) | TensorRT-LLM ([disagg serving](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md)); DistServe/Mooncake ideas; SGLang Mooncake TransferEngine; Dynamo feature matrix ([matrix](https://docs.dynamo.nvidia.com/dynamo/resources/feature-matrix.md)) | **Partial.** `PipelinePlan`/`PipelineTransfer`, `DeviceMesh`, `ReferenceTransport`, NCCL transport exist (`crates/incin-backends/src/dist/`). Missing: KV-transfer protocol (metadata + bulk), prefill/decode role split, any network transport for KV bytes. |
| M12 | Quantized GEMM zoo (Marlin INT4, FP8 W8A8, NVFP4) + FlashAttention/FlashInfer/TRTLLM-Gen kernels | vLLM ([docs](https://docs.vllm.ai/en/latest/)); [llama.cpp backends](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file) | **Partial.** `quantized_matmul` Q8_0 on CPU+CUDA only; no INT4/FP8, no WGPU/Metal quant path, no FlashAttention kernel (SDPA is composed reference). |
| M13 | CUDA/HIP graphs (launch-overhead amortization) | vLLM ([docs](https://docs.vllm.ai/en/latest/)) | **Missing.** No graph capture on any backend. |
| M14 | AOT compiled serving + dynamic-shape handling (symbolic shapes, cross-level opt) | [Relax paper](https://arxiv.org/abs/2311.02103) (ASPLOS 2025); [Relax VM](https://tvm.apache.org/docs//arch/relax_vm.html); [MLC install](https://llm.mlc.ai/docs/install/tvm.html) | **Partial.** incin already types shapes statically (`KvCache` capacity-in-type, const head counts) and has `CompiledPlan` + codegen; what it lacks vs Relax is *symbolic* dynamic shapes — variable sequence length is `Dyn` escape hatch, not a tracked relation. |
| M15 | No-GIL Rust core, tiny deployable binary, HF-native safetensors/tokenizers | [Candle README](https://github.com/huggingface/candle/blob/main/README.md) (serverless goal, GIL removal); [candle-core](https://docs.rs/candle-core/0.9.1/candle_core/index.html) | **Exists.** incin is Rust throughout; `cargo incin doctor`, `inspect` already ship as binaries. Gap vs Candle: Candle has `candle-vllm` OpenAI server + `atoma-infer` (PagedAttention+FlashAttention2) — the *serving layer*, which incin also lacks. |
| M16 | GGUF ubiquity: format + 1.5–8-bit K-quants + CPU SIMD + every-GPU backends + hybrid offload | [llama.cpp README](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file); [performance tuning](https://ggml-org-llama-cpp.mintlify.app/advanced/performance-tuning) | **Partial (export-only).** incin *writes* GGUF (F32/Q8_0/Q4_0/Q4_K) and *inspects* it, but cannot *load/run* GGUF weights: no GGUF importer, no K-quant dequant/matmul zoo, no imatrix path. Running the zoo needs: GGUF reader → tensor mapping → per-block dequant kernels per backend → `quantized_matmul` widened per scheme. |
| M17 | Sampler (temperature/top-k/top-p/min-p/penalties) + OpenAI-compatible server + tokenizers | llama.cpp (`llama-server`), vLLM OpenAI server, SGLang runtime | **Missing entirely.** No sampling code, no tokenizer dependency, no HTTP surface. Every serving scenario bottoms out here. |

## 2. Gap analysis, ordered by serving impact

**G1 — No model-level generate loop (prefill → decode → sample). Size: M. HW: none.**
What breaks: *everything* serving. Concrete scenario: TinyGpt today can only do a
single teacher-forced forward (`gpt_decoder_model.rs`); asking it to "complete a
prompt" has no code path — there is no sampler (M17), no per-step loop calling
`forward_with_cache`, no EOS/stop handling. A 1-request chat completion is
impossible, let alone a batch.
Build on: `forward_with_cache` + `KvCache` + `prefill_memory` (all exist and
tested at the attention layer); `doctor` for device selection.
Exit-relevant: this is the single highest-leverage first item (see §4, stage 1).

**G2 — Contiguous per-sequence KV, no paging/sharing (M1/M2/M5). Size: L. HW: none for
host-side design; GPU kernels later.**
What breaks: memory economics. Concrete scenario: batch 32 of 8k-context chat on
one 80 GB GPU. KV bytes/token/layer ≈ `2 (K+V) × kv_heads × head_dim × dtype_bytes`.
For a GQA 7B-class model (32 layers, 8 kv heads, head_dim 128, fp16): 32×8k×32×2×8×128×2B
≈ 137 GB — OOM before weights. vLLM's pager packs this to ~allocated-tokens-only plus
shares the common system-prompt prefix across the 32 sequences; incin's contiguous
`capacity`-typed cache additionally *reserves max length up front per sequence*,
so fragmentation + reservation multiply the waste. Without this, batch-32 8k serving
is impossible on any single GPU.
Build on: `KvCache` (append/len/kv/reset contract), `CacheCapacityExceeded` as the
preemption signal, capability-gated backend rows.
Sub-steps: (a) block-table + non-contiguous attention read path on CPU (host-side,
provable); (b) prefix/radix match on token hashes; (c) GPU paged kernels.

**G3 — No scheduler: no continuous batching, no chunked prefill, no TTFT/SLO policy
(M3/M4/M7). Size: L. HW: none for reference scheduler.**
What breaks: throughput and tail latency. Concrete scenario: 8 concurrent chat
sessions with mixed 4k-prefill + 1-tok decodes — without iteration-level scheduling
each request runs to completion exclusively; decode throughput collapses to ~1/8 of a
continuous-batched server and a 4k prefill blocks all decodes behind it (no chunking),
spiking p99 inter-token latency into seconds.
Build on: `dist/` planner vocabulary (`PipelinePlanBuilder`, tuning service),
`CompiledPlan` for the per-iteration graph.
Sub-steps: (a) deterministic single-replica reference scheduler (host-side, scripted
arrival traces, fully testable without GPU); (b) chunked-prefill budgeting;
(c) CUDA-graph + overlap variants.

**G4 — GGUF import + K-quant zoo missing (M16/M12). Size: L. HW: CPU first, then per-backend.**
What breaks: model access. Concrete scenario: user points incin at any
`Qwen3-8B-Q4_K_M.gguf` from the Hub — today this fails: incin can *write* Q4_K
shapes but has no GGUF weight reader, no K-quant block dequant, no imatrix path, so
the entire Hub zoo (≈ the default local-inference distribution format) is unloadable.
"Speaks GGUF" today means export/inspect only.
Build on: `inspect_gguf` metadata reader, `GgufExporter`/`QuantScheme` layout code
(knows the format's tensor entries), `Quantize`/`quantized_matmul` rows, capability
gating per scheme×backend.
Sub-steps: (a) GGUF reader → `StateSnapshot` mapping for F32/Q8_0; (b) Q4_0/Q4_K/Q6_K
dequant + matmul on CPU; (c) per-accelerator kernels.

**G5 — No quantized / low-precision decode math beyond Q8_0 (M9/M12). Size: M–L.
HW: CUDA device for FP8/NVFP4 proof.**
What breaks: cost per token. Concrete scenario: serving Llama-3.1-8B at fp16 needs
~16 GB weights + ~KV at full precision; FP8 weights + FP8 KV halves both, roughly
doubling max batch on the same card. Without it incin serves at ~2× the memory cost
of TRT-LLM/vLLM baselines.
Build on: `DTypeKey`/`DTypeRegistry` (custom dtype seam), `Q8_0` precedent,
`QuantScheme` file-type plumbing, `doctor` ISA/GPU-capability probes.

**G6 — No fused attention kernel (M8/M12/M13). Size: M. HW: CUDA device for proof;
CPU tiling provable locally.**
What breaks: prefill speed and memory. Concrete scenario: 4k-token prefill on the
composed SDPA path materializes `[B,H,T,T]` scores (4k² × heads × batch fp32 —
~GBs, cf. `104-fused-attention-kv.md`); FlashAttention-style tiling keeps it O(T)
memory and ~2–4× faster. Every prefill pays this until a fused kernel lands.
Build on: `ScaledDotProductAttention` row (dispatch seam + CPU reference + `#83`
oracle convention), autotune/tuning service for tile-size selection.

**G7 — No speculative decoding (M10). Size: M. HW: needs M3 + CUDA device.**
What breaks: low-batch latency leadership. Concrete scenario: single-user chat
(decode batch 1, memory-bandwidth bound) — EAGLE-3 style drafting yields 2–3× lower
inter-token latency on TRT-LLM/vLLM; without any draft/verify path incin cannot
compete on the most visible interactive metric.
Build on: G1 loop + G3 scheduler lookahead slots + `KvCache` rollback (new).

**G8 — No disaggregated prefill/decode or KV transfer (M11). Size: L. HW: fleet
(≥2 nodes/GPUs + network transport).**
What breaks: scale-out SLO isolation. Concrete scenario: mixed prefill-heavy +
interactive fleet — without role split, long prefills share GPUs with decodes and
destroy decode p99; with disaggregation (NIXL-style KV ship overlapped with compute)
each pool scales independently. This is strictly stage 3.
Build on: `ReferenceTransport` + NCCL transport + `PipelineTransfer` + tuning
topology fingerprints.

**G9 — Dynamic shapes are `Dyn` escape, not symbolic (M14). Size: M. HW: none.**
What breaks: AOT-compiled variable-length serving. Concrete scenario: compiling the
per-iteration graph (G3) for arbitrary arrival lengths — today each distinct shape
re-plans; Relax-style symbolic shapes would compile once and run all lengths.
Build on: const head counts + static `KvCache` geometry (the static pole already
exists); `CompiledPlan`; codegen IR.

**G10 — No constrained/structured decoding (M6). Size: S–M. HW: none.**
What breaks: agent/JSON-mode workloads. Concrete scenario: function-calling server
needing guaranteed-schema JSON — without grammar-guided masking every output needs
a retry loop; SGLang's compressed FSM emits multiple constrained tokens per step.
Build on: G1 sampler seam (new).

## 3. Staged roadmap (exit criteria are tests, not vibes)

### Stage 1 — Locally provable (CPU + WGPU-680M + host-side)

1. **S1.1 Model-level generate loop + sampler (G1, M17). FIRST ITEM — highest leverage.**
   Greedy + temperature/top-k/top-p sampler; `TinyGpt`-scale `generate(prompt,
   max_tokens, stop)` on CPU calling `forward_with_cache`; EOS handling.
   Tests: determinism (greedy, fixed seed → byte-identical output); prefill+decode
   equivalence (generate-N ≡ full-forward logits for the same prefix, like the
   existing state round-trip test); stop-condition tests; CPU tok/s reported for a
   fixed tiny config (baseline number, not a threshold).
2. **S1.2 Decode-path parity across backends.** Same loop over CPU and WGPU-680M;
   logits-match test within tolerance; `doctor` reports which path was selected.
3. **S1.3 GGUF import for F32/Q8_0 (G4a).** Reader → state mapping; round-trip test
   (export → import → identical logits); load a real Hub GGUF's metadata in a test
   fixture (small file, checked-in or pinned hash).
4. **S1.4 Reference scheduler on scripted traces (G3a).** Host-side FCFS +
   iteration driver over the S1.1 loop with arrival scripts; metrics harness
   (TTFT, inter-token latency, p50/p99); no GPU needed — simulated time + real
   CPU execution.
5. **S1.5 KV measurement + block-table design spike (G2a).** Measure KV bytes/token
   for TinyGpt configs; prototype paged read path on CPU with equivalence test vs
   contiguous `KvCache`; batch-8 chat TTFT/p99 test on CPU as the tracked number.
6. **S1.6 CPU K-quant dequant + matmul (G4b).** Q4_0/Q4_K dequant equivalence vs
   reference vectors; perplexity-delta bound on a tiny model.

Stage-1 exit: `generate()` tested on CPU+WGPU; scripted-trace scheduler with p99
numbers; GGUF F32/Q8_0 import round-trip; KV bytes/token measured and published in
the test output; batch-8 CPU chat TTFT/p99 recorded as the baseline to beat.

### Stage 2 — Needs CUDA device

1. **S2.1 Fused attention kernel (G6).** Tiled online-softmax SDPA vs `#83` oracle
   exactness test; prefill 4k tokens tok/s test on device; causal block-skipping.
2. **S2.2 Paged KV on GPU (G2b/c).** Block-table kernels; equivalence vs CPU paged
   path; batch-32 8k-context residency test (fits where contiguous OOMs — the G2
   scenario as a regression test).
3. **S2.3 Continuous batching + chunked prefill on device (G3b).** Mixed
   prefill/decode co-batch test; chunked 4k-prefill TTFT bound test; CUDA-graph
   capture for the steady-state decode step (M13).
4. **S2.4 FP8/KV-quant + INT4 GEMM (G5, M12).** Weight + KV FP8 paths; accuracy-hold
   test (perplexity delta bound); memory-halving assertion on the G5 scenario.
5. **S2.5 Prefix/radix cache (M5).** Hit-rate test on scripted shared-prefix
   traces; skip-prefill equivalence test; LRU eviction test.
6. **S2.6 Speculative decoding (G7).** NGram drafter first (no training needed),
   then EAGLE-style head; acceptance-rate + identical-output (byte-match vs
   non-speculative) tests; inter-token-latency improvement test at batch 1.
7. **S2.7 K-quant zoo on CUDA + sampler/server surface.** Port S1.6 kernels;
   OpenAI-compatible endpoint smoke test (M17 completion).

Stage-2 exit: prefill-4k tok/s ≥ tracked baseline with exactness proof; batch-32 8k
residency; chunked-prefill TTFT bound; FP8 accuracy-hold; radix hit-rate on scripted
traces; speculative byte-match + latency win; HTTP smoke test.

### Stage 3 — Needs fleet / hardware

1. **S3.1 Tensor/pipeline-parallel inference at scale.** `DeviceMesh` TP/PP for
   decode (training-heritage planners exist; inference needs the no-grad,
   cache-coherent variants); multi-GPU equivalence test.
2. **S3.2 Disaggregated prefill/decode (G8).** KV-transfer protocol over NCCL/next
   transport; prefill→decode handoff test on 2 ranks; overlapped-transfer test;
   decode-p99-isolation test under prefill load.
3. **S3.3 Fleet scheduler (KV-aware routing, SLA planner, migration).**
   Dynamo-matrix features ([matrix](https://docs.dynamo.nvidia.com/dynamo/resources/feature-matrix.md)):
   KV-aware routing, SLA-based planner, request migration/cancellation — each with
   a scripted-fleet test on reference transports before native ones.
4. **S3.4 Symbolic dynamic shapes for the serving graph (G9).** Compile-once
   per-iteration artifact; shape-genericity test (one artifact, many lengths).

Stage-3 exit: 2-rank disaggregated handoff test green; decode-p99 isolation number;
fleet-script tests on reference transport; single compiled artifact serving all
lengths in the tested range.

## 4. Condensed roadmap + the single highest-leverage first item

```
Stage 1 (local):  generate loop + sampler → WGPU parity → GGUF import (F32/Q8_0)
                  → scripted scheduler + metrics → paged-KV CPU spike + KV bytes/token
                  → CPU K-quant dequant
Stage 2 (CUDA):   fused attention → paged KV on GPU → continuous batch + chunked
                  prefill (+CUDA graphs) → FP8/INT4 → radix cache → speculative
                  decoding → HTTP surface
Stage 3 (fleet):  TP/PP decode → disaggregated P/D + KV transfer → fleet scheduler
                  (routing/SLA/migration) → symbolic shapes
```

**Highest leverage: S1.1 — the model-level generate loop + sampler.** Reason: every
other mechanism is unprovable end-to-end without it (no TTFT without generation, no
acceptance test without a sampler, no scheduler without a step function), it needs
no hardware, and it converts the already-tested `forward_with_cache` primitive into
the first user-visible inference act (TinyGpt completes a prompt). All stage-1
throughput/latency numbers anchor to it.

## 5. incin-specific advantages to lean into (3–5, each with a beating scenario)

1. **Typed shapes catch batching bugs at compile time.** Const head counts
   (`D_MODEL/N_HEADS/N_KV_HEADS` with `const { assert! }`), static `KvCache`
   geometry, dtype-parameterized modules. Scenario: adding GQA to a serving build —
   a mismatched kv-head expansion fails at the construction call site (proven by
   compile-fail fixtures today) while a vLLM/SGLang paged-KV shape mismatch surfaces
   as a midnight CUDA illegal-memory-access in production.
2. **Fail-closed capability discovery for heterogeneous fleets.** Capability tables
   + `doctor` + per-row refusal (`quantized_matmul` has no backward rule and says
   so). Scenario: rolling out Q4_K serving across a mixed CPU/iGPU/CUDA fleet — incin
   reports per-node scheme×backend support before allocating and refuses cleanly
   where unsupported; vLLM/SGLang equivalents surface as kernel-not-implemented
   crashes mid-rollout.
3. **GGUF in *and* out with a refuse-to-lie contract.** Exporter already refuses
   unimplemented schemes rather than writing lying headers. Scenario: a team
   fine-tunes, quantizes, and ships a local-first model — one toolchain trains,
   exports verifiable GGUF, and (post-S1.3/S1.6) reloads it bit-identically, where
   today that loop spans transformers + llama.cpp + conversion scripts with silent
   dtype/layout drift between them.
4. **No-GIL Rust core with a deterministic reference for every distributed claim.**
   `ReferenceTransport` establishes collective values/counts/adjoins before NCCL is
   allowed to optimize them; the same pattern extends to schedulers and KV transfer.
   Scenario: debugging a prefill/decode handoff checksum mismatch — replay against
   the deterministic reference on a laptop instead of bisecting NCCL behavior on a
   fleet; Python-first stacks (vLLM, SGLang, TRT-LLM's Python API) cannot offer a
   hardware-independent oracle of the same strength.
5. **Inference-only-by-construction is already typed.** `NoGrad` decode,
   `GradMode::Disabled.restrict`, `Buffer` vs parameter, `Frozen`, backward-less
   `quantized_matmul`. Scenario: enabling KV-cache quantization or int4 decode —
   the type system guarantees no tape, no gradient storage, no optimizer state is
   ever allocated for the decode path; in PyTorch-based servers that guarantee is a
   `torch.no_grad()` context away from being silently dropped.

## Sources relied on most

- [PagedAttention paper (SOSP 2023)](https://arxiv.org/html/2309.06180v1) — M1–M4
  mechanisms, block manager, preemption, sharing.
- [vLLM docs](https://docs.vllm.ai/en/latest/) and
  [block manager API](https://docs.vllm.ai/en/v0.9.0/api/vllm/core/block_manager.html) —
  continuous batching, chunked prefill, prefix caching, CUDA graphs, quantization list.
- [SGLang paper](https://arxiv.org/html/2312.07104v2) and
  [radix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) —
  RadixAttention, compressed FSM, overlapped scheduling; HiCache
  [design](https://docs.sglang.io/docs/advanced_features/hicache_design).
- [TensorRT-LLM](https://developer.nvidia.com/tensorrt-llm):
  [KV cache incl. FP8](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md),
  [speculative decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md),
  [disaggregated serving](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md).
- [Dynamo feature matrix](https://docs.dynamo.nvidia.com/dynamo/resources/feature-matrix.md) —
  disaggregated serving / KV-aware routing / SLA planner / migration cross-check.
- [Relax paper](https://arxiv.org/abs/2311.02103) + [Relax VM](https://tvm.apache.org/docs//arch/relax_vm.html) +
  [MLC](https://llm.mlc.ai/docs/install/tvm.html) — AOT + symbolic dynamic shapes.
- [Candle README](https://github.com/huggingface/candle/blob/main/README.md) +
  [candle-core](https://docs.rs/candle-core/0.9.1/candle_core/index.html) — Rust
  serving state (candle-vllm, atoma-infer with PagedAttention).
- [llama.cpp](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file) +
  [performance tuning](https://ggml-org-llama-cpp.mintlify.app/advanced/performance-tuning) —
  GGUF zoo, K-quants, backend breadth, hybrid offload.
