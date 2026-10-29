# MoE design space: are custom designs specifically made for incin possible?

Research lane R-MoE, branch `feat/custom-autograd-dtype`. No implementation,
no decisions — proposals with evidence. The maintainer decides.

**Question.** Issue #102 settles *how routing is typed* (offset-array
formulation, option C target / option B interim; E/TOPK const; aux loss in
`Output` tuple; gate-weight-only gradients — see
[102-nameable-spans.md](102-nameable-spans.md) and
[102-moe-typing.md](102-moe-typing.md)). This memo asks the next question:
given that typing settlement *and* incin's actual subsystems, which MoE
designs are specifically natural here — things incin's typed/proof machinery
makes easy, checkable, or possible at all — and what should they be?

**Short answer.** Yes. Eight proposals below, all grounded in subsystems that
already exist (`exec/catalog`, capability rows, `Execute` dispatch,
`Router`/`MoE`/`expert_offsets`/`grouped_matmul`, `CollectiveKind::AllToAll`,
Q8_0 block encodings, `FusedAttention` precedent, telemetry emitter). The
highest-leverage pick is P1 (dropless grouped-GEMM MoE over the option-C
offsets): it is the execution target #102 already names, it matches the
industry trajectory (Megablocks → vLLM/SGLang grouped GEMM), and every other
proposal composes with it.

In-tree status at time of writing: `MoE`/`Router` dense masked forward exists
(`crates/incin-core/src/nn/moe.rs`); `Routing::expert_offsets` builds `[E+1]`
offsets with no host interop; `grouped_matmul` takes the shaped path with a
CPU implementation (`grouped_matmul_impl`) and a flat-buffer-only CUDA path;
`TopK`/`Argsort`/`Gather`/`Scatter`/`Bincount`/`ScatterAdd`/`NonZero` exist on
CPU with CUDA coverage queued (#87, #88); `FusedAttention` just landed as a
CPU-reference-plus-capability-row fused op (#104).

## 1. Design axes

| Axis | Options | Who uses what | incin implication |
|---|---|---|---|
| Granularity | Coarse (8 full-FFN experts) / fine-grained (64–256 quarter-FFN) / neuron-level | Mixtral 8x7B: 8 experts top-2; GLaM: 64 experts top-2, 1.2T params; DeepSeekMoE: 64 quarter-FFN + shared; DeepSeek-V3: 256 routed + 1 shared, top-8; Qwen3-MoE: 128 routed top-8, no shared; Qwen3-Next: 512 routed + 1 shared; MoNE: neuron experts | `E` is already a const generic (`Router<E, TOPK, …>`); expert width is a caller-chosen `FeedForward` shape. Fine granularity = larger `E`, smaller expert `D_FF` — no new mechanism, but state paths (`experts.0…E-1`) and checkpoint refusal on `E` mismatch must hold at 256-expert scale. Proposal P1 parameterizes this. |
| Shared vs routed | None / isolated shared experts (always-on dense path) | Mixtral, Qwen3-MoE: none. DeepSeekMoE 16B: 2 shared; DeepSeek-V3: 1 shared; Qwen2-MoE: 8 shared; Qwen3-Next: 1 shared | A shared expert is a plain dense `Module` composed *beside* the routed `MoE` — no routing, no offsets, full gradient. The composition question (aux-tuple threading through `Sequential`/`TransformerDecoderLayer`) is the open #101 item [102-nameable-spans.md](102-nameable-spans.md) §9 names. P1. |
| Token-choice vs expert-choice | Token-choice top-k (each token picks experts) / expert-choice (each expert picks top-k tokens) | Switch/GShard/Mixtral/DeepSeek/Qwen: token-choice. Zhou et al. 2022: expert-choice, 2x convergence speedup, perfect balance, no aux loss | Expert-choice gives every expert a **fixed bucket size** — a compile-time extent. That sidesteps the `n_e` naming problem (§5 of the spans memo) *entirely*: option-A padding becomes exact with zero drop. The most type-system-friendly routing ever proposed. P8. |
| Gating nonlinearity | Softmax+renormalize / sigmoid+normalize / softmax-then-topK vs topK-then-softmax | In-tree `Router`: softmax → topk → gather → renormalize. DeepSeek-V3: sigmoid affinities, normalized over selected (`norm_topk_prob`). NVIDIA upcycling study: softmax-then-topK preserves dense equivalence on upcycle, topK-then-softmax does not | `RouterBackend` already lists `Softmax/TopK/Gather/SumKeepDim/Div/Mul/Sub` as the routing-head bound set; a sigmoid variant swaps one row. The upcycling ordering result constrains P6's importer. |
| Load balance | Aux loss (`E·ΣfᵢPᵢ`) / loss-free dynamic bias / optimal-transport assignment / z-loss on logits | Switch: aux loss + capacity factor. ST-MoE: + z-loss. BASE: linear assignment, no aux loss. DeepSeek-V3: **auxiliary-loss-free** per-expert bias `bᵢ` added to affinities for routing only, `±γ` per step; + small sequence-wise aux loss | #102 puts the aux scalar in `Output` (tuple). Loss-free bias is instead `NoGrad` state updated by an explicit rule — the tape *cannot* route gradient through it by type, which is exactly the V3 semantics ("bias used for routing only"). P2. Telemetry hooks: P7. |
| Dropless vs capacity | Capacity-factor pad/drop / adaptive capacity (Tutel 0/negative) / dropless grouped | Switch/GShard/Tutel-positive: capacity factor. Tutel 0: min-that-fits; negative: capped dropless. Megablocks dMoE, DeepSeek-V3: dropless | #102: A = explicit bounded config, B = CPU interim, C = target. The industry moved A → C (Megablocks block-sparse → grouped GEMM; vLLM `mlp_impl='grouped'`). P1 is that move inside incin. |
| Attention pairing | GQA/MQA/SWA / MLA (latent KV) | Mixtral: SWA + GQA. DeepSeek-V2/V3: MLA (70KB/token vs 516KB LLaMA-405B). Qwen3-Next: GQA every 4th layer + linear attention | `MultiHeadAttention` already parameterizes `N_HEADS`/`N_KV_HEADS` (GQA/MQA as parameter, not module); MLA is absent. MoE + MLA co-design (small active params + small KV) is out of scope here but the pairing axis is recorded so MoE proposals don't assume MHA. |
| Expert placement | Data/model parallel / expert parallel + device/node-limited routing / 2DH all-to-all | DeepSeek-V2: device-limited (≤M devices/token); V3: node-limited (≤4 nodes). Tutel: 2DH + flexible all-to-all, 5.75x on 2048 A100. Hybrid-EP: near-hardware-limit EP all-to-all | `CollectiveKind::AllToAll` exists with a typed adjoint (`AllToAll → AllToAll`); `MeshSpec` axes (`Data`, `TensorParallel`, `Pipeline`) exist; expert axis does not. P5. |
| Precision | BF16 / FP8 blockwise (1×128 act, 128×128 weight, E4M3 everywhere) / FP4 NVFP4 (blk16) / MXFP4 (blk32) | DeepSeek-V3: FP8 training, <0.25% loss error, FP8 dispatch + BF16 combine. TensorRT-LLM: NVFP4/MXFP4 MoE (CUTLASS/TRTLLM backends); `DENSEGEMM` blog: dense-over-all-experts wins at M=64–208 low-latency. Qwen3 technically supports FP8 | #93 landed Q8_0 + block-encoding contract; #94 FP8 queued; #95 FP4 queued. Block scales are already `StorageEncoding::block`-shaped. P3. |
| Expert FFN form | ReLU MLP / SwiGLU (gate+up+down) / fused gate+up | Mixtral/DeepSeek/Qwen experts: SwiGLU. TritonMoE: fused gate+up kernel (shared A-tile loads, in-register SiLU), +1.15x | `FeedForwardKind` exists in-tree; a fused gate+up is a kernel-fusion item under P4, not a module item. |

## 2. Kernel-primitive inventory

| Primitive | SOTA impl | incin status |
|---|---|---|
| argsort-based permute + gather/combine (`padded_gather`/`padded_scatter`) | Megablocks dMoE (`dmoe.py`); TritonMoE 5-kernel pipeline (router / permute / gate+up / down-GEMM / unpermute) | **Exists (CPU).** `argsort`/`gather`/`scatter`/`index_select` on CPU; CUDA queued (#87, #88). `expert_offsets` (bincount→cumsum→concat, no host interop) already constructs the sort metadata the permute needs. |
| Grouped GEMM, offsets-driven (`lhs [T,K] × rhs [E,K,N]`, `offsets [E+1]`) | CUTLASS grouped GEMM (incl. Blackwell blockscaled); vLLM `fused_moe_kernel` (`sorted_token_ids` + `expert_ids` + `num_tokens_post_padded`); AITER `grouped_gemm`; SGLang DeepGEMM MoE | **Exists (CPU) / partial (CUDA).** `op::GroupedMatMul` catalog row + CPU `grouped_matmul_impl` + flat-buffer-only CUDA path; rides the `embedding` capability group (union-dtype trick documented in `declarations.rs`). Missing: static-shape overload (§8 of spans memo), performant CUDA grouped kernel. P1. |
| Batched / grouped top-k (per-group top-k for fine-grained routing) | vLLM `fused_topk`/`grouped_topk` (+ bias params, `e_score_correction_bias`); AITER strided grouped-topk router (Kimi-K3) | **Missing.** `TopK` exists (CPU full-dtype; CUDA f32-only per capability rows) but no grouped variant. Needed at 128–512-expert scale (Qwen3/Qwen3-Next regime). P4/P1 follow-on. |
| EP utilities: `moe_align_block_size`, `expert_map` (global→local expert ids) | vLLM `FusedMoEParallelConfig` + `expert_map` buffer; round-robin placement | **Missing.** No expert axis on the mesh; no local/global expert-id map. P5. |
| Fused permute/unpermute + dispatch/combine | SGLang `moe_permute_unpermute`; Tutel fast encode/decode (−90% mem); vLLM `moe_permute_unpermute` | **Missing.** Candidate `op::FusedMoE`-family row following the `FusedAttention` precedent (own capability group, CPU reference, composed fallback). P4. |
| All-to-all dispatch/combine (incl. 2DH hierarchical, fused comm+GEMM) | Tutel 2DH + flexible all-to-all; FasterMoE dynamic scheduling; CCFuser fused comm+GEMM; DeepSeek-V3 near-zero-overlap EP (20 SMs suffice) | **Type exists, execution missing.** `CollectiveKind::AllToAll` + adjoint in `dist/collective.rs`; `MeshSpec`/`DeviceMesh` logical/physical split in `dist/mesh.rs`; no expert placement, no backend transport (#99). P5. |
| FP8 blockwise grouped GEMM (1×128 / 128×128 scales, E4M3, FP32 accum) | DeepSeek-V3 + DeepGEMM (open-sourced); NVIDIA TransformerEngine `Float8BlockScaling`; HF `FineGrainedFP8Config` (`weight_block_size=(128,128)`) | **Missing.** Blocked on #94 (dtypes) + #85 (GEMM) + per-op quantized admission (#93 decisions 3/8). P3. |
| FP4 block-scaled grouped GEMM (NVFP4 blk16 / MXFP4 blk32) | TensorRT-LLM CUTLASS/TRTLLM NVFP4 MoE; CUTLASS Blackwell blockscaled grouped GEMM examples; Quartet (MXFP4 native training, ~2x over FP8 on RTX 5090) | **Missing.** Blocked on #95 (+ #94 de-risking, #85/#93 prerequisites, sm100 gate). P3 follow-on. |
| Quantized MoE (AWQ/Marlin GPTQ-MoE, W4A8) | vLLM `fused_marlin_moe`, `cutlass_moe_fp8/fp4`, `deep_gemm_moe` | **Missing.** Connects to #93 (Q8_0 landed, admission tables explicitly deferred as documented gap). P3. |
| Capacity-bounded dispatch (static per-expert capacity, pad/drop) | Tutel capacity modes; Switch capacity factor 1.0–1.25 | **Missing (by decision).** Option A stays an explicit config, never the default (#102). Smallest kernel item on this page; listed for completeness, not proposed. |
| Router z-loss / aux-loss fused kernels | Megatron-Core `z_loss_func`, `switch_load_balancing_loss_func` (fused flag); `MoEAuxLossAutoScaler` | **Missing.** Trivially composable from existing rows today (softmax/logsumexp exist); fused form belongs with P2/P7. |

## 3. Custom MoE proposals specific to incin

Conventions: **Size** S (< 1 week sketch-to-PR scale), M (multi-week, cross-cutting
bounds), L (backend kernels and/or multi-issue programs). **Needs** names catalog
ops, capability rows, backends, dtypes, and dist primitives concretely.
All sketches are `proposed — sketch only`.

### P1. Dropless grouped-GEMM MoE layer over the option-C offsets (target path) — M

**What.** Implement the §7 target: `router.forward` → `flatten`+`argsort`
permute → `repeat_rows`/`gather` into the static `[T*K, D]` buffer →
`grouped_matmul` against stacked `[E, K, N]` experts with the `[E+1]`
`expert_offsets` → inverse-permutation `scatter_add` combine → `(output, aux)`
tuple. Same public type as the interim dense-masked path (option B stays
behind it on CPU until grouped GEMM lands on each backend). Parameterize
granularity (`E`, expert `D_FF`) and an always-on shared-expert module beside
the routed array (DeepSeekMoE/Qwen2-style isolation; Qwen3-style `num_shared=0`
is the same type with the field absent).

**Why incin makes it natural.** The hard part of dropless MoE — naming the
dynamic partition — is already solved *by the type system* via option C:
`n_e` is a row range, never an extent; `expert_offsets`, `bincount`,
`grouped_matmul` (offsets excluded from the cotangent, like `scatter_add`'s
index operand) all exist and are already composed in prose in the spans memo
§3/§7. The `E`/`TOPK`-const + `StatePath`-stable `[Expert; E]` decisions give
256-expert checkpoints refusal semantics for free. No other framework gets the
dropless-vs-capacity choice as a *type-level* execution-strategy swap behind
one signature; here B and C are literally the same `Module::Output`.

**Needs.** Catalog: static-shape `grouped_matmul` overload (`MatMulShape`-style
rule; today it returns `Dense<Dyn>`), #100's `BroadcastCompatible` + typed
gather/scatter index bounds for the gate-weight application pairs (spans memo
§8). Capability rows: none new (rides `embedding` group). Backends: CPU now
(`grouped_matmul_impl` exists); CUDA grouped GEMM perf work (#85 → #103) +
routing-prims CUDA (#87, #88). Dist: none (single-device first; P5 later).

**Unlocks.** The maintainer's stated end-state for #102; perf parity trajectory
with Megablocks/vLLM; the substrate every proposal below composes with.

### P2. Auxiliary-loss-free balancing as typed `NoGrad` bias state — S/M

**What.** DeepSeek-V3-style loss-free balancing: per-expert bias `bᵢ` added to
affinities *for the routing decision only*; gate values still derive from the
original scores; after each step, `bᵢ ±= γ` by overload/underload. Keep the
#102 aux-tuple (and its small sequence-wise complement, as V3 does) — the bias
is an *additional* mechanism, not a signature change.

**Why incin makes it natural — arguably possible *cleanly* only here.** V3's
semantics ("bias used for routing only, no interference gradients") is
currently a comment in every codebase that implements it. In incin it can be a
*type fact*: the bias lives in a `NoGrad` buffer (like the offsets tile), so
the tape cannot record a cotangent through it — the exact failure mode the
Loss-Free Balancing paper exists to prevent becomes unrepresentable. The
update rule itself fits the precedent of catalog optimizer-step ops
(`SgdStep`/`AdamStep`/`AdamWStep` already in `table.rs`): an explicit,
auditable rule rather than smuggled gradient. Determinism of the
`scatter_add` combine (#103 guarantee, #102 contract) keeps the
load measurement reproducible.

**Needs.** Catalog: a small bias-update op (or host-side rule over existing
rows) + `Buffer`-style `NoGrad` expert-bias state on `Router`. Capability
rows: existing arithmetic rows. Backends: none new. Telemetry: bias/load
values feed P7.

**Unlocks.** Removes the aux-coefficient hyperparameter the V3 report blames
for performance degradation; makes incin's MoE the only implementation where
"routing-only" is compiler-checked rather than convention.

### P3. FP8-per-expert-scale MoE riding #94 (then FP4) — L

**What.** Quantized grouped GEMM where each expert's `[K, N]` slice carries its
own block scales: activations tile-wise `1×128`, weights block-wise
`128×128`, E4M3 throughout with FP32 accumulation (the DeepSeek-V3 recipe,
<0.25% loss error vs BF16); dispatch in FP8, combine in BF16 (V3 keeps combine
high-precision). FP4 (NVFP4 blk16 / MXFP4 blk32) follows the same shape once
#95 lands.

**Why incin makes it natural.** Three preconditions SOTA had to invent are
already architecture here: (i) `StorageEncoding::block` + `DTypeKey`
versioning models per-expert block scales as *dtype-level* facts (#93
decisions 1/6, #95 memo); (ii) per-op dtype admission is specified
(compile error when statically known, descriptive error under `Dyn` — #93
decisions 3/8), which is exactly the "which grouped-GEMM variants accept
which K" table this needs; (iii) #94 already mandates scale-as-separate-`f32`
at construction (`to_f8_scaled`), so a per-expert-scale constructor is the
same pattern with `E` scale tensors. The offsets tile stays i64 throughout —
quantization never touches the routing metadata.

**Needs.** Catalog: quantized `GroupedMatMul` variant rows; `DTypeRule`
population for the MoE op set (the explicitly deferred gap, #93 decision 7).
Capability rows: new quantized-grouped rows, `Unsupported` pre-Hopper (per
#94 memo). Backends: CUDA (#85 GEMM + #94 dtypes); CPU pure-Rust reference
encode/decode mandatory for the #83 oracle (per #95 memo). Dtypes: E4M3/E5M2
(#94), then NVFP4/MXFP4 (#95). Dist: FP8 dispatch halves EP bytes (V3) —
couples with P5.

**Unlocks.** 2x (FP8) to ~2x-again (FP4: B200 7700 vs 3850 TFLOPS measured)
MoE throughput; weight residency for 128–256-expert layers; the hardware's
native language (block scales) spoken by the type system.

### P4. Fused route+permute+GEMA kernel family (`op::FusedMoE`) — L

**What.** A fused MoE kernel family behind one catalog row, staged like
`FusedAttention` (#104): single-descriptor dispatch covering
router-topk → align/pad (`moe_align_block_size`-equivalent) → grouped expert
GEMMs (incl. fused gate+up with shared A-tile loads, in-register SiLU per
TritonMoE, +1.15x) → weighted combine; backends without the kernel run the
composed P1 path through the same module surface.

**Why incin makes it natural.** #104 just proved the pattern end to end: a
fused op as *its own capability group* (`fused_attention = [FusedAttention]`,
honest dtype story per-backend), CPU reference kernel
(`cpu/ops/shape_ops/fused_attention.rs`), single tape entry, composed
fallback. MoE fusion is the same shape with bigger payoff (TritonMoE: 15.4x
from replacing the expert loop with one grouped GEMM; vLLM's kernel family —
`fused_moe`, `fused_marlin_moe`, `cutlass_moe`, `deep_gemm_moe`,
`moe_permute_unpermute` — shows the family structure). The catalog's
descriptor-per-operand contract + capability-row gating is precisely the
machinery for "fused where available, composed otherwise."

**Needs.** Catalog: new `op::FusedMoE` row (+ grouped-topk variant at
fine-grained scale). Capability rows: new `fused_moe` group (CUDA empty until
a kernel ships, mirroring `fused_attention`). Backends: CPU reference first;
CUDA Triton/CUTLASS-class kernel later; ROCm via AITER-shaped Triton further
out (#6). Needs P1's static overload settled first (the fused row's shape rule
is P1's rule).

**Unlocks.** Inference-latency regime (vLLM/SGLang parity track); a home for
every future MoE kernel (FP8/FP4 variants plug into the family, not the
module).

### P5. Expert-parallel all-to-all dispatch/combine over the dist mesh — L

**What.** Expert parallelism as placement over mesh axes: experts sharded
across ranks, tokens dispatched/combined with paired all-to-alls; a
device/node-limited routing policy (V2: ≤M devices/token; V3: ≤4 nodes) as a
typed routing constraint so communication cost is bounded *by construction*;
recompute the P1 offsets per-rank after dispatch (global `[E+1]` → local
`[E_local+1]` is a pure function of the expert map).

**Why incin makes it natural.** The logical half of distributed correctness is
already typed: `MeshSpec` axes with compiler-checked shard counts, and
`CollectiveKind::AllToAll` whose adjoint is itself (dispatch/combine duality
as a one-line proof in `dist/collective.rs`). The logical/physical split
(`MeshSpec` vs bound `DeviceMesh`, `TopologyProbe` at the boundary per
PROPOSALS.md §3.8) means the limited-routing policy can be checked
logically (at most M devices per token) while the transport stays a backend
concern. Tutel's lesson (static parallelism vs dynamic workload → adaptive
switching) maps onto capability-gated backend selection, an existing
mechanism.

**Needs.** Dist: expert mesh axis + placement rules (#99); `expert_map`
global→local buffer (vLLM precedent); backend all-to-all transport (missing on
all backends). Catalog: none new (dispatch = permute + collective; combine =
collective + weighted `scatter_add`). Couples with P3 (FP8 dispatch halves
bytes) and P7 (per-rank load telemetry drives EPLB later).

**Unlocks.** Multi-device MoE at all (the regime where MoE matters); V3-style
near-zero-overlap EP as a future backend optimization behind an unchanged
module type.

### P6. MoEfication importer: dense→MoE surgery as a typed state transform — M

**What.** Convert a trained dense checkpoint into MoE two ways, as a state-level
constructor: (a) **sparse upcycling** — duplicate each FFN into `E` experts +
random router (Komatsuzaki et al.; Mixtral was trained from scratch but the
upcycling recipe — softmax-*then*-topK to preserve dense equivalence at init —
is established by the NVIDIA study); (b) **MoEfication/splitting** — partition
one FFN's width into `E` narrower experts (LLaMA-MoE: split SwiGLU FFN,
continual pre-train 200B tokens, beats dense at matched active params).

**Why incin makes it natural.** State traversal is generic over concrete shapes
with `StatePath`s (`experts.0…`), and #102 already specifies checkpoint
refusal on `E` mismatch — so "this dense FFN became these `E` experts" is
expressible as a total, checkable function over state snapshots, with the
router init and the softmax-then-topK ordering as auditable constructor
arguments. Import paths exist on both sides (safetensors index, GGUF in #93).
`FreezeExpert`/`UnfreezeExpert` typestate gives the standard upcycling
workflow (freeze experts, train router; unfreeze, joint train) as type
transitions rather than scripts.

**Needs.** Catalog: none (constructor, not execution). State: a
dense→`[Expert; E]` split/duplicate transform + router seeding; import
integration (safetensors/GGUF). Needs #102 signature + #101 composition for
the produced layer to slot into `TransformerDecoderLayer`. Optional later:
co-activation clustering (MoEfication proper) as an analysis pass over
`ComputeStats`.

**Unlocks.** Every dense incin model becomes MoE raw material; the cheapest
path to a large MoE (no from-scratch training); a showcase for typed state
surgery.

### P7. Load-balance telemetry as first-class trace identities — S

**What.** Emit the routing intermediates that diagnose collapse as structured
telemetry events: tokens-per-expert histogram (the `bincount` counts P1
already computes), the aux scalar(s), loss-free biases (P2), per-rank loads
(P5), dropped/padded counts if the option-A config is ever enabled. Trace
identities (stable per-layer/per-step keys), not log lines.

**Why incin makes it natural.** The values are already materialized as tensors
on the P1 path (`counts`, aux scalar, offsets) — telemetry is a readback tap,
not new computation. `incin-telemetry` already has an emitter/reporter/run-dir
structure; host readback rows (`ToHost*Vec`) exist in the catalog. The typed
angle: because offsets/counts are `NoGrad` values with static extents
(`[E+1]`, `[E]`), their telemetry schemas are fixed at compile time per
layer type — dashboards can be generated from the type, and EPLB-style
replacers (vLLM `enable_eplb`) later consume the same identities.

**Needs.** Catalog: existing readback rows only. Telemetry: new event kinds in
`incin-telemetry`; layer-type-derived schema. Blocked on nothing (rides P1
intermediates; useful already on the option-B interim).

**Unlocks.** Routing-collapse debugging (the failure mode that "looks like a
convergence problem," #102); the measurement substrate for P2's bias updates
and any future expert-parallel load balancer.

### P8. Expert-choice routing: fixed buckets that typecheck — S/M

**What.** An expert-choice `Router` variant: each expert selects its top-k
tokens (score-gated), so every expert's bucket is a **compile-time constant**.
No aux loss (balance is structural), variable experts per token, 2x
convergence speedup reported (Zhou et al. 2022, vs Switch top-1 / GShard top-2
at matched compute).

**Why incin makes it natural — the strongest type-system fit on this page.**
The spans memo concludes data-dependent extents are unnamed *everywhere*
(PyTorch, Megablocks, Tutel, JAX, XLA). Expert-choice dissolves the premise:
bucket sizes are caller-chosen consts, so per-expert operands are ordinary
static shapes, option-A padding is exact with zero drop, and `Module::Output`
composition needs no `Dyn` span at all. It is the one routing family where
incin's static-shape claim holds end to end *without* offsets. Caveats are
honest and known: expert-choice is non-causal (train-inference mismatch for
autoregressive decoding; recent null-expert work recovers data-sparsity inside
token-choice instead) — so this is proposed as an encoder/bidirectional-track
variant, matching its SOTA usage, not as the default decoder router.

**Needs.** Catalog: `TopK` along the token dim (exists) + the same #100 index
bounds as P1 (simpler instance: index extent is const, not `T*K`). Capability
rows: existing. Backends: CPU now; CUDA with #87/#88. Composes with P7
(dead-token ratio is the natural telemetry) and P3 (fixed buckets quantize
trivially).

**Unlocks.** A no-`Dyn`, no-offsets MoE path proving the type system's full
strength on at least one routing family; a hedge if #100's index bounds slip
(P8 needs the easy half of them).

## 4. Sequencing

| Order | Proposal | Blocked on | Notes |
|---|---|---|---|
| 1 | P1 core (dropless grouped-GEMM, CPU) | #100 bounds (§8 spans memo); static `grouped_matmul` overload | CPU `grouped_matmul_impl` exists — the gap is typing, not kernels. Shared-expert + granularity parameters ride along (module-only). |
| 2 | P7 telemetry | Nothing (taps P1/B-interim intermediates) | Do early: it instruments everything after it. |
| 3 | P2 loss-free bias | #102 aux-tuple placement (decided); P7 for observation | Small, high-signal; no kernel work. |
| 4 | P8 expert-choice | #100 index bounds (easy half); #87/#88 for CUDA | Independent of P1's grouped path; encoder-track. |
| 5 | P6 MoEfication importer | #102 signature; #101 composition; safetensors/GGUF paths (#93) | No execution work; parallelizable with 1–4. |
| 6 | P1 CUDA performance | #85 (batched GEMM) → #103 grouped; #87/#88 routing prims | The perf half of P1; Megablocks/vLLM parity lives here. |
| 7 | P4 fused family | P1 shape rules; #104 pattern; CUDA kernel program | CPU reference first (fused_attention precedent), fast kernels later. |
| 8 | P3 FP8 MoE | #94 (dtypes) + #85 (GEMM) + #93 admission tables | Per-expert scales = block encodings at `E` multiplicity. FP4 (#95) after. |
| 9 | P5 expert parallelism | #99 (placement, transport); helped by P3 (bytes) + P7 (loads) | The scale-out endgame; explicitly "later" per #102. |

Cross-cutting: #101 (`Module`/`Sequential`/aux-tuple composition with
`TransformerDecoderLayer`) gates every proposal's final wiring — it is the
shared signature dependency, not any proposal's private blocker. #83
(conformance oracle) must cover each new numerical path (CPU reference
requirement already established for FP4 in the #95 memo; same rule applies to
P1-CUDA, P3, P4).

## 5. Sources relied on most

Architecture: Switch Transformer (Fedus et al. 2021,
[arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)) for capacity
factor + aux loss; DeepSeekMoE (Dai et al. 2024,
[arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)) for fine-grained
+ shared experts; DeepSeek-V2/V3 technical reports
([arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434),
[arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)) for
device/node-limited routing, MLA, dropless training, FP8 blockwise recipe,
and auxiliary-loss-free bias balancing (Wang et al. 2024a,
[openreview.net/forum?id=y1iU5czYpE](https://openreview.net/forum?id=y1iU5czYpE));
Mixtral (Jiang et al. 2024, [arxiv.org/pdf/2401.04088](https://arxiv.org/pdf/2401.04088))
for top-2 SwiGLU; GLaM (Du et al. 2022,
[arxiv.org/abs/2112.06905](https://arxiv.org/abs/2112.06905)) for 64-expert
top-2 at 1.2T; Qwen2/Qwen3 reports
([arxiv.org/html/2407.10671v2](https://arxiv.org/html/2407.10671v2),
[arxiv.org/html/arXiv%3A2505.09388](https://arxiv.org/html/arXiv%3A2505.09388))
for fine-grained + shared-then-not ablations; expert-choice (Zhou et al. 2022,
[arxiv.org/abs/2202.09368](https://arxiv.org/abs/2202.09368)); ST-MoE z-loss
via Megatron-Core
([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.moe_utils.html));
upcycling (He et al. 2024, [arxiv.org/pdf/2410.07524v2](https://arxiv.org/pdf/2410.07524v2);
LLaMA-MoE, [arxiv.org/pdf/2406.16554](https://arxiv.org/pdf/2406.16554)).

Systems/kernels: MegaBlocks (Gale et al., MLSys'23,
[arxiv.org/abs/2211.15841](https://arxiv.org/abs/2211.15841)) — the offsets
formulation that *is* option C; Tutel (Hwang et al.,
[arxiv.org/abs/2206.03382](https://arxiv.org/abs/2206.03382)) for adaptive
capacity + 2DH all-to-all; vLLM fused MoE
([github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py),
[modular_kernel.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/modular_kernel.py))
for the kernel-family structure; TritonMoE
([arxiv.org/pdf/2605.23911v1](https://arxiv.org/pdf/2605.23911v1)) for the
5-kernel pipeline and skew analysis; TensorRT-LLM MoE backends + DENSEGEMM
([nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html),
[deployment-guide-for-deepseek-r1-on-trtllm.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/deployment-guide-for-deepseek-r1-on-trtllm.md))
for FP8/FP4 MoE support matrices.

Quant/hardware: DeepSeek-V3 FP8 framework (§3.3) + DeepGEMM; TransformerEngine
`Float8BlockScaling`
([docs.nvidia.com](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html));
Blackwell FP4 claims (B200 9 PFLOPS dense FP4 / 4.5 FP8; microbench 7702 vs
3850 TFLOPS at ~96% peak; NVFP4 MLPerf 1.9x over FP8;
[developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference](https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference.html));
AMD MI300X + AITER (fused MoE 3x, DeepSeek-R1 2.1x e2e;
[github.com/ROCm/aiter](https://github.com/ROCm/aiter),
[grouped GEMM API](https://rocm.github.io/aiter/api/gemm.html)); AMD 2T-MoE
training whitepaper (EP+CP+PP sharding on MI300X clusters).

In-repo companions: [102-nameable-spans.md](102-nameable-spans.md) (option C
target / B interim; E/TOPK const; aux tuple; gate-weight-only gradients),
[102-moe-typing.md](102-moe-typing.md) (issue #102 text),
[103-routing-primitives.md](103-routing-primitives.md),
[100-broadcast-pairs.md](100-broadcast-pairs.md),
[101-attention-modules.md](101-attention-modules.md),
[104-fused-attention-kv.md](104-fused-attention-kv.md),
[94-fp8.md](94-fp8.md), [95-fp4-blackwell.md](95-fp4-blackwell.md),
[93-quantized-contract.md](93-quantized-contract.md),
[99-fsdp-tp-pp.md](99-fsdp-tp-pp.md).
