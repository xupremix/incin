# 0.2.0 research index

29 design reports, one per open issue/theme, produced by parallel research
agents grounded in the tree (every Finding cites file:symbol evidence; full
worked examples in each file). Sequencing follows #105 with these corrections
the research surfaced.

## Execution order (dependency-driven)

**Tier 0 — unblock everything (S each, do first)**
1. #82 GPU CI: fix the dist2-network/multinode resolver defect + register an
   ephemeral self-hosted NVIDIA runner. Everything hardware-gated is
   unfalsifiable until this exists.
2. #89 creation/readback rows (S): the primitives already exist on all three
   accelerators — pure registration. First conformance-harness customers.
3. #7 Metal trainer selection (S): mirrored constants + parity test.

**Tier 1 — decisions (gate the model-authoring chapter)**
4. #100 broadcast pairs (M): BroadcastShape already exists — re-bind
   where_cond/masked_fill only. Gates #101/#102.
5. #93 quantized contract (M): NumericDType marker, block-aware sharding,
   type-enforced NoGrad. Gates #94/#95/#1.
6. #102 MoE typing (M): static outer shapes + E+1 offset array.

**Tier 2 — correctness bugs first**
7. #103 scatter_add (S within M): scatter overwrites = silent wrong answers
   today. Then the other 7 routing primitives (nonzero waits on #102).
8. #90 dtype-parametric matmul (M): the f32 transmute is a latent unsoundness
   for bf16 storage; GemmComputeType mapping. Gates #2.

**Tier 3 — training on GPU**
9. #84 losses/dropout/norms (L), #4 LayerNorm backward (M), #2 precision
   policy (M), #85 cuBLASLt (M) — all four close against #83's harness.
10. #83 conformance harness (M): extend external/conformance.rs; registry-
    driven enumeration; JSON artifact -> verified-on column.

**Tier 4 — breadth + depth**
11. #86 pointwise macro (M), #87 reductions (L), #88 indexing (L, safety),
    #106 last 6 (M, kernels partially exist unwired), #91 WGPU tiers
    (Tier 0 = S: register already-implemented kernels).
12. #101 attention modules (M/L), #104 KvCache first (S/M) then fused
    attention (L), #8 execute_impls! (M), #96 descriptor-keyed subsystems (L),
    #94 FP8 (L) -> #95 FP4 (M) -> #3 (M) -> #1 (M), #99 FSDP tier (M) ->
    TP -> PP, #6 ROCm (M), streaming writes (M, 0.3.0 lane).

## Cross-cutting findings
- **Dead/unwired code is a theme**: embedding.cu (non-conforming, unreferenced),
  adaptive-pool kernels, WGPU binary/scalar modes, quant.cu, cuBLASLt — several
  issues are partly "wire + register", not "write".
- **#83 is the multiplier**: every close-out criterion across 12 issues reads
  "verified by the harness"; its extension is the highest-leverage M in the
  milestone.
- **#82's resolver defect** (NCCL jobs on GPU-less runners) is a live bug, not
  a gap.
- **Gate discipline**: public-API baselines will drift on #100/#2/#8/#96;
  no_std check applies to any core shapes/dtype work (#100/#93/#96);
  generated docs regenerate on every catalog/capability touch (#103, #89).
