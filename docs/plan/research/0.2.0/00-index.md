# 0.2.0 research index

29 design reports, one per open issue/theme, produced by parallel research
agents grounded in the tree (every Finding cites file:symbol evidence).
Sequencing follows #105 with the corrections the research surfaced.

## Status of this document

The reports below were written against the tree as of 26 Aug 2026. Work has
landed since. This section records what changed, so a reader can tell a
finding that still holds from one that has been overtaken; the reports
themselves are left as written rather than edited in place.

**Completed since the sweep**

- **#7 Metal trainer selection** -- closed. Landed as `900c5310`, merged in
  `8655205e`. Tier 0 item 3 is done.
- **#82's resolver defect** -- the live half is fixed. `54174f68` stops NCCL
  jobs being scheduled onto GPU-less runners, merged in `831974c1`. The issue
  stays open for the other half: no self-hosted NVIDIA runner is registered
  yet, so everything hardware-gated remains unfalsifiable.
- **#89 creation/readback rows** -- implemented on CUDA, WGPU and Metal, nine
  rows each. The report's "pure registration" estimate held, with one addition
  it did not anticipate: a host readback must reject a non-contiguous operand
  rather than copying the whole allocation, or a strided view silently returns
  the wrong window and reports success.
- **#4 LayerNorm backward** -- closed. Fused backward replaying saved
  mean/rstd, measured against CPU on hardware; the training cell flipped only
  after parity.
- **#103's first slice** -- `scatter_add`, `log_softmax`, `logsumexp_dim`,
  `logsumexp_keepdim` and now `one_hot` are catalog operations with CPU
  executors, capability rows and conformance fixtures. Still absent: `sort`,
  `nonzero`, `repeat_interleave`, `bincount`, `grouped_matmul`. The report's
  "absent from the 174-operation catalog" is therefore stale for four of the
  eight; its `DuplicateIndexRule::Accumulate` warning and its determinism
  contract stand.
- **#124 topk rewrite** -- closed. `Tensor::topk` travels `execute_shaped_n`
  with one proof per output instead of untyped dispatch plus unchecked
  `from_parts`.
- **#114 optimizers** -- closed. All four optimizers verified against closed
  forms on hardware.
- **#84 narrowed by measurement** -- the loss half is done on CUDA (mse, l1,
  bce-with-logits, cross-entropy, dropout replay, softmax/rms/transpose-view
  tape entries); what remains is group/instance norms and batch-norm
  backward (#123).
- **CUDA movement rows widened** -- transpose, broadcast, narrow and concat
  claim the dense storage set on CUDA after a byte-exact hardware matrix;
  catalog 174 operations, 164 backend-executable, oracle floor 163.

**Corrections the later work forced**

- **#91's Tier 0 estimate is wrong in kind, not degree.** The report calls it
  "register already-implemented kernels -- S". That holds only for the
  non-differentiable subset. `maximum`/`minimum`/`abs_diff` sit in a group
  declared `training = true`, and the fused shader modes push no tape entry,
  so registering them as found would have advertised trainable operations that
  silently drop gradients. They needed gradients written, not just rows.

  The remaining Tier 0 modes are blocked outright. The catalog types all six
  comparisons and three logicals as `DTypeId::Bool`
  (`exec/catalog/inference.rs`, `exec/catalog/descriptor.rs`), CPU has a real
  `CpuBuffer::Bool`, and WGPU has no bool dtype anywhere -- it declares `f32`
  and nothing else across every row. Those modes write 0.0/1.0 into an f32
  buffer, so registering them would claim an f32 result for an operation the
  catalog types boolean. This is "add a dtype to the backend, then wire, then
  register", not "register", and it also blocks Tier 1's `where_cond`.

- **#103 cites the wrong enum variant; its substantive claim holds.** The
  public `scatter` declares `DuplicateIndexRule::LastWriteWins`, not `Reject`.
  The refusal in CPU's `Execute<op::Scatter>` is the backend declining to
  implement `Reject` semantics, not a duplicate-index check, so under the rule
  that actually runs colliding indices still give success and one silently
  dropped contribution. `DuplicateIndexRule` has only two variants, so
  `scatter_add` is inexpressible rather than miscomputed. Note before starting:
  the enum derives `Serialize`/`Deserialize` and is not `#[non_exhaustive]`,
  so a third variant is a wire-format change and a downstream break.

- **#83 diverged from its own recommendation.** The report says "extend
  `external/conformance.rs`". The harness that shipped is a new module at
  `crates/incin-backends/src/conformance/` (`fixtures`, `operands`, `plan`,
  `shaped`); `external/conformance.rs` is untouched. Not a defect, but a
  reader following the recommendation goes to the wrong file. The harness
  currently poses CPU only, so accelerator rows are still verified by
  hand-written per-backend tests rather than by the oracle.

- **#90's transmute finding is right about the hazard and wrong about both its
  scope and its failure mode.** The report locates it in the CUDA matmul
  executor and describes it as "bf16 today = silent byte misread". There are
  nineteen unconditional `transmute::<f32>` calls across five files
  (`ops/matmul.rs`, `ops/conv.rs`, `ops/pool.rs`, `ops/shape.rs`), not one, and
  none of them had a dtype guard.

  The failure depends on which way the width goes, and cudarc decides it:
  `transmute::<S>(len)` returns `None` when `len * size_of::<S>()` exceeds the
  allocation. A **narrower** dtype (`f16`, `bf16`, `u8`) therefore asks for
  more bytes than exist, so the transmute fails and the `unwrap` panics. A
  **wider** dtype (`f64`) satisfies the byte check, so the transmute succeeds
  and the kernel reads the first half of the allocation as `f32`. The silent
  misread the report describes is real, but it is the `f64` case, not the
  `bf16` one.

  Both were unreachable, held off only by the capability rows being
  `F32_ONLY` while `validate_cuda_storage_dtype` deliberately accepts `f64`,
  `f16`, `bf16`, `i64`, `q8_0` and `bool` as storage. Widening a `matmul`,
  `conv` or `pool` row, which is what #90, #86, #87 and #106 all propose,
  would have made them reachable with nothing in the code to stop it.

  `validate_cuda_f32_kernel` now guards all nine entry points, so a wider row
  produces the same typed `UnsupportedDType` refusal the rest of the backend
  gives instead of a panic or a wrong answer. That does not implement #90; it
  removes the trap #90 would otherwise spring.

- **The "dead/unwired code" theme is broader than the report states.** It
  names WGPU's binary/scalar shader modes. `transpose` was a further instance:
  a complete WGSL kernel with a correct tape entry, in
  `wgpu/backend/shape_ops.rs`, that nothing advertised, so dispatch refused an
  operation the backend could already perform. Worth grepping each backend for
  implemented-but-unregistered kernels before estimating any of #86-#92.

**WGPU coverage has moved.** #91 opens with "WGPU advertises 46 of the 158".
It now advertises 63: +9 from #89, then `maximum`, `minimum`, `abs_diff`,
`softmax`, `transpose`, `flatten`, `squeeze`, `unsqueeze`. `softmax` is
composed from `max_keepdim`/`sub`/`exp`/`sum_keepdim`/`log` in the numerically
stable form CPU uses, so its `training = true` is the tape replay rather than
new hand-derived math. Tier 1 still needs `layer_norm`, `rms_norm`, `concat`,
`slice`, `narrow`, `masked_fill`, `where_cond`, `tril`, `triu`, `embedding`,
`bmm` and `linear`.

## Execution order (dependency-driven)

**Tier 0 — unblock everything (S each, do first)**
1. #82 GPU CI: resolver defect fixed; the ephemeral self-hosted NVIDIA runner
   is still outstanding. Everything hardware-gated is unfalsifiable until it
   exists.
2. ~~#89 creation/readback rows~~ -- done.
3. ~~#7 Metal trainer selection~~ -- done.

**Tier 1 — decisions (gate the model-authoring chapter)**
4. #100 broadcast pairs (M): BroadcastShape already exists — re-bind
   where_cond/masked_fill only. Gates #101/#102. Note for whoever takes this:
   `BroadcastShape`'s `#[diagnostic::on_unimplemented]` message never fires,
   because the incompatible branch returns `()` as its bottom, so the trait
   resolves and the failing obligation lands on `(): Dim` one layer down where
   no curated message is attached. A named uninhabited bottom carrying both
   operands would fix it.
5. #93 quantized contract (M): NumericDType marker, block-aware sharding,
   type-enforced NoGrad. Gates #94/#95/#1.
6. #102 MoE typing (M): static outer shapes + E+1 offset array.

**Tier 2 — correctness bugs first**
7. #103 scatter_add (S within M): see the correction above. Then the other 7
   routing primitives (nonzero waits on #102).
8. #90 dtype-parametric matmul (M): the f32 transmute is a latent unsoundness
   for bf16 storage; GemmComputeType mapping. Gates #2.

**Tier 3 — training on GPU**
9. #84 losses/dropout/norms (L), #4 LayerNorm backward (M), #2 precision
   policy (M), #85 cuBLASLt (M) — all four close against #83's harness.
10. #83 conformance harness (M): module exists and poses CPU. The remaining
    work is extending it to pose the accelerators, plus the JSON artifact
    feeding a verified-on column in capabilities.md.

**Tier 4 — breadth + depth**
11. #86 pointwise macro (M), #87 reductions (L), #88 indexing (L, safety),
    #106 last 6 (M, kernels partially exist unwired), #91 WGPU tiers
    (Tier 0 partly done; the rest blocked on a WGPU bool dtype).
12. #101 attention modules (M/L), #104 KvCache first (S/M) then fused
    attention (L), #8 execute_impls! (M), #96 descriptor-keyed subsystems (L),
    #94 FP8 (L) -> #95 FP4 (M) -> #3 (M) -> #1 (M), #99 FSDP tier (M) ->
    TP -> PP, #6 ROCm (M), streaming writes (M, 0.3.0 lane).

## Cross-cutting findings
- **Dead/unwired code is a theme**: embedding.cu (non-conforming, unreferenced),
  adaptive-pool kernels, WGPU binary/scalar modes, WGPU `transpose`, quant.cu,
  cuBLASLt — several issues are partly "wire + register", not "write". Grep for
  implemented-but-unadvertised kernels before estimating.
- **#83 is the multiplier**: every close-out criterion across 12 issues reads
  "verified by the harness". It now exists for CPU; extending it to the
  accelerators is the highest-leverage remaining M in the milestone.
- **Gate discipline**: public-API baselines will drift on #100/#2/#8/#96;
  no_std check applies to any core shapes/dtype work (#100/#93/#96);
  generated docs regenerate on every catalog/capability touch (#103, #89, #91).
  A new facade feature also needs rows in `docs/book/src/feature_flags.md`,
  `docs/FEATURE_MATRIX.md` and `[package.metadata.incin.feature-contract]`, or
  `tools/check-docs.py` and the xtask suite fail.
