#91 WGPU gap in tiers (112 ops) — L (Tier 0: S)

> **Status (2026-09-22):** Batch A landed as `c9c0d035` — 24
> pointwise/trig/scalar operations runtime-verified on the local Vulkan
> adapter against CPU-twin references, plus `sin`/`clamp` backward kernels
> (`wgpu_pointwise_gap.rs`, 579 lines, 8 tests). Rows are `f32`-only
> contiguous. WGPU suite: 653+ green. Skipped rows (cmp/logical/indexing)
> remain blocked on a multi-dtype advertising policy — WGPU declares no
> bool dtype, while the catalog types comparisons as `bool`. Tier 1
> (`layer_norm`, `rms_norm`, `concat`, `slice`, `narrow`, `masked_fill`,
> `where_cond`, `tril`/`triu`, `embedding`, `bmm`, `linear`) still open.
Finding: 46/158, all f32 contiguous; only accelerator with CI execution (lavapipe). Kernels EXIST AHEAD OF REGISTRATION: shaders/binary.wgsl op_modes 7-17 (all cmp_*, logical, maximum/minimum) + scalar.wgsl scalar ops are implemented but unadvertised. shape.wgsl handles slice/transpose/broadcast to rank 6. Tier 1 (attention path, 17): softmax norms layout masked_fill embedding bmm. Tier 2: losses+norms+backward. Tier 3: ~55 breadth.
Recommendation: Tier 0 first (register implemented modes honestly, cmp_* as bool-producing); Tier 1 softmax (workgroup tree per reduce.wgsl pattern) norms embedding where_cond bmm; make device init limit-aware ONCE (max_compute_workgroup_size_x, conditional shader-f16) before Tier 3 template sweep; chunk >128MiB storage bindings or refuse.
Risk: lavapipe vs native divergence until #82; no f64 ever; f16 behind shader-f16; no float atomics (matters for #103 scatter_add); rank ceiling 6 stated in rows.
