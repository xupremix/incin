# What's not finished yet

Kept separate from the rest of the book so it can be updated independently,
and so nothing above has to hedge every sentence. Everything here was
verified directly against the source, not inferred from documentation  -
where a claim depends on something more likely to drift (operation counts,
op tables), it points at the generated document that stays current instead
of repeating a number that won't.

## Blocks real usage today

- **GPU training is advertised with partial execution evidence, not a full
  hardware proof.** See [Backends](./backends.md) for the generated counts.
  All three previews now advertise a training-shaped subset: the losses,
  the normalization family, `embedding`, and `dropout` are present on CPU,
  CUDA, WGPU, and Metal (composed or native, with training rows). WGPU also
  covers `conv2d`/pooling and attention end-to-end; Metal still has an empty
  spatial group (no `conv2d`/pooling) and refuses bool-mask ops
  (`masked_fill`/`where_cond`) plus the `cmp_*`/`logical_*` families.
  Runtime-verified today: CPU (complete), and the WGPU software-adapter path
  — Batch A pointwise/trig/scalar (`c9c0d035`), Batch B structural/norm/loss
  rows (`631e5a1d`), Batch C (`instance_norm`/dropout/BCE/SDPA), attention
  e2e including a training smoke that `backward`s to finite non-zero
  projection gradients (`0623e762`), and cross-entropy with backward
  (`wgpu_cross_entropy.rs`). Every-PR CI runs that software-adapter suite
  (job `wgpu`, lavapipe on ubuntu). CUDA value tests are `#[ignore]`d
  pending `HARDWARE_CUDA_RUNNER`, but they are hardware-proven on real
  NVIDIA devices: matmul parity, losses, fused attention, quantize STE
  and the two-rank NCCL harness all run green outside CI (the weekly
  matrix's CUDA job is still skipped because `HARDWARE_CUDA_RUNNER` is
  unset, issues #82 and #83). Metal's host-side suites run on any OS but
  do not prove a Metal device; the macOS Apple-Silicon hardware job is the
  device run. Verified training in this book's
  [Building models](./building_models.md) chapter is still CPU-first.
- **Attention has a native online-softmax kernel on CPU and CUDA.**
  Evaluation and zero-dropout inference can route through the catalog's
  `fused_attention` row (single descriptor dispatch, single tape entry,
  recompute backward, GQA + causal); the composed
  `scaled_dot_product_attention` row remains for backends without the
  kernel. Training with attention-weight dropout keeps the manual
  score/softmax chain on CPU because the fused row has no dropout
  operand. A typed [`KvCache`] handles incremental decode
  (`MultiHeadAttention::forward_with_cache`). Above eight query rows the
  CUDA forward covers four of them per block, sharing one key/value load
  and one block reduction per key, so global key/value streaming and the
  reduction count both fall by four and a causal block stops at the last
  row's key bound. What that is worth is measured, not assumed: on a
  GTX 1650 SUPER (CC 7.5) a 1x4x512x4096x64 forward takes 0.449/0.451 ms
  tiled against 0.473/0.478 ms one-row-per-block, about 5%, because the
  kernel accumulates in `f64` to match the CPU twin and is bound by
  double-precision arithmetic rather than by the traffic the tiling
  removes. What remains for #104 is fine-grained
  block-sparse causal skipping and tensor-core tiling - and the latter
  needs a decision this framework has not made, because CC 7.5 tensor
  cores have no `f64` path and the current parity guarantee rests on the
  `f64` intermediate.

[`KvCache`]: https://docs.rs/incin/latest/incin/nn/struct.KvCache.html

## Facade gaps (the functionality exists, but not through `incin`)

- **`BatchNorm2d` cannot carry its running statistics out of a forward
  pass.** The layer now has both modes - it normalizes by the batch's own
  mean and variance in training mode and by `running_mean`/`running_var` in
  evaluation mode, which is what `BatchNorm1d` has always done - but
  `forward` never *writes* those buffers. They reach the layer as shared
  references, and the execution contract does not carry mutations through
  them, so the `momentum` argument is currently read and never spent. A
  model that trains and then switches to evaluation mode therefore needs its
  running statistics supplied from outside (a checkpoint, or a
  `collect_state`/restore pass) or should stay in training mode, which is
  sound when the evaluation batch is large enough for its statistics to
  stand in for the population's. `vision_live` takes the second route
  deliberately, because its test pass is one forward over 1000 images.
- **The CPU `conv2d` is the throughput ceiling for a training example.**
  It is im2col plus a batched matmul, which is the right shape, and the
  optional `cpu-blas` feature hands large f32 GEMMs to a blocked,
  register-tiled kernel: the same four-step CIFAR-10 batch measured 2m15s on
  a default build and 31s with the feature (4.4x) on a 4-core CPU. Without
  `cpu-blas` the CPU sustains roughly 120 MFLOP/s through these
  convolutions, which is what sets the model sizes and epoch counts any
  example can afford. A default-build conv2d on a blocked GEMM is the
  remaining step, and it is a kernel change rather than a feature flag.
- **Scoped gradient policy** is intentionally explicit through
  `incin_core::exec::GradMode::Disabled.scope` and has no facade alias.
- **The lower-level `save_safetensors`/`load_safetensors` helpers** remain
  available under `incin_core::nn::save` for compatibility, while normal
  facade users should use `incin::prelude::{Format, ModelExt}`. The facade
  path supports the same typed snapshot contract through `ModelExt::save`
  and `ModelExt::load`.
- **Facade-level QAT (issue #93).** `Tensor::quantize`/`dequantize` exist and
  the CPU executor records the straight-through gradient, but a gradient-
  marked tensor cannot be quantized in place: gradient tracking only admits
  floating dtypes, so `weight.quantize(-1)` is refused until you
  `.detach()` first. The STE round-trip therefore runs on the canonical
  dispatch surface (`incin_core::exec::dispatch`, proven in
  `quantize_ste.rs` and the `quantization_qat` example), not through the
  `Tensor` method chain, and `quantized_matmul` has no `Tensor` method at
  all. See [Quantization](./quantization.md) and the how-to chapter.
- **No shape-only test backend.** There used to be a `DummyBackend` behind a
  `test-utils` feature; it stored a shape instead of data and claimed to
  execute every operation, so a test written against it passed whether or not
  the operation could run. It is gone. `incin::test_utils` now gates
  deterministic fault injection only, and a test that needs a backend uses a
  real one.

## Architecture in progress (affects contributors more than users)

- **Backend decomposition is still in progress.** Ordinary tensor operations use
  the per-operation descriptor execution path. The seven remaining broad
  operation-family traits have been removed from production source. Remaining
  work is splitting large backend files and making exceptional execution sites
  that cannot fit `Execute<O>` easier to maintain.
- **Accelerator inherent helpers are crate-private now.** `add`, `matmul`,
  `conv2d` and siblings on `WgpuBackendImpl`, `CudaBackendImpl`,
  `MetalBackendImpl`, and `DispatchBackend` are `pub(crate)` (or absent):
  they are not part of the public surface. Use the descriptor path described
  in [The target API and canonical dispatch](./target_api.md).
- **Distributed training** has executed tiers and the rest is planning.
  FSDP/ZeRO-1 and ZeRO-2 lower onto the trainer's synchronizer seam (#99),
  proven on CPU against scripted peers; ZeRO-3 (parameter sharding),
  tensor-parallel execution (waiting on partitioned matmul, #85/#90), and
  pipeline-parallel schedule execution are planning surfaces only. The
  NCCL transport is hardware-proven (single-process two-GPU loopback plus
  the two-process two-rank harness, including a heterogeneous pair), and
  meshes bind mixed architectures by decision; multi-host training itself
  remains ungated ambition, and the host-side collectives round-trip every
  tensor, optimizer state stays full-size per rank (owned-slice authority,
  not `1/N` memory), and global gradient clipping is unsupported while
  gradients are masked.
- **The automatic `Trainer`** (`incin::experimental::training`, `train`
  feature) has a real single-device training loop (`fit`), a
  `GradientSynchronizer` seam for data-parallel mean-reduction (#97), and
  an FSDP sharding seam for ZeRO-1/ZeRO-2 execution (#99). Single-rank
  aggregation and the sharded walks are proven on CPU; multi-device plans
  still refuse to run without the matching synchronizer
  (`TrainError::CollectivesUnavailable` / `TrainError::FsdpUnavailable`),
  and real multi-rank transport remains gated on the unset
  `HARDWARE_CUDA_RUNNER` (#82). `Trainer::fit` does not shard its input
  data: pair it with `incin-data`'s `DistributedSampler` (#98) yourself.
- **Compiled execution** is only the CPU reference evaluator under
  `incin::experimental::compiled`. It has no stable facade contract, optimized
  backend, deployment target, or portable artifact ABI; its serialized plan
  snapshots are local preview data only.
- **Only Apple Silicon and a software WGPU adapter are covered by automated
  runs.** The weekly `hardware.yml` run has a registered macOS runner (real
  Apple Silicon for the `metal` suite; `HARDWARE_METAL_RUNNER` is optional
  with a `macos-latest` fallback). The CUDA and native-WGPU jobs resolve
  their runners from repository variables that are still unset and report
  themselves skipped rather than queueing forever. Separately, every-PR CI
  runs a WGPU **software-adapter** job (lavapipe on ubuntu), so WGPU has
  execution coverage without NVIDIA hardware. The native CUDA backend is
  compile-checked in CI; its value tests stay `#[ignore]`d until
  `HARDWARE_CUDA_RUNNER` exists, but they carry real-device evidence from
  hardware runs outside CI. This is the mechanism behind
  [Backends](./backends.md) calling CPU the only backend verified across
  the complete catalog, and WGPU the only accelerator with automated
  execution evidence for its subset.

## Where the current, generated truth lives

- `docs/capabilities.md`: exactly which operations each backend supports,
  for which dtypes, regenerated from the actual registrations.
- `docs/OPERATION_SEMANTICS.md`: the full semantic contract (broadcasting,
  dtype, gradient, output rules) for every catalog operation.
- `audit-evidence/FND-005/cpu-migration-status.md`: canonical-path
  migration status, machine-checked against source on every test run.

If any claim in this book ever disagrees with one of those three, the
generated document is right and this book is stale, please file it as
such.
