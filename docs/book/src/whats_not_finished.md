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
  (job `wgpu`, lavapipe on ubuntu). CUDA remains compile-gated: hundreds of
  `#[ignore = "requires CUDA hardware"]` value tests (cuBLASLt, matmul
  dtypes, losses, train ops) pass only on a machine with a device, and the
  weekly matrix's CUDA job is still skipped because `HARDWARE_CUDA_RUNNER`
  is unset (issues #82 and #83). Metal's host-side suites run on any OS but
  do not prove a Metal device; the macOS Apple-Silicon hardware job is the
  device run. Verified training in this book's
  [Building models](./building_models.md) chapter is still CPU-first.
- **Attention still has no online-softmax/flash kernel.** Evaluation and
  zero-dropout inference route through the catalog's composed
  `scaled_dot_product_attention` row (a single descriptor dispatch; the CPU
  backend materializes scores the same way the old hand-composed path did).
  WGPU and Metal advertise that composed row too; WGPU has e2e forward and
  training-smoke coverage for it. Training with attention-weight dropout
  keeps the manual score/softmax chain on CPU because the fused row has no
  dropout operand. A typed [`KvCache`] handles incremental decode
  (`MultiHeadAttention::forward_with_cache`); what remains for #104 is a
  true flash-style kernel with block-sparse causal skipping.

[`KvCache`]: https://docs.rs/incin/latest/incin/nn/struct.KvCache.html

## Facade gaps (the functionality exists, but not through `incin`)

- **Scoped gradient policy** is intentionally explicit through
  `incin_core::exec::GradMode::Disabled.scope` and has no facade alias.
- **The lower-level `save_safetensors`/`load_safetensors` helpers** remain
  available under `incin_core::nn::save` for compatibility, while normal
  facade users should use `incin::prelude::{Format, ModelExt}`. The facade
  path supports the same typed snapshot contract through `ModelExt::save`
  and `ModelExt::load`.
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
- **Distributed training** has one executed tier and the rest is planning.
  FSDP/ZeRO-1 and ZeRO-2 lower onto the trainer's synchronizer seam (#99),
  proven on CPU against scripted peers; ZeRO-3 (parameter sharding),
  tensor-parallel execution (waiting on partitioned matmul, #85/#90), and
  pipeline-parallel schedule execution are planning surfaces only. No
  NCCL-wired synchronizer ships (`HARDWARE_CUDA_RUNNER` unset, #82), the
  host-side collectives round-trip every tensor, optimizer state stays
  full-size per rank (owned-slice authority, not `1/N` memory), and global
  gradient clipping is unsupported while gradients are masked.
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
  `HARDWARE_CUDA_RUNNER` exists. This is the mechanism behind
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
