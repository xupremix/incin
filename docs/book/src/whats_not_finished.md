# What's not finished yet

Kept separate from the rest of the book so it can be updated independently,
and so nothing above has to hedge every sentence. Everything here was
verified directly against the source, not inferred from documentation  -
where a claim depends on something more likely to drift (operation counts,
op tables), it points at the generated document that stays current instead
of repeating a number that won't.

## Blocks real usage today

- **GPU training is advertised, not verified.** See [Backends](./backends.md);
  the previews cover basic arithmetic, reductions, `matmul`, and
  `conv2d`/pooling; WGPU adds thirteen unary activations, and CUDA adds
  `softmax` and `rms_norm`, the normalization family through `batch_norm`
  with training rows, and the loss functions, `embedding`, and `dropout`.
  WGPU and Metal still lack all of those. Two narrower surfaces *are*
  runtime-verified: CPU, and WGPU's #91 Batch A (24 pointwise/trig/scalar
  operations plus `sin`/`clamp` backward, executed on the local Vulkan
  adapter against CPU-twin references, `c9c0d035`). CUDA's cuBLASLt matmul
  path (`4023ceed`) is compile-gated only — six host policy tests pass,
  six value tests are `#[ignore]`d pending the #82 runner. No GPU execution
  runs in CI, so the CUDA training path remains a declared capability
  awaiting evidence (issues #82 and #83), and verified training in this
  book's [Building models](./building_models.md) chapter is CPU-only.
- **Attention still has no online-softmax/flash kernel.** Evaluation and
  zero-dropout inference route through the catalog's composed
  `scaled_dot_product_attention` row (a single descriptor dispatch; the CPU
  backend materializes scores the same way the old hand-composed path did).
  Training with attention-weight dropout keeps the manual score/softmax chain
  because the fused row has no dropout operand. A typed [`KvCache`] handles
  incremental decode (`MultiHeadAttention::forward_with_cache`); what remains
  for #104 is a true flash-style kernel with block-sparse causal skipping.

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
- **The accelerator backends still expose operation helpers that bypass
  canonical dispatch.** `WgpuBackendImpl`, `CudaBackendImpl`,
  `MetalBackendImpl`, and `DispatchBackend` carry public inherent `add`,
  `matmul`, `conv2d` and siblings that take runtime dimensions, mint no
  descriptor, and consult no capability table. The CPU backend has none; it
  was contracted already, which is part of why it is the complete one. Do not
  build on these: they are slated to become crate-private. Use the descriptor
  path described in [The target API and canonical
  dispatch](./target_api.md).
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
- **Only Apple Silicon is covered by the scheduled hardware matrix.** The
  weekly `hardware.yml` run has a registered macOS runner, so the aarch64 CPU
  path is exercised on real hardware. The CUDA and WGPU native-adapter jobs
  resolve their runners from repository variables that are currently unset,
  and report themselves skipped rather than queueing forever. So the native
  CUDA backend is compile-checked in CI and has no automated execution
  coverage on an NVIDIA device, and WGPU's execution coverage comes from a
  software adapter. This is the mechanism behind [Backends](./backends.md)
  calling CPU the only backend verified by execution.

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
