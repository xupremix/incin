# What's not finished yet

Kept separate from the rest of the book so it can be updated independently,
and so nothing above has to hedge every sentence. Everything here was
verified directly against the source, not inferred from documentation  -
where a claim depends on something more likely to drift (operation counts,
op tables), it points at the generated document that stays current instead
of repeating a number that won't.

## Blocks real usage today

- **GPU training.** See [Backends](./backends.md) - the previews cover
  basic arithmetic, reductions, `matmul`, and `conv2d`/pooling, and WGPU adds
  thirteen unary activations. None of them has normalization, a loss
  function, `embedding`, or `dropout`, so training anything in this book's
  [Building models](./building_models.md) chapter is CPU-only right now.
- **No transformer/attention modules.** The raw
  `scaled_dot_product_attention` operation exists; there is no
  `MultiHeadAttention` or `TransformerEncoderLayer` composed module, and no
  `GRU` alongside `LSTM`/`RNN`. Building a transformer means hand-composing
  it from primitives.

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
  descriptor, and consult no capability table. The CPU backend has none - it
  was contracted already, which is part of why it is the complete one. Do not
  build on these: they are slated to become crate-private. Use the descriptor
  path described in [The target API and canonical
  dispatch](./target_api.md).
- **Distributed training** (`FSDP`, tensor/pipeline parallelism) has a
  complete planning layer behind the `distributed` feature but no execution
  path yet - a design surface, not a training feature to reach for.
- **The automatic `Trainer`** (`incin::experimental::training`, `train`
  feature) has a real single-device training loop (`fit`), but explicitly
  refuses a multi-device plan (`TrainError::CollectivesUnavailable`) rather
  than doing something wrong.
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

- `docs/capabilities.md` - exactly which operations each backend supports,
  for which dtypes, regenerated from the actual registrations.
- `docs/OPERATION_SEMANTICS.md` - the full semantic contract (broadcasting,
  dtype, gradient, output rules) for every catalog operation.
- `audit-evidence/FND-005/cpu-migration-status.md` - canonical-path
  migration status, machine-checked against source on every test run.

If any claim in this book ever disagrees with one of those three, the
generated document is right and this book is stale - please file it as
such.
