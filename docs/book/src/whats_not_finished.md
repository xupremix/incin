# What's not finished yet

Kept separate from the rest of the book so it can be updated independently,
and so nothing above has to hedge every sentence. Everything here was
verified directly against the source, not inferred from documentation —
where a claim depends on something more likely to drift (operation counts,
op tables), it points at the generated document that stays current instead
of repeating a number that won't.

## Blocks real usage today

- **GPU training.** See [Backends](./backends.md) — CUDA and WGPU cover
  basic arithmetic, reductions, `matmul`, and `conv2d`/pooling; neither has
  any activation, normalization, loss, `embedding`, or `dropout`. Metal is
  narrower still. Training anything in this book's [Building
  models](./building_models.md) chapter is CPU-only right now.
- **No transformer/attention modules.** The raw
  `scaled_dot_product_attention` operation exists; there is no
  `MultiHeadAttention` or `TransformerEncoderLayer` composed module, and no
  `GRU` alongside `LSTM`/`RNN`. Building a transformer means hand-composing
  it from primitives.
- **No gradient clipping.** Nothing in `incin_core::optim` clips gradients
  by norm or value. Learning rate scheduling is fine (`ConstantLR`,
  `LinearLR`, `CosineAnnealingLR`, `StepLR`) — this is specifically about
  clipping.

## Facade gaps (the functionality exists, but not through `incin`)

- **Scoped gradient policy** is intentionally explicit through
  `incin_core::exec::GradMode::Disabled.scope` and has no facade alias.
- **The lower-level `save_safetensors`/`load_safetensors` helpers** remain
  available under `incin_core::nn::save` for compatibility, while normal
  facade users should use `incin::prelude::{Format, ModelExt}`. The facade
  path currently supports the same typed snapshot contract through
  `ModelExt::save` and `ModelExt::load`.

## Architecture in progress (affects contributors more than users)

- **Backend decomposition is still in progress.** Ordinary tensor operations use
  the per-operation descriptor execution path. The seven remaining broad
  operation-family traits have been removed from production source. Remaining
  work is splitting large backend files and making exceptional execution sites
  that cannot fit `Execute<O>` easier to maintain.
- **Distributed training** (`FSDP`, tensor/pipeline parallelism) has a
  complete planning layer behind the `distributed` feature but no execution
  path yet — a design surface, not a training feature to reach for.
- **The automatic `Trainer`** (`incin::experimental::training`, `train`
  feature) has a real single-device training loop (`fit`), but explicitly
  refuses a multi-device plan (`TrainError::CollectivesUnavailable`) rather
  than doing something wrong.

## Where the current, generated truth lives

- `docs/capabilities.md` — exactly which operations each backend supports,
  for which dtypes, regenerated from the actual registrations.
- `docs/OPERATION_SEMANTICS.md` — the full semantic contract (broadcasting,
  dtype, gradient, output rules) for every catalog operation.
- `audit-evidence/FND-005/cpu-migration-status.md` — canonical-path
  migration status, machine-checked against source on every test run.

If any claim in this book ever disagrees with one of those three, the
generated document is right and this book is stale — please file it as
such.
