# Final sweep - 2026-08-19

Scope requested: UX, model creation, tensor/layer APIs, dtypes, devices,
backends and their APIs, operations receiving pre-checked validation/metadata
rather than re-deriving it, model import/export, datasets, and the
Hugging Face Hub. All items below are verified by a passing test run, not
just by compiling; live network verification is noted where it applies.

## Fixed

- **HF Hub model loading was completely broken for its actual use case.**
  `HubRepo::load_safetensors` called `load_safetensors_snapshot`, which
  refuses any safetensors file lacking incin's own `incin.format.version`
  metadata key. Every real Hub file lacks that key - it was never written by
  incin - so downloading and loading *any* real-world model always failed.
  Added `deserialize_snapshot_safetensors_foreign` /
  `load_foreign_safetensors_snapshot` (same parsing, same role-metadata
  lookup, no version gate - matching what `import_model!`'s compile-time
  reader already does for foreign files per `saving_loading.md`), and pointed
  `hub.rs` at it. Verified live against `hf-internal-testing/tiny-random-gpt2`
  on the real Hub (network available in this environment): download, cache
  hit on second call, and safetensors load all succeed end to end.
  `crates/incin-core/src/serialize.rs`, `crates/incin-core/src/nn/save.rs`,
  `crates/incin-data/src/hub.rs`.
- **`HubApi` had no dataset-repo access**, only `.model()`. The underlying
  `hf-hub` client already exposes `.dataset()`; added
  `HubApi::dataset(repo_id)` and `download_dataset()` mirroring the model
  path. Deliberately did not add dataset *loading* (parquet/CSV/etc.) - that
  is new-feature work, not stabilization; `HubRepo::get` already downloads
  any named file from a dataset repo, same as it does for a model repo.
- **`hub.rs` had zero test coverage.** Added offline tests (env-var
  construction, `.model()`/`.dataset()` handle construction) plus one
  `#[ignore]`d live test exercising the real client end to end - see above.
  Combined what would have been two live tests into one: two separate tests
  each mutating the same process-global `INCIN_HUB_CACHE_DIR` race under
  Rust's default parallel test execution; this was caught by actually running
  the ignored test live rather than trusting it once it compiled.
- **`clip_grad_value` did not exist.** `whats_not_finished.md` named it
  explicitly ("No gradient clipping by value"). Added as the per-element
  counterpart to the existing `clip_grad_norm`'s whole-group rescale.
  Deliberately *not* added as a new method on `OptimizerBackend` itself:
  `Execute<op::Clamp>` is CPU-only (CUDA/WGPU/Metal do not implement it, and
  in fact `OptimizerBackend` itself is not currently satisfied by CUDA at all
 - a pre-existing gap, not something this sweep introduced or fixed).
  Adding the bound to `OptimizerBackend` would have broken every backend's
  existing conformance the moment it landed. Instead added a separate
  `ValueClippingBackend<K>: OptimizerBackend<K>` trait with its own blanket
  impl; `clip_grad_value` requires that, `Adam`/`SGD`/`AdamW`/the Trainer
  still only require `OptimizerBackend`. Verified this doesn't regress CUDA/
  WGPU/Metal by force-instantiating `OptimizerBackend` for each backend type
  via a throwaway compile check before and after - same pre-existing failure
  either way, confirming the addition is additive. Re-exported from the
  facade (`incin::optim::clip_grad_value`) alongside `clip_grad_norm`.
- **Dead, redundant code from the previous session's misdiagnosis removed.**
  `map_binary_strided_serial_f32` in `elementwise_kernel.rs` (added to close
  a broadcast-stride AVX2 gap that turned out to already be covered by the
  pre-existing `map_iteration_avx2_f32`, confirmed by that work's own
  benchmark reporting "No change in performance detected") has been deleted;
  `execute_strided_f32` is back to calling `map_iteration_avx2_f32` directly.
  The `broadcast_strided_fast_path_matches_scalar_reference` regression test
  was kept - it exercises the public `execute_binary` entry point, not the
  deleted function, so it remains valid coverage of the AVX2 broadcast path.

## Checked, no action needed

- `docs/book/src/transformer.md` was suspected of contradicting
  `whats_not_finished.md`'s "no transformer/attention modules" claim. Read in
  full: it explicitly documents hand-composition only and says the stable
  surface does not provide a reusable `MultiHeadAttention`. No doc-vs-code
  drift.
- Safetensors/postcard save-load round trip (`incin-core` test suite) and the
  `incin-data` loader/transform/vision suite: all passing, unrelated to this
  sweep's changes, confirmed still green after them.

## Explicitly deferred (named, not silently skipped)

- **CUDA/WGPU/Metal operation breadth** (normalization, loss, embedding,
  dropout - the actual blocker on GPU training per `whats_not_finished.md`).
  Unverifiable in this environment (no accelerator hardware), and writing
  kernels for three backends is new-feature work at a scale well past a
  sweep.
- **`MultiHeadAttention`/`TransformerEncoderLayer`/`GRU` composed modules.**
  Named explicitly in `whats_not_finished.md` as absent. Real, valuable, and
  entirely CPU-testable - but it is new public API surface, not a
  stabilization fix, and the largest single item that was in scope for this
  request. Left as the most concrete next-step recommendation.
- **Distributed execution path.** Planning layer is real and complete per
  the book; there is no execution path. Out of scope for a sweep.
- **The systemic "operations re-derive metadata instead of consuming
  already-validated `TensorMeta`/`LayoutClass`" pattern.** Confirmed real:
  114 call sites of `is_contiguous`/`validated_numel`/`checked_numel`/
  `broadcast_shape` inside `crates/incin-backends/src/cpu/ops/` against 7
  sites anywhere reading `TensorMeta`/`LayoutClass`/`ShapeEvidence` directly
  (all 7 in `canonical.rs`, none in the kernel files themselves). This is the
  literal gap the user's "operations should receive pre-checked information"
  request describes, and it is also independently documented as a tracked
  debt item in `gen-ledger.py`'s own deviation notes for `EXE-007`/`EXE-009`.
  Both of those rows are marked complete now, so the blocker they cited is
  resolved - but threading validated layout/contiguity metadata through 114
  call sites across `reduce.rs`, `elementwise.rs`, `elementwise_kernel.rs`,
  `conv.rs`, `pool.rs`, `norm.rs`, `matmul.rs`, and `shape_ops.rs` is a
  correctness-sensitive, multi-file refactor on its own scale, not a sweep
  item. Some re-checking here is deliberate defense-in-depth (see
  `canonical.rs`'s `admitted()`: "a backend that only refuses when its caller
  remembers to ask is a backend whose capability output is advisory") and
  should not all be removed - only the shape/layout re-derivation that exists
  purely for dispatch, not for safety, is the actual target.

## Verification run

`cargo test -p incin-core --features std`, `cargo test -p incin --features
cpu` (full suite including doctests and book_docs compile tests), `cargo
test -p incin-data --features hub` (including the live network test, run
once explicitly with `--ignored`), `cargo test -p incin-backends --lib`
(elementwise_kernel/reduce/simd modules) - all passing, zero failures.
`cargo fmt --check` clean. `cargo clippy --all-targets -D warnings` clean for
every touched crate. `cargo check` clean (warnings-only, pre-existing) for
`cuda`, `wgpu`, and `metal` feature sets, confirming none of the above broke
non-CPU backend compilation.

Governance note, unrelated to this sweep and not touched here: `tools/
gen-ledger.py` (the documented single source of truth for the original
100-task ledger) is stale relative to `docs/plan/ledger.toml`/`PROPOSALS.md`'s
table - 494 lines of diff between what it generates and what's committed,
including several rows it still marks incomplete that the committed mirror
and the actual source both show as done. Flagging per prior session's
finding; not fixed here since it's bookkeeping, not implementation.
