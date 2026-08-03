# FND-002 — DONE

The FND-002 acceptance gate passes at the recorded pre-commit checkout
`e906a5b4a9128788b7122d621c53734bee55774d` with the changes in this task's
diff.

## Frozen contract

- `Dyn` is a freely constructible unit zero-sized marker.
- Checked element counts and byte lengths have private payloads, named checked
  constructors, and read-only accessors.
- Opaque IDs, compiled buffer slots, proof-bearing metadata, gradients, dataset
  state, and runtime accelerator selectors cannot be forged through public
  fields.
- Runtime accelerator selectors describe only a requested ordinal; construction
  does not claim hardware availability.
- Tensor metadata is created only through checked constructors and is exposed
  through a shared read-only view.
- Deserialization of tuning cache records, liveness intervals, and memory plans
  reruns the same validation required by ordinary construction.
- Element counts, byte lengths, strides, spans, offsets, slicing, reshape,
  concat/stack, allocation lengths, model dimensions, dataset dimensions, and
  accelerator launch dimensions use checked arithmetic at their trust or
  allocation boundaries.

## Acceptance evidence

The archived final runs pass the resolved feature matrix, exact workspace
Clippy gate, core and CPU package suites, the full workspace suite, isolated
compile-pass/compile-fail contracts, serialization and arithmetic boundary
tests, workspace doctests, compiled-feature doctests, rustdoc warnings, focused
formatting, and diff hygiene. Initial failing lint/test runs are retained in the
log and were followed by passing reruns; they are not presented as successes.

The workspace-wide `cargo fmt --all -- --check` command still exits 1 because
of formatting drift outside this task's Rust diff. The focused changed-file
rustfmt check and `git diff --check` both exit 0.

No accelerator hardware execution claim is made. CUDA, Metal, WGPU, and Candle
feature compilation was checked independently; the software-only workspace
suite exercised the default enabled paths.

