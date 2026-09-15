# Prototyping status

This is a capability report for researchers deciding whether current Incin is
useful as a model-prototyping base. It records what is executable today and
what is blocked by a missing operation or layer, rather than treating a type
that can be named as a working implementation.

## Working foundation

- Static, dynamic, partially dynamic, and named-axis tensor shapes compile in
  the same model.
- CPU `Linear`, activations, losses, `RNN`, `DataLoader`, `AdamW`, autograd,
  and visitor-backed state paths are available and covered by examples/tests.
- The repository examples compile with `cargo check -p incin --examples`.
- The Book’s current snippets are Cargo-doctested with:
  `cargo test -p incin --features 'backend-authoring' --doc`.

## Transformer / attention assessment

The repository now has a focused CPU Transformer proof in
`crates/incin/tests/transformer_block.rs`. It runs a four-token, single-head
self-attention block with query/key/value projections, transposed key, scaled
QK scores, softmax, attention-value multiplication, output projection,
residual addition, a GELU feed-forward block, backward propagation, AdamW,
and a typed state snapshot round-trip. A companion test and the
`compile_fixture_transformer_static.rs`, `compile_fixture_transformer_mixed.rs`,
and `compile_fixture_transformer_dyn.rs` examples cover static, partially
dynamic, and runtime-heavy shape forms.

The compile baseline in `docs/benchmarks/compile-2026-08-15.md` records the
script's one-clean-before-sequential-cases sample plus its incremental check
for this proof alongside tiny tensor, MLP, and CNN fixtures. Use
`CLEAN_EACH=1 tools/bench-compile.sh` when each case must start from a clean
`incin` package build. The test observes finite output and nonzero gradients for the query,
key, value, output, and both feed-forward projection groups.

That fixture is no longer the whole story. `incin::nn` now carries
`MultiHeadAttention` (with grouped-query attention and rotary positions),
`FeedForward`, and the `TransformerEncoderLayer`/`TransformerDecoderLayer`
pair, and `crates/incin/tests/gpt_decoder_model.rs` trains a decoder-only
model end to end on CPU. Causal masking, multi-head reshaping and dropout are
covered there and in `crates/incin/tests/transformer_layers.rs`, which checks
the module's attention against the hand-composed block above.

What still must not be inferred from any of it is portable accelerator
execution: all of this is CPU-verified only. Cross-attention and a fused
attention kernel are also absent, and the layers are written against `Dyn`
rather than static shapes.

## Modern training

AdamW and checkpoint/state contracts are available independently of the
transformer layers. Learning-rate scheduling and gradient accumulation
are not presented as stable first-class APIs in this snapshot; users can
express a small manual loop, but those paths need dedicated fixtures before
being promoted as framework guarantees.
