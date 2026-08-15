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
  `cargo test -p incin --features 'target-api backend-authoring' --doc`.

## Transformer / attention assessment

A complete Transformer proof is not currently claimed. The framework has the
tensor primitives needed for pieces such as matmul, reshape, transpose,
softmax, masking, residual addition, and feed-forward layers, but there is no
stable public multi-head-attention or Transformer module. Backend capability
coverage also differs: the accelerator backends do not currently provide the
full activation, normalization, loss, and dropout set used by a trainable
Transformer. Consequently a CPU-only hand composition may demonstrate pieces,
but it is not yet a portable modern-architecture prototype with a validated
backward path.

The honest next milestone is a focused attention operation/module fixture that
proves causal masking, head reshaping, residuals, normalization, dropout, and
backward together. Until that exists, examples must not label a partial
matmul/reshape composition as a Transformer implementation.

## Modern training

AdamW and checkpoint/state contracts are available independently of the
missing attention module. Learning-rate scheduling and gradient accumulation
are not presented as stable first-class APIs in this snapshot; users can
express a small manual loop, but those paths need dedicated fixtures before
being promoted as framework guarantees.
