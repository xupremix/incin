# A small Transformer-style block

The executable Transformer proof uses `Param::as_tensor()` and
`Gradients::require(&parameter)` for parameter gradients. It does not reach
into a backend gradient store, so the proof follows the same typed lookup path
available to application code.

The current stable composition surface is enough to write and train a small
CPU self-attention block directly. The executable proof is
[`crates/incin/tests/transformer_block.rs`](../../../crates/incin/tests/transformer_block.rs):
it performs scaled single-head attention, an MLP residual, backward, checks
finite outputs and nonzero gradients for every projection group, takes an
AdamW step, and round-trips the parameter state.

The proof also includes a static-shape attention composition and six compile
fixtures covering tiny tensors, MLPs, CNNs, static shapes, mixed shapes, and
`Dyn`-heavy code. Run the baseline with:

```text
CLEAN=1 tools/bench-compile.sh
```

This is intentionally a small, honest building block. The stable public
surface does not yet provide a reusable multi-head attention module, and this
example does not claim causal masking, normalization, dropout, portable GPU
training, or a complete decoder implementation. Those are separate composition
and backend contracts rather than features hidden in this example.

For the weight-shared variant — one block iterated several times, the
looped-transformer shape — see
[`crates/incin/tests/looped_transformer.rs`](../../../crates/incin/tests/looped_transformer.rs):
same geometry, three iterations over shared parameters, one gradient per
weight accumulated across iterations, and one copy of the weights in the
state snapshot. No custom operation is involved; sharing falls out of the
tape's accumulation rule.

One masking limitation to know about before designing around it: shape
equality is reflexive only, so a `[T, T]` causal mask does not meet `[B, H,
T, T]` scores at the type level even though the backend would broadcast
them. Broadcast the mask explicitly (for example with `broadcast_left` to the
full score shape) before combining it with the scores.

The canonical source is included below so the Book, executable example, and
integration proof stay together:

```rust,ignore
{{#include ../../../crates/incin/tests/transformer_block.rs}}
```
