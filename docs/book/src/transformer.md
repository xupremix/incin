# A small Transformer-style block

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

The canonical source is included below so the Book, executable example, and
integration proof stay together:

```rust,ignore
{{#include ../../../crates/incin/tests/transformer_block.rs}}
```
