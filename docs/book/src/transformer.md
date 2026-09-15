# Transformers

`incin::nn` provides the pieces a decoder-only or encoder-only transformer is
made of, so building one is composition rather than reassembly from `matmul`,
`transpose` and `softmax`:

| Module | What it is |
|---|---|
| `MultiHeadAttention` | Four projections, optional rotary positions, optional causal mask. Grouped-query and multi-query attention are the `n_kv_heads` argument, not separate modules. |
| `FeedForward` | The position-wise half: `Relu`, `Gelu`, or the gated `SwiGlu`. |
| `TransformerEncoderLayer` | Self-attention and feed-forward, each behind a residual and a `LayerNorm`. Every position sees every other. |
| `TransformerDecoderLayer` | The same layer, masked: a position sees only itself and its predecessors. |

## The direction is a type, not a flag

The two layers are one struct with a direction marker:

```rust,ignore
pub type TransformerEncoderLayer<B, K = f32, Train = Trainable> =
    TransformerLayer<Bidirectional, B, K, Train>;
pub type TransformerDecoderLayer<B, K = f32, Train = Trainable> =
    TransformerLayer<Causal, B, K, Train>;
```

An encoder layer and a decoder-only layer differ in exactly one bit, so two
structs would be two copies of one forward pass, and a `causal: bool` field
would make masking a value that no signature could require. The marker gives
both: one dataflow, and two types that cannot be substituted for one another.

`build` takes the direction's word for it. Whatever `config.attention.causal`
was set to is overwritten with the marker's constant and the corrected config
is what gets stored, so `layer.config.attention.causal` reads back what the
layer does rather than what was asked for.

## A decoder-only model

The executable proof is
[`crates/incin/tests/gpt_decoder_model.rs`](../../../crates/incin/tests/gpt_decoder_model.rs):
token ids in, logits over the vocabulary out, trained on CPU with AdamW, with
the whole model round-tripping through `collect_state`/`load_state`. It is
included in full at the end of this chapter, so the book, the compiled test
and the proof cannot drift apart.

Two things about its shape are worth knowing before you copy it.

**The blocks are named fields, not an array.** `Sequential` is a *pair*
combinator, `Sequential<L1, L2>`, so a depth-`N` stack is either a
right-nested tower of pairs or one field per block. There is no
`Sequential<[Layer; N]>`, and a `Vec<Layer>` field would not traverse either:
the state and parameter visitors are implemented for `Option<L>` and for
module types, not for collections. One field per block also keeps the
checkpoint paths readable (`block0.attention.query.weight`).

**Rotary positions need no second embedding.** The rotation is applied to
queries and keys inside attention, after the head split, so it cannot be
expressed as a term added to the input the way a learned absolute position
table is. The cosine and sine tables are `Buffer`s rather than `Param`s: they
round-trip through a checkpoint and receive no gradient, which the test
asserts directly.

## Grouped-query attention is an argument

```rust,ignore
// Eight query heads over two key/value heads: grouped-query attention.
// n_kv_heads == n_heads is ordinary multi-head; n_kv_heads == 1 is multi-query.
MultiHeadAttention::<Cpu>::build(512, 8, 2, AttentionConfig::causal(), (), ())?
```

Key and value project to `n_kv_heads * head_dim` rather than to `d_model`,
which is the point: with two key/value heads out of eight, those projections
and the cache they will feed are a quarter the size.

## Shapes are dynamic here

Both layers are written against `Dyn` rather than a static shape, and their
head counts are runtime fields rather than const parameters. The reason is the
causal mask: masking needs the `[T, T]` mask and the `[B, H, T, T]` scores to
meet, and shape equality is reflexive only, so that pairing cannot be stated
through the typed path today. The layers work around it internally; a caller
building a mask by hand still has to broadcast it explicitly to the full score
shape before combining it with the scores.

The two invariants a const parameterization would have proved at compile time
are checked in `build` instead, and reported with the offending numbers named:
`d_model` must divide evenly into `n_heads`, and `n_heads` into `n_kv_heads`.

## What these layers are not

- **Not cross-attending.** They attend to their own input only. A layer that
  also attends to an encoder's output takes two tensors, and `Module` is
  parameterized by one input.
- **Not fused.** Every module composes catalog operations, so it runs on any
  backend advertising them rather than on the subset that has an attention
  kernel. A fused path can be selected underneath the same surface later.
- **Not GPU-verified.** Training anything in this chapter is CPU-only right
  now; see [What is not finished](./whats_not_finished.md).

## The earlier hand-composed proof

[`crates/incin/tests/transformer_block.rs`](../../../crates/incin/tests/transformer_block.rs)
predates these modules and is still in the tree. It hand-assembles a
four-token, single-head block and its own doc comment records what it leaves
out: masking, normalization, dropout and multi-head packing. Those are exactly
what the modules above add, and
[`crates/incin/tests/transformer_layers.rs`](../../../crates/incin/tests/transformer_layers.rs)
checks the module's attention against that hand-written dataflow so the
replacement is proved against something that already worked.

For the weight-shared variant — one block iterated several times, the
looped-transformer shape — see
[`crates/incin/tests/looped_transformer.rs`](../../../crates/incin/tests/looped_transformer.rs):
same geometry, three iterations over shared parameters, one gradient per
weight accumulated across iterations, and one copy of the weights in the state
snapshot. No custom operation is involved; sharing falls out of the tape's
accumulation rule.

The compile baseline covers the transformer fixtures alongside the tiny
tensor, MLP and CNN cases. Run it with:

```text
CLEAN=1 tools/bench-compile.sh
```

## The model, in full

```rust,ignore
{{#include ../../../crates/incin/tests/gpt_decoder_model.rs}}
```
