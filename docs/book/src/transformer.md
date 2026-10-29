# Transformers

`incin::nn` provides the pieces a decoder-only or encoder-only transformer is
made of, so building one is composition rather than reassembly from `matmul`,
`transpose` and `softmax`:

| Module | What it is |
|---|---|
| `MultiHeadAttention` | Four projections, optional rotary positions, optional causal mask. Grouped-query and multi-query attention are the `N_KV_HEADS` const parameter, not separate modules. |
| `CrossAttention` | The same four projections over two inputs: queries from one sequence, keys and values from an encoder's memory (issue #101). `forward` takes the tuple `(query, memory)`. |
| `FeedForward` | The position-wise half: `Relu`, `Gelu`, or the gated `SwiGlu`. |
| `TransformerEncoderLayer` | Self-attention and feed-forward, each behind a residual and a `LayerNorm`. Every position sees every other. |
| `TransformerDecoderLayer` | The same layer, masked: a position sees only itself and its predecessors. |

## The direction is a type, not a flag

The two layers are one struct with a direction marker, and their widths are
const parameters (issue #101):

```rust,ignore
pub type TransformerEncoderLayer<
    const D_MODEL: usize,
    const N_HEADS: usize,
    const N_KV_HEADS: usize,
    const D_FF: usize,
    B,
    K = f32,
    Train = Trainable,
> = TransformerLayer<Bidirectional, D_MODEL, N_HEADS, N_KV_HEADS, D_FF, B, K, Train>;
pub type TransformerDecoderLayer<
    const D_MODEL: usize,
    const N_HEADS: usize,
    const N_KV_HEADS: usize,
    const D_FF: usize,
    B,
    K = f32,
    Train = Trainable,
> = TransformerLayer<Causal, D_MODEL, N_HEADS, N_KV_HEADS, D_FF, B, K, Train>;
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

## Grouped-query attention is a const parameter

```rust,ignore
// Eight query heads over two key/value heads: grouped-query attention.
// N_KV_HEADS == N_HEADS is ordinary multi-head; N_KV_HEADS == 1 is multi-query.
MultiHeadAttention::<512, 8, 2, Cpu>::build(AttentionConfig::causal(), (), ())?
```

Key and value project to `N_KV_HEADS * head_dim` rather than to `D_MODEL`,
which is the point: with two key/value heads out of eight, those projections
and the cache they will feed are a quarter the size.

## Head configuration is compile-time (issue #101)

`D_MODEL`, `N_HEADS`, `N_KV_HEADS` and `D_FF` are const parameters, so both
head invariants — `D_MODEL` divisible by `N_HEADS`, and `N_HEADS` divisible by
`N_KV_HEADS` — are `const { assert!(..) }` inside `build`: a mismatched
configuration fails at compile time at the construction site, named in the
error ("d_model must be divisible by n_heads", "n_heads must be divisible by
n_kv_heads"). The compile-fail fixtures
`crates/incin-core/tests/compile_fail/attention_d_model_head_mismatch.rs` and
`attention_head_kv_mismatch.rs` pin that, so a regression to a runtime check
breaks the compile baseline.

The tensor *shapes* are the part that stays dynamic: both layers are written
against `Dyn`, because the causal mask needs the `[T, T]` mask and the
`[B, H, T, T]` scores to meet and that pairing still runs through `Dyn`. The
one head property that cannot be const-proven — rotary needs an even head
width, and the table extent depends on the runtime config — remains a build
error naming the odd `head_dim`.

## Cross-attention takes two inputs (issue #101)

`CrossAttention` is the same four projections with a different dataflow: the
queries come from one `[batch, seq, d_model]` sequence and the keys and
values from another — an encoder's memory — so `forward` takes a tuple and
the two lengths are free to differ:

```rust,ignore
let attention = CrossAttention::<512, 8, 2, Cpu>::build(
    AttentionConfig::default(), (), ())?;
let y = attention.forward((query, memory))?; // [batch, seq_query, 512]
```

The head const-parameters and compile-time invariants are exactly
`MultiHeadAttention`'s. The causal flag, when set, masks by *position*:
query row `i` sees memory columns `0 ..= i`, a rectangular
`[seq_query, seq_memory]` mask rather than the square `[T, T]` one. Memory
that stays fixed across decode steps is projected once with
`prefill_memory` into a `KvCache`, and then read — not appended to — by
`forward_with_cache(step, query_pos, &cache)`, where `query_pos` continues
both the rotary positions and the mask diagonal across chunk boundaries.

The layers above do not embed it: a `TransformerDecoderLayer` stays
single-input because `Sequential` composes single-input modules, so a
cross-attending stack reaches for `CrossAttention` directly.

## Cached decode (issue #104)

`MultiHeadAttention::forward_with_cache` runs one generation step against a
caller-owned `KvCache` — a preallocated `[batch, kv_heads, capacity,
head_dim]` buffer whose type fixes the capacity:

```rust,ignore
let mut cache = KvCache::<s![1, 2, 256, 64], Cpu, f32>::new(())?;
for token in prompt_and_samples {
    let step = embed(token); // [1, 1, d_model], NoGrad
    let logits = model.attention.forward_with_cache(step, &mut cache)?;
}
```

Each call projects only the new tokens, rotates keys and queries at their
absolute positions (`cache.len() ..`), appends the rotated keys/values, and
attends against the full stored prefix with a causal mask sliced to this
chunk. The cache is **not** module state: generation owns it, passes `&mut`,
and `reset()` starts the next sequence without reallocating. An append that
would pass capacity fails with `Error::CacheCapacityExceeded` rather than
growing the buffer.

## What these layers are not

- **Not cross-attending.** They attend to their own input only. For an
  encoder's memory use `CrossAttention` (above), whose `forward` takes the
  tuple `(query, memory)` — `Module` is parameterized by one input, which is
  why cross-attention is a separate module rather than a flag on these.
- **Not a flash kernel.** Evaluation and zero-dropout inference dispatch the
  catalog's `scaled_dot_product_attention` row (one descriptor instead of the
  composed score/softmax/attend chain); training with attention dropout still
  runs the manual path because the fused row has no dropout operand. See
  [What is not finished](./whats_not_finished.md) for what remains of #104.
- **Not fully GPU-proven.** Training anything in this chapter is still
  CPU-first for the complete catalog. WGPU now runs attention end-to-end
  against a CPU twin and has a training smoke with finite non-zero
  projection gradients (`wgpu_attention.rs`, landed with `0623e762`), and
  cross-entropy trains on WGPU too — but CUDA/Metal device runs and the
  full stacked model on a GPU remain compile-gated or host-side. See
  [What is not finished](./whats_not_finished.md).

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
