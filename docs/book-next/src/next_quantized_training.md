# Quantized training: STE today, low precision tomorrow

The [stable quantization chapter](../#/quantization) documents the user
contract: Q8_0 quantization, the admission rules, and what is explicitly
not a goal (GPTQ/AWQ). This chapter is about the training half — how
gradients flow through quantization at all — and the precision formats
lining up behind it.

## The straight-through estimator, concretely

`quantize` rounds to the grid; rounding has a zero derivative almost
everywhere, so true backpropagation would never train. The estimator
replaces the backward rule with the identity: forward computes
`dequantize(quantize(x))`, backward passes the cotangent through unchanged.
Both `quantize` and `dequantize` record one tape entry each on CPU and
CUDA, and `quantized_matmul` has no backward rule anywhere by design (#93).

The pairing to hold in your head is *forward honest, backward pragmatic*.
At the facade level the round-trip is ordinary code — 64 elements make two
full Q8_0 blocks along the last axis, and the `f16` block scale keeps the
result approximate, not exact:

```rust
use incin::prelude::*;

let t = Cpu.full(shape![64], 0.5)?;
let q = t.quantize(-1)?;
let back = q.dequantize::<f32>()?;
let values = back.to_vec1::<f32>()?;
assert!(values.iter().all(|v| (v - 0.5).abs() < 1e-2));
# Ok::<(), incin::Error>(())
```

The gradient half lives one level down, on the canonical dispatch
surface, and is proven in `crates/incin-backends/tests/quantize_ste.rs`:
forward computes `dequantize(quantize(x))`, backward passes the cotangent
through unchanged. One facade sharp edge remains, documented in [What's
not finished](../#/whats_not_finished): a gradient-marked tensor cannot be
quantized in place — gradient tracking admits only floating dtypes, so
`.detach()` comes first and the STE round-trip is a dispatch-surface story
rather than a `Tensor` method chain.

## What comes after Q8_0

NVFP4 and MXFP4 block codecs exist end to end on CPU as logical markers
(`crates/incin-core/src/tensor/dtype/builtin.rs`): the format plumbing is
real, the training story is not yet written. `fp8` scaled round-trips hold
within tolerance. The sequence is deliberate — markers first, codecs
second, STE rules third, facade last — because each step is only
trustworthy with the previous one's tests beneath it.

The boundary: quantized operations are backend-authoring functionality, not
a stable `Tensor` method surface, and straight-through training is claimed
on CPU and CUDA only. Anything else is a plan, tracked under #93/#94/#95.
