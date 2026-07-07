# Kindle Framework Testing Checklist

This document tracks all functionality that needs to be tested across the `kindle` workspace. For every item, we verify its specific parameter permutations, struct implementations, and ensure it is fully documented with usage examples.

## 1. Tensor Operations (`kindle-core/src/tensor/ops/`)

### 1.1 Unary Operations
- `[x]` `abs()`
  - `[x]` Test permutations: positive, negative, zero, very small numbers, very large numbers, NaN, Inf.
  - `[x]` Documentation & Usage Example
- `[x]` `relu()`
  - `[x]` Test permutations: positive (unchanged), negative (zeroed), zero.
  - `[x]` Documentation & Usage Example
- `[x]` `gelu()`
  - `[x]` Test permutations: standard normal values, extreme negatives/positives.
  - `[x]` Documentation & Usage Example
- `[x]` `swish()`
  - `[x]` Test permutations: test curve matching standard beta=1 definitions.
  - `[x]` Documentation & Usage Example
- `[x]` `softmax(dim)`
  - `[x]` Test permutations: dim 0, dim 1, intermediate dims, negative dims, very large/small values (numerical stability).
  - `[x]` Documentation & Usage Example
- `[x]` `neg()`
  - `[x]` Test permutations: zero, positives, negatives.
  - `[x]` Documentation & Usage Example
- `[x]` `sqrt()`
  - `[x]` Test permutations: positive, zero, negative (handling NaN).
  - `[x]` Documentation & Usage Example
- `[x]` `exp()`
  - `[x]` Test permutations: large positive (overflow bounds), zero, negative values.
  - `[x]` Documentation & Usage Example
- `[x]` `log()`
  - `[x]` Test permutations: positive, zero (-Inf), negative (NaN).
  - `[x]` Documentation & Usage Example
- `[x]` `tanh()`
  - `[x]` Test permutations: asymptotes at +1 and -1, near zero values.
  - `[x]` Documentation & Usage Example
- `[x]` `sigmoid()`
  - `[x]` Test permutations: asymptotes at 0 and 1, origin.
  - `[x]` Documentation & Usage Example
- `[x]` `mul_scalar(f64)`
  - `[x]` Test permutations: zero, positive, negative, fractional, large scalars.
  - `[x]` Documentation & Usage Example
- `[x]` `add_scalar(f64)`
  - `[x]` Test permutations: zero, positive, negative, fractional scalars.
  - `[x]` Documentation & Usage Example

### 1.2 Binary Operations
- `[x]` `add(rhs)`
  - `[x]` Test permutations: positive + positive, negative + negative, zeroes.
  - `[x]` Documentation & Usage Example
- `[x]` `sub(rhs)`
  - `[x]` Test permutations: lhs > rhs, lhs < rhs, identical tensors.
  - `[x]` Documentation & Usage Example
- `[x]` `mul(rhs)`
  - `[x]` Test permutations: zeroes, identity matrix (for matmul vs element-wise), negative terms.
  - `[x]` Documentation & Usage Example
- `[x]` `div(rhs)`
  - `[x]` Test permutations: standard division, division by zero, floating point precision limits.
  - `[x]` Documentation & Usage Example

### 1.3 Broadcasting Operations
- `[x]` `broadcast_add(rhs)`, `broadcast_sub(rhs)`, `broadcast_mul(rhs)`, `broadcast_div(rhs)`
  - `[x]` Test permutations: matching shapes, differing trailing dims, broadcast scalar to tensor, broadcast vector to matrix, multi-dim broadcasting.
  - `[x]` Documentation & Usage Example
- `[x]` Standard ops delegation (`core::ops::Add`, `Sub`, `Mul`, `Div`)
  - `[x]` Test permutations: operator overloading invokes correct broadcast traits.
  - `[x]` Documentation & Usage Example

### 1.4 Reduction Operations
- `[x]` `sum_all()`, `mean_all()`, `max_all()`, `min_all()`
  - `[x]` Test permutations: large tensors, 1D tensors, zero-filled tensors, tensors with NaNs.
  - `[x]` Documentation & Usage Example
- `[x]` `sum_dim(dim)`, `sum_keepdim(dim)`
  - `[x]` Test permutations: first dim, last dim, intermediate dim, negative dims.
  - `[x]` Documentation & Usage Example
- `[x]` `mean_dim(dim)`, `mean_keepdim(dim)`
  - `[x]` Test permutations: first dim, last dim, intermediate dim, negative dims.
  - `[x]` Documentation & Usage Example
- `[x]` `max_dim(dim)`, `max_keepdim(dim)`
  - `[x]` Test permutations: all dims, duplicate maximums.
  - `[x]` Documentation & Usage Example
- `[x]` `min_dim(dim)`, `min_keepdim(dim)`
  - `[x]` Test permutations: all dims, duplicate minimums.
  - `[x]` Documentation & Usage Example

### 1.5 Manipulation Operations
- `[x]` `reshape(new_shape)`
  - `[x]` Test permutations: valid reshape, flat to multi-dim, multi-dim to flat.
  - `[x]` Documentation & Usage Example
- `[x]` `transpose(dim0, dim1)`
  - `[x]` Test permutations: adjacent dims, non-adjacent dims, reversing tensor.
  - `[x]` Documentation & Usage Example
- `[x]` `flatten(start, end)`
  - `[x]` Test permutations: full flatten, partial flatten (e.g., preserving batch dim).
  - `[x]` Documentation & Usage Example
- `[x]` `narrow(dim, start, length)`
  - `[x]` Test permutations: start=0, middle extraction, extract to end, out-of-bounds safety.
  - `[x]` Documentation & Usage Example
- `[x]` `squeeze(dim)`
  - `[x]` Test permutations: single dim=1, multiple dim=1, attempting to squeeze dim>1 (should fail/noop).
  - `[x]` Documentation & Usage Example
- `[x]` `broadcast_as(shape)`, `broadcast_left(shape)`
  - `[x]` Test permutations: up-casting 1D to 3D, broadcasting inner dimensions.
  - `[x]` Documentation & Usage Example

### 1.6 Indexing & Slicing
- `[x]` `dyn_slice()`
  - `[x]` Test permutations: exact indices (`1`), ranges (`1..3`), inferred spans (`..`), ellipses (`...`).
  - `[x]` Documentation & Usage Example
- `[x]` `concat(rhs, axis)` / `try_concat(rhs, dim)`
  - `[x]` Test permutations: concatenating identical shapes, concatenating mismatched sizes on concatenation axis, zero-length concats.
  - `[x]` Documentation & Usage Example
- `[x]` `stack(rhs, axis)` / `try_stack(rhs, dim)`
  - `[x]` Test permutations: all dims, pushing new outer dims, pushing new inner dims.
  - `[x]` Documentation & Usage Example

### 1.7 Loss Functions
- `[x]` `mse_loss(pred, target, reduction)`
  - `[x]` Test permutations: Reduction::Mean, Sum, None. Perfect match (zero loss), extreme differences.
  - `[x]` Documentation & Usage Example
- `[x]` `cross_entropy_loss(pred, target, reduction)`
  - `[x]` Test permutations: Reduction::Mean, Sum, None. 1D targets (indices), 2D targets (probabilities).
  - `[x]` Documentation & Usage Example

---

## 2. Neural Network Modules & Structs (`kindle-core/src/nn/`)

For every struct, we must test its constructor, functional implementation, parameter permutations, and trait implementations.

### 2.1 Struct: `Param<S, B>`
- `[x]` `zeros()`, `ones()`, `rand()`, `randn()`
  - `[x]` Test permutations: different static shapes, extreme shapes (e.g. 1x1).
- `[x]` `as_tensor()`, `from_tensor()`
  - `[x]` Test permutations: mapping back and forth cleanly.
- `[x]` Documentation & Usage Example

### 2.2 Struct: `Linear`
- `[x]` Constructor `new(in_dim, out_dim)`
- `[x]` `forward(x)`
  - `[x]` Test permutations: `has_bias=true`, `has_bias=false`, 1D input, 2D input (batched), 3D input (sequence-batched).
- `[x]` `state_dict()`, `parameters()`
- `[x]` Documentation & Usage Example

### 2.3 Struct: `Conv2d` / `ConvTranspose2d`
- `[x]` `forward(x)`
  - `[x]` Test permutations: `stride=(1,1)`, `stride=(2,2)`, `padding=(0,0)`, `padding=(1,1)`, `dilation>1`, `groups=in_channels` (depthwise).
  - `[x]` Test permutations (ConvTranspose2d): `output_padding>0`.
- `[x]` `state_dict()`, `parameters()`
- `[x]` Documentation & Usage Example

### 2.4 Struct: `Embedding`
- `[x]` `forward(indices)`
  - `[x]` Test permutations: valid indices, out-of-bounds indices, 1D vs batched 2D index inputs.
- `[x]` Documentation & Usage Example

### 2.5 Struct: `LayerNorm` / `BatchNorm2d`
- `[x]` `forward(x)`
  - `[x]` Test permutations: varying `eps` values, training mode (updates running stats), eval mode (uses running stats).
- `[x]` Documentation & Usage Example

### 2.6 Struct: `Sequential`
- `[x]` `forward(x)`
  - `[x]` Test permutations: empty sequence, single layer, deeply nested sequences.
- `[x]` Documentation & Usage Example

---

## 3. Optimizers (`kindle-core/src/optim/`)

### 3.1 Struct: `SGD`
- `[x]` `step(grads)`
  - `[x]` Test permutations: varied learning rates (0.1, 0.001), 0.0 learning rate (no-op).
- `[x]` Documentation & Usage Example

### 3.2 Struct: `Adam`
- `[x]` `step(grads)`
  - `[x]` Test permutations: extreme beta1/beta2 values, varied `eps`, repeated steps (momentum scaling over time).
- `[x]` Documentation & Usage Example

### 3.3 Struct: `AdamW`
- `[x]` `step(grads)`
  - `[x]` Test permutations: `weight_decay=0.0`, `weight_decay>0.0` (ensure isolated decay logic vs Adam).
- `[x]` Documentation & Usage Example

---

## 4. Procedural Macros (`kindle-macros`)

### 4.1 `s![]`
- `[x]` Test permutations: integer literals (`s![1, 3, 224, 224]`), named dimension symbols (`s![Batch, Channels]`), mixed parameters, `dyn` dimension definitions.
- `[x]` Error handling: test syntax errors trigger `compile_fail`.
- `[x]` Documentation & Usage Example

### 4.2 `idx![]`
- `[x]` Test permutations: single index (`5`), range (`2..5`), range-from (`2..`), range-to (`..5`), full range (`..`), ellipsis (`...`), inferred dimension (`-1`). Combinations of the above.
- `[x]` Error handling: test out-of-bounds or non-parseable structures trigger `compile_fail`.
- `[x]` Documentation & Usage Example

### 4.3 `#[module]`
- `[x]` Test permutations: structs with primitive fields (f64), structs with Param/Tensors, structs with nested modules, generic bounds handling, `#[kindle(skip)]` parameter skips.
- `[x]` Documentation & Usage Example

### 4.4 `import_model!()`
- `[x]` Test permutations: basic linear model, convolutional network, varying ONNX opset versions.
- `[x]` Documentation & Usage Example

---

## 5. Backends (`kindle-backends`)

- `[x]` `CandleBackend`: Execute full permutations matrix on CPU.
- `[x]` `NdarrayBackend`: Execute full permutations matrix on CPU.
- `[x]` Device movement: Ensure `to_device(Cpu)` -> `to_device(Cuda)` permutations properly transfer nested structures.
- `[x]` Documentation & Usage Example

---

## 6. End-to-End Tracing / Autograd
- `[x]` Backward Pass: Test scalar loss tracking through unary/binary/reduction permutations.
- `[x]` Tape Memory: Validate graph construction constraints.
- `[x]` Documentation & Usage Example

---

## 7. Data Pipeline (`kindle-data`)

### 7.1 Datasets
- `[x]` `Dataset` Trait Implementation
  - `[x]` Test permutations: in-memory datasets, streaming/lazy datasets, varying tensor shapes per item.
  - `[x]` Documentation & Usage Example

### 7.2 DataLoaders & Samplers
- `[x]` `DataLoader` Batching & Iteration
  - `[x]` Test permutations: `batch_size=1`, `batch_size>1`, `drop_last=true`, `drop_last=false`.
  - `[x]` Documentation & Usage Example
- `[x]` Samplers
  - `[x]` Test permutations: sequential sampling (ordered), random sampling (shuffled indices), distributed sampling (mocked partitions).
  - `[x]` Documentation & Usage Example

---

## 8. Serialization (SafeTensors)

- `[x]` Model Weight Loading & Saving
  - `[x]` Test permutations: saving a complex `#[module]` to disk and loading it back, verifying weights match exactly.
  - `[x]` Error handling: attempting to load a Safetensors file with mismatched tensor shapes or missing keys.
  - `[x]` Documentation & Usage Example
