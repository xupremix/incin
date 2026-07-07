# Kindle Framework Testing Checklist

This document tracks all functionality that needs to be tested across the `kindle` workspace. For every item, we verify its specific parameter permutations, struct implementations, and ensure it is fully documented with usage examples.

## 1. Tensor Operations (`kindle-core/src/tensor/ops/`)

### 1.1 Unary Operations
- `[ ]` `abs()`
  - `[ ]` Test permutations: positive, negative, zero, very small numbers, very large numbers, NaN, Inf.
  - `[ ]` Documentation & Usage Example
- `[ ]` `relu()`
  - `[ ]` Test permutations: positive (unchanged), negative (zeroed), zero.
  - `[ ]` Documentation & Usage Example
- `[ ]` `gelu()`
  - `[ ]` Test permutations: standard normal values, extreme negatives/positives.
  - `[ ]` Documentation & Usage Example
- `[ ]` `swish()`
  - `[ ]` Test permutations: test curve matching standard beta=1 definitions.
  - `[ ]` Documentation & Usage Example
- `[ ]` `softmax(dim)`
  - `[ ]` Test permutations: dim 0, dim 1, intermediate dims, negative dims, very large/small values (numerical stability).
  - `[ ]` Documentation & Usage Example
- `[ ]` `neg()`
  - `[ ]` Test permutations: zero, positives, negatives.
  - `[ ]` Documentation & Usage Example
- `[ ]` `sqrt()`
  - `[ ]` Test permutations: positive, zero, negative (handling NaN).
  - `[ ]` Documentation & Usage Example
- `[ ]` `exp()`
  - `[ ]` Test permutations: large positive (overflow bounds), zero, negative values.
  - `[ ]` Documentation & Usage Example
- `[ ]` `log()`
  - `[ ]` Test permutations: positive, zero (-Inf), negative (NaN).
  - `[ ]` Documentation & Usage Example
- `[ ]` `tanh()`
  - `[ ]` Test permutations: asymptotes at +1 and -1, near zero values.
  - `[ ]` Documentation & Usage Example
- `[ ]` `sigmoid()`
  - `[ ]` Test permutations: asymptotes at 0 and 1, origin.
  - `[ ]` Documentation & Usage Example
- `[ ]` `mul_scalar(f64)`
  - `[ ]` Test permutations: zero, positive, negative, fractional, large scalars.
  - `[ ]` Documentation & Usage Example
- `[ ]` `add_scalar(f64)`
  - `[ ]` Test permutations: zero, positive, negative, fractional scalars.
  - `[ ]` Documentation & Usage Example

### 1.2 Binary Operations
- `[ ]` `add(rhs)`
  - `[ ]` Test permutations: positive + positive, negative + negative, zeroes.
  - `[ ]` Documentation & Usage Example
- `[ ]` `sub(rhs)`
  - `[ ]` Test permutations: lhs > rhs, lhs < rhs, identical tensors.
  - `[ ]` Documentation & Usage Example
- `[ ]` `mul(rhs)`
  - `[ ]` Test permutations: zeroes, identity matrix (for matmul vs element-wise), negative terms.
  - `[ ]` Documentation & Usage Example
- `[ ]` `div(rhs)`
  - `[ ]` Test permutations: standard division, division by zero, floating point precision limits.
  - `[ ]` Documentation & Usage Example

### 1.3 Broadcasting Operations
- `[ ]` `broadcast_add(rhs)`, `broadcast_sub(rhs)`, `broadcast_mul(rhs)`, `broadcast_div(rhs)`
  - `[ ]` Test permutations: matching shapes, differing trailing dims, broadcast scalar to tensor, broadcast vector to matrix, multi-dim broadcasting.
  - `[ ]` Documentation & Usage Example
- `[ ]` Standard ops delegation (`core::ops::Add`, `Sub`, `Mul`, `Div`)
  - `[ ]` Test permutations: operator overloading invokes correct broadcast traits.
  - `[ ]` Documentation & Usage Example

### 1.4 Reduction Operations
- `[ ]` `sum_all()`, `mean_all()`, `max_all()`, `min_all()`
  - `[ ]` Test permutations: large tensors, 1D tensors, zero-filled tensors, tensors with NaNs.
  - `[ ]` Documentation & Usage Example
- `[ ]` `sum_dim(dim)`, `sum_keepdim(dim)`
  - `[ ]` Test permutations: first dim, last dim, intermediate dim, negative dims.
  - `[ ]` Documentation & Usage Example
- `[ ]` `mean_dim(dim)`, `mean_keepdim(dim)`
  - `[ ]` Test permutations: first dim, last dim, intermediate dim, negative dims.
  - `[ ]` Documentation & Usage Example
- `[ ]` `max_dim(dim)`, `max_keepdim(dim)`
  - `[ ]` Test permutations: all dims, duplicate maximums.
  - `[ ]` Documentation & Usage Example
- `[ ]` `min_dim(dim)`, `min_keepdim(dim)`
  - `[ ]` Test permutations: all dims, duplicate minimums.
  - `[ ]` Documentation & Usage Example

### 1.5 Manipulation Operations
- `[ ]` `reshape(new_shape)`
  - `[ ]` Test permutations: valid reshape, flat to multi-dim, multi-dim to flat.
  - `[ ]` Documentation & Usage Example
- `[ ]` `transpose(dim0, dim1)`
  - `[ ]` Test permutations: adjacent dims, non-adjacent dims, reversing tensor.
  - `[ ]` Documentation & Usage Example
- `[ ]` `flatten(start, end)`
  - `[ ]` Test permutations: full flatten, partial flatten (e.g., preserving batch dim).
  - `[ ]` Documentation & Usage Example
- `[ ]` `narrow(dim, start, length)`
  - `[ ]` Test permutations: start=0, middle extraction, extract to end, out-of-bounds safety.
  - `[ ]` Documentation & Usage Example
- `[ ]` `squeeze(dim)`
  - `[ ]` Test permutations: single dim=1, multiple dim=1, attempting to squeeze dim>1 (should fail/noop).
  - `[ ]` Documentation & Usage Example
- `[ ]` `broadcast_as(shape)`, `broadcast_left(shape)`
  - `[ ]` Test permutations: up-casting 1D to 3D, broadcasting inner dimensions.
  - `[ ]` Documentation & Usage Example

### 1.6 Indexing & Slicing
- `[ ]` `dyn_slice()`
  - `[ ]` Test permutations: exact indices (`1`), ranges (`1..3`), inferred spans (`..`), ellipses (`...`).
  - `[ ]` Documentation & Usage Example
- `[ ]` `concat(rhs, axis)` / `try_concat(rhs, dim)`
  - `[ ]` Test permutations: concatenating identical shapes, concatenating mismatched sizes on concatenation axis, zero-length concats.
  - `[ ]` Documentation & Usage Example
- `[ ]` `stack(rhs, axis)` / `try_stack(rhs, dim)`
  - `[ ]` Test permutations: all dims, pushing new outer dims, pushing new inner dims.
  - `[ ]` Documentation & Usage Example

### 1.7 Loss Functions
- `[ ]` `mse_loss(pred, target, reduction)`
  - `[ ]` Test permutations: Reduction::Mean, Sum, None. Perfect match (zero loss), extreme differences.
  - `[ ]` Documentation & Usage Example
- `[ ]` `cross_entropy_loss(pred, target, reduction)`
  - `[ ]` Test permutations: Reduction::Mean, Sum, None. 1D targets (indices), 2D targets (probabilities).
  - `[ ]` Documentation & Usage Example

---

## 2. Neural Network Modules & Structs (`kindle-core/src/nn/`)

For every struct, we must test its constructor, functional implementation, parameter permutations, and trait implementations.

### 2.1 Struct: `Param<S, B>`
- `[ ]` `zeros()`, `ones()`, `rand()`, `randn()`
  - `[ ]` Test permutations: different static shapes, extreme shapes (e.g. 1x1).
- `[ ]` `as_tensor()`, `from_tensor()`
  - `[ ]` Test permutations: mapping back and forth cleanly.
- `[ ]` Documentation & Usage Example

### 2.2 Struct: `Linear`
- `[ ]` Constructor `new(in_dim, out_dim)`
- `[ ]` `forward(x)`
  - `[ ]` Test permutations: `has_bias=true`, `has_bias=false`, 1D input, 2D input (batched), 3D input (sequence-batched).
- `[ ]` `state_dict()`, `parameters()`
- `[ ]` Documentation & Usage Example

### 2.3 Struct: `Conv2d` / `ConvTranspose2d`
- `[ ]` `forward(x)`
  - `[ ]` Test permutations: `stride=(1,1)`, `stride=(2,2)`, `padding=(0,0)`, `padding=(1,1)`, `dilation>1`, `groups=in_channels` (depthwise).
  - `[ ]` Test permutations (ConvTranspose2d): `output_padding>0`.
- `[ ]` `state_dict()`, `parameters()`
- `[ ]` Documentation & Usage Example

### 2.4 Struct: `Embedding`
- `[ ]` `forward(indices)`
  - `[ ]` Test permutations: valid indices, out-of-bounds indices, 1D vs batched 2D index inputs.
- `[ ]` Documentation & Usage Example

### 2.5 Struct: `LayerNorm` / `BatchNorm2d`
- `[ ]` `forward(x)`
  - `[ ]` Test permutations: varying `eps` values, training mode (updates running stats), eval mode (uses running stats).
- `[ ]` Documentation & Usage Example

### 2.6 Struct: `Sequential`
- `[ ]` `forward(x)`
  - `[ ]` Test permutations: empty sequence, single layer, deeply nested sequences.
- `[ ]` Documentation & Usage Example

---

## 3. Optimizers (`kindle-core/src/optim/`)

### 3.1 Struct: `SGD`
- `[ ]` `step(grads)`
  - `[ ]` Test permutations: varied learning rates (0.1, 0.001), 0.0 learning rate (no-op).
- `[ ]` Documentation & Usage Example

### 3.2 Struct: `Adam`
- `[ ]` `step(grads)`
  - `[ ]` Test permutations: extreme beta1/beta2 values, varied `eps`, repeated steps (momentum scaling over time).
- `[ ]` Documentation & Usage Example

### 3.3 Struct: `AdamW`
- `[ ]` `step(grads)`
  - `[ ]` Test permutations: `weight_decay=0.0`, `weight_decay>0.0` (ensure isolated decay logic vs Adam).
- `[ ]` Documentation & Usage Example

---

## 4. Procedural Macros (`kindle-macros`)

### 4.1 `s![]`
- `[ ]` Test permutations: integer literals (`s![1, 3, 224, 224]`), named dimension symbols (`s![Batch, Channels]`), mixed parameters, `dyn` dimension definitions.
- `[ ]` Error handling: test syntax errors trigger `compile_fail`.
- `[ ]` Documentation & Usage Example

### 4.2 `idx![]`
- `[ ]` Test permutations: single index (`5`), range (`2..5`), range-from (`2..`), range-to (`..5`), full range (`..`), ellipsis (`...`), inferred dimension (`-1`). Combinations of the above.
- `[ ]` Error handling: test out-of-bounds or non-parseable structures trigger `compile_fail`.
- `[ ]` Documentation & Usage Example

### 4.3 `#[module]`
- `[ ]` Test permutations: structs with primitive fields (f64), structs with Param/Tensors, structs with nested modules, generic bounds handling, `#[kindle(skip)]` parameter skips.
- `[ ]` Documentation & Usage Example

### 4.4 `import_model!()`
- `[ ]` Test permutations: basic linear model, convolutional network, varying ONNX opset versions.
- `[ ]` Documentation & Usage Example

---

## 5. Backends (`kindle-backends`)

- `[ ]` `CandleBackend`: Execute full permutations matrix on CPU.
- `[ ]` `NdarrayBackend`: Execute full permutations matrix on CPU.
- `[ ]` Device movement: Ensure `to_device(Cpu)` -> `to_device(Cuda)` permutations properly transfer nested structures.
- `[ ]` Documentation & Usage Example

---

## 6. End-to-End Tracing / Autograd
- `[ ]` Backward Pass: Test scalar loss tracking through unary/binary/reduction permutations.
- `[ ]` Tape Memory: Validate graph construction constraints.
- `[ ]` Documentation & Usage Example

---

## 7. Data Pipeline (`kindle-data`)

### 7.1 Datasets
- `[ ]` `Dataset` Trait Implementation
  - `[ ]` Test permutations: in-memory datasets, streaming/lazy datasets, varying tensor shapes per item.
  - `[ ]` Documentation & Usage Example

### 7.2 DataLoaders & Samplers
- `[ ]` `DataLoader` Batching & Iteration
  - `[ ]` Test permutations: `batch_size=1`, `batch_size>1`, `drop_last=true`, `drop_last=false`.
  - `[ ]` Documentation & Usage Example
- `[ ]` Samplers
  - `[ ]` Test permutations: sequential sampling (ordered), random sampling (shuffled indices), distributed sampling (mocked partitions).
  - `[ ]` Documentation & Usage Example

---

## 8. Serialization (SafeTensors)

- `[ ]` Model Weight Loading & Saving
  - `[ ]` Test permutations: saving a complex `#[module]` to disk and loading it back, verifying weights match exactly.
  - `[ ]` Error handling: attempting to load a Safetensors file with mismatched tensor shapes or missing keys.
  - `[ ]` Documentation & Usage Example
