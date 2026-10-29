# Normalization in training: BatchNorm's two modes

Normalization layers look stateless until you train through them.
BatchNorm normalizes by the batch's own statistics while training and by
accumulated running statistics in evaluation — two data paths through one
module, with a known hole between them. This chapter walks both paths and
names the hole precisely.

## The two modes, honestly separated

The layer honors `TrainMode` and starts in training mode
(`crates/incin-core/src/nn/batch_norm.rs`): training normalizes each
channel by the batch's mean and variance; evaluation normalizes by
`running_mean` and `running_var`. Before the layer honored the mode it
normalized by `running_mean = 0` and `running_var = 1` on every call, which
made every evaluation wrong in a way that looked numerically plausible —
the worst kind of bug, and the reason the mode split exists.

At the operation level, `Tensor::batch_norm` always trains
(`training: true` is hardcoded in
`crates/incin-core/src/tensor/ops/module.rs`): five inputs, one output,
same shape back:

```rust
use incin::prelude::*;

let x = Cpu.tensor([[1.0f32, 2.0], [3.0, 4.0]])?;
let weight = Cpu.ones(vec![2])?;
let bias = Cpu.zeros(vec![2])?;
let running_mean = Cpu.zeros(vec![2])?;
let running_var = Cpu.ones(vec![2])?;
let y = x.batch_norm(&weight, &bias, &running_mean, &running_var, 1e-5)?;
assert_eq!(y.dims().dims(), &[2, 2]);
// Training mode normalizes each channel's batch mean to (near) zero.
let v = y.to_vec1::<f32>()?;
assert!(((v[0] + v[2]) / 2.0).abs() < 1e-4);
assert!(((v[1] + v[3]) / 2.0).abs() < 1e-4);
# Ok::<(), incin::Error>(())
```

CUDA agrees on the contract and adds the training dynamics: the training
forward launches the kernel, saves statistics, and pushes one tape entry
(`launch_batch_norm_training` plus the backward launcher in
`crates/incin-backends/src/cuda/backend/norm.rs`, #123). A recording
forward that saved no statistics is a named error, not a silent wrong
gradient.

## The hole: statistics flow in, never out

Here is the boundary, stated exactly once so no other chapter has to
hedge: the running statistics are *not* updated by any forward, on any
backend, and cannot be under the current execution contract. They arrive
as shared references, and mutating an operand is not something the
contract carries — CPU's `batch_norm_training_impl` and CUDA's kernel
agree on this. Training mode does not read them at all (normalization
uses batch statistics), so they are accepted for arity and ignored. The
`momentum` argument is read and never spent.

In practice that leaves two honest options, both used in-tree. Supply the
statistics from outside — a checkpoint, or a `collect_state`/restore pass
— or leave the model in training mode for evaluation, which is sound
whenever the evaluation batch is large enough for its statistics to stand
in for the population's. Closing the hole means either a layout-agnostic
mutation path for buffers or a training-mode write-back outside the
`forward` contract; until one lands, this chapter stays exactly this
short.
