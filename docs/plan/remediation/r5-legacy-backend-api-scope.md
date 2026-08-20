# R5 scope: the legacy backend operation API

A measurement, not a change. Nothing in this document has been applied; the
tree is exactly as it was before the experiment described here. It exists so
that R5 is planned against what the code actually is rather than against an
estimate.

## What R5 asks for

`docs/plan/remediation/architecture-stabilization.md`, decision 8: *public
backend operation helpers that bypass canonical execution become crate-private
or are removed*, with completion evidence *downstream fixtures cannot call
legacy operation methods*.

## What is actually public

Inherent `pub fn` operation helpers on the backend implementations, counted on
the current tree:

| File | Public inherent methods |
|---|---:|
| `crates/incin-backends/src/dispatch.rs` | 135 |
| `crates/incin-backends/src/wgpu/backend.rs` | 132 |
| `crates/incin-backends/src/cuda/backend.rs` | 120 |
| `crates/incin-backends/src/metal/backend.rs` | 112 |

These are `add`, `matmul`, `conv2d`, `mean_dim`, `softmax` and their siblings:
storage-in, storage-out helpers that take runtime dimensions and never mint a
descriptor. `crates/incin-backends/src/cpu/mod.rs` has none - the CPU backend
is already contracted, which is why it is the one described as complete.

Only `new` (wgpu, cuda, metal) and `metadata` (dispatch) are genuine public
API among them.

## Call sites, and why the change looked cheap

Outside their defining files, these names are referenced in five places:

| Location | References |
|---|---:|
| `src/wgpu/executor.rs` | 14 |
| `src/cuda/executor.rs` | 12 |
| `src/metal/executor.rs` | 6 |
| `src/dist/nccl.rs` | 3 |
| `tests/architecture_regression.rs` | 4 |

The first four are inside the crate and unaffected by `pub(crate)`. The four
in `architecture_regression.rs` are `::new()` constructors, which stay public.
Only four test call sites use a genuine operation helper - 
`TestBackend::{matmul, reshape, conv2d}` in `tests/wgpu_executor.rs` and
`WgpuB::transpose` in `tests/tensor_meta.rs` - each as the reference half of a
"descriptor execution matches the backend helper" comparison.

## What the experiment found

Rewriting all 501 to `pub(crate)` compiles. The crate then reports **30 groups
of dead code under `--features cpu,wgpu`, and 60 under `--all-features`.**

That is the finding: most of this layer has no caller inside the crate at all.
It was reachable only from outside, so `pub` was the only thing keeping the
compiler quiet about it. The whole `impl<D: Device> DispatchBackend<D>` helper
block - `add`, `sub`, `mul`, `div`, `reshape`, `softmax`, the quantization
trio, `cross_entropy_loss` - is orphaned; `dispatch_executor.rs` superseded it
and the old layer was never removed.

The dead set cascades. Once `dispatch.rs`'s helpers go, the CUDA helpers they
called go with them, and then `cuda/ops/{norm,embedding,loss,pool,conv}.rs`'s
launchers and their kernel source constants go too. Under `--all-features` the
cascade already reaches `launch_layer_norm`, `launch_batch_norm`,
`launch_embedding_forward`, `EMBEDDING_SRC`, `LOSS_KERNEL`, `QUANT_KERNEL`,
and the `tuning.rs` normalization candidate selection.

## Why it was not applied

Two reasons, both about who should make the call rather than about difficulty:

1. **The visibility change and the deletion are inseparable.** `pub(crate)` on
   an uncalled function is a `-D warnings` failure, and the CI clippy gate runs
   with `-D warnings`. There is no intermediate state: either the layer stays
   public, or it becomes crate-private and the orphaned part is deleted in the
   same change.

2. **The deletion reaches CUDA and Metal kernel code that cannot be executed
   here.** Removing `launch_layer_norm` and its kernel source is safe if the
   canonical executor really does not use it and unrecoverable-by-compiler if
   some feature combination does. That is a judgement for someone who can run
   the backend, not one to make from a green `cargo check`.

## What R5 should do

1. Delete `dispatch.rs`'s legacy helper block outright. It is superseded, not
   merely unused, and nothing outside the crate should have been calling a
   runtime-dispatch shim in the first place.
2. Follow the cascade into `cuda/`, `metal/`, and `wgpu/`, deleting each
   helper the canonical executor does not call, verifying on hardware per
   backend rather than by compilation alone.
3. Make the survivors `pub(crate)`.
4. Convert the four helper-comparison tests. Comparing descriptor execution
   against the helper only proved the two agreed; against a hand-computed
   value, or against the CPU backend, it proves the result is right. That is a
   stronger test and it does not need the helper to be reachable.
5. Add a consumer fixture that calls `WgpuBackendImpl::add` and must fail to
   compile, which is the completion evidence the phase asks for.
