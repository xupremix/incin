# Reviewed stable API to canonical operation mapping

This review covers stable tensor construction, manipulation, arithmetic,
pointwise, reduction, transfer, autograd, module/loss, optimizer, and quantized
entry points in the current checkout. The Rust rows in
`incin_core::exec::OPERATION_CATALOG` are authoritative. Each semantic identity
appears there once; the paths below are aliases, typed variants, or deliberate
compositions and therefore do not create a second identity.

## Direct identities added outside the legacy backend traits

| Stable path | Canonical identity |
|---|---|
| `Tensor::sample` | `Sample` |
| `Tensor::dot` | `Dot` |
| `Tensor::outer` | `Outer` |
| `Tensor::chunk` | `Chunk` |
| `Tensor::split` | `Split` |
| `Tensor::add_` | `AddInPlace` |
| `Tensor::sub_` | `SubInPlace` |
| `Tensor::mul_` | `MulInPlace` |
| `Tensor::div_` | `DivInPlace` |
| `Tensor::zero_` | `ZeroInPlace` |
| `Tensor::fill_` | `FillInPlace` |
| `Tensor::require_grad` | `RequireGrad` |
| `Tensor::detach` | `Detach` |
| `Tensor::backward` | `Backward` |
| `Linear::forward` | `Linear` |
| `RMSNorm::forward` | `RmsNorm` |
| `Dropout::forward` | `Dropout` |
| `RNN::forward` | `Rnn` |
| `LSTM::forward` | `Lstm` |
| `SGD::step` | `SgdStep` |
| `Adam::step` | `AdamStep` |

## Stable aliases and typed variants

| Stable paths | Canonical identity |
|---|---|
| `Tensor::from_slice`, tensor-data argument conversion | `TensorFromData` |
| `Tensor::reshape`, `try_reshape`, `reshape_idx`, `to_shape`, `into_shape` | `ReshapeExact` |
| `Tensor::slice`, `dyn_slice`, `slice_idx` | `SliceExact` |
| `Tensor::get` | checked composition of `SliceExact` and `SqueezeExact` |
| `Tensor::broadcast_to`, `expand` | `BroadcastAs` |
| `Tensor::try_narrow` | `Narrow` |
| `Tensor::try_squeeze` | `SqueezeExact` |
| `Tensor::try_concat_slice`, `concat`, `try_concat` | `ConcatExact` |
| `Tensor::try_stack_slice`, `stack`, `try_stack`, `try_stack_tensors` | `StackExact` |
| `Tensor::mul_scalar` | `MulScalar` |
| `Tensor::add_scalar` | `AddScalar` |
| `Tensor::to_scalar` | `ToHostFloatScalar` or `ToHostIntScalar`, selected by dtype |
| `Tensor::to_vec1` | `ToHostFloatVec` or `ToHostIntVec`, selected by dtype |
| loss convenience methods and their `_with` forms | the corresponding exact loss identity |
| typed and dynamic reduction-axis variants | the corresponding exact all/dim/keepdim identity |
| tensor and module `conv2d` entry points | `Conv2dExact` |
| tensor and module pooling entry points | `MaxPool2d`, `AvgPool2d`, or `AdaptiveAvgPool2dExact` |

## Deliberate exclusions

Metadata accessors (`shape`, `dims`, `dtype`, `device`, `rank`, `numel`, and
placement/gradient queries) do not execute a semantic tensor operation. Raw
storage constructors and resharding bridges are backend-authoring or
experimental contracts, not stable end-user execution identities. Module
builders and optimizer schedulers create configuration/state; their tensor
effects are represented by the exact forward or step identities above.

The 142 legacy operation-family methods are mapped separately in
`old-trait-to-descriptor.md`; that list was reproduced from the current source
and is not inferred from a historical count.
