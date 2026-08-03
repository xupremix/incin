# CPU canonical migration status

Generated from `CPU_CAPABILITIES` and `incin_core::exec::OPERATION_CATALOG`; the Rust source is authoritative. "Migrated" means the CPU backend advertises the exact identity and therefore, by the compile-time proof in `cpu::canonical`, implements `Execute<Descriptor<op::...>>` for it. It does not mean the operation is unreachable through the legacy operation-family traits: those remain the path the stable tensor surface uses.

The denominator is the number of operations that `Execute<Descriptor<O>>` can carry at all, not the whole catalog. An operation whose `ExecutionSite` is not backend-executable is listed separately with the reason: it is a gap in the execution trait rather than an unwritten executor, and counting it here would describe work that cannot be done without changing the contract first.

**145 of 161 backend-executable operations migrated**, out of 174 catalog operations in total. The remaining 16 executable operations are still reachable only through the legacy operation-family traits.

## Backend-executable operations

| Operation | Site | Migrated | Legacy source |
|---|---|:--:|---|
| `tensor_from_data` | `Creation` | no | `TensorArgsData` |
| `tensor_from_bytes` | `Creation` | no | `Tensor::from_bytes` |
| `tensor_to_bytes` | `HostReadback` | no | `Tensor::to_bytes` |
| `zeros` | `Creation` | yes | `CreationOps::zeros` |
| `ones` | `Creation` | yes | `CreationOps::ones` |
| `rand` | `Creation` | yes | `CreationOps::rand` |
| `randn` | `Creation` | yes | `CreationOps::randn` |
| `var_zeros` | `Creation` | no | `CreationOps::var_zeros` |
| `var_ones` | `Creation` | no | `CreationOps::var_ones` |
| `var_rand` | `Creation` | no | `CreationOps::var_rand` |
| `var_randn` | `Creation` | no | `CreationOps::var_randn` |
| `full` | `Creation` | yes | `CreationOps::full` |
| `arange` | `Creation` | yes | `CreationOps::arange` |
| `linspace` | `Creation` | yes | `CreationOps::linspace` |
| `sample` | `Creation` | no | `Tensor::sample` |
| `relu` | `Kernel` | yes | `FloatOps::relu` |
| `step` | `Kernel` | yes | `FloatOps::step` |
| `mish` | `Kernel` | yes | `FloatOps::mish` |
| `elu` | `Kernel` | yes | `FloatOps::elu` |
| `gelu` | `Kernel` | yes | `FloatOps::gelu` |
| `abs` | `Kernel` | yes | `FloatOps::abs` |
| `exp` | `Kernel` | yes | `FloatOps::exp` |
| `neg` | `Kernel` | yes | `FloatOps::neg` |
| `sqrt` | `Kernel` | yes | `FloatOps::sqrt` |
| `log` | `Kernel` | yes | `FloatOps::log` |
| `tanh` | `Kernel` | yes | `FloatOps::tanh` |
| `sigmoid` | `Kernel` | yes | `FloatOps::sigmoid` |
| `swish` | `Kernel` | yes | `FloatOps::swish` |
| `softmax` | `Kernel` | yes | `FloatOps::softmax` |
| `add_scalar` | `Kernel` | yes | `FloatOps::add_scalar_float` |
| `mul_scalar` | `Kernel` | yes | `FloatOps::mul_scalar_float` |
| `powf` | `Kernel` | yes | `FloatOps::powf` |
| `clamp` | `Kernel` | yes | `FloatOps::clamp` |
| `sign` | `Kernel` | yes | `FloatOps::sign` |
| `floor` | `Kernel` | yes | `FloatOps::floor` |
| `ceil` | `Kernel` | yes | `FloatOps::ceil` |
| `round` | `Kernel` | yes | `FloatOps::round` |
| `log2` | `Kernel` | yes | `FloatOps::log2` |
| `log10` | `Kernel` | yes | `FloatOps::log10` |
| `sin` | `Kernel` | yes | `FloatOps::sin` |
| `cos` | `Kernel` | yes | `FloatOps::cos` |
| `tan` | `Kernel` | yes | `FloatOps::tan` |
| `asin` | `Kernel` | yes | `FloatOps::asin` |
| `acos` | `Kernel` | yes | `FloatOps::acos` |
| `atan` | `Kernel` | yes | `FloatOps::atan` |
| `atan2` | `Kernel` | yes | `FloatOps::atan2` |
| `sinh` | `Kernel` | yes | `FloatOps::sinh` |
| `cosh` | `Kernel` | yes | `FloatOps::cosh` |
| `asinh` | `Kernel` | yes | `FloatOps::asinh` |
| `acosh` | `Kernel` | yes | `FloatOps::acosh` |
| `atanh` | `Kernel` | yes | `FloatOps::atanh` |
| `erf` | `Kernel` | yes | `FloatOps::erf` |
| `rsqrt` | `Kernel` | yes | `FloatOps::rsqrt` |
| `trunc` | `Kernel` | yes | `FloatOps::trunc` |
| `frac` | `Kernel` | yes | `FloatOps::frac` |
| `fmod` | `Kernel` | yes | `FloatOps::fmod` |
| `remainder` | `Kernel` | yes | `FloatOps::remainder` |
| `add` | `Kernel` | yes | `NumericOps::add` |
| `sub` | `Kernel` | yes | `NumericOps::sub` |
| `mul` | `Kernel` | yes | `NumericOps::mul` |
| `div` | `Kernel` | yes | `NumericOps::div` |
| `sub_scalar` | `Kernel` | yes | `TensorOps::sub_scalar` |
| `div_scalar` | `Kernel` | yes | `TensorOps::div_scalar` |
| `maximum` | `Kernel` | yes | `TensorOps::maximum` |
| `minimum` | `Kernel` | yes | `TensorOps::minimum` |
| `abs_diff` | `Kernel` | yes | `TensorOps::abs_diff` |
| `lerp` | `Kernel` | yes | `TensorOps::lerp` |
| `cmp_eq` | `Kernel` | yes | `TensorOps::cmp_eq` |
| `cmp_ne` | `Kernel` | yes | `TensorOps::cmp_ne` |
| `cmp_lt` | `Kernel` | yes | `TensorOps::cmp_lt` |
| `cmp_le` | `Kernel` | yes | `TensorOps::cmp_le` |
| `cmp_gt` | `Kernel` | yes | `TensorOps::cmp_gt` |
| `cmp_ge` | `Kernel` | yes | `TensorOps::cmp_ge` |
| `logical_and` | `Kernel` | yes | `TensorOps::logical_and` |
| `logical_or` | `Kernel` | yes | `TensorOps::logical_or` |
| `logical_not` | `Kernel` | yes | `TensorOps::logical_not` |
| `reshape` | `Kernel` | yes | `TensorOps::reshape` |
| `transpose` | `Kernel` | yes | `TensorOps::transpose` |
| `matmul` | `Kernel` | yes | `TensorOps::matmul` |
| `dot` | `Kernel` | yes | `Tensor::dot` |
| `outer` | `Kernel` | yes | `Tensor::outer` |
| `broadcast_as` | `Kernel` | yes | `TensorOps::broadcast_as` |
| `narrow` | `Kernel` | yes | `TensorOps::narrow` |
| `squeeze` | `Kernel` | yes | `TensorOps::squeeze` |
| `stack` | `Kernel` | yes | `TensorOps::stack` |
| `concat` | `Kernel` | yes | `TensorOps::concat` |
| `slice` | `Kernel` | yes | `TensorOps::slice` |
| `flatten` | `Kernel` | yes | `TensorOps::flatten` |
| `where_cond` | `Kernel` | yes | `TensorOps::where_cond` |
| `gather` | `Kernel` | yes | `TensorOps::gather` |
| `scatter` | `Kernel` | yes | `TensorOps::scatter` |
| `index_select` | `Kernel` | yes | `TensorOps::index_select` |
| `masked_fill` | `Kernel` | yes | `TensorOps::masked_fill` |
| `unsqueeze` | `Kernel` | yes | `TensorOps::unsqueeze` |
| `repeat` | `Kernel` | yes | `TensorOps::repeat` |
| `pad` | `Kernel` | yes | `TensorOps::pad` |
| `triu` | `Kernel` | yes | `TensorOps::triu` |
| `tril` | `Kernel` | yes | `TensorOps::tril` |
| `diag` | `Kernel` | yes | `TensorOps::diag` |
| `chunk` | `Kernel` | yes | `Tensor::chunk` |
| `split` | `Kernel` | yes | `Tensor::split` |
| `addmm` | `Kernel` | yes | `TensorOps::addmm` |
| `bmm` | `Kernel` | yes | `TensorOps::bmm` |
| `scaled_dot_product_attention` | `Kernel` | yes | `TensorOps::scaled_dot_product_attention` |
| `unfold` | `Kernel` | yes | `TensorOps::unfold` |
| `pixel_shuffle` | `Kernel` | yes | `TensorOps::pixel_shuffle` |
| `group_norm` | `Kernel` | yes | `TensorOps::group_norm` |
| `instance_norm` | `Kernel` | yes | `TensorOps::instance_norm` |
| `broadcast_left` | `Kernel` | yes | `TensorOps::broadcast_left` |
| `float_to_scalar` | `HostReadback` | no | `TensorOps::float_to_scalar` |
| `float_to_vec1` | `HostReadback` | no | `TensorOps::float_to_vec1` |
| `int_to_scalar` | `HostReadback` | no | `TensorOps::int_to_scalar` |
| `int_to_vec1` | `HostReadback` | no | `TensorOps::int_to_vec1` |
| `to_dtype` | `Kernel` | yes | `TensorOps::tensor_to_dtype` |
| `sum_all` | `Kernel` | yes | `ReductionOps::sum_all` |
| `mean_all` | `Kernel` | yes | `ReductionOps::mean_all` |
| `max_all` | `Kernel` | yes | `ReductionOps::max_all` |
| `min_all` | `Kernel` | yes | `ReductionOps::min_all` |
| `sum_dim` | `Kernel` | yes | `ReductionOps::sum_dim` |
| `sum_keepdim` | `Kernel` | yes | `ReductionOps::sum_keepdim` |
| `mean_dim` | `Kernel` | yes | `ReductionOps::mean_dim` |
| `mean_keepdim` | `Kernel` | yes | `ReductionOps::mean_keepdim` |
| `max_dim` | `Kernel` | yes | `ReductionOps::max_dim` |
| `max_keepdim` | `Kernel` | yes | `ReductionOps::max_keepdim` |
| `min_dim` | `Kernel` | yes | `ReductionOps::min_dim` |
| `min_keepdim` | `Kernel` | yes | `ReductionOps::min_keepdim` |
| `argmax` | `Kernel` | yes | `ReductionOps::argmax` |
| `argmin` | `Kernel` | yes | `ReductionOps::argmin` |
| `prod_all` | `Kernel` | yes | `ReductionOps::prod_all` |
| `prod_dim` | `Kernel` | yes | `ReductionOps::prod_dim` |
| `cumsum` | `Kernel` | yes | `ReductionOps::cumsum` |
| `topk` | `Kernel` | yes | `ReductionOps::topk` |
| `argsort` | `Kernel` | yes | `ReductionOps::argsort` |
| `norm` | `Kernel` | yes | `Tensor::norm` |
| `var_all` | `Kernel` | yes | `Tensor::var_all` |
| `var_dim` | `Kernel` | yes | `Tensor::var_dim` |
| `var_keepdim` | `Kernel` | yes | `Tensor::var_keepdim` |
| `std_all` | `Kernel` | yes | `Tensor::std_all` |
| `std_dim` | `Kernel` | yes | `Tensor::std_dim` |
| `std_keepdim` | `Kernel` | yes | `Tensor::std_keepdim` |
| `layer_norm` | `Kernel` | yes | `ModuleOps::layer_norm` |
| `batch_norm` | `Kernel` | yes | `ModuleOps::batch_norm` |
| `embedding` | `Kernel` | no | `ModuleOps::embedding` |
| `conv1d` | `Kernel` | yes | `ModuleOps::conv1d` |
| `conv2d` | `Kernel` | yes | `ModuleOps::conv2d` |
| `conv_transpose2d` | `Kernel` | yes | `ModuleOps::conv_transpose2d` |
| `max_pool2d` | `Kernel` | yes | `ModuleOps::max_pool2d` |
| `avg_pool2d` | `Kernel` | yes | `ModuleOps::avg_pool2d` |
| `adaptive_avg_pool2d` | `Kernel` | yes | `ModuleOps::adaptive_avg_pool2d` |
| `linear` | `Kernel` | yes | `Linear::forward` |
| `rms_norm` | `Kernel` | yes | `RMSNorm::forward` |
| `dropout` | `Kernel` | yes | `Dropout::forward` |
| `rnn` | `Kernel` | no | `RNN::forward` |
| `lstm` | `Kernel` | no | `LSTM::forward` |
| `mse_loss` | `Kernel` | yes | `LossOps::mse_loss` |
| `l1_loss` | `Kernel` | yes | `LossOps::l1_loss` |
| `bce_with_logits_loss` | `Kernel` | yes | `LossOps::bce_with_logits_loss` |
| `cross_entropy_loss` | `Kernel` | no | `LossOps::cross_entropy_loss` |
| `quantize` | `Kernel` | yes | `QuantizedOps::quantize` |
| `dequantize` | `Kernel` | yes | `QuantizedOps::dequantize` |
| `quantized_matmul` | `Kernel` | yes | `QuantizedOps::quantized_matmul` |

## Operations the execution contract cannot carry

These are not pending migrations. Each one needs a change to `Execute`/`ExecutionRequest` before an executor for it could be written, and until then the stable tensor surface reaches it through the legacy path by necessity rather than by omission.

| Operation | Site | Why | Legacy source |
|---|---|---|---|
| `add_in_place` | `Mutation` | writes through an operand; execution borrows operands shared | `Tensor::add_` |
| `sub_in_place` | `Mutation` | writes through an operand; execution borrows operands shared | `Tensor::sub_` |
| `mul_in_place` | `Mutation` | writes through an operand; execution borrows operands shared | `Tensor::mul_` |
| `div_in_place` | `Mutation` | writes through an operand; execution borrows operands shared | `Tensor::div_` |
| `zero_in_place` | `Mutation` | writes through an operand; execution borrows operands shared | `Tensor::zero_` |
| `fill_in_place` | `Mutation` | writes through an operand; execution borrows operands shared | `Tensor::fill_` |
| `to_device` | `DeviceTransfer` | produces storage on another backend, which the executor cannot name | `TransferTo` |
| `require_grad` | `GraphState` | acts on autograd state, not on an allocation | `Tensor::require_grad` |
| `detach` | `GraphState` | acts on autograd state, not on an allocation | `Tensor::detach` |
| `backward` | `GraphState` | acts on autograd state, not on an allocation | `Tensor::backward` |
| `sgd_step` | `Mutation` | writes through an operand; execution borrows operands shared | `SGD::step` |
| `adam_step` | `Mutation` | writes through an operand; execution borrows operands shared | `Adam::step` |
| `adamw_step` | `Mutation` | writes through an operand; execution borrows operands shared | `OptimizerOps::adamw_step` |
