# CPU canonical migration status

Generated from `CPU_CAPABILITIES` and `incin_core::exec::OPERATION_CATALOG`; the Rust source is authoritative. "Migrated" means the CPU backend advertises the exact identity and therefore, by the compile-time proof in `cpu::canonical`, implements `Execute<op::...>` for it. The legacy operation-family traits remain only as backend-local adapters for special kernels and compatibility tests; they are not the stable tensor execution path.

The denominator is the number of operations that `Execute<O>` can carry at all, not the whole catalog. An operation whose `ExecutionSite` is not backend-executable is listed separately with the reason: it is a gap in the execution trait rather than an unwritten executor, and counting it here would describe work that cannot be done without changing the contract first.

**162 of 162 backend-executable operations migrated**, out of 178 catalog operations in total.

## Backend-executable operations

| Operation | Site | Migrated | Catalog source mapping |
|---|---|:--:|---|
| `tensor_from_data` | `Creation` | yes | `TensorArgsData` |
| `tensor_from_bytes` | `Creation` | yes | `Tensor::from_bytes` |
| `tensor_to_bytes` | `HostReadback` | yes | `Tensor::to_bytes` |
| `zeros` | `Creation` | yes | `Descriptor<op::Zeros>` |
| `ones` | `Creation` | yes | `Descriptor<op::Ones>` |
| `rand` | `Creation` | yes | `Descriptor<op::UniformRandom>` |
| `randn` | `Creation` | yes | `Descriptor<op::NormalRandom>` |
| `var_zeros` | `Creation` | yes | `Descriptor<op::VariableZeros>` |
| `var_ones` | `Creation` | yes | `Descriptor<op::VariableOnes>` |
| `var_rand` | `Creation` | yes | `Descriptor<op::VariableUniformRandom>` |
| `var_randn` | `Creation` | yes | `Descriptor<op::VariableNormalRandom>` |
| `full` | `Creation` | yes | `Descriptor<op::Full>` |
| `arange` | `Creation` | yes | `Descriptor<op::Arange>` |
| `linspace` | `Creation` | yes | `Descriptor<op::Linspace>` |
| `relu` | `Kernel` | yes | `::relu` |
| `step` | `Kernel` | yes | `::step` |
| `mish` | `Kernel` | yes | `::mish` |
| `elu` | `Kernel` | yes | `::elu` |
| `gelu` | `Kernel` | yes | `::gelu` |
| `abs` | `Kernel` | yes | `::abs` |
| `exp` | `Kernel` | yes | `::exp` |
| `neg` | `Kernel` | yes | `::neg` |
| `sqrt` | `Kernel` | yes | `::sqrt` |
| `log` | `Kernel` | yes | `::log` |
| `tanh` | `Kernel` | yes | `::tanh` |
| `sigmoid` | `Kernel` | yes | `::sigmoid` |
| `swish` | `Kernel` | yes | `::swish` |
| `softmax` | `Kernel` | yes | `::softmax` |
| `log_softmax` | `Kernel` | yes | `::log_softmax` |
| `add_scalar` | `Kernel` | yes | `::add_scalar_float` |
| `mul_scalar` | `Kernel` | yes | `::mul_scalar_float` |
| `powf` | `Kernel` | yes | `::powf` |
| `clamp` | `Kernel` | yes | `::clamp` |
| `sign` | `Kernel` | yes | `::sign` |
| `floor` | `Kernel` | yes | `::floor` |
| `ceil` | `Kernel` | yes | `::ceil` |
| `round` | `Kernel` | yes | `::round` |
| `log2` | `Kernel` | yes | `::log2` |
| `log10` | `Kernel` | yes | `::log10` |
| `sin` | `Kernel` | yes | `::sin` |
| `cos` | `Kernel` | yes | `::cos` |
| `tan` | `Kernel` | yes | `::tan` |
| `asin` | `Kernel` | yes | `::asin` |
| `acos` | `Kernel` | yes | `::acos` |
| `atan` | `Kernel` | yes | `::atan` |
| `atan2` | `Kernel` | yes | `::atan2` |
| `sinh` | `Kernel` | yes | `::sinh` |
| `cosh` | `Kernel` | yes | `::cosh` |
| `asinh` | `Kernel` | yes | `::asinh` |
| `acosh` | `Kernel` | yes | `::acosh` |
| `atanh` | `Kernel` | yes | `::atanh` |
| `erf` | `Kernel` | yes | `::erf` |
| `rsqrt` | `Kernel` | yes | `::rsqrt` |
| `trunc` | `Kernel` | yes | `::trunc` |
| `frac` | `Kernel` | yes | `::frac` |
| `fmod` | `Kernel` | yes | `::fmod` |
| `remainder` | `Kernel` | yes | `::remainder` |
| `add` | `Kernel` | yes | `::add` |
| `sub` | `Kernel` | yes | `::sub` |
| `mul` | `Kernel` | yes | `::mul` |
| `div` | `Kernel` | yes | `::div` |
| `sub_scalar` | `Kernel` | yes | `::sub_scalar` |
| `div_scalar` | `Kernel` | yes | `::div_scalar` |
| `maximum` | `Kernel` | yes | `::maximum` |
| `minimum` | `Kernel` | yes | `::minimum` |
| `abs_diff` | `Kernel` | yes | `::abs_diff` |
| `lerp` | `Kernel` | yes | `::lerp` |
| `cmp_eq` | `Kernel` | yes | `::cmp_eq` |
| `cmp_ne` | `Kernel` | yes | `::cmp_ne` |
| `cmp_lt` | `Kernel` | yes | `::cmp_lt` |
| `cmp_le` | `Kernel` | yes | `::cmp_le` |
| `cmp_gt` | `Kernel` | yes | `::cmp_gt` |
| `cmp_ge` | `Kernel` | yes | `::cmp_ge` |
| `logical_and` | `Kernel` | yes | `::logical_and` |
| `logical_or` | `Kernel` | yes | `::logical_or` |
| `logical_not` | `Kernel` | yes | `::logical_not` |
| `reshape` | `Kernel` | yes | `::reshape` |
| `transpose` | `Kernel` | yes | `::transpose` |
| `matmul` | `Kernel` | yes | `::matmul` |
| `dot` | `Kernel` | yes | `Tensor::dot` |
| `outer` | `Kernel` | yes | `Tensor::outer` |
| `broadcast_as` | `Kernel` | yes | `::broadcast_as` |
| `narrow` | `Kernel` | yes | `::narrow` |
| `squeeze` | `Kernel` | yes | `::squeeze` |
| `stack` | `Kernel` | yes | `::stack` |
| `concat` | `Kernel` | yes | `::concat` |
| `slice` | `Kernel` | yes | `::slice` |
| `flatten` | `Kernel` | yes | `::flatten` |
| `where_cond` | `Kernel` | yes | `::where_cond` |
| `gather` | `Kernel` | yes | `::gather` |
| `scatter` | `Kernel` | yes | `::scatter` |
| `scatter_add` | `Kernel` | yes | `::scatter_add` |
| `index_select` | `Kernel` | yes | `::index_select` |
| `masked_fill` | `Kernel` | yes | `::masked_fill` |
| `unsqueeze` | `Kernel` | yes | `::unsqueeze` |
| `repeat` | `Kernel` | yes | `::repeat` |
| `pad` | `Kernel` | yes | `::pad` |
| `triu` | `Kernel` | yes | `::triu` |
| `tril` | `Kernel` | yes | `::tril` |
| `diag` | `Kernel` | yes | `::diag` |
| `chunk` | `Kernel` | yes | `Tensor::chunk` |
| `split` | `Kernel` | yes | `Tensor::split` |
| `addmm` | `Kernel` | yes | `::addmm` |
| `bmm` | `Kernel` | yes | `::bmm` |
| `scaled_dot_product_attention` | `Kernel` | yes | `::scaled_dot_product_attention` |
| `unfold` | `Kernel` | yes | `::unfold` |
| `pixel_shuffle` | `Kernel` | yes | `::pixel_shuffle` |
| `group_norm` | `Kernel` | yes | `::group_norm` |
| `instance_norm` | `Kernel` | yes | `::instance_norm` |
| `broadcast_left` | `Kernel` | yes | `::broadcast_left` |
| `float_to_scalar` | `HostReadback` | yes | `::float_to_scalar` |
| `float_to_vec1` | `HostReadback` | yes | `::float_to_vec1` |
| `int_to_scalar` | `HostReadback` | yes | `::int_to_scalar` |
| `int_to_vec1` | `HostReadback` | yes | `::int_to_vec1` |
| `to_dtype` | `Kernel` | yes | `::tensor_to_dtype` |
| `sum_all` | `Kernel` | yes | `::sum_all` |
| `mean_all` | `Kernel` | yes | `::mean_all` |
| `max_all` | `Kernel` | yes | `::max_all` |
| `min_all` | `Kernel` | yes | `::min_all` |
| `sum_dim` | `Kernel` | yes | `::sum_dim` |
| `sum_keepdim` | `Kernel` | yes | `::sum_keepdim` |
| `mean_dim` | `Kernel` | yes | `::mean_dim` |
| `mean_keepdim` | `Kernel` | yes | `::mean_keepdim` |
| `max_dim` | `Kernel` | yes | `::max_dim` |
| `max_keepdim` | `Kernel` | yes | `::max_keepdim` |
| `min_dim` | `Kernel` | yes | `::min_dim` |
| `min_keepdim` | `Kernel` | yes | `::min_keepdim` |
| `argmax` | `Kernel` | yes | `::argmax` |
| `argmin` | `Kernel` | yes | `::argmin` |
| `logsumexp_dim` | `Kernel` | yes | `::logsumexp_dim` |
| `logsumexp_keepdim` | `Kernel` | yes | `::logsumexp_keepdim` |
| `prod_all` | `Kernel` | yes | `::prod_all` |
| `prod_dim` | `Kernel` | yes | `::prod_dim` |
| `cumsum` | `Kernel` | yes | `::cumsum` |
| `topk` | `Kernel` | yes | `::topk` |
| `argsort` | `Kernel` | yes | `::argsort` |
| `norm` | `Kernel` | yes | `Tensor::norm` |
| `var_all` | `Kernel` | yes | `Tensor::var_all` |
| `var_dim` | `Kernel` | yes | `Tensor::var_dim` |
| `var_keepdim` | `Kernel` | yes | `Tensor::var_keepdim` |
| `std_all` | `Kernel` | yes | `Tensor::std_all` |
| `std_dim` | `Kernel` | yes | `Tensor::std_dim` |
| `std_keepdim` | `Kernel` | yes | `Tensor::std_keepdim` |
| `layer_norm` | `Kernel` | yes | `::layer_norm` |
| `batch_norm` | `Kernel` | yes | `::batch_norm` |
| `embedding` | `Kernel` | yes | `::embedding` |
| `conv1d` | `Kernel` | yes | `::conv1d` |
| `conv2d` | `Kernel` | yes | `::conv2d` |
| `conv_transpose2d` | `Kernel` | yes | `::conv_transpose2d` |
| `max_pool2d` | `Kernel` | yes | `::max_pool2d` |
| `avg_pool2d` | `Kernel` | yes | `::avg_pool2d` |
| `adaptive_avg_pool2d` | `Kernel` | yes | `::adaptive_avg_pool2d` |
| `linear` | `Kernel` | yes | `Linear::forward` |
| `rms_norm` | `Kernel` | yes | `RMSNorm::forward` |
| `dropout` | `Kernel` | yes | `Dropout::forward` |
| `mse_loss` | `Kernel` | yes | `Loss::mse_loss` |
| `l1_loss` | `Kernel` | yes | `Loss::l1_loss` |
| `bce_with_logits_loss` | `Kernel` | yes | `Loss::bce_with_logits_loss` |
| `cross_entropy_loss` | `Kernel` | yes | `Loss::cross_entropy_loss` |
| `quantize` | `Kernel` | yes | `Descriptor<op::Quantize>` |
| `dequantize` | `Kernel` | yes | `Descriptor<op::Dequantize>` |
| `quantized_matmul` | `Kernel` | yes | `Descriptor<op::QuantizedMatMul>` |

## Why the rest have no executor

None of these is an unwritten function. Each names a limit of the descriptor or capability contract that has to change before an executor for it could be written at all, so the remaining count and the remaining work are not the same number.

| Operation | What blocks it |
|---|---|

## Operations the execution contract cannot carry

These are not pending migrations. Each one needs a change to `Execute`/`ExecutionRequest` before an executor for it could be written, and until then the stable tensor surface reaches it only through the documented non-backend-executable boundary rather than by omission.

| Operation | Site | Why | Catalog source mapping |
|---|---|---|---|
| `sample` | `Composed` | the frontend composition owns the execution semantics | `Tensor::sample` |
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
| `rnn` | `Composed` | the frontend composition owns the execution semantics | `RNN::forward` |
| `lstm` | `Composed` | the frontend composition owns the execution semantics | `LSTM::forward` |
| `sgd_step` | `Mutation` | writes through an operand; execution borrows operands shared | `SGD::step` |
| `adam_step` | `Mutation` | writes through an operand; execution borrows operands shared | `Adam::step` |
| `adamw_step` | `Mutation` | writes through an operand; execution borrows operands shared | `AdamW::step` |
