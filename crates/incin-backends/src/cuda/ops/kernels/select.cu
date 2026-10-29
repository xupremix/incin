#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Mask-driven selection, the counterpart to `compare.cu`'s mask *producers*:
// these two kernels are what a `bool` mask is actually for. Both read a
// `bool` mask as `unsigned char` -- the same 0x00/0x01 encoding `compare.cu`
// writes -- alongside float data, and write a fresh result in the same float
// dtype. Both require every operand pre-broadcast to one contiguous shape and
// element count by the Rust launcher, the same precondition `compare.cu`
// states, so neither kernel does any stride or shape bookkeeping of its own.
//
// Broadcasting a lower-rank `bool` mask up to that shared shape is the
// launcher's job too, and since #122 it rides `shape.cu`'s own width-
// parametric `shape_op_8bit` through `shape::launch_broadcast`, like every
// other dtype -- the dedicated `broadcast_bool_op` this file used to carry
// was deleted as redundant.
//
// One template serves every float storage dtype (`f32`/`f64`/`f16`/`bf16`,
// issue #90 slice 3), the same shape `embedding.cu` uses: the mask stays
// `unsigned char` in all of them, only the value pointers and -- for
// `masked_fill` -- the scalar parameter change type. Each dtype gets its own
// `extern "C"` entry point so the Rust launcher's entry-point map stays
// fail-closed: a dtype without an entry can never reach a buffer
// reinterpreted as the wrong type.

// `where_cond(mask, on_true, on_false)`: picks `on_true[idx]` where the mask
// is set, `on_false[idx]` otherwise.
template <typename T>
__device__ void where_cond_impl(
    const unsigned char* __restrict__ mask,
    const T* __restrict__ on_true,
    const T* __restrict__ on_false,
    T* __restrict__ out,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = mask[idx] ? on_true[idx] : on_false[idx];
}

// `masked_fill(input, mask, value)`: overwrites the masked positions with a
// scalar, an attribute rather than an operand, so it is a kernel parameter
// here rather than a third pointer.
template <typename T>
__device__ void masked_fill_impl(
    const T* __restrict__ input,
    const unsigned char* __restrict__ mask,
    T value,
    T* __restrict__ out,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = mask[idx] ? value : input[idx];
}

extern "C" __global__ void where_cond_op(
    const unsigned char* __restrict__ mask,
    const float* __restrict__ on_true,
    const float* __restrict__ on_false,
    float* __restrict__ out,
    int numel
) {
    where_cond_impl<float>(mask, on_true, on_false, out, numel);
}

extern "C" __global__ void where_cond_op_f64(
    const unsigned char* __restrict__ mask,
    const double* __restrict__ on_true,
    const double* __restrict__ on_false,
    double* __restrict__ out,
    int numel
) {
    where_cond_impl<double>(mask, on_true, on_false, out, numel);
}

extern "C" __global__ void where_cond_op_f16(
    const unsigned char* __restrict__ mask,
    const __half* __restrict__ on_true,
    const __half* __restrict__ on_false,
    __half* __restrict__ out,
    int numel
) {
    where_cond_impl<__half>(mask, on_true, on_false, out, numel);
}

extern "C" __global__ void where_cond_op_bf16(
    const unsigned char* __restrict__ mask,
    const __nv_bfloat16* __restrict__ on_true,
    const __nv_bfloat16* __restrict__ on_false,
    __nv_bfloat16* __restrict__ out,
    int numel
) {
    where_cond_impl<__nv_bfloat16>(mask, on_true, on_false, out, numel);
}

extern "C" __global__ void masked_fill_op(
    const float* __restrict__ input,
    const unsigned char* __restrict__ mask,
    float value,
    float* __restrict__ out,
    int numel
) {
    masked_fill_impl<float>(input, mask, value, out, numel);
}

extern "C" __global__ void masked_fill_op_f64(
    const double* __restrict__ input,
    const unsigned char* __restrict__ mask,
    double value,
    double* __restrict__ out,
    int numel
) {
    masked_fill_impl<double>(input, mask, value, out, numel);
}

extern "C" __global__ void masked_fill_op_f16(
    const __half* __restrict__ input,
    const unsigned char* __restrict__ mask,
    __half value,
    __half* __restrict__ out,
    int numel
) {
    masked_fill_impl<__half>(input, mask, value, out, numel);
}

extern "C" __global__ void masked_fill_op_bf16(
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ mask,
    __nv_bfloat16 value,
    __nv_bfloat16* __restrict__ out,
    int numel
) {
    masked_fill_impl<__nv_bfloat16>(input, mask, value, out, numel);
}
