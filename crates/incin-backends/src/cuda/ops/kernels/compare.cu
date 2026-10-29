#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Elementwise numeric comparison, writing a boolean result (0 or 1) per
// element as `unsigned char` -- CUDA's own encoding matches Rust's `bool`
// (one byte, values 0x00/0x01). Both operands share one dtype and must
// already share one contiguous shape and element count; the Rust launcher
// never calls this with mismatched shapes, so there is no broadcasting or
// stride bookkeeping here the way the general elementwise kernels need.
//
// `op_mode` selects which comparison to apply: 0=eq, 1=ne, 2=lt, 3=le,
// 4=gt, 5=ge (any other value falls through to ge, but the Rust side never
// sends one).
//
// One template serves every float storage dtype (`f32`/`f64`/`f16`/`bf16`,
// issue #90 slice 4), the same shape `select.cu` uses: the output stays
// `unsigned char` in all of them, only the operand pointers change type.
// Each dtype gets its own `extern "C"` entry point so the Rust launcher's
// entry-point map stays fail-closed: a dtype without an entry can never
// reach a buffer reinterpreted as the wrong type.
template <typename T>
__device__ void compare_impl(
    const T* __restrict__ lhs,
    const T* __restrict__ rhs,
    unsigned char* __restrict__ out,
    int op_mode,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    T a = lhs[idx];
    T b = rhs[idx];
    unsigned char result;
    switch (op_mode) {
        case 0: result = (a == b) ? 1 : 0; break;
        case 1: result = (a != b) ? 1 : 0; break;
        case 2: result = (a <  b) ? 1 : 0; break;
        case 3: result = (a <= b) ? 1 : 0; break;
        case 4: result = (a >  b) ? 1 : 0; break;
        default: result = (a >= b) ? 1 : 0; break;
    }
    out[idx] = result;
}

extern "C" __global__ void compare_op(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    unsigned char* __restrict__ out,
    int op_mode,
    int numel
) {
    compare_impl<float>(lhs, rhs, out, op_mode, numel);
}

extern "C" __global__ void compare_op_f64(
    const double* __restrict__ lhs,
    const double* __restrict__ rhs,
    unsigned char* __restrict__ out,
    int op_mode,
    int numel
) {
    compare_impl<double>(lhs, rhs, out, op_mode, numel);
}

extern "C" __global__ void compare_op_f16(
    const __half* __restrict__ lhs,
    const __half* __restrict__ rhs,
    unsigned char* __restrict__ out,
    int op_mode,
    int numel
) {
    compare_impl<__half>(lhs, rhs, out, op_mode, numel);
}

extern "C" __global__ void compare_op_bf16(
    const __nv_bfloat16* __restrict__ lhs,
    const __nv_bfloat16* __restrict__ rhs,
    unsigned char* __restrict__ out,
    int op_mode,
    int numel
) {
    compare_impl<__nv_bfloat16>(lhs, rhs, out, op_mode, numel);
}
