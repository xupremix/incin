// Elementwise numeric comparison, writing a boolean result (0 or 1) per
// element as `unsigned char` -- CUDA's own encoding matches Rust's `bool`
// (one byte, values 0x00/0x01). Both operands are `float` and must already
// share one contiguous shape and element count; the Rust launcher never
// calls this with mismatched shapes, so there is no broadcasting or stride
// bookkeeping here the way the general elementwise kernels need.
//
// `op_mode` selects which comparison to apply: 0=eq, 1=ne, 2=lt, 3=le,
// 4=gt, 5=ge (any other value falls through to ge, but the Rust side never
// sends one).
extern "C" __global__ void compare_op(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    unsigned char* __restrict__ out,
    int op_mode,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float a = lhs[idx];
    float b = rhs[idx];
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
