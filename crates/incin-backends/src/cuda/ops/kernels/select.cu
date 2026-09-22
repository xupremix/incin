// Mask-driven selection, the counterpart to `compare.cu`'s mask *producers*:
// these two kernels are what a `bool` mask is actually for. Both read a
// `bool` mask as `unsigned char` -- the same 0x00/0x01 encoding `compare.cu`
// writes -- alongside `float` data, and write a fresh `float` result. Both
// require every operand pre-broadcast to one contiguous shape and element
// count by the Rust launcher, the same precondition `compare.cu` states, so
// neither kernel does any stride or shape bookkeeping of its own.
//
// Broadcasting a lower-rank `bool` mask up to that shared shape is the
// launcher's job too, and since #122 it rides `shape.cu`'s own width-
// parametric `shape_op_8bit` through `shape::launch_broadcast`, like every
// other dtype -- the dedicated `broadcast_bool_op` this file used to carry
// was deleted as redundant.

// `where_cond(mask, on_true, on_false)`: picks `on_true[idx]` where the mask
// is set, `on_false[idx]` otherwise.
extern "C" __global__ void where_cond_op(
    const unsigned char* __restrict__ mask,
    const float* __restrict__ on_true,
    const float* __restrict__ on_false,
    float* __restrict__ out,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = mask[idx] ? on_true[idx] : on_false[idx];
}

// `masked_fill(input, mask, value)`: overwrites the masked positions with a
// scalar, an attribute rather than an operand, so it is a kernel parameter
// here rather than a third pointer.
extern "C" __global__ void masked_fill_op(
    const float* __restrict__ input,
    const unsigned char* __restrict__ mask,
    float value,
    float* __restrict__ out,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = mask[idx] ? value : input[idx];
}
