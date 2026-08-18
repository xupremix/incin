// Mask-driven selection, the counterpart to `compare.cu`'s mask *producers*:
// these two kernels are what a `bool` mask is actually for. Both read a
// `bool` mask as `unsigned char` -- the same 0x00/0x01 encoding `compare.cu`
// writes -- alongside `float` data, and write a fresh `float` result. Both
// require every operand pre-broadcast to one contiguous shape and element
// count by the Rust launcher, the same precondition `compare.cu` states, so
// neither kernel does any stride or shape bookkeeping of its own.
//
// That broadcast is what the third kernel below, `broadcast_bool_op`, does
// for the `bool` mask itself, before either kernel above ever runs: a mask
// can legitimately arrive at a lower rank than the data it selects between
// (`where_cond`'s output shape is the broadcast of all three operands), and
// `shape.cu`'s own `shape_op` cannot answer that -- its data pointers are a
// hardcoded `float*`/`float*`, four bytes per element, and a `bool` buffer
// is one. This kernel is `shape_op`'s `op_mode == 3` case, verbatim, ported
// to `unsigned char`; the params layout it reads is exactly the one
// `shape.cu`'s own header comment documents, produced by the same
// `prepare_shape_params(3, ...)` on the Rust side.

typedef unsigned int uint32_t;

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

// Broadcasts a `bool` mask to `out_shape` -- `shape.cu`'s `shape_op`
// `op_mode == 3` case, ported from `float`/`float` to `unsigned char`/
// `unsigned char`. `idx` ranges over `out_shape` (`params[2]` elements);
// `params[3..9]` is `out_shape` and `params[9..15]` is the mask's own shape,
// both right-aligned/padded to 6 dims with leading 1s, exactly as
// `shape.cu`'s header comment documents.
extern "C" __global__ void broadcast_bool_op(
    const unsigned char* __restrict__ inp,
    unsigned char* __restrict__ out,
    const uint32_t* __restrict__ params
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = params[2];
    if (idx >= n) return;

    uint32_t multi_idx[6] = {0, 0, 0, 0, 0, 0};
    uint32_t rem = idx;
    for (int d = 5; d >= 0; d--) {
        uint32_t s = params[3 + d];
        multi_idx[d] = rem % s;
        rem = rem / s;
    }
    uint32_t in_flat = 0;
    uint32_t in_stride = 1;
    for (int d = 5; d >= 0; d--) {
        uint32_t s = params[9 + d];
        uint32_t out_coord = multi_idx[d];
        uint32_t in_coord = (s == 1) ? 0 : out_coord;
        in_flat += in_coord * in_stride;
        in_stride *= s;
    }
    out[idx] = inp[in_flat];
}
