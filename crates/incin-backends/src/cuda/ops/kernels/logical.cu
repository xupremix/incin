// Elementwise logical connectives over `bool` storage, encoded the same way
// `compare.cu`/`select.cu` do: one byte per element, `unsigned char` 0x00/0x01
// matching Rust's `bool` directly. Both operands (for the binaries) and the
// single operand (for `logical_not_op`) are read pre-broadcast to one
// contiguous shape and element count by the Rust launcher, so none of these
// kernels does any stride or shape bookkeeping.
extern "C" __global__ void logical_and_op(
    const unsigned char* __restrict__ lhs,
    const unsigned char* __restrict__ rhs,
    unsigned char* __restrict__ out,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = (lhs[idx] != 0 && rhs[idx] != 0) ? 1 : 0;
}

extern "C" __global__ void logical_or_op(
    const unsigned char* __restrict__ lhs,
    const unsigned char* __restrict__ rhs,
    unsigned char* __restrict__ out,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = (lhs[idx] != 0 || rhs[idx] != 0) ? 1 : 0;
}

extern "C" __global__ void logical_not_op(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ out,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = (input[idx] == 0) ? 1 : 0;
}
