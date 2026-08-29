// N-dimensional shape operations: narrow (slice), paste (scatter into a
// zeroed larger buffer — narrow's backward), transpose, broadcast. Direct
// CUDA port of wgpu/shaders/shape.wgsl's op_mode 0/1/2/3.
//
// params layout (uint32_t[21], uploaded once per launch):
//   params[0]      = op_mode (0=narrow, 1=paste, 2=transpose, 3=broadcast)
//   params[1]      = rank
//   params[2]      = n_elements — output count for narrow/transpose/
//                    broadcast, but INPUT count for paste (paste iterates
//                    the smaller side and scatters into the larger,
//                    pre-zeroed output; the caller is responsible for
//                    zero-initializing `out` before launch)
//   params[3..9]   = out_shape, padded to 6 dims with leading 1s
//   params[9..15]  = inp_shape, padded to 6 dims with leading 1s
//   params[15..21] = aux: narrow/paste start offsets, or transpose's
//                    output-dim -> input-dim map, padded with 0s
//
// NVRTC has no default header search path, so `uint32_t` isn't available
// without an explicit typedef (unlike `size_t`, which is compiler builtin).
typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned long long uint64_t;

template <typename T>
__device__ void shape_op_impl(
    const T* __restrict__ inp,
    T* __restrict__ out,
    const uint32_t* __restrict__ params
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = params[2];
    if (idx >= n) return;

    const uint32_t op = params[0];

    if (op == 0) { // narrow (slice): idx ranges over out_shape
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
            uint32_t start = params[15 + d];
            uint32_t in_coord = multi_idx[d] + start;
            in_flat += in_coord * in_stride;
            in_stride *= s;
        }
        out[idx] = inp[in_flat];
    } else if (op == 1) { // paste: idx ranges over inp_shape (the smaller side)
        uint32_t multi_idx[6] = {0, 0, 0, 0, 0, 0};
        uint32_t rem = idx;
        for (int d = 5; d >= 0; d--) {
            uint32_t s = params[9 + d];
            multi_idx[d] = rem % s;
            rem = rem / s;
        }
        uint32_t out_flat = 0;
        uint32_t out_stride = 1;
        for (int d = 5; d >= 0; d--) {
            uint32_t s = params[3 + d];
            uint32_t start = params[15 + d];
            uint32_t out_coord = multi_idx[d] + start;
            out_flat += out_coord * out_stride;
            out_stride *= s;
        }
        out[out_flat] = inp[idx];
    } else if (op == 2) { // transpose: idx ranges over out_shape
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
            uint32_t mapped_dim = params[15 + d];
            uint32_t in_coord = multi_idx[mapped_dim];
            in_flat += in_coord * in_stride;
            in_stride *= s;
        }
        out[idx] = inp[in_flat];
    } else if (op == 3) { // broadcast: idx ranges over out_shape
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
}

extern "C" __global__ void shape_op(
    const float* __restrict__ inp,
    float* __restrict__ out,
    const uint32_t* __restrict__ params
) {
    shape_op_impl<float>(inp, out, params);
}

extern "C" __global__ void shape_op_8bit(
    const uint8_t* __restrict__ inp,
    uint8_t* __restrict__ out,
    const uint32_t* __restrict__ params
) {
    shape_op_impl<uint8_t>(inp, out, params);
}

extern "C" __global__ void shape_op_16bit(
    const uint16_t* __restrict__ inp,
    uint16_t* __restrict__ out,
    const uint32_t* __restrict__ params
) {
    shape_op_impl<uint16_t>(inp, out, params);
}

extern "C" __global__ void shape_op_32bit(
    const uint32_t* __restrict__ inp,
    uint32_t* __restrict__ out,
    const uint32_t* __restrict__ params
) {
    shape_op_impl<uint32_t>(inp, out, params);
}

extern "C" __global__ void shape_op_64bit(
    const uint64_t* __restrict__ inp,
    uint64_t* __restrict__ out,
    const uint32_t* __restrict__ params
) {
    shape_op_impl<uint64_t>(inp, out, params);
}
