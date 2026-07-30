#include <metal_stdlib>
using namespace metal;

// MSL Reduction Operations Kernel (Sum, Max, Min)
kernel void reduce_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    constant uint& op_type [[buffer(3)]],
    threadgroup float* shared_data [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    float val = 0.0f;
    if (op_type == 1) { // Max
        val = -1e38f;
    } else if (op_type == 2) { // Min
        val = 1e38f;
    }

    if (gid < numel) {
        val = in[gid];
    }
    shared_data[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (op_type == 0) { // Sum
                shared_data[tid] += shared_data[tid + s];
            } else if (op_type == 1) { // Max
                shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
            } else if (op_type == 2) { // Min
                shared_data[tid] = min(shared_data[tid], shared_data[tid + s]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[group_id] = shared_data[0];
    }
}
