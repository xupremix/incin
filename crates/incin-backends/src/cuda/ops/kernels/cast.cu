#include <cuda_fp16.h>
#include <cuda_bf16.h>

// NVRTC has no default header search path for host libc headers (only for
// the CUDA toolkit's own headers, via --include-path), so <stdint.h> isn't
// resolvable; define the one typedef we need instead.
typedef long long int64_t;

// Runtime type codes shared with `cuda/ops/cast.rs`:
//   source: 0=f32, 1=f64, 2=f16, 3=bf16, 4=i64, 5=bool
//   target: 0=f32, 1=f64, 2=f16, 3=bf16, 4=i64   (no bool/u8/u32/q8_0)
//
// One load-to-f64 / store-from-f64 path mirrors CPU's `tensor_to_dtype_storage`,
// which decodes every source through `get() -> f64` and narrows with `as`
// casts (saturating) and `half::{f16,bf16}::from_f64`.
__device__ __forceinline__ double cast_load(const void* src, int code, int64_t i) {
    switch (code) {
        case 0: return (double)((const float*)src)[i];
        case 1: return ((const double*)src)[i];
        case 2: return (double)__half2float(((const __half*)src)[i]);
        case 3: return (double)__bfloat162float(((const __nv_bfloat16*)src)[i]);
        case 4: return (double)((const int64_t*)src)[i];
        case 5: return ((const bool*)src)[i] ? 1.0 : 0.0;
        default: return 0.0;
    }
}

__device__ __forceinline__ void cast_store(void* dst, int code, int64_t i, double v) {
    switch (code) {
        case 0:
            ((float*)dst)[i] = (float)v;
            break;
        case 1:
            ((double*)dst)[i] = v;
            break;
        case 2:
            ((half*)dst)[i] = __double2half(v);
            break;
        case 3:
            // f64 -> bf16 through f32: bf16 keeps 9 significant bits, f32 keeps
            // 24, so the intermediate is wide enough that a single rounding at
            // bf16 precision is preserved for every finite input (same path the
            // host round-trip used).
            ((__nv_bfloat16*)dst)[i] = __float2bfloat16((float)v);
            break;
        case 4: {
            // Saturating f64 -> i64 matching Rust's `as`: NaN -> 0,
            // values at or beyond 2^63 saturate, otherwise truncate toward zero.
            int64_t r;
            if (!(v == v)) {
                r = 0;
            } else if (v >= 9223372036854775808.0) {
                r = 0x7fffffffffffffffLL;
            } else if (v <= -9223372036854775808.0) {
                r = (-0x7fffffffffffffffLL - 1LL);
            } else {
                r = (int64_t)v;
            }
            ((int64_t*)dst)[i] = r;
            break;
        }
        default:
            break;
    }
}

extern "C" __global__ void to_dtype_cast(
    const void* __restrict__ src,
    void* __restrict__ dst,
    const int64_t n,
    const int src_code,
    const int dst_code,
    const int64_t offset
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v = cast_load(src, src_code, i + offset);
    cast_store(dst, dst_code, i, v);
}
