// Native single-pass online-softmax attention (issue #104).
//
// Forward is one block per query row: streaming key/value rows update the
// running maximum `m`, normalizer `l` and per-thread accumulator, so no
// `[seq_q, seq_kv]` score matrix is ever materialized. Backward stores
// nothing but per-row `(m, l)` and recomputes weights from them in three
// row-parallel kernels (delta, dq, dkv), so concurrent blocks never share
// an output row and no atomics are needed. Accumulation is in double for
// every storage dtype (f32/f64/f16/bf16 convert on load, round once on
// store), matching CPU's f64 intermediates within test tolerance.
//
// GQA maps query head `hq` to key/value head `hq / groups` by integer
// division; causal masking keeps key `j` for query `i` iff
// `j <= i + kv_lead`, exactly the CPU kernel's rule (square inputs give
// the lower triangle, `seq_kv > seq_q` lets a decode step see the cached
// prefix plus itself).
//
// `blockDim.x` covers `head_dim` (validated <= 1024 at launch); reductions
// use shared memory so any head dim works. The decode loop is
// straight-line over the key index with the causal test on `j` only -
// never an `if (d == axis)`-style loop-carried conditional (see
// repeat_interleave's NVVM range-split note).
//
// Above a small sequence length the forward also runs the tiled variant
// below (`ATTN_QBLOCK` query rows per block), which is where the
// remaining global-memory traffic went. Same arithmetic in the same
// order: only the K/V loads are shared.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Query rows per block in the tiled forward. Each thread keeps this many
// (q, m, l, acc) tuples in registers and streams the key/value rows once
// for all of them, so global K/V traffic drops by this factor.
#define ATTN_QBLOCK 4

// Load/store arithmetic per storage dtype: everything substantial runs
// in double, conversions happen once at the buffer edge.
template <typename T>
struct AttnArit;
template <>
struct AttnArit<float> {
    static __device__ __forceinline__ double load(float v) { return (double)v; }
    static __device__ __forceinline__ float store(double v) { return (float)v; }
};
template <>
struct AttnArit<double> {
    static __device__ __forceinline__ double load(double v) { return v; }
    static __device__ __forceinline__ double store(double v) { return v; }
};
template <>
struct AttnArit<__half> {
    static __device__ __forceinline__ double load(__half v) { return (double)__half2float(v); }
    static __device__ __forceinline__ __half store(double v) { return __float2half((float)v); }
};
template <>
struct AttnArit<__nv_bfloat16> {
    static __device__ __forceinline__ double load(__nv_bfloat16 v) {
        return (double)__bfloat162float(v);
    }
    static __device__ __forceinline__ __nv_bfloat16 store(double v) {
        return __float2bfloat16((float)v);
    }
};

// Block-wide sum over blockDim.x doubles in `red` (size blockDim.x).
// Handles non-power-of-two widths via the bounds check.
__device__ double block_reduce_sum(double val, double *red) {
    int tid = threadIdx.x;
    int n = blockDim.x;
    red[tid] = val;
    __syncthreads();
    for (int s = (n + 1) / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    return red[0];
}

// The same reduction for `QB` independent values at once: one tree walk,
// `log2(n)` barriers, instead of `QB` walks. `red` is `QB * blockDim.x`
// doubles, one slice per value. This is what makes a query tile pay off -
// sharing the key/value loads alone leaves the barrier count (and so the
// time) flat, because a block-wide reduction per key is what the kernel
// actually spends its cycles on.
//
// The leading barrier is load-bearing: every thread reads the previous
// call's totals out of `red` after that call's final barrier, so without
// it a thread that races ahead could overwrite `red[0]` before a slower
// peer has read it. (`block_reduce_sum`, above, has that latent race; it
// is left as it is because the one-row kernel's barrier count is its
// decode-path cost.)
template <int QB>
__device__ void block_reduce_sum_q(double (&val)[QB], double *red) {
    int tid = threadIdx.x;
    int n = blockDim.x;
    __syncthreads();
    for (int r = 0; r < QB; r++) {
        red[r * n + tid] = val[r];
    }
    __syncthreads();
    for (int s = (n + 1) / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < n) {
            for (int r = 0; r < QB; r++) {
                red[r * n + tid] += red[r * n + tid + s];
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < QB; r++) {
        val[r] = red[r * n];
    }
}

template <typename T>
__device__ void attn_fwd_row(
    const T *__restrict__ q,
    const T *__restrict__ k,
    const T *__restrict__ v,
    T *__restrict__ o,
    double *__restrict__ ml,
    int B,
    int Hq,
    int Hkv,
    int Sq,
    int Skv,
    int D,
    int groups,
    int kv_lead,
    double scale,
    int causal,
    long long q_off,
    long long k_off,
    long long v_off,
    long long o_off,
    double *red) {
    int tid = threadIdx.x;
    long long row = (long long)blockIdx.x;
    long long rows = (long long)B * Hq * Sq;
    if (row >= rows || tid >= D) {
        return;
    }
    int tmp = (int)row;
    int i = tmp % Sq;
    tmp /= Sq;
    int hq = tmp % Hq;
    int b = tmp / Hq;
    int hkv = hq / groups;

    long long q_row = q_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
    double qd = AttnArit<T>::load(q[q_row + tid]);

    double m = -1e300;
    double l = 0.0;
    double acc = 0.0;
    for (int j = 0; j < Skv; j++) {
        if (causal && j > i + kv_lead) {
            break;
        }
        long long k_row = k_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
        double s = scale * block_reduce_sum(qd * AttnArit<T>::load(k[k_row + tid]), red);
        double m_new = m > s ? m : s;
        double w_old = exp(m - m_new);
        double w_new = exp(s - m_new);
        l = l * w_old + w_new;
        long long v_row = v_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
        acc = acc * w_old + w_new * AttnArit<T>::load(v[v_row + tid]);
        m = m_new;
    }
    long long o_row = o_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
    if (l == 0.0) {
        // Fully masked row (only an empty key extent can do this): zeros,
        // matching the CPU kernel's documented rule.
        o[o_row + tid] = AttnArit<T>::store(0.0);
        ml[2 * row] = -1e300;
        ml[2 * row + 1] = 0.0;
    } else {
        o[o_row + tid] = AttnArit<T>::store(acc / l);
        ml[2 * row] = m;
        ml[2 * row + 1] = l;
    }
}

// Tiled forward: one block per `QB` query rows of one (b, hq) head, so
// the key/value row a thread loads for key `j` is reused by every query
// row in the block and global K/V traffic falls by `QB`. The tile lives
// in a register, not shared memory: `blockDim.x` already spans the whole
// head dim, so one element per key per thread is the entire tile.
//
// The arithmetic is unchanged and, with one exception below, so is its
// order: a query row still sees keys 0..min(Skv-1, i + kv_lead) in
// ascending order through the same online-softmax recurrence, so the
// output is bit-identical to `attn_fwd_row`. The exception is that a
// causal block now walks keys up to the *last* row's bound and masks the
// rows below it. A masked key is excluded from its row's dot product and
// then given a score of -1e300, which leaves `m` untouched (the running
// max is always at least a real score once key 0 is reached, and key 0 is
// never masked), contributes `exp(-1e300 - m) == 0.0` to `l`, and
// contributes `0.0 * v` to `acc` - so it is arithmetically inert rather
// than skipped.
//
// The j loop is straight-line and its bound is computed before the loop
// from the block's last query row, which also skips the fully-masked tail
// of a causal block (block-sparse causal skipping, coarsely).
template <typename T, int QB>
__device__ void attn_fwd_qblock(
    const T *__restrict__ q,
    const T *__restrict__ k,
    const T *__restrict__ v,
    T *__restrict__ o,
    double *__restrict__ ml,
    int B,
    int Hq,
    int Hkv,
    int Sq,
    int Skv,
    int D,
    int groups,
    int kv_lead,
    double scale,
    int causal,
    long long q_off,
    long long k_off,
    long long v_off,
    long long o_off,
    double *red) {
    int tid = threadIdx.x;
    if (tid >= D) {
        return;
    }
    // blockIdx enumerates (b, hq, tile) - decomposed against the *tiled*
    // row count so a sequence length that is not a multiple of QB still
    // maps to the right tile.
    int tiles = (Sq + QB - 1) / QB;
    long long block = (long long)blockIdx.x;
    int tmp = (int)(block / tiles);
    int i0 = (int)(block - (long long)tmp * tiles) * QB;
    int hq = tmp % Hq;
    int b = tmp / Hq;
    int hkv = hq / groups;

    // Ragged last tile: rows past Sq contribute nothing and are not stored.
    int live = Sq - i0;
    if (live > QB) {
        live = QB;
    }
    long long head_q = ((long long)b * Hq + hq) * Sq;
    long long head_kv = ((long long)b * Hkv + hkv) * Skv;

    double qd[QB], m[QB], l[QB], acc[QB];
    for (int r = 0; r < QB; r++) {
        m[r] = -1e300;
        l[r] = 0.0;
        acc[r] = 0.0;
        qd[r] = (r < live) ? AttnArit<T>::load(q[q_off + (head_q + i0 + r) * D + tid])
                           : 0.0;
    }

    int j_hi = Skv - 1;
    if (causal && i0 + live - 1 + kv_lead < j_hi) {
        j_hi = i0 + live - 1 + kv_lead;
    }
    for (int j = 0; j <= j_hi; j++) {
        // One global load per (block, key): every query row in the block
        // reuses both values from registers.
        long long kv_row = (head_kv + j) * D;
        double kd = AttnArit<T>::load(k[k_off + kv_row + tid]);
        double vd = AttnArit<T>::load(v[v_off + kv_row + tid]);
        // Causal rows below this key contribute nothing to their own dot,
        // so they contribute nothing to the shared reduction either; the
        // running maximum then leaves them untouched below.
        double scores[QB];
        for (int r = 0; r < QB; r++) {
            scores[r] = (causal && j > i0 + r + kv_lead) ? 0.0 : qd[r] * kd;
        }
        block_reduce_sum_q<QB>(scores, red);
        for (int r = 0; r < QB; r++) {
            double s = (causal && j > i0 + r + kv_lead) ? -1e300 : scale * scores[r];
            double m_new = m[r] > s ? m[r] : s;
            double w_old = exp(m[r] - m_new);
            double w_new = exp(s - m_new);
            l[r] = l[r] * w_old + w_new;
            acc[r] = acc[r] * w_old + w_new * vd;
            m[r] = m_new;
        }
    }
    for (int r = 0; r < QB; r++) {
        if (r >= live) {
            break;
        }
        long long row = head_q + i0 + r;
        if (l[r] == 0.0) {
            o[o_off + row * D + tid] = AttnArit<T>::store(0.0);
            ml[2 * row] = -1e300;
            ml[2 * row + 1] = 0.0;
        } else {
            o[o_off + row * D + tid] = AttnArit<T>::store(acc[r] / l[r]);
            ml[2 * row] = m[r];
            ml[2 * row + 1] = l[r];
        }
    }
}

// delta[b, hq, i] = dot(dO, O): hoisted out of the dk/dv kernel so that
// kernel stays a single pass with no redundant reductions.
template <typename T>
__device__ void attn_delta_row(
    const T *__restrict__ grad,
    const T *__restrict__ o,
    double *__restrict__ delta,
    int B,
    int Hq,
    int Sq,
    int D,
    long long g_off,
    long long o_off,
    double *red) {
    int tid = threadIdx.x;
    long long row = (long long)blockIdx.x;
    long long rows = (long long)B * Hq * Sq;
    if (row >= rows || tid >= D) {
        return;
    }
    int tmp = (int)row;
    int i = tmp % Sq;
    tmp /= Sq;
    int hq = tmp % Hq;
    int b = tmp / Hq;
    long long g_row = g_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
    long long o_row = o_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
    double d = block_reduce_sum(
        AttnArit<T>::load(grad[g_row + tid]) * AttnArit<T>::load(o[o_row + tid]), red);
    if (tid == 0) {
        delta[row] = d;
    }
}

// dq[b, hq, i] rowwise: recompute this row's weights from the stored
// (m, l), then ds = p * (dO.v - delta), dq += scale * ds * K.
// Each block owns its dq row exclusively.
template <typename T>
__device__ void attn_dq_row(
    const T *__restrict__ q,
    const T *__restrict__ k,
    const T *__restrict__ v,
    const T *__restrict__ grad,
    const double *__restrict__ ml,
    const double *__restrict__ delta,
    T *__restrict__ dq,
    int B,
    int Hq,
    int Hkv,
    int Sq,
    int Skv,
    int D,
    int groups,
    int kv_lead,
    double scale,
    int causal,
    long long q_off,
    long long k_off,
    long long v_off,
    long long g_off,
    long long dq_off,
    double *red) {
    int tid = threadIdx.x;
    long long row = (long long)blockIdx.x;
    long long rows = (long long)B * Hq * Sq;
    if (row >= rows || tid >= D) {
        return;
    }
    int tmp = (int)row;
    int i = tmp % Sq;
    tmp /= Sq;
    int hq = tmp % Hq;
    int b = tmp / Hq;
    int hkv = hq / groups;

    double m = ml[2 * row];
    double l = ml[2 * row + 1];
    double dl = delta[row];
    long long g_row = g_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
    double dq_d = 0.0;
    if (l != 0.0) {
        for (int j = 0; j < Skv; j++) {
            if (causal && j > i + kv_lead) {
                break;
            }
            long long k_row =
                k_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
            long long v_row =
                v_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
            double s = scale * block_reduce_sum(
                AttnArit<T>::load(q[q_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D + tid]) *
                    AttnArit<T>::load(k[k_row + tid]),
                red);
            double p = exp(s - m) / l;
            double dot_dv = block_reduce_sum(
                AttnArit<T>::load(grad[g_row + tid]) * AttnArit<T>::load(v[v_row + tid]), red);
            double ds = p * (dot_dv - dl);
            dq_d += scale * ds * AttnArit<T>::load(k[k_row + tid]);
        }
    }
    long long dq_row = dq_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
    dq[dq_row + tid] = AttnArit<T>::store(dq_d);
}

// dk/dv rows are each owned by one (b, hkv, j) block: loop the query rows
// (and GQA group) that observe j, accumulating with no atomics.
template <typename T>
__device__ void attn_dkv_row(
    const T *__restrict__ q,
    const T *__restrict__ k,
    const T *__restrict__ v,
    const T *__restrict__ grad,
    const double *__restrict__ ml,
    const double *__restrict__ delta,
    T *__restrict__ dk,
    T *__restrict__ dv,
    int B,
    int Hq,
    int Hkv,
    int Sq,
    int Skv,
    int D,
    int groups,
    int kv_lead,
    double scale,
    int causal,
    long long q_off,
    long long k_off,
    long long v_off,
    long long g_off,
    long long dk_off,
    long long dv_off,
    double *red) {
    int tid = threadIdx.x;
    long long row = (long long)blockIdx.x;
    long long rows = (long long)B * Hkv * Skv;
    if (row >= rows || tid >= D) {
        return;
    }
    int tmp = (int)row;
    int j = tmp % Skv;
    tmp /= Skv;
    int hkv = tmp % Hkv;
    int b = tmp / Hkv;

    double dk_d = 0.0;
    double dv_d = 0.0;
    int i_start = 0;
    if (causal) {
        i_start = j - kv_lead;
        if (i_start < 0) {
            i_start = 0;
        }
    }
    long long k_row = k_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
    long long v_row = v_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
    double kd = AttnArit<T>::load(k[k_row + tid]);
    double vd = AttnArit<T>::load(v[v_row + tid]);
    for (int i = i_start; i < Sq; i++) {
        for (int h = 0; h < groups; h++) {
            int hq = hkv * groups + h;
            long long qgrow = ((long long)b * Hq + hq) * Sq + i;
            double m = ml[2 * qgrow];
            double l = ml[2 * qgrow + 1];
            if (l == 0.0) {
                continue;
            }
            long long q_row = q_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
            long long g_row = g_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
            double s = scale * block_reduce_sum(AttnArit<T>::load(q[q_row + tid]) * kd, red);
            double p = exp(s - m) / l;
            double dot_dv = block_reduce_sum(AttnArit<T>::load(grad[g_row + tid]) * vd, red);
            double ds = p * (dot_dv - delta[qgrow]);
            double qd = AttnArit<T>::load(q[q_row + tid]);
            double gd = AttnArit<T>::load(grad[g_row + tid]);
            dk_d += scale * ds * qd;
            dv_d += p * gd;
        }
    }
    long long dk_row = dk_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
    long long dv_row = dv_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
    dk[dk_row + tid] = AttnArit<T>::store(dk_d);
    dv[dv_row + tid] = AttnArit<T>::store(dv_d);
}

// Exported entry points, one pair per storage dtype the capability row
// admits. The wrappers only forward arguments; all logic lives in the
// templates above so f32 and f64 stay a single implementation.

extern "C" __global__ void attn_fwd_f32(
    const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
    float *__restrict__ o, double *__restrict__ ml,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long o_off) {
    extern __shared__ double red[];
    attn_fwd_row<float>(
        q, k, v, o, ml, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, o_off, red);
}

extern "C" __global__ void attn_fwd_f64(
    const double *__restrict__ q, const double *__restrict__ k, const double *__restrict__ v,
    double *__restrict__ o, double *__restrict__ ml,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long o_off) {
    extern __shared__ double red[];
    attn_fwd_row<double>(
        q, k, v, o, ml, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, o_off, red);
}

extern "C" __global__ void attn_fwd_tiled_f32(
    const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
    float *__restrict__ o, double *__restrict__ ml,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long o_off) {
    extern __shared__ double red[];
    attn_fwd_qblock<float, ATTN_QBLOCK>(
        q, k, v, o, ml, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, o_off, red);
}

extern "C" __global__ void attn_fwd_tiled_f64(
    const double *__restrict__ q, const double *__restrict__ k, const double *__restrict__ v,
    double *__restrict__ o, double *__restrict__ ml,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long o_off) {
    extern __shared__ double red[];
    attn_fwd_qblock<double, ATTN_QBLOCK>(
        q, k, v, o, ml, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, o_off, red);
}

extern "C" __global__ void attn_delta_f32(
    const float *__restrict__ grad, const float *__restrict__ o, double *__restrict__ delta,
    int B, int Hq, int Sq, int D, long long g_off, long long o_off) {
    extern __shared__ double red[];
    attn_delta_row<float>(grad, o, delta, B, Hq, Sq, D, g_off, o_off, red);
}

extern "C" __global__ void attn_delta_f64(
    const double *__restrict__ grad, const double *__restrict__ o, double *__restrict__ delta,
    int B, int Hq, int Sq, int D, long long g_off, long long o_off) {
    extern __shared__ double red[];
    attn_delta_row<double>(grad, o, delta, B, Hq, Sq, D, g_off, o_off, red);
}

extern "C" __global__ void attn_dq_f32(
    const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
    const float *__restrict__ grad, const double *__restrict__ ml, const double *__restrict__ delta,
    float *__restrict__ dq,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long g_off, long long dq_off) {
    extern __shared__ double red[];
    attn_dq_row<float>(
        q, k, v, grad, ml, delta, dq, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, g_off, dq_off, red);
}

extern "C" __global__ void attn_dq_f64(
    const double *__restrict__ q, const double *__restrict__ k, const double *__restrict__ v,
    const double *__restrict__ grad, const double *__restrict__ ml, const double *__restrict__ delta,
    double *__restrict__ dq,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long g_off, long long dq_off) {
    extern __shared__ double red[];
    attn_dq_row<double>(
        q, k, v, grad, ml, delta, dq, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, g_off, dq_off, red);
}

extern "C" __global__ void attn_dkv_f32(
    const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
    const float *__restrict__ grad, const double *__restrict__ ml, const double *__restrict__ delta,
    float *__restrict__ dk, float *__restrict__ dv,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long g_off,
    long long dk_off, long long dv_off) {
    extern __shared__ double red[];
    attn_dkv_row<float>(
        q, k, v, grad, ml, delta, dk, dv, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, g_off, dk_off, dv_off, red);
}

extern "C" __global__ void attn_dkv_f64(
    const double *__restrict__ q, const double *__restrict__ k, const double *__restrict__ v,
    const double *__restrict__ grad, const double *__restrict__ ml, const double *__restrict__ delta,
    double *__restrict__ dk, double *__restrict__ dv,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long g_off,
    long long dk_off, long long dv_off) {
    extern __shared__ double red[];
    attn_dkv_row<double>(
        q, k, v, grad, ml, delta, dk, dv, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, g_off, dk_off, dv_off, red);
}

// Half-precision entry points (f16/bf16 storage, double arithmetic):
// same templates instantiated at the two half types.

extern "C" __global__ void attn_fwd_f16(
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    __half *__restrict__ o, double *__restrict__ ml,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long o_off) {
    extern __shared__ double red[];
    attn_fwd_row<__half>(
        q, k, v, o, ml, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, o_off, red);
}

extern "C" __global__ void attn_fwd_bf16(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ o, double *__restrict__ ml,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long o_off) {
    extern __shared__ double red[];
    attn_fwd_row<__nv_bfloat16>(
        q, k, v, o, ml, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, o_off, red);
}

extern "C" __global__ void attn_fwd_tiled_f16(
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    __half *__restrict__ o, double *__restrict__ ml,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long o_off) {
    extern __shared__ double red[];
    attn_fwd_qblock<__half, ATTN_QBLOCK>(
        q, k, v, o, ml, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, o_off, red);
}

extern "C" __global__ void attn_fwd_tiled_bf16(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ o, double *__restrict__ ml,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long o_off) {
    extern __shared__ double red[];
    attn_fwd_qblock<__nv_bfloat16, ATTN_QBLOCK>(
        q, k, v, o, ml, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, o_off, red);
}

extern "C" __global__ void attn_delta_f16(
    const __half *__restrict__ grad, const __half *__restrict__ o, double *__restrict__ delta,
    int B, int Hq, int Sq, int D, long long g_off, long long o_off) {
    extern __shared__ double red[];
    attn_delta_row<__half>(grad, o, delta, B, Hq, Sq, D, g_off, o_off, red);
}

extern "C" __global__ void attn_delta_bf16(
    const __nv_bfloat16 *__restrict__ grad, const __nv_bfloat16 *__restrict__ o,
    double *__restrict__ delta,
    int B, int Hq, int Sq, int D, long long g_off, long long o_off) {
    extern __shared__ double red[];
    attn_delta_row<__nv_bfloat16>(grad, o, delta, B, Hq, Sq, D, g_off, o_off, red);
}

extern "C" __global__ void attn_dq_f16(
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    const __half *__restrict__ grad, const double *__restrict__ ml, const double *__restrict__ delta,
    __half *__restrict__ dq,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long g_off, long long dq_off) {
    extern __shared__ double red[];
    attn_dq_row<__half>(
        q, k, v, grad, ml, delta, dq, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, g_off, dq_off, red);
}

extern "C" __global__ void attn_dq_bf16(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const __nv_bfloat16 *__restrict__ grad,
    const double *__restrict__ ml, const double *__restrict__ delta,
    __nv_bfloat16 *__restrict__ dq,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long g_off, long long dq_off) {
    extern __shared__ double red[];
    attn_dq_row<__nv_bfloat16>(
        q, k, v, grad, ml, delta, dq, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, g_off, dq_off, red);
}

extern "C" __global__ void attn_dkv_f16(
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    const __half *__restrict__ grad, const double *__restrict__ ml, const double *__restrict__ delta,
    __half *__restrict__ dk, __half *__restrict__ dv,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long g_off,
    long long dk_off, long long dv_off) {
    extern __shared__ double red[];
    attn_dkv_row<__half>(
        q, k, v, grad, ml, delta, dk, dv, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, g_off, dk_off, dv_off, red);
}

extern "C" __global__ void attn_dkv_bf16(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, const __nv_bfloat16 *__restrict__ grad,
    const double *__restrict__ ml, const double *__restrict__ delta,
    __nv_bfloat16 *__restrict__ dk, __nv_bfloat16 *__restrict__ dv,
    int B, int Hq, int Hkv, int Sq, int Skv, int D, int groups,
    int kv_lead, double scale, int causal,
    long long q_off, long long k_off, long long v_off, long long g_off,
    long long dk_off, long long dv_off) {
    extern __shared__ double red[];
    attn_dkv_row<__nv_bfloat16>(
        q, k, v, grad, ml, delta, dk, dv, B, Hq, Hkv, Sq, Skv, D, groups, kv_lead, scale, causal,
        q_off, k_off, v_off, g_off, dk_off, dv_off, red);
}
