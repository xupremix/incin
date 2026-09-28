// Native single-pass online-softmax attention (issue #104).
//
// Forward is one block per query row: streaming key/value rows update the
// running maximum `m`, normalizer `l` and per-thread accumulator, so no
// `[seq_q, seq_kv]` score matrix is ever materialized. Backward stores
// nothing but per-row `(m, l)` and recomputes weights from them in three
// row-parallel kernels (delta, dq, dkv), so concurrent blocks never share
// an output row and no atomics are needed. Accumulation is in double for
// both float and double operands; stores round once.
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
    double qd = (double)q[q_row + tid];

    double m = -1e300;
    double l = 0.0;
    double acc = 0.0;
    for (int j = 0; j < Skv; j++) {
        if (causal && j > i + kv_lead) {
            break;
        }
        long long k_row = k_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
        double s = scale * block_reduce_sum(qd * (double)k[k_row + tid], red);
        double m_new = m > s ? m : s;
        double w_old = exp(m - m_new);
        double w_new = exp(s - m_new);
        l = l * w_old + w_new;
        long long v_row = v_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
        acc = acc * w_old + w_new * (double)v[v_row + tid];
        m = m_new;
    }
    long long o_row = o_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
    if (l == 0.0) {
        // Fully masked row (only an empty key extent can do this): zeros,
        // matching the CPU kernel's documented rule.
        o[o_row + tid] = (T)0.0;
        ml[2 * row] = -1e300;
        ml[2 * row + 1] = 0.0;
    } else {
        o[o_row + tid] = (T)(acc / l);
        ml[2 * row] = m;
        ml[2 * row + 1] = l;
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
        (double)grad[g_row + tid] * (double)o[o_row + tid], red);
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
                (double)q[q_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D + tid] *
                    (double)k[k_row + tid],
                red);
            double p = exp(s - m) / l;
            double dot_dv = block_reduce_sum(
                (double)grad[g_row + tid] * (double)v[v_row + tid], red);
            double ds = p * (dot_dv - dl);
            dq_d += scale * ds * (double)k[k_row + tid];
        }
    }
    long long dq_row = dq_off + ((long long)b * Hq + hq) * Sq * D + (long long)i * D;
    dq[dq_row + tid] = (T)dq_d;
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
    double kd = (double)k[k_row + tid];
    double vd = (double)v[v_row + tid];
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
            double s = scale * block_reduce_sum((double)q[q_row + tid] * kd, red);
            double p = exp(s - m) / l;
            double dot_dv = block_reduce_sum((double)grad[g_row + tid] * vd, red);
            double ds = p * (dot_dv - delta[qgrow]);
            double qd = (double)q[q_row + tid];
            double gd = (double)grad[g_row + tid];
            dk_d += scale * ds * qd;
            dv_d += p * gd;
        }
    }
    long long dk_row = dk_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
    long long dv_row = dv_off + ((long long)b * Hkv + hkv) * Skv * D + (long long)j * D;
    dk[dk_row + tid] = (T)dk_d;
    dv[dv_row + tid] = (T)dv_d;
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
