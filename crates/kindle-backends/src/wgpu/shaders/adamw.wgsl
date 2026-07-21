// Fused AdamW update (in-place).
// All arrays are flat f32 buffers of length N.
//
// params (as f32 bits):
//   [0] = N (u32 as u32)
//   [1] = lr     (f32 bits)
//   [2] = beta1  (f32 bits)
//   [3] = beta2  (f32 bits)
//   [4] = eps    (f32 bits)
//   [5] = wd     (weight_decay, f32 bits)
//   [6] = bc1    (1 - beta1^t, f32 bits)
//   [7] = bc2    (1 - beta2^t, f32 bits)

@group(0) @binding(0) var<storage, read_write> param_buf : array<f32>;
@group(0) @binding(1) var<storage, read>       grad_buf  : array<f32>;
@group(0) @binding(2) var<storage, read_write> m_buf     : array<f32>;
@group(0) @binding(3) var<storage, read_write> v_buf     : array<f32>;
@group(0) @binding(4) var<storage, read>       hp : array<u32>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let N = hp[0];
    if idx >= N { return; }

    let lr    = bitcast<f32>(hp[1]);
    let beta1 = bitcast<f32>(hp[2]);
    let beta2 = bitcast<f32>(hp[3]);
    let eps   = bitcast<f32>(hp[4]);
    let wd    = bitcast<f32>(hp[5]);
    let bc1   = bitcast<f32>(hp[6]);
    let bc2   = bitcast<f32>(hp[7]);

    let p = param_buf[idx];
    let g = grad_buf[idx];

    // Multiplicative weight decay to match fused_adamw.cu
    let p_wd = p - lr * wd * p;

    let mi = beta1 * m_buf[idx] + (1.0 - beta1) * g;
    let vi = beta2 * v_buf[idx] + (1.0 - beta2) * g * g;

    param_buf[idx] = p_wd - lr * mi / (sqrt(vi) + eps);
    m_buf[idx]     = mi;
    v_buf[idx]     = vi;
}
