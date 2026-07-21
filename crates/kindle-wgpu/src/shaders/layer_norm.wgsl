@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<f32>;

// params[0] = eps
// params[1] = norm_size
// params[2] = has_bias (1.0 or 0.0)
// params[3] = batch_size

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let batch_size = u32(params[3]);
    if batch_idx >= batch_size { return; }

    let norm_size = u32(params[1]);
    let eps = params[0];
    let has_bias = u32(params[2]);

    let base = batch_idx * norm_size;
    
    var sum = 0.0;
    for (var i = 0u; i < norm_size; i = i + 1u) {
        sum = sum + inp[base + i];
    }
    let mean = sum / f32(norm_size);

    var var_sum = 0.0;
    for (var i = 0u; i < norm_size; i = i + 1u) {
        let diff = inp[base + i] - mean;
        var_sum = var_sum + diff * diff;
    }
    let variance = var_sum / f32(norm_size);
    let std = sqrt(variance + eps);

    for (var i = 0u; i < norm_size; i = i + 1u) {
        let norm = (inp[base + i] - mean) / std;
        var b = 0.0;
        if has_bias > 0u {
            b = beta[i];
        }
        out[base + i] = norm * gamma[i] + b;
    }
}
