@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> rm: array<f32>;
@group(0) @binding(4) var<storage, read> rv: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<storage, read> params: array<f32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ch = global_id.x;
    let c = u32(params[1]);
    if ch >= c { return; }

    let eps = params[0];
    let spatial = u32(params[2]);
    let batch = u32(params[3]);
    
    var mean = 0.0;
    var variance = 1.0;
    
    if params[6] > 0.0 {
        mean = rm[ch];
        variance = rv[ch];
    } else {
        var sum = 0.0;
        for (var n = 0u; n < batch; n = n + 1u) {
            for (var s = 0u; s < spatial; s = s + 1u) {
                sum = sum + inp[n * c * spatial + ch * spatial + s];
            }
        }
        mean = sum / f32(batch * spatial);

        var var_sum = 0.0;
        for (var n = 0u; n < batch; n = n + 1u) {
            for (var s = 0u; s < spatial; s = s + 1u) {
                let diff = inp[n * c * spatial + ch * spatial + s] - mean;
                var_sum = var_sum + diff * diff;
            }
        }
        variance = var_sum / f32(batch * spatial);
    }
    
    let std = sqrt(variance + eps);
    var g = 1.0;
    if params[4] > 0.0 { g = gamma[ch]; }
    var b = 0.0;
    if params[5] > 0.0 { b = beta[ch]; }

    for (var n = 0u; n < batch; n = n + 1u) {
        for (var s = 0u; s < spatial; s = s + 1u) {
            let idx = n * c * spatial + ch * spatial + s;
            out[idx] = (inp[idx] - mean) / std * g + b;
        }
    }
}
