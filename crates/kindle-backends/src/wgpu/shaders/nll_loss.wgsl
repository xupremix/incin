@group(0) @binding(0) var<storage, read> log_sm: array<f32>;
@group(0) @binding(1) var<storage, read> target_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

// params[0] = batch_size
// params[1] = n_classes

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch = params[0];
    let n_classes = params[1];
    if idx >= batch { return; }

    let t = u32(target_buf[idx]);
    if t < n_classes {
        out[idx] = -log_sm[idx * n_classes + t];
    } else {
        out[idx] = 0.0;
    }
}
