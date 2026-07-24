@group(0) @binding(0) var<storage, read_write> t: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

// params[0] = C (number of channels)
// params[1] = spatial (H * W)

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let C = params[0];
    let spatial = params[1];
    
    // idx = n * C * spatial + c * spatial + s
    let c = (idx / spatial) % C;
    
    // We don't have total_elements passed explicitly, but out-of-bounds
    // will just fault or return 0, which is fine if we dispatch exactly the right number.
    // However, it's safer to pass total_elements. We can pass it as params[2].
    let total = params[2];
    if idx >= total { return; }
    
    t[idx] = t[idx] + bias[c];
}
