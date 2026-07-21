@group(0) @binding(0) var<storage, read> indices: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

// params[0] = seq_len
// params[1] = embed_dim
// params[2] = vocab_size

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let seq_len = params[0];
    let embed_dim = params[1];
    let vocab_size = params[2];

    let total_elements = seq_len * embed_dim;
    if idx >= total_elements { return; }

    let seq_idx = idx / embed_dim;
    let embed_idx = idx % embed_dim;

    let v_idx = u32(indices[seq_idx]);
    if v_idx < vocab_size {
        out[idx] = weight[v_idx * embed_dim + embed_idx];
    } else {
        out[idx] = 0.0;
    }
}
