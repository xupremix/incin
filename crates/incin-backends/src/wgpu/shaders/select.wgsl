// Selection between two value buffers by a bool mask (physically f32 0.0/1.0).
// op_mode: 0=where (out = mask ? a : b), 1=masked_fill (out = mask ? value : a)
// Maximum rank: none — buffers are pre-flattened to equal length by the
// host-side broadcast in `backend/indexing.rs`.
//
// params[0] = op_mode
// params[1] = n_elements
// params[2] = f32 fill value bit pattern (masked_fill only)

@group(0) @binding(0) var<storage, read> mask: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[1];
    if idx >= n { return; }

    let op = params[0];
    let take_a = mask[idx] != 0.0;
    if op == 0u {
        // where_cond: a is on_true, b is on_false.
        out[idx] = select(b[idx], a[idx], take_a);
    } else {
        // masked_fill: constant fill under the mask, input elsewhere.
        let value = bitcast<f32>(params[2]);
        out[idx] = select(a[idx], value, take_a);
    }
}
