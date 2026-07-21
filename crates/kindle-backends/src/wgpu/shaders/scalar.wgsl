// Scalar operations: add_scalar, mul_scalar
// op_mode: 0=add_scalar, 1=mul_scalar

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
// params[0] = op_mode, params[1] = n_elements
// params[2..3] = f32 scalar (reinterpreted as u32 bits via bitcast)

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[1];
    if idx >= n { return; }

    let scalar = bitcast<f32>(params[2]);
    let op = params[0];

    if op == 0u {
        out[idx] = inp[idx] + scalar;
    } else {
        out[idx] = inp[idx] * scalar;
    }
}
