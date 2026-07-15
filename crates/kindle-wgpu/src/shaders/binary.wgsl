// Binary elementwise operations: add, sub, mul, div
// op_mode: 0=add, 1=sub, 2=mul, 3=div

@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

// params[0] = op_mode, params[1] = n_elements

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[1];
    if idx >= n { return; }

    let op = params[0];
    if op == 0u {
        out[idx] = lhs[idx] + rhs[idx];
    } else if op == 1u {
        out[idx] = lhs[idx] - rhs[idx];
    } else if op == 2u {
        out[idx] = lhs[idx] * rhs[idx];
    } else {
        out[idx] = lhs[idx] / rhs[idx];
    }
}
