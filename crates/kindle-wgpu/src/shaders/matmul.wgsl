@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Shape {
    M: u32,
    K: u32,
    N: u32,
}
@group(0) @binding(3) var<storage, read> shape: Shape;

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row < shape.M && col < shape.N) {
        var sum = 0.0;
        for (var k = 0u; k < shape.K; k = k + 1u) {
            sum = sum + lhs[row * shape.K + k] * rhs[k * shape.N + col];
        }
        out[row * shape.N + col] = sum;
    }
}
