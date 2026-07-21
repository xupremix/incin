@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Shape {
    M: u32,
    K: u32,
    N: u32,
    batch: u32,
    lhs_stride_b: u32,
    rhs_stride_b: u32,
}
@group(0) @binding(3) var<storage, read> shape: Shape;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) block_idx: vec3<u32>,
    @builtin(local_invocation_id) thread_idx: vec3<u32>
) {
    let b = block_idx.z;
    let row = block_idx.y * TILE_SIZE + thread_idx.y;
    let col = block_idx.x * TILE_SIZE + thread_idx.x;

    let M = shape.M;
    let K = shape.K;
    let N = shape.N;

    let lhs_offset = b * shape.lhs_stride_b;
    let rhs_offset = b * shape.rhs_stride_b;
    let out_offset = b * M * N;

    var sum = 0.0;
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let k_col = t * TILE_SIZE + thread_idx.x;
        if (row < M && k_col < K) {
            tile_a[thread_idx.y][thread_idx.x] = lhs[lhs_offset + row * K + k_col];
        } else {
            tile_a[thread_idx.y][thread_idx.x] = 0.0;
        }

        let k_row = t * TILE_SIZE + thread_idx.y;
        if (k_row < K && col < N) {
            tile_b[thread_idx.y][thread_idx.x] = rhs[rhs_offset + k_row * N + col];
        } else {
            tile_b[thread_idx.y][thread_idx.x] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[thread_idx.y][k] * tile_b[k][thread_idx.x];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        out[out_offset + row * N + col] = sum;
    }
}
