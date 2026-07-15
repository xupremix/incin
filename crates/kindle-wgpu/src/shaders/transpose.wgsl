// Transpose a 2D matrix [rows, cols] -> [cols, rows]
// params[0] = rows, params[1] = cols

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

// Use tile for coalesced memory access
var<workgroup> tile: array<array<f32, 17>, 16>; // 17 avoids bank conflicts

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let rows = params[0];
    let cols = params[1];

    let in_row = wg_id.y * 16u + local_id.y;
    let in_col = wg_id.x * 16u + local_id.x;

    if in_row < rows && in_col < cols {
        tile[local_id.y][local_id.x] = inp[in_row * cols + in_col];
    }
    workgroupBarrier();

    let out_row = wg_id.x * 16u + local_id.y;
    let out_col = wg_id.y * 16u + local_id.x;

    if out_row < cols && out_col < rows {
        out[out_row * rows + out_col] = tile[local_id.x][local_id.y];
    }
}
