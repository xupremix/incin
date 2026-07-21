// N-dimensional shape operations: slice, paste, transpose, broadcast
// op_mode: 0=slice, 1=paste, 2=transpose, 3=broadcast
// Maximum rank supported: 6

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

// params layout:
// params[0] = op_mode
// params[1] = rank
// params[2] = n_elements
// params[3..9]   = out_shape (padded to 6 with 1s)
// params[9..15]  = inp_shape (padded to 6 with 1s)
// params[15..21] = aux (starts for slice/paste, perm for transpose)

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[2];
    if idx >= n { return; }

    let op = params[0];
    let rank = params[1];

    if op == 0u { // slice
        // out_shape determines our multi_idx
        var multi_idx = array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
        var rem = idx;
        for (var d = 5u; d >= 0u; d = d - 1u) {
            let s = params[3u + d];
            multi_idx[d] = rem % s;
            rem = rem / s;
            if d == 0u { break; }
        }

        // input flat idx
        var in_flat = 0u;
        var in_stride = 1u;
        for (var d = 5u; d >= 0u; d = d - 1u) {
            let s = params[9u + d];
            let start = params[15u + d];
            let in_coord = multi_idx[d] + start;
            in_flat = in_flat + in_coord * in_stride;
            in_stride = in_stride * s;
            if d == 0u { break; }
        }
        out[idx] = inp[in_flat];
    } else if op == 1u { // paste
        // inp_shape determines our multi_idx (since n_elements = inp size)
        var multi_idx = array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
        var rem = idx;
        for (var d = 5u; d >= 0u; d = d - 1u) {
            let s = params[9u + d];
            multi_idx[d] = rem % s;
            rem = rem / s;
            if d == 0u { break; }
        }

        // output flat idx
        var out_flat = 0u;
        var out_stride = 1u;
        for (var d = 5u; d >= 0u; d = d - 1u) {
            let s = params[3u + d];
            let start = params[15u + d];
            let out_coord = multi_idx[d] + start;
            out_flat = out_flat + out_coord * out_stride;
            out_stride = out_stride * s;
            if d == 0u { break; }
        }
        out[out_flat] = inp[idx];
    } else if op == 2u { // transpose
        // out_shape determines our multi_idx
        var multi_idx = array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
        var rem = idx;
        for (var d = 5u; d >= 0u; d = d - 1u) {
            let s = params[3u + d];
            multi_idx[d] = rem % s;
            rem = rem / s;
            if d == 0u { break; }
        }

        var in_flat = 0u;
        var in_stride = 1u;
        for (var d = 5u; d >= 0u; d = d - 1u) {
            let s = params[9u + d];
            // aux holds a map from input_dim to output_dim
            // so we look up the coordinate in multi_idx using the mapped dim
            let mapped_dim = params[15u + d]; 
            let in_coord = multi_idx[mapped_dim];
            in_flat = in_flat + in_coord * in_stride;
            in_stride = in_stride * s;
            if d == 0u { break; }
        }
        out[idx] = inp[in_flat];
    } else if op == 3u { // broadcast
        var multi_idx = array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
        var rem = idx;
        for (var d = 5u; d >= 0u; d = d - 1u) {
            let s = params[3u + d];
            multi_idx[d] = rem % s;
            rem = rem / s;
            if d == 0u { break; }
        }

        var in_flat = 0u;
        var in_stride = 1u;
        for (var d = 5u; d >= 0u; d = d - 1u) {
            let s = params[9u + d];
            let out_coord = multi_idx[d];
            // If dimension was broadcasted (size 1), coord is 0
            let in_coord = select(out_coord, 0u, s == 1u);
            in_flat = in_flat + in_coord * in_stride;
            in_stride = in_stride * s;
            if d == 0u { break; }
        }
        out[idx] = inp[in_flat];
    }
}
