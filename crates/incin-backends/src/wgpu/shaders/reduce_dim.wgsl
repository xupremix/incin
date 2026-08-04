// Single dimension reduction: sum_dim, mean_dim, max_dim, min_dim, argmax, argmin, prod_dim
// op_mode: 0=sum, 1=mean, 2=max, 3=min, 4=argmax, 5=argmin, 6=product

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_n = params[3];
    if idx >= out_n { return; }

    let op = params[0];
    let dim_size = params[1];
    let inner_stride = params[2];

    let inner_idx = idx % inner_stride;
    let outer_idx = idx / inner_stride;
    let base_offset = outer_idx * (dim_size * inner_stride) + inner_idx;

    if op == 0u || op == 1u { // sum / mean
        var acc = 0.0;
        for (var k = 0u; k < dim_size; k = k + 1u) {
            acc = acc + inp[base_offset + k * inner_stride];
        }
        if op == 1u {
            acc = acc / f32(dim_size);
        }
        out[idx] = acc;
    } else if op == 2u { // max
        var acc = -3.4028235e+38; // -FLT_MAX
        for (var k = 0u; k < dim_size; k = k + 1u) {
            let v = inp[base_offset + k * inner_stride];
            if v > acc { acc = v; }
        }
        out[idx] = acc;
    } else if op == 3u { // min
        var acc = 3.4028235e+38; // FLT_MAX
        for (var k = 0u; k < dim_size; k = k + 1u) {
            let v = inp[base_offset + k * inner_stride];
            if v < acc { acc = v; }
        }
        out[idx] = acc;
    } else if op == 4u { // argmax
        var acc = -3.4028235e+38;
        var best_k = 0u;
        for (var k = 0u; k < dim_size; k = k + 1u) {
            let v = inp[base_offset + k * inner_stride];
            // Match CPU behavior which uses > for first max
            if v > acc || k == 0u {
                acc = v;
                best_k = k;
            }
        }
        out[idx] = bitcast<f32>(best_k);
    } else if op == 5u { // argmin
        var acc = 3.4028235e+38;
        var best_k = 0u;
        for (var k = 0u; k < dim_size; k = k + 1u) {
            let v = inp[base_offset + k * inner_stride];
            if v < acc || k == 0u {
                acc = v;
                best_k = k;
            }
        }
        out[idx] = bitcast<f32>(best_k);
    } else if op == 6u { // product
        var acc = 1.0;
        for (var k = 0u; k < dim_size; k = k + 1u) {
            acc = acc * inp[base_offset + k * inner_stride];
        }
        out[idx] = acc;
    }
}
