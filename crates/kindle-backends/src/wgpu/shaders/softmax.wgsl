// Softmax over a batch: shape [batch, n]
// Each workgroup handles one row (batch element)
// params[0] = batch, params[1] = n (row length)

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

var<workgroup> wg_max: f32;
var<workgroup> wg_sum: f32;
var<workgroup> shared_vals: array<f32, 256>;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let batch = params[0];
    let n = params[1];
    let row = wg_id.x;
    let lid = local_id.x;

    if row >= batch { return; }

    let base = row * n;

    // Step 1: find max for numerical stability
    var local_max = -3.4028235e+38;
    var i = lid;
    loop {
        if i >= n { break; }
        local_max = max(local_max, inp[base + i]);
        i += 256u;
    }
    shared_vals[lid] = local_max;
    workgroupBarrier();

    // Tree-reduce max
    var s = 128u;
    loop {
        if s == 0u { break; }
        if lid < s {
            shared_vals[lid] = max(shared_vals[lid], shared_vals[lid + s]);
        }
        workgroupBarrier();
        if s == 1u { break; }
        s = s >> 1u;
    }
    if lid == 0u { wg_max = shared_vals[0]; }
    workgroupBarrier();

    let row_max = wg_max;

    // Step 2: compute exp(x - max) and partial sum
    var local_sum = 0.0f;
    var j = lid;
    loop {
        if j >= n { break; }
        let e = exp(inp[base + j] - row_max);
        out[base + j] = e;
        local_sum += e;
        j += 256u;
    }
    shared_vals[lid] = local_sum;
    workgroupBarrier();

    // Tree-reduce sum
    s = 128u;
    loop {
        if s == 0u { break; }
        if lid < s {
            shared_vals[lid] = shared_vals[lid] + shared_vals[lid + s];
        }
        workgroupBarrier();
        if s == 1u { break; }
        s = s >> 1u;
    }
    if lid == 0u { wg_sum = shared_vals[0]; }
    workgroupBarrier();

    let row_sum = wg_sum;

    // Step 3: normalize
    var k = lid;
    loop {
        if k >= n { break; }
        out[base + k] = out[base + k] / row_sum;
        k += 256u;
    }
}
