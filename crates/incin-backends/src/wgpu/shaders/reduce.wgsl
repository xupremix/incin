// Reduce along all elements (sum_all, mean_all) or a single dim (sum_dim, mean_dim)
// Two-pass: this shader does a parallel reduction within workgroups, outputting per-workgroup sums.
// The host then sums workgroup results or re-dispatches.
//
// params[0] = n_elements (total elements)
// params[1] = reduce_mode: 0 = sum, 1 = max, 2 = min, 3 = product

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

var<workgroup> shared_data: array<f32, 256>;

@compute
@workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let n = params[0];
    let mode = params[1];
    let lid = local_id.x;
    let gid = global_id.x;

    // Load with identity element
    var val: f32;
    if mode == 1u {
        val = -3.4028235e+38; // -FLT_MAX
    } else if mode == 2u {
        val = 3.4028235e+38;  // FLT_MAX
    } else if mode == 3u {
        val = 1.0; // product identity
    } else {
        val = 0.0;
    }

    if gid < n {
        val = inp[gid];
    }
    shared_data[lid] = val;
    workgroupBarrier();

    // Tree reduction in shared memory
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if lid < stride {
            if mode == 1u {
                shared_data[lid] = max(shared_data[lid], shared_data[lid + stride]);
            } else if mode == 2u {
                shared_data[lid] = min(shared_data[lid], shared_data[lid + stride]);
            } else if mode == 3u {
                shared_data[lid] = shared_data[lid] * shared_data[lid + stride];
            } else {
                shared_data[lid] = shared_data[lid] + shared_data[lid + stride];
            }
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride = stride >> 1u;
    }

    if lid == 0u {
        out[wg_id.x] = shared_data[0];
    }
}
