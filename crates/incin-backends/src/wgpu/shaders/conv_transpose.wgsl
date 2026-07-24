@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
    batch: u32,
    c_in: u32,
    c_out: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dil_h: u32,
    dil_w: u32,
    groups: u32,
}
@group(0) @binding(3) var<storage, read> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let total_out = p.batch * p.c_out * p.h_out * p.w_out;
    if (id >= total_out) {
        return;
    }

    let w_out = id % p.w_out;
    let h_out = (id / p.w_out) % p.h_out;
    let c_out_idx = (id / (p.h_out * p.w_out)) % p.c_out;
    let b = id / (p.c_out * p.h_out * p.w_out);

    var sum = 0.0;
    
    let c_in_per_group = p.c_in / p.groups;
    let c_out_per_group = p.c_out / p.groups;
    let g = c_out_idx / c_out_per_group;

    let c_in_start = g * c_in_per_group;

    for (var ci = 0u; ci < c_in_per_group; ci = ci + 1u) {
        let c_in_actual = c_in_start + ci;
        for (var kh_i = 0u; kh_i < p.kh; kh_i = kh_i + 1u) {
            for (var kw_i = 0u; kw_i < p.kw; kw_i = kw_i + 1u) {
                
                let h_in_num = h_out + p.pad_h - kh_i * p.dil_h;
                let w_in_num = w_out + p.pad_w - kw_i * p.dil_w;

                if (h_in_num % p.stride_h == 0u && w_in_num % p.stride_w == 0u) {
                    let h_in_idx = h_in_num / p.stride_h;
                    let w_in_idx = w_in_num / p.stride_w;

                    if (h_in_idx >= 0u && h_in_idx < p.h_in && w_in_idx >= 0u && w_in_idx < p.w_in) {
                        let in_idx = b * (p.c_in * p.h_in * p.w_in) + c_in_actual * (p.h_in * p.w_in) + h_in_idx * p.w_in + w_in_idx;
                        
                        // weight shape for conv_transpose2d is usually [in_channels, out_channels/groups, kh, kw]
                        // PyTorch format: weight is [C_in, C_out/groups, kH, kW]
                        let w_idx = c_in_actual * (c_out_per_group * p.kh * p.kw) + (c_out_idx % c_out_per_group) * (p.kh * p.kw) + kh_i * p.kw + kw_i;
                        
                        sum = sum + inp[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    
    out[id] = sum;
}
