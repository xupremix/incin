@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

// params: 
// 0: N
// 1: C_in
// 2: H_in
// 3: W_in
// 4: C_out
// 5: H_out
// 6: W_out
// 7: Kh
// 8: Kw
// 9: stride_h
// 10: stride_w
// 11: pad_h
// 12: pad_w
// 13: dil_h
// 14: dil_w
// 15: groups

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    let N = params[0];
    let C_in = params[1];
    let H_in = params[2];
    let W_in = params[3];
    let C_out = params[4];
    let H_out = params[5];
    let W_out = params[6];
    let Kh = params[7];
    let Kw = params[8];
    let stride_h = params[9];
    let stride_w = params[10];
    let pad_h = i32(params[11]);
    let pad_w = i32(params[12]);
    let dil_h = params[13];
    let dil_w = params[14];
    let groups = params[15];
    
    let total_out = N * C_out * H_out * W_out;
    if idx >= total_out { return; }
    
    let ow = idx % W_out;
    var rem = idx / W_out;
    let oh = rem % H_out;
    rem = rem / H_out;
    let oc = rem % C_out;
    let n = rem / C_out;
    
    let group = oc / (C_out / groups);
    let c_in_per_group = C_in / groups;
    let in_c_start = group * c_in_per_group;
    let in_c_end = in_c_start + c_in_per_group;
    
    var sum = 0.0;
    
    for (var ic = in_c_start; ic < in_c_end; ic = ic + 1u) {
        let weight_base = oc * c_in_per_group * Kh * Kw + (ic - in_c_start) * Kh * Kw;
        let in_base = n * C_in * H_in * W_in + ic * H_in * W_in;
        
        for (var kh = 0u; kh < Kh; kh = kh + 1u) {
            for (var kw = 0u; kw < Kw; kw = kw + 1u) {
                let ih = i32(oh * stride_h) + i32(kh * dil_h) - pad_h;
                let iw = i32(ow * stride_w) + i32(kw * dil_w) - pad_w;
                
                if ih >= 0 && ih < i32(H_in) && iw >= 0 && iw < i32(W_in) {
                    let w_idx = weight_base + kh * Kw + kw;
                    let i_idx = in_base + u32(ih) * W_in + u32(iw);
                    sum = sum + inp[i_idx] * weight[w_idx];
                }
            }
        }
    }
    
    out[idx] = sum;
}
