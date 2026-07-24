@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

// params[0] = mode (0: adaptive_avg, 1: avg, 2: max)
// params[1] = N
// params[2] = C
// params[3] = H
// params[4] = W
// params[5] = OH
// params[6] = OW
// params[7] = kernel_h
// params[8] = kernel_w
// params[9] = stride_h
// params[10] = stride_w
// params[11] = pad_h
// params[12] = pad_w
// params[13] = dilation_h
// params[14] = dilation_w

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    let mode = params[0];
    let N = params[1];
    let C = params[2];
    let H = params[3];
    let W = params[4];
    let OH = params[5];
    let OW = params[6];
    
    let total_out = N * C * OH * OW;
    if idx >= total_out { return; }
    
    let oj = idx % OW;
    var rem = idx / OW;
    let oi = rem % OH;
    rem = rem / OH;
    let ci = rem % C;
    let bi = rem / C;
    
    let base_in = bi * C * H * W + ci * H * W;
    
    if mode == 0u { // adaptive_avg
        let h_start = oi * H / OH;
        let h_end = ((oi + 1u) * H + OH - 1u) / OH;
        let w_start = oj * W / OW;
        let w_end = ((oj + 1u) * W + OW - 1u) / OW;
        
        var sum = 0.0;
        var cnt = 0u;
        for (var hi = h_start; hi < h_end; hi = hi + 1u) {
            for (var wi = w_start; wi < w_end; wi = wi + 1u) {
                sum = sum + inp[base_in + hi * W + wi];
                cnt = cnt + 1u;
            }
        }
        out[idx] = sum / f32(cnt);
    } else {
        let kh = params[7];
        let kw = params[8];
        let sh = params[9];
        let sw = params[10];
        let ph = i32(params[11]);
        let pw = i32(params[12]);
        let dh = i32(params[13]);
        let dw = i32(params[14]);
        
        if mode == 1u { // avg
            var sum = 0.0;
            for (var ki = 0u; ki < kh; ki = ki + 1u) {
                for (var kj = 0u; kj < kw; kj = kj + 1u) {
                    let hi = i32(oi * sh + ki) - ph;
                    let wi = i32(oj * sw + kj) - pw;
                    if hi >= 0 && hi < i32(H) && wi >= 0 && wi < i32(W) {
                        sum = sum + inp[base_in + u32(hi) * W + u32(wi)];
                    }
                }
            }
            out[idx] = sum / f32(kh * kw);
        } else if mode == 2u { // max
            var max_val = -3.4028235e+38; // -FLT_MAX
            for (var ki = 0u; ki < kh; ki = ki + 1u) {
                for (var kj = 0u; kj < kw; kj = kj + 1u) {
                    let hi = i32(oi * sh) + i32(ki) * dh - ph;
                    let wi = i32(oj * sw) + i32(kj) * dw - pw;
                    if hi >= 0 && hi < i32(H) && wi >= 0 && wi < i32(W) {
                        let v = inp[base_in + u32(hi) * W + u32(wi)];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
            }
            out[idx] = max_val;
        }
    }
}
