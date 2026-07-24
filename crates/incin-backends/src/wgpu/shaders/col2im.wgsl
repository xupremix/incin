// col2im for conv2d backward: [N, C_in*Kh*Kw, Oh*Ow] -> inp matrix [N, C_in, H_in, W_in]
// Each thread accumulates into one element of the input image.
// Note: Atomic operations are required if multiple threads accumulate to the same pixel,
// but since we assign one thread per input pixel, we can loop over the col matrix contributions.
//
// params layout (all u32):
//   0: N (batch)
//   1: C_in
//   2: H_in,  3: W_in
//   4: H_out, 5: W_out
//   6: Kh,    7: Kw
//   8: stride_h, 9: stride_w
//  10: pad_h,   11: pad_w
//  12: dil_h,   13: dil_w

@group(0) @binding(0) var<storage, read>       col_in  : array<f32>;
@group(0) @binding(1) var<storage, read_write> inp_out : array<f32>;
@group(0) @binding(2) var<storage, read>       params  : array<u32>;

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    let N      = params[0];
    let C_in   = params[1];
    let H_in   = params[2];
    let W_in   = params[3];
    let H_out  = params[4];
    let W_out  = params[5];
    let Kh     = params[6];
    let Kw     = params[7];
    let sh     = params[8];
    let sw     = params[9];
    let ph     = params[10];
    let pw     = params[11];
    let dh     = params[12];
    let dw     = params[13];

    let total_inp = N * C_in * H_in * W_in;
    if idx >= total_inp { return; }

    // Decompose idx: [n, c, h, w]
    let w = idx % W_in;
    let h = (idx / W_in) % H_in;
    let c = (idx / (H_in * W_in)) % C_in;
    let n = idx / (C_in * H_in * W_in);

    var val = 0.0f;

    let col_channels = C_in * Kh * Kw;
    let spatial_out = H_out * W_out;

    // We need to find all (kh, kw) such that they contributed to this (h, w).
    // The relationship is:
    // ih = oh * sh + kh * dh - ph  => oh = (ih + ph - kh * dh) / sh
    // iw = ow * sw + kw * dw - pw  => ow = (iw + pw - kw * dw) / sw
    // We only sum valid (oh, ow) that are integer solutions and within bounds.
    
    for (var kh = 0u; kh < Kh; kh = kh + 1u) {
        for (var kw = 0u; kw < Kw; kw = kw + 1u) {
            let h_num = h + ph - kh * dh;
            let w_num = w + pw - kw * dw;

            // In WGSL, subtraction of u32 that goes negative will wrap, so we can cast to i32 to be safe.
            let h_num_i = i32(h) + i32(ph) - i32(kh * dh);
            let w_num_i = i32(w) + i32(pw) - i32(kw * dw);

            if h_num_i >= 0 && w_num_i >= 0 && h_num_i % i32(sh) == 0 && w_num_i % i32(sw) == 0 {
                let oh = u32(h_num_i) / sh;
                let ow = u32(w_num_i) / sw;

                if oh < H_out && ow < W_out {
                    let c_k = c * (Kh * Kw) + kh * Kw + kw;
                    let col_idx = n * col_channels * spatial_out + c_k * spatial_out + oh * W_out + ow;
                    val = val + col_in[col_idx];
                }
            }
        }
    }

    inp_out[idx] = val;
}
