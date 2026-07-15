// im2col for conv2d: [N, C_in, H, W] -> col matrix [N, C_in*Kh*Kw, Oh*Ow]
// Each thread writes one output element of the col matrix.
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

@group(0) @binding(0) var<storage, read>       inp     : array<f32>;
@group(0) @binding(1) var<storage, read_write> col_out : array<f32>;
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

    // Total output elements = N * C_in * Kh * Kw * H_out * W_out
    let col_channels = C_in * Kh * Kw;
    let spatial_out  = H_out * W_out;
    let total = N * col_channels * spatial_out;
    if idx >= total { return; }

    // Decompose idx: [n, c_k, oh_ow]
    let ow     = idx % W_out;
    let oh     = (idx / W_out) % H_out;
    let c_k    = (idx / spatial_out) % col_channels;
    let n      = idx / (col_channels * spatial_out);

    let kw     = c_k % Kw;
    let kh     = (c_k / Kw) % Kh;
    let c      = c_k / (Kh * Kw);

    let ih_i = i32(oh * sh + kh * dh) - i32(ph);
    let iw_i = i32(ow * sw + kw * dw) - i32(pw);

    var val = 0.0f;
    if ih_i >= 0 && iw_i >= 0 && u32(ih_i) < H_in && u32(iw_i) < W_in {
        let in_idx = n * C_in * H_in * W_in
                   + c * H_in * W_in
                   + u32(ih_i) * W_in
                   + u32(iw_i);
        val = inp[in_idx];
    }

    // col_out[n, c_k, oh*W_out+ow]
    let out_idx = n * col_channels * spatial_out + c_k * spatial_out + oh * W_out + ow;
    col_out[out_idx] = val;
}
