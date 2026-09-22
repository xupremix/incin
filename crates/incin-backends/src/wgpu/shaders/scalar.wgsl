// Scalar operations: add_scalar, mul_scalar, sub_scalar, div_scalar,
//                    powf, clamp
// op_mode: 0=add_scalar, 1=mul_scalar, 2=sub_scalar, 3=div_scalar,
//          4=powf, 5=clamp

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
// params[0] = op_mode, params[1] = n_elements
// params[2] = f32 scalar (reinterpreted as u32 bits via bitcast);
//             for clamp this slot is the min bound
// params[3] = f32 max bound, read only by clamp (mode 5)

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[1];
    if idx >= n { return; }

    let op = params[0];

    if op == 0u {
        out[idx] = inp[idx] + bitcast<f32>(params[2]);
    } else if op == 1u {
        out[idx] = inp[idx] * bitcast<f32>(params[2]);
    } else if op == 2u {
        out[idx] = inp[idx] - bitcast<f32>(params[2]);
    } else if op == 3u {
        out[idx] = inp[idx] / bitcast<f32>(params[2]);
    } else if op == 4u {
        // WGSL's `pow` goes through exp(y*ln(x)) and returns NaN for any
        // negative base, even integer exponents. CPU's f32::powf handles
        // (-2)^2 = 4, so branch on sign and apply the parity flip for
        // integer exponents to stay in lockstep.
        let base = inp[idx];
        let expo = bitcast<f32>(params[2]);
        if (base >= 0.0) {
            out[idx] = pow(base, expo);
        } else {
            let n = round(expo);
            if (abs(expo - n) > 1e-6) {
                // Non-integer power of a negative base is NaN on both
                // sides; route through pow so the bit pattern matches.
                out[idx] = pow(base, expo);
            } else {
                let mag = pow(-base, n);
                let half = n * 0.5;
                if (half == floor(half)) {
                    out[idx] = mag;
                } else {
                    out[idx] = -mag;
                }
            }
        }
    } else {
        // clamp: params[2] = min, params[3] = max. The attribute contract
        // has already refused NaN bounds and min > max.
        let lo = bitcast<f32>(params[2]);
        let hi = bitcast<f32>(params[3]);
        out[idx] = clamp(inp[idx], lo, hi);
    }
}
