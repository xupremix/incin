// Binary elementwise operations: add, sub, mul, div
// op_mode: 0=add, 1=sub, 2=mul, 3=div

@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

// params[0] = op_mode, params[1] = n_elements

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEFF: f32 = 0.044715;

fn gelu_grad(g: f32, x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    let tanh_inner = tanh(inner);
    let cdf = 0.5 * (1.0 + tanh_inner);
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x * x);
    let pdf = 0.5 * x * sech2 * d_inner;
    return g * (cdf + pdf);
}

fn elu_grad(g: f32, x: f32) -> f32 {
    if (x > 0.0) {
        return g;
    } else {
        return g * exp(x);
    }
}

fn mish_grad(g: f32, x: f32) -> f32 {
    let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    let t_sp = tanh(sp);
    let sig_x = 1.0 / (1.0 + exp(-x));
    let sech2 = 1.0 - t_sp * t_sp;
    let deriv = t_sp + x * sech2 * sig_x;
    return g * deriv;
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[1];
    if idx >= n { return; }

    let op = params[0];
    if op == 0u {
        out[idx] = lhs[idx] + rhs[idx];
    } else if op == 1u {
        out[idx] = lhs[idx] - rhs[idx];
    } else if op == 2u {
        out[idx] = lhs[idx] * rhs[idx];
    } else if op == 3u {
        out[idx] = lhs[idx] / rhs[idx];
    } else if op == 4u {
        out[idx] = gelu_grad(lhs[idx], rhs[idx]);
    } else if op == 5u {
        out[idx] = elu_grad(lhs[idx], rhs[idx]);
    } else if op == 6u {
        out[idx] = mish_grad(lhs[idx], rhs[idx]);
    }
}
