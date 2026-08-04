// Unary elementwise operations
// op_mode: 0=relu, 1=gelu, 2=tanh, 3=sigmoid, 4=abs, 5=neg, 6=sqrt, 7=exp, 8=log, 9=swish,
//          10=step, 11=mish, 12=elu, 13=logical_not

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

// params[0] = op_mode, params[1] = n_elements

const PI: f32 = 3.14159265358979323846;
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;  // sqrt(2/pi)
const GELU_COEFF: f32 = 0.044715;

fn gelu_approx(x: f32) -> f32 {
    // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    return 0.5 * x * (1.0 + tanh(inner));
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params[1];
    if idx >= n { return; }

    let x = inp[idx];
    let op = params[0];

    if op == 0u {
        // ReLU
        out[idx] = max(x, 0.0);
    } else if op == 1u {
        // GELU (tanh approx)
        out[idx] = gelu_approx(x);
    } else if op == 2u {
        // Tanh
        out[idx] = tanh(x);
    } else if op == 3u {
        // Sigmoid
        out[idx] = 1.0 / (1.0 + exp(-x));
    } else if op == 4u {
        // Abs
        out[idx] = abs(x);
    } else if op == 5u {
        // Neg
        out[idx] = -x;
    } else if op == 6u {
        // Sqrt
        out[idx] = sqrt(x);
    } else if op == 7u {
        // Exp
        out[idx] = exp(x);
    } else if op == 8u {
        // Log
        out[idx] = log(x);
    } else if op == 9u {
        // Swish = x * sigmoid(x)
        out[idx] = x / (1.0 + exp(-x));
    } else if op == 10u {
        // Step
        if x > 0.0 {
            out[idx] = 1.0;
        } else {
            out[idx] = 0.0;
        }
    } else if op == 11u {
        // Mish = x * tanh(softplus(x))
        let sp = select(log(1.0 + exp(x)), x, x > 20.0);
        out[idx] = x * tanh(sp);
    } else if op == 13u {
        // Logical NOT
        out[idx] = select(0.0, 1.0, x == 0.0);
    } else {
        // ELU (alpha = 1.0)
        if x > 0.0 {
            out[idx] = x;
        } else {
            out[idx] = exp(x) - 1.0;
        }
    }
}
