// Unary elementwise operations
// op_mode: 0=relu, 1=gelu, 2=tanh, 3=sigmoid, 4=abs, 5=neg, 6=sqrt, 7=exp, 8=log, 9=swish,
//          10=step, 11=mish, 12=elu,
//          13=sign, 14=floor, 15=ceil, 16=round, 17=log2, 18=log10,
//          19=sin, 20=cos, 21=tan, 22=asin, 23=acos, 24=atan,
//          25=sinh, 26=cosh, 27=asinh, 28=acosh, 29=atanh,
//          30=erf, 31=rsqrt, 32=trunc, 33=frac, 34=logical_not
//
// Every mode above is dispatched by `backend/elementwise.rs`, except
// 34=logical_not, which `backend/compare.rs` dispatches (#91): a bool
// operand - physical f32 0.0/1.0, the encoding
// `wgpu/storage.rs::physical_element_bytes` documents - in, the negation
// as 0.0/1.0 out, labeled `Bool` by the caller. That representation is the
// one `masked_fill`/`where_cond` already consume, which is what settled
// the question that once kept this mode (and the whole `logical` group in
// `capability/declarations.rs`) deliberately absent.
//
// Host-parity notes (each was chosen because the obvious WGSL builtin
// disagrees with the CPU reference on a class of inputs):
//   - `sign` hand-rolls the three-way compare: WGSL's `sign` returns NaN for
//     a NaN input, while CPU's kernel falls through both comparisons to 0.
//   - `round` hand-rolls half-away-from-zero: WGSL's `round` is banker's
//     rounding (2.5 -> 2), Rust's `f32::round` is half-away (2.5 -> 3).
//   - `frac` is `x - trunc(x)`: WGSL's `fract` is floor-based, which
//     disagrees on every negative non-integer (-1.5 -> 0.5, not -0.5).
//   - `erf` is the same A&S 7.1.26 rational approximation CPU's
//     `erf_approx_f64` uses, evaluated in f32 (no f64 guarantee here).

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

fn round_half_away(x: f32) -> f32 {
    // Rust `f32::round`: halves go away from zero. WGSL `round` is
    // round-to-even and would disagree on every half-integer.
    if (x >= 0.0) {
        return floor(x + 0.5);
    } else {
        return ceil(x - 0.5);
    }
}

fn sign_cpu(x: f32) -> f32 {
    // CPU: `if value > 0 {1} else if value < 0 {-1} else {0}` -- a NaN
    // answers false to both comparisons and lands in the else (0).
    if (x > 0.0) {
        return 1.0;
    } else if (x < 0.0) {
        return -1.0;
    } else {
        return 0.0;
    }
}

fn erf_aands(x: f32) -> f32 {
    // A&S 7.1.26, same coefficients as CPU's `erf_approx_f64`.
    let sign = select(1.0, -1.0, x < 0.0);
    let v = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * v);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t;
    return sign * (1.0 - poly * exp(-v * v));
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
    } else if op == 12u {
        // ELU (alpha = 1.0)
        if x > 0.0 {
            out[idx] = x;
        } else {
            out[idx] = exp(x) - 1.0;
        }
    } else if op == 13u {
        out[idx] = sign_cpu(x);
    } else if op == 14u {
        out[idx] = floor(x);
    } else if op == 15u {
        out[idx] = ceil(x);
    } else if op == 16u {
        out[idx] = round_half_away(x);
    } else if op == 17u {
        out[idx] = log2(x);
    } else if op == 18u {
        // log10 via natural log: naga rejects the `log10` identifier even
        // though the WGSL spec names it, so derive it the portable way.
        out[idx] = log(x) / 2.302585092994046;  // ln(10)
    } else if op == 19u {
        out[idx] = sin(x);
    } else if op == 20u {
        out[idx] = cos(x);
    } else if op == 21u {
        out[idx] = tan(x);
    } else if op == 22u {
        out[idx] = asin(x);
    } else if op == 23u {
        out[idx] = acos(x);
    } else if op == 24u {
        out[idx] = atan(x);
    } else if op == 25u {
        out[idx] = sinh(x);
    } else if op == 26u {
        out[idx] = cosh(x);
    } else if op == 27u {
        out[idx] = asinh(x);
    } else if op == 28u {
        out[idx] = acosh(x);
    } else if op == 29u {
        out[idx] = atanh(x);
    } else if op == 30u {
        out[idx] = erf_aands(x);
    } else if op == 31u {
        out[idx] = 1.0 / sqrt(x);
    } else if op == 32u {
        out[idx] = trunc(x);
    } else if op == 33u {
        // Trunc-based fractional part, matching Rust `f32::fract`, not
        // WGSL's floor-based `fract`.
        out[idx] = x - trunc(x);
    } else if op == 34u {
        // logical_not over the bool-as-f32 encoding: CPU computes
        // `elementwise_cmp(x, x, |a, _| a == 0.0)`, i.e. true exactly
        // where the operand is 0.0, and false for the physical 1.0 of a
        // true bool - the same 0.0/1.0 in, 1.0/0.0 out swap.
        out[idx] = select(0.0, 1.0, x == 0.0);
    } else {
        // Unreachable through `backend/elementwise.rs`: every mode dispatched
        // there is named above. WGSL cannot trap, so the default repeats the
        // ELU computation rather than leaving `out` unwritten; any value
        // produced here means the dispatcher passed a mode that does not
        // exist, which is a bug at the call site rather than in this shader.
        if x > 0.0 {
            out[idx] = x;
        } else {
            out[idx] = exp(x) - 1.0;
        }
    }
}
